"""Step 7 of ``docs/training_data/PRETRAIN_V1_PLAN.md``: the pilot training view, from the canonical
record (``canonical.py``). Every choice here is a view choice [proposed]; the canonical record is
unchanged, so another view is a re-run, not a re-extraction (§4b).

    python -m data_pipeline.v1.pilot_view build  --out data/pretrain/v1.0-pilot --set smoke
    python -m data_pipeline.v1.pilot_view export --out data/pretrain/v1.0-pilot --set smoke --format chat

``build`` writes ``<out>/views/pilot_v1/<set>/``:
* ``images/<log>/<token>.jpg``: CAM_F0 downscaled to ``--width`` (640 × 360 by default: 220 visual
  tokens for Qwen3.5, whose images snap to 32 px steps; native 1920 × 1080 would be 2,040, D);
* ``train.jsonl``, ``dev.jsonl``: one sample per line: ``id``, ``image`` (relative to the view
  directory), ``prompt``, ``target`` (text), ``target_xy`` / ``target_yaw`` (the 8 continuous poses,
  0.5 … 4 s), and ``meta`` for slicing results (city, command, decisions, speed);
* ``view.json``: the view's settings, counts, and the constant-velocity baseline on each split.

``export --format chat`` writes ``chat_<split>.jsonl`` in the OpenAI ``messages`` format with the
image inlined as a base64 data URL: what Together.ai's vision fine-tuning takes, and what Hugging
Face chat templates read.

The prompt holds only what NAVSIM shows a planner at test time: the front image, ego speed,
acceleration and yaw rate, 1.5 s of past positions, and the route command.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional, Sequence

import numpy as np

from data_pipeline.v1 import provenance, tables
from data_pipeline.v1.openscene_index import PATH_OFFSETS_S

VIEW = "pilot_v1"
PROMPT_VERSION = "pilot_v1.0"
WIDTH = 640
JPEG_QUALITY = 90
HISTORY_S = (-1.5, -1.0, -0.5)
TARGET_S = tuple(np.round(np.arange(0.5, 4.01, 0.5), 1))
COMMAND_TEXT = {"left": "turn left", "straight": "go straight", "right": "turn right", "unknown": "none given"}
SYSTEM = "You are the motion planner of a self-driving car."
PROMPT = (
    "Front camera image (CAM_F0).\n"
    "Ego state: speed {speed:.1f} m/s, acceleration {accel:+.1f} m/s², yaw rate {yaw_rate:+.2f} rad/s.\n"
    "Past positions (x forward, y left, metres) at -1.5, -1.0, -0.5 s: {history}.\n"
    "Route command: {command}.\n"
    "Predict the path for the next 4 seconds: 8 positions (x, y) in metres, one every 0.5 s."
)
COLUMNS = ("frame_id", "log_name", "token", "our_split", "city", "image_path", "path_xyyaw", "path_valid",
           "speed", "accel_lon", "yaw_rate", "command")
PAIR = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


def _pairs(xy: np.ndarray) -> str:
    return " ".join(f"({x:.1f}, {y:.1f})" for x, y in np.round(xy, 1) + 0.0)  # no "-0.0"


def parse_path(text: str, n: int = len(TARGET_S)) -> Optional[np.ndarray]:
    """The first ``n`` "(x, y)" pairs of a model's answer, or None if there are fewer."""
    pairs = PAIR.findall(text)
    return np.array(pairs[:n], dtype=np.float64) if len(pairs) >= n else None


def sample(frames: tables.Columns, k: int) -> Dict[str, Any]:
    path = np.asarray(frames["path_xyyaw"][k], dtype=np.float64)
    hist = path[[int(np.flatnonzero(np.isclose(PATH_OFFSETS_S, t))[0]) for t in HISTORY_S]]
    fut = path[[int(np.flatnonzero(np.isclose(PATH_OFFSETS_S, t))[0]) for t in TARGET_S]]
    prompt = PROMPT.format(speed=float(frames["speed"][k]), accel=round(float(frames["accel_lon"][k]), 1) + 0.0,
                           yaw_rate=round(float(frames["yaw_rate"][k]), 2) + 0.0, history=_pairs(hist[:, :2] + 0.0),
                           command=COMMAND_TEXT.get(str(frames["command"][k]), "none given"))
    log, token = str(frames["log_name"][k]), str(frames["token"][k])
    return {
        "id": str(frames["frame_id"][k]),
        "image": f"images/{log}/{token}.jpg",
        "system": SYSTEM,
        "prompt": prompt,
        "target": _pairs(fut[:, :2]),
        "target_xy": np.round(fut[:, :2], 4).tolist(),
        "target_yaw": np.round(fut[:, 2], 5).tolist(),
        "meta": {"split": str(frames["our_split"][k]), "city": str(frames["city"][k]),
                 "command": str(frames["command"][k]), "speed": round(float(frames["speed"][k]), 3)},
    }


def constant_velocity(frames: tables.Columns, rows: np.ndarray) -> Dict[str, float]:
    """ADE / FDE (m) of driving on at the anchor's speed along the current heading: the bar to beat."""
    if len(rows) == 0:
        return {}
    t = np.array(TARGET_S)
    idx = [int(np.flatnonzero(np.isclose(PATH_OFFSETS_S, s))[0]) for s in TARGET_S]
    truth = np.stack([np.asarray(frames["path_xyyaw"][k], dtype=np.float64)[idx, :2] for k in rows])
    guess = np.zeros_like(truth)
    guess[:, :, 0] = frames["speed"][rows][:, None] * t[None, :]
    err = np.linalg.norm(guess - truth, axis=2)
    return {"ade_m": round(float(err.mean()), 3), "fde_m": round(float(err[:, -1].mean()), 3),
            "samples": int(len(rows))}


def _resize(job) -> int:
    from PIL import Image

    src, dst, width = job
    if os.path.exists(dst):
        return 0
    with Image.open(src) as image:
        height = round(image.height * width / image.width)
        small = image.convert("RGB").resize((width, height), Image.BILINEAR, reducing_gap=3.0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    small.save(dst + ".part", format="JPEG", quality=JPEG_QUALITY)
    os.replace(dst + ".part", dst)
    return 1


def view_dir(out_dir: str, set_name: str) -> str:
    return os.path.join(out_dir, "views", VIEW, set_name)


def build(out_dir: str, set_name: str = "smoke", width: int = WIDTH, workers: int = 8) -> Dict[str, Any]:
    member = f"in_{set_name.replace('-', '_')}"
    frames = tables.read(os.path.join(out_dir, "frames.parquet"), list(COLUMNS) + [member])
    rows = np.flatnonzero(frames[member])
    target = view_dir(out_dir, set_name)
    os.makedirs(target, exist_ok=True)
    jobs = [(os.path.join(out_dir, *str(frames["image_path"][k]).split("/")),
             os.path.join(target, "images", str(frames["log_name"][k]), f"{frames['token'][k]}.jpg"), width)
            for k in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        written = sum(pool.map(_resize, jobs, chunksize=64))
    counts = {}
    for split in ("train", "dev"):
        chosen = rows[frames["our_split"][rows] == split]
        with open(os.path.join(target, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for k in chosen:
                f.write(json.dumps(sample(frames, k), ensure_ascii=False) + "\n")
        counts[split] = int(len(chosen))
    lengths = [len(sample(frames, k)["prompt"]) for k in rows[:200]]
    spec = {
        "view": VIEW,
        "prompt_version": PROMPT_VERSION,
        "set": set_name,
        "created_utc": provenance.utc_now(),
        "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
        "source": "frames.parquet (canonical record, step 4)",
        "samples": counts,
        "image": {"width": width, "height": round(1080 * width / 1920), "jpeg_quality": JPEG_QUALITY,
                  "resized_now": int(written),
                  "visual_tokens_qwen3_5": _qwen_visual_tokens(width, round(1080 * width / 1920), 32),
                  "visual_tokens_qwen2_5_vl": _qwen_visual_tokens(width, round(1080 * width / 1920), 28),
                  "visual_tokens_note": "D: each side rounded to the nearest multiple of patch × merge "
                                        "(Qwen2VLImageProcessor), one token per merged patch"},
        "history_s": list(HISTORY_S),
        "target_s": list(TARGET_S),
        "number_format": "metres, 1 decimal in text; continuous values in target_xy / target_yaw",
        "prompt_chars_mean": round(float(np.mean(lengths)), 1) if lengths else None,
        "system": SYSTEM,
        "prompt_template": PROMPT,
        "constant_velocity_baseline": {split: constant_velocity(frames, rows[frames["our_split"][rows] == split])
                                       for split in ("train", "dev")},
    }
    with open(os.path.join(target, "view.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return spec


def _qwen_visual_tokens(width: int, height: int, step: int) -> int:
    return max(1, round(width / step)) * max(1, round(height / step))


def chat_record(item: Dict[str, Any], image_bytes: bytes) -> Dict[str, Any]:
    """One sample in the OpenAI ``messages`` format, image first, as a base64 data URL."""
    url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    return {"messages": [
        {"role": "system", "content": [{"type": "text", "text": item["system"]}]},  # Together: a text part, not a string
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": item["prompt"]}]},
        {"role": "assistant", "content": item["target"]},
    ]}


def export(out_dir: str, set_name: str = "smoke", fmt: str = "chat", limit: Optional[int] = None) -> Dict[str, Any]:
    if fmt != "chat":
        raise ValueError(f"unknown format {fmt}")
    target = view_dir(out_dir, set_name)
    out = {}
    for split in ("train", "dev"):
        path = os.path.join(target, f"chat_{split}.jsonl")
        n = 0
        with open(os.path.join(target, f"{split}.jsonl"), encoding="utf-8") as src, \
                open(path + ".part", "w", encoding="utf-8", newline="\n") as dst:
            for line in src:
                if limit is not None and n >= limit:
                    break
                item = json.loads(line)
                with open(os.path.join(target, *item["image"].split("/")), "rb") as f:
                    dst.write(json.dumps(chat_record(item, f.read())) + "\n")
                n += 1
        os.replace(path + ".part", path)
        out[split] = {"samples": n, "bytes": os.path.getsize(path), "sha256": provenance.sha256_file(path)}
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step 7: the pilot training view (image + prompt + 8 poses).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "export"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="dataset directory, e.g. data/pretrain/v1.0-pilot")
        p.add_argument("--set", default="smoke", choices=("smoke", "v1-pilot"))
        if name == "build":
            p.add_argument("--width", type=int, default=WIDTH, help="image width in the view (px)")
            p.add_argument("--workers", type=int, default=8)
        else:
            p.add_argument("--format", default="chat", choices=("chat",))
            p.add_argument("--limit", type=int, default=None, help="first N samples per split (a quick test file)")
    args = parser.parse_args(argv)
    if args.command == "build":
        spec = build(args.out, args.set, args.width, args.workers)
        print(json.dumps({k: spec[k] for k in ("samples", "image", "constant_velocity_baseline")}, indent=1))
    else:
        print(json.dumps(export(args.out, args.set, args.format, args.limit), indent=1))


if __name__ == "__main__":
    main()
