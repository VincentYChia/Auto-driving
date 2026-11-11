"""Step 7 of ``docs/training_data/PRETRAIN_V1_PLAN.md``: the pilot training view, from the canonical
record (``canonical.py``). Every choice here is a view choice [proposed]; the canonical record is
unchanged, so another view is a re-run, not a re-extraction (§4b).

    python -m data_pipeline.v1.pilot_view build  --out data/pretrain/v1.0-pilot --set smoke
    python -m data_pipeline.v1.pilot_view build  --out data/pretrain/v1.0-xl --set v1-xl --storage zip --workers 64
    python -m data_pipeline.v1.pilot_view export --out data/pretrain/v1.0-pilot --set smoke --format chat

``build`` writes ``<out>/views/pilot_v1/<set>/``:
* ``images/<log>/<token>.jpg``: CAM_F0 downscaled to ``--width`` (640 × 360 by default: 220 visual
  tokens for Qwen3.5, whose images snap to 32 px steps; native 1920 × 1080 would be 2,040, D).
  With ``--storage zip`` [proposed] the same JPEG bytes go into one file instead, ``images.zip``
  (uncompressed, members sorted, member name = the jsonl ``image`` field, fixed timestamps), so a
  200k-sample view costs one inode, not 200k (Anvil's file quota). ``pilot_sft/view_images.py``
  reads either layout;
* ``train.jsonl``, ``dev.jsonl``: one sample per line: ``id``, ``image`` (relative to the view
  directory), ``prompt``, ``target`` (text), ``target_xy`` / ``target_yaw`` (the 8 continuous poses,
  0.5 … 4 s), and ``meta`` for slicing results (city, command, decisions, speed);
* ``view.json``: the view's settings, counts, storage, measured image sizes, and the
  constant-velocity baseline on each split.

Source images are read through ``image_store.read_image``: the loose original if it exists, else its
member in the archive's pack (``images/packs/<archive>.zip``).

``export --format chat`` writes ``chat_<split>.jsonl`` in the OpenAI ``messages`` format with the
image inlined as a base64 data URL: what Together.ai's vision fine-tuning takes, and what Hugging
Face chat templates read.

The prompt holds only what NAVSIM shows a planner at test time: the front image, ego speed,
acceleration and yaw rate, 1.5 s of past positions, and the route command. The yaw rate is
differenced from the current and -0.5 s headings (pilot_v1.1); pilot_v1.0 used canonical
``yaw_rate``, a central difference through the +0.5 s pose, which NAVSIM's AgentInput does not have.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_pipeline.v1 import image_store, provenance, tables
from navsim_tools import geometry
from data_pipeline.v1.openscene_index import PATH_OFFSETS_S

VIEW = "pilot_v1"
PROMPT_VERSION = "pilot_v1.1"  # 1.0 -> 1.1: past-only yaw rate (2026-11-03)
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
# 1.1 -> 1.2 [proposed 2026-11-09]: the past as 4 poses with heading at -1.5, -1.0, -0.5 and 0 s (the last is the
# origin), the history ReCogDrive gives its VLM (4 timesteps of (x, y, heading), 2 decimals), written in our prompt's
# wording; everything else is unchanged. Views of each version live side by side (VIEW_ROOTS); 1.1 stays the default.
POSES_S = (-1.5, -1.0, -0.5, 0.0)
PROMPT_V12 = (
    "Front camera image (CAM_F0).\n"
    "Ego state: speed {speed:.1f} m/s, acceleration {accel:+.1f} m/s², yaw rate {yaw_rate:+.2f} rad/s.\n"
    "Past poses (x forward and y left in metres, heading in radians) at -1.5, -1.0, -0.5 and 0 s: {poses}.\n"
    "Route command: {command}.\n"
    "Predict the path for the next 4 seconds: 8 positions (x, y) in metres, one every 0.5 s."
)
TEMPLATES = {"pilot_v1.1": PROMPT, "pilot_v1.2": PROMPT_V12}
PROMPT_VERSIONS = tuple(TEMPLATES)
VIEW_ROOTS = {"pilot_v1.1": VIEW, "pilot_v1.2": "pilot_v1.2"}
COLUMNS = ("frame_id", "log_name", "token", "our_split", "city", "image_path", "path_xyyaw", "path_valid",
           "speed", "accel_lon", "command")
PAIR = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
# Derived sets [proposed 2026-11-10] (data_pipeline/v1/derived_sets.py): cut from a built set with no canonical step,
# so frames.parquet has no in_<set> column for them; their membership is their token files' (member_rows)
DERIVED_SETS = ("v1-full-thin", "dev-v2", "v1-hard")
SETS = ("smoke", "v1-pilot", "v1-large", "v1-xl", "v1-xl-nd", "v1-nt", "v1-xl-mix", "v1-nt-mix", "v1-all",
        "v1-full") + DERIVED_SETS  # not v1-full-devlogs: the rows of dev-v2's held-out logs are in no view (plan §12)
STORAGE = ("loose", "zip")
ZIP_NAME = "images.zip"  # pilot_sft/view_images.ZIP_NAME
ZIP_CHUNK = 4096  # resized images held in memory at once by a zip build (about 40 kB each at 640 px, D)
ZIP_DATE = image_store.DATE_TIME  # the earliest a zip can store: no build time in the bytes


def _pairs(xy: np.ndarray) -> str:
    return " ".join(f"({x:.1f}, {y:.1f})" for x, y in np.round(xy, 1) + 0.0)  # no "-0.0"


def _poses(xyh: Any) -> str:
    return " ".join(f"({x:.2f}, {y:.2f}, {h:.2f})" for x, y, h in np.round(np.asarray(xyh, dtype=np.float64), 2) + 0.0)


def render_prompt(version: str, speed: float, accel: float, yaw_rate: float, poses: Any, command: str) -> str:
    """The prompt of ``version``. ``poses``: (x, y, heading) at ``POSES_S`` in the anchor's frame, the last the origin.

    ``speed`` is formatted by the template; pass ``accel`` and ``yaw_rate`` already rounded (as ``sample`` does)."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (len(POSES_S), 3):
        raise ValueError(f"need {len(POSES_S)} poses (x, y, heading), got shape {poses.shape}")
    if version == "pilot_v1.1":
        return PROMPT.format(speed=speed, accel=accel, yaw_rate=yaw_rate, history=_pairs(poses[:3, :2] + 0.0),
                             command=command)
    if version == "pilot_v1.2":
        return PROMPT_V12.format(speed=speed, accel=accel, yaw_rate=yaw_rate, poses=_poses(poses), command=command)
    raise ValueError(f"unknown prompt version {version}; known: {PROMPT_VERSIONS}")


def parse_path(text: str, n: int = len(TARGET_S)) -> Optional[np.ndarray]:
    """The first ``n`` "(x, y)" pairs of a model's answer, or None if there are fewer."""
    pairs = PAIR.findall(text)
    return np.array(pairs[:n], dtype=np.float64) if len(pairs) >= n else None


NOW, PREV = (int(np.flatnonzero(np.isclose(PATH_OFFSETS_S, t))[0]) for t in (0.0, -0.5))


def past_yaw_rate(path: np.ndarray) -> float:
    """(heading now - heading 0.5 s ago) / 0.5 s, in rad/s: no pose after the anchor."""
    return float(geometry.normalize_angle(path[NOW, 2] - path[PREV, 2]) / (PATH_OFFSETS_S[NOW] - PATH_OFFSETS_S[PREV]))


def sample(frames: tables.Columns, k: int, version: str = PROMPT_VERSION) -> Dict[str, Any]:
    path = np.asarray(frames["path_xyyaw"][k], dtype=np.float64)
    poses = path[[int(np.flatnonzero(np.isclose(PATH_OFFSETS_S, t))[0]) for t in POSES_S]]
    fut = path[[int(np.flatnonzero(np.isclose(PATH_OFFSETS_S, t))[0]) for t in TARGET_S]]
    prompt = render_prompt(version, speed=float(frames["speed"][k]),
                           accel=round(float(frames["accel_lon"][k]), 1) + 0.0,
                           yaw_rate=round(past_yaw_rate(path), 2) + 0.0, poses=poses,
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


def resize_bytes(src_bytes: bytes, width: int = WIDTH) -> bytes:
    """The view's JPEG of one image: ``width`` px wide, aspect kept, bilinear, quality ``JPEG_QUALITY``."""
    from PIL import Image

    with Image.open(io.BytesIO(src_bytes)) as image:
        height = round(image.height * width / image.width)
        small = image.convert("RGB").resize((width, height), Image.BILINEAR, reducing_gap=3.0)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _resize(job) -> int:
    """Resize file ``src`` to file ``dst`` unless ``dst`` exists (``predict_navsim.py`` uses this too)."""
    src, dst, width = job
    if os.path.exists(dst):
        return 0
    with open(src, "rb") as f:
        small = resize_bytes(f.read(), width)
    _write_file(dst, small)
    return 1


def _write_file(dst: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)


def _read_source(out_dir: str, image_path: str, source_archive: int) -> bytes:
    """One original CAM_F0 JPEG: the loose file, or its member in the archive's pack (``image_store``)."""
    return image_store.read_image(out_dir, image_path, source_archive)


def _jpeg_size(data: Any) -> Tuple[int, int]:
    """(width, height) of a JPEG given as bytes or as a path; Pillow reads the header only."""
    from PIL import Image

    with Image.open(io.BytesIO(data) if isinstance(data, bytes) else data) as image:
        return image.size


def _resize_loose(job) -> Tuple[int, Tuple[int, int]]:
    """(1 if written now else 0, (width, height) of the view image) for one loose-layout image."""
    out_dir, image_path, source_archive, dst, width = job
    if os.path.exists(dst):  # measured from the file's header, so a re-run does not read every image
        return 0, _jpeg_size(dst)
    small = resize_bytes(_read_source(out_dir, image_path, source_archive), width)
    _write_file(dst, small)
    return 1, _jpeg_size(small)


def _resize_member(job) -> bytes:
    out_dir, image_path, source_archive, width = job
    return resize_bytes(_read_source(out_dir, image_path, source_archive), width)


def _zip_info(name: str) -> zipfile.ZipInfo:
    """A member header that depends on the name only: fixed time, Unix ``rw-r--r--``, stored (as ``image_store``'s packs)."""
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = image_store.CREATE_SYSTEM
    info.external_attr = image_store.EXTERNAL_ATTR
    return info


def write_zip(path: str, names: Sequence[str], jobs: Sequence[Any], workers: int,
              chunk: int = ZIP_CHUNK) -> List[Tuple[int, int]]:
    """Resize ``jobs`` in parallel, ``chunk`` at a time, into ``path`` as members ``names`` (sorted).

    Written to ``path + ".part"``, synced, then renamed, so ``path`` is complete or absent. Returns the
    measured (width, height) of every member, in ``names`` order.
    """
    if list(names) != sorted(names) or len(set(names)) != len(names):
        raise ValueError("zip members must be unique and sorted")
    sizes: List[Tuple[int, int]] = []
    try:
        with open(path + ".part", "wb") as f:
            with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive, \
                    ProcessPoolExecutor(max_workers=workers) as pool:
                for start in range(0, len(jobs), chunk):
                    part = jobs[start : start + chunk]
                    for name, data in zip(names[start : start + chunk],
                                          pool.map(_resize_member, part, chunksize=max(1, len(part) // (4 * workers)))):
                        archive.writestr(_zip_info(name), data)
                        sizes.append(_jpeg_size(data))
            f.flush()
            os.fsync(f.fileno())
    except Exception:  # e.g. a source image missing from its pack: no multi-GB .part left on scratch
        if os.path.exists(path + ".part"):
            os.remove(path + ".part")
        raise
    os.replace(path + ".part", path)
    return sizes


def token_splits(out_dir: str, set_name: str) -> Dict[str, str]:
    """Token -> split of the set's token files (``splits/<set>_tokens_{train,dev}.txt``)."""
    split_of: Dict[str, str] = {}
    for split in ("train", "dev"):
        with open(os.path.join(out_dir, "splits", f"{set_name}_tokens_{split}.txt"), encoding="utf-8") as f:
            split_of.update((line.strip(), split) for line in f if line.strip())
    return split_of


def member_rows(out_dir: str, set_name: str, tokens: np.ndarray) -> np.ndarray:
    """Rows of a derived set (``DERIVED_SETS``): the frames whose token is in its token files, every one a frame."""
    listed = token_splits(out_dir, set_name)
    rows = np.flatnonzero(np.array([str(t) in listed for t in tokens], dtype=bool))
    if len(rows) != len(listed):
        raise ValueError(f"{len(listed) - len(rows)} tokens of {set_name} have no frame in frames.parquet "
                         "(a derived set must lie inside the folder's frames)")
    return rows


def set_split(out_dir: str, set_name: str, tokens: np.ndarray, rows: np.ndarray, index_split: np.ndarray) -> np.ndarray:
    """``index_split`` with each of ``rows`` given its split in the set's token files (``splits/<set>_tokens_*.txt``).

    The set's files decide a sample's split, not the index's ``our_split``: they agree for every set before
    v1.0-full, whose re-split dev releases val logs to training (PRETRAIN_V1_PLAN.md §12). Raises when a row's token
    is in neither file.
    """
    split_of = token_splits(out_dir, set_name)
    out = np.array(index_split, dtype=object, copy=True)
    missing = [str(tokens[k]) for k in rows if str(tokens[k]) not in split_of]
    if missing:
        raise ValueError(f"{len(missing)} frames of {set_name} are in neither of its token files, e.g. {missing[0]}")
    out[rows] = [split_of[str(tokens[k])] for k in rows]
    return out


def view_dir(out_dir: str, set_name: str, version: str = PROMPT_VERSION) -> str:
    return os.path.join(out_dir, "views", VIEW_ROOTS[version], set_name)


def build(out_dir: str, set_name: str = "smoke", width: int = WIDTH, workers: int = 8,
          storage: str = "loose", prompt_version: str = PROMPT_VERSION) -> Dict[str, Any]:
    """Write the view of ``set_name``. ``storage="zip"`` [proposed] packs the images into ``images.zip``.

    A loose build skips images that already exist; a zip build always rewrites ``images.zip`` (the bytes
    are the same for the same inputs). Either way ``train.jsonl`` / ``dev.jsonl`` are the same. A loose
    build refuses a view folder that holds ``images.zip``: the readers would serve the zip's images.
    """
    if storage not in STORAGE:
        raise ValueError(f"unknown storage {storage}")
    if prompt_version not in TEMPLATES:
        raise ValueError(f"unknown prompt version {prompt_version}; known: {PROMPT_VERSIONS}")
    stale_zip = os.path.join(view_dir(out_dir, set_name, prompt_version), ZIP_NAME)
    if storage == "loose" and os.path.exists(stale_zip):
        raise FileExistsError(f"{stale_zip} exists, and the readers (ViewImages, export) prefer it to loose "
                              "images: move it aside, or build with --storage zip")
    if set_name in DERIVED_SETS:
        frames = tables.read(os.path.join(out_dir, "frames.parquet"), list(COLUMNS) + ["source_archive"])
        rows = member_rows(out_dir, set_name, frames["token"])
    else:
        member = f"in_{set_name.replace('-', '_')}"
        frames = tables.read(os.path.join(out_dir, "frames.parquet"), list(COLUMNS) + ["source_archive", member])
        rows = np.flatnonzero(frames[member])
    index_split = frames["our_split"]
    frames["our_split"] = set_split(out_dir, set_name, frames["token"], rows, index_split)
    target = view_dir(out_dir, set_name, prompt_version)
    os.makedirs(target, exist_ok=True)

    def source(k: int) -> Tuple[str, str, int]:
        return out_dir, str(frames["image_path"][k]), int(frames["source_archive"][k])

    images_zip = None
    if storage == "loose":
        jobs = [(*source(k), os.path.join(target, "images", str(frames["log_name"][k]), f"{frames['token'][k]}.jpg"),
                 width) for k in rows]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_resize_loose, jobs, chunksize=64))
        written, sizes = sum(d[0] for d in done), [d[1] for d in done]
    else:
        name_of = {int(k): f"images/{frames['log_name'][k]}/{frames['token'][k]}.jpg" for k in rows}  # sample()["image"]
        order = sorted(name_of, key=name_of.get)
        names = [name_of[k] for k in order]
        path = os.path.join(target, ZIP_NAME)
        sizes = write_zip(path, names, [(*source(k), width) for k in order], workers)
        written = len(names)
        images_zip = {"file": ZIP_NAME, "members": len(names), "bytes": os.path.getsize(path),
                      "sha256": provenance.sha256_file(path), "compression": "stored"}
    measured = Counter(f"{w}x{h}" for w, h in sizes)
    height = int(Counter(h for _, h in sizes).most_common(1)[0][0]) if sizes else round(1080 * width / 1920)
    counts = {}
    for split in ("train", "dev"):
        chosen = rows[frames["our_split"][rows] == split]
        with open(os.path.join(target, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for k in chosen:
                f.write(json.dumps(sample(frames, k, prompt_version), ensure_ascii=False) + "\n")
        counts[split] = int(len(chosen))
    lengths = [len(sample(frames, k, prompt_version)["prompt"]) for k in rows[:200]]
    spec = {
        "view": VIEW_ROOTS[prompt_version],
        "prompt_version": prompt_version,
        "set": set_name,
        "created_utc": provenance.utc_now(),
        "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
        "source": "frames.parquet (canonical record, step 4)",
        "storage": storage,
        "samples": counts,
        "image": {"width": width, "height": height, "jpeg_quality": JPEG_QUALITY,
                  "resized_now": int(written),
                  "measured_sizes": dict(sorted(measured.items())),
                  "height_note": "V: the most common height among the view's images (measured_sizes); "
                                 "computed as 1080 · width / 1920 only for an empty view",
                  "visual_tokens_qwen3_5": _qwen_visual_tokens(width, height, 32),
                  "visual_tokens_qwen2_5_vl": _qwen_visual_tokens(width, height, 28),
                  "visual_tokens_note": "D: each side rounded to the nearest multiple of patch × merge "
                                        "(Qwen2VLImageProcessor), one token per merged patch"},
        "images_zip": images_zip,
        "split_source": "the set's token files (splits/<set>_tokens_{train,dev}.txt)",
        "split_differs_from_index": int((frames["our_split"][rows] != index_split[rows]).sum()),
        "history_s": list(HISTORY_S) if prompt_version == "pilot_v1.1" else list(POSES_S),
        "target_s": list(TARGET_S),
        "number_format": ("metres, 1 decimal in text; continuous values in target_xy / target_yaw"
                          if prompt_version == "pilot_v1.1" else
                          "metres, 1 decimal in text, past poses 2 decimals with heading in radians; continuous "
                          "values in target_xy / target_yaw"),
        "prompt_chars_mean": round(float(np.mean(lengths)), 1) if lengths else None,
        "system": SYSTEM,
        "prompt_template": TEMPLATES[prompt_version],
        "constant_velocity_baseline": {split: constant_velocity(frames, rows[frames["our_split"][rows] == split])
                                       for split in ("train", "dev")},
    }
    provenance.write_text(os.path.join(target, "view.json"), json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
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


def export(out_dir: str, set_name: str = "smoke", fmt: str = "chat", limit: Optional[int] = None,
           prompt_version: str = PROMPT_VERSION) -> Dict[str, Any]:
    if fmt != "chat":
        raise ValueError(f"unknown format {fmt}")
    target = view_dir(out_dir, set_name, prompt_version)
    out = {}
    with _ViewReader(target) as images:
        for split in ("train", "dev"):
            path = os.path.join(target, f"chat_{split}.jsonl")
            n = 0
            with open(os.path.join(target, f"{split}.jsonl"), encoding="utf-8") as src, \
                    open(path + ".part", "w", encoding="utf-8", newline="\n") as dst:
                for line in src:
                    if limit is not None and n >= limit:
                        break
                    item = json.loads(line)
                    dst.write(json.dumps(chat_record(item, images.read(item["image"]))) + "\n")
                    n += 1
            os.replace(path + ".part", path)
            out[split] = {"samples": n, "bytes": os.path.getsize(path), "sha256": provenance.sha256_file(path)}
    return out


class _ViewReader:
    """A view's image bytes: the ``images.zip`` member if the view is packed, else the loose file.

    The same rule as ``pilot_sft/view_images.ViewImages``, kept here so ``data_pipeline`` does not
    import from ``pilot_sft`` (the trainer reads the view, not the other way round).
    """

    def __init__(self, target: str):
        self.target = target
        path = os.path.join(target, ZIP_NAME)
        self.archive = zipfile.ZipFile(path) if os.path.isfile(path) else None

    def read(self, rel: str) -> bytes:
        if self.archive is not None:
            return self.archive.read(rel)
        with open(os.path.join(self.target, *rel.split("/")), "rb") as f:
            return f.read()

    def __enter__(self) -> "_ViewReader":
        return self

    def __exit__(self, *exc) -> None:
        if self.archive is not None:
            self.archive.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step 7: the pilot training view (image + prompt + 8 poses).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "export"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="dataset directory, e.g. data/pretrain/v1.0-pilot")
        p.add_argument("--set", default="smoke", choices=SETS)
        p.add_argument("--prompt-version", default=PROMPT_VERSION, choices=PROMPT_VERSIONS,
                       help="pilot_v1.2 [proposed]: 4 past poses with heading; written to views/pilot_v1.2/<set>")
        if name == "build":
            p.add_argument("--width", type=int, default=WIDTH, help="image width in the view (px)")
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--storage", default="loose", choices=STORAGE,
                           help="loose: images/<log>/<token>.jpg; zip: one images.zip [proposed]")
        else:
            p.add_argument("--format", default="chat", choices=("chat",))
            p.add_argument("--limit", type=int, default=None, help="first N samples per split (a quick test file)")
    args = parser.parse_args(argv)
    if args.command == "build":
        spec = build(args.out, args.set, args.width, args.workers, args.storage, args.prompt_version)
        print(json.dumps({k: spec[k] for k in ("samples", "storage", "image", "images_zip",
                                               "constant_velocity_baseline")}, indent=1))
    else:
        print(json.dumps(export(args.out, args.set, args.format, args.limit, args.prompt_version), indent=1))


if __name__ == "__main__":
    main()
