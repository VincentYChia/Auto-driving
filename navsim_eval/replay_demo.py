"""Re-planning replay: a pilot re-plans every 0.5 s over a ~20 s stretch of a navtest log (for showing, not scoring).

NAVSIM v1 scores one plan per scene: made at the scene's planning moment, checked over the next 4 s, never redone.
This replays a longer stretch of the same log instead. At every logged frame (2 Hz) the model receives exactly what
a NAVSIM agent receives at that frame (its AgentInput: the front camera, 4 ego poses, velocity, acceleration and the
route command) and plans the next 4 s. The car then moves as the human drove it, not as the model planned: this is
open loop, so the page shows how each new plan reacts to what the camera and the ego state show (a lead car
braking, a view opening up), not where the model's own driving would lead [proposed 2026-11-11].

Three steps:

    # 1. NAVSIM env, CPU (Anvil): each drive's frames, its model inputs and its map render
    python navsim_eval/replay_demo.py export --out $E/replay/v1 --drive 77c3cf2f="left turn" --drive ...
    #    -> v1/replay_inputs.json (export_mvlad_inputs records, one per frame), v1/drives.json, v1/map_<k>.png,
    #       v1/cam_f0/<token>.jpg
    # 2. pilot env, GPU: one plan per frame
    python pilot_sft/predict_navsim.py --inputs v1/replay_inputs.json --model Qwen/Qwen3.5-0.8B \\
        --adapter <run>/adapter --out <run>/replay_predictions
    # 3. any env with numpy and Pillow: the page
    python navsim_eval/replay_demo.py page --drives $E/replay/v1 --predictions <run>/replay_predictions \\
        --name "0.8B pilot (navtest 0.829)" --html <out>.html

A drive is centred on a navtest scene (``--drive <token or its first characters>=<label>``): from ``--before`` s
before its planning moment to ``--after`` s after, cut where the log lacks the 1.5 s of history an input needs or
the 4 s of recorded future the human's path needs. Everything is stored in the drive frame: the ego pose at that
planning moment (x forward, y left, heading counter-clockwise), the frame NAVSIM's own figure of the scene uses.

The page embeds nuPlan camera images and map renders: keep it out of the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim_tools import geometry  # noqa: E402

TEMPLATE = Path(__file__).with_name("replay_template.html")
PLACEHOLDER = "__REPLAY_DATA_JSON__"
STEP = 0.5  # s between logged frames and between plan poses
NUM_POSES = 8  # the 4 s plan
NUM_HISTORY = 4  # AgentInput frames: -1.5, -1.0, -0.5 and 0 s
COMMANDS = ["left", "straight", "right", "unknown"]  # NAVSIM's one-hot driving_command, in index order
MAP_MARGIN = 30.0  # m of map around the drive's path
MAP_PX_PER_M = 5.0
MAP_MAX_PX = 2400
AGENT_PAD = 10.0  # m: road users that come this close to the map render's edge are kept


# ----------------------------------------------------------------------------------------------- pure parts

def window(n_frames: int, center: int, before_s: float, after_s: float, step: float = STEP,
           num_history: int = NUM_HISTORY, num_future: int = NUM_POSES) -> Tuple[int, int]:
    """``(first, last)``: the indices of the frames a drive re-plans at, ``before_s`` before ``center`` to
    ``after_s`` after it, cut so every one has ``num_history - 1`` frames before it and ``num_future`` after it."""
    if not 0 <= center < n_frames:
        raise ValueError(f"centre frame {center} outside a log of {n_frames} frames")
    first = max(center - int(round(before_s / step)), num_history - 1)
    last = min(center + int(round(after_s / step)), n_frames - 1 - num_future)
    if first > center or last < center:
        raise ValueError(f"frame {center} of {n_frames} lacks the history or future a re-plan needs")
    return first, last


def steady_frames(timestamps_us: Sequence[int], step: float = STEP, tol: float = 0.05) -> bool:
    """True when consecutive log timestamps (microseconds) are ``step`` seconds apart, within ``tol`` s."""
    gaps = np.diff(np.asarray(timestamps_us, dtype=np.float64)) / 1e6
    return bool(len(gaps) == 0 or np.all(np.abs(gaps - step) <= tol))


def command_text(one_hot) -> str:
    c = np.asarray(one_hot, dtype=np.float64).reshape(-1)
    return COMMANDS[int(np.argmax(c))] if c.size == 4 and c.max() > 0 else "unknown"


def plan_in_drive(ego_pose, waypoints) -> Optional[np.ndarray]:
    """A plan's ``(N, 2)`` waypoints, given in the ego frame of the frame it was made at (pose ``ego_pose`` in the
    drive frame), in the drive frame. None stays None."""
    if waypoints is None:
        return None
    wp = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    local = np.column_stack([wp, np.zeros(len(wp))])
    return geometry.relative_to_absolute(np.asarray(ego_pose, dtype=np.float64), local)[:, :2]


def human_future(ego: np.ndarray, k: int, num_future: int = NUM_POSES) -> np.ndarray:
    """The human's recorded positions over the 4 s after frame ``k``: ``ego[k + 1 : k + 1 + num_future, :2]``."""
    out = np.asarray(ego, dtype=np.float64)[k + 1: k + 1 + num_future, :2]
    if len(out) != num_future:
        raise ValueError(f"frame {k}: {len(out)} recorded future poses, need {num_future}")
    return out


def replan_stats(plans: Sequence[Optional[np.ndarray]], humans: Sequence[np.ndarray]) -> List[Dict]:
    """Per re-plan frame, all in metres:

    - ``ade`` and ``fde``: the plan against the human's recorded next 4 s (mean and final distance);
    - ``change``: how far the plan moved since the frame before, over the 3.5 s both plans cover: the mean distance
      between the previous plan's poses at t + 1.0 ... 4.0 s and this plan's at t + 0.5 ... 3.5 s (the same moments,
      seen from the ground). None for the first frame or next to a missing plan.

    ``plans[k]`` (or None) and ``humans[k]`` are ``(8, 2)`` drive-frame positions at t_k + 0.5 ... t_k + 4 s."""
    out: List[Dict] = []
    for k, (plan, human) in enumerate(zip(plans, humans)):
        row: Dict[str, Optional[float]] = {"ade": None, "fde": None, "change": None}
        if plan is not None:
            d = np.linalg.norm(plan - human, axis=1)
            row["ade"], row["fde"] = float(d.mean()), float(d[-1])
            prev = plans[k - 1] if k > 0 else None
            if prev is not None:
                row["change"] = float(np.linalg.norm(prev[1:] - plan[:-1], axis=1).mean())
        out.append(row)
    return out


def reaches(origins: np.ndarray, plans: Sequence[Optional[np.ndarray]], humans: Sequence[np.ndarray]):
    """``(plan reach, human reach)`` per frame: straight-line distance from the car (``origins[k]``, x and y) to the
    last pose of the plan and of the human's recorded path. None for a missing plan."""
    plan_r, human_r = [], []
    for o, plan, human in zip(np.asarray(origins, dtype=np.float64), plans, humans):
        human_r.append(float(np.hypot(*(human[-1] - o[:2]))))
        plan_r.append(None if plan is None else float(np.hypot(*(plan[-1] - o[:2]))))
    return plan_r, human_r


def map_frame(points: np.ndarray, margin: float = MAP_MARGIN, px_per_m: float = MAP_PX_PER_M,
              max_px: int = MAP_MAX_PX) -> Dict:
    """The map render covering drive-frame ``points`` (N, 2) with ``margin`` metres: its centre ``(cx, cy)`` in the
    drive frame (the render is axis-aligned with the drive frame), its half sizes along x (up) and y (left) and its
    pixel width (along y), at ``px_per_m`` or less so that neither side passes ``max_px``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lo, hi = p.min(axis=0) - margin, p.max(axis=0) + margin
    cx, cy = (lo + hi) / 2
    half_x, half_y = (hi - lo) / 2
    ppm = min(px_per_m, max_px / (2 * max(half_x, half_y)))
    return {"cx": float(cx), "cy": float(cy), "half_x": float(half_x), "half_y": float(half_y),
            "width_px": int(round(2 * half_y * ppm)), "px_per_m": float(ppm)}


def drive_tracks(ego_global: Sequence, boxes: Sequence, names: Sequence, tokens: Sequence, origin,
                 frame: Dict, pad: float = AGENT_PAD) -> List[Dict]:
    """Road users over K frames in the drive frame (origin: the global ego pose at the drive's planning moment).

    ``ego_global[k]``: the global ego pose of frame k; ``boxes[k]``: that frame's NAVSIM boxes (x, y, z, length,
    width, height, heading) in its own ego frame; ``names`` and ``tokens``: their types and track tokens. Keeps the
    tracks that come within ``pad`` metres of the map render ``frame`` (map_frame) at some frame. Each track:
    ``type``, ``l``, ``w`` (where first seen) and ``p``, K entries of ``[x, y, heading]`` or None where absent."""
    origin = np.asarray(origin, dtype=np.float64)
    k_frames = len(ego_global)
    tracks: Dict[str, Dict] = {}
    for k in range(k_frames):
        if len(tokens[k]) == 0:
            continue
        b = np.asarray(boxes[k], dtype=np.float64).reshape(len(tokens[k]), -1)
        rel = geometry.change_frame(b[:, [0, 1, 6]], np.asarray(ego_global[k], dtype=np.float64), origin)
        for i, token in enumerate(tokens[k]):
            t = tracks.get(token)
            if t is None:
                t = tracks[token] = {"type": str(names[k][i]), "l": round(float(b[i, 3]), 2),
                                     "w": round(float(b[i, 4]), 2), "p": [None] * k_frames}
            t["p"][k] = [round(float(rel[i, 0]), 2), round(float(rel[i, 1]), 2), round(float(rel[i, 2]), 3)]
    keep = []
    for t in tracks.values():
        for p in t["p"]:
            if p is not None and abs(p[0] - frame["cx"]) <= frame["half_x"] + pad and \
                    abs(p[1] - frame["cy"]) <= frame["half_y"] + pad:
                keep.append(t)
                break
    return keep


def rounded(a, decimals: int = 2):
    """A nested list of floats rounded for the page (None stays None)."""
    if a is None:
        return None
    return np.round(np.asarray(a, dtype=np.float64), decimals).tolist()


def drive_payload(drive: Dict, predictions: Dict[str, np.ndarray], cameras: Dict[str, str]) -> Dict:
    """One drive of the page from its drives.json entry, the model's plans by frame token and the front-camera data
    URIs by frame token."""
    ego = np.asarray(drive["ego"], dtype=np.float64)  # drive frame, frames first - 3 ... last + 8
    off = drive["replan_offset"]  # index in ``ego`` of the first re-plan frame
    replans = drive["replans"]
    plans = [plan_in_drive(ego[off + k], predictions.get(r["token"])) for k, r in enumerate(replans)]
    humans = [human_future(ego, off + k) for k in range(len(replans))]
    stats = replan_stats(plans, humans)
    plan_r, human_r = reaches(ego[off: off + len(replans)], plans, humans)
    frames = []
    for k, r in enumerate(replans):
        frames.append({"t": r["t"], "token": r["token"], "speed": round(r["speed"], 2), "command": r["command"],
                       "cam": cameras.get(r["token"]), "plan": rounded(plans[k]),
                       "ade": stats[k]["ade"], "fde": stats[k]["fde"], "change": stats[k]["change"],
                       "reach": plan_r[k], "human_reach": human_r[k]})
    return {"label": drive["label"], "token": drive["token"], "log": drive["log"], "map": drive["map"],
            "t_first": replans[0]["t"], "t_last": replans[-1]["t"], "replan_offset": off,
            "ego": rounded(ego, 3), "frames": frames, "tracks": drive["tracks"], "map_frame": drive["map_frame"],
            "map_src": drive.get("map_src"), "missing_plans": sum(p is None for p in plans)}


def build_html(payload: Dict, template: Optional[str] = None) -> str:
    text = Path(template or TEMPLATE).read_text(encoding="utf-8")
    if text.count(PLACEHOLDER) != 1:
        raise ValueError(f"the template must hold {PLACEHOLDER} exactly once")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return text.replace(PLACEHOLDER, data)


# ----------------------------------------------------------------------------------------------- export (NAVSIM env)

def resolve(prefix: str, tokens: Sequence[str]) -> str:
    hits = [t for t in tokens if t.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"--drive {prefix}: {len(hits)} navtest tokens start with it (need exactly one)")
    return hits[0]


def export(args) -> None:
    import pickle

    from navsim.common.dataclasses import Scene, SceneFilter

    from navsim_eval.export_mvlad_inputs import build_records
    from navsim_eval.render_demo import bev_map_png, navsim_styles
    from navsim_tools import loader as nav_loader
    from navsim_tools import raw_frames

    log_path, sensor_path = nav_loader.default_paths("test", args.navsim_log_path, args.sensor_blobs_path)
    with open(args.navtest_inputs, encoding="utf-8") as f:
        log_of = {r["id"]: r["navsim"]["log_name"] for r in json.load(f)}
    out = Path(args.out)
    (out / "cam_f0").mkdir(parents=True, exist_ok=True)

    drives, all_tokens, logs = [], [], set()
    for spec in args.drive:
        prefix, _, label = spec.partition("=")
        token = resolve(prefix, sorted(log_of))
        log = log_of[token]
        with open(log_path / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        center = next(i for i, fr in enumerate(frames) if fr["token"] == token)
        first, last = window(len(frames), center, args.before, args.after)
        span = frames[first - (NUM_HISTORY - 1): last + 1 + NUM_POSES]
        if not steady_frames([fr["timestamp"] for fr in span]):
            raise SystemExit(f"{token}: the log's frames in this window are not 0.5 s apart")
        origin = raw_frames.global_ego_pose(frames[center], navsim_yaw=True)
        ego_global = np.stack([raw_frames.global_ego_pose(fr, navsim_yaw=True) for fr in span])
        ego = geometry.absolute_to_relative(origin, ego_global)
        mframe = map_frame(ego[:, :2])
        anns = [fr.get("anns") or {} for fr in span]
        boxes = [np.asarray(a.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64).reshape(-1, 7) for a in anns]
        tracks = drive_tracks(ego_global, boxes, [list(a.get("gt_names", [])) for a in anns],
                              [list(a.get("track_tokens", [])) for a in anns], origin, mframe)
        off = NUM_HISTORY - 1
        # road users at the re-plan frames only (the slider's range)
        for t in tracks:
            t["p"] = t["p"][off: off + last - first + 1]
        tracks = [t for t in tracks if any(p is not None for p in t["p"])]
        replans = []
        for j, fr in enumerate(frames[first: last + 1]):
            replans.append({"token": fr["token"], "t": round((first + j - center) * STEP, 2),
                            "speed": float(np.hypot(*raw_frames.ego_velocity(fr))),
                            "command": command_text(fr["driving_command"])})
        map_name = frames[center]["map_location"]
        centre_global = geometry.relative_to_absolute(origin, [[mframe["cx"], mframe["cy"], 0.0]])[0]
        png = bev_map_png(Scene._build_map_api(map_name), centre_global, mframe["half_x"], mframe["half_y"],
                          mframe["width_px"])
        name = f"map_{len(drives):02d}.png"
        (out / name).write_bytes(png)
        drives.append({"label": label or "drive", "token": token, "log": log, "map": map_name,
                       "origin_global": origin.tolist(), "ego": ego.tolist(), "replan_offset": off,
                       "replans": replans, "tracks": tracks, "map_frame": mframe, "map_png": name})
        all_tokens += [r["token"] for r in replans]
        logs.add(log)
        print(f"[replay] {token[:8]} {label}: frames {first}..{last} of {len(frames)} ({len(replans)} re-plans, "
              f"{(last - first) * STEP:.1f} s), {len(tracks)} road users, map {mframe['half_x'] * 2:.0f} x "
              f"{mframe['half_y'] * 2:.0f} m", flush=True)

    # the model inputs: NAVSIM's own AgentInput at every re-plan frame, as export_mvlad_inputs builds them for navtest
    scene_filter = SceneFilter(num_history_frames=NUM_HISTORY, num_future_frames=0, frame_interval=1,
                               has_route=False, log_names=sorted(logs), tokens=all_tokens)
    loader = nav_loader.make_scene_loader(log_path, sensor_path, scene_filter)
    records, missing, gap = build_records(loader, NUM_HISTORY, sensor_path, "navtest-replay", "available")
    if missing or gap > 1e-3 or len(records) != len(set(all_tokens)):
        raise SystemExit(f"{len(records)} records for {len(set(all_tokens))} frames, {missing} images missing, "
                         f"ego-history gap {gap:.2e} m")
    for r in records:
        dst = out / "cam_f0" / f"{r['id']}.jpg"
        shutil.copyfile(r["image"], dst)
        r["image"] = str(dst)
    with open(out / "replay_inputs.json", "w", encoding="utf-8") as f:
        json.dump(records, f)
    styles = navsim_styles()
    with open(out / "drives.json", "w", encoding="utf-8") as f:
        json.dump({"drives": drives, "styles": styles, "before": args.before, "after": args.after,
                   "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}, f)
    print(f"[replay] wrote {len(records)} inputs, {len(drives)} drives -> {out} (ego-history gap {gap:.1e} m)")


# ----------------------------------------------------------------------------------------------- page

def page(args) -> None:
    import base64

    from PIL import Image

    from navsim_eval.predictions import load_predictions

    root = Path(args.drives)
    with open(root / "drives.json", encoding="utf-8") as f:
        data = json.load(f)
    preds = load_predictions(args.predictions)

    def camera_uri(token: str) -> Optional[str]:
        path = root / "cam_f0" / f"{token}.jpg"
        if not path.exists():
            return None
        import io

        im = Image.open(path).convert("RGB")
        if im.width > args.cam_px:
            im = im.resize((args.cam_px, round(im.height * args.cam_px / im.width)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=args.cam_quality, optimize=True, progressive=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def map_uri(name: str) -> str:
        import io

        im = Image.open(root / name).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                                                             dither=Image.Dither.NONE)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    drives = []
    for d in data["drives"]:
        cams = {r["token"]: camera_uri(r["token"]) for r in d["replans"]}
        d = dict(d, map_src=map_uri(d["map_png"]))
        drives.append(drive_payload(d, preds, cams))
        print(f"[page] {d['token'][:8]} {d['label']}: {len(d['replans'])} re-plans, "
              f"{drives[-1]['missing_plans']} without a plan", flush=True)
    payload = {"name": args.name, "drives": drives, "styles": data["styles"], "step": STEP,
               "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "navtest_pdms": args.navtest_pdms}
    html = build_html(payload)
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.html).write_text(html, encoding="utf-8")
    print(f"[page] wrote {args.html}: {len(drives)} drives, {len(html) / 1e6:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="NAVSIM env: each drive's frames, model inputs and map render")
    e.add_argument("--out", required=True)
    e.add_argument("--drive", action="append", required=True, help="<navtest token or its first characters>=<label>")
    e.add_argument("--before", type=float, default=8.0, help="s before the scene's planning moment")
    e.add_argument("--after", type=float, default=12.0, help="s after it")
    e.add_argument("--navtest-inputs", default=os.path.join(os.environ.get("NAVSIM_EXP_ROOT", "."), "mvlad",
                                                            "navtest_inputs.json"),
                   help="export_mvlad_inputs.py's navtest records: the log of each navtest token")
    e.add_argument("--navsim-log-path", default=None)
    e.add_argument("--sensor-blobs-path", default=None)
    g = sub.add_parser("page", help="the page, from an export and the model's predictions on it")
    g.add_argument("--drives", required=True, help="the export's --out folder")
    g.add_argument("--predictions", required=True, help="predict_navsim.py's --out folder (or a JSON/JSONL file)")
    g.add_argument("--name", required=True, help="the model's name on the page")
    g.add_argument("--navtest-pdms", type=float, default=None, help="the model's full-navtest PDMS, for the page")
    g.add_argument("--html", required=True)
    g.add_argument("--cam-px", type=int, default=640)
    g.add_argument("--cam-quality", type=int, default=75)
    args = p.parse_args()
    if args.cmd == "export":
        export(args)
    else:
        page(args)


if __name__ == "__main__":
    main()
