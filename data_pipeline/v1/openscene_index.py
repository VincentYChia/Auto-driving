"""Steps 1–2 of ``docs/training_data/PRETRAIN_V1_PLAN.md``: index every valid anchor frame in the
OpenScene trainval metadata. No images are read.

Input (``--metadata``), either:
* a directory of per-log pickles: ``download_navsim_data.sh trainval-logs`` puts them in
  ``$OPENSCENE_DATA_ROOT/navsim_logs/trainval``; or
* ``openscene_metadata_trainval.tgz`` as a local path or URL (``METADATA_URL``), read as a stream
  and never extracted.

Output, in ``<out>/index/``:
* ``candidates.parquet``: one row per anchor frame whose 1.5 s history and 4 s future exist with no
  timing gap (rule 1 of the plan, §3). The other rules are columns, applied by ``selection.py``, so
  the audit can count what each rule removes.
* ``logs.parquet``: one row per log: frame counts, timing, CAM_F0 calibration, split.
* ``manifest.json``: inputs, parameters, counts and checksums. Aggregate only, so it can be committed.

What the metadata holds (V, OpenScene's ``create_openscene_metadata.py``):
* one frame per 10th LiDAR sweep (2 Hz); ``timestamp`` is in microseconds;
* frames whose LiDAR file or a camera image is missing are dropped, so a log can have silent gaps.
  NAVSIM windows the frame list without checking time (``navsim/common/dataloader.py``); every
  window here is checked;
* ``driving_command``: one-hot (left, forward, right, unknown), from the route 20 m ahead;
* ``ego_dynamic_state``: [vx, vy, ax, ay] in the ego (rear-axle) frame.

    python -m data_pipeline.v1.openscene_index --metadata DIR_TGZ_OR_URL --navsim-root ~/navsim \\
        --out data/pretrain/v1.0-pilot [--max-logs 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import posixpath
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from glob import glob
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow

from data_pipeline import labels
from data_pipeline.v1 import navsim_splits, provenance, safe_pickle, stats, tables
from navsim_tools import geometry, raw_frames

DATASET_VERSION = "v1.0-pilot"
METADATA_URL = (
    "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/"
    "openscene-v1.1/openscene_metadata_trainval.tgz"
)
CAMERA = "CAM_F0"

DT_US = 500_000  # nominal frame spacing (0.5 s)
DT_TOL_US = 50_000  # rule 1: every gap in the window is 0.5 s ± 0.05 s
PAST_REQUIRED = 3  # rule 1: frames i-3 .. i, 1.5 s of history
FUTURE_REQUIRED = 8  # rule 1: frames i+1 .. i+8, 4 s of future
PAST_STORED = 8  # stored path: -4.0 s ...
FUTURE_STORED = 12  # ... to +6.0 s
OFFSETS = np.arange(-PAST_STORED, FUTURE_STORED + 1)  # in frames
PATH_OFFSETS_S = OFFSETS * (DT_US / 1e6)
CORE = (OFFSETS >= -PAST_REQUIRED) & (OFFSETS <= FUTURE_REQUIRED)  # the window rule 1 requires
ANCHOR = PAST_STORED  # column of offset 0
MAX_STEP_M = 25.0  # rule 3: drop if any 0.5 s step is longer ...
MAX_TURN_RAD = 0.5  # ... or turns more
SPEED_TOL_MPS = 1.0  # rule 3 flag: logged speed vs speed from the path
STANDSTILL_SPEED = labels.STOP_SPEED  # rule 6: slower than this ...
STANDSTILL_MOVE_M = 0.5  # ... and moving less than this over the window
DT_HIST_EDGES_S = (0.0, 0.45, 0.49, 0.499, 0.501, 0.51, 0.55, 1.0, 5.0, stats.INF)

# Calibration columns of logs.parquet, named after their source keys: their meaning (camera-to-ego
# vs camera-to-LiDAR) is checked against images in step 4, not assumed here.
CALIBRATION = (  # (column, key, size); cam_f0_* keys are in cams[CAM_F0], lidar2ego_* in the frame
    ("cam_f0_intrinsic", "cam_intrinsic", 9),
    ("cam_f0_distortion", "distortion", 5),
    ("cam_f0_sensor2lidar_rotation", "sensor2lidar_rotation", 9),
    ("cam_f0_sensor2lidar_translation", "sensor2lidar_translation", 3),
    ("lidar2ego_translation", "lidar2ego_translation", 3),
    ("lidar2ego_rotation", "lidar2ego_rotation", 4),
)


def parameters() -> Dict[str, Any]:
    return {
        "dt_s": DT_US / 1e6,
        "dt_tolerance_s": DT_TOL_US / 1e6,
        "history_required_s": PAST_REQUIRED * DT_US / 1e6,
        "future_required_s": FUTURE_REQUIRED * DT_US / 1e6,
        "path_offsets_s": PATH_OFFSETS_S.tolist(),
        "max_step_m": MAX_STEP_M,
        "max_turn_rad": MAX_TURN_RAD,
        "speed_tolerance_mps": SPEED_TOL_MPS,
        "standstill_speed_mps": STANDSTILL_SPEED,
        "standstill_move_m": STANDSTILL_MOVE_M,
        "decision_rules_version": labels.RULES_VERSION,
        "path_frame": "ego rear axle at the anchor frame: x forward, y left; metres, radians",
    }


# --- reading -------------------------------------------------------------------------------------


def iter_log_pickles(source: str, max_logs: Optional[int] = None) -> Iterator[Tuple[str, bytes]]:
    """Yield ``(file name, pickle bytes)`` per log, in directory or archive order."""
    for count, item in enumerate(_iter_source(source)):
        if max_logs is not None and count >= max_logs:
            return
        yield item


def _iter_source(source: str) -> Iterator[Tuple[str, bytes]]:
    if source.startswith(("http://", "https://")):
        # The timeout applies to each read, so a stalled transfer fails instead of hanging the job.
        with urllib.request.urlopen(source, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
            yield from _iter_tar(tar)
    elif os.path.isdir(source):
        for path in sorted(glob(os.path.join(source, "*.pkl"))):
            with open(path, "rb") as f:
                yield os.path.basename(path), f.read()
    else:
        with tarfile.open(source, mode="r|gz") as tar:
            yield from _iter_tar(tar)


def _iter_tar(tar: tarfile.TarFile) -> Iterator[Tuple[str, bytes]]:
    # Members are read into memory, never extracted, so their paths are only labels.
    for member in tar:
        if member.isfile() and member.name.endswith(".pkl"):
            yield posixpath.basename(member.name), tar.extractfile(member).read()


# --- one log -------------------------------------------------------------------------------------


def _flat(value: Any, size: int) -> np.ndarray:
    """``value`` as a flat float64 array; NaN if it is missing or has the wrong size."""
    if value is None:
        return np.full(size, np.nan)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr if arr.size == size else np.full(size, np.nan)


def _calibration(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    cam = (frame.get("cams") or {}).get(CAMERA)
    if not cam:
        return None
    return np.concatenate(
        [_flat((frame if column.startswith("lidar2ego") else cam).get(key), size) for column, key, size in CALIBRATION]
    )


def _command(frame: Dict[str, Any]) -> Tuple[str, bool]:
    one_hot = raw_frames.driving_command(frame)
    is_one_hot = len(one_hot) == 4 and set(one_hot) <= {0, 1} and sum(one_hot) == 1
    return labels.describe_command(one_hot), is_one_hot


def _chain_ok(link_ok: np.ndarray, idx: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """``[i, j]``: frame ``idx[i, j]`` exists and every link between it and anchor ``i`` is ok."""
    bad = np.concatenate([[0], np.cumsum(~link_ok)])  # bad[k] = number of bad links among 0 .. k-1
    anchor = np.arange(len(idx))[:, None]
    return inside & (bad[np.maximum(idx, anchor)] == bad[np.minimum(idx, anchor)])


def _relative(origins: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Global ``poses`` (m, k, 3) in the frames of global ``origins`` (m, 3).

    Batched ``geometry.absolute_to_relative``.
    """
    d = poses - origins[:, None, :]
    c, s = np.cos(origins[:, 2])[:, None], np.sin(origins[:, 2])[:, None]
    out = np.empty_like(d)
    out[..., 0] = c * d[..., 0] + s * d[..., 1]
    out[..., 1] = -s * d[..., 0] + c * d[..., 1]
    out[..., 2] = geometry.normalize_angle(d[..., 2])
    return out


def index_log(
    frames: Sequence[Dict[str, Any]],
    splits: navsim_splits.NavsimSplits,
    source_name: str = "",
    pickle_sha256: str = "",
) -> Tuple[Dict[str, Any], tables.Columns]:
    """The log's summary row and its candidate rows."""
    if not frames:
        raise ValueError(f"{source_name}: the log has no frames")
    log_names = {str(f["log_name"]) for f in frames}
    if len(log_names) != 1:
        raise ValueError(f"{source_name}: frames from several logs: {sorted(log_names)[:3]}")
    log_name = log_names.pop()
    order = sorted(range(len(frames)), key=lambda k: int(frames[k]["timestamp"]))
    was_sorted = order == list(range(len(frames)))
    frames = [frames[k] for k in order]
    n = len(frames)

    t_us = np.array([int(f["timestamp"]) for f in frames], dtype=np.int64)
    poses = np.stack([raw_frames.global_ego_pose(f) for f in frames])
    dynamics = np.stack([np.concatenate([raw_frames.ego_velocity(f), raw_frames.ego_acceleration(f)]) for f in frames])
    tokens = np.array([str(f["token"]) for f in frames], dtype=object)
    in_navtrain = np.array([t in splits.navtrain_tokens for t in tokens], dtype=bool)
    scenes = np.array([str(f.get("scene_token")) for f in frames], dtype=object)
    cities = np.array([str(f.get("map_location")) for f in frames], dtype=object)
    commands = [_command(f) for f in frames]
    cam_paths = [raw_frames.camera_paths(f).get(CAMERA) or "" for f in frames]
    has_route = np.array([len(f.get("roadblock_ids") or ()) > 0 for f in frames])
    calibrations = [_calibration(f) for f in frames]

    # Links between consecutive frames: timing (rule 1) and pose sanity (rule 3).
    dt_us = np.diff(t_us)
    gap_ok = np.abs(dt_us - DT_US) <= DT_TOL_US
    steps = np.diff(poses, axis=0)
    step_m = np.hypot(steps[:, 0], steps[:, 1])
    turn_rad = np.abs(geometry.normalize_angle(steps[:, 2]))
    sane = (step_m <= MAX_STEP_M) & (turn_rad <= MAX_TURN_RAD)

    # Windows: row i holds frames i-8 .. i+12. A path point is valid if the frame exists and the
    # chain of links back to the anchor passes both checks.
    idx = np.arange(n)[:, None] + OFFSETS[None, :]
    inside = (idx >= 0) & (idx < n)
    idx = np.clip(idx, 0, n - 1)
    timing_ok = _chain_ok(gap_ok, idx, inside)
    sane_ok = _chain_ok(sane, idx, inside)
    cand = np.flatnonzero(timing_ok[:, CORE].all(axis=1))  # rule 1
    has_room = (np.arange(n) >= PAST_REQUIRED) & (np.arange(n) + FUTURE_REQUIRED <= n - 1)

    w = idx[cand]
    valid = timing_ok[cand] & sane_ok[cand]
    rel = _relative(poses[cand], poses[w])
    t_rel = (t_us[w] - t_us[cand][:, None]) / 1e6
    vel, acc = dynamics[cand, :2], dynamics[cand, 2:]
    speed = np.hypot(vel[:, 0], vel[:, 1])
    prev, nxt = cand - 1, cand + 1
    dt2 = (t_us[nxt] - t_us[prev]) / 1e6
    chord = poses[nxt, :2] - poses[prev, :2]
    speed_from_path = np.hypot(chord[:, 0], chord[:, 1]) / dt2
    yaw_rate = geometry.normalize_angle(poses[nxt, 2] - poses[prev, 2]) / dt2
    links = cand[:, None] + np.arange(-PAST_REQUIRED, FUTURE_REQUIRED)[None, :]
    moved = np.hypot(rel[:, CORE, 0], rel[:, CORE, 1]).max(axis=1)
    future = rel[:, ANCHOR + 1 : ANCHOR + 1 + FUTURE_REQUIRED]  # +0.5 .. +4.0 s
    _, scene_codes = np.unique(scenes, return_inverse=True)
    scene_codes = scene_codes.reshape(-1)

    nuplan = splits.nuplan_split(log_name)
    log_reason = splits.guarded_logs.get(log_name, "")
    log_split = "excluded" if log_reason else ("dev" if nuplan == "val" else "train")
    cand_tokens = tokens[cand]
    reasons = np.array([log_reason or splits.guarded_tokens.get(t, "") for t in cand_tokens], dtype=object)
    m = len(cand)

    path = rel.astype(np.float32)
    path[~valid] = np.nan
    path_t = t_rel.astype(np.float32)
    path_t[~valid] = np.nan
    rows: tables.Columns = {
        "frame_id": np.array([f"openscene/{log_name}/{t}" for t in cand_tokens], dtype=object),
        "log_name": np.full(m, log_name, dtype=object),
        "token": cand_tokens,
        "scene_token": scenes[cand],
        "t_us": t_us[cand],
        "frame_index": cand.astype(np.int32),
        "nuplan_split": np.full(m, nuplan, dtype=object),
        "our_split": np.array(["excluded" if r else log_split for r in reasons], dtype=object),
        "guard_reason": reasons,
        "city": cities[cand],
        "traffic_side": np.array(["left" if c.startswith("sg-") else "right" for c in cities[cand]], dtype=object),
        "in_navtrain": in_navtrain[cand],
        "has_cam_f0": np.array([bool(cam_paths[i]) for i in cand], dtype=bool),
        "cam_f0_path": np.array([cam_paths[i] for i in cand], dtype=object),
        "vx": vel[:, 0].astype(np.float32),
        "vy": vel[:, 1].astype(np.float32),
        "ax": acc[:, 0].astype(np.float32),
        "ay": acc[:, 1].astype(np.float32),
        "speed": speed.astype(np.float32),
        "accel_lon": acc[:, 0].astype(np.float32),
        "accel_lat": acc[:, 1].astype(np.float32),
        "yaw_rate": yaw_rate.astype(np.float32),
        "speed_from_path": speed_from_path.astype(np.float32),
        "command": np.array([commands[i][0] for i in cand], dtype=object),
        "command_onehot_ok": np.array([commands[i][1] for i in cand], dtype=bool),
        "has_route": has_route[cand],
        "crosses_scene_boundary": (scene_codes[w[:, CORE]] != scene_codes[cand][:, None]).any(axis=1),
        "max_step_m": step_m[links].max(axis=1).astype(np.float32) if m else np.zeros(0, np.float32),
        "max_turn_rad": turn_rad[links].max(axis=1).astype(np.float32) if m else np.zeros(0, np.float32),
        "pose_ok": sane_ok[cand][:, CORE].all(axis=1),
        "speed_consistent": np.abs(speed - speed_from_path) <= SPEED_TOL_MPS,
        "standstill": (speed < STANDSTILL_SPEED) & (moved < STANDSTILL_MOVE_M),
        "max_time_err_s": np.abs(t_rel[:, CORE] - PATH_OFFSETS_S[CORE]).max(axis=1).astype(np.float32),
        "decision_lon": np.array(
            [labels.longitudinal_action(v, f[:, :2], dt=0.5, horizon_s=3.0) for v, f in zip(speed, future)],
            dtype=object,
        ),
        "decision_lat": np.array([labels.lateral_action(f, dt=0.5, horizon_s=4.0) for f in future], dtype=object),
        "path_xyyaw": path,
        "path_t_s": path_t,
        "path_valid": valid,
    }

    present = [c for c in calibrations if c is not None]
    reference = present[0] if present else None
    log_row: Dict[str, Any] = {
        "log_name": log_name,
        "source_member": source_name,
        "pickle_sha256": pickle_sha256,
        "name_matches_file": source_name in ("", f"{log_name}.pkl"),
        "n_frames": n,
        "n_candidates": m,
        "n_no_room": int((~has_room).sum()),  # too close to the log's start or end for a window
        "n_gap_windows": int(has_room.sum()) - m,  # room, but a timing gap inside the window
        "was_sorted": was_sorted,
        "n_dup_timestamps": int((dt_us == 0).sum()),
        "n_dup_tokens": n - len(set(tokens.tolist())),
        "n_gaps": int((~gap_ok).sum()),
        "n_pose_jumps": int((~sane).sum()),
        "dt_hist": np.array(stats.bin_counts(dt_us / 1e6, DT_HIST_EDGES_S), dtype=np.int64),
        "t_start_us": int(t_us[0]),
        "t_end_us": int(t_us[-1]),
        "n_scenes": len(set(scenes.tolist())),
        "city": Counter(cities.tolist()).most_common(1)[0][0],
        "city_varies": len(set(cities.tolist())) > 1,
        "vehicle_name": str(frames[0].get("vehicle_name") or ""),
        "nuplan_split": nuplan,
        "our_split": log_split,
        "guard_reason": log_reason,
        "n_no_cam_f0": sum(1 for p in cam_paths if not p),
        "n_no_route": int((~has_route).sum()),
        "n_bad_command": sum(1 for _, ok in commands if not ok),
        "n_navtrain_frames": int(in_navtrain.sum()),
        "n_navtrain_candidates": int(in_navtrain[cand].sum()),
        "calib_varies": bool(present) and not np.allclose(np.stack(present), reference, atol=1e-9, equal_nan=True),
    }
    start = 0
    for column, _, size in CALIBRATION:
        log_row[column] = reference[start : start + size] if reference is not None else np.full(size, np.nan)
        start += size
    return log_row, rows


# --- all logs ------------------------------------------------------------------------------------


def _column(values: List[Any]) -> np.ndarray:
    if isinstance(values[0], str):
        return np.array(values, dtype=object)
    if isinstance(values[0], (list, tuple, np.ndarray)):
        return np.stack([np.asarray(v) for v in values])
    return np.array(values)


def _describe_source(source: str) -> Dict[str, Any]:
    if source.startswith(("http://", "https://")):
        return {"kind": "url", "name": source}
    if os.path.isdir(source):
        return {"kind": "directory", "name": os.path.basename(os.path.normpath(source))}
    return {"kind": "tgz", "name": os.path.basename(source), "bytes": os.path.getsize(source)}


def build_index(
    source: str,
    splits: navsim_splits.NavsimSplits,
    out_dir: str,
    max_logs: Optional[int] = None,
    progress_every: int = 50,
) -> Dict[str, Any]:
    """Index every log in ``source`` and write ``<out_dir>/index/``. Returns the manifest."""
    started = time.time()
    index_dir = os.path.join(out_dir, "index")
    os.makedirs(index_dir, exist_ok=True)
    log_rows: List[Dict[str, Any]] = []
    parts: List[tables.Columns] = []
    for name, data in iter_log_pickles(source, max_logs):
        try:
            frames = safe_pickle.loads(data)
        except pickle.UnpicklingError as exc:
            raise pickle.UnpicklingError(f"{name}: {exc}") from exc
        log_row, rows = index_log(frames, splits, name, hashlib.sha256(data).hexdigest())
        log_rows.append(log_row)
        parts.append(rows)
        if len(log_rows) % progress_every == 0:
            print(
                f"[index] {len(log_rows)} logs, {sum(len(p['token']) for p in parts):,} candidates, "
                f"{time.time() - started:.0f} s",
                file=sys.stderr,
                flush=True,
            )
    if not log_rows:
        raise RuntimeError(f"no *.pkl logs found in {source}")
    duplicates = [k for k, v in Counter(r["log_name"] for r in log_rows).items() if v > 1]
    if duplicates:
        raise ValueError(f"logs appear more than once in {source}: {duplicates[:5]}")

    candidates = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    logs = {k: _column([r[k] for r in log_rows]) for k in log_rows[0]}
    outputs = {
        "candidates.parquet": (candidates, {"parameters": parameters()}),
        "logs.parquet": (logs, {"dt_hist_edges_s": stats.edges_json(DT_HIST_EDGES_S)}),
    }
    manifest: Dict[str, Any] = {
        "dataset": f"pretrain {DATASET_VERSION}",
        "stage": "index: PRETRAIN_V1_PLAN.md steps 1-2",
        "created_utc": provenance.utc_now(),
        "code": {
            "module": __name__,
            "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
            "repo_dirty": provenance.git_dirty(provenance.REPO_ROOT),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "navsim": {
            "devkit_commit": splits.devkit_commit,
            "files": splits.files,
            "navtrain_tokens": len(splits.navtrain_tokens),
            "guarded_logs": len(splits.guarded_logs),
            "guarded_tokens": len(splits.guarded_tokens),
        },
        "input": {**_describe_source(source), "max_logs": max_logs, "logs_read": len(log_rows)},
        "parameters": parameters(),
        "counts": {
            "logs": len(log_rows),
            "frames": int(logs["n_frames"].sum()),
            "candidates": len(candidates["token"]),
        },
        "outputs": {},
    }
    for name, (columns, meta) in outputs.items():
        path = os.path.join(index_dir, name)
        tables.write(path, columns, metadata=meta)
        manifest["outputs"][name] = {"rows": len(next(iter(columns.values()))), "sha256": provenance.sha256_file(path)}
    manifest["runtime_s"] = round(time.time() - started, 1)
    with open(os.path.join(index_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Index the OpenScene trainval metadata (plan steps 1-2).")
    parser.add_argument("--metadata", required=True, help=f"directory of *.pkl logs, or the .tgz (path or URL: {METADATA_URL})")
    parser.add_argument("--navsim-root", required=True, help="NAVSIM devkit checkout, for the split lists")
    parser.add_argument("--out", required=True, help="dataset directory, e.g. data/pretrain/v1.0-pilot")
    parser.add_argument("--max-logs", type=int, default=None, help="index only the first N logs (smoke test)")
    args = parser.parse_args(argv)
    manifest = build_index(args.metadata, navsim_splits.load(args.navsim_root), args.out, max_logs=args.max_logs)
    print(json.dumps({"counts": manifest["counts"], "runtime_s": manifest["runtime_s"]}, indent=1))


if __name__ == "__main__":
    main()
