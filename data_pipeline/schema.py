"""Canonical sample record (schema v0.1). One JSON object per NAVSIM scene token.

Design rules (see ``docs/DATASET_SCHEMA.md``):

* store raw continuous values; tokenization and prompts are derived views;
* everything is in the current ego frame (NAVSIM local: rear axle, x forward, y left, yaw rad);
* inputs an agent may use at test time live outside ``privileged``; anything else (future
  object states, route, traffic lights) lives inside it and must never reach a student's input;
* each record says where its object tracks came from (``agents.source``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from navsim_tools import raw_frames

SCHEMA_VERSION = "0.1.0"
DT_S = 0.5
GT_AGENT_SOURCE = "gt_annotations"


def _r(values, digits: int) -> List[float]:
    return [round(float(v), digits) for v in np.asarray(values).reshape(-1)]


def _round_state(state: Optional[List[Optional[float]]]) -> Optional[List[Optional[float]]]:
    if state is None:
        return None
    digits = (3, 3, 3, 4, 3, 3)  # x, y, z, yaw, vx, vy
    return [None if v is None else round(float(v), d) for v, d in zip(state, digits)]


def _tracks_list(tracks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for token, entry in sorted(tracks.items()):
        out.append(
            {
                "track_token": token,
                "category": entry["category"],
                "size_lwh": _r(entry["size_lwh"], 3),
                "states": [_round_state(s) for s in entry["states"]],
            }
        )
    return out


def build_sample(
    frames: Sequence[Dict[str, Any]],
    current_idx: int,
    num_future: int,
    split: str,
    data_split: str,
    max_range_m: Optional[float] = 100.0,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one canonical record from a NAVSIM scene's raw frame dicts."""
    if len(frames) < current_idx + 1 + num_future:
        raise ValueError(f"scene has {len(frames)} frames, need {current_idx + 1 + num_future}")
    history_idx = list(range(current_idx + 1))
    future_idx = list(range(current_idx + 1, current_idx + 1 + num_future))
    current = frames[current_idx]

    history_poses = raw_frames.ego_history_local(frames, current_idx)
    future_poses = raw_frames.future_poses_local(frames, current_idx, num_future)
    hist_tracks = raw_frames.agent_tracks(frames, current_idx, history_idx, max_range=max_range_m)
    fut_tracks = raw_frames.agent_tracks(frames, current_idx, future_idx, max_range=max_range_m)

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": current["token"],
        "source": {
            "dataset": "navsim",
            "split": split,
            "data_split": data_split,
            "log_name": current["log_name"],
            "scene_token": current.get("scene_token"),
            "map_location": current.get("map_location"),
            "timestamp_us": int(current["timestamp"]),
        },
        "timeline": {
            "dt_s": DT_S,
            "history_offsets_s": [round((i - current_idx) * DT_S, 2) for i in history_idx],
            "future_offsets_s": [round((i - current_idx) * DT_S, 2) for i in future_idx],
            "history_tokens": [frames[i]["token"] for i in history_idx],
            "future_tokens": [frames[i]["token"] for i in future_idx],
        },
        "ego": {
            "history_poses": [_r(p, 4) for p in history_poses],
            "history_velocity": [_r(raw_frames.ego_velocity(frames[i]), 3) for i in history_idx],
            "history_acceleration": [_r(raw_frames.ego_acceleration(frames[i]), 3) for i in history_idx],
            "history_driving_command": [raw_frames.driving_command(frames[i]) for i in history_idx],
        },
        "target": {"future_poses": [_r(p, 4) for p in future_poses]},
        "agents": {
            "source": GT_AGENT_SOURCE,
            "frame_offsets_s": [round((i - current_idx) * DT_S, 2) for i in history_idx],
            "max_range_m": max_range_m,
            "tracks": _tracks_list(hist_tracks),
        },
        "privileged": {
            "agents_future": {
                "frame_offsets_s": [round((i - current_idx) * DT_S, 2) for i in future_idx],
                "tracks": _tracks_list(fut_tracks),
            },
            "roadblock_ids": list(current.get("roadblock_ids") or []),
            "traffic_lights": [[str(lane), bool(red)] for lane, red in (current.get("traffic_lights") or [])],
        },
        "sensors": {
            "data_split": data_split,
            "cameras": {
                name: [raw_frames.camera_paths(frames[i]).get(name) for i in history_idx]
                for name in sorted(raw_frames.camera_paths(current))
            },
            "lidar": [frames[i].get("lidar_path") for i in history_idx],
            "calibration": raw_frames.camera_calibration(current),
        },
        "provenance": dict(provenance or {}),
    }


def _finite(values) -> bool:
    return all(v is None or math.isfinite(v) for v in values)


def validate_sample(sample: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if a record breaks the v0.1 structure."""
    def fail(msg: str) -> None:
        raise ValueError(f"{sample.get('sample_id')}: {msg}")

    if sample.get("schema_version") != SCHEMA_VERSION:
        fail(f"schema_version {sample.get('schema_version')} != {SCHEMA_VERSION}")
    n_hist = len(sample["timeline"]["history_offsets_s"])
    n_fut = len(sample["timeline"]["future_offsets_s"])
    ego = sample["ego"]
    for key in ("history_poses", "history_velocity", "history_acceleration", "history_driving_command"):
        if len(ego[key]) != n_hist:
            fail(f"ego.{key} has {len(ego[key])} entries, expected {n_hist}")
    last = ego["history_poses"][-1]
    if any(abs(v) > 1e-3 for v in last):
        fail(f"current ego pose should be the origin, got {last}")
    if len(sample["target"]["future_poses"]) != n_fut:
        fail("target.future_poses length mismatch")
    for pose in ego["history_poses"] + sample["target"]["future_poses"]:
        if len(pose) != 3 or not _finite(pose):
            fail(f"bad pose {pose}")
    for group, n in ((sample["agents"], n_hist), (sample["privileged"]["agents_future"], n_fut)):
        for track in group["tracks"]:
            if len(track["states"]) != n:
                fail(f"track {track['track_token']} has {len(track['states'])} states, expected {n}")
            for state in track["states"]:
                if state is not None and (len(state) != 6 or not _finite(state)):
                    fail(f"bad state {state} for track {track['track_token']}")
    for name, paths in sample["sensors"]["cameras"].items():
        if len(paths) != n_hist:
            fail(f"camera {name} has {len(paths)} paths, expected {n_hist}")
