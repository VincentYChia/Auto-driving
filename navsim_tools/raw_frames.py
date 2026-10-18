"""Read NAVSIM raw frame dicts (the per-frame entries of ``SceneLoader.scene_frames_dicts``).

Field names and frame conventions are taken from ``navsim/common/dataclasses.py`` and
``navsim/planning/scenario_builder/navsim_scenario_utils.py``:

* ego pose: ``ego2global_translation`` + ``ego2global_rotation`` (w, x, y, z quaternion)
* ``ego_dynamic_state`` = [vx, vy, ax, ay] in that frame's ego axes
* annotations (``anns``): ``gt_boxes`` rows are [x, y, z, length, width, height, yaw] in that
  frame's ego frame; ``gt_velocity_3d`` is in that frame's ego axes (NAVSIM rotates it by the
  ego heading to get global velocity); ``track_tokens`` persist across frames.

Everything here is pure numpy so it can be unit-tested without the devkit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from navsim_tools import geometry

BOX_X, BOX_Y, BOX_Z, BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT, BOX_YAW = range(7)


def global_ego_pose(frame: Dict[str, Any]) -> np.ndarray:
    tx, ty = frame["ego2global_translation"][:2]
    return np.array([tx, ty, geometry.yaw_from_quaternion(frame["ego2global_rotation"])], dtype=np.float64)


def ego_velocity(frame: Dict[str, Any]) -> np.ndarray:
    return np.asarray(frame["ego_dynamic_state"][:2], dtype=np.float64)


def ego_acceleration(frame: Dict[str, Any]) -> np.ndarray:
    return np.asarray(frame["ego_dynamic_state"][2:4], dtype=np.float64)


def driving_command(frame: Dict[str, Any]) -> List[int]:
    return [int(v) for v in np.asarray(frame["driving_command"]).reshape(-1)]


def camera_paths(frame: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Camera name -> image path relative to the sensor-blobs root (None if missing)."""
    out: Dict[str, Optional[str]] = {}
    for name, cam in frame.get("cams", {}).items():
        out[name] = cam.get("data_path") if cam else None
    return out


def camera_calibration(frame: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, cam in frame.get("cams", {}).items():
        if not cam:
            continue
        out[name] = {
            key: np.asarray(cam[key], dtype=np.float64).tolist()
            for key in ("cam_intrinsic", "sensor2lidar_rotation", "sensor2lidar_translation", "distortion")
            if cam.get(key) is not None
        }
    return out


def ego_history_local(frames: Sequence[Dict[str, Any]], current_idx: int) -> np.ndarray:
    """Ego poses of frames ``0..current_idx`` in the current ego frame (oldest first)."""
    poses = np.stack([global_ego_pose(f) for f in frames[: current_idx + 1]])
    return geometry.absolute_to_relative(poses[-1], poses)


def future_poses_local(frames: Sequence[Dict[str, Any]], current_idx: int, num_future: int) -> np.ndarray:
    """Ego poses of the next ``num_future`` frames in the current ego frame (NAVSIM's GT target)."""
    origin = global_ego_pose(frames[current_idx])
    future = np.stack([global_ego_pose(f) for f in frames[current_idx + 1 : current_idx + 1 + num_future]])
    return geometry.absolute_to_relative(origin, future)


def _finite_or_none(values) -> List[Optional[float]]:
    return [None if not math.isfinite(float(v)) else float(v) for v in values]


def agent_tracks(
    frames: Sequence[Dict[str, Any]],
    current_idx: int,
    frame_indices: Sequence[int],
    max_range: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Ground-truth objects over ``frame_indices``, ego-motion compensated into the current frame.

    Returns ``track_token -> {"category", "size_lwh", "states"}``. ``states[j]`` belongs to
    ``frame_indices[j]`` and is ``[x, y, z, yaw, vx, vy]`` in the current ego frame (velocity
    rotated into current-frame axes; components may be None if the log has NaN), or None
    when the object is not annotated in that frame. ``max_range`` keeps only tracks that come
    within that distance (metres) of the current ego position in any of the given frames.
    """
    origin = global_ego_pose(frames[current_idx])
    tracks: Dict[str, Dict[str, Any]] = {}
    for slot, frame_idx in enumerate(frame_indices):
        frame = frames[frame_idx]
        anns = frame.get("anns") or {}
        boxes = np.asarray(anns.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64).reshape(-1, 7)
        if len(boxes) == 0:
            continue
        names = list(anns["gt_names"])
        velocities = np.asarray(anns["gt_velocity_3d"], dtype=np.float64).reshape(len(boxes), -1)
        tokens = list(anns["track_tokens"])
        frame_pose = global_ego_pose(frame)
        local_se2 = boxes[:, [BOX_X, BOX_Y, BOX_YAW]]
        in_current = geometry.change_frame(local_se2, frame_pose, origin)
        vel_current = geometry.rotate_vectors(velocities[:, :2], frame_pose[2] - origin[2])
        for i, token in enumerate(tokens):
            entry = tracks.get(token)
            if entry is None:
                entry = {
                    "category": str(names[i]),
                    "size_lwh": [float(boxes[i, BOX_LENGTH]), float(boxes[i, BOX_WIDTH]), float(boxes[i, BOX_HEIGHT])],
                    "states": [None] * len(frame_indices),
                    "_min_dist": math.inf,
                }
                tracks[token] = entry
            x, y, yaw = in_current[i]
            entry["states"][slot] = [float(x), float(y), float(boxes[i, BOX_Z]), float(yaw)] + _finite_or_none(
                vel_current[i]
            )
            entry["_min_dist"] = min(entry["_min_dist"], float(math.hypot(x, y)))
    out: Dict[str, Dict[str, Any]] = {}
    for token, entry in tracks.items():
        if max_range is not None and entry["_min_dist"] > max_range:
            continue
        entry.pop("_min_dist")
        out[token] = entry
    return out
