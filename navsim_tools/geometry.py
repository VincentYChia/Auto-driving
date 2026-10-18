"""SE(2) helpers matching NAVSIM's conventions.

Frames follow NAVSIM: poses are ``(x, y, yaw)``, x forward, y left, yaw in radians
(counter-clockwise), and "local" means relative to the ego rear axle at the current frame.
``absolute_to_relative`` mirrors ``convert_absolute_to_relative_se2_array`` in
``navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py``.
"""

from __future__ import annotations

import numpy as np


def normalize_angle(angle):
    """Wrap angles to [-pi, pi], as NAVSIM's ``normalize_angle`` does."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_from_quaternion(wxyz) -> float:
    """Yaw of a ``(w, x, y, z)`` quaternion; equals ``pyquaternion.Quaternion(*q).yaw_pitch_roll[0]``."""
    w, x, y, z = (float(v) for v in wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def absolute_to_relative(origin, poses) -> np.ndarray:
    """Express global ``(N, 3)`` poses in the frame of the global ``origin`` pose."""
    origin = np.asarray(origin, dtype=np.float64)
    rel = np.array(poses, dtype=np.float64, copy=True).reshape(-1, 3) - origin[None, :]
    rel[:, :2] = rel[:, :2] @ rotation(-origin[2]).T
    rel[:, 2] = normalize_angle(rel[:, 2])
    return rel


def relative_to_absolute(origin, local_poses) -> np.ndarray:
    """Inverse of :func:`absolute_to_relative`."""
    origin = np.asarray(origin, dtype=np.float64)
    local = np.asarray(local_poses, dtype=np.float64).reshape(-1, 3)
    out = np.empty_like(local)
    out[:, :2] = local[:, :2] @ rotation(origin[2]).T + origin[None, :2]
    out[:, 2] = normalize_angle(local[:, 2] + origin[2])
    return out


def change_frame(local_poses, source_origin, target_origin) -> np.ndarray:
    """Re-express poses given in the frame of ``source_origin`` in the frame of ``target_origin``.

    Both origins are global ``(x, y, yaw)`` poses. Used to move objects annotated in a past
    frame's ego frame into the current ego frame (ego-motion compensation).
    """
    return absolute_to_relative(target_origin, relative_to_absolute(source_origin, local_poses))


def rotate_vectors(vectors, angle: float) -> np.ndarray:
    """Rotate ``(N, 2)`` vectors by ``angle`` (radians, counter-clockwise)."""
    v = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    return v @ rotation(angle).T


def waypoints_to_poses(waypoints_xy, min_step: float = 0.05) -> np.ndarray:
    """Turn ``(N, 2)`` local waypoints into ``(N, 3)`` poses by adding a heading.

    The heading of waypoint ``i`` is the direction of travel from waypoint ``i-1`` (the
    origin for ``i = 0``). Steps shorter than ``min_step`` metres keep the previous heading
    (0 at the start), so a stationary plan does not get a random heading.
    """
    xy = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
    poses = np.zeros((len(xy), 3), dtype=np.float64)
    poses[:, :2] = xy
    prev_xy, prev_heading = np.zeros(2), 0.0
    for i, point in enumerate(xy):
        step = point - prev_xy
        if np.hypot(step[0], step[1]) >= min_step:
            prev_heading = float(np.arctan2(step[1], step[0]))
        poses[i, 2] = prev_heading
        prev_xy = point
    return poses


def backfill_history(history_xy, total: int) -> np.ndarray:
    """Extend an oldest-first ``(K, 2)`` history to ``total`` points by constant-velocity backfill.

    The velocity is the mean displacement per step over the available history. With a single
    point (or no motion) the backfilled points repeat the oldest point. Returns oldest-first.
    """
    hist = np.asarray(history_xy, dtype=np.float64).reshape(-1, 2)
    missing = total - len(hist)
    if missing <= 0:
        return hist[-total:]
    if len(hist) >= 2:
        per_step = (hist[-1] - hist[0]) / (len(hist) - 1)
    else:
        per_step = np.zeros(2)
    steps = np.arange(missing, 0, -1, dtype=np.float64)[:, None]
    backfill = hist[0][None, :] - steps * per_step[None, :]
    return np.concatenate([backfill, hist], axis=0)
