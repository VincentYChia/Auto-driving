"""Rule-based meta-action labels derived from the ground-truth future (v0.1, thresholds [proposed]).

These give the slow reasoner a compact, checkable "decision" vocabulary and give the pilot a
decision token to condition on. They use privileged information (the recorded future), so they
are training targets only, never test-time inputs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

RULES_VERSION = "0.1.0"

LONGITUDINAL = ("stopped", "stop", "start", "accelerate", "decelerate", "keep_speed")
LATERAL = ("stationary", "turn_left", "turn_right", "lane_change_left", "lane_change_right",
           "curve_left", "curve_right", "straight")

STOP_SPEED = 0.3          # m/s: below this the car counts as standing still
SPEED_CHANGE_ABS = 1.0    # m/s: minimum change to call accelerate/decelerate ...
SPEED_CHANGE_REL = 0.15   # ... or this fraction of the current speed, whichever is larger
TURN_DEG = 25.0           # heading change at 4 s for a turn
CURVE_DEG = 8.0           # heading change at 4 s for a curve
LANE_CHANGE_MAX_DEG = 15.0
LANE_CHANGE_LATERAL_M = 1.5
MIN_FORWARD_M = 2.0       # below this forward travel the lateral label is "stationary"


def _speeds(future_xy: np.ndarray, dt: float) -> np.ndarray:
    pts = np.vstack([np.zeros((1, 2)), future_xy])
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt


def longitudinal_action(current_speed: float, future_xy, dt: float = 0.5, horizon_s: float = 3.0) -> str:
    future_xy = np.asarray(future_xy, dtype=np.float64).reshape(-1, 2)
    speeds = _speeds(future_xy, dt)
    end = min(int(round(horizon_s / dt)), len(speeds))
    v_end = float(np.mean(speeds[max(end - 2, 0) : end]))  # mean over the last second before the horizon
    v0 = float(current_speed)
    threshold = max(SPEED_CHANGE_ABS, SPEED_CHANGE_REL * v0)
    if v0 < STOP_SPEED and v_end < STOP_SPEED:
        return "stopped"
    if v_end < STOP_SPEED:
        return "stop"
    if v0 < STOP_SPEED:
        return "start"
    if v_end - v0 > threshold:
        return "accelerate"
    if v0 - v_end > threshold:
        return "decelerate"
    return "keep_speed"


def lateral_action(future_poses, dt: float = 0.5, horizon_s: float = 4.0) -> str:
    poses = np.asarray(future_poses, dtype=np.float64).reshape(-1, 3)
    idx = min(int(round(horizon_s / dt)), len(poses)) - 1
    x, y, yaw = poses[idx]
    if x < MIN_FORWARD_M:
        return "stationary"
    yaw_deg = math.degrees(yaw)
    if yaw_deg > TURN_DEG:
        return "turn_left"
    if yaw_deg < -TURN_DEG:
        return "turn_right"
    # A constant-curvature path ending at heading `yaw` has lateral offset x * tan(yaw / 2);
    # only the offset beyond that (a sideways shift without matching heading change) is a lane change.
    residual = y - x * math.tan(yaw / 2.0)
    if abs(yaw_deg) < LANE_CHANGE_MAX_DEG and residual > LANE_CHANGE_LATERAL_M:
        return "lane_change_left"
    if abs(yaw_deg) < LANE_CHANGE_MAX_DEG and residual < -LANE_CHANGE_LATERAL_M:
        return "lane_change_right"
    if yaw_deg > CURVE_DEG:
        return "curve_left"
    if yaw_deg < -CURVE_DEG:
        return "curve_right"
    return "straight"


def meta_action(sample: Dict[str, Any]) -> Dict[str, str]:
    """Longitudinal + lateral meta-action for a canonical sample."""
    dt = float(sample["timeline"]["dt_s"])
    vx, vy = sample["ego"]["history_velocity"][-1]
    future = np.asarray(sample["target"]["future_poses"], dtype=np.float64)
    return {
        "longitudinal": longitudinal_action(math.hypot(vx, vy), future[:, :2], dt),
        "lateral": lateral_action(future, dt),
        "rules_version": RULES_VERSION,
    }


def describe_command(one_hot: Sequence[int]) -> str:
    names = ("left", "straight", "right", "unknown")
    one_hot = list(one_hot)
    if len(one_hot) != 4 or sum(one_hot) != 1:
        return "unknown"
    return names[one_hot.index(1)]
