"""What the driver did over the next 4 s, read from the recorded path: one maneuver label per sample.

Shared by both workstreams (``navsim_tools`` is the shared layer; neither workstream imports the other):
``navsim_eval/failure_analysis.py`` groups navtest scenes and training views by it, and
``data_pipeline/v1/token_sets.py`` builds the v1-xl-mix training set from it. Pure numpy.

Input: ``human`` (n, 8, 3), the poses (x forward, y left, yaw) at +0.5 … +4.0 s in the sample's own frame,
and ``speed`` (n,), the speed now in m/s. ``future_poses`` takes them from a stored path. The rules, in
priority order (each later rule overrides the earlier ones, so a turn wins over everything):

1. keep going, unless:
2. sideways shift: more than 2 m lateral at 4 s (a lane change or a curve);
3. brake to a stop: at least 2 m/s now, under 0.5 m/s over the last 0.5 s;
4. start from a stop / stay stopped: under 0.5 m/s now, and travels at least / less than 1 m in 4 s;
5. turn left / right: the heading changes by more than ``TURN_RAD`` (about 20 degrees).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

TURN_RAD = 0.35  # heading change over 4 s that counts as a turn (about 20 degrees)
SPEED_EDGES = [0.5, 4.0, 8.0, 12.0]  # m/s
SPEED_NAMES = ["stopped (<0.5 m/s)", "slow (0.5-4)", "medium (4-8)", "fast (8-12)", "very fast (>12)"]
HORIZON_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)  # the poses the labels read

TURN_LEFT = "turn left (>20 deg)"
TURN_RIGHT = "turn right (>20 deg)"
SIDEWAYS = "sideways shift (>2 m, lane change or curve)"
BRAKE = "brake to a stop"
START = "start from a stop"
STAY = "stay stopped"
KEEP = "keep going"
LABELS = (TURN_LEFT, TURN_RIGHT, SIDEWAYS, BRAKE, START, STAY, KEEP)


def maneuver(human: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """What the human did over the next 4 s, one label per scene (later assignments take priority)."""
    heading = np.arctan2(np.sin(human[:, -1, 2]), np.cos(human[:, -1, 2]))
    travelled = np.linalg.norm(human[:, -1, :2], axis=1)
    end_speed = np.linalg.norm(human[:, -1, :2] - human[:, -2, :2], axis=1) / 0.5
    lateral = human[:, -1, 1]
    label = np.full(len(human), KEEP, dtype=object)
    label[np.abs(lateral) > 2.0] = SIDEWAYS
    label[(speed >= 2.0) & (end_speed < 0.5)] = BRAKE
    label[(speed < 0.5) & (travelled >= 1.0)] = START
    label[(speed < 0.5) & (travelled < 1.0)] = STAY
    label[heading > TURN_RAD] = TURN_LEFT
    label[heading < -TURN_RAD] = TURN_RIGHT
    return label


def speed_band(speed: np.ndarray) -> np.ndarray:
    return np.array(SPEED_NAMES, dtype=object)[np.digitize(speed, SPEED_EDGES)]


def future_poses(path_xyyaw: np.ndarray, offsets_s: Sequence[float]) -> np.ndarray:
    """The poses at ``HORIZON_S`` of each row's stored path, as float64 (n, 8, 3): ``maneuver``'s ``human``.

    ``path_xyyaw`` (n, k, 3) holds the poses at ``offsets_s`` (k,) seconds from the anchor, e.g. the v1
    index's ``path_xyyaw`` with ``openscene_index.PATH_OFFSETS_S`` (float32, so the values are exact).
    """
    offsets = np.asarray(offsets_s, dtype=np.float64)
    columns = []
    for t in HORIZON_S:
        hit = np.flatnonzero(np.isclose(offsets, t))
        if len(hit) != 1:
            raise ValueError(f"the path has no single pose at +{t} s (offsets {offsets.tolist()})")
        columns.append(int(hit[0]))
    return np.asarray(path_xyyaw)[:, columns, :].astype(np.float64)
