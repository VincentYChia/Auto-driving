"""Read saved trajectory predictions and turn them into NAVSIM poses (pure numpy)."""

from __future__ import annotations

import glob
import json
import os
from typing import Dict

import numpy as np

from navsim_tools import geometry


def load_predictions(path: str) -> Dict[str, np.ndarray]:
    """Read ``{token: (N, 2) waypoints}`` from a JSON file, a JSONL file, or a directory of ``*.jsonl`` shards.

    JSONL rows look like ``{"token": ..., "waypoints": [[x, y], ...]}``; a JSON file maps
    token -> waypoints. Later files win on duplicate tokens.
    """
    files = sorted(glob.glob(os.path.join(path, "*.jsonl"))) if os.path.isdir(path) else [path]
    files = [f for f in files if not os.path.basename(f).startswith("errors_")]
    preds: Dict[str, np.ndarray] = {}
    for file in files:
        with open(file) as f:
            if file.endswith(".jsonl"):
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                rows = [{"token": k, "waypoints": v} for k, v in json.load(f).items()]
        for row in rows:
            if "waypoints" in row:
                preds[row["token"]] = np.asarray(row["waypoints"], dtype=np.float64).reshape(-1, 2)
    return preds


def poses_from_waypoints(waypoints_xy: np.ndarray, num_poses: int) -> np.ndarray:
    """First ``num_poses`` waypoints as ``(num_poses, 3)`` float32 poses with travel-direction headings."""
    waypoints_xy = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
    if len(waypoints_xy) < num_poses:
        raise ValueError(f"prediction has {len(waypoints_xy)} waypoints, need {num_poses}")
    return geometry.waypoints_to_poses(waypoints_xy[:num_poses]).astype(np.float32)
