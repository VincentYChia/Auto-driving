"""MVLAD-AD planning prompt, rebuilt for NAVSIM inputs.

The text matches what the shipped planning LoRA saw during training: the "long" prompt from
``data_prep/gen_data_planning.py`` after the instruction rewrite in ``train/llava/train/train.py``
(``tests/test_mvlad_prompt.py`` checks this against both files).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from navsim_tools import geometry

MVLAD_HISTORY_POINTS = 10  # MVLAD-AD was trained with 10 past waypoints at 0.5 s (t = -4.5 s ... 0 s)

SYSTEM_MESSAGE = (
    "You are an autonomous driving labeller. You are given a sequence of front-view images, "
    "the 5-second historical ego waypoints in the *current ego frame* (x-forward, y-left), "
    "and a driving rationale. Your task is to predict the future ego-frame waypoints for the next "
    "10 timesteps (≈0.5 s interval). Provide raw coordinates only.\n"
)

USER_TEMPLATE = (
    "This is a frame from a front camera video captured.\n\n"
    "The 5-second historical ego-frame waypoints (relative to the last observed frame) are {history}.\n\n"
    "Generate the predicted ten future waypoints in the action token format:\n    "
)


def format_waypoints(points_xy, precision: int = 2) -> str:
    return ", ".join(f"[{x:.{precision}f}, {y:.{precision}f}]" for x, y in np.asarray(points_xy).reshape(-1, 2))


def history_for_prompt(history_xy: Sequence[Sequence[float]], mode: str = "pad_cv") -> np.ndarray:
    """Adapt NAVSIM's short history (4 frames = 1.5 s) to MVLAD-AD's 10-point format.

    ``pad_cv``: backfill the missing older points at constant velocity (keeps the trained format).
    ``available``: pass only the frames NAVSIM provides.
    """
    hist = np.asarray(history_xy, dtype=np.float64).reshape(-1, 2)
    if mode == "pad_cv":
        return geometry.backfill_history(hist, MVLAD_HISTORY_POINTS)
    if mode == "available":
        return hist
    raise ValueError(f"unknown history mode: {mode}")


def build_human_message(history_xy, mode: str = "pad_cv") -> str:
    history = format_waypoints(history_for_prompt(history_xy, mode))
    return SYSTEM_MESSAGE + "<image>" + USER_TEMPLATE.format(history=history)
