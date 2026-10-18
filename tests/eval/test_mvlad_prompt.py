"""The NAVSIM prompt must equal what MVLAD-AD's planning LoRA saw in training."""

import re
from pathlib import Path

import numpy as np

from navsim_eval import mvlad_prompt
from navsim_eval.predictions import load_predictions, poses_from_waypoints

REPO = Path(__file__).resolve().parents[2]


def _training_replacement():
    src = (REPO / "train/llava/train/train.py").read_text()
    old = re.search(r'^\s*old = "(.*)"$', src, flags=re.M).group(1)
    new = re.search(r'^\s*new = "(.*)"$', src, flags=re.M).group(1)
    return old, new


def _original_prompt(history: str) -> str:
    """Rebuild data_prep/gen_data_planning.py's long prompt (FUT_LEN = 10) and check it against the file."""
    src = (REPO / "data_prep/gen_data_planning.py").read_text()
    assert "Generate the predicted future waypoints in the format [x_1, y_1], [x_2, y_2], ..., [x_{FUT_LEN}, y_{FUT_LEN}]" in src
    system = (
        "You are an autonomous driving labeller. You are given a sequence of front-view images, "
        "the 5-second historical ego waypoints in the *current ego frame* (x-forward, y-left), "
        "and a driving rationale. Your task is to predict the future ego-frame waypoints for the next "
        "10 timesteps (≈0.5 s interval). Provide raw coordinates only.\n"
    )
    user = (
        "This is a frame from a front camera video captured.\n\n"
        f"The 5-second historical ego-frame waypoints (relative to the last observed frame) are {history}.\n\n"
        "Generate the predicted future waypoints in the format [x_1, y_1], [x_2, y_2], ..., [x_10, y_10]. "
        "Write the raw text, not markdown or LaTeX. Future waypoints:\n    "
    )
    for fragment in ("You are an autonomous driving labeller.", "This is a frame from a front camera video captured.",
                     "Provide raw coordinates only."):
        assert fragment in src
    return system + "<image>" + user


def test_prompt_matches_training_time_text():
    hist4 = np.array([[-15.0, -0.3], [-10.0, -0.1], [-5.0, 0.0], [0.0, 0.0]])
    ours = mvlad_prompt.build_human_message(hist4, mode="pad_cv")
    history_text = mvlad_prompt.format_waypoints(mvlad_prompt.history_for_prompt(hist4, "pad_cv"))
    old, new = _training_replacement()
    assert _original_prompt(history_text).replace(old, new) == ours
    assert history_text.count("[") == mvlad_prompt.MVLAD_HISTORY_POINTS
    assert history_text.endswith("[0.00, 0.00]")


def test_available_mode_keeps_navsim_history():
    hist4 = np.array([[-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    assert mvlad_prompt.format_waypoints(mvlad_prompt.history_for_prompt(hist4, "available")).count("[") == 4


def test_predictions_round_trip(tmp_path):
    shard = tmp_path / "shard_000_of_001.jsonl"
    shard.write_text('{"token": "a", "waypoints": [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0]]}\n')
    (tmp_path / "errors_000_of_001.jsonl").write_text('{"token": "b", "error": "x"}\n')
    preds = load_predictions(str(tmp_path))
    assert set(preds) == {"a"}
    poses = poses_from_waypoints(preds["a"], 8)
    assert poses.shape == (8, 3) and poses.dtype == np.float32
    np.testing.assert_allclose(poses[:, 2], 0.0)
