import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft"))

import predict_navsim  # noqa: E402
from data_pipeline.v1 import pilot_view  # noqa: E402


def test_navsim_record_gets_the_training_prompt():
    offsets = pilot_view.PATH_OFFSETS_S
    path = np.zeros((len(offsets), 3))
    path[:, 0] = offsets * 7.3
    path[:, 1] = offsets ** 2 * 0.2
    path[:, 2] = offsets * -0.06
    frames = {"frame_id": ["f"], "log_name": ["log"], "token": ["t"], "our_split": ["dev"], "city": ["c"],
              "path_xyyaw": [path], "speed": [math.hypot(7.3, 0.4)], "accel_lon": [-0.62], "command": ["left"]}
    rows = [int(np.flatnonzero(np.isclose(offsets, t))[0]) for t in (-1.5, -1.0, -0.5, 0.0)]
    record = {"id": "t", "navsim": {"ego_history_local": path[rows].tolist(), "ego_velocity": [7.3, 0.4],
                                    "ego_acceleration": [-0.62, 0.1], "driving_command": [1, 0, 0, 0]}}
    ours = predict_navsim.item(record)
    assert ours["prompt"] == pilot_view.sample(frames, 0)["prompt"] and ours["system"] == pilot_view.SYSTEM
    assert "Route command: turn left." in ours["prompt"]
    assert predict_navsim.constant_velocity(record)[-1][0] == pytest.approx(4.0 * math.hypot(7.3, 0.4), abs=1e-3)
