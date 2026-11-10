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


def _scene():
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
    return frames, record


def test_v12_navsim_record_gets_the_training_prompt():
    frames, record = _scene()
    ours = predict_navsim.item(record, "pilot_v1.2")
    assert ours["prompt"] == pilot_view.sample(frames, 0, "pilot_v1.2")["prompt"]
    line = next(l for l in ours["prompt"].split("\n") if l.startswith("Past poses"))
    # 4 poses (x, y, heading) at -1.5, -1.0, -0.5, 0 s, 2 decimals, the last the origin
    assert line.endswith("(-10.95, 0.45, 0.09) (-7.30, 0.20, 0.06) (-3.65, 0.05, 0.03) (0.00, 0.00, 0.00).")


def test_v11_unchanged_by_the_v12_option():
    frames, record = _scene()
    assert pilot_view.sample(frames, 0)["prompt"] == pilot_view.sample(frames, 0, "pilot_v1.1")["prompt"]
    path = frames["path_xyyaw"][0]  # the pre-1.2 formula, verbatim
    hist = path[[int(np.flatnonzero(np.isclose(pilot_view.PATH_OFFSETS_S, t))[0]) for t in pilot_view.HISTORY_S]]
    old = pilot_view.PROMPT.format(speed=float(frames["speed"][0]), accel=round(-0.62, 1) + 0.0,
                                   yaw_rate=round(pilot_view.past_yaw_rate(path), 2) + 0.0,
                                   history=pilot_view._pairs(hist[:, :2] + 0.0), command="turn left")
    assert pilot_view.sample(frames, 0)["prompt"] == old
    assert predict_navsim.item(record) == predict_navsim.item(record, "pilot_v1.1")


def test_run_prompt_version_reads_run_json(tmp_path):
    import json
    (tmp_path / "adapter").mkdir()
    assert predict_navsim.run_prompt_version(str(tmp_path / "adapter")) == "pilot_v1.1"
    (tmp_path / "run.json").write_text(json.dumps({"prompt_version": "pilot_v1.2"}))
    assert predict_navsim.run_prompt_version(str(tmp_path / "adapter")) == "pilot_v1.2"
    assert predict_navsim.run_prompt_version(str(tmp_path / "epoch_1")) == "pilot_v1.2"


def test_baselines_and_probes_read_v12_poses():
    import baselines
    frames, _ = _scene()
    v11 = {"prompt": pilot_view.sample(frames, 0, "pilot_v1.1")["prompt"]}
    v12 = {"prompt": pilot_view.sample(frames, 0, "pilot_v1.2")["prompt"]}
    a, b = baselines.prompt_numbers(v11), baselines.prompt_numbers(v12)
    assert a[:3] == b[:3] and np.allclose(a[3], b[3], atol=0.051) and len(b[3]) == 6
