"""Shared pytest fixtures. NAVSIM-dependent fixtures skip when the devkit is not importable."""

import pickle

import pytest

from tests.shared import synthetic_navsim as syn

NUM_FRAMES = 20  # one synthetic log; NAVSIM's 4 + 10 frame windows give 7 scenes


@pytest.fixture()
def synthetic_dataset(tmp_path):
    """A NAVSIM-format log on disk: (log_dir, sensor_dir, SceneFilter)."""
    pytest.importorskip("navsim")
    pytest.importorskip("nuplan")
    from navsim.common.dataclasses import SceneFilter

    logs = tmp_path / "navsim_logs" / "trainval"
    sensors = tmp_path / "sensor_blobs" / "trainval"
    logs.mkdir(parents=True)
    frames = syn.make_frames(NUM_FRAMES)
    with open(logs / "synthetic_log.pkl", "wb") as f:
        pickle.dump(frames, f)
    for frame in frames:
        path = sensors / frame["cams"]["CAM_F0"]["data_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    return logs, sensors, scene_filter
