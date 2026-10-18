"""Runs the NAVSIM-facing code against the real devkit on a synthetic log.

Skipped unless the NAVSIM devkit is importable (it needs its own Python 3.9 environment).
"""

import gzip
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("navsim")
pytest.importorskip("nuplan")

from navsim.common.dataclasses import SceneFilter  # noqa: E402

from data_pipeline import extract_navsim, schema  # noqa: E402
from navsim_tools import export_mvlad_inputs, loader  # noqa: E402
from tests import synthetic_navsim as syn  # noqa: E402

NUM_FRAMES = 20


@pytest.fixture()
def synthetic_dataset(tmp_path):
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


def test_split_configs_load():
    navtest_filter, data_split = loader.load_split("navtest")
    assert data_split == "test" and navtest_filter.num_history_frames == 4
    assert len(navtest_filter.tokens) > 10000
    navtrain_filter, data_split = loader.load_split("navtrain")
    assert data_split == "trainval" and len(navtrain_filter.tokens) > 100000


def test_export_matches_agent_input(synthetic_dataset):
    logs, sensors, scene_filter = synthetic_dataset
    scene_loader = loader.make_scene_loader(logs, sensors, scene_filter)
    assert sorted(scene_loader.tokens) == [f"tok{i:02d}" for i in range(3, 3 + NUM_FRAMES - 13)]
    records, missing, gap = export_mvlad_inputs.build_records(scene_loader, 4, sensors, "synthetic", "pad_cv")
    assert missing == 0 and gap < 1e-6
    first = records[0]
    assert first["id"] == "tok03" and first["image"].endswith("synthetic_log/CAM_F0/0003.jpg")
    assert first["conversations"][0]["value"].count("[") == 10
    assert first["navsim"]["driving_command"] == [0, 1, 0, 0]


def test_extract_batch_with_navsim_crosscheck(synthetic_dataset, tmp_path):
    logs, sensors, scene_filter = synthetic_dataset
    out_dir = tmp_path / "canonical"
    out_dir.mkdir()
    result = extract_navsim._process_batch(
        {
            "batch_id": 0, "log_names": ["synthetic_log"], "scene_filter": scene_filter,
            "log_path": str(logs), "sensor_path": str(sensors), "out_dir": str(out_dir),
            "split": "synthetic", "data_split": "trainval", "max_range_m": 100.0,
            "provenance": {"extractor": "test"}, "overwrite": False,
        }
    )
    assert result["status"] == "written" and result["samples"] == NUM_FRAMES - 13 and not result["errors"]
    with gzip.open(out_dir / "samples-00000.jsonl.gz", "rt") as f:
        samples = [json.loads(line) for line in f]
    for sample in samples:
        schema.validate_sample(sample)
    assert samples[0]["sample_id"] == "tok03"
    # rerun is a no-op (resume)
    assert extract_navsim._process_batch({**result, **{
        "batch_id": 0, "log_names": ["synthetic_log"], "scene_filter": scene_filter, "log_path": str(logs),
        "sensor_path": str(sensors), "out_dir": str(out_dir), "split": "synthetic", "data_split": "trainval",
        "max_range_m": 100.0, "provenance": {}, "overwrite": False}})["status"] == "exists"


def test_precomputed_agent_and_hydra_override(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import navsim

    preds = tmp_path / "preds"
    preds.mkdir()
    waypoints = [[float(i), 0.1 * i] for i in range(1, 11)]
    (preds / "shard_000_of_001.jsonl").write_text(json.dumps({"token": "tokA", "waypoints": waypoints}) + "\n")

    config_dir = str(navsim.__path__[0] + "/planning/script/config/pdm_scoring")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[
                "train_test_split=navtest",
                "experiment_name=test",
                "agent._target_=navsim_tools.precomputed_agent.PrecomputedTrajectoryAgent",
                f"+agent.predictions_path={preds}",
            ],
        )
    agent = instantiate(cfg.agent)
    agent.initialize()
    assert agent.requires_scene and agent.name() == "PrecomputedTrajectoryAgent"
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token="tokA"))
    trajectory = agent.compute_trajectory(agent_input=None, scene=scene)
    assert trajectory.poses.shape == (8, 3)
    np.testing.assert_allclose(trajectory.poses[:, :2], np.asarray(waypoints[:8]), atol=1e-6)
    with pytest.raises(KeyError):
        agent.compute_trajectory(agent_input=None, scene=SimpleNamespace(scene_metadata=SimpleNamespace(initial_token="missing")))
