"""NAVSIM evaluation workstream against the real devkit (skipped without NAVSIM)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("navsim")
pytest.importorskip("nuplan")

from navsim_eval import export_mvlad_inputs  # noqa: E402
from navsim_tools import loader  # noqa: E402


def test_export_matches_agent_input(synthetic_dataset):
    logs, sensors, scene_filter = synthetic_dataset
    scene_loader = loader.make_scene_loader(logs, sensors, scene_filter)
    records, missing, gap = export_mvlad_inputs.build_records(scene_loader, 4, sensors, "synthetic", "pad_cv")
    assert missing == 0 and gap < 1e-6
    first = records[0]
    assert first["id"] == "tok03" and first["image"].endswith("synthetic_log/CAM_F0/0003.jpg")
    assert first["conversations"][0]["value"].count("[") == 10
    assert first["navsim"]["driving_command"] == [0, 1, 0, 0]


def test_precomputed_agent_and_hydra_override(tmp_path):
    import navsim
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

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
                "agent._target_=navsim_eval.precomputed_agent.PrecomputedTrajectoryAgent",
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
