"""Shared NAVSIM access layer against the real devkit (skipped without NAVSIM)."""

import pytest

pytest.importorskip("navsim")
pytest.importorskip("nuplan")

from navsim_tools import loader  # noqa: E402
from tests.conftest import NUM_FRAMES  # noqa: E402


def test_split_configs_load():
    navtest_filter, data_split = loader.load_split("navtest")
    assert data_split == "test" and navtest_filter.num_history_frames == 4
    assert len(navtest_filter.tokens) > 10000
    navtrain_filter, data_split = loader.load_split("navtrain")
    assert data_split == "trainval" and len(navtrain_filter.tokens) > 100000


def test_scene_loader_windows(synthetic_dataset):
    logs, sensors, scene_filter = synthetic_dataset
    scene_loader = loader.make_scene_loader(logs, sensors, scene_filter)
    assert sorted(scene_loader.tokens) == [f"tok{i:02d}" for i in range(3, 3 + NUM_FRAMES - 13)]
