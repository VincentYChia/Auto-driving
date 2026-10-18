import json

import pytest

from data_pipeline import schema
from tests import synthetic_navsim as syn


def build(**kw):
    frames = syn.make_frames()
    return schema.build_sample(frames, 3, 10, "navtrain", "trainval", provenance={"extractor": "test"}, **kw)


def test_build_and_validate_round_trip():
    sample = build()
    schema.validate_sample(sample)
    again = json.loads(json.dumps(sample))  # must be plain JSON (no NaN, no numpy types)
    schema.validate_sample(again)
    assert again == sample
    assert "NaN" not in json.dumps(sample)


def test_fields_and_conventions():
    sample = build()
    assert sample["sample_id"] == "tok03"
    assert sample["timeline"]["history_offsets_s"] == [-1.5, -1.0, -0.5, 0.0]
    assert sample["timeline"]["future_offsets_s"][0] == 0.5 and len(sample["target"]["future_poses"]) == 10
    assert sample["ego"]["history_poses"][-1] == [0.0, 0.0, 0.0]
    assert sample["agents"]["source"] == schema.GT_AGENT_SOURCE
    assert {t["track_token"] for t in sample["agents"]["tracks"]} == {"track_car", "track_ped"}
    assert {t["track_token"] for t in sample["privileged"]["agents_future"]["tracks"]} == {"track_car", "track_ped"}
    assert sample["sensors"]["cameras"]["CAM_F0"][-1] == "synthetic_log/CAM_F0/0003.jpg"
    assert len(sample["sensors"]["lidar"]) == 4
    assert sample["privileged"]["roadblock_ids"] == ["rb1"]


def test_validate_rejects_broken_records():
    sample = build()
    sample["ego"]["history_poses"][-1] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        schema.validate_sample(sample)
    sample = build()
    sample["agents"]["tracks"][0]["states"].pop()
    with pytest.raises(ValueError):
        schema.validate_sample(sample)


def test_short_scene_rejected():
    with pytest.raises(ValueError):
        schema.build_sample(syn.make_frames(8), 3, 10, "navtrain", "trainval")
