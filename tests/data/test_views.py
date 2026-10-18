import json

import pytest

from data_pipeline import motion_text, schema, views
from data_pipeline.label_client import parse_label
from tests.shared import synthetic_navsim as syn

ROOT = "/data/sensor_blobs/trainval"


@pytest.fixture()
def sample():
    return schema.build_sample(syn.make_frames(), 3, 10, "navtrain", "trainval")


def test_pilot_variants(sample):
    front = views.pilot_example(sample, ROOT, "front")
    motion = views.pilot_example(sample, ROOT, "front_motion")
    history = views.pilot_example(sample, ROOT, "front_history")
    assert front["images"] == [f"{ROOT}/synthetic_log/CAM_F0/0003.jpg"]
    assert len(history["images"]) == 4 and history["messages"][0]["content"].startswith("<image>" * 4)
    assert "Nearby objects" in motion["messages"][0]["content"]
    assert "Nearby objects" not in front["messages"][0]["content"]
    assert len(front["target_poses"]) == views.NUM_TARGET_POSES
    assert front["messages"][1]["content"].count("[") == views.NUM_TARGET_POSES
    assert "Route command: straight." in front["messages"][0]["content"]
    with pytest.raises(ValueError):
        views.pilot_example(sample, ROOT, "nope")


def test_motion_text_lists_nearest_dynamic_objects(sample):
    rows = motion_text.summarize_tracks(sample["agents"], max_objects=5)
    assert [r["category"] for r in rows] == ["pedestrian", "vehicle"]  # nearest first; far car out of range
    text = motion_text.render(rows)
    assert "moved" in text and "speed unknown" in text  # pedestrian velocity was NaN


def _future_numbers(sample):
    poses = sample["target"]["future_poses"]
    return [f"({poses[i][0]:.1f}, {poses[i][1]:.1f})" for i in (1, 3, 5, 7)]


def test_teacher_sees_future_but_reasoner_input_does_not(sample):
    request = views.teacher_request(sample, ROOT)
    teacher_text = request["body"]["messages"][0]["content"][1]["text"]
    assert all(n in teacher_text for n in _future_numbers(sample))
    label = {"custom_id": sample["sample_id"], "narration": "Busy road.", "reasoning": "Keep lane.",
             "decision": request["meta_action"], "label_source": {"model": "m"}}
    example = views.reasoner_example(sample, label, ROOT)
    user_text = example["messages"][0]["content"]
    assert not any(n in user_text for n in _future_numbers(sample))
    assert "recorded future" not in user_text
    assert example["messages"][1]["content"].startswith("Narration: Busy road.")
    assert json.loads(json.dumps(example)) == example


def test_parse_label():
    reply = 'Sure:\n```json\n{"narration": "a", "reasoning": "b", "decision": {"longitudinal": "keep_speed"}}\n```'
    assert parse_label(reply)["reasoning"] == "b"
    assert parse_label("no json here") is None
    assert parse_label('{"narration": 3}') is None
