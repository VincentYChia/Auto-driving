"""Training-data prototype against the real devkit (skipped without NAVSIM)."""

import gzip
import json

import pytest

pytest.importorskip("navsim")
pytest.importorskip("nuplan")

from data_pipeline import extract_navsim, schema  # noqa: E402
from tests.conftest import NUM_FRAMES  # noqa: E402


def _job(logs, sensors, scene_filter, out_dir):
    return {
        "batch_id": 0, "log_names": ["synthetic_log"], "scene_filter": scene_filter,
        "log_path": str(logs), "sensor_path": str(sensors), "out_dir": str(out_dir),
        "split": "synthetic", "data_split": "trainval", "max_range_m": 100.0,
        "provenance": {"extractor": "test"}, "overwrite": False,
    }


def test_extract_batch_with_navsim_crosscheck(synthetic_dataset, tmp_path):
    logs, sensors, scene_filter = synthetic_dataset
    out_dir = tmp_path / "canonical"
    out_dir.mkdir()
    result = extract_navsim._process_batch(_job(logs, sensors, scene_filter, out_dir))
    assert result["status"] == "written" and result["samples"] == NUM_FRAMES - 13 and not result["errors"]
    with gzip.open(out_dir / "samples-00000.jsonl.gz", "rt") as f:
        samples = [json.loads(line) for line in f]
    for sample in samples:
        schema.validate_sample(sample)
    assert samples[0]["sample_id"] == "tok03"
    assert extract_navsim._process_batch(_job(logs, sensors, scene_filter, out_dir))["status"] == "exists"  # resume
