"""``pilot_sft/heads/summarize.py`` and ``drive_map.py``: the drive-level interval is wider than the scene-level
one when scenes within a drive are correlated, sample folders group by sampling setting, and log names map to
drives."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import drive_map  # noqa: E402
import summarize  # noqa: E402


def test_drive_interval_is_wider_when_scenes_cluster_by_drive():
    rng = np.random.default_rng(0)
    drives = np.repeat(np.arange(11), 180)  # 11 drives x 180 scenes, like navdev
    x = rng.normal(0, 1, 11)[drives] * 0.1 + rng.normal(0, 0.1, len(drives))  # a shared offset per drive
    lo_d, hi_d = summarize.drive_ci(x, drives.astype(str))
    lo_s, hi_s = summarize.scene_ci(x)
    assert (hi_d - lo_d) > 2 * (hi_s - lo_s)
    assert lo_d < x.mean() < hi_d


def test_drive_interval_is_none_without_drives():
    assert summarize.drive_ci(np.zeros(5), None) is None


def test_sample_folder_names_group_by_scale():
    m = summarize.SAMPLE.match
    assert m("sample_3").groups() == (None, "3")
    assert m("scale0.5_sample_0").groups() == ("0.5", "0")
    assert m("single") is None


def test_log_names_map_to_drives():
    assert drive_map.drive_of_log("2021.06.14.18.33.41_veh-35_04905_05090") == "2021.06.14.18.33.41_veh-35"
    assert drive_map.drive_of_log("/x/sensor_blobs/test/2021.05.25.14.16.10_veh-35_00083_00485/CAM_F0/a.jpg") \
        == "2021.05.25.14.16.10_veh-35"
    with pytest.raises(ValueError):
        drive_map.drive_of_log("not-a-log")
