"""``pilot_sft/heads/report_html.py``: method pairs with every seed (method_seeds.py's output) on the report page."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pilot_sft" / "heads"))

import report_html  # noqa: E402

MS = {"navtest": {"methods": {}, "pairs": {
          "A-D": {"single": {"difference": 2.19, "ci": [1.1, 3.34], "seeds": [3, 2], "scenes": 12146},
                  "best_of_6": {"difference": -1.6, "ci": [-2.16, -0.99], "seeds": [3, 2], "scenes": 12146}},
          "D-R": {"single": {"difference": -2.07, "ci": [-3.46, -0.7], "seeds": [2, 1], "scenes": 12146}}},
                  "seed_sd_added_for_single_seed": {"single": 0.5, "best_of_6": 0.49}},
      "navdev": {"methods": {}, "pairs": {}, "seed_sd_added_for_single_seed": {}}}


def test_seed_pairs_become_page_rows_in_points():
    sp = report_html.page_data({}, MS)["seedPairs"]
    rows = {r["pair"]: r for r in sp["navtest"]["pairs"]}
    assert rows["A-D"]["single"] == {"x": 2.19, "lo": 1.1, "hi": 3.34} and rows["A-D"]["seeds"] == [3, 2]
    assert rows["A-D"]["best_of_6"]["x"] == -1.6
    assert "best_of_6" not in rows["D-R"] and rows["D-R"]["seeds"] == [2, 1]  # regression has no samples
    assert sp["navtest"]["fill"] == {"single": 0.5, "best_of_6": 0.49} and sp["navdev"]["pairs"] == []


def test_the_continuity_series_rows_carry_r17s_test():
    d = {"difference": -0.02, "ci_drives": [-0.03, -0.01], "scenes": 12146, "beyond_seed_noise": True}
    small = {"difference": 0.004, "ci_drives": [-0.002, 0.01], "scenes": 12146, "beyond_seed_noise": False}
    agg = {"series": {
        "a2-bin025-l12-18-24": {"config": "a2", "variant": "bin025", "navtest_single": small, "measured_effect": False},
        "a0-l12-18-24": {"config": "a0", "variant": "", "navtest_single": d, "navdev_single": small, "measured_effect": True},
        "a3-no-soft-l12-18-24": {"config": "a3", "variant": "no-soft", "navtest_single": small, "measured_effect": False}}}
    sb = {"best": "a2", "run": "a2-l12-18-24", "single": {"difference": 0.9, "ci": [0.1, 1.7], "seeds": [3, 3],
                                                          "best_seeds": [85.1, 85.3, 85.0], "a1_seeds": [84.67, 84.2, 85.2]}}
    data = report_html.page_data(agg, {"navtest": {"pairs": {}, "series_best_vs_a1": sb}})
    rows = data["series"]
    assert [r["run"] for r in rows] == ["a0-l12-18-24", "a2-bin025-l12-18-24", "a3-no-soft-l12-18-24"]  # series first
    assert rows[0]["label"].startswith("A0: plain bins") and rows[0]["measured"] is True
    assert rows[0]["navtest_single"] == {"x": -2.0, "lo": -3.0, "hi": -1.0, "beyond": True}
    assert rows[1]["label"] == "A2 with 0.25 m bins" and rows[2]["label"] == "A3 without soft labels"
    assert [r["short"] for r in rows] == ["A0", "A2 0.25 m bins", "A3 no-soft"]  # chart labels stay short
    assert data["seriesBest"]["navtest"]["best"] == "a2" and "navdev" not in data["seriesBest"]
    page = report_html.build(agg, {"title": "t"}, {"navtest": {"pairs": {}, "series_best_vs_a1": sb}})
    assert 'id="ch-series"' in page and 'id="tb-seriesbest"' in page


def test_the_page_carries_them_and_works_without_them():
    page = report_html.build({}, {"title": "t"}, MS)
    assert 'id="ch-seedpairs"' in page and 'id="tb-seedpairs"' in page
    data = json.loads(page.split('<script type="application/json" id="data">')[1].split("</script>")[0])
    assert data["seedPairs"]["navtest"]["pairs"][0]["pair"] == "A-D"
    assert report_html.page_data({})["seedPairs"] == {}
