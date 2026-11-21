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


def test_phase4_rows_per_split_and_recipe():
    p4 = {"RL4": {"methods": {"A": {"single": {"seeds": [86.55, 82.55, 85.55], "mean": 84.88, "sd": 2.08,
                                              "checkpoints": ["rl_500", "start", "rl_250"]}}},
                  "change_from_phase3": {"A": {"single": {"difference": 2.33, "ci": [0.9, 3.8], "seeds": 3, "scenes": 12}}},
                  "pairs": {"A-B": {"single": {"difference": 1.33, "ci": [-0.2, 2.9], "seeds": [3, 1], "scenes": 12}}}}}
    data = report_html.page_data({}, {"navtest": {"pairs": {}, "phase4": p4}, "navdev": {"pairs": {}}})
    r = data["phase4"]["navtest"]["RL4"]
    assert r["methods"][0]["method"] == "A" and r["methods"][0]["single"]["checkpoints"][1] == "start"
    assert r["change"][0]["single"] == {"x": 2.33, "lo": 0.9, "hi": 3.8, "seeds": 3}
    assert r["pairs"][0]["pair"] == "A-B" and r["pairs"][0]["single"]["seeds"] == [3, 1]
    assert "navdev" not in data["phase4"]
    assert 'id="ph4"' in report_html.build({}, {"title": "t"}, {"navtest": {"pairs": {}, "phase4": p4}})


def test_selector_rows_per_chain_with_adoption():
    nav = {"scenes": 743, "candidates": 7, "selected_pdms": 0.83, "own_pick_pdms": 0.8186, "selected_minus_own": 0.0114,
           "selected_minus_own_ci": [0.002, 0.02], "best_of_k_pdms": 0.9, "switch_rate": 0.12,
           "switches": {"wins": 50, "losses": 30, "ties": 9}, "picks_failing_nc_or_dac": {"selected": 40, "own_pick": 52}}
    agg = {"selectors": {
        "selector-rehearsal-xfit-boundary": {"chain": "rehearsal", "variant": "xfit-boundary", "heldout": {"epoch": 9}, "navdev": nav},
        "selector-rehearsal-xfit-s0": {"chain": "rehearsal", "variant": "xfit-s0", "heldout": {"epoch": 12, "delta": 0.02},
                                       "train": {"elapsed_s": 2400.0}, "navdev": nav},
        "selector-a4-xfit-s0": {"chain": "a4", "variant": "xfit-s0", "heldout": {}}},
        "methods": {"selector_rehearsal_adoption": {"adopted": True, "gain_claimed": True, "role": "rehearsal", "x": 1}}}
    rows = report_html.page_data(agg)["selectors"]
    assert [r["chain"] for r in rows] == ["a4", "rehearsal"]  # contenders first, the rehearsal last
    rh = rows[1]
    assert [v["variant"] for v in rh["variants"]] == ["xfit-s0", "xfit-boundary"]  # R14's order
    assert rh["variants"][0]["navdev"]["gain"] == {"x": 1.14, "lo": 0.2, "hi": 2.0} and rh["variants"][0]["navdev"]["own"] == 81.86
    assert rh["adoption"] == {"adopted": True, "gain_claimed": True, "role": "rehearsal"}
    assert "navdev" not in rows[0]["variants"][0]  # trained, not yet applied to navdev
    assert 'id="sel"' in report_html.build(agg, {"title": "t"})
