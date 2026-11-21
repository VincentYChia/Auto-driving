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


def test_the_page_carries_them_and_works_without_them():
    page = report_html.build({}, {"title": "t"}, MS)
    assert 'id="ch-seedpairs"' in page and 'id="tb-seedpairs"' in page
    data = json.loads(page.split('<script type="application/json" id="data">')[1].split("</script>")[0])
    assert data["seedPairs"]["navtest"]["pairs"][0]["pair"] == "A-D"
    assert report_html.page_data({})["seedPairs"] == {}
