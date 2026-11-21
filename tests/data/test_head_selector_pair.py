"""``pilot_sft/heads/selector_pair.py``: two selectors' picks on the same scenes, paired by drive."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import selector_pair as sp  # noqa: E402


def write(d, picks):
    (d / "navdev").mkdir(parents=True)
    (d / "navdev" / "selection.jsonl").write_text("".join(
        json.dumps({"token": t, "chosen": c, "true_chosen": v, "true_own": 0.8}) + "\n" for t, (c, v) in picks.items()))


def test_pair(tmp_path):
    a, b = tmp_path / "sel-a", tmp_path / "sel-b"
    write(a, {"t1": (2, 0.9), "t2": (0, 0.8), "t3": (1, 0.7), "t4": (0, 0.6)})
    write(b, {"t1": (0, 0.8), "t2": (0, 0.8), "t3": (1, 0.7), "t5": (3, 0.5)})  # t4, t5: only one selector scored them
    r = sp.pair(str(a), str(b), {"t1": "d1", "t2": "d1", "t3": "d2", "t4": "d2", "t5": "d3"})
    assert r["scenes"] == 3 and r["drives"] == 2 and r["a"] == "sel-a"
    assert r["a_minus_b"] == round(100 * 0.1 / 3, 2) and r["different_picks"] == round(1 / 3, 4)
    assert r["ci"][0] <= r["a_minus_b"] <= r["ci"][1]
