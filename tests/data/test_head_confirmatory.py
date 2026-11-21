"""``pilot_sft/heads/confirmatory.py``: R4's four navtest tests with Holm's correction, run once at the end."""

import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pilot_sft" / "heads"))
sys.path.insert(0, str(HERE))

import confirmatory as cf  # noqa: E402
from test_head_method_seeds import CHOICE, DRIVES, TOKENS, rl_choice, write  # noqa: E402,F401


def test_holm_steps_down_and_stops_at_the_first_failure():
    h = cf.holm({"C1": 0.001, "C2": 0.03, "C3": 0.04, "C4": 0.013})
    # sorted: C1 0.001 <= 0.05/4, C4 0.013 <= 0.05/3, then C2 0.03 > 0.05/2 stops it, so C3 is not rejected either
    assert [k for k in ("C1", "C4", "C2", "C3") if h[k]["rejected"]] == ["C1", "C4"]
    assert h["C1"]["threshold"] == 0.0125 and h["C2"]["threshold"] == 0.025 and h["C3"]["threshold"] == 0.05
    assert (h["C1"]["p_holm"], h["C4"]["p_holm"], h["C2"]["p_holm"], h["C3"]["p_holm"]) == (0.004, 0.039, 0.06, 0.06)


def test_the_four_tests_from_the_registered_recipe_and_r5s_leader(tmp_path):
    exp, cache = tmp_path / "exp", tmp_path / "cache"
    cache.mkdir()
    (cache / "drives_navtest.json").write_text(json.dumps(DRIVES))
    heads = exp / "heads"
    ch = heads / "choices"
    ch.mkdir(parents=True)
    for m, run in (("A", "a1-l12-18-24"), ("D", "d-l12-18-24")):
        (ch / f"{m}.json").write_text(json.dumps({"run": run, "choice": CHOICE}))
    (ch / "R.json").write_text(json.dumps({"run": "r-l12-18-24", "choice": {**CHOICE, "scale": None}}))
    assert cf.main(str(exp), str(cache)) is None  # nothing before phase 4 and R5
    (ch / "rl_recipe.json").write_text(json.dumps({"recipe": 4}))
    (ch / "decision_phase5.json").write_text(json.dumps({"leader": "A", "clear_winner": False}))
    pd.DataFrame({"token": TOKENS + ["average"], "score": [0.85] * 13}).pipe(
        lambda d: (exp / "deskcheck-h1_navtest_v1" / "run").mkdir(parents=True) or d.to_csv(
            exp / "deskcheck-h1_navtest_v1" / "run" / "scores.csv", index=False))
    for i, (sfx, base) in enumerate((("", 0.82), ("-s1", 0.81), ("-s2", 0.83))):
        write(exp, f"a1-l12-18-24{sfx}", "navtest", base)  # A before RL (phase 3)
        write(exp, f"r-l12-18-24{sfx}", "navtest", base)  # R exactly level with A: C3 must not be claimed
        write(exp, f"d-l12-18-24{sfx}", "navtest", 0.79)
        for m, run, after in (("A", "a1-l12-18-24", 0.86 + 0.001 * i), ("D", "d-l12-18-24", 0.80)):
            rl_choice(heads, f"{run}{sfx}-rl", f"{run}{sfx}", "rl_1500")
            write(exp, f"{run}{sfx}-rl", "navtest", after, "rl_1500")
    out = cf.main(str(exp), str(cache))
    t = out["tests"]
    assert out["chosen_method"] == "A" and out["recipe"] == "RL4"
    assert t["C1"]["difference"] == 6.1 and t["C1"]["rejected"] and t["C1"]["p"] == 0.0
    assert t["C2"]["difference"] == 1.65 and t["C2"]["seeds"] == [3, 1] and t["C2"]["rejected"]  # H1: one fixed model
    assert t["C3"]["difference"] == 0.0 and not t["C3"]["rejected"] and t["C3"]["p"] > 0.5
    assert t["C4"]["seeds"] == 3 and t["C4"]["difference"] == 4.1 and t["C4"]["rejected"]  # +4, +5, +3.2 by seed
