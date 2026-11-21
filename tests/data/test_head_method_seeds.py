"""``pilot_sft/heads/method_seeds.py``: method pairs over every seed (R5's hierarchical bootstrap), and R17's
three-seed comparison of the continuity series' best with A1."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import method_seeds as ms  # noqa: E402

TOKENS = [f"tok{i}" for i in range(12)]
DRIVES = {t: f"drive{i % 4}" for i, t in enumerate(TOKENS)}
CHOICE = {"epoch": 15, "steps": 4, "scale": 1.5}


def write(exp, run, split, base, weights="epoch_15"):
    """Single path = base + 0.001 * i; the 6 samples a little below and above (best-of-6 = base + 0.01 + ...)."""
    for n, shift in [("single", 0.0)] + [(f"scale1.5_sample_{j}", -0.01 + 0.004 * j) for j in range(6)]:
        d = Path(exp) / f"heads-{run}-{split}_{weights}-steps4-{n}_{split}_v1" / "run"
        d.mkdir(parents=True, exist_ok=True)
        s = [base + shift + 0.001 * i for i in range(len(TOKENS))]
        pd.DataFrame({"token": TOKENS + ["average"], "score": s + [float(np.mean(s))]}).to_csv(d / "scores.csv", index=False)


@pytest.fixture
def tree(tmp_path):
    exp, cache = tmp_path / "exp", tmp_path / "cache"
    cache.mkdir()
    for sp in ("navdev", "navtest"):
        (cache / f"drives_{sp}.json").write_text(json.dumps(DRIVES))
    ch = exp / "heads" / "choices"
    ch.mkdir(parents=True)
    (ch / "A.json").write_text(json.dumps({"run": "a1-l12-18-24", "choice": CHOICE}))
    (ch / "B.json").write_text(json.dumps({"run": "b-l12-18-24", "choice": CHOICE}))
    for sp in ("navdev", "navtest"):
        for run, base in (("a1-l12-18-24", 0.82), ("a1-l12-18-24-s1", 0.81), ("a1-l12-18-24-s2", 0.83),
                          ("b-l12-18-24", 0.80), ("b-l12-18-24-s1", 0.80), ("b-l12-18-24-s2", 0.80)):
            write(exp, run, sp, base)
    return exp, cache


def test_pairs_use_every_seed(tree):
    exp, cache = tree
    out = ms.main(str(exp), str(cache))
    a = out["navtest"]["methods"]["A"]["single"]
    assert a["seeds"] == [82.55, 81.55, 83.55] and a["mean"] == 82.55 and a["sd"] == 1.0
    p = out["navtest"]["pairs"]["A-B"]["single"]
    assert p["difference"] == 2.0 and p["seeds"] == [3, 3] and p["ci"][0] < 2.0 < p["ci"][1]
    assert "series_best_vs_a1" not in out["navtest"]  # no phase-2 result yet


def test_r17_compares_the_series_best_with_a1_over_three_seeds(tree):
    exp, cache = tree
    heads = exp / "heads"
    (heads / "choices" / "A_series.json").write_text(json.dumps({"best": "a2"}))
    (heads / "a2-l12-18-24").mkdir()
    (heads / "a2-l12-18-24" / "choices_navdev.json").write_text(json.dumps(CHOICE))
    write(exp, "a2-l12-18-24", "navtest", 0.85)
    write(exp, "a2-l12-18-24", "navdev", 0.85)
    assert "series_best_vs_a1" not in ms.main(str(exp), str(cache))["navtest"]  # its seeds 1 and 2 not yet scored
    for sd in (1, 2):
        for sp in ("navdev", "navtest"):
            write(exp, f"a2-l12-18-24-s{sd}", sp, 0.85)
    sb = ms.main(str(exp), str(cache))["navtest"]["series_best_vs_a1"]
    assert sb["best"] == "a2" and sb["run"] == "a2-l12-18-24" and "R17" in sb["rule"]
    assert sb["single"]["difference"] == 3.0 and sb["single"]["seeds"] == [3, 3]
    assert sb["single"]["best_seeds"] == [85.55] * 3 and sb["single"]["a1_seeds"] == [82.55, 81.55, 83.55]
    assert sb["best_of_6"]["difference"] == 3.0


def rl_choice(heads, rl, sft, checkpoint):
    d = heads / rl
    d.mkdir(parents=True, exist_ok=True)
    (d / "choices_navdev.json").write_text(json.dumps({"run": rl, "start": {"run": sft, "weights": "epoch_20"},
                                                       "checkpoint": checkpoint, "steps": 4, "scale": 1.5}))


def test_phase4_compares_methods_after_rl_and_each_methods_change_seed_by_seed(tree):
    exp, cache = tree
    heads = exp / "heads"
    assert "phase4" not in ms.main(str(exp), str(cache))["navtest"]  # no RL recipe chosen yet
    (heads / "choices" / "rl_recipe.json").write_text(json.dumps({"recipe": 4}))
    # RL4 (the registered recipe keeps the plain "-rl" name): A's seeds 0 and 2 at an RL checkpoint, seed 1 kept its
    # start (update 0, epoch 20); B's one seed
    for sp in ("navdev", "navtest"):
        rl_choice(heads, "a1-l12-18-24-rl", "a1-l12-18-24", "rl_500")
        write(exp, "a1-l12-18-24-rl", sp, 0.86, "rl_500")
        rl_choice(heads, "a1-l12-18-24-s1-rl", "a1-l12-18-24-s1", "start")
        write(exp, "a1-l12-18-24-s1", sp, 0.82, "epoch_20")
        rl_choice(heads, "a1-l12-18-24-s2-rl", "a1-l12-18-24-s2", "rl_250")
        write(exp, "a1-l12-18-24-s2-rl", sp, 0.85, "rl_250")
        rl_choice(heads, "b-l12-18-24-rl", "b-l12-18-24", "rl_1500")
        write(exp, "b-l12-18-24-rl", sp, 0.83, "rl_1500")
        rl_choice(heads, "b-l12-18-24-rl6", "b-l12-18-24", "rl_750")  # RL6: only B so far
        write(exp, "b-l12-18-24-rl6", sp, 0.84, "rl_750")
    p4 = ms.main(str(exp), str(cache))["navtest"]["phase4"]
    assert set(p4) == {"RL4", "RL6"} and set(p4["RL6"]["methods"]) == {"B"}  # A's RL6 seeds not chosen yet
    a = p4["RL4"]["methods"]["A"]["single"]
    assert a["seeds"] == [86.55, 82.55, 85.55] and a["mean"] == 84.88 and a["checkpoints"] == ["rl_500", "start", "rl_250"]
    ch = p4["RL4"]["change_from_phase3"]
    assert ch["A"]["single"]["difference"] == 2.33 and ch["A"]["single"]["seeds"] == 3  # +4, +1, +2 against seeds 0-2
    assert ch["B"]["single"]["difference"] == 3.0 and ch["B"]["single"]["ci"] == [3.0, 3.0]  # one seed: drives only
    assert p4["RL4"]["pairs"]["A-B"]["single"]["difference"] == 1.33
    assert p4["RL4"]["methods"]["B"]["best_of_6"]["seeds"] == [84.55]
