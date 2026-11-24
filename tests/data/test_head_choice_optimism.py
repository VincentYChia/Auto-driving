"""``pilot_sft/heads/choice_optimism.py``: the optimism of a navdev-chosen checkpoint, by cross-fitting over drives."""

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import choice_optimism as co  # noqa: E402


def make(true_means, n_drives=12, per_drive=40, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    toks = [f"t{i}" for i in range(n_drives * per_drive)]
    drives = {t: f"d{i // per_drive}" for i, t in enumerate(toks)}
    drive_effect = {f"d{d}": rng.normal(0, 0.1) for d in range(n_drives)}
    scores = {u: pd.Series([mu + drive_effect[drives[t]] + rng.normal(0, noise) for t in toks], index=toks)
              for u, mu in true_means.items()}
    return scores, drives


def test_equal_checkpoints_show_optimism_and_cross_fitting_removes_it():
    # 11 checkpoints with the same true mean: the in-sample maximum is optimistic, the cross-fitted score is not
    scores, drives = make({u: 0.85 for u in range(0, 5500, 500)})
    r = co.cross_fit(scores, drives, n_splits=200)
    assert r["in_sample"] > r["cross_fitted"] and r["optimism"] > 0
    assert abs(r["cross_fitted"] - 85.0) < 1.0
    assert r["scenes"] == 480 and r["drives"] == 12 and set(r["by_update"]) == set(range(0, 5500, 500))


def test_a_clearly_better_checkpoint_is_found_either_way():
    means = {u: 0.80 for u in range(0, 5500, 500)}
    means[2500] = 0.90
    scores, drives = make(means, noise=0.05)
    r = co.cross_fit(scores, drives, n_splits=100)
    assert r["in_sample_choice"] == 2500 and r["choice_agreement"] == 1.0
    assert abs(r["optimism"]) < 0.5  # no luck needed when one checkpoint is truly better


def test_a_tie_goes_to_the_earlier_update():
    toks = [f"t{i}" for i in range(20)]
    drives = {t: f"d{i % 4}" for i, t in enumerate(toks)}
    s = pd.Series([0.8] * 20, index=toks)
    r = co.cross_fit({0: s, 500: s.copy()}, drives, n_splits=5)
    assert r["in_sample_choice"] == 0 and r["optimism"] == 0.0
