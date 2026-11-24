"""``pilot_sft/heads/oof_pair.py``: two chains' out-of-fold candidates on the same scenes, paired over drives."""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import oof_pair  # noqa: E402

LOGS = [f"2021.0{m}.1{m}.10.00.00_veh-3{m}_00001_00100" for m in range(1, 7)]


def write_set(path, scores, names=("single", "s0", "s1")):
    """``scores``: token -> list of candidate PDMS in ``names`` order (None = unscorable)."""
    with open(path, "w", encoding="utf-8") as f:
        for t, ss in scores.items():
            res = [{"error": "x"} if s is None else {"score": s} for s in ss]
            f.write(json.dumps({"token": t, "names": list(names), "results": res}) + "\n")
    Path(str(path) + ".done").write_text("")


@pytest.fixture
def world(tmp_path):
    toks = [(f"t{d}x{i}", LOGS[d]) for d in range(6) for i in range(4)]
    meta = tmp_path / "meta.jsonl"
    meta.write_text("".join(json.dumps({"id": f"openscene/{log}/{t}"}) + "\n" for t, log in toks))
    heads = tmp_path / "heads"
    for x in ("a4", "a6"):
        (heads / f"selector-{x}").mkdir(parents=True)
    return heads, meta, [t for t, _ in toks]


def test_a_paired_difference_per_set_and_pooled_with_drive_intervals(world):
    heads, meta, toks = world
    half0, half1 = toks[:12], toks[12:]
    # a6's own pick is 0.02 better everywhere; their best candidates are equal; a4 has one unscorable sample
    write_set(heads / "selector-a4" / "candidates_oof_half0.jsonl", {t: [0.80, 0.90, None] for t in half0})
    write_set(heads / "selector-a6" / "candidates_oof_half0.jsonl", {t: [0.82, 0.90, 0.85] for t in half0})
    write_set(heads / "selector-a4" / "candidates_oof_half1.jsonl", {t: [0.70, 0.90, 0.80] for t in half1})
    write_set(heads / "selector-a6" / "candidates_oof_half1.jsonl", {t: [0.72, 0.90, 0.80] for t in half1})
    r = oof_pair.main("a4", "a6", ["oof_half0", "oof_half1"], str(heads), str(meta))
    h0 = r["sets"]["oof_half0"]
    assert (h0["scenes"], h0["drives"]) == (12, 3)
    assert h0["own_pick"]["a_minus_b"] == -2.0 and h0["own_pick"]["ci"] == [-2.0, -2.0]
    assert h0["best_of_k"]["a_minus_b"] == 0.0
    assert h0["mean_candidate"]["a"] == 56.67  # the unscorable sample counts as 0: (0.80 + 0.90 + 0) / 3
    p = r["pooled"]
    assert (p["scenes"], p["drives"]) == (24, 6) and p["own_pick"]["a_minus_b"] == -2.0
    assert p["own_pick"]["a"] == 75.0 and p["own_pick"]["b"] == 77.0


def test_a_set_not_scored_for_both_chains_is_left_out(world):
    heads, meta, toks = world
    write_set(heads / "selector-a4" / "candidates_oof_half0.jsonl", {t: [0.8, 0.9, 0.9] for t in toks[:12]})
    write_set(heads / "selector-a6" / "candidates_oof_half0.jsonl", {t: [0.8, 0.9, 0.9] for t in toks[:12]})
    write_set(heads / "selector-a4" / "candidates_oof_half1.jsonl", {t: [0.8, 0.9, 0.9] for t in toks[12:]})
    (heads / "selector-a6" / "candidates_oof_half1.jsonl").write_text("")  # still scoring: no .done marker
    r = oof_pair.main("a4", "a6", ["oof_half0", "oof_half1"], str(heads), str(meta))
    assert r["sets"]["oof_half1"] is None and r["pooled"]["scenes"] == 12
