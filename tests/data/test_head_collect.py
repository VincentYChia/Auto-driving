"""``pilot_sft/heads/collect_results.py`` on a fake score tree: references, each run at its choices (with the
paired difference to H1, sub-scores and best-of-6 diversity), the layer test, method pairs, and RL runs (update 0
from the start's folders, the paired change from the start, the training curve)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import collect_results as cr  # noqa: E402

TOKENS = [f"tok{i}" for i in range(12)]
DRIVES = {t: f"drive{i % 4}" for i, t in enumerate(TOKENS)}


def write_scores(exp, name, scores, extra=None):
    d = Path(exp) / name / "run"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"token": t, "valid": True, "score": s, "no_at_fault_collisions": 1.0, "ego_progress": s / 2,
             **(extra or {})} for t, s in zip(TOKENS, scores)]
    rows.append({"token": "average", "valid": True, "score": float(np.mean(scores)), "no_at_fault_collisions": 1.0,
                 "ego_progress": 0.0})
    pd.DataFrame(rows).to_csv(d / "scores.csv", index=False)


def write_paths(folder, shift):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "shard_000_of_001.jsonl", "w") as f:
        for t in TOKENS:
            f.write(json.dumps({"token": t, "waypoints": [[x + shift, 0.0] for x in range(8)]}) + "\n")


def config(exp, run, sp, weights, steps, scale, base):
    """Single path = base + 0.001 * index; sample j = base - 0.01 + 0.004 * j (so best-of-6 = sample 5)."""
    heads = Path(exp) / "heads"
    single = [base + 0.001 * i for i in range(len(TOKENS))]
    write_scores(exp, cr.Results.name(run, sp, weights, steps, "single"), single)
    write_paths(heads / run / f"{sp}_{weights}" / f"steps{steps}" / "single", 0.0)
    for j, n in enumerate(cr.Results.sample_names(scale)):
        write_scores(exp, cr.Results.name(run, sp, weights, steps, n),
                     [base - 0.01 + 0.004 * j + 0.001 * i for i in range(len(TOKENS))])
        write_paths(heads / run / f"{sp}_{weights}" / f"steps{steps}" / n, 0.5 * j)
    return np.mean(single)


@pytest.fixture
def tree(tmp_path):
    exp, cache = tmp_path / "exp", tmp_path / "cache"
    cache.mkdir()
    for sp in ("navdev", "navtest"):
        (cache / f"drives_{sp}.json").write_text(json.dumps(DRIVES))
    write_scores(exp, "h1-text_navdev_v1", [0.80] * len(TOKENS))
    write_scores(exp, "deskcheck-h1_navtest_v1", [0.85] * len(TOKENS))
    write_scores(exp, "cv_navdev_v1", [0.20] * len(TOKENS))
    heads = exp / "heads"
    for run, base, kind in (("a1-l24", 0.81, "masked"), ("a1-l12-18-24", 0.82, "masked"), ("d-l24", 0.78, "anchor")):
        rd = heads / run
        rd.mkdir(parents=True)
        (rd / "choices_navdev.json").write_text(json.dumps({"run": run, "epoch": 15, "steps": 4, "scale": 1.5}))
        (rd / "run.json").write_text(json.dumps({"config": {"kind": kind, "layers": "24"}, "params": 10}))
        (rd / "log.jsonl").write_text(json.dumps({"epoch": 15, "dev_ade": 0.7, "dev_best6_ade": 0.5}) + "\n")
        for sp in ("navdev", "navtest"):
            config(exp, run, sp, "epoch_15", 4, 1.5, base)
    (heads / "choices").mkdir()
    (heads / "choices" / "A.json").write_text(json.dumps({
        "method": "A", "run": "a1-l12-18-24", "choice": {"epoch": 15}, "candidates": {"a1-l24": 0.81, "a1-l12-18-24": 0.82}}))
    (heads / "choices" / "D.json").write_text(json.dumps({
        "method": "D", "run": "d-l24", "choice": {"epoch": 15}, "candidates": {"d-l24": 0.78}}))
    (heads / "choices" / "A_rl_start.json").write_text(json.dumps({"method": "A", "epoch": 20}))
    # an RL run from a1-l12-18-24 epoch 20: the start at 0.80, update 250 at 0.79, update 500 at 0.77
    config(exp, "a1-l12-18-24", "navdev", "epoch_20", 4, 1.5, 0.80)
    rl = heads / "a1-l12-18-24-rlrecipe0"
    rl.mkdir()
    (rl / "run.json").write_text(json.dumps({
        "args": {"run": str(heads / "a1-l12-18-24"), "weights": "epoch_20", "steps": 4, "scale": 1.5, "lr": 1e-5,
                 "kl_beta": 0.0, "g": 8, "scenes_per_update": 32, "updates": 500, "seed": 0},
        "start": {"run": str(heads / "a1-l12-18-24"), "weights": "epoch_20"}, "rl_scenes": 100}))
    for u, base in ((250, 0.79), (500, 0.77)):
        (rl / f"rl_{u}.pt").write_text("x")
        config(exp, "a1-l12-18-24-rlrecipe0", "navdev", f"rl_{u}", 4, 1.5, base)
    (rl / "log.jsonl").write_text("".join(json.dumps({"update": u, "reward": 0.8, "best_of_g": 0.9, "kl": 0.0,
                                                       "equal_reward_groups": 0.1 + u / 1000}) + "\n"
                                          for u in range(1, 121)))
    return exp, cache


def test_collect(tree):
    exp, cache = tree
    out = cr.Results(str(exp), str(cache)).collect()
    ref = out["references"]
    assert ref["h1_text_navdev"]["pdms"] == 0.8 and ref["constant_velocity_navdev"]["pdms"] == 0.2
    assert ref["h1_text_navdev"]["scenes"] == len(TOKENS)  # the "average" row is not a scene
    a = out["runs"]["a1-l24"]
    single = 0.81 + 0.001 * np.mean(range(len(TOKENS)))
    assert a["navdev_single"]["pdms"] == pytest.approx(single, abs=1e-4)
    assert a["navdev_single"]["vs_h1_text"]["difference"] == pytest.approx(single - 0.80, abs=1e-4)
    assert a["navdev_best_of_6"]["pdms"] == pytest.approx(single - 0.01 + 0.02, abs=1e-4)  # sample 5 is best
    assert a["navdev_best_of_6"]["diversity"]["endpoint_m"] > 0
    assert a["navdev_single"]["subscores"]["no_at_fault_collisions"] == 1.0
    assert a["navtest_single"]["vs_h1_text"]["difference"] == pytest.approx(single - 0.85, abs=1e-4)
    assert a["dev_ade"] == 0.7 and a["kind"] == "masked"
    lo, hi = a["navdev_single"]["ci_drives"]
    assert lo <= a["navdev_single"]["pdms"] <= hi
    # layer test: 12/18/24 minus 24, a constant 0.01 per scene, so the interval is the point
    lt = out["layer_tests"]["A"]["navdev_single"]
    assert lt["difference"] == pytest.approx(0.01, abs=1e-6) and lt["ci_drives"] == pytest.approx([0.01, 0.01], abs=1e-6)
    assert "D" not in out["layer_tests"]  # D has one layer setting so far
    cmp = out["comparisons"]["A-D"]
    assert cmp["runs"] == ["a1-l12-18-24", "d-l24"]
    assert cmp["navdev_single"]["difference"] == pytest.approx(0.04, abs=1e-6)
    assert cmp["navtest_best_of_6"]["difference"] == pytest.approx(0.04, abs=1e-6)
    assert "a1-l12-18-24-rlrecipe0" not in out["runs"]
    r = out["rl"]["a1-l12-18-24-rlrecipe0"]
    assert r["start"] == {"run": "a1-l12-18-24", "weights": "epoch_20"}
    assert set(r["checkpoints"]) == {"0", "250", "500"}
    c0, c500 = r["checkpoints"]["0"], r["checkpoints"]["500"]
    assert c0["single"]["pdms"] == pytest.approx(0.80 + 0.0055, abs=1e-4) and "vs_start" not in c0["single"]
    assert c500["single"]["vs_start"]["difference"] == pytest.approx(-0.03, abs=1e-6)
    assert c500["best_of_6"]["vs_start"]["difference"] == pytest.approx(-0.03, abs=1e-6)
    assert [b["updates"] for b in r["curve"]] == [[1, 50], [51, 100], [101, 120]]
    assert r["updates_done"] == 120 and r["settings"]["lr"] == 1e-5
    assert set(out["methods"]) == {"A", "D", "A_rl_start"}
    json.dumps(out)  # serialisable, no per-scene series left in it


def test_the_continuity_series_is_read_against_a1_with_r17s_test(tree):
    exp, cache = tree
    heads = exp / "heads"
    # A1 (a1-l12-18-24) scores 0.82 + 0.001 * i; A0 2 points below it, A2 0.5 above, A2's 0.25 m bins 2 above;
    # A1's phase-3 seed and its layer-24 run are not part of the series
    for run, base in (("a0-l12-18-24", 0.80), ("a2-l12-18-24", 0.825), ("a2-bin025-l12-18-24", 0.84),
                      ("a1-l12-18-24-s1", 0.83)):
        rd = heads / run
        rd.mkdir(parents=True)
        (rd / "choices_navdev.json").write_text(json.dumps({"run": run, "epoch": 15, "steps": 4, "scale": 1.5}))
        for sp in ("navdev", "navtest"):
            config(exp, run, sp, "epoch_15", 4, 1.5, base)
    out = cr.Results(str(exp), str(cache)).collect()["series"]
    assert set(out) == {"a0-l12-18-24", "a2-l12-18-24", "a2-bin025-l12-18-24"}
    a0, a2, b = out["a0-l12-18-24"], out["a2-l12-18-24"], out["a2-bin025-l12-18-24"]
    assert a0["against"] == "a1-l12-18-24" and (a0["config"], a0["variant"]) == ("a0", "")
    assert a0["navtest_single"]["difference"] == pytest.approx(-0.02, abs=1e-6) and a0["measured_effect"] is True
    assert a2["navtest_single"]["difference"] == pytest.approx(0.005, abs=1e-6) and a2["measured_effect"] is False
    assert (b["config"], b["variant"], b["measured_effect"]) == ("a2", "bin025", True)
    assert b["navdev_best_of_6"]["beyond_seed_noise"] is True  # every measure carries the test, for reporting


def test_selectors_carry_their_choice_training_and_navdev_measures(tree):
    exp, cache = tree
    heads = exp / "heads"
    d = heads / "selector-rehearsal-xfit-s0"
    (d / "navdev").mkdir(parents=True)
    (d / "choice.json").write_text(json.dumps({"epoch": 12, "weights": "epoch_12", "delta": 0.02,
                                               "heldout_selected_pdms": 0.851, "epochs_done": 20, "history": []}))
    (d / "run.json").write_text(json.dumps({"params": 7200000, "scenes": {"train": 70000, "heldout": 10626},
                                            "elapsed_s": 2400.0, "peak_gpu_gb": 6.1, "candidates_per_scene": 7,
                                            "extra_candidates_per_scene": 0}))
    (d / "navdev" / "report.json").write_text(json.dumps({"scenes": 2000, "weights": "epoch_12", "delta": 0.02, "report": {
        "scenes": 743, "candidates": 7, "selected_pdms": 0.83, "own_pick_pdms": 0.8186, "selected_minus_own": 0.0114,
        "selected_minus_own_ci": [0.002, 0.02], "best_of_k_pdms": 0.90, "switch_rate": 0.12, "not_kept": 1}}))
    (heads / "selector-rehearsal-gen-half0").mkdir()  # a half generator, not a selector
    out = cr.Results(str(exp), str(cache)).collect()["selectors"]
    assert set(out) == {"selector-rehearsal-xfit-s0"}
    r = out["selector-rehearsal-xfit-s0"]
    assert (r["chain"], r["variant"]) == ("rehearsal", "xfit-s0")
    assert r["heldout"] == {"epoch": 12, "delta": 0.02, "heldout_selected_pdms": 0.851, "epochs_done": 20}
    assert r["train"]["elapsed_s"] == 2400.0 and r["train"]["scenes"]["train"] == 70000
    assert r["navdev"]["selected_minus_own_ci"] == [0.002, 0.02] and r["navdev"]["scenes"] == 743
    assert "not_kept" not in r["navdev"] and "navtest" not in r


def test_each_chains_scored_candidate_sets_carry_their_headroom(tree):
    exp, cache = tree
    d = exp / "heads" / "selector-rehearsal"
    d.mkdir(parents=True)
    rows = [{"token": "t1", "names": ["single", "sample_0"], "results": [
        {"score": 0.8, "no_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0},
        {"score": 0.9, "no_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0}]}]
    for h in (0, 1):
        (d / f"candidates_oof_half{h}.jsonl").write_text("".join(json.dumps(r) + chr(10) for r in rows))
    (d / "candidates_oof_half0.jsonl.done").write_text("")  # half 1 still being scored: left out
    out = cr.Results(str(exp), str(cache)).collect()["selector_candidates"]
    assert list(out) == ["rehearsal"] and list(out["rehearsal"]) == ["oof_half0"]
    assert out["rehearsal"]["oof_half0"]["own_pick_pdms"] == 80.0 and out["rehearsal"]["oof_half0"]["best_of_k_pdms"] == 90.0


def test_cli(tree, tmp_path):
    exp, cache = tree
    import subprocess
    res = subprocess.run([sys.executable, str(Path(cr.__file__)), "--out", str(tmp_path / "agg.json"), "--exp",
                          str(exp), "--cache-dir", str(cache)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert json.loads((tmp_path / "agg.json").read_text())["rl"]


def test_rl_settings_include_r13s_options(tree):
    """R12's runs predate R13's options, so they read as the registered recipe; an R13 run carries its own and the
    BC weight it measured."""
    exp, cache = tree
    r = cr.Results(str(exp), str(cache)).collect()["rl"]["a1-l12-18-24-rlrecipe0"]
    assert r["settings"]["adv"] == "grpo" and r["settings"]["bc_balance"] == 0.0 and r["settings"]["lp_sd_floor"] == 0.0
    assert "bc_weight" not in r
    rj = Path(exp) / "heads" / "a1-l12-18-24-rlrecipe0" / "run.json"
    j = json.loads(rj.read_text())
    j["args"].update(adv="dr", bc_balance=1.0, lp_sd_floor=0.1)
    j["bc"] = {"weight": 0.37, "balance": 1.0}
    rj.write_text(json.dumps(j))
    r = cr.Results(str(exp), str(cache)).collect()["rl"]["a1-l12-18-24-rlrecipe0"]
    assert r["settings"]["adv"] == "dr" and r["settings"]["lp_sd_floor"] == 0.1 and r["bc_weight"] == 0.37
