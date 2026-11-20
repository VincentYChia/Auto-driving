"""Phase 6's selector chain in ``pilot_sft/heads/driver.py`` (docs/HEADS_PLAN.md R14): from a spec file, each step
queued once and only when its inputs exist: the half generators and in-fold candidates (GPU), candidate scoring
(~/queue2), the selectors, adoption on navdev by the point estimate, and one navtest evaluation."""

import json
import os
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pilot_sft" / "heads"))
sys.path.insert(0, str(HERE))

import driver  # noqa: E402
from test_head_driver import tree  # noqa: E402,F401  (the fixture)

LOGS = ["2021.05.12.19.36.12_veh-35_00005_00204", "2021.06.07.12.54.00_veh-38_01100_01500",
        "2021.07.24.20.37.45_veh-17_00015_00375", "2021.08.17.18.54.02_veh-45_00665_01065"]
SPEC = {"kind": "masked", "layers": "12,18,24", "flags": ["--soft-sigma", "1"], "rl_start_epoch": 20, "recipe": 4,
        "updates": 1500, "save_every": 250, "deployed_run": "a1-l12-18-24-rl", "deployed_weights": "rl_1500",
        "steps": 8, "scale": 1.5, "k": 6}


@pytest.fixture
def chain(tree, monkeypatch):  # noqa: F811
    ids = [f"openscene/{l}/tok{d}x{i}" for d, l in enumerate(LOGS) for i in range(5)]
    cache = tree / "cache"
    (cache / "train").mkdir(parents=True)
    (cache / "train" / "meta.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in ids))
    view = tree / "rlview"
    view.mkdir()
    (view / "train.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in ids[::2]))
    monkeypatch.setitem(driver.P, "cache", str(cache))
    monkeypatch.setitem(driver.P, "rl_view", str(view))
    monkeypatch.setitem(driver.P, "queue2_root", str(tree / "queue2_root"))
    choices = Path(driver.P["heads"]) / "choices"
    choices.mkdir(parents=True, exist_ok=True)
    (choices / "selector_spec_A.json").write_text(json.dumps(SPEC))
    return tree


def gpu_jobs():
    return sorted(os.listdir(driver.P["queue"]))


def cpu_jobs():
    d = Path(driver.P["queue2_root"]) / "pending"
    return sorted(os.listdir(d)) if d.exists() else []


def written(folder):
    (Path(folder) / "timing.json").parent.mkdir(parents=True, exist_ok=True)
    (Path(folder) / "timing.json").write_text("{}")


def test_selector_chain_from_spec_to_navtest(chain, monkeypatch):
    H = Path(driver.P["heads"])
    status = {}
    driver.selectors(status)
    halves = H / "halves"
    assert (halves / "rl_tokens_all.txt").read_text().split() == \
        (halves / "rl_tokens_half0.txt").read_text().split() + (halves / "rl_tokens_half1.txt").read_text().split()
    assert gpu_jobs() == ["60ah0_selector-a-gen-half0.sh", "60ah1_selector-a-gen-half1.sh", "61a_selector-a-infold.sh"]
    assert cpu_jobs() == ["57aboundary_score-selector-a-boundary.sh"]  # generator-independent: needs no candidates
    h0 = (Path(driver.P["queue"]) / "60ah0_selector-a-gen-half0.sh").read_text()
    assert "--soft-sigma 1 --drive-half 0" in h0 and "--weights epoch_20" in h0 and "--updates 1500" in h0
    assert f"--only-tokens {halves}/rl_tokens_half0.txt" in h0.split("rl_head.py")[1].split("\n")[0]  # RL: own half
    oof = [l for l in h0.splitlines() if "predict_head.py" in l and "navtrain" in l][0]
    assert f"rl_tokens_half1.txt" in oof and "--weights rl_1500" in oof and "--k 6 --scales 1.5" in oof  # other half
    assert "--adv dr" in h0 and "--lp-sd-floor 0" in h0  # RL4 on a kind without Gaussian steps
    assert any("--split navdev" in l for l in h0.splitlines())  # half against deployed, on navdev
    bnd = (Path(driver.P["queue2_root"]) / "pending" / "57aboundary_score-selector-a-boundary.sh").read_text()
    assert "boundary_paths.py make" in bnd and "boundary_paths.py pick" in bnd and f"--port {driver.SCORE_PORT}" in bnd
    assert status["phase6"]["a"]["scored"] == {"half0": False, "half1": False, "infold": False, "boundary": False}

    # candidates written: their scoring is queued on the CPU queue, once
    written(H / "selector-a-gen-half0-rl" / "navtrain_rl_1500")
    written(H / "selector-a-gen-half1-rl" / "navtrain_rl_1500")
    driver.selectors({})
    driver.selectors({})
    assert cpu_jobs() == ["57aboundary_score-selector-a-boundary.sh", "57ahalf0_score-selector-a-half0.sh",
                          "57ahalf1_score-selector-a-half1.sh"]
    assert not [j for j in gpu_jobs() if j.startswith("62")]  # no selector before its candidates are scored

    # cross-fitted candidates scored: the cross-fitted selectors and the 3 plain ablations, not the in-fold control
    for h in (0, 1):
        (H / "selector-a" / f"candidates_oof_half{h}.jsonl.done").write_text("")
    driver.selectors({})
    sel = [j for j in gpu_jobs() if j.startswith("62")]
    assert sel == ["62a0_selector-a-xfit-s0.sh", "62a1_selector-a-xfit-s1.sh", "62a3_selector-a-xfit-pdms.sh",
                   "62a4_selector-a-xfit-listwise.sh", "62a5_selector-a-xfit-imitation.sh"]
    s0 = (Path(driver.P["queue"]) / "62a0_selector-a-xfit-s0.sh").read_text()
    assert s0.count("--candidates") == 2 and "--seed 0" in s0 and os.path.join("a1-l12-18-24-rl", "rl_1500.pt") in s0
    assert "apply_selector.py" in s0 and os.path.join("navdev_rl_1500", "steps8") in s0 and "drives_navdev.json" in s0
    assert "--target pdms" in (Path(driver.P["queue"]) / "62a3_selector-a-xfit-pdms.sh").read_text()

    # in-fold and boundary scored: the control and the boundary ablation
    written(H / "a1-l12-18-24-rl" / "navtrain_rl_1500")
    driver.selectors({})
    assert "57ainfold_score-selector-a-infold.sh" in cpu_jobs()
    (H / "selector-a" / "candidates_infold.jsonl.done").write_text("")
    (H / "selector-a" / "perturbed.jsonl.done").write_text("")
    driver.selectors({})
    inf = (Path(driver.P["queue"]) / "62a2_selector-a-infold-s0.sh").read_text()
    assert inf.count("--candidates") == 1 and "candidates_infold.jsonl" in inf
    b = (Path(driver.P["queue"]) / "62a6_selector-a-xfit-boundary.sh").read_text()
    assert "--extra-candidates" in b and os.path.join("boundary", "scores.jsonl") in b and b.count("--candidates") == 2

    # adoption by the point estimate; the gain is not claimed while the interval includes 0
    rep = H / "selector-a-xfit-s0" / "navdev"
    rep.mkdir(parents=True)
    (rep / "report.json").write_text(json.dumps({"report": {"selected_minus_own": 0.004, "selected_minus_own_ci": [-0.002, 0.011],
                                                            "selected_pdms": 0.866, "own_pick_pdms": 0.862}}))
    st = driver.selector_chain("a", SPEC)
    ad = json.loads((H / "choices" / "selector_a_adoption.json").read_text())
    assert ad["adopted"] is True and ad["gain_claimed"] is False and st["adopted"] is True
    t = (Path(driver.P["queue"]) / "350_navtest-selector-a.sh").read_text()
    assert "--cache $C/navtest" in t and os.path.join("navtest_rl_1500", "steps8") in t

    # one navtest evaluation of the selected paths, by NAVSIM's own scorer
    nt = H / "selector-a-xfit-s0" / "navtest"
    (nt / "selected").mkdir(parents=True)
    (nt / "selected" / "shard_000_of_001.jsonl").write_text("{}\n")
    (nt / "report.json").write_text("{}")
    seen = []
    monkeypatch.setattr(driver, "score", lambda folder, split: seen.append((folder, split)) or pd.Series([0.9, 0.88]))
    st = driver.selector_chain("a", SPEC)
    assert seen == [(str(nt / "selected"), "navtest")] and st["navtest"] == 0.89
    driver.selector_chain("a", SPEC)
    assert len(seen) == 1  # scored once


def test_a_rejected_selector_gets_no_navtest_run(chain):
    H = Path(driver.P["heads"])
    rep = H / "selector-a-xfit-s0" / "navdev"
    rep.mkdir(parents=True)
    (rep / "report.json").write_text(json.dumps({"report": {"selected_minus_own": -0.001, "selected_minus_own_ci": [-0.01, 0.006],
                                                            "selected_pdms": 0.861, "own_pick_pdms": 0.862}}))
    st = driver.selector_chain("a", SPEC)
    assert st["adopted"] is False and "350_navtest-selector-a.sh" not in gpu_jobs()


def test_without_rl_the_halves_predict_from_the_supervised_epoch(chain):
    spec = {**SPEC, "updates": 0, "deployed_run": "a1-l12-18-24", "deployed_weights": "epoch_20"}
    driver.selector_chain("a", spec)
    h1 = (Path(driver.P["queue"]) / "60ah1_selector-a-gen-half1.sh").read_text()
    assert "rl_head.py" not in h1 and "selector-a-gen-half1 --weights epoch_20 --split navtrain" in h1
