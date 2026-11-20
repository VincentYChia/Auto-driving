"""``pilot_sft/heads/driver.py`` with a fake NAVSIM scorer: the R10 choice order and tie-breaks, the layer rule, and
the jobs it queues (navtest paths, phase-3 seeds, phase-2 series), without ever writing a job twice."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import driver  # noqa: E402

TOKENS = [f"tok{i}" for i in range(20)]


def fake_scorer(folder, split, exp_name, **_):
    """PDMS from the folder's name: epoch 15 > the others, steps 4 > the others, scale 1.5 > the others, and the
    12/18/24 runs a little better than layer 24 alone (A only)."""
    base = 0.80
    if "epoch_15" in folder:
        base += 0.02
    if "steps4" in folder:
        base += 0.01
    if "scale1.5" in folder:
        base += 0.03
    if "l12-18-24" in folder and folder.split(os.sep)[-4].startswith("a1"):
        base += 0.005
    j = int(folder.rsplit("_", 1)[-1]) if "sample_" in folder else 0
    rows = [{"token": t, "valid": True, "score": min(1.0, base + 0.001 * ((i + j) % 5))} for i, t in enumerate(TOKENS)]
    d = Path(driver.P["exp"]) / exp_name / "run"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "scores.csv", index=False)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    for k in ("exp", "heads", "queue", "queue_root", "repo", "h1"):
        monkeypatch.setitem(driver.P, k, str(tmp_path / k))
    monkeypatch.setitem(driver.P, "heads", str(tmp_path / "exp" / "heads"))
    monkeypatch.setitem(driver.P, "queue", str(tmp_path / "queue_root" / "pending"))
    monkeypatch.setattr(driver, "SCORER", fake_scorer)
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: None)  # summarize.py is not run in tests
    sf = tmp_path / "repo" / "navsim_eval/navsim_configs/train_test_split/scene_filter"
    sf.mkdir(parents=True)
    (sf / "navdev.yaml").write_text("tokens:\n" + "".join(f"  - '{t}'\n" for t in TOKENS))
    (tmp_path / "exp" / "metric_cache_navdev_v1").mkdir(parents=True)
    os.makedirs(driver.P["queue"], exist_ok=True)
    return tmp_path


def make_run(run, kind, epochs=driver.EPOCHS, split="navdev"):
    g = driver.GRID[kind]
    rd = Path(driver.P["heads"]) / run
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "final.pt").write_text("x")
    for e in epochs:
        base = rd / f"{split}_epoch_{e}"
        for st in g["steps"]:
            names = ["single"] + [os.path.basename(f) for x in g["scales"]
                                  for f in driver.sample_folders(str(base / f"steps{st}"), x, g["k"])]
            for n in names:
                d = base / f"steps{st}" / n
                d.mkdir(parents=True, exist_ok=True)
                (d / "shard_000_of_001.jsonl").write_text("{}\n")
        (base / "timing.json").write_text("{}")


def test_choices_follow_the_r10_order(tree):
    make_run("a1-l24", "masked")
    c = driver.choose_run("a1-l24", "masked")
    assert (c["epoch"], c["steps"], c["scale"]) == (15, 4, 1.5)
    assert json.loads((Path(driver.P["heads"]) / "a1-l24" / "choices_navdev.json").read_text())["epoch"] == 15


def test_regression_has_no_sampling_choice(tree):
    make_run("r-l24", "regression")
    c = driver.choose_run("r-l24", "regression")
    assert c["scale"] is None and c["navdev_best_of_6"] is None and c["epoch"] == 15


def test_nothing_is_chosen_before_every_epoch_exists(tree):
    make_run("d-l24", "anchor", epochs=[5, 10])
    assert driver.choose_run("d-l24", "anchor") is None


def test_layer_choice_and_the_queued_phases(tree):
    for run in ("a1-l24", "a1-l12-18-24"):
        make_run(run, "masked")
    lc = driver.choose_layers("A")
    assert lc["layers"] == "12,18,24" and lc["run"] == "a1-l12-18-24"
    driver.seed_runs("A", lc)
    driver.series(lc)
    driver.queue_navtest("a1-l12-18-24", "masked", lc["choice"])
    jobs = sorted(os.listdir(driver.P["queue"]))
    assert "39a1_train-a1-l12-18-24-s1.sh" in jobs and "39a2_train-a1-l12-18-24-s2.sh" in jobs
    assert {"400_train-a0-l12-18-24.sh", "401_train-a2-l12-18-24.sh", "402_train-a3-l12-18-24.sh",
            "403_train-a4-l12-18-24.sh"} <= set(jobs)
    assert "350_navtest-a1-l12-18-24.sh" in jobs
    seed = (Path(driver.P["queue"]) / "39a1_train-a1-l12-18-24-s1.sh").read_text()
    assert "--seed 1" in seed and "--weights epoch_15" in seed and "--scales 1.5" in seed
    a0 = (Path(driver.P["queue"]) / "400_train-a0-l12-18-24.sh").read_text()
    assert "--no-metric-embed" in a0 and "--layers 12,18,24" in a0
    a4 = (Path(driver.P["queue"]) / "403_train-a4-l12-18-24.sh").read_text()
    assert "--soft-sigma 1" in a4 and "--l1-weight 1" in a4 and "--offset" in a4
    # a second pass writes nothing new, and a job already moved to running/ is not written again
    before = set(os.listdir(driver.P["queue"]))
    os.makedirs(Path(driver.P["queue_root"]) / "running", exist_ok=True)
    os.replace(Path(driver.P["queue"]) / "400_train-a0-l12-18-24.sh",
               Path(driver.P["queue_root"]) / "running" / "400_train-a0-l12-18-24.sh")
    driver.seed_runs("A", lc)
    driver.series(lc)
    assert set(os.listdir(driver.P["queue"])) == before - {"400_train-a0-l12-18-24.sh"}


def test_a_layer_tie_goes_to_layer_24(tree, monkeypatch):
    for run in ("d-l24", "d-l12-18-24"):
        make_run(run, "anchor")
    assert driver.choose_layers("D")["layers"] == "24"  # the fake scores give D's two runs equal PDMS


def test_navtest_scores_at_the_choices_are_reported_only(tree):
    make_run("b-l24", "ddim")
    c = driver.choose_run("b-l24", "ddim")
    make_run("b-l24", "ddim", epochs=[c["epoch"]], split="navtest")
    # navtest has 12,146 scenes; the fake scorer writes 20, so the run's navtest summary must refuse it
    with pytest.raises(ValueError):
        driver.score_navtest_run("b-l24", "ddim", c)


def test_rl_start_uses_best_of_6_and_rl_waits_for_the_navtrain_cache(tree):
    for run in ("a1-l24", "a1-l12-18-24"):
        make_run(run, "masked")
    lc = driver.choose_layers("A")
    start = driver.rl_start("A", lc)
    assert start["epoch"] == 15  # the fake makes epoch 15 best for samples too
    for s in (1, 2):
        (Path(driver.P["heads"]) / f"{lc['run']}-s{s}").mkdir(parents=True, exist_ok=True)
        (Path(driver.P["heads"]) / f"{lc['run']}-s{s}" / "final.pt").write_text("x")
    status = {"navdev_ready": True}
    driver.phase4(status)
    assert not [j for j in os.listdir(driver.P["queue"]) if j.startswith("45")]  # no cache yet: nothing queued
    (Path(driver.P["exp"]) / "metric_cache_navtrain_v1").mkdir(parents=True)
    (Path(driver.P["exp"]) / "metric_cache_navtrain_v1" / ".ready").write_text("")
    driver.phase4(status)
    rl_jobs = sorted(j for j in os.listdir(driver.P["queue"]) if j.startswith("45"))
    assert rl_jobs == ["45a0_rl-a1-l12-18-24.sh", "45a1_rl-a1-l12-18-24-s1.sh", "45a2_rl-a1-l12-18-24-s2.sh"]
    body = (Path(driver.P["queue"]) / "45a1_rl-a1-l12-18-24-s1.sh").read_text()
    assert "--weights epoch_15" in body and "--seed 1" in body and "--scale 1.5" in body and "--steps 4" in body


def test_decision_needs_a_gap_beyond_drive_and_seed_noise():
    rng = np.random.default_rng(0)
    tokens = [f"t{i}" for i in range(220)]
    drives = {t: f"d{i % 11}" for i, t in enumerate(tokens)}
    base = pd.Series(rng.uniform(0.5, 1.0, len(tokens)), index=tokens)

    def method(offset, seeds):  # per seed: a run-level shift; per scene: method-specific noise, as in real scores
        return [(base + offset + rng.normal(0, 0.005) + rng.normal(0, 0.1, len(tokens))).clip(0, 1)
                for _ in range(seeds)]

    clear = driver.decide({"A": method(0.08, 3), "D": method(0.0, 3), "B": method(-0.02, 1), "C": method(-0.03, 1)}, drives)
    assert clear["leader"] == "A" and clear["clear_winner"]
    tie = driver.decide({"A": method(0.002, 3), "D": method(0.0, 3), "B": method(-0.05, 1), "C": method(-0.05, 1)}, drives)
    assert tie["leader"] in ("A", "D") and not tie["clear_winner"]
