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
    for k in ("exp", "heads", "queue", "queue_root", "queue2_root", "repo", "h1"):
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


def test_the_series_best_gets_two_more_seeds_when_it_is_not_a1(tree, monkeypatch):
    def scorer(folder, split, exp_name, **kw):  # A2 above every other configuration of the series (R17)
        fake_scorer(folder, split, exp_name, **kw)
        if f"{os.sep}a2-" in folder:
            f = Path(driver.P["exp"]) / exp_name / "run" / "scores.csv"
            d = pd.read_csv(f)
            d["score"] = d["score"] + 0.02
            d.to_csv(f, index=False)

    monkeypatch.setattr(driver, "SCORER", scorer)
    for run in ("a1-l24", "a1-l12-18-24", "a0-l12-18-24", "a2-l12-18-24", "a3-l12-18-24", "a4-l12-18-24"):
        make_run(run, "masked")
    lc = driver.choose_layers("A")
    driver.series(lc)
    assert json.loads((Path(driver.P["heads"]) / "choices" / "A_series.json").read_text())["best"] == "a2"
    jobs = sorted(j for j in os.listdir(driver.P["queue"]) if j.startswith("z_43"))  # queued last: no choice reads them
    assert jobs == ["z_431_train-a2-l12-18-24-s1.sh", "z_432_train-a2-l12-18-24-s2.sh"]
    assert sorted(j for j in os.listdir(driver.P["queue"]) if j.startswith("z_4")) == sorted(
        j for j in os.listdir(driver.P["queue"]) if j[2:4] in ("41", "42", "43"))  # every follow-up run is last
    driver.series(lc)  # queued once, under the prefixed name
    assert len([j for j in os.listdir(driver.P["queue"]) if "431_train" in j]) == 1
    body = (Path(driver.P["queue"]) / "z_432_train-a2-l12-18-24-s2.sh").read_text()
    assert "--seed 2" in body and "--soft-sigma 1" in body and "--split navtest" in body and "--weights epoch_15" in body
    # their scores, once written, go to choices/A_series_best_seeds.json
    for sd in (1, 2):
        make_run(f"a2-l12-18-24-s{sd}", "masked", epochs=[15])
    driver.series(lc)
    res = json.loads((Path(driver.P["heads"]) / "choices" / "A_series_best_seeds.json").read_text())
    assert set(res["runs"]) == {"a2-l12-18-24-s1", "a2-l12-18-24-s2"} and "navdev_single_pdms" in res["runs"]["a2-l12-18-24-s1"]


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


def recipe_setup():
    """Phase 1 done for A and D, and the navtrain cache ready: the recipe runs get queued. Returns A's and D's layer
    choices and ``finish(r)``, which writes recipe r's navdev paths at 250 and 500 updates."""
    for run, kind in (("a1-l24", "masked"), ("a1-l12-18-24", "masked"), ("d-l24", "anchor"), ("d-l12-18-24", "anchor")):
        make_run(run, kind)
    la, ld = driver.choose_layers("A"), driver.choose_layers("D")
    assert driver.rl_start("A", la)["epoch"] == 15  # the fake makes epoch 15 best for samples too
    status = {"navdev_ready": True}
    driver.phase4(status)
    # no navtrain cache: no recipe run and no phase-4 run is queued
    assert not [j for j in os.listdir(driver.P["queue"]) if j.startswith(("351", "45"))]
    (Path(driver.P["exp"]) / "metric_cache_navtrain_v1").mkdir(parents=True)
    (Path(driver.P["exp"]) / "metric_cache_navtrain_v1" / ".ready").write_text("")
    driver.phase4(status)

    def finish(r):
        for lc, kind in ((la, "masked"), (ld, "anchor")):
            if driver.recipe_run(r, kind) != r:
                continue  # another recipe's run stands for this one
            g = driver.GRID[kind]
            rd = Path(driver.P["heads"]) / f"{lc['run']}-rlrecipe{r}"
            for u in driver.RECIPE_CHECKPOINTS:
                base = rd / f"navdev_rl_{u}" / f"steps{lc['choice']['steps']}"
                names = ["single"] + ([os.path.basename(f) for f in driver.sample_folders(str(base), lc["choice"]["scale"], g["k"])]
                                      if lc["choice"]["scale"] is not None else [])
                for n in names:
                    (base / n).mkdir(parents=True, exist_ok=True)
                    (base / n / "shard_000_of_001.jsonl").write_text("{}" + chr(10))
                (rd / f"navdev_rl_{u}" / "timing.json").write_text("{}")
        for s in (1, 2):
            for lc in (la, ld):
                (Path(driver.P["heads"]) / f"{lc['run']}-s{s}").mkdir(parents=True, exist_ok=True)
                (Path(driver.P["heads"]) / f"{lc['run']}-s{s}" / "final.pt").write_text("x")

    return la, ld, finish, status


def test_rl_recipes_come_first_and_phase_4_is_held_when_every_recipe_hurts(tree):
    la, ld, finish, status = recipe_setup()
    jobs = sorted(os.listdir(driver.P["queue"]))
    recipe_jobs = [j for j in jobs if j.startswith("351")]
    # R12's RL0-RL3 and R13's RL4-RL7 on A and D, except A's RL7: without Gaussian steps it is RL0
    assert len(recipe_jobs) == 15 and not [j for j in jobs if j.startswith("45")]  # recipes first, no phase 4 yet
    assert "3517a_rlrecipe-a1-l12-18-24.sh" not in jobs and "3517d_rlrecipe-d-l24.sh" in jobs
    # the runner starts jobs in name order: the recipe runs go after the 350_ path jobs, before phase-1 training
    assert all("350_~" < j < "360" for j in recipe_jobs)
    q = Path(driver.P["queue"])
    rl2 = (q / "3512a_rlrecipe-a1-l12-18-24.sh").read_text()
    assert "--lr 3e-06" in rl2 and "--kl-beta 0.04" in rl2 and "--g 8" in rl2 and "--updates 500" in rl2
    assert "--adv grpo --bc-balance 0 --lp-sd-floor 0" in rl2
    assert "--g 16 --scenes-per-update 16" in (q / "3513d_rlrecipe-d-l24.sh").read_text()
    assert "--adv dr --bc-balance 0 --lp-sd-floor 0 " in (q / "3514a_rlrecipe-a1-l12-18-24.sh").read_text()  # A: no floor
    assert "--adv dr --bc-balance 0 --lp-sd-floor 0.1" in (q / "3514d_rlrecipe-d-l24.sh").read_text()
    assert "--adv grpo --bc-balance 1 --lp-sd-floor 0.1" in (q / "3515d_rlrecipe-d-l24.sh").read_text()
    assert "--adv dr --bc-balance 1 --lp-sd-floor 0.1" in (q / "3516d_rlrecipe-d-l24.sh").read_text()
    assert "--adv grpo --bc-balance 0 --lp-sd-floor 0.1" in (q / "3517d_rlrecipe-d-l24.sh").read_text()
    # recipe 0 finishes first: its checkpoint choices are saved, and no recipe is chosen while the others run
    finish(0)
    driver.phase4(status)
    assert (Path(driver.P["heads"]) / "a1-l12-18-24-rlrecipe0" / "choices_navdev.json").exists()
    assert not (Path(driver.P["heads"]) / "choices" / "rl_recipe.json").exists()
    # a later pass reads recipe 0's saved choices back from JSON (string keys) while scoring the rest
    for r in list(driver.RECIPES)[1:]:
        finish(r)
    driver.phase4(status)
    rec = json.loads((Path(driver.P["heads"]) / "choices" / "rl_recipe.json").read_text())
    assert set(rec["mean_gain_navdev_single"]) == {str(r) for r in driver.RECIPES}
    assert rec["every_recipe_lowers_both_methods"] and rec["phase4"].startswith("held")  # the fake scores RL below
    assert rec["runs_standing_in"] == {"RL7 on A": "RL0 (the same settings for masked)"}
    assert rec["gain_by_method"]["7"]["A"] == rec["gain_by_method"]["0"]["A"]
    assert not [j for j in os.listdir(driver.P["queue"]) if j.startswith("45")]  # R13: phase 4 waits for the researcher
    assert status["rl_recipe"] is None
    ch = json.loads((Path(driver.P["heads"]) / "a1-l12-18-24-rlrecipe0" / "choices_navdev.json").read_text())
    assert ch["checkpoint"] == "start"  # RL only lowered the score here, so update 0 is kept


def test_phase_4_runs_the_recipe_that_helps(tree, monkeypatch):
    def scorer(folder, split, exp_name, **kw):  # RL6's paths score above the start, the other recipes' below
        fake_scorer(folder, split, exp_name, **kw)
        if "rlrecipe6" in folder:
            f = Path(driver.P["exp"]) / exp_name / "run" / "scores.csv"
            df = pd.read_csv(f)
            df["score"] = (df["score"] + 0.08).clip(upper=1.0)
            df.to_csv(f, index=False)

    monkeypatch.setattr(driver, "SCORER", scorer)
    la, ld, finish, status = recipe_setup()
    # an R12-only choice written before R13 existed is kept under another name, and the stage goes on
    old = Path(driver.P["heads"]) / "choices" / "rl_recipe.json"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps({"recipe": 3, "mean_gain_navdev_single": {str(r): -0.01 for r in range(4)},
                               "every_recipe_lowers_both_methods": True}))
    for r in driver.RECIPES:
        finish(r)
    driver.phase4(status)
    assert (old.parent / "rl_recipe_RL0-RL3.json").exists()
    rec = json.loads(old.read_text())
    assert rec["recipe"] == 6 and not rec["every_recipe_lowers_both_methods"] and rec["phase4"] == "started with RL6"
    assert status["rl_recipe"] == 6
    jobs = set(os.listdir(driver.P["queue"]))
    assert {"45a0_rl-a1-l12-18-24.sh", "45a1_rl-a1-l12-18-24-s1.sh", "45a2_rl-a1-l12-18-24-s2.sh"} <= jobs
    # R16: the other phase-4 recipe runs too, with the same seeds and updates, under its own name
    assert {"45a0r4_rl4-a1-l12-18-24.sh", "45a1r4_rl4-a1-l12-18-24-s1.sh", "45d2r4_rl4-d-l24-s2.sh"} <= jobs
    rl4 = (Path(driver.P["queue"]) / "45a1r4_rl4-a1-l12-18-24-s1.sh").read_text()
    assert "OUT=$H/a1-l12-18-24-s1-rl4" in rl4 and "--adv dr --bc-balance 0 " in rl4 and "--updates 1500" in rl4
    # phase 4 runs after the phase-3 seeds and the continuity series (the plan's order); phase 2's follow-up runs
    # (leave-one-out, bin sizes, R17 seeds) feed no choice and are queued last, after everything else
    driver.seed_runs("A", la)
    driver.series(la)
    names = os.listdir(driver.P["queue"])
    diffusion = [j for j in names if j.startswith(("39", "40"))]
    rl = [j for j in names if j.startswith("45")]
    assert len(diffusion) >= 6 and max(diffusion) < min(rl)
    assert max(rl + ["47a4_rlfull4-x.sh", "62a40_selector-x.sh"]) < min(["z_41s_train-x.sh", "z_4205_train-x.sh"])
    body = (Path(driver.P["queue"]) / "45a1_rl-a1-l12-18-24-s1.sh").read_text()
    assert "--weights epoch_15" in body and "--seed 1" in body and "--lr 1e-05" in body and "--updates 1500" in body
    assert "--adv dr --bc-balance 1 --lp-sd-floor 0 " in body  # A has no Gaussian steps
    assert "--adv dr --bc-balance 1 --lp-sd-floor 0.1" in (Path(driver.P["queue"]) / "45d1_rl-d-l24-s1.sh").read_text()


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


def test_every_pass_writes_its_health_even_when_it_fails(tree, monkeypatch):
    """driver_status.json is written only by a pass that completes, so a failing driver could look healthy there;
    driver_health.json is written after every pass, with the error and how many passes in a row have failed."""
    health = Path(driver.P["heads"]) / "driver_health.json"
    monkeypatch.setattr(sys, "argv", ["driver.py", "--once"])
    monkeypatch.setattr(driver, "one_pass", lambda: (_ for _ in ()).throw(KeyError(250)))
    driver.main()
    h = json.loads(health.read_text())
    assert h["ok"] is False and h["error"] == "KeyError: 250" and h["failed_in_a_row"] == 1 and h["pass_ended"]
    monkeypatch.setattr(driver, "one_pass", lambda: {})
    driver.main()
    h = json.loads(health.read_text())
    assert h["ok"] is True and "error" not in h and h["failed_in_a_row"] == 0
