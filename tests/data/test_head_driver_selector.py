"""Phase 6's selector chain in ``pilot_sft/heads/driver.py`` (docs/HEADS_PLAN.md R14): from a spec file, each step
queued once and only when its inputs exist: the half generators and in-fold candidates (GPU), candidate scoring
(~/queue2), the selectors, adoption on navdev by the point estimate, and one navtest evaluation."""

import json
import re
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
    assert gpu_jobs() == ["470a-60h0_selector-a-gen-half0.sh", "470a-60h1_selector-a-gen-half1.sh", "48a-61_selector-a-infold.sh"]
    assert cpu_jobs() == ["57boundary_score-selector-shared-boundary.sh"]  # generator-independent: one shared set
    h0 = (Path(driver.P["queue"]) / "470a-60h0_selector-a-gen-half0.sh").read_text()
    assert "--soft-sigma 1 --drive-half 0" in h0 and "--weights epoch_20" in h0 and "--updates 1500" in h0
    assert f"--only-tokens {halves}/rl_tokens_half0.txt" in h0.split("rl_head.py")[1].split("\n")[0]  # RL: own half
    oof = [l for l in h0.splitlines() if "predict_head.py" in l and "navtrain" in l][0]
    assert f"rl_tokens_half1.txt" in oof and "--weights rl_1500" in oof and "--k 6 --scales 1.5" in oof  # other half
    assert "--adv dr" in h0 and "--lp-sd-floor 0" in h0  # RL4 on a kind without Gaussian steps
    assert any("--split navdev" in l for l in h0.splitlines())  # half against deployed, on navdev
    bnd = (Path(driver.P["queue2_root"]) / "pending" / "57boundary_score-selector-shared-boundary.sh").read_text()
    assert "boundary_paths.py make" in bnd and "boundary_paths.py pick" in bnd and f"--port {driver.SCORE_PORT}" in bnd
    assert status["phase6"]["a"]["scored"] == {"half0": False, "half1": False, "infold": False, "boundary": False}

    # candidates written: their scoring is queued on the CPU queue, once
    written(H / "selector-a-gen-half0-rl" / "navtrain_rl_1500")
    written(H / "selector-a-gen-half1-rl" / "navtrain_rl_1500")
    driver.selectors({})
    driver.selectors({})
    assert cpu_jobs() == ["57ahalf0_score-selector-a-half0.sh", "57ahalf1_score-selector-a-half1.sh",
                          "57boundary_score-selector-shared-boundary.sh"]
    assert not [j for j in gpu_jobs() if "-62" in j]  # no selector before its candidates are scored

    # cross-fitted candidates scored: the cross-fitted selectors and the 3 plain ablations, not the in-fold control
    for h in (0, 1):
        (H / "selector-a" / f"candidates_oof_half{h}.jsonl.done").write_text("")
    driver.selectors({})
    sel = [j for j in gpu_jobs() if "-62" in j]
    assert sel == ["470a-620_selector-a-xfit-s0.sh", "48a-621_selector-a-xfit-s1.sh", "48a-623_selector-a-xfit-pdms.sh",
                   "48a-624_selector-a-xfit-listwise.sh", "48a-625_selector-a-xfit-imitation.sh"]
    s0 = (Path(driver.P["queue"]) / "470a-620_selector-a-xfit-s0.sh").read_text()
    assert s0.count("--candidates") == 2 and "--seed 0" in s0 and os.path.join("a1-l12-18-24-rl", "rl_1500.pt") in s0
    assert "apply_selector.py" in s0 and os.path.join("navdev_rl_1500", "steps8") in s0 and "drives_navdev.json" in s0
    assert "--target pdms" in (Path(driver.P["queue"]) / "48a-623_selector-a-xfit-pdms.sh").read_text()

    # in-fold and boundary scored: the control and the boundary ablation
    written(H / "a1-l12-18-24-rl" / "navtrain_rl_1500")
    driver.selectors({})
    assert "57ainfold_score-selector-a-infold.sh" in cpu_jobs()
    (H / "selector-a" / "candidates_infold.jsonl.done").write_text("")
    (H / "selector-shared" / "perturbed.jsonl.done").write_text("")
    driver.selectors({})
    inf = (Path(driver.P["queue"]) / "48a-622_selector-a-infold-s0.sh").read_text()
    assert inf.count("--candidates") == 1 and "candidates_infold.jsonl" in inf
    b = (Path(driver.P["queue"]) / "48a-626_selector-a-xfit-boundary.sh").read_text()
    assert "--extra-candidates" in b and os.path.join("boundary", "scores.jsonl") in b and b.count("--candidates") == 2

    # adoption by the point estimate; the gain is not claimed while the interval includes 0
    rep = H / "selector-a-xfit-s0" / "navdev"
    rep.mkdir(parents=True)
    (rep / "report.json").write_text(json.dumps({"report": {"selected_minus_own": 0.004, "selected_minus_own_ci": [-0.002, 0.011],
                                                            "selected_pdms": 0.866, "own_pick_pdms": 0.862}}))
    st = driver.selector_chain("a", SPEC)
    ad = json.loads((H / "choices" / "selector_a_adoption.json").read_text())
    assert ad["adopted"] is True and ad["gain_claimed"] is False and st["adopted"] is True
    assert "350_navtest-selector-a.sh" not in gpu_jobs()  # navtest waits for the final-system choice (R16)


def test_a_rejected_selector_gets_no_navtest_run(chain):
    H = Path(driver.P["heads"])
    rep = H / "selector-a-xfit-s0" / "navdev"
    rep.mkdir(parents=True)
    (rep / "report.json").write_text(json.dumps({"report": {"selected_minus_own": -0.001, "selected_minus_own_ci": [-0.01, 0.006],
                                                            "selected_pdms": 0.861, "own_pick_pdms": 0.862}}))
    st = driver.selector_chain("a", SPEC)
    assert st["adopted"] is False and "350_navtest-selector-a.sh" not in gpu_jobs()


def test_a_job_queued_first_is_never_written_again_under_either_name(tree):  # noqa: F811
    assert driver.queue_job("47a4_rlfull4-x", "what", "echo\n", first=True)
    assert gpu_jobs() == ["0_47a4_rlfull4-x.sh"]
    assert not driver.queue_job("47a4_rlfull4-x", "what", "echo\n")  # pending under the prefixed name
    done = Path(driver.P["queue_root"]) / "done"
    done.mkdir(parents=True)
    os.replace(Path(driver.P["queue"]) / "0_47a4_rlfull4-x.sh", done / "0_47a4_rlfull4-x.sh")
    assert not driver.queue_job("47a4_rlfull4-x", "what", "echo\n", first=True) and gpu_jobs() == []
    assert sorted(["0_47a4_rlfull4-x.sh", "350_navtest-a.sh", "39r1_train-r.sh"])[0] == "0_47a4_rlfull4-x.sh"


def test_a_rehearsal_spec_runs_only_its_variants_ahead_of_the_queue(chain):
    H = Path(driver.P["heads"])
    spec = {**SPEC, "updates": 0, "recipe": None, "deployed_run": "a1-l12-18-24", "deployed_weights": "epoch_20",
            "variants": ["xfit-s0", "xfit-boundary"], "first": True, "role": "rehearsal"}
    driver.selector_chain("rehearsal", spec)
    assert gpu_jobs() == ["0_60rehearsalh0_selector-rehearsal-gen-half0.sh", "0_60rehearsalh1_selector-rehearsal-gen-half1.sh"]
    assert cpu_jobs() == ["0_57boundary_score-selector-shared-boundary.sh"]  # no in-fold set: no variant needs it
    h0 = (Path(driver.P["queue"]) / "0_60rehearsalh0_selector-rehearsal-gen-half0.sh").read_text()
    assert "rl_head.py" not in h0 and "--drive-half 0" in h0 and "--weights epoch_20 --split navtrain" in h0
    written(H / "selector-rehearsal-gen-half0" / "navtrain_epoch_20")
    written(H / "selector-rehearsal-gen-half1" / "navtrain_epoch_20")
    driver.selector_chain("rehearsal", spec)
    assert cpu_jobs() == ["0_57boundary_score-selector-shared-boundary.sh", "0_57rehearsalhalf0_score-selector-rehearsal-half0.sh",
                          "0_57rehearsalhalf1_score-selector-rehearsal-half1.sh"]
    for h in (0, 1):
        (H / "selector-rehearsal" / f"candidates_oof_half{h}.jsonl.done").write_text("")
    driver.selector_chain("rehearsal", spec)
    assert [j for j in gpu_jobs() if "62" in j] == ["0_62rehearsal0_selector-rehearsal-xfit-s0.sh"]  # boundary not scored yet
    (H / "selector-shared" / "perturbed.jsonl.done").write_text("")
    st = driver.selector_chain("rehearsal", spec)
    assert [j for j in gpu_jobs() if "62" in j] == ["0_62rehearsal0_selector-rehearsal-xfit-s0.sh",
                                                     "0_62rehearsal6_selector-rehearsal-xfit-boundary.sh"]
    assert st["scored"] == {"half0": True, "half1": True, "boundary": True}
    # a contender's chain later reuses the shared boundary set: not queued again under its plain name
    driver.selector_chain("a", SPEC)
    assert "57boundary_score-selector-shared-boundary.sh" not in cpu_jobs()
    assert "48a-61_selector-a-infold.sh" in gpu_jobs() and "470a-60h0_selector-a-gen-half0.sh" in gpu_jobs()  # all of R14's sets
    # its adoption record says what it is; the final system reads only the contenders' specs (R16)
    rep = H / "selector-rehearsal-xfit-s0" / "navdev"
    rep.mkdir(parents=True)
    (rep / "report.json").write_text(json.dumps({"report": {"selected_minus_own": 0.004, "selected_minus_own_ci": [-0.002, 0.011],
                                                            "selected_pdms": 0.866, "own_pick_pdms": 0.862}}))
    driver.selector_chain("rehearsal", spec)
    assert json.loads((H / "choices" / "selector_rehearsal_adoption.json").read_text())["role"] == "rehearsal"
    (H / "choices" / "selector_spec_A.json").unlink()
    (H / "choices" / "selector_spec_rehearsal.json").write_text(json.dumps(spec))
    driver.final_system({})
    assert not (H / "choices" / "final_system.json").exists()


def test_the_rehearsal_spec_is_the_phase1_configuration_without_rl(tree):  # noqa: F811
    import rehearsal_spec
    make_run("a1-l24", "masked")
    make_run("a1-l12-18-24", "masked")
    spec = rehearsal_spec.rehearsal_spec("A")
    lc, start = driver.choose_layers("A"), driver.rl_start("A", driver.choose_layers("A"))
    assert (spec["deployed_run"], spec["deployed_weights"], spec["updates"]) == (lc["run"], f"epoch_{start['epoch']}", 0)
    assert (spec["steps"], spec["scale"], spec["k"]) == (lc["choice"]["steps"], lc["choice"]["scale"], driver.GRID["masked"]["k"])
    assert spec["first"] is True and spec["variants"] == ["xfit-s0", "xfit-boundary"] and spec["flags"] == []
    assert rehearsal_spec.rehearsal_spec("A") == spec  # written once
    assert ("A", 4) in driver.contenders() and not any("rehearsal" in m for m, _ in driver.contenders())


def test_without_rl_the_halves_predict_from_the_supervised_epoch(chain):
    spec = {**SPEC, "updates": 0, "deployed_run": "a1-l12-18-24", "deployed_weights": "epoch_20"}
    driver.selector_chain("a", spec)
    h1 = (Path(driver.P["queue"]) / "470a-60h1_selector-a-gen-half1.sh").read_text()
    assert "rl_head.py" not in h1 and "selector-a-gen-half1 --weights epoch_20 --split navtrain" in h1


# ---- R15: the winner's tuned configuration, its full RL stage, and the selector spec ---------------------------
from test_head_driver import fake_scorer, make_run  # noqa: E402


def full_rl_outputs(rd, steps, scale, k=6):
    for u in driver.FULL_RL_CHECKPOINTS:
        base = rd / f"navdev_rl_{u}" / f"steps{steps}"
        for n in ["single"] + [os.path.basename(f) for f in driver.sample_folders(str(base), scale, k)]:
            (base / n).mkdir(parents=True, exist_ok=True)
            (base / n / "shard_000_of_001.jsonl").write_text("{}\n")
        (rd / f"navdev_rl_{u}" / "timing.json").write_text("{}")



def test_phase6_takes_a_under_both_recipes_through_full_rl_to_the_selector_specs(tree, monkeypatch):  # noqa: F811
    for run in ("a1-l24", "a1-l12-18-24", "a0-l12-18-24", "a2-l12-18-24", "a3-l12-18-24", "a4-l12-18-24"):
        make_run(run, "masked")
    choices = Path(driver.P["heads"]) / "choices"
    choices.mkdir(parents=True, exist_ok=True)
    (choices / "rl_recipe.json").write_text(json.dumps({"recipe": 4}))
    status = {}
    driver.phase6(status)  # no phase-5 decision needed (R16)
    tuned = json.loads((choices / "A_tuned.json").read_text())
    assert tuned["run"] == "a1-l12-18-24" and tuned["label"] == "post-decision tuning"  # the fake favours A1
    assert set(tuned["candidates"]) == {"a1-l12-18-24", "a0-l12-18-24", "a2-l12-18-24", "a3-l12-18-24", "a4-l12-18-24"}
    assert json.loads((choices / "A_tuned_rl_start.json").read_text())["epoch"] == 15
    j4 = (Path(driver.P["queue"]) / "47a4_rlfull4-a1-l12-18-24.sh").read_text()
    j6 = (Path(driver.P["queue"]) / "47a6_rlfull6-a1-l12-18-24.sh").read_text()
    assert "--updates 5000 --save-every 500" in j4 and "--adv dr --bc-balance 0 " in j4 and "--weights epoch_15" in j4
    assert "--adv dr --bc-balance 1 " in j6 and "a1-l12-18-24-rlfull6" in j6  # RL6: Dr. GRPO plus BC
    assert "for u in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000" in j4
    assert not list(choices.glob("selector_spec_*.json"))  # no checkpoint chosen yet

    for r in (4, 6):
        full_rl_outputs(Path(driver.P["heads"]) / f"a1-l12-18-24-rlfull{r}", 4, 1.5)

    def scorer(folder, split, exp_name, **kw):  # the RL checkpoint at 3,000 updates is the best on navdev
        fake_scorer(folder, split, exp_name, **kw)
        if "navdev_rl_3000" in folder:
            f = Path(driver.P["exp"]) / exp_name / "run" / "scores.csv"
            d = pd.read_csv(f)
            d["score"] = d["score"] + 0.05
            d.to_csv(f, index=False)

    monkeypatch.setattr(driver, "SCORER", scorer)
    driver.phase6(status)
    for r in (4, 6):
        spec = json.loads((choices / f"selector_spec_A{r}.json").read_text())
        assert (spec["deployed_run"], spec["deployed_weights"], spec["updates"]) == (f"a1-l12-18-24-rlfull{r}", "rl_3000", 3000)
        assert (spec["recipe"], spec["steps"], spec["scale"], spec["k"], spec["rl_start_epoch"]) == (r, 4, 1.5, 6, 15)
        assert status["phase6_full"][f"A{r}"]["checkpoint"] == "rl_3000"
    assert not [j for j in os.listdir(driver.P["queue"]) if j.startswith("350_navtest-a1")]  # no navtest yet (R16)


def rl_outputs(rd, steps, scale, updates, k=6):
    """navdev paths of the given saves of a full RL stage."""
    for u in updates:
        base = rd / f"navdev_rl_{u}" / f"steps{steps}"
        for n in ["single"] + [os.path.basename(f) for f in driver.sample_folders(str(base), scale, k)]:
            (base / n).mkdir(parents=True, exist_ok=True)
            (base / n / "shard_000_of_001.jsonl").write_text("{}\n")
        (rd / f"navdev_rl_{u}" / "timing.json").write_text("{}")


def peak_at(peak):
    """A scorer whose navdev single path rises with the update count up to ``peak`` and falls after it."""
    def scorer(folder, split, exp_name, **kw):
        fake_scorer(folder, split, exp_name, **kw)
        m = re.search(r"navdev_rl_(\d+)", folder)
        if m:
            u = int(m.group(1))
            f = Path(driver.P["exp"]) / exp_name / "run" / "scores.csv"
            d = pd.read_csv(f)
            d["score"] = d["score"] + 0.01 * (u if u <= peak else 2 * peak - u) / 1000
            d.to_csv(f, index=False)
    return scorer


def phase6_a_only(tree, monkeypatch):  # noqa: F811
    for run in ("a1-l24", "a1-l12-18-24", "a0-l12-18-24", "a2-l12-18-24", "a3-l12-18-24", "a4-l12-18-24"):
        make_run(run, "masked")
    choices = Path(driver.P["heads"]) / "choices"
    choices.mkdir(parents=True, exist_ok=True)
    (choices / "rl_recipe.json").write_text(json.dumps({"recipe": 4}))
    monkeypatch.setattr(driver, "contenders", lambda: [("A", 4)])
    return choices, Path(driver.P["heads"]) / "a1-l12-18-24-rlfull4"


def test_the_full_rl_stage_continues_while_its_last_save_is_the_best(tree, monkeypatch):  # noqa: F811
    choices, rd = phase6_a_only(tree, monkeypatch)
    monkeypatch.setattr(driver, "SCORER", peak_at(7500))
    driver.phase6({})
    rl_outputs(rd, 4, 1.5, range(500, 5001, 500))
    driver.phase6({})  # round 1: the best save is its last (5,000), so round 2 is queued and nothing is chosen
    assert not (rd / "choices_navdev.json").exists() and not (choices / "selector_spec_A4.json").exists()
    j2 = (Path(driver.P["queue"]) / "47a4r2_rlfull4-a1-l12-18-24.sh").read_text()
    assert "--updates 10000 --save-every 500 --resume" in j2 and "for u in 500 1000" in j2 and " 10000; do" in j2
    assert "[ -f $OUT/navdev_rl_$u/timing.json ] ||" in j2  # the saves predicted in round 1 are not predicted again
    rl_outputs(rd, 4, 1.5, range(5500, 10001, 500))
    status = {}
    driver.phase6(status)  # round 2 peaks at 7,500: the stage ends there
    ch = json.loads((rd / "choices_navdev.json").read_text())
    assert (ch["checkpoint"], ch["updates"], ch["rounds"]) == ("rl_7500", 7500, 2) and "R18" in ch["rule"]
    spec = json.loads((choices / "selector_spec_A4.json").read_text())
    assert (spec["deployed_weights"], spec["updates"]) == ("rl_7500", 7500)  # the half generators run 7,500 updates
    assert status["phase6_full"]["A4"]["rounds"] == 2
    assert not [j for j in os.listdir(driver.P["queue"]) if j.startswith("47a4r3")]


def test_the_full_rl_stage_stops_after_four_rounds(tree, monkeypatch):  # noqa: F811
    choices, rd = phase6_a_only(tree, monkeypatch)
    monkeypatch.setattr(driver, "SCORER", peak_at(10 ** 6))  # still rising at every save
    for k in range(1, 5):
        driver.phase6({})
        rl_outputs(rd, 4, 1.5, range(500, 5000 * k + 1, 500))
    driver.phase6({})
    ch = json.loads((rd / "choices_navdev.json").read_text())
    assert (ch["checkpoint"], ch["rounds"]) == ("rl_20000", 4)
    jobs = sorted(j for j in os.listdir(driver.P["queue"]) if j.startswith("47a4"))
    assert jobs == ["47a4_rlfull4-a1-l12-18-24.sh", "47a4r2_rlfull4-a1-l12-18-24.sh", "47a4r3_rlfull4-a1-l12-18-24.sh",
                    "47a4r4_rlfull4-a1-l12-18-24.sh"]
    assert "--updates 20000" in (Path(driver.P["queue"]) / "47a4r4_rlfull4-a1-l12-18-24.sh").read_text()


def test_phase6_queues_tuning_runs_and_holds_methods_without_their_flags(tree, monkeypatch):  # noqa: F811
    for run in ("d-l24", "d-l12-18-24", "b-l24", "b-l12-18-24"):
        make_run(run, "anchor" if run.startswith("d") else "ddim")
    choices = Path(driver.P["heads"]) / "choices"
    choices.mkdir(parents=True, exist_ok=True)
    (choices / "phase6_gate_bc.json").write_text(json.dumps({"B": {"joins": True}, "C": {"joins": False}}))
    assert driver.contenders() == [("A", 4), ("A", 6), ("D", 4), ("D", 6), ("B", 4), ("B", 6)]
    monkeypatch.setattr(driver, "R15_READY", {"D"})  # B admitted, but without its training flags: held
    driver.phase6({})
    tag = driver.layers_tag(driver.choose_layers("D")["layers"])
    jobs = sorted(j for j in os.listdir(driver.P["queue"]) if j.startswith("46"))
    assert jobs == [f"46d{i}_train-d-r15{i}-{tag}.sh" for i in range(4)]  # queued once for both of D's recipes
    assert "--anchor-trunc 0.02" in (Path(driver.P["queue"]) / f"46d2_train-d-r152-{tag}.sh").read_text()
    assert not [j for j in os.listdir(driver.P["queue"]) if j.startswith("350_navtest-d-r15")]  # nothing before a choice
    make_run(f"d-r150-{tag}", "anchor")  # its navdev grid done: its navtest paths are queued for the report (R1)
    driver.phase6({})
    assert f"350_navtest-d-r150-{tag}.sh" in os.listdir(driver.P["queue"])
    monkeypatch.setattr(driver, "R15_READY", {"B", "C", "D"})  # B's flags exist: its 4 runs follow
    driver.phase6({})
    jobs = sorted(j for j in os.listdir(driver.P["queue"]) if j.startswith("46b"))
    assert jobs == [f"46b{i}_train-b-r15{i}-{tag}.sh" for i in range(4)]
    assert "--target eps --schedule linear" in (Path(driver.P["queue"]) / f"46b3_train-b-r153-{tag}.sh").read_text()
    assert not [j for j in os.listdir(driver.P["queue"]) if j.startswith("47")]  # nothing until the tuning is done


def phase4_seed0(method, recipe, base=4, seed=0):
    """Write a phase-4 run's navdev choice and its chosen paths (checkpoint rl_250), for one supervised seed."""
    lc = driver.choose_layers(method)
    assert driver.rl_start(method, lc)
    sft = lc["run"] if seed == 0 else f"{lc['run']}-s{seed}"
    rl = driver.rl_name(sft, recipe, base)
    c, g = lc["choice"], driver.GRID[lc["kind"]]
    rd = Path(driver.P["heads"]) / rl
    paths = rd / "navdev_rl_250" / f"steps{c['steps']}"
    for n in ["single"] + [os.path.basename(f) for f in driver.sample_folders(str(paths), c["scale"], g["k"])]:
        (paths / n).mkdir(parents=True, exist_ok=True)
        (paths / n / "shard_000_of_001.jsonl").write_text("{}\n")
    (rd / "choices_navdev.json").write_text(json.dumps(
        {"run": rl, "start": {"run": sft, "weights": "epoch_15"}, "checkpoint": "rl_250", "updates": 250,
         "steps": c["steps"], "scale": c["scale"], "navdev_single_pdms": 0.8, "navdev_best_of_6": 0.9,
         "single_by_checkpoint": {"0": 0.8, "250": 0.8}, "best_of_6_by_checkpoint": {"0": 0.9, "250": 0.9}}))


def test_b_and_c_join_phase6_only_if_not_clearly_below_a_and_d(tree, monkeypatch):  # noqa: F811
    from test_head_driver import TOKENS
    monkeypatch.setitem(driver.P, "cache", str(tree / "cache" / "h1"))
    (tree / "cache").mkdir(parents=True, exist_ok=True)
    (tree / "cache" / "drives_navdev.json").write_text(json.dumps({t: f"drive{i % 4}" for i, t in enumerate(TOKENS)}))

    def scorer(folder, split, exp_name, **kw):  # B level with A and D, C clearly below
        fake_scorer(folder, split, exp_name, **kw)
        shift = 0.02 if f"{os.sep}b-l" in folder else (-0.1 if f"{os.sep}c-l" in folder else 0.0)
        if shift and "_rl_" in folder:
            f = Path(driver.P["exp"]) / exp_name / "run" / "scores.csv"
            d = pd.read_csv(f)
            d["score"] = d["score"] + shift
            d.to_csv(f, index=False)

    monkeypatch.setattr(driver, "SCORER", scorer)
    for m, kind in (("A", "masked"), ("D", "anchor"), ("B", "ddim"), ("C", "flow")):
        pre = {"A": "a1", "D": "d", "B": "b", "C": "c"}[m]
        make_run(f"{pre}-l24", kind)
        make_run(f"{pre}-l12-18-24", kind)
    choices = Path(driver.P["heads"]) / "choices"
    choices.mkdir(parents=True, exist_ok=True)
    (choices / "rl_recipe.json").write_text(json.dumps({"recipe": 4}))
    assert driver.bc_gate() is None  # not before every phase-4 run is chosen
    for m in ("A", "D", "B", "C"):
        for r in (4, 6):
            phase4_seed0(m, r)
    assert driver.bc_gate() is None  # A's and D's seeds 1 and 2 are still missing (amended R16: every seed)
    for m in ("A", "D"):
        for r in (4, 6):
            for sd in (1, 2):
                phase4_seed0(m, r, seed=sd)
    g = driver.bc_gate()
    assert g["B"]["joins"] is True and g["C"]["joins"] is False
    assert g["B"]["tests"]["single"]["leads"] is True  # B scores 0.02 above A and D in this fake
    c = g["C"]["tests"]
    assert c["single"]["best_ad_minus_it_ci"][0] > 0 and c["best_of_6"]["best_ad_minus_it_ci"][0] > 0
    assert "amended" in g["rule"]
    assert driver.contenders() == [("A", 4), ("A", 6), ("D", 4), ("D", 6), ("B", 4), ("B", 6)]


def test_the_final_system_is_chosen_on_navdev_before_any_navtest_score(tree, monkeypatch):  # noqa: F811
    from test_head_driver import TOKENS
    monkeypatch.setitem(driver.P, "cache", str(tree / "cache" / "h1"))
    (tree / "cache").mkdir(parents=True, exist_ok=True)
    (tree / "cache" / "drives_navdev.json").write_text(json.dumps({t: f"drive{i % 4}" for i, t in enumerate(TOKENS)}))
    H, choices = Path(driver.P["heads"]), Path(driver.P["heads"]) / "choices"
    choices.mkdir(parents=True, exist_ok=True)
    adopted = {"A4": 0.90, "A6": None, "D4": 0.85, "D6": None}  # the selected paths' true navdev PDMS, or not adopted
    for x, val in adopted.items():
        dep = f"gen-{x.lower()}"
        (choices / f"selector_spec_{x}.json").write_text(json.dumps(
            {"deployed_run": dep, "deployed_weights": "rl_2500", "steps": 2, "scale": 1.0, "k": 6}))
        nav = H / dep / "navdev_rl_2500" / "steps2" / "single"
        nav.mkdir(parents=True)
        (nav / "shard_000_of_001.jsonl").write_text("{}\n")
        (choices / f"selector_{x.lower()}_adoption.json").write_text(json.dumps({"adopted": val is not None}))
        if val is not None:
            d = H / f"selector-{x.lower()}-xfit-s0" / "navdev"
            d.mkdir(parents=True)
            (d / "selection.jsonl").write_text("".join(json.dumps({"token": t, "true_chosen": val}) + "\n" for t in TOKENS)
                                               + json.dumps({"token": "unscored-dev-scene", "chosen": 0}) + "\n")
            # a dev scene outside NAVSIM's navdev filter has a selected path but no true score (the 2026-11-26 KeyError)
    status = {}
    driver.final_system(status)
    fs = json.loads((choices / "final_system.json").read_text())
    assert fs["final"] == "A4" and fs["navdev_system_pdms"]["A4"] == 0.9 and not fs["tie"]
    assert set(fs["final_minus_other"]) == {"A6", "D4", "D6"} and fs["final_minus_other"]["D4"]["difference"] == 0.05
    jobs = sorted(j for j in os.listdir(driver.P["queue"]) if j.startswith("350_navtest"))
    assert jobs == [f"350_navtest-gen-{x}-rl_2500.sh" for x in ("a4", "a6", "d4", "d6")]
    assert "--k 6 --scales 1" in (Path(driver.P["queue"]) / "350_navtest-gen-a4-rl_2500.sh").read_text()  # candidates
    assert "--k 0" in (Path(driver.P["queue"]) / "350_navtest-gen-a6-rl_2500.sh").read_text()  # own pick only

    # navtest paths written: an adopted selector is applied once its candidates are scored on navtest (it reads their
    # score files), then every system path is scored once
    for x in adopted:
        t = H / f"gen-{x.lower()}" / "navtest_rl_2500"
        (t / "steps2" / "single").mkdir(parents=True)
        (t / "timing.json").write_text("{}")
    asked = []
    monkeypatch.setattr(driver, "score", lambda folder, split: asked.append(os.path.relpath(folder, H)) or (
        pd.Series([0.9] * 3) if "sample_5" not in folder else None))  # the last candidate is not scorable yet
    driver.final_system(status)
    assert not {"350_navtest-selector-a4.sh", "350_navtest-selector-d4.sh"} & set(os.listdir(driver.P["queue"]))
    cand = os.path.join("gen-a4", "navtest_rl_2500", "steps2")
    assert [a for a in asked if a.startswith("gen-a4")] == [os.path.join(cand, n)
                                                            for n in ["single"] + [f"sample_{j}" for j in range(6)]]
    monkeypatch.setattr(driver, "score", lambda folder, split: pd.Series([0.9] * 3))
    driver.final_system(status)
    assert {"350_navtest-selector-a4.sh", "350_navtest-selector-d4.sh"} <= set(os.listdir(driver.P["queue"]))
    assert "--exp" in (Path(driver.P["queue"]) / "350_navtest-selector-a4.sh").read_text()
    for x in ("a4", "d4"):
        sel = H / f"selector-{x}-xfit-s0" / "navtest"
        (sel / "selected").mkdir(parents=True)
        (sel / "report.json").write_text("{}")
    got = {"selector-a4": 0.91, "selector-d4": 0.9, "gen-a6": 0.86, "gen-d6": 0.84,
           "gen-a4": 0.88, "gen-d4": 0.87}  # the adopted contenders' candidates, already scored
    monkeypatch.setattr(driver, "score", lambda folder, split: pd.Series(
        [next(v for k, v in got.items() if k in folder)] * 3) if split == "navtest" else None)
    driver.final_system(status)
    fn = json.loads((choices / "final_navtest.json").read_text())
    assert fn["final"] == "A4" and fn["headline_navtest_pdms"] == 0.91
    assert fn["navtest_system_pdms"] == {"A4": 0.91, "A6": 0.86, "D4": 0.9, "D6": 0.84}
    assert status["final_system"] == {"final": "A4", "navtest": 0.91}


def test_a_contenders_chain_runs_before_more_full_rl_but_after_tuning(chain):
    """The reorders of 2026-11-22: a contender's decisive chain jobs (half generators, the adopted variant) sort
    ahead of every full RL stage and behind the tuning runs; its in-fold control, seed 1 and ablations sort after
    every full RL stage and before phase 2's follow-ups. The rehearsal keeps its old names."""
    H = Path(driver.P["heads"])
    driver.selector_chain("a", SPEC)
    written(H / "selector-a-gen-half0-rl" / "navtrain_rl_1500")
    written(H / "selector-a-gen-half1-rl" / "navtrain_rl_1500")
    written(H / "a1-l12-18-24-rl" / "navtrain_rl_1500")
    for f in ("candidates_oof_half0", "candidates_oof_half1", "candidates_infold"):
        (H / "selector-a").mkdir(parents=True, exist_ok=True)
        (H / "selector-a" / f"{f}.jsonl.done").write_text("")
    (H / "selector-shared").mkdir(parents=True, exist_ok=True)
    (H / "selector-shared" / "perturbed.jsonl.done").write_text("")
    driver.selector_chain("a", SPEC)
    others = ["46d1_train-d-r151-l12-18-24.sh", "47a6_rlfull6-a3-l12-18-24.sh", "47d4r2_rlfull4-d-r150-l12-18-24.sh",
              "350_navtest-x.sh", "z_431_train-a3-l12-18-24-s1.sh"]
    names = sorted(gpu_jobs() + others)
    assert names == ["350_navtest-x.sh", "46d1_train-d-r151-l12-18-24.sh", "470a-60h0_selector-a-gen-half0.sh",
                     "470a-60h1_selector-a-gen-half1.sh", "470a-620_selector-a-xfit-s0.sh",
                     "47a6_rlfull6-a3-l12-18-24.sh", "47d4r2_rlfull4-d-r150-l12-18-24.sh",
                     "48a-61_selector-a-infold.sh", "48a-621_selector-a-xfit-s1.sh",
                     "48a-622_selector-a-infold-s0.sh", "48a-623_selector-a-xfit-pdms.sh",
                     "48a-624_selector-a-xfit-listwise.sh", "48a-625_selector-a-xfit-imitation.sh",
                     "48a-626_selector-a-xfit-boundary.sh", "z_431_train-a3-l12-18-24-s1.sh"]
    assert driver.ADOPTED_VARIANT in driver.SEL_VARIANTS
    assert driver.chain_job("rehearsal", "60", "h0_selector-rehearsal-gen-half0", True) == "60rehearsalh0_selector-rehearsal-gen-half0"
    assert driver.chain_job("rehearsal", "62", "6_x", True, decisive=False) == "62rehearsal6_x"  # the rehearsal: unchanged
    assert driver.chain_job("a4", "62", "0_selector-a4-xfit-s0", False) < driver.chain_job("a6", "60", "h0_x", False)


# ---- R19: when RL improves no save of the tuned configuration -------------------------------------------------
def r19_world(tree, monkeypatch):  # noqa: F811
    """A4 whose tuned configuration is A3 (not the phase-1 A1), with every one of A3's full-RL saves below its start."""
    choices, _ = phase6_a_only(tree, monkeypatch)
    lc = driver.choose_layers("A")
    tuned = {"method": "A", "run": "a3-l12-18-24", "kind": "masked", "layers": lc["layers"], "flags": ["--soft-sigma", "1"],
             "choice": driver.choose_run("a3-l12-18-24", "masked"), "rule": "test"}
    (choices / "A_tuned.json").write_text(json.dumps(tuned))
    return choices, lc


def lower_rl_of(run, by, also=None):
    """A scorer that lowers every navdev save of ``run``'s RL folders by ``by`` (and raises ``also``'s rl_3000)."""
    def scorer(folder, split, exp_name, **kw):
        fake_scorer(folder, split, exp_name, **kw)
        delta = -by if (f"{run}{os.sep}navdev_rl_" in folder or f"{run}/navdev_rl_" in folder) else 0.0
        if also and "navdev_rl_3000" in folder and also in folder:
            delta += 0.05
        if delta:
            f = Path(driver.P["exp"]) / exp_name / "run" / "scores.csv"
            d = pd.read_csv(f)
            d["score"] = d["score"] + delta
            d.to_csv(f, index=False)
    return scorer


def test_r19_runs_the_phase1_configuration_when_rl_improves_no_save_and_deploys_the_better(tree, monkeypatch):  # noqa: F811
    choices, lc = r19_world(tree, monkeypatch)
    monkeypatch.setattr(driver, "SCORER", lower_rl_of("a3-l12-18-24-rlfull4", 0.05, also="a1-l12-18-24-rlfull4"))
    driver.phase6({})
    assert (Path(driver.P["queue"]) / "47a4_rlfull4-a3-l12-18-24.sh").exists()
    full_rl_outputs(Path(driver.P["heads"]) / "a3-l12-18-24-rlfull4", 4, 1.5)
    status = {}
    driver.phase6(status)  # A3 chooses update 0: R19 queues A1's full RL stage, and no spec is written yet
    assert json.loads((Path(driver.P["heads"]) / "a3-l12-18-24-rlfull4" / "choices_navdev.json").read_text())["checkpoint"] == "start"
    j = (Path(driver.P["queue"]) / "47a4_rlfull4-a1-l12-18-24.sh").read_text()
    assert "--run $H/a1-l12-18-24 --weights epoch_15" in j and "--updates 5000 --save-every 500" in j
    assert not (choices / "selector_spec_A4.json").exists() and status["phase6_full"]["A4"]["r19"]["triggered"]
    full_rl_outputs(Path(driver.P["heads"]) / "a1-l12-18-24-rlfull4", 4, 1.5)
    driver.phase6(status)  # A1's stage chooses rl_3000, above A3's start: the contender deploys A1's RL save
    spec = json.loads((choices / "selector_spec_A4.json").read_text())
    assert (spec["deployed_run"], spec["deployed_weights"], spec["updates"]) == ("a1-l12-18-24-rlfull4", "rl_3000", 3000)
    assert (spec["flags"], spec["rl_start_epoch"], spec["configuration"]) == ([], 15, "a1-l12-18-24") and "R19" in spec["rule"]
    r19 = json.loads((choices / "r19_a4.json").read_text())
    assert r19["deployed"] == "main" and r19["main"]["checkpoint"] == "rl_3000" and r19["tuned"]["checkpoint"] == "start"


def test_r19_keeps_the_tuned_configuration_on_a_tie_and_never_runs_without_a_trigger(tree, monkeypatch):  # noqa: F811
    choices, lc = r19_world(tree, monkeypatch)
    tch = {"checkpoint": "start", "updates": 0, "navdev_single_pdms": 0.83}
    calls = []

    def stage(method, recipe, run, tlc, start, rl):
        calls.append(run)
        return tch if run == "a3-l12-18-24" else {"checkpoint": "rl_500", "updates": 500, "navdev_single_pdms": 0.83}

    monkeypatch.setattr(driver, "full_rl_stage", stage)
    driver.phase6({})
    assert calls == ["a3-l12-18-24", "a1-l12-18-24"]
    spec = json.loads((choices / "selector_spec_A4.json").read_text())  # a tie: the tuned configuration's start
    assert (spec["deployed_run"], spec["deployed_weights"], spec["configuration"]) == ("a3-l12-18-24", "epoch_15", "a3-l12-18-24")
    assert json.loads((choices / "r19_a4.json").read_text())["deployed"] == "tuned"
    # the tuned configuration chose an RL save: R19 does not run
    (choices / "selector_spec_A4.json").unlink()
    tch.update(checkpoint="rl_1000", updates=1000, navdev_single_pdms=0.85)
    calls.clear()
    driver.phase6({})
    assert calls == ["a3-l12-18-24"]
    assert json.loads((choices / "selector_spec_A4.json").read_text())["deployed_weights"] == "rl_1000"


def test_the_half_generators_are_scored_on_navdev_once_their_paths_exist(chain):
    """R14's measure: the half generators against the deployed one, on navdev (reporting)."""
    H = Path(driver.P["heads"])
    st = driver.selector_chain("a", SPEC)
    assert "halves_navdev_single" not in st  # nothing predicted yet
    base = H / "selector-a-gen-half0-rl" / "navdev_rl_1500"
    (base / "steps8" / "single").mkdir(parents=True)
    (base / "steps8" / "single" / "shard_000_of_001.jsonl").write_text("{}\n")
    (base / "timing.json").write_text("{}")
    st = driver.selector_chain("a", SPEC)
    assert set(st["halves_navdev_single"]) == {"half0"} and 0.7 < st["halves_navdev_single"]["half0"] < 0.9


# ---- R20: seeds 1 and 2 of the final system (A4) and of the navtest leader (C6) ----------------------------------
R20_WORLD = {"A": ("a3-l12-18-24", "masked", ["--soft-sigma", "1", "--l1-weight", "1"], 5, 4, 4, 1.5),
             "C": ("c-r152-l12-18-24", "flow", ["--time-sampling", "beta"], 10, 6, 2, 2.0)}


def r20_world(tree, monkeypatch, adopted=None):  # noqa: F811
    """R16 done: A4 final, neither A4 nor C6 adopted its selector, both deployed an RL save of the tuned run."""
    from test_head_driver import TOKENS
    monkeypatch.setitem(driver.P, "cache", str(tree / "cache" / "h1"))
    (tree / "cache").mkdir(parents=True, exist_ok=True)
    for split in ("navdev", "navtest"):
        (tree / "cache" / f"drives_{split}.json").write_text(json.dumps({t: f"drive{i % 4}" for i, t in enumerate(TOKENS)}))
    H, choices = Path(driver.P["heads"]), Path(driver.P["heads"]) / "choices"
    choices.mkdir(parents=True, exist_ok=True)
    (choices / "final_navtest.json").write_text(json.dumps({"final": "A4"}))
    (choices / "final_system.json").write_text(json.dumps(
        {"final": "A4", "adopted": adopted or {"A4": False, "A6": True, "C6": False}}))
    for m, (run, kind, fl, start, recipe, steps, scale) in R20_WORLD.items():
        (choices / f"{m}_tuned.json").write_text(json.dumps({"method": m, "run": run, "kind": kind, "layers": "12,18,24",
                                                             "flags": fl, "choice": {"epoch": 15, "steps": steps,
                                                                                     "scale": scale}}))
        (choices / f"{m}_tuned_rl_start.json").write_text(json.dumps({"run": run, "epoch": start}))
        (choices / f"selector_spec_{m}{recipe}.json").write_text(json.dumps(
            {"deployed_run": f"{run}-rlfull{recipe}", "deployed_weights": "rl_2500"}))  # as A4's spec of Nov 23
        rd = H / f"{run}-rlfull{recipe}"
        (rd / "navdev_rl_2500").mkdir(parents=True)
        (rd / "navdev_rl_2500" / "timing.json").write_text("{}")
        (rd / "choices_navdev.json").write_text(json.dumps(
            {"run": rd.name, "start": {"run": run, "weights": f"epoch_{start}"}, "checkpoint": "rl_2500", "updates": 2500,
             "steps": steps, "scale": scale, "single_by_checkpoint": {"0": 0.8, "2500": 0.85},
             "best_of_6_by_checkpoint": {"0": 0.9, "2500": 0.92}}))
    for s in (1, 2):  # R17 queued A3's two seeds under its own job names, with the same flags
        (Path(driver.P["queue"]) / f"z_43{s}_train-a3-l12-18-24-s{s}.sh").write_text("#!/bin/bash\n")
    return H, choices


def test_r20_runs_seeds_1_and_2_of_a4_and_c6_through_the_pipeline_and_reports_c6_minus_a4(tree, monkeypatch):  # noqa: F811
    from test_head_driver import TOKENS
    H, choices = r20_world(tree, monkeypatch)
    q = Path(driver.P["queue"])
    status = {}
    driver.seed_replication(status)  # pass 1: only C's seeds need training; A's are R17's runs
    assert sorted(j for j in os.listdir(q) if "train" in j) == [
        "z_431_train-a3-l12-18-24-s1.sh", "z_432_train-a3-l12-18-24-s2.sh",
        "z_44c1_train-c-r152-l12-18-24-s1.sh", "z_44c2_train-c-r152-l12-18-24-s2.sh"]
    j = (q / "z_44c2_train-c-r152-l12-18-24-s2.sh").read_text()
    assert "--kind flow --layers 12,18,24" in j and "--seed 2 --time-sampling beta" in j and "R20" in j
    assert not [f for f in os.listdir(q) if "rlfull" in f] and status["r20"]["C6 seed 1"] == "training"

    for m, (run, kind, _, start, _, _, _) in R20_WORLD.items():  # the four seed runs finish training
        for s in (1, 2):
            make_run(f"{run}-s{s}", kind, epochs=[start])
    driver.seed_replication(status)  # pass 2: each seed's full RL stage, from seed 0's starting epoch, queued last
    rl = sorted(f for f in os.listdir(q) if "rlfull" in f)
    assert rl == ["z_47a4s1_rlfull4-a3-l12-18-24-s1.sh", "z_47a4s2_rlfull4-a3-l12-18-24-s2.sh",
                  "z_47c6s1_rlfull6-c-r152-l12-18-24-s1.sh", "z_47c6s2_rlfull6-c-r152-l12-18-24-s2.sh"]
    a = (q / rl[0]).read_text()
    assert "--run $H/a3-l12-18-24-s1 --weights epoch_5" in a and "--seed 1 --updates 5000" in a
    assert "--adv dr --bc-balance 0 " in a and "OUT=$H/a3-l12-18-24-s1-rlfull4" in a and "RL seed 1 (R20)" in a
    c = (q / rl[3]).read_text()
    assert "--weights epoch_10" in c and "--seed 2 --updates 5000" in c and "--adv dr --bc-balance 1 " in c
    assert not (q / "47a4_rlfull4-a3-l12-18-24.sh").exists()  # seed 0's stage is R16's, read back, never re-queued

    monkeypatch.setattr(driver, "SCORER", peak_at(2500))
    for m, (run, kind, _, start, recipe, steps, scale) in R20_WORLD.items():
        for s in (1, 2):
            rl_outputs(H / f"{run}-s{s}-rlfull{recipe}", steps, scale, range(500, 5001, 500))
    driver.seed_replication(status)  # pass 3: R11 chooses each seed's checkpoint on navdev; navtest queued once
    ch = json.loads((H / "c-r152-l12-18-24-s1-rlfull6" / "choices_navdev.json").read_text())
    assert (ch["checkpoint"], ch["rounds"]) == ("rl_2500", 1) and "R20" in ch["rule"]
    nt = (q / "350_navtest-a3-l12-18-24-s2-rlfull4.sh").read_text()
    assert "--weights rl_2500 --split navtest" in nt and "--k 6 --scales 1.5" in nt
    assert not (choices / "r20_seed_replication.json").exists()  # no navtest score yet

    for m, (run, _, _, _, recipe, _, _) in R20_WORLD.items():
        for rd in (f"{run}-rlfull{recipe}", f"{run}-s1-rlfull{recipe}", f"{run}-s2-rlfull{recipe}"):
            (H / rd / "navtest_rl_2500").mkdir(parents=True, exist_ok=True)
            (H / rd / "navtest_rl_2500" / "timing.json").write_text("{}")

    def value(folder, split):  # C6 is 0.015 above A4 on navtest and 0.01 below it on navdev; seeds differ a little
        sd = 2 if "-s2-" in folder else (1 if "-s1-" in folder else 0)
        base = {("a3", "navtest"): 0.88, ("c-", "navtest"): 0.895, ("a3", "navdev"): 0.88, ("c-", "navdev"): 0.87}
        b = next(v for (k, sp), v in base.items() if sp == split and os.path.basename(os.path.dirname(
            os.path.dirname(os.path.dirname(folder)))).startswith(k))
        return pd.Series([b + 0.002 * sd + 0.001 * (1 if i % 2 else -1) for i in range(len(TOKENS))], index=TOKENS)

    monkeypatch.setattr(driver, "score", value)
    driver.seed_replication(status)  # pass 4: every seed scored, the contrast written
    r = json.loads((choices / "r20_seed_replication.json").read_text())
    assert [row["seed"] for row in r["contenders"]["A4"]["seeds"]] == [0, 1, 2]
    assert [row["checkpoint"] for row in r["contenders"]["C6"]["seeds"]] == ["rl_2500"] * 3
    assert r["contenders"]["A4"]["navtest_mean"] == 0.882 and r["contenders"]["C6"]["navtest_mean"] == 0.897
    assert r["contenders"]["C6"]["navtest_between_seed_sd"] == 0.002
    nt, nd = r["C6_minus_A4"]["navtest"], r["C6_minus_A4"]["navdev"]
    assert nt["difference"] == 0.015 and 0 < nt["ci"][0] <= 0.015 <= nt["ci"][1] and nt["seeds"] == {"A4": 3, "C6": 3}
    assert nd["difference"] == -0.01 and nd["ci"][0] <= -0.01 <= nd["ci"][1] < 0  # the sign follows C6 - A4
    assert "R20" in r["rule"] and status["r20"]["written"].endswith("r20_seed_replication.json")


def test_r20_waits_for_r16s_navtest_and_holds_a_contender_with_an_adopted_selector(tree, monkeypatch):  # noqa: F811
    H, choices = r20_world(tree, monkeypatch, adopted={"A4": False, "C6": True})
    (choices / "final_navtest.json").unlink()
    driver.seed_replication({})  # before R16's navtest scores: nothing at all
    assert sorted(os.listdir(driver.P["queue"])) == ["z_431_train-a3-l12-18-24-s1.sh", "z_432_train-a3-l12-18-24-s2.sh"]
    (choices / "final_navtest.json").write_text(json.dumps({"final": "A4"}))
    status = {}
    driver.seed_replication(status)
    assert status["r20"]["C6"].startswith("held") and not [f for f in os.listdir(driver.P["queue"]) if "c-r152" in f]
    assert status["r20"]["A4 seed 1"] == "training"  # A4's seeds proceed (R17's training jobs, nothing re-queued)
