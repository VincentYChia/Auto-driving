"""The phase driver of the path-generator study (docs/HEADS_PLAN.md), on the lab 4090 desktop's CPUs.

It keeps the study moving when no Claude session is watching:
- scores navdev path sets in the pre-registered order (R10) and writes each training run's choices;
- makes each method's layer choice;
- queues the next GPU jobs into ``~/queue/pending``: navtest paths at each run's choices, phase 3's seed runs,
  phase 2's continuity series, R12's RL recipe runs and phase 4's RL runs;
- chooses the RL recipe (R12) and each RL run's checkpoint (R11) on navdev, and makes phase 5's decision (R5);
- scores navtest for the report.

The GPU runner starts pending jobs in name order, so the name prefix sets the priority:
``350_`` navtest/navdev paths, ``351{recipe}{method}_`` R12/R13 recipe runs (they gate all of phase 4, so they go
before the remaining phase-1 training), ``36x``/``38x`` phase-1 training, ``39{method}{seed}_`` phase-3 seeds,
``40x``-``42x`` phase 2, then ``45{method}{seed}_`` phase-4 RL. Phase 4 sorts after every phase-2 and phase-3 job,
as in the plan's order (phases 2-3, then 4), so the diffusion comparison finishes before the RL stage starts
(2026-11-20). The order changes when results arrive, never what they are.

Every result is a file, so a restart repeats nothing that is done. It never trains, deletes or overwrites a job
file, and never touches Anvil. Phase 6's selector (R14) runs as one chain per spec file
``choices/selector_spec_<X>.json`` (``selector_chain``): ``60``/``61`` its candidates, ``62`` its selectors on the GPU,
``57`` candidate scoring in ``~/queue2`` (CPU). Phase 6's full RL stage, which a spec follows, is not automated yet.

    nohup ~/miniforge3/envs/pilot/bin/python pilot_sft/heads/driver.py --loop 300 \\
        >> ~/navsim_workspace/exp/heads/driver.log 2>&1 &
    python pilot_sft/heads/driver.py --once          # one pass, then exit
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

HOME = os.path.expanduser("~")
P = {  # paths; tests point them at a temporary folder
    "exp": f"{HOME}/navsim_workspace/exp",
    "heads": f"{HOME}/navsim_workspace/exp/heads",
    "cache": f"{HOME}/navsim_workspace/head_cache/h1_l12-18-24",
    "queue": f"{HOME}/queue/pending",
    "queue_root": f"{HOME}/queue",
    "queue2_root": f"{HOME}/queue2",
    "rl_view": f"{HOME}/navsim_workspace/training_data/pilot_v1/v1-nt-p12",
    "repo": f"{HOME}/Auto-driving",
    "navsim_py": f"{HOME}/miniforge3/envs/navsim/bin/python",
    "pilot_py": f"{HOME}/miniforge3/envs/pilot/bin/python",
    "devkit": f"{HOME}/navsim-v1.1",
    "h1": f"{HOME}/navsim_workspace/exp/pilot_sft/qwen35-0.8b-v1hard-p12-visfull-ep4",
}
GRID = {  # R7 and R10: each method's step grid, default steps, sampling settings, paths per setting
    "masked": {"steps": [1, 2, 4, 8], "default": 4, "scales": [0.5, 1.0, 1.5], "k": 6},
    "ddim": {"steps": [1, 2, 5, 10], "default": 5, "scales": [0.5, 1.0, 1.5], "k": 6},
    "flow": {"steps": [1, 2, 5], "default": 2, "scales": [0.5, 1.0, 2.0], "k": 6},
    "anchor": {"steps": [1, 2], "default": 2, "scales": [0.5, 1.0, 2.0], "k": 6},
    "regression": {"steps": [1], "default": 1, "scales": [], "k": 0},
}
EPOCHS = [5, 10, 15, 20]
PHASE1 = [  # (method, kind, layers, run name): the 10 phase-1 training runs
    ("A", "masked", "24", "a1-l24"), ("A", "masked", "12,18,24", "a1-l12-18-24"),
    ("D", "anchor", "24", "d-l24"), ("D", "anchor", "12,18,24", "d-l12-18-24"),
    ("B", "ddim", "24", "b-l24"), ("B", "ddim", "12,18,24", "b-l12-18-24"),
    ("C", "flow", "24", "c-l24"), ("C", "flow", "12,18,24", "c-l12-18-24"),
    ("R", "regression", "24", "r-l24"), ("R", "regression", "12,18,24", "r-l12-18-24"),
]
SERIES = {  # phase 2: A's continuity series as feature sets (R10); A1 = {"metric"}
    "a0": set(), "a1": {"metric"}, "a2": {"metric", "soft"}, "a3": {"metric", "soft", "l1"},
    "a4": {"metric", "soft", "l1", "offset"},
}
FLAGS = {"soft": ["--soft-sigma", "1"], "l1": ["--l1-weight", "1"], "offset": ["--offset"]}


def log(msg: str) -> None:
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".part", path)


def layers_tag(layers: str) -> str:
    return "l" + layers.replace(",", "-")


# ---- scoring --------------------------------------------------------------------------------------------------
def navdev_tokens() -> Optional[set]:
    f = os.path.join(P["repo"], "navsim_eval/navsim_configs/train_test_split/scene_filter/navdev.yaml")
    if not os.path.exists(f):
        return None
    lines = open(f, encoding="utf-8").read().split("\n")
    i = lines.index("tokens:")
    return {l.strip()[3:-1] for l in lines[i + 1:] if l.strip().startswith("- '")}


def ready(split: str) -> bool:
    """navdev needs its scene filter and its metric cache on this machine; navtest is always ready."""
    if split == "navtest":
        return True
    return navdev_tokens() is not None and os.path.isdir(os.path.join(P["exp"], "metric_cache_navdev_v1"))


def score_name(folder: str) -> str:
    """``heads-<run>-<split>_<weights>-steps<S>-<name>``, as scripts/desk/score_head_run.sh names it."""
    return "heads-" + os.path.relpath(folder, P["heads"]).replace(os.sep, "-").replace("/", "-")


def load_scores(exp_name: str) -> Optional[pd.Series]:
    files = sorted(glob.glob(os.path.join(P["exp"], exp_name, "*", "*.csv")))
    if not files:
        return None
    d = pd.read_csv(files[-1])
    d = d[d.token != "average"].set_index("token")
    if not d.valid.astype(bool).all():
        raise ValueError(f"{exp_name}: invalid scenes")
    return d["score"].astype(float)


def run_scorer(folder: str, split: str, exp_name: str, threads: int = 12, agent: Optional[List[str]] = None) -> None:
    env = dict(os.environ, NAVSIM_DEVKIT_ROOT=P["devkit"], OPENSCENE_DATA_ROOT=f"{HOME}/navsim_workspace/dataset",
               NUPLAN_MAPS_ROOT=f"{HOME}/navsim_workspace/dataset/maps", NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
               NAVSIM_EXP_ROOT=P["exp"], PYTHONPATH=P["repo"])
    agent = agent or ["agent._target_=navsim_eval.precomputed_agent.PrecomputedTrajectoryAgent",
                      f"+agent.predictions_path={folder}"]
    cmd = ["taskset", "-c", "16-31", "nice", "-n", "5", P["navsim_py"],
           f"{P['devkit']}/navsim/planning/script/run_pdm_score.py", f"train_test_split={split}",
           f"metric_cache_path={P['exp']}/metric_cache_{split}_v1",
           f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{P['repo']}/navsim_eval/navsim_configs]",
           f"worker.threads_per_node={threads}", f"experiment_name={exp_name}", *agent]
    t0 = time.time()
    with open(os.path.join(folder if os.path.isdir(folder) else P["heads"], f"score_{split}.log"), "w") as out:
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=P["repo"], env=env).returncode
    log(f"scored {exp_name} in {time.time() - t0:.0f} s (exit {rc})")


SCORER: Callable[[str, str, str], None] = run_scorer  # tests replace this


def score(folder: str, split: str) -> Optional[pd.Series]:
    """Per-scene PDMS of one path folder on ``split``, scoring it first if needed; None if it cannot be scored."""
    if not os.path.exists(os.path.join(folder, "shard_000_of_001.jsonl")) or not ready(split):
        return None
    exp_name = f"{score_name(folder)}_{split}_v1"
    s = load_scores(exp_name)
    if s is None:
        SCORER(folder, split, exp_name)
        s = load_scores(exp_name)
    if s is None:
        log(f"no scores for {exp_name}")
        return None
    want = 12146 if split == "navtest" else len(navdev_tokens())
    if len(s) != want:
        raise ValueError(f"{exp_name}: {len(s)} scenes, expected {want}")
    return s


def best_of(folders: List[str], split: str) -> Optional[float]:
    scores = [score(f, split) for f in folders]
    if any(s is None for s in scores):
        return None
    m = pd.concat(scores, axis=1)
    return float(m.max(axis=1).mean())


# ---- per-run choices (R10) ------------------------------------------------------------------------------------
def sample_folders(base: str, scale: float, k: int) -> List[str]:
    name = (lambda j: f"sample_{j}") if scale == 1.0 else (lambda j: f"scale{scale:g}_sample_{j}")
    return [os.path.join(base, name(j)) for j in range(k)]


def choose_run(run: str, kind: str) -> Optional[dict]:
    """The run's navdev choices (epoch, steps, sampling setting), in the R10 order; None until all are scored."""
    rd = os.path.join(P["heads"], run)
    out = os.path.join(rd, "choices_navdev.json")
    if os.path.exists(out):
        return read_json(out)
    g = GRID[kind]
    if not all(os.path.exists(os.path.join(rd, f"navdev_epoch_{e}", "timing.json")) for e in EPOCHS):
        return None
    stage1 = {}
    for e in EPOCHS:
        s = score(os.path.join(rd, f"navdev_epoch_{e}", f"steps{g['default']}", "single"), "navdev")
        if s is None:
            return None
        stage1[e] = round(float(s.mean()), 4)
    epoch = max(EPOCHS, key=lambda e: (stage1[e], -e))  # a tie goes to the earlier epoch
    stage2 = {}
    for st in g["steps"]:
        s = score(os.path.join(rd, f"navdev_epoch_{epoch}", f"steps{st}", "single"), "navdev")
        if s is None:
            return None
        stage2[st] = round(float(s.mean()), 4)
    steps = max(g["steps"], key=lambda st: (stage2[st], -st))  # a tie goes to fewer steps
    stage3, scale = {}, None
    for x in g["scales"]:
        b = best_of(sample_folders(os.path.join(rd, f"navdev_epoch_{epoch}", f"steps{steps}"), x, g["k"]), "navdev")
        if b is None:
            return None
        stage3[f"{x:g}"] = round(b, 4)
    if g["scales"]:
        scale = max(g["scales"], key=lambda x: (stage3[f"{x:g}"], x == 1.0))  # a tie goes to the training setting
    choice = {"run": run, "kind": kind, "epoch": epoch, "steps": steps, "scale": scale,
              "navdev_single_pdms": stage2[steps], "navdev_best_of_6": stage3.get(f"{scale:g}") if scale else None,
              "stage1_epoch_single": stage1, "stage2_steps_single": stage2, "stage3_scale_best_of_6": stage3,
              "rule": "docs/HEADS_PLAN.md R10", "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    write_json(out, choice)
    log(f"{run}: choices on navdev {json.dumps({k: choice[k] for k in ('epoch', 'steps', 'scale', 'navdev_single_pdms', 'navdev_best_of_6')})}")
    return choice


# ---- jobs -----------------------------------------------------------------------------------------------------
HEADER = "#!/bin/bash\n# Queued by pilot_sft/heads/driver.py on {when} (docs/HEADS_PLAN.md): {what}\ncd ~/Auto-driving\nsource scripts/anvil/config.local.sh\nC=$HOME/navsim_workspace/head_cache/h1_l12-18-24\nPY=~/miniforge3/envs/pilot/bin/python\n"


FIRST = "0_"  # a job file named with this prefix sorts before every other job, so the runner starts it next
LAST = "z_"  # ... and with this one after every other, so it runs only when nothing else is waiting


def queue_job(name: str, what: str, body: str, queue: str = "queue", first: bool = False, last: bool = False) -> bool:
    """Write ~/queue/pending/<name>.sh (GPU, one job at a time) or, with ``queue="queue2"``, ~/queue2/pending (CPU
    only) once. A job already pending, running or done in either queue, under any of its file names, is never
    written again. The runners take jobs in name order; ``first`` writes ``0_<name>.sh``, which goes ahead of the
    rest, and ``last`` writes ``z_<name>.sh``, which waits until nothing else is pending."""
    for root in (P["queue_root"], P["queue2_root"]):
        for sub in ("pending", "running", "done", "parked", "dropped", "failed"):
            if any(os.path.exists(os.path.join(root, sub, f"{pre}{name}.sh")) for pre in ("", FIRST, LAST)):
                return False
    pending = P["queue"] if queue == "queue" else os.path.join(P["queue2_root"], "pending")
    path = os.path.join(pending, f"{FIRST if first else (LAST if last else '')}{name}.sh")
    os.makedirs(pending, exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER.format(when=f"{dt.datetime.now():%Y-%m-%d %H:%M}", what=what) + body)
    os.chmod(path + ".part", 0o755)
    os.replace(path + ".part", path)
    log(f"queued {name}: {what}")
    return True


def predict_cmd(run: str, split: str, cache: str, epoch: int, steps: str, k: int, scale) -> str:
    sc = f"--scales {scale:g}" if (k and scale is not None) else ""
    return (f"taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $HOME/navsim_workspace/exp/heads/{run} "
            f"--split {split} --cache $C/{cache} --weights epoch_{epoch} --steps {steps} --k {k} {sc}\n")


def queue_navtest(run: str, kind: str, choice: dict) -> None:
    g = GRID[kind]
    if os.path.exists(os.path.join(P["heads"], run, f"navtest_epoch_{choice['epoch']}", "timing.json")):
        return
    body = predict_cmd(run, "navtest", "navtest", choice["epoch"], ",".join(map(str, g["steps"])),
                       g["k"] if choice["scale"] is not None else 0, choice["scale"])
    queue_job(f"350_navtest-{run}", f"navtest paths of {run} at its navdev choices (epoch {choice['epoch']}, all "
                                    f"steps, 6 samples at setting {choice['scale']})", body)


def train_body(run: str, kind: str, layers: str, seed: int, flags: List[str]) -> str:
    return ("RUN=$HOME/navsim_workspace/exp/heads/" + run + "\n"
            "for attempt in 1 2 3; do\n"
            f"  taskset -c 16-31 $PY pilot_sft/heads/train_head.py --cache $C --kind {kind} --layers {layers} "
            f"--out $RUN --epochs 20 --seed {seed} {' '.join(flags)} --resume && break\n"
            "  echo \"attempt $attempt failed, resuming\"\ndone\n"
            "[ -f $RUN/final.pt ] || { echo \"no final.pt\"; exit 1; }\n")


def report_navtest(run: str, kind: str, choice: Optional[dict]) -> Optional[dict]:
    """R1: navtest is reported for every configuration (R17 reads phase 2's series there): once a run has its navdev
    choices, queue its navtest paths at them and score them. Nothing before the choice; no choice reads the result."""
    if not choice:
        return None
    queue_navtest(run, kind, choice)
    return score_navtest_run(run, kind, choice)


def score_navtest_run(run: str, kind: str, choice: dict) -> Optional[dict]:
    """navtest scores at the run's choices (reporting only): singles at every step count, 6 samples at the chosen
    steps and setting. Returns the summary once complete."""
    rd = os.path.join(P["heads"], run)
    base = os.path.join(rd, f"navtest_epoch_{choice['epoch']}")
    if not os.path.exists(os.path.join(base, "timing.json")):
        return None
    out = os.path.join(rd, "navtest_at_choices.json")
    if os.path.exists(out):
        return read_json(out)
    g = GRID[kind]
    singles = {}
    for st in g["steps"]:
        s = score(os.path.join(base, f"steps{st}", "single"), "navtest")
        if s is None:
            return None
        singles[st] = round(float(s.mean()), 4)
    b6 = None
    if choice["scale"] is not None:
        b6 = best_of(sample_folders(os.path.join(base, f"steps{choice['steps']}"), choice["scale"], g["k"]), "navtest")
        if b6 is None:
            return None
        b6 = round(b6, 4)
    res = {"run": run, "epoch": choice["epoch"], "steps": choice["steps"], "scale": choice["scale"],
           "navtest_single_pdms_at_choice": singles[choice["steps"]], "navtest_single_by_steps": singles,
           "navtest_best_of_6": b6, "note": "reporting only; no choice uses navtest (R1)"}
    write_json(out, res)
    log(f"{run}: navtest at its choices {json.dumps({k: res[k] for k in ('navtest_single_pdms_at_choice', 'navtest_best_of_6')})}")
    subprocess.run([P["pilot_py"], os.path.join(P["repo"], "pilot_sft/heads/summarize.py"), "--run", rd, "--exp",
                    P["exp"], "--split", "navtest"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res


# ---- method-level choices and the next phases -----------------------------------------------------------------
def choose_layers(method: str) -> Optional[dict]:
    out = os.path.join(P["heads"], "choices", f"{method}.json")
    if os.path.exists(out):
        return read_json(out)
    runs = [(kind, layers, run) for m, kind, layers, run in PHASE1 if m == method]
    ch = {run: choose_run(run, kind) for kind, layers, run in runs}
    if any(c is None for c in ch.values()):
        return None
    best = max(runs, key=lambda r: (ch[r[2]]["navdev_single_pdms"], r[1] == "24"))  # a tie goes to layer 24 alone
    res = {"method": method, "kind": best[0], "layers": best[1], "run": best[2], "choice": ch[best[2]],
           "candidates": {r[2]: ch[r[2]]["navdev_single_pdms"] for r in runs}, "rule": "docs/HEADS_PLAN.md R10",
           "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    write_json(out, res)
    log(f"method {method}: layers {best[1]} ({res['candidates']})")
    return res


def seed_runs(method: str, lc: dict) -> None:
    """Phase 3: seeds 1 and 2 of the method's phase-1 configuration at its chosen layers. Each reuses seed 0's
    choices (R10), with navdev and navtest paths at those choices only."""
    c, kind = lc["choice"], lc["kind"]
    g = GRID[kind]
    for s in (1, 2):
        run = f"{lc['run']}-s{s}"
        k = g["k"] if c["scale"] is not None else 0
        body = (train_body(run, kind, lc["layers"], s, []) +
                predict_cmd(run, "navdev", "dev", c["epoch"], str(c["steps"]), k, c["scale"]) +
                predict_cmd(run, "navtest", "navtest", c["epoch"], str(c["steps"]), k, c["scale"]))
        queue_job(f"39{method.lower()}{s}_train-{run}", f"phase 3, seed {s} of {lc['run']} (choices of seed 0 reused)", body)


def seed_results(method: str, lc: dict) -> Dict[str, dict]:
    c, kind = lc["choice"], lc["kind"]
    g = GRID[kind]
    out = {}
    for s in (1, 2):
        run = f"{lc['run']}-s{s}"
        rd = os.path.join(P["heads"], run)
        res = {}
        for split, sub in (("navdev", f"navdev_epoch_{c['epoch']}"), ("navtest", f"navtest_epoch_{c['epoch']}")):
            base = os.path.join(rd, sub, f"steps{c['steps']}")
            if not os.path.exists(os.path.join(rd, sub, "timing.json")):
                continue
            sv = score(os.path.join(base, "single"), split)
            if sv is not None:
                res[f"{split}_single_pdms"] = round(float(sv.mean()), 4)
            if c["scale"] is not None:
                b = best_of(sample_folders(base, c["scale"], g["k"]), split)
                if b is not None:
                    res[f"{split}_best_of_6"] = round(b, 4)
        if res:
            out[run] = res
    if out:
        write_json(os.path.join(P["heads"], "choices", f"{method}_seeds.json"), out)
    return out


def flags(fs: set) -> List[str]:
    """The training flags of a set of A's continuity features (phase 2's series, R10)."""
    out = [] if "metric" in fs else ["--no-metric-embed"]
    for f in ("soft", "l1", "offset"):
        if f in fs:
            out += FLAGS[f]
    return out


def series(lc: dict) -> None:
    """Phase 2 (analysis only, R2): A's continuity series at A's layers, then leave-one-out from the series' best,
    then the 0.25 m and 0.05 m bins on it, and R17's seeds. The follow-up runs feed no choice (A's tuned configuration
    reads the series alone, R15), so they are queued last and run when nothing else waits (reorder of 2026-11-21)."""
    layers, tag = lc["layers"], layers_tag(lc["layers"])
    name_of = {cfg: (lc["run"] if cfg == "a1" else f"{cfg}-{tag}") for cfg in SERIES}

    for i, cfg in enumerate(["a0", "a2", "a3", "a4"]):
        run = name_of[cfg]
        queue_job(f"40{i}_train-{run}", f"phase 2, configuration {cfg.upper()} at layers {layers}",
                  train_body(run, "masked", layers, 0, flags(SERIES[cfg])) +
                  f"bash scripts/desk/navdev_grid.sh $RUN masked\n")
    ch = {cfg: choose_run(name_of[cfg], "masked") for cfg in SERIES}
    for cfg, c in ch.items():
        if c and cfg != "a1":
            report_navtest(name_of[cfg], "masked", c)
    if any(v is None for v in ch.values()):
        return
    order = list(SERIES)
    best = max(order, key=lambda cfg: (ch[cfg]["navdev_single_pdms"], -order.index(cfg)))  # a tie goes to the simpler
    write_json(os.path.join(P["heads"], "choices", "A_series.json"),
               {"best": best, "single_pdms": {cfg: ch[cfg]["navdev_single_pdms"] for cfg in order},
                "note": "analysis only (R2); not used in the method decision"})
    feats = SERIES[best]
    known = {frozenset(v) for v in SERIES.values()}
    for f in sorted(feats):
        rest = feats - {f}
        if frozenset(rest) in known:
            continue
        run = f"{best}-no-{f}-{tag}"
        queue_job(f"41{f[0]}_train-{run}", f"phase 2, leave-one-out: {best.upper()} without '{f}'",
                  train_body(run, "masked", layers, 0, flags(rest)) + "bash scripts/desk/navdev_grid.sh $RUN masked\n",
                  last=True)
        report_navtest(run, "masked", choose_run(run, "masked"))
    for b in ("0.25", "0.05"):
        run = f"{best}-bin{b.replace('.', '')}-{tag}"
        queue_job(f"42{b.replace('.', '')}_train-{run}", f"phase 2, {best.upper()} with {b} m bins",
                  train_body(run, "masked", layers, 0, flags(feats) + ["--bin-m", b]) +
                  "bash scripts/desk/navdev_grid.sh $RUN masked\n", last=True)
        report_navtest(run, "masked", choose_run(run, "masked"))
    if best != "a1":  # R17: the series' best gets seeds 1 and 2, at its own R10 choices, against A1's three seeds
        cb, g = ch[best], GRID["masked"]
        res = {}
        for sd in (1, 2):
            run = f"{name_of[best]}-s{sd}"
            queue_job(f"43{sd}_train-{run}", f"phase 2 (R17), seed {sd} of the continuity series' best, {best.upper()}",
                      train_body(run, "masked", layers, sd, flags(feats)) +
                      predict_cmd(run, "navdev", "dev", cb["epoch"], str(cb["steps"]), g["k"], cb["scale"]) +
                      predict_cmd(run, "navtest", "navtest", cb["epoch"], str(cb["steps"]), g["k"], cb["scale"]),
                      last=True)
            r = {}
            for split in ("navdev", "navtest"):
                base = os.path.join(P["heads"], run, f"{split}_epoch_{cb['epoch']}", f"steps{cb['steps']}")
                if not os.path.exists(os.path.join(os.path.dirname(base), "timing.json")):
                    continue
                sv = score(os.path.join(base, "single"), split)
                if sv is not None:
                    r[f"{split}_single_pdms"] = round(float(sv.mean()), 4)
                b6 = best_of(sample_folders(base, cb["scale"], g["k"]), split)
                if b6 is not None:
                    r[f"{split}_best_of_6"] = round(b6, 4)
            if r:
                res[run] = r
        if res:
            write_json(os.path.join(P["heads"], "choices", "A_series_best_seeds.json"),
                       {"best": best, "runs": res, "rule": "docs/HEADS_PLAN.md R17"})


#
# ---- phase 4: the matched short RL stage (R11) and the method decision (R5) ----------------------------------
RL_SEEDS = {"A": (0, 1, 2), "D": (0, 1, 2), "B": (0,), "C": (0,)}
RL_CHECKPOINTS = (250, 500, 750, 1000, 1250, 1500)


def navtrain_ready() -> bool:
    """The RL reward needs the navtrain metric cache on this machine, marked ready once its paths are rewritten."""
    return os.path.exists(os.path.join(P["exp"], "metric_cache_navtrain_v1", ".ready"))


def rl_start(method: str, lc: dict) -> Optional[dict]:
    """R11: the epoch with the highest navdev best-of-6 at the chosen steps and setting, on seed 0's run."""
    out = os.path.join(P["heads"], "choices", f"{method}_rl_start.json")
    if os.path.exists(out):
        return read_json(out)
    c, g = lc["choice"], GRID[lc["kind"]]
    rd = os.path.join(P["heads"], lc["run"])
    b6 = {}
    for e in EPOCHS:
        b = best_of(sample_folders(os.path.join(rd, f"navdev_epoch_{e}", f"steps{c['steps']}"), c["scale"], g["k"]),
                    "navdev")
        if b is None:
            return None
        b6[e] = round(b, 4)
    epoch = max(EPOCHS, key=lambda e: (b6[e], -e))  # a tie goes to the earlier epoch
    res = {"method": method, "epoch": epoch, "navdev_best_of_6_by_epoch": b6, "rule": "docs/HEADS_PLAN.md R11"}
    write_json(out, res)
    log(f"method {method}: RL starts from epoch {epoch} ({b6})")
    return res


RECIPES = {  # R12 (RL0-RL3) and R13 (RL4-RL7): the candidate RL recipes, the same for every method
    0: {"lr": 1e-5, "kl_beta": 0.0, "g": 8, "scenes": 32},
    1: {"lr": 1e-5, "kl_beta": 0.04, "g": 8, "scenes": 32},
    2: {"lr": 3e-6, "kl_beta": 0.04, "g": 8, "scenes": 32},
    3: {"lr": 3e-6, "kl_beta": 0.04, "g": 16, "scenes": 16},
    4: {"lr": 1e-5, "kl_beta": 0.0, "g": 8, "scenes": 32, "adv": "dr", "lp_floor": 0.1},
    5: {"lr": 1e-5, "kl_beta": 0.0, "g": 8, "scenes": 32, "bc": 1.0, "lp_floor": 0.1},
    6: {"lr": 1e-5, "kl_beta": 0.0, "g": 8, "scenes": 32, "adv": "dr", "bc": 1.0, "lp_floor": 0.1},
    7: {"lr": 1e-5, "kl_beta": 0.0, "g": 8, "scenes": 32, "lp_floor": 0.1},
}
GAUSSIAN_STEPS = ("anchor", "ddim", "flow")  # kinds whose log-probability has Gaussian steps (R13's SD floor)
RECIPE_METHODS = ("A", "D")
RECIPE_CHECKPOINTS = (250, 500)


def recipe_settings(recipe: int, kind: str) -> dict:
    """The recipe's settings as they act on this kind: the SD floor does nothing without Gaussian steps."""
    s = {"adv": "grpo", "bc": 0.0, "lp_floor": 0.0, **RECIPES[recipe]}
    if kind not in GAUSSIAN_STEPS:
        s["lp_floor"] = 0.0
    return s


def recipe_run(recipe: int, kind: str) -> int:
    """The earliest recipe that acts identically on this kind; its run stands for both (R13: A's RL7 is RL0)."""
    return min(q for q in RECIPES if recipe_settings(q, kind) == recipe_settings(recipe, kind))


def rl_body(lc: dict, sft: str, rl: str, epoch: int, seed: int, recipe: int, checkpoints) -> str:
    c, g, r = lc["choice"], GRID[lc["kind"]], recipe_settings(recipe, lc["kind"])
    k = g["k"] if c["scale"] is not None else 0
    sc = f"--scales {c['scale']:g}" if k else ""
    return f"""H=$HOME/navsim_workspace/exp/heads; NC=$HOME/navsim_workspace/exp/metric_cache_navtrain_v1; OUT=$H/{rl}
export NAVSIM_DEVKIT_ROOT=$HOME/navsim-v1.1 OPENSCENE_DATA_ROOT=$HOME/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=$HOME/navsim_workspace/dataset/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$HOME/navsim_workspace/exp
until [ -f $NC/.ready ]; do sleep 300; done
mkdir -p $OUT
# update 0, the start: navdev paths of the supervised run at the starting epoch, if not written yet
[ -f $H/{sft}/navdev_epoch_{epoch}/timing.json ] || taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $H/{sft} \\
  --weights epoch_{epoch} --split navdev --cache $C/dev --steps {c['steps']} --k {k} {sc}
PYTHONPATH=$HOME/Auto-driving setsid nohup taskset -c 16-31 nice -n 5 ~/miniforge3/envs/navsim/bin/python pilot_sft/heads/pdm_server.py serve --cache $NC --workers 12 --port 47411 > $OUT/server.log 2>&1 < /dev/null &
for i in $(seq 120); do grep -q serving $OUT/server.log && break; sleep 5; done
taskset -c 16-31 $PY pilot_sft/heads/rl_head.py --run $H/{sft} --weights epoch_{epoch} --out $OUT \\
  --rl-scenes $HOME/navsim_workspace/training_data/pilot_v1/v1-nt-p12 --steps {c['steps']} --scale {(c['scale'] or 1.0):g} \\
  --lr {r['lr']:g} --kl-beta {r['kl_beta']:g} --g {r['g']} --scenes-per-update {r['scenes']} \\
  --adv {r['adv']} --bc-balance {r['bc']:g} --lp-sd-floor {r['lp_floor']:g} \\
  --seed {seed} --updates {checkpoints[-1]} --save-every {checkpoints[0]} --resume
rc=$?
pkill -f "pdm_server.py serve --cache $NC"
[ $rc -eq 0 ] || exit $rc
for u in {' '.join(map(str, checkpoints))}; do
  [ -f $OUT/navdev_rl_$u/timing.json ] || taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $OUT --weights rl_$u \\
    --split navdev --cache $C/dev --steps {c['steps']} --k {k} {sc}
done
"""


PHASE4_RECIPES = (4, 6)  # R16: phase 4 runs RL4 (as registered) and RL6 on every method, same seeds and updates


def phase4_recipes(base: int) -> tuple:
    """The recipes phase 4 runs: R12/R13's chosen one first, then R16's others."""
    return (base,) + tuple(r for r in PHASE4_RECIPES if r != base)


def rl_name(sft: str, recipe: int, base: int) -> str:
    """Phase 4's RL run of ``sft``: the registered recipe's run keeps the plain name, R16's other recipe a suffix."""
    return f"{sft}-rl" if recipe == base else f"{sft}-rl{recipe}"


def queue_rl(method: str, lc: dict, start: dict, recipe: int, base: Optional[int] = None) -> None:
    """Phase 4 proper (R11; R16: each recipe of ``phase4_recipes``): every RL seed of the method, once its supervised
    seed exists."""
    base = recipe if base is None else base
    for s in RL_SEEDS[method]:
        sft = lc["run"] if s == 0 else f"{lc['run']}-s{s}"
        if not os.path.exists(os.path.join(P["heads"], sft, "final.pt")):
            continue
        job = (f"45{method.lower()}{s}_rl-{sft}" if recipe == base
               else f"45{method.lower()}{s}r{recipe}_rl{recipe}-{sft}")
        queue_job(job, f"phase 4, RL{recipe} (seed {s}) of {sft} from epoch {start['epoch']}",
                  rl_body(lc, sft, rl_name(sft, recipe, base), start["epoch"], s, recipe, RL_CHECKPOINTS))


def checkpoint_scores(rl: str, lc: dict, sft: str, epoch: int, checkpoints) -> Optional[tuple]:
    """navdev single-path PDMS and best-of-6 at update 0 (the start, no RL) and each saved checkpoint, as
    ({update: value}, {update: value}); None until every one is predicted and scored."""
    rd = os.path.join(P["heads"], rl)
    c, g = lc["choice"], GRID[lc["kind"]]
    single, b6 = {}, {}
    for u in (0,) + tuple(checkpoints):
        wdir = (os.path.join(P["heads"], sft, f"navdev_epoch_{epoch}") if u == 0 else os.path.join(rd, f"navdev_rl_{u}"))
        if not os.path.exists(os.path.join(wdir, "timing.json")):
            return None
        base = os.path.join(wdir, f"steps{c['steps']}")
        s = score(os.path.join(base, "single"), "navdev")
        b = best_of(sample_folders(base, c["scale"], g["k"]), "navdev") if c["scale"] is not None else None
        if s is None or (c["scale"] is not None and b is None):
            return None
        single[u], b6[u] = round(float(s.mean()), 4), (round(b, 4) if b is not None else None)
    return single, b6


def best_update(single: dict) -> int:
    """R11's rule: the highest navdev single-path PDMS; a tie goes to the earlier update."""
    return max(single, key=lambda x: (single[x], -x))


def rl_choose(rl: str, lc: dict, sft: str, epoch: int, checkpoints, navtest: bool = True,
              extra: Optional[dict] = None) -> Optional[dict]:
    """R11/R12: among update 0 (the start, no RL) and the saved checkpoints, the one with the highest navdev
    single-path PDMS (a tie goes to the earlier one). Queues its navtest paths when ``navtest``. ``extra`` adds keys
    to the written choice (and may replace its rule)."""
    rd = os.path.join(P["heads"], rl)
    out = os.path.join(rd, "choices_navdev.json")
    if os.path.exists(out):
        res = read_json(out)
        for k in ("single_by_checkpoint", "best_of_6_by_checkpoint"):  # JSON keys are strings; callers use updates
            res[k] = {int(u): v for u, v in res[k].items()}
    else:
        c = lc["choice"]
        sc = checkpoint_scores(rl, lc, sft, epoch, checkpoints)
        if sc is None:
            return None
        single, b6 = sc
        u = best_update(single)
        res = {"run": rl, "start": {"run": sft, "weights": f"epoch_{epoch}"},
               "checkpoint": "start" if u == 0 else f"rl_{u}", "updates": u, "steps": c["steps"], "scale": c["scale"],
               "navdev_single_pdms": single[u], "navdev_best_of_6": b6[u], "single_by_checkpoint": single,
               "best_of_6_by_checkpoint": b6, "rule": "docs/HEADS_PLAN.md R11, R12 (update 0 allowed)",
               "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", **(extra or {})}
        write_json(out, res)
        log(f"{rl}: RL checkpoint {res['checkpoint']} (navdev single {single[u]}; by checkpoint {single})")
    if navtest:  # also for a choice made in an earlier pass, so a pass that stopped after choosing loses nothing
        queue_rl_navtest(res, lc)
    return res


def queue_rl_navtest(ch: dict, lc: dict) -> None:
    """Queue the navtest paths of an RL run's chosen checkpoint (update 0 means its start), once; reporting (R1)."""
    run_dir, weights = ((ch["start"]["run"], ch["start"]["weights"]) if ch["checkpoint"] == "start"
                        else (ch["run"], ch["checkpoint"]))
    if os.path.exists(os.path.join(P["heads"], run_dir, f"navtest_{weights}", "timing.json")):
        return
    k = GRID[lc["kind"]]["k"] if ch["scale"] is not None else 0
    sc = f"--scales {ch['scale']:g}" if k else ""
    queue_job(f"350_navtest-{ch['run']}", f"navtest paths of {ch['run']} at its navdev checkpoint {ch['checkpoint']}",
              f"taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $HOME/navsim_workspace/exp/heads/{run_dir} "
              f"--weights {weights} --split navtest --cache $C/navtest --steps {ch['steps']} --k {k} {sc}\n")


def chosen_paths(ch: dict, split: str) -> str:
    """The path folder (single) of an RL run's chosen checkpoint on ``split``."""
    run_dir, weights = ((ch["start"]["run"], ch["start"]["weights"]) if ch["checkpoint"] == "start"
                        else (ch["run"], ch["checkpoint"]))
    return os.path.join(P["heads"], run_dir, f"{split}_{weights}", f"steps{ch['steps']}")


def rl_navtest(rl: str, ch: dict, lc: dict) -> Optional[dict]:
    out = os.path.join(P["heads"], rl, "navtest_at_choices.json")
    if os.path.exists(out):
        return read_json(out)
    base = chosen_paths(ch, "navtest")
    if not os.path.exists(os.path.join(os.path.dirname(base), "timing.json")):
        return None
    s = score(os.path.join(base, "single"), "navtest")
    b = best_of(sample_folders(base, ch["scale"], GRID[lc["kind"]]["k"]), "navtest") if ch["scale"] is not None else None
    if s is None or (ch["scale"] is not None and b is None):
        return None
    res = {"run": rl, "checkpoint": ch["checkpoint"], "navtest_single_pdms_at_choice": round(float(s.mean()), 4),
           "navtest_best_of_6": round(b, 4) if b is not None else None, "note": "reporting only; no choice uses navtest (R1)"}
    write_json(out, res)
    log(f"{rl}: navtest {json.dumps(res)}")
    return res


def decide(per_method: Dict[str, List[pd.Series]], drives: Dict[str, str], n: int = 10000, seed: int = 0,
           fill: Optional[float] = None) -> dict:
    """R5 with R3's hierarchical bootstrap. Each resample redraws drives (with all their scenes) and every method's
    seeds with replacement. A method with one seed gets an added normal draw with the larger of A's and D's
    between-seed SDs. The leader continues only if its interval against every other method excludes 0."""
    tokens = sorted(set.intersection(*[set(s.index) for ss in per_method.values() for s in ss]))
    drv = np.array([drives[t] for t in tokens])
    uniq, inv = np.unique(drv, return_inverse=True)
    mats = {m: np.stack([s.loc[tokens].values for s in ss]) for m, ss in per_method.items()}  # [seeds, scenes]
    point = {m: float(v.mean()) for m, v in mats.items()}
    seed_sd = {m: float(v.mean(1).std(ddof=1)) for m, v in mats.items() if v.shape[0] > 1}
    if fill is None:  # R5: the larger of A's and D's between-seed SDs
        fill = max([seed_sd.get(m, 0.0) for m in ("A", "D")] or [0.0])
    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n) for m in mats}
    for i in range(n):
        di = rng.integers(0, len(uniq), len(uniq))
        w = np.bincount(di, minlength=len(uniq))[inv]  # each scene's weight: how often its drive was drawn
        for m, v in mats.items():
            si = rng.integers(0, v.shape[0], v.shape[0])
            draws[m][i] = float((v[si].mean(0) * w).sum() / w.sum()) + (rng.normal(0, fill) if v.shape[0] == 1 else 0.0)
    leader = max(point, key=point.get)
    vs = {}
    for m in mats:
        if m != leader:
            d = draws[leader] - draws[m]
            vs[m] = {"difference": round(point[leader] - point[m], 4),
                     "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    clear = all(v["ci"][0] > 0 for v in vs.values())
    return {"leader": leader, "clear_winner": clear, "point_navdev_single_pdms": {m: round(v, 4) for m, v in point.items()},
            "seeds": {m: int(v.shape[0]) for m, v in mats.items()}, "between_seed_sd": {m: round(v, 4) for m, v in seed_sd.items()},
            "sd_added_for_single_seed_methods": round(fill, 4), "leader_minus_other": vs, "scenes": len(tokens),
            "drives": int(len(uniq)), "rule": "docs/HEADS_PLAN.md R5 (navdev, after RL); navtest tests C1-C4 come after"}


def recipe_stage() -> Optional[int]:
    """R12 and R13: run every candidate recipe on A and on D (seed 0, from their R11 starting epochs, 500 updates),
    then choose the recipe whose better checkpoint (250 or 500) gives the highest navdev single-path change from the
    start, averaged over A and D (a tie goes to the earlier recipe). These runs are in no method comparison. If
    every recipe lowers both methods, phase 4 is held for the researcher (R13): None."""
    out = os.path.join(P["heads"], "choices", "rl_recipe.json")
    if os.path.exists(out):
        j = read_json(out)
        if {int(r) for r in j["mean_gain_navdev_single"]} >= set(RECIPES):
            return None if j["every_recipe_lowers_both_methods"] else j["recipe"]
        os.replace(out, out.replace(".json", f"_RL0-RL{max(map(int, j['mean_gain_navdev_single']))}.json"))  # kept
    if not navtrain_ready():
        return None
    ready_methods = {}
    for m in RECIPE_METHODS:  # both methods need their layer choice and starting epoch before any recipe run
        lc = choose_layers(m)
        start = rl_start(m, lc) if lc else None
        if not start:
            return None
        ready_methods[m] = (lc, start)
    for r in RECIPES:  # every recipe run is queued before any scoring, which can take a while
        for m in RECIPE_METHODS:
            lc, start = ready_methods[m]
            if recipe_run(r, lc["kind"]) == r:
                queue_job(f"351{r}{m.lower()}_rlrecipe-{lc['run']}",
                          f"R{12 if r < 4 else 13} recipe RL{r} on {lc['run']} (seed 0, from epoch {start['epoch']}, "
                          "500 updates)", rl_body(lc, lc["run"], f"{lc['run']}-rlrecipe{r}", start["epoch"], 0, r,
                                                  RECIPE_CHECKPOINTS))
    gains, per_recipe, stand_in = {}, {}, {}
    for r in RECIPES:
        per = {}
        for m in RECIPE_METHODS:
            lc, start = ready_methods[m]
            q = recipe_run(r, lc["kind"])
            rl = f"{lc['run']}-rlrecipe{q}"
            if q != r:
                stand_in[f"RL{r} on {m}"] = f"RL{q} (the same settings for {lc['kind']})"
            ch = rl_choose(rl, lc, lc["run"], start["epoch"], RECIPE_CHECKPOINTS, navtest=False)
            if ch:
                sb = ch["single_by_checkpoint"]
                per[m] = round(max(sb[u] for u in RECIPE_CHECKPOINTS) - sb[0], 4)
        if len(per) == len(RECIPE_METHODS):
            per_recipe[r], gains[r] = per, round(float(np.mean(list(per.values()))), 4)
    if len(gains) < len(RECIPES):
        return None
    best = max(RECIPES, key=lambda r: (gains[r], -r))
    hold = all(v < 0 for p in per_recipe.values() for v in p.values())
    res = {"recipe": best, "settings": RECIPES[best], "mean_gain_navdev_single": gains, "gain_by_method": per_recipe,
           "every_recipe_lowers_both_methods": hold, "runs_standing_in": stand_in,
           "phase4": ("held: every recipe lowered both methods' navdev single path; the researcher decides the next "
                      "step (R13)") if hold else f"started with RL{best}",
           "rule": "docs/HEADS_PLAN.md R12, R13", "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    write_json(out, res)
    log(f"R12/R13 recipe: RL{best}{' (phase 4 held)' if hold else ''} {json.dumps(res)}")
    return None if hold else best


def phase4(status: dict) -> None:
    rl_methods = {}
    recipe = recipe_stage() if status["navdev_ready"] else None
    status["rl_recipe"] = recipe
    for method in ("A", "D", "B", "C"):
        lc = choose_layers(method) if status["navdev_ready"] else None
        if not lc:
            continue
        start = rl_start(method, lc)
        if not start:
            continue
        status.setdefault("phase4", {})[method] = {"start_epoch": start["epoch"]}
        if recipe is None:
            continue
        seeds_done = all(os.path.exists(os.path.join(P["heads"], f"{lc['run']}-s{s}", "final.pt"))
                         for s in RL_SEEDS[method] if s > 0)
        for r in phase4_recipes(recipe):
            if seeds_done and navtrain_ready():  # a job waiting for the cache would hold the GPU queue
                queue_rl(method, lc, start, r, recipe)
            runs = []
            for s in RL_SEEDS[method]:
                sft = lc["run"] if s == 0 else f"{lc['run']}-s{s}"
                rl = rl_name(sft, r, recipe)
                ch = rl_choose(rl, lc, sft, start["epoch"], RL_CHECKPOINTS)
                if ch:
                    runs.append(ch)
                    rl_navtest(rl, ch, lc)
            key = "rl_runs_chosen" if r == recipe else f"rl{r}_runs_chosen"
            status["phase4"][method][key] = [(ch["run"], ch["checkpoint"]) for ch in runs]
            if len(runs) == len(RL_SEEDS[method]):
                rl_methods.setdefault(r, {})[method] = [score(os.path.join(chosen_paths(ch, "navdev"), "single"), "navdev")
                                                        for ch in runs]
    for r, per in rl_methods.items():  # R5 per recipe: reported, no longer deciding (R16)
        out = os.path.join(P["heads"], "choices", "decision_phase5.json" if r == recipe else f"decision_phase5_rl{r}.json")
        if set(per) >= {"A", "D", "B", "C"} and not os.path.exists(out):
            drives = read_json(os.path.join(os.path.dirname(P["cache"]), "drives_navdev.json"))
            res = decide(per, drives)
            write_json(out, {**res, "recipe": r, "role": "reported only; the final system is chosen by R16",
                             "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"})
            log(f"phase 5 (R5, RL{r}, reported only): {json.dumps(res)}")


# ---- phase 6: the selector (R14), one chain per spec file -----------------------------------------------------
# A spec ``choices/selector_spec_<X>.json`` names the deployed path generator the selector serves and how it was
# made. It is written after the full RL stage, or by the researcher (e.g. a runner-up's selector). Keys: ``kind``,
# ``layers``, ``flags`` (the configuration), ``rl_start_epoch``, ``recipe``, ``updates`` (the deployed generator's
# chosen update count; 0 = no RL), ``save_every``, ``deployed_run``, ``deployed_weights``, ``steps``, ``scale``, ``k``.
SEL_VARIANTS = {  # R14: the cross-fitted selector (2 seeds), the in-fold control, the ablations (seed 0 each)
    "xfit-s0": ("xfit", 0, []), "xfit-s1": ("xfit", 1, []), "infold-s0": ("infold", 0, []),
    "xfit-pdms": ("xfit", 0, ["--target", "pdms"]), "xfit-listwise": ("xfit", 0, ["--listwise", "0.25"]),
    "xfit-imitation": ("xfit", 0, ["--imitation", "0.1"]), "xfit-boundary": ("xfit", 0, ["boundary"]),
}
SCORE_PORT = 47412  # candidate scoring in ~/queue2, apart from the RL jobs' scoring service on 47411
NAVSIM_ENV = ("export NAVSIM_DEVKIT_ROOT=$HOME/navsim-v1.1 OPENSCENE_DATA_ROOT=$HOME/navsim_workspace/dataset\n"
              "export NUPLAN_MAPS_ROOT=$HOME/navsim_workspace/dataset/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 "
              "NAVSIM_EXP_ROOT=$HOME/navsim_workspace/exp\n")


def ensure_halves() -> str:
    """Phase 6's cross-fitting split (R14), written once: halves.json, each half's RL-scene tokens, and all of them."""
    import drive_map
    d = os.path.join(P["heads"], "halves")
    if not os.path.exists(os.path.join(d, "rl_tokens_all.txt")):
        s = drive_map.write_halves(os.path.join(P["cache"], "train", "meta.jsonl"), P["rl_view"], d)
        with open(os.path.join(d, "rl_tokens_all.txt.part"), "w", encoding="utf-8") as f:
            for h in (0, 1):
                f.write(open(os.path.join(d, f"rl_tokens_half{h}.txt"), encoding="utf-8").read())
        os.replace(os.path.join(d, "rl_tokens_all.txt.part"), os.path.join(d, "rl_tokens_all.txt"))
        log(f"phase 6 halves: drives {s['drives']}, RL scenes {s['rl_scenes']}")
    return d


def score_body(paths: str, out: str, boundary_from: Optional[str] = None) -> str:
    """A ~/queue2 (CPU) job: the PDM scoring service on the navtrain cache, then every candidate of ``paths``.
    With ``boundary_from`` (a token file), it first writes the boundary perturbations into ``paths``, and picks the 4
    boundary paths after scoring."""
    srv = f"pdm_server.py serve --cache $NC --workers 8 --port {SCORE_PORT}"
    make = (f"[ -f {paths}/perturbations.json ] || taskset -c 24-31 $PY pilot_sft/heads/boundary_paths.py make "
            f"--cache $C/train --only-tokens {boundary_from} --out {paths}\n") if boundary_from else ""
    pick = (f"taskset -c 24-31 $PY pilot_sft/heads/boundary_paths.py pick --paths {paths} --scores {out} "
            f"--out {os.path.dirname(out)}/boundary\n") if boundary_from else ""
    return (f"NC=$HOME/navsim_workspace/exp/metric_cache_navtrain_v1\n{NAVSIM_ENV}{make}"
            f"PYTHONPATH=$HOME/Auto-driving setsid nohup taskset -c 24-31 nice -n 5 {P['navsim_py']} "
            f"pilot_sft/heads/{srv} > {out}.server.log 2>&1 < /dev/null &\n"
            f"for i in $(seq 120); do grep -q serving {out}.server.log && break; sleep 5; done\n"
            f"taskset -c 24-31 $PY pilot_sft/heads/score_candidates.py --paths {paths} --out {out} --port {SCORE_PORT}\n"
            f"rc=$?\npkill -f \"{srv}\"\n[ $rc -eq 0 ] || exit $rc\n{pick}touch {out}.done\n")


def half_body(spec: dict, x: str, h: int, halves: str) -> str:
    """Half ``h``'s path generator (R14): the chosen configuration trained on that half of the training drives, the
    deployed RL recipe for the same number of updates on that half's RL scenes, then the 7 candidates for the other
    half's RL scenes (out of fold) and its navdev paths (half against deployed)."""
    sft = f"selector-{x}-gen-half{h}"
    flags = list(spec.get("flags", [])) + ["--drive-half", str(h)]
    body = train_body(sft, spec["kind"], spec["layers"], 0, flags)
    k, sc = spec["k"], f"--scales {spec['scale']:g}"
    if spec["updates"] > 0:
        r = recipe_settings(spec["recipe"], spec["kind"])
        body += (f"H=$HOME/navsim_workspace/exp/heads; NC=$HOME/navsim_workspace/exp/metric_cache_navtrain_v1; "
                 f"OUT=$H/{sft}-rl\n{NAVSIM_ENV}mkdir -p $OUT\n"
                 "PYTHONPATH=$HOME/Auto-driving setsid nohup taskset -c 16-31 nice -n 5 ~/miniforge3/envs/navsim/bin/python "
                 "pilot_sft/heads/pdm_server.py serve --cache $NC --workers 12 --port 47411 > $OUT/server.log 2>&1 < /dev/null &\n"
                 "for i in $(seq 120); do grep -q serving $OUT/server.log && break; sleep 5; done\n"
                 f"taskset -c 16-31 $PY pilot_sft/heads/rl_head.py --run $H/{sft} --weights epoch_{spec['rl_start_epoch']} "
                 f"--out $OUT --rl-scenes {P['rl_view']} --only-tokens {halves}/rl_tokens_half{h}.txt "
                 f"--steps {spec['steps']} --scale {spec['scale']:g} --lr {r['lr']:g} --kl-beta {r['kl_beta']:g} "
                 f"--g {r['g']} --scenes-per-update {r['scenes']} --adv {r['adv']} --bc-balance {r['bc']:g} "
                 f"--lp-sd-floor {r['lp_floor']:g} --seed 0 --updates {spec['updates']} "
                 f"--save-every {spec.get('save_every', spec['updates'])} --resume\n"
                 "rc=$?\npkill -f \"pdm_server.py serve --cache $NC --workers 12 --port 47411\"\n[ $rc -eq 0 ] || exit $rc\n")
        run, weights = f"{sft}-rl", f"rl_{spec['updates']}"
    else:
        run, weights = sft, f"epoch_{spec['rl_start_epoch']}"
    for split, cache, only in (("navtrain", "train", f"--only-tokens {halves}/rl_tokens_half{1 - h}.txt"),
                               ("navdev", "dev", "")):
        body += (f"taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $HOME/navsim_workspace/exp/heads/{run} "
                 f"--weights {weights} --split {split} --cache $C/{cache} {only} --steps {spec['steps']} --k {k} {sc}\n")
    return body


def variant_needs(v: str) -> List[str]:
    """The candidate sets selector variant ``v`` trains on (R14)."""
    data, _, extra = SEL_VARIANTS[v]
    return (["half0", "half1"] if data == "xfit" else ["infold"]) + (["boundary"] if extra == ["boundary"] else [])


ADOPTED_VARIANT = "xfit-s0"  # R14: the selector whose navdev result decides adoption, and so R16's final system


def chain_job(x: str, step: str, rest: str, first: bool, decisive: bool = True) -> str:
    """A selector-chain GPU job's file name (reorders of 2026-11-22; every contender still gets every job).
    - A contender's decisive jobs (the half generators and the ``ADOPTED_VARIANT`` selector) sort ahead of every full
      RL stage (``47...``), so a contender whose full RL is done gets its adoption next: the first complete
      generator-plus-selector system arrives about a day sooner, and the chain's first run after RL (half generators
      with RL, never run at scale) shows any fault early. The tuning runs (``46...``) still go first.
    - Its other jobs (the in-fold control, seed 1 and the ablations: R14 reports them, no choice reads them) sort
      after every full RL stage (``48...``), so they fill the GPU's gaps and wait for no decision.
    - The rehearsal (``first``) keeps the names its finished jobs carry."""
    if first:
        return f"{step}{x}{rest}"
    return f"470{x}-{step}{rest}" if decisive else f"48{x}-{step}{rest}"


def selector_chain(x: str, spec: dict) -> dict:
    """R14 for one deployed path generator, one step per pass, each job queued once: halves, the two half generators
    and the in-fold candidates (GPU), scoring every candidate set (CPU, ~/queue2), the selectors (GPU), adoption on
    navdev, then one navtest evaluation. Returns its state for driver_status.json.

    Two optional spec keys serve a rehearsal of the chain (a spec that is no contender, used for no choice):
    ``variants`` (a subset of SEL_VARIANTS; only the candidate sets they need are made and scored) and ``first``
    (its jobs go ahead of the rest of each queue)."""
    st: Dict[str, object] = {"spec": spec.get("deployed_run")}
    halves = ensure_halves()
    H, S = P["heads"], f"steps{spec['steps']}"
    variants = [v for v in SEL_VARIANTS if v in spec.get("variants", SEL_VARIANTS)]
    needed = {n for v in variants for n in variant_needs(v)}
    first = bool(spec.get("first", False))
    oof, sets = {}, {}
    for h in (0, 1):
        run, weights = ((f"selector-{x}-gen-half{h}-rl", f"rl_{spec['updates']}") if spec["updates"] > 0
                        else (f"selector-{x}-gen-half{h}", f"epoch_{spec['rl_start_epoch']}"))
        oof[h] = os.path.join(H, run, f"navtrain_{weights}", S)
        if f"half{h}" in needed:
            queue_job(chain_job(x, "60", f"h{h}_selector-{x}-gen-half{h}", first),
                      f"phase 6 (R14): selector {x}, half-{h} path generator and its out-of-fold candidates",
                      half_body(spec, x, h, halves), first=first)
        sets[f"half{h}"] = (oof[h], os.path.join(H, f"selector-{x}", f"candidates_oof_half{h}.jsonl"))
        # R14's measure "the half generators against the deployed one, on navdev" (reporting): each half generator's
        # navdev single path, scored once its paths exist (collect_results.py then shows it with the RL runs)
        if os.path.exists(os.path.join(H, run, f"navdev_{weights}", "timing.json")):
            v = score(os.path.join(H, run, f"navdev_{weights}", S, "single"), "navdev")
            if v is not None:
                st.setdefault("halves_navdev_single", {})[f"half{h}"] = round(float(v.mean()), 4)
    dep = os.path.join(H, spec["deployed_run"], f"navtrain_{spec['deployed_weights']}", S)
    if "infold" in needed:
        queue_job(chain_job(x, "61", f"_selector-{x}-infold", first, decisive=False),
                  f"phase 6 (R14): selector {x}, in-fold candidates of the deployed generator",
                  f"taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $HOME/navsim_workspace/exp/heads/"
                  f"{spec['deployed_run']} --weights {spec['deployed_weights']} --split navtrain --cache $C/train "
                  f"--only-tokens {halves}/rl_tokens_all.txt --steps {spec['steps']} --k {spec['k']} --scales {spec['scale']:g}\n",
                  first=first)
    sets["infold"] = (dep, os.path.join(H, f"selector-{x}", "candidates_infold.jsonl"))
    shared = os.path.join(H, "selector-shared")  # the boundary paths depend on no generator: one set for every chain
    sets["boundary"] = (os.path.join(shared, "perturbed"), os.path.join(shared, "perturbed.jsonl"))
    scored = {}
    for name, (paths, out) in sets.items():
        if name not in needed:
            continue
        ready_paths = name == "boundary" or os.path.exists(os.path.join(os.path.dirname(paths), "timing.json"))
        if ready_paths:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            job = "57boundary_score-selector-shared-boundary" if name == "boundary" else f"57{x}{name}_score-selector-{x}-{name}"
            queue_job(job, f"phase 6 (R14): PDM scores of every {name} candidate" + ("" if name == "boundary" else f", selector {x}"),
                      score_body(paths, out, f"{halves}/rl_tokens_all.txt" if name == "boundary" else None), queue="queue2",
                      first=first)
        scored[name] = os.path.exists(out + ".done")
    st["scored"] = scored
    gen_w = os.path.join(H, spec["deployed_run"], f"{spec['deployed_weights']}.pt")
    nav = os.path.join(H, spec["deployed_run"], f"navdev_{spec['deployed_weights']}", S)
    drives = os.path.dirname(P["cache"])
    for v in variants:
        data, seed, extra = SEL_VARIANTS[v]
        need = variant_needs(v)
        if extra == ["boundary"]:
            extra = ["--extra-candidates", os.path.join(shared, "boundary"), os.path.join(shared, "boundary", "scores.jsonl")]
        if not all(scored[n] for n in need):
            continue
        cands = " ".join(f"--candidates {sets[n][0]} {sets[n][1]}" for n in need if n != "boundary")
        out = f"$HOME/navsim_workspace/exp/heads/selector-{x}-{v}"
        queue_job(chain_job(x, "62", f"{list(SEL_VARIANTS).index(v)}_selector-{x}-{v}", first,
                            decisive=v == ADOPTED_VARIANT),
                  f"phase 6 (R14): selector {x}, {v}",
                  f"taskset -c 16-31 $PY pilot_sft/heads/train_selector.py --cache $C/train --generator {gen_w} {cands} "
                  f"--scale {spec['scale']:g} --k {spec['k']} --seed {seed} {' '.join(extra)} --out {out} --resume\n"
                  f"taskset -c 16-31 $PY pilot_sft/heads/apply_selector.py --selector {out} --cache $C/dev --paths {nav} "
                  f"--scale {spec['scale']:g} --k {spec['k']} --out {out}/navdev --exp {P['exp']} --split navdev "
                  f"--drives {drives}/drives_navdev.json\n", first=first)
    # adoption on navdev (R14): the cross-fitted seed-0 selector, by the point estimate; a gain is claimed only if
    # its drive interval excludes 0. Then one navtest evaluation, scored by NAVSIM's own scorer.
    rep_path = os.path.join(H, f"selector-{x}-xfit-s0", "navdev", "report.json")
    out = os.path.join(H, "choices", f"selector_{x}_adoption.json")
    if os.path.exists(rep_path) and not os.path.exists(out):
        r = read_json(rep_path)["report"]
        res = {"selector": f"selector-{x}-xfit-s0", "adopted": r["selected_minus_own"] > 0,
               "gain_claimed": r["selected_minus_own_ci"][0] > 0, "navdev_selected_minus_own": r["selected_minus_own"],
               "ci_drives": r["selected_minus_own_ci"], "navdev_selected_pdms": r["selected_pdms"],
               "navdev_own_pick_pdms": r["own_pick_pdms"], "rule": "docs/HEADS_PLAN.md R14",
               **({"role": spec["role"]} if "role" in spec else {}),
               "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
        write_json(out, res)
        log(f"selector {x}: adoption on navdev {json.dumps(res)}")
    if os.path.exists(out):  # navtest only after the final system is chosen on navdev (R16, final_system)
        st["adopted"] = read_json(out)["adopted"]
    return st


# ---- phase 6: post-decision tuning and the full RL stage (R15) ------------------------------------------------
FULL_RL_EVERY = 500
FULL_RL_CHECKPOINTS = tuple(range(FULL_RL_EVERY, 5001, FULL_RL_EVERY))  # 5,000 updates, saved every 500
R15_TUNING = {  # R15: 4 runs over each method's own main settings (A: its phase-2 series, already run)
    "B": [["--target", "eps"], ["--target", "v"], ["--schedule", "linear"], ["--target", "eps", "--schedule", "linear"]],
    "C": [["--flow-sigma", "0.3"], ["--flow-sigma", "1.0"], ["--time-sampling", "beta"], ["--time-sampling", "logit-normal"]],
    "D": [["--anchors", "10"], ["--anchors", "40"], ["--anchor-trunc", "0.02"], ["--anchor-trunc", "0.1"]],
}
R15_READY: set = {"B", "C", "D"}  # methods whose R15 training flags exist in train_head.py


FULL_RL_ROUND = 5000  # R18: the full RL stage runs in rounds of 5,000 updates ...
FULL_RL_ROUNDS = 4  # ... and continues while a round's last save is R11's choice, up to 20,000 updates


def full_rl_stage(method: str, recipe: int, run: str, tlc: dict, start: int, rl: str) -> Optional[dict]:
    """R15 with R18: the full RL stage of ``run`` in rounds of 5,000 updates, saved every 500, each round resuming
    the last (jobs ``47{m}{r}_`` then ``47{m}{r}r{k}_``). After each round R11's rule chooses among update 0 and every
    save so far; the stage continues only while that choice is the round's last save, up to 4 rounds. Returns the
    final choice (also written as the run's choices_navdev.json), or None while a round runs or is scored."""
    if os.path.exists(os.path.join(P["heads"], rl, "choices_navdev.json")):
        return rl_choose(rl, tlc, run, start, (), navtest=False)  # chosen before: read back
    for k in range(1, FULL_RL_ROUNDS + 1):
        cps = tuple(range(FULL_RL_EVERY, FULL_RL_ROUND * k + 1, FULL_RL_EVERY))
        queue_job(f"47{method.lower()}{recipe}{'' if k == 1 else f'r{k}'}_rlfull{recipe}-{run}",
                  f"phase 6 (R16, R15, R18): full RL stage, RL{recipe}, of {run} from epoch {start}, round {k}: "
                  f"to {cps[-1]:,} updates", rl_body(tlc, run, rl, start, 0, recipe, cps))
        sc = checkpoint_scores(rl, tlc, run, start, cps)
        if sc is None:
            return None
        if best_update(sc[0]) == cps[-1] and k < FULL_RL_ROUNDS:
            continue  # navdev still rising at the round's last save: the next round (logged once, when queued)
        return rl_choose(rl, tlc, run, start, cps, navtest=False,
                         extra={"rounds": k, "rule": "docs/HEADS_PLAN.md R11, R15, R18 (update 0 allowed)"})
    return None


R16_CONTENDERS = (("A", 4), ("A", 6), ("D", 4), ("D", 6))  # R16: at least A and D, each under RL4 and RL6


def drive_interval(x: np.ndarray, groups, n: int = 10000) -> List[float]:
    """95% interval of the mean of per-scene values ``x`` by the drive-level bootstrap (R3)."""
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    sm, c = np.bincount(inv, weights=x), np.bincount(inv)
    i = np.random.default_rng(0).integers(0, len(u), size=(n, len(u)))
    m = sm[i].sum(1) / c[i].sum(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def bc_gate() -> Optional[dict]:
    """R16 (amended 21:42): B and C join phase 6, under both recipes, if after phase 4 the interval of (the best of A
    and D) minus it includes 0, or it leads, on navdev single path or best-of-6. The interval is R5's hierarchical
    bootstrap over drives and seeds: every phase-4 seed at its chosen checkpoint, with the larger of A's and D's
    between-seed SDs added for single-seed methods. Each method takes its better recipe and A and D their best
    (method, recipe), by point estimate. None until every phase-4 run is chosen."""
    out = os.path.join(P["heads"], "choices", "phase6_gate_bc.json")
    if os.path.exists(out):
        return read_json(out)
    rf = os.path.join(P["heads"], "choices", "rl_recipe.json")
    if not os.path.exists(rf):
        return None
    base = read_json(rf)["recipe"]
    vals = {}  # (method, recipe) -> measure -> one per-scene series per seed
    for m in ("A", "D", "B", "C"):
        lc = choose_layers(m)
        start = rl_start(m, lc) if lc else None
        if not start:
            return None
        for r in phase4_recipes(base):
            per = {"single": [], "best_of_6": []}
            for sd in RL_SEEDS[m]:
                sft = lc["run"] if sd == 0 else f"{lc['run']}-s{sd}"
                ch = rl_choose(rl_name(sft, r, base), lc, sft, start["epoch"], RL_CHECKPOINTS, navtest=False)
                if not ch:
                    return None
                paths = chosen_paths(ch, "navdev")
                single = score(os.path.join(paths, "single"), "navdev")
                if ch["scale"] is not None:
                    b6 = pd.concat([score(f, "navdev") for f in sample_folders(paths, ch["scale"], GRID[lc["kind"]]["k"])],
                                   axis=1).max(axis=1)
                else:
                    b6 = single
                per["single"].append(single)
                per["best_of_6"].append(b6)
            vals[(m, r)] = per
    drives = read_json(os.path.join(os.path.dirname(P["cache"]), "drives_navdev.json"))
    mean = lambda ss: float(np.mean([x.mean() for x in ss]))  # noqa: E731
    res = {"rule": "docs/HEADS_PLAN.md R16 (amended 2026-11-20 21:42)", "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    for m in ("B", "C"):
        res[m] = {"joins": False, "tests": {}}
        for meas in ("single", "best_of_6"):
            best = max(((x, r) for x in ("A", "D") for r in phase4_recipes(base)), key=lambda k: mean(vals[k][meas]))
            mine = max(phase4_recipes(base), key=lambda r: mean(vals[(m, r)][meas]))
            sds = [float(np.std([x.mean() for x in vals[(x, r)][meas]], ddof=1))
                   for x in ("A", "D") for r in phase4_recipes(base) if len(vals[(x, r)][meas]) > 1]
            d = decide({"AD": vals[best][meas], m: vals[(m, mine)][meas]}, drives, fill=max(sds or [0.0]))
            leads = d["leader"] == m
            ci = None if leads else d["leader_minus_other"][m]["ci"]
            res[m]["tests"][meas] = {"recipe": mine, "against": f"{best[0]} RL{best[1]}", "leads": leads,
                                     "point": d["point_navdev_single_pdms"], "best_ad_minus_it_ci": ci,
                                     "seed_sd_added": d["sd_added_for_single_seed_methods"]}
            res[m]["joins"] = res[m]["joins"] or leads or ci[0] <= 0
    write_json(out, res)
    log(f"phase 6 gate for B and C (R16, amended): {json.dumps(res)}")
    return res


def contenders() -> List[tuple]:
    """R16's phase-6 contenders: (method, recipe) pairs. B and C follow A and D once the gate admits them."""
    out = list(R16_CONTENDERS)
    g = bc_gate()
    for m in ("B", "C"):
        if g and g[m]["joins"]:
            out += [(m, 4), (m, 6)]
    return out


def tuned_config(method: str, lc: dict) -> Optional[dict]:
    """R15: the phase-1 configuration or one of its 4 tuning runs, by navdev single path at each run's R10 choices
    (a tie goes to the phase-1 configuration). Returns {run, kind, layers, flags, choice}."""
    out = os.path.join(P["heads"], "choices", f"{method}_tuned.json")
    if os.path.exists(out):
        return read_json(out)
    tag, kind = layers_tag(lc["layers"]), lc["kind"]
    cands = [(lc["run"], [])]
    if method == "A":  # its phase-2 series, already queued by series(): A0, A2, A3, A4 against A1
        cands += [(f"{cfg}-{tag}", flags(SERIES[cfg])) for cfg in ("a0", "a2", "a3", "a4")]
    elif method in R15_READY:
        for i, fl in enumerate(R15_TUNING[method]):
            run = f"{method.lower()}-r15{i}-{tag}"
            queue_job(f"46{method.lower()}{i}_train-{run}", f"phase 6 (R15), tuning run {i} of {method}: {' '.join(fl)}",
                      train_body(run, kind, lc["layers"], 0, fl) + f"bash scripts/desk/navdev_grid.sh $RUN {kind}\n")
            cands.append((run, fl))
    else:
        log(f"phase 6: {method}'s R15 tuning runs need their training flags in train_head.py first")
        return None
    ch = [(run, fl, choose_run(run, kind)) for run, fl in cands]
    for run, _, c in ch:
        if run != lc["run"]:
            report_navtest(run, kind, c)
    if any(c is None for _, _, c in ch):
        return None
    run, fl, c = max(ch, key=lambda r: (r[2]["navdev_single_pdms"], r[0] == lc["run"]))
    res = {"method": method, "run": run, "kind": kind, "layers": lc["layers"], "flags": fl, "choice": c,
           "candidates": {r: x["navdev_single_pdms"] for r, _, x in ch}, "rule": "docs/HEADS_PLAN.md R15",
           "label": "post-decision tuning", "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    write_json(out, res)
    log(f"phase 6: {method}'s tuned configuration {run} ({res['candidates']})")
    return res


def phase6(status: dict) -> None:
    """R16 with R15 and R14, for every contender: its method's tuned configuration, the full RL stage with its own
    recipe, the chosen checkpoint, and the selector spec that starts ``selector_chain``. No navtest yet (R16)."""
    for method, recipe in contenders():
        x = f"{method}{recipe}"
        st = status.setdefault("phase6_full", {}).setdefault(x, {})
        lc = choose_layers(method)
        tuned = tuned_config(method, lc) if lc else None
        if not tuned:
            continue
        st["tuned"] = tuned["run"]
        tlc = {"kind": tuned["kind"], "choice": tuned["choice"], "run": tuned["run"]}
        out = os.path.join(P["heads"], "choices", f"{method}_tuned_rl_start.json")
        if not os.path.exists(out):  # R11 on the tuned run: the epoch with the highest navdev best-of-6
            c, g = tuned["choice"], GRID[tuned["kind"]]
            b6 = {}
            for e in EPOCHS:
                base = os.path.join(P["heads"], tuned["run"], f"navdev_epoch_{e}", f"steps{c['steps']}")
                if c["scale"] is not None:
                    b = best_of(sample_folders(base, c["scale"], g["k"]), "navdev")
                else:
                    v = score(os.path.join(base, "single"), "navdev")
                    b = None if v is None else float(v.mean())
                if b is None:
                    break
                b6[e] = round(b, 4)
            if len(b6) < len(EPOCHS):
                continue
            write_json(out, {"run": tuned["run"], "epoch": max(EPOCHS, key=lambda e: (b6[e], -e)),
                             "navdev_best_of_6_by_epoch": b6, "rule": "docs/HEADS_PLAN.md R11 via R15"})
        start = read_json(out)["epoch"]
        rl = f"{tuned['run']}-rlfull{recipe}"
        ch = full_rl_stage(method, recipe, tuned["run"], tlc, start, rl)
        if not ch:
            continue
        st["rounds"] = ch.get("rounds", 1)
        st["checkpoint"] = ch["checkpoint"]
        dep = (tuned, start, rl, ch)
        if ch["checkpoint"] == "start" and tuned["run"] != lc["run"]:
            dep = r19_fallback(x, method, recipe, lc, dep, st)
            if dep is None:
                continue
        cfg, start, rl, ch = dep
        spec = os.path.join(P["heads"], "choices", f"selector_spec_{x}.json")
        if not os.path.exists(spec):
            dep_run, dep_w = (cfg["run"], f"epoch_{start}") if ch["checkpoint"] == "start" else (rl, ch["checkpoint"])
            write_json(spec, {"method": method, "kind": cfg["kind"], "layers": cfg["layers"], "flags": cfg["flags"],
                              "rl_start_epoch": start, "recipe": recipe, "updates": ch["updates"],
                              "save_every": FULL_RL_EVERY, "deployed_run": dep_run, "deployed_weights": dep_w,
                              "steps": cfg["choice"]["steps"], "scale": cfg["choice"]["scale"],
                              "k": GRID[cfg["kind"]]["k"], "configuration": cfg["run"],
                              "rule": "docs/HEADS_PLAN.md R14, R15, R16" + (", R19" if "r19" in st else ""),
                              "written": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"})
            log(f"phase 6: selector spec for {x}: deployed {dep_run} {dep_w}")


def r19_fallback(x: str, method: str, recipe: int, lc: dict, tuned_dep: tuple, st: dict) -> Optional[tuple]:
    """R19: the tuned configuration's full RL stage chose update 0, so the method's phase-1 configuration gets the
    same full RL stage (R18's rounds, from phase 4's R11 starting epoch), and the contender deploys the higher of the
    two R11 choices by navdev single path, a tie staying with the tuned configuration. Returns (configuration, start
    epoch, RL run, choice) of the deployed one, or None while the phase-1 configuration's stage runs or is scored.
    The decision is written once, as ``choices/r19_<x>.json``."""
    tuned, tstart, trl, tch = tuned_dep
    main = {"method": method, "run": lc["run"], "kind": lc["kind"], "layers": lc["layers"], "flags": [],
            "choice": lc["choice"]}
    rs = rl_start(method, lc)
    if not rs:
        return None
    mstart, mrl = rs["epoch"], f"{lc['run']}-rlfull{recipe}"
    st["r19"] = {"triggered": True, "tuned": {"run": trl, "checkpoint": "start", "navdev_single_pdms": tch["navdev_single_pdms"]}}
    mch = full_rl_stage(method, recipe, main["run"], {"kind": main["kind"], "choice": main["choice"], "run": main["run"]},
                        mstart, mrl)
    if not mch:
        return None
    use_main = mch["navdev_single_pdms"] > tch["navdev_single_pdms"]  # a tie stays with the tuned configuration
    st["r19"].update(main={"run": mrl, "checkpoint": mch["checkpoint"], "navdev_single_pdms": mch["navdev_single_pdms"]},
                     deployed="main" if use_main else "tuned")
    out = os.path.join(P["heads"], "choices", f"r19_{x.lower()}.json")
    if not os.path.exists(out):
        write_json(out, {"contender": x, "tuned": st["r19"]["tuned"], "main": st["r19"]["main"],
                         "deployed": st["r19"]["deployed"], "rule": "docs/HEADS_PLAN.md R19",
                         "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"})
        log(f"phase 6: R19 for {x}: tuned {trl} chose update 0 ({tch['navdev_single_pdms']}); phase-1 configuration "
            f"{mrl} chose {mch['checkpoint']} ({mch['navdev_single_pdms']}); deployed: {st['r19']['deployed']}")
    return (main, mstart, mrl, mch) if use_main else tuned_dep


def selectors(status: dict) -> None:
    for f in sorted(glob.glob(os.path.join(P["heads"], "choices", "selector_spec_*.json"))):
        x = os.path.basename(f)[len("selector_spec_"):-len(".json")].lower()
        status.setdefault("phase6", {})[x] = selector_chain(x, read_json(f))


def system_scores(x: str, spec: dict, adopted: bool) -> Optional[pd.Series]:
    """A contender's per-scene navdev PDMS of its system path: the selected path if its selector is adopted (from the
    selector's navdev selection), otherwise the deployed generator's own pick."""
    nav = os.path.join(P["heads"], spec["deployed_run"], f"navdev_{spec['deployed_weights']}", f"steps{spec['steps']}")
    own = score(os.path.join(nav, "single"), "navdev")
    if not adopted or own is None:
        return own
    f = os.path.join(P["heads"], f"selector-{x}-xfit-s0", "navdev", "selection.jsonl")
    if not os.path.exists(f):
        return None
    rows = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
    return pd.Series({r["token"]: r["true_chosen"] for r in rows}, dtype=float)


def final_system(status: dict) -> None:
    """R16: once every contender's selector is adopted or not on navdev, the final system is the contender with the
    highest navdev PDMS of its system path (point estimate; intervals against it reported; an exact tie goes to the
    researcher, ``choices/final_system_researcher.json``). Only then is each contender's system path scored once on
    navtest, the final system's being the headline."""
    H, C = P["heads"], os.path.join(P["heads"], "choices")
    out = os.path.join(C, "final_system.json")
    specs = {f"{m}{r}": os.path.join(C, f"selector_spec_{m}{r}.json") for m, r in contenders()}
    if not os.path.exists(out):
        if not all(os.path.exists(f) for f in specs.values()):
            return
        ad = {x: os.path.join(C, f"selector_{x.lower()}_adoption.json") for x in specs}
        if not all(os.path.exists(f) for f in ad.values()):
            return
        sys_s = {x: system_scores(x.lower(), read_json(specs[x]), read_json(ad[x])["adopted"]) for x in specs}
        if any(v is None for v in sys_s.values()):
            return
        means = {x: round(float(v.mean()), 4) for x, v in sys_s.items()}
        top = sorted(means, key=means.get, reverse=True)
        tie = len(top) > 1 and means[top[0]] == means[top[1]]
        own = os.path.join(C, "final_system_researcher.json")
        final = read_json(own)["final"] if (tie and os.path.exists(own)) else (None if tie else top[0])
        if final is None:
            status["final_system"] = {"waiting": "an exact tie goes to the researcher", **means}
            log(f"final system (R16): a tie at the top, waiting for the researcher: {means}")
            return
        drives = read_json(os.path.join(os.path.dirname(P["cache"]), "drives_navdev.json"))
        vs = {}
        for x, v in sys_s.items():
            if x != final:
                d = (sys_s[final] - v).dropna()
                vs[x] = {"difference": round(float(d.mean()), 4),
                         "ci_drives": drive_interval(d.values, [drives[t] for t in d.index])}
        res = {"final": final, "tie": tie, "navdev_system_pdms": means,
               "adopted": {x: read_json(ad[x])["adopted"] for x in specs}, "final_minus_other": vs,
               "rule": "docs/HEADS_PLAN.md R16: navdev generator-plus-selector score; R5 reported only",
               "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
        write_json(out, res)
        log(f"final system (R16): {final} ({means})")
    fs = read_json(out)
    status["final_system"] = {"final": fs["final"]}
    drives_t = f"{os.path.dirname(P['cache'])}/drives_navtest.json"
    results = {}
    for x, f in specs.items():
        if x not in fs["adopted"]:
            continue  # a contender admitted after the choice is not part of it
        spec, adopted = read_json(f), fs["adopted"][x]
        test = os.path.join(H, spec["deployed_run"], f"navtest_{spec['deployed_weights']}")
        k = spec["k"] if (adopted and spec["scale"] is not None) else 0
        sc = f"--scales {spec['scale']:g}" if k else ""
        queue_job(f"350_navtest-{spec['deployed_run']}-{spec['deployed_weights']}",
                  f"phase 6 (R16): navtest paths of contender {x}'s deployed path generator, after the final choice",
                  f"taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $HOME/navsim_workspace/exp/heads/"
                  f"{spec['deployed_run']} --weights {spec['deployed_weights']} --split navtest --cache $C/navtest "
                  f"--steps {spec['steps']} --k {k} {sc}\n")
        if not os.path.exists(os.path.join(test, "timing.json")):
            continue
        paths = os.path.join(test, f"steps{spec['steps']}")
        if adopted:
            # apply_selector.py --exp reads every candidate's NAVSIM score file (R14's measures on navtest, reported
            # beside the headline) and stops without them, so the own pick and the K samples are scored first
            cands = [os.path.join(paths, "single")] + sample_folders(paths, spec["scale"], spec["k"])
            if any(score(f, "navtest") is None for f in cands):
                continue
            sel = os.path.join(H, f"selector-{x.lower()}-xfit-s0", "navtest")
            queue_job(f"350_navtest-selector-{x.lower()}", f"phase 6 (R16): contender {x}'s selector on navtest, once",
                      f"taskset -c 16-31 $PY pilot_sft/heads/apply_selector.py --selector "
                      f"$HOME/navsim_workspace/exp/heads/selector-{x.lower()}-xfit-s0 --cache $C/navtest --paths {paths} "
                      f"--scale {spec['scale']:g} --k {spec['k']} --out {sel} --exp {P['exp']} --split navtest "
                      f"--drives {drives_t}\n")
            if not os.path.exists(os.path.join(sel, "report.json")):
                continue
            v = score(os.path.join(sel, "selected"), "navtest")
        else:
            v = score(os.path.join(paths, "single"), "navtest")
        if v is not None:
            results[x] = round(float(v.mean()), 4)
    fin = os.path.join(C, "final_navtest.json")
    if len(results) == len(fs["adopted"]) and not os.path.exists(fin):
        write_json(fin, {"final": fs["final"], "headline_navtest_pdms": results[fs["final"]], "navtest_system_pdms": results,
                         "scorer": "NAVSIM v1.1 run_pdm_score.py, full navtest", "rule": "docs/HEADS_PLAN.md R16"})
        log(f"final system on navtest (R16): {fs['final']} {results[fs['final']]} ({results})")
    if os.path.exists(fin):
        status["final_system"]["navtest"] = read_json(fin)["headline_navtest_pdms"]


def references() -> dict:
    """H1's text output and constant velocity on navdev, the references (phase 0)."""
    out = {}
    if not ready("navdev"):
        return out
    h1 = os.path.join(P["h1"], "navdev_predictions")
    if os.path.exists(os.path.join(h1, "shard_000_of_001.jsonl")):
        s = load_scores("h1-text_navdev_v1")
        if s is None:
            SCORER(h1, "navdev", "h1-text_navdev_v1")
            s = load_scores("h1-text_navdev_v1")
        if s is not None:
            out["h1_text_navdev_pdms"] = round(float(s.mean()), 4)
    s = load_scores("cv_navdev_v1")
    if s is None:
        SCORER(P["heads"], "navdev", "cv_navdev_v1", agent=["agent=constant_velocity_agent"]) if SCORER is run_scorer \
            else SCORER(P["heads"], "navdev", "cv_navdev_v1")
        s = load_scores("cv_navdev_v1")
    if s is not None:
        out["constant_velocity_navdev_pdms"] = round(float(s.mean()), 4)
    return out


def one_pass() -> dict:
    status = {"time": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", "navdev_ready": ready("navdev"), "runs": {},
              "methods": {}}
    status["references"] = references()
    for method, kind, layers, run in PHASE1:
        rd = os.path.join(P["heads"], run)
        st = {"trained": os.path.exists(os.path.join(rd, "final.pt")),
              "navdev_grid": sum(os.path.exists(os.path.join(rd, f"navdev_epoch_{e}", "timing.json")) for e in EPOCHS)}
        c = choose_run(run, kind) if st["trained"] and status["navdev_ready"] else None
        if c:
            st["choice"] = {k: c[k] for k in ("epoch", "steps", "scale", "navdev_single_pdms", "navdev_best_of_6")}
            queue_navtest(run, kind, c)
            t = score_navtest_run(run, kind, c)
            if t:
                st["navtest"] = {k: t[k] for k in ("navtest_single_pdms_at_choice", "navtest_best_of_6")}
        status["runs"][run] = st
    for method in ("A", "D", "B", "C", "R"):
        lc = choose_layers(method) if status["navdev_ready"] else None
        if lc:
            status["methods"][method] = {"layers": lc["layers"], "run": lc["run"]}
            seed_runs(method, lc)
            status["methods"][method]["seeds"] = seed_results(method, lc)
            if method == "A":
                series(lc)
    status["navtrain_ready"] = navtrain_ready()
    phase4(status)
    if status["navdev_ready"] and navtrain_ready():
        phase6(status)
        selectors(status)
        final_system(status)
    write_json(os.path.join(P["heads"], "driver_status.json"), status)
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes; 0 = one pass")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    failed_in_a_row = 0
    while True:
        start = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
        health = {"pass_started": start, "ok": True}
        try:
            one_pass()
            failed_in_a_row = 0
        except Exception as e:  # keep the loop alive; the log and driver_health.json show what failed
            failed_in_a_row += 1
            health.update(ok=False, error=f"{type(e).__name__}: {e}")
            log(f"pass failed: {type(e).__name__}: {e}")
        # driver_status.json is written only by a pass that completes; this file is written after every pass, so a
        # driver that keeps failing shows up in it, not only in the log
        health.update(pass_ended=f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", failed_in_a_row=failed_in_a_row)
        write_json(os.path.join(P["heads"], "driver_health.json"), health)
        if args.once or not args.loop:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
