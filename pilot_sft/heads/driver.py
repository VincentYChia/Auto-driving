"""The phase driver of the path-generator study (docs/HEADS_PLAN.md), on the lab 4090 desktop's CPUs.

It keeps the study moving when no Claude session is watching:
- scores navdev path sets in the pre-registered order (R10) and writes each training run's choices;
- makes each method's layer choice;
- queues the next GPU jobs into ``~/queue/pending``: navtest paths at each run's choices, phase 3's seed runs,
  and phase 2's continuity series;
- scores navtest for the report.

Every result is a file, so a restart repeats nothing that is done. It never trains, deletes or overwrites a job
file, and never touches Anvil. Queuing stops at the end of phases 2 and 3: phase 4 (RL) needs its own code.

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


def queue_job(name: str, what: str, body: str) -> bool:
    """Write ~/queue/pending/<name>.sh once. A job already pending, running or done is never written again."""
    for sub in ("pending", "running", "done", "parked", "dropped", "failed"):
        if os.path.exists(os.path.join(P["queue_root"], sub, f"{name}.sh")):
            return False
    path = os.path.join(P["queue"], f"{name}.sh")
    os.makedirs(P["queue"], exist_ok=True)
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


def series(lc: dict) -> None:
    """Phase 2 (analysis only, R2): A's continuity series at A's layers, then leave-one-out from the series' best,
    then the 0.25 m and 0.05 m bins on it."""
    layers, tag = lc["layers"], layers_tag(lc["layers"])
    name_of = {cfg: (lc["run"] if cfg == "a1" else f"{cfg}-{tag}") for cfg in SERIES}

    def flags(fs: set) -> List[str]:
        out = [] if "metric" in fs else ["--no-metric-embed"]
        for f in ("soft", "l1", "offset"):
            if f in fs:
                out += FLAGS[f]
        return out

    for i, cfg in enumerate(["a0", "a2", "a3", "a4"]):
        run = name_of[cfg]
        queue_job(f"40{i}_train-{run}", f"phase 2, configuration {cfg.upper()} at layers {layers}",
                  train_body(run, "masked", layers, 0, flags(SERIES[cfg])) +
                  f"bash scripts/desk/navdev_grid.sh $RUN masked\n")
    ch = {cfg: choose_run(name_of[cfg], "masked") for cfg in SERIES}
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
                  train_body(run, "masked", layers, 0, flags(rest)) + "bash scripts/desk/navdev_grid.sh $RUN masked\n")
    for b in ("0.25", "0.05"):
        run = f"{best}-bin{b.replace('.', '')}-{tag}"
        queue_job(f"42{b.replace('.', '')}_train-{run}", f"phase 2, {best.upper()} with {b} m bins",
                  train_body(run, "masked", layers, 0, flags(feats) + ["--bin-m", b]) +
                  "bash scripts/desk/navdev_grid.sh $RUN masked\n")


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


RECIPES = {  # R12: the candidate RL recipes, the same for every method
    0: {"lr": 1e-5, "kl_beta": 0.0, "g": 8, "scenes": 32},
    1: {"lr": 1e-5, "kl_beta": 0.04, "g": 8, "scenes": 32},
    2: {"lr": 3e-6, "kl_beta": 0.04, "g": 8, "scenes": 32},
    3: {"lr": 3e-6, "kl_beta": 0.04, "g": 16, "scenes": 16},
}
RECIPE_METHODS = ("A", "D")
RECIPE_CHECKPOINTS = (250, 500)


def rl_body(lc: dict, sft: str, rl: str, epoch: int, seed: int, recipe: int, checkpoints) -> str:
    c, g, r = lc["choice"], GRID[lc["kind"]], RECIPES[recipe]
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
  --seed {seed} --updates {checkpoints[-1]} --save-every {checkpoints[0]} --resume
rc=$?
pkill -f "pdm_server.py serve --cache $NC"
[ $rc -eq 0 ] || exit $rc
for u in {' '.join(map(str, checkpoints))}; do
  taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $OUT --weights rl_$u --split navdev --cache $C/dev \\
    --steps {c['steps']} --k {k} {sc}
done
"""


def queue_rl(method: str, lc: dict, start: dict, recipe: int) -> None:
    """Phase 4 proper (R11, with R12's chosen recipe): every RL seed of the method, once its supervised seed exists."""
    for s in RL_SEEDS[method]:
        sft = lc["run"] if s == 0 else f"{lc['run']}-s{s}"
        if not os.path.exists(os.path.join(P["heads"], sft, "final.pt")):
            continue
        queue_job(f"396{method.lower()}{s}_rl-{sft}",
                  f"phase 4, RL{recipe} (seed {s}) of {sft} from epoch {start['epoch']}",
                  rl_body(lc, sft, f"{sft}-rl", start["epoch"], s, recipe, RL_CHECKPOINTS))


def rl_choose(rl: str, lc: dict, sft: str, epoch: int, checkpoints, navtest: bool = True) -> Optional[dict]:
    """R11/R12: among update 0 (the start, no RL) and the saved checkpoints, the one with the highest navdev
    single-path PDMS (a tie goes to the earlier one). Queues its navtest paths when ``navtest``."""
    rd = os.path.join(P["heads"], rl)
    out = os.path.join(rd, "choices_navdev.json")
    if os.path.exists(out):
        return read_json(out)
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
    u = max(single, key=lambda x: (single[x], -x))
    res = {"run": rl, "start": {"run": sft, "weights": f"epoch_{epoch}"},
           "checkpoint": "start" if u == 0 else f"rl_{u}", "updates": u, "steps": c["steps"], "scale": c["scale"],
           "navdev_single_pdms": single[u], "navdev_best_of_6": b6[u], "single_by_checkpoint": single,
           "best_of_6_by_checkpoint": b6, "rule": "docs/HEADS_PLAN.md R11, R12 (update 0 allowed)",
           "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    write_json(out, res)
    log(f"{rl}: RL checkpoint {res['checkpoint']} (navdev single {single[u]}; by checkpoint {single})")
    if navtest:
        run_dir, weights = (sft, f"epoch_{epoch}") if u == 0 else (rl, f"rl_{u}")
        if not os.path.exists(os.path.join(P["heads"], run_dir, f"navtest_{weights}", "timing.json")):
            k = g["k"] if c["scale"] is not None else 0
            sc = f"--scales {c['scale']:g}" if k else ""
            queue_job(f"350_navtest-{rl}", f"navtest paths of {rl} at its navdev checkpoint {res['checkpoint']}",
                      f"taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run $HOME/navsim_workspace/exp/heads/{run_dir} "
                      f"--weights {weights} --split navtest --cache $C/navtest --steps {c['steps']} --k {k} {sc}\n")
    return res


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


def decide(per_method: Dict[str, List[pd.Series]], drives: Dict[str, str], n: int = 10000, seed: int = 0) -> dict:
    """R5 with R3's hierarchical bootstrap. Each resample redraws drives (with all their scenes) and every method's
    seeds with replacement. A method with one seed gets an added normal draw with the larger of A's and D's
    between-seed SDs. The leader continues only if its interval against every other method excludes 0."""
    tokens = sorted(set.intersection(*[set(s.index) for ss in per_method.values() for s in ss]))
    drv = np.array([drives[t] for t in tokens])
    uniq, inv = np.unique(drv, return_inverse=True)
    mats = {m: np.stack([s.loc[tokens].values for s in ss]) for m, ss in per_method.items()}  # [seeds, scenes]
    point = {m: float(v.mean()) for m, v in mats.items()}
    seed_sd = {m: float(v.mean(1).std(ddof=1)) for m, v in mats.items() if v.shape[0] > 1}
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
    """R12: run every candidate recipe on A and on D (seed 0, from their R11 starting epochs, 500 updates), then
    choose the recipe whose better checkpoint (250 or 500) gives the highest navdev single-path change from the
    start, averaged over A and D (a tie goes to the earlier recipe). These runs are in no method comparison."""
    out = os.path.join(P["heads"], "choices", "rl_recipe.json")
    if os.path.exists(out):
        return read_json(out)["recipe"]
    if not navtrain_ready():
        return None
    ready_methods = {}
    for m in RECIPE_METHODS:  # both methods need their layer choice and starting epoch before any recipe run
        lc = choose_layers(m)
        start = rl_start(m, lc) if lc else None
        if not start:
            return None
        ready_methods[m] = (lc, start)
    gains, per_recipe = {}, {}
    for r in RECIPES:
        per = {}
        for m in RECIPE_METHODS:
            lc, start = ready_methods[m]
            rl = f"{lc['run']}-rlrecipe{r}"
            queue_job(f"395{r}{m.lower()}_rlrecipe-{lc['run']}",
                      f"R12 recipe RL{r} on {lc['run']} (seed 0, from epoch {start['epoch']}, 500 updates)",
                      rl_body(lc, lc["run"], rl, start["epoch"], 0, r, RECIPE_CHECKPOINTS))
            ch = rl_choose(rl, lc, lc["run"], start["epoch"], RECIPE_CHECKPOINTS, navtest=False)
            if ch:
                sb = ch["single_by_checkpoint"]
                per[m] = round(max(sb[u] for u in RECIPE_CHECKPOINTS) - sb[0], 4)
        if len(per) == len(RECIPE_METHODS):
            per_recipe[r], gains[r] = per, round(float(np.mean(list(per.values()))), 4)
    if len(gains) < len(RECIPES):
        return None
    best = max(RECIPES, key=lambda r: (gains[r], -r))
    res = {"recipe": best, "settings": RECIPES[best], "mean_gain_navdev_single": gains, "gain_by_method": per_recipe,
           "every_recipe_lowers_both_methods": all(v < 0 for p in per_recipe.values() for v in p.values()),
           "rule": "docs/HEADS_PLAN.md R12", "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    write_json(out, res)
    log(f"R12 recipe: RL{best} {json.dumps(res)}")
    return best


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
        if seeds_done and navtrain_ready():  # a job waiting for the cache would hold the GPU queue
            queue_rl(method, lc, start, recipe)
        runs = []
        for s in RL_SEEDS[method]:
            sft = lc["run"] if s == 0 else f"{lc['run']}-s{s}"
            ch = rl_choose(f"{sft}-rl", lc, sft, start["epoch"], RL_CHECKPOINTS)
            if ch:
                runs.append(ch)
                rl_navtest(f"{sft}-rl", ch, lc)
        status["phase4"][method]["rl_runs_chosen"] = [(ch["run"], ch["checkpoint"]) for ch in runs]
        if len(runs) == len(RL_SEEDS[method]):
            rl_methods[method] = [score(os.path.join(chosen_paths(ch, "navdev"), "single"), "navdev") for ch in runs]
    out = os.path.join(P["heads"], "choices", "decision_phase5.json")
    if set(rl_methods) >= {"A", "D", "B", "C"} and not os.path.exists(out):
        drives = read_json(os.path.join(os.path.dirname(P["cache"]), "drives_navdev.json"))
        res = decide(rl_methods, drives)
        write_json(out, {**res, "decided": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"})
        log(f"phase 5 decision: {json.dumps(res)}")


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
    write_json(os.path.join(P["heads"], "driver_status.json"), status)
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes; 0 = one pass")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        try:
            one_pass()
        except Exception as e:  # keep the loop alive; the log shows what failed
            log(f"pass failed: {type(e).__name__}: {e}")
        if args.once or not args.loop:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
