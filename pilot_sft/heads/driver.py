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
