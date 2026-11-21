"""Run on the desktop: python pilot_sft/heads/dac_headroom.py > dac_headroom.json. Selector headroom on drivable-area failures (navtest, reporting only): for each path generator at its choices
(layers 12/18/24, seed 0), by route command, the share of scenes whose own pick fails DAC, the share of those where
one of its 6 samples passes DAC, and the own pick, best-of-7 and mean candidate PDMS. Prints one JSON object."""
import glob
import json
import os

import numpy as np
import pandas as pd

EXP = os.path.expanduser("~/navsim_workspace/exp")
H = os.path.join(EXP, "heads")
CACHE = os.path.expanduser("~/navsim_workspace/head_cache/h1_l12-18-24/navtest")
COMMANDS = ("turn left", "go straight", "turn right", "none given")


def frame(name):
    f = sorted(glob.glob(os.path.join(EXP, name, "*", "*.csv")))
    d = pd.read_csv(f[-1])
    return d[d.token != "average"].set_index("token")


cmd = {}
with open(os.path.join(CACHE, "meta.jsonl")) as f:
    for line in f:
        m = json.loads(line)
        cmd[m["id"].rsplit("/", 1)[-1]] = COMMANDS[int(np.argmax(m["ego"][-4:]))]
out = {}
for run in ("a1-l12-18-24", "b-l12-18-24", "c-l12-18-24", "d-l12-18-24"):
    c = json.load(open(os.path.join(H, run, "choices_navdev.json")))
    base = f"heads-{run}-navtest_epoch_{c['epoch']}-steps{c['steps']}"
    names = ["single"] + [f"sample_{j}" if c["scale"] == 1.0 else f"scale{c['scale']:g}_sample_{j}" for j in range(6)]
    fr = [frame(f"{base}-{n}_navtest_v1") for n in names]
    toks = sorted(set.intersection(*[set(x.index) for x in fr]))
    pd_ = np.stack([x.loc[toks, "score"].values for x in fr], 1)
    dac = np.stack([x.loc[toks, "drivable_area_compliance"].values for x in fr], 1)
    cm = np.array([cmd[t] for t in toks])
    res = {}
    for k in ("go straight", "turn left", "turn right", "all"):
        m = np.ones(len(toks), bool) if k == "all" else cm == k
        own_fail = dac[m, 0] == 0
        rescue = (dac[m, 1:] > 0).any(1) & own_fail
        res[k] = {"scenes": int(m.sum()), "own_dac_fail_pct": round(100 * float(own_fail.mean()), 2),
                  "rescuable_pct_of_fails": round(100 * float(rescue.sum() / max(own_fail.sum(), 1)), 1),
                  "own_pdms": round(100 * float(pd_[m, 0].mean()), 2), "best_of_7_pdms": round(100 * float(pd_[m].max(1).mean()), 2),
                  "mean_candidate_pdms": round(100 * float(pd_[m].mean()), 2)}
    out[run] = res
print(json.dumps(out))
