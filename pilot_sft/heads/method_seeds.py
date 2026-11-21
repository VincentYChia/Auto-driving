"""Method comparisons with every phase-3 seed (reporting; run on the desktop): python pilot_sft/heads/method_seeds.py

For each method's chosen configuration (choices/<M>.json), every seed run that has NAVSIM scores at seed 0's choices
(the run, <run>-s1, <run>-s2) gives a per-scene single path and best-of-6. Each pair of methods is compared by the
difference of their seed-averaged means, with R5's hierarchical bootstrap: each resample redraws drives (with all
their scenes) and each method's seeds. A method with one seed gets the larger between-seed SD of the methods with
three, as an added normal draw (R5). navtest is reporting only (R1); navdev is where choices are made. Prints JSON.
"""
import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

EXP = os.path.expanduser("~/navsim_workspace/exp")
H = os.path.join(EXP, "heads")
CACHE_ROOT = os.path.expanduser("~/navsim_workspace/head_cache")
METHODS = ("A", "B", "C", "D", "R")


def scores(name):
    f = sorted(glob.glob(os.path.join(EXP, name, "*", "*.csv")))
    if not f:
        return None
    d = pd.read_csv(f[-1])
    return d[d.token != "average"].set_index("token")["score"].astype(float)


def per_seed(lc, split):
    """{"single": [series per seed], "best_of_6": [...]} at seed 0's choices, for the seeds that are scored."""
    c = lc["choice"]
    k = 6 if c["scale"] is not None else 0
    names = ["single"] + [f"sample_{j}" if c["scale"] == 1.0 else f"scale{c['scale']:g}_sample_{j}" for j in range(k)]
    out = {"single": [], "best_of_6": [], "runs": []}
    for run in [lc["run"], f"{lc['run']}-s1", f"{lc['run']}-s2"]:
        fr = [scores(f"heads-{run}-{split}_epoch_{c['epoch']}-steps{c['steps']}-{n}_{split}_v1") for n in names]
        if any(x is None for x in fr):
            continue
        out["runs"].append(run)
        out["single"].append(fr[0])
        if k:
            out["best_of_6"].append(pd.concat(fr[1:], axis=1).max(axis=1))
    return out


def hier(a, b, drives, fill, n=10000, seed=0):
    """Mean difference a - b (lists of per-scene series, one per seed) and its hierarchical-bootstrap 95% interval."""
    toks = sorted(set.intersection(*[set(s.index) for s in a + b]))
    drv = np.array([drives[t] for t in toks])
    uniq, inv = np.unique(drv, return_inverse=True)
    A = np.stack([s.loc[toks].values for s in a])
    B = np.stack([s.loc[toks].values for s in b])
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        ma = A[rng.integers(0, len(A), len(A))].mean(0)
        mb = B[rng.integers(0, len(B), len(B))].mean(0)
        da = (ma * w).sum() / w.sum() + (rng.normal(0, fill) if len(A) == 1 else 0.0)
        db = (mb * w).sum() / w.sum() + (rng.normal(0, fill) if len(B) == 1 else 0.0)
        d[i] = da - db
    point = float(A.mean() - B.mean())
    return {"difference": round(100 * point, 2), "ci": [round(100 * float(np.percentile(d, 2.5)), 2),
                                                       round(100 * float(np.percentile(d, 97.5)), 2)],
            "seeds": [len(A), len(B)], "scenes": len(toks)}


out = {}
for split in ("navtest", "navdev"):
    drives = json.load(open(os.path.join(CACHE_ROOT, f"drives_{split}.json")))
    data = {}
    for m in METHODS:
        f = os.path.join(H, "choices", f"{m}.json")
        if os.path.exists(f):
            data[m] = per_seed(json.load(open(f)), split)
    res = {"methods": {}, "pairs": {}}
    for meas in ("single", "best_of_6"):
        sds = [float(np.std([s.mean() for s in data[m][meas]], ddof=1)) for m in data if len(data[m][meas]) >= 3]
        fill = max(sds) if sds else 0.0
        for m, v in data.items():
            if v[meas]:
                means = [round(100 * float(s.mean()), 2) for s in v[meas]]
                res["methods"].setdefault(m, {})[meas] = {"seeds": means, "mean": round(float(np.mean(means)), 2),
                                                          "sd": round(float(np.std(means, ddof=1)), 2) if len(means) > 1 else None}
        for m1, m2 in combinations([m for m in data if data[m][meas]], 2):
            res["pairs"].setdefault(f"{m1}-{m2}", {})[meas] = hier(data[m1][meas], data[m2][meas], drives, fill)
        res.setdefault("seed_sd_added_for_single_seed", {})[meas] = round(100 * fill, 2)
    out[split] = res
print(json.dumps(out))
