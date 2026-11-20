"""Run on the desktop: python pilot_sft/heads/seed_pairs.py > seed_pairs.json. Phase 3's between-seed differences, paired on the same scenes with the drive-level bootstrap: each seed run against
seed 0 of its configuration, single path and best-of-6 at seed 0's choices, on navdev and navtest. Prints JSON."""
import glob
import json
import os

import numpy as np
import pandas as pd

EXP = os.path.expanduser("~/navsim_workspace/exp")
H = os.path.join(EXP, "heads")
CACHE_ROOT = os.path.expanduser("~/navsim_workspace/head_cache")


def scores(name):
    f = sorted(glob.glob(os.path.join(EXP, name, "*", "*.csv")))
    if not f:
        return None
    d = pd.read_csv(f[-1])
    return d[d.token != "average"].set_index("token")["score"].astype(float)


def folder_scores(run, split, epoch, steps, names):
    out = []
    for n in names:
        s = scores(f"heads-{run}-{split}_epoch_{epoch}-steps{steps}-{n}_{split}_v1")
        if s is None:
            return None
        out.append(s)
    return out


def ci(x, groups, n=10000):
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    sm, c = np.bincount(inv, weights=x), np.bincount(inv)
    i = np.random.default_rng(0).integers(0, len(u), size=(n, len(u)))
    m = sm[i].sum(1) / c[i].sum(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


drives = {sp: json.load(open(os.path.join(CACHE_ROOT, f"drives_{sp}.json"))) for sp in ("navdev", "navtest")}
out = {}
for method in ("A", "B", "C", "D", "R"):
    f = os.path.join(H, "choices", f"{method}.json")
    if not os.path.exists(f):
        continue
    lc = json.load(open(f))
    c = lc["choice"]
    k = 6 if c["scale"] is not None else 0
    samples = [f"sample_{j}" if c["scale"] == 1.0 else f"scale{c['scale']:g}_sample_{j}" for j in range(k)]
    for s in (1, 2):
        run = f"{lc['run']}-s{s}"
        if not os.path.isdir(os.path.join(H, run)):
            continue
        res = {}
        for split in ("navdev", "navtest"):
            a = folder_scores(lc["run"], split, c["epoch"], c["steps"], ["single"] + samples)
            b = folder_scores(run, split, c["epoch"], c["steps"], ["single"] + samples)
            if a is None or b is None:
                continue
            toks = sorted(set(a[0].index) & set(b[0].index))
            g = [drives[split][t] for t in toks]
            d1 = b[0].loc[toks].values - a[0].loc[toks].values
            res[f"{split}_single"] = {"seed0": round(float(a[0].loc[toks].mean()), 4), f"seed{s}": round(float(b[0].loc[toks].mean()), 4),
                                      "difference": round(float(d1.mean()), 4), "ci_drives": ci(d1, g)}
            if k:
                ba = pd.concat([x.loc[toks] for x in a[1:]], axis=1).max(axis=1).values
                bb = pd.concat([x.loc[toks] for x in b[1:]], axis=1).max(axis=1).values
                res[f"{split}_best_of_6"] = {"seed0": round(float(ba.mean()), 4), f"seed{s}": round(float(bb.mean()), 4),
                                             "difference": round(float((bb - ba).mean()), 4), "ci_drives": ci(bb - ba, g)}
        if res:
            out[run] = res
print(json.dumps(out))
