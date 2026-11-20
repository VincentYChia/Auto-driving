"""Collect every scored path-generator configuration into one aggregate JSON for reports (no per-scene data).

    python pilot_sft/heads/collect_results.py --out <json>

For every training run with navdev choices (``choices_navdev.json``), at those choices it reports:
- the single path and best-of-6, on navdev (where the choices were made) and on navtest (report only, R1);
- each with a 95% interval resampling drives (R3), and the paired difference to H1's text output on the same
  scenes, with a drive-level interval;
- plus the run's dev ADE at the chosen epoch, the method decisions so far, and the references.

Runs on the desktop with the score CSVs (pilot env).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

HOME = os.path.expanduser("~")
EXP = f"{HOME}/navsim_workspace/exp"
HEADS = f"{EXP}/heads"
REF = {"navdev": "h1-text_navdev_v1", "navtest": "deskcheck-h1_navtest_v1"}
CV = {"navdev": "cv_navdev_v1", "navtest": "cv_navtest_v1"}


def load(name: str) -> Optional[pd.Series]:
    f = sorted(glob.glob(f"{EXP}/{name}/*/*.csv"))
    if not f:
        return None
    d = pd.read_csv(f[-1])
    return d[d.token != "average"].set_index("token")["score"].astype(float)


def drive_ci(x: np.ndarray, groups: np.ndarray, n: int = 10000):
    u, inv = np.unique(groups, return_inverse=True)
    s, c = np.bincount(inv, weights=x), np.bincount(inv)
    i = np.random.default_rng(0).integers(0, len(u), size=(n, len(u)))
    m = s[i].sum(1) / c[i].sum(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def stats(s: pd.Series, ref: Optional[pd.Series], drives: dict) -> dict:
    g = np.array([drives[t] for t in s.index])
    out = {"pdms": round(float(s.mean()), 4), "ci_drives": drive_ci(s.values, g), "scenes": int(len(s)),
           "drives": int(len(set(g)))}
    if ref is not None:
        common = s.index.intersection(ref.index)
        d = s.loc[common].values - ref.loc[common].values
        out["vs_h1_text"] = {"difference": round(float(d.mean()), 4),
                             "ci_drives": drive_ci(d, np.array([drives[t] for t in common]))}
    return out


def folder_scores(run: str, split: str, weights: str, steps: int, name: str) -> Optional[pd.Series]:
    return load(f"heads-{run}-{split}_{weights}-steps{steps}-{name}_{split}_v1")


def best6(run: str, split: str, weights: str, steps: int, scale) -> Optional[pd.Series]:
    if scale is None:
        return None
    names = [f"sample_{j}" if scale == 1.0 else f"scale{scale:g}_sample_{j}" for j in range(6)]
    ss = [folder_scores(run, split, weights, steps, n) for n in names]
    if any(s is None for s in ss):
        return None
    return pd.concat(ss, axis=1).max(axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    drives = {sp: json.load(open(f"{HOME}/navsim_workspace/head_cache/drives_{sp}.json")) for sp in ("navdev", "navtest")}
    refs = {sp: load(REF[sp]) for sp in REF}
    out = {"references": {}, "runs": {}, "methods": {}}
    for sp in ("navdev", "navtest"):
        for label, name in (("h1_text", REF[sp]), ("constant_velocity", CV[sp])):
            s = load(name)
            if s is not None:
                out["references"][f"{label}_{sp}"] = stats(s, None, drives[sp])
    for cf in sorted(glob.glob(f"{HEADS}/*/choices_navdev.json")):
        run = os.path.basename(os.path.dirname(cf))
        c = json.load(open(cf))
        weights = c.get("checkpoint") or f"epoch_{c['epoch']}"
        rec = {"choice": {k: c.get(k) for k in ("epoch", "checkpoint", "steps", "scale")}}
        for sp in ("navdev", "navtest"):
            s = folder_scores(run, sp, weights, c["steps"], "single")
            if s is not None:
                rec[f"{sp}_single"] = stats(s, refs[sp], drives[sp])
            b = best6(run, sp, weights, c["steps"], c.get("scale"))
            if b is not None:
                rec[f"{sp}_best_of_6"] = stats(b, refs[sp], drives[sp])
        logf = f"{HEADS}/{run}/log.jsonl"
        if os.path.exists(logf) and c.get("epoch"):
            for line in open(logf):
                r = json.loads(line)
                if r.get("epoch") == c["epoch"] and "dev_ade" in r:
                    rec["dev_ade"], rec["dev_best6_ade"] = r["dev_ade"], r.get("dev_best6_ade")
        rj = f"{HEADS}/{run}/run.json"
        if os.path.exists(rj):
            j = json.load(open(rj))
            rec["kind"] = (j.get("config") or {}).get("kind")
            rec["layers"] = (j.get("config") or {}).get("layers")
            rec["params"], rec["train_seconds"] = j.get("params"), j.get("elapsed_s")
        out["runs"][run] = rec
    for mf in sorted(glob.glob(f"{HEADS}/choices/*.json")):
        out["methods"][os.path.basename(mf)[:-5]] = json.load(open(mf))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{len(out['runs'])} runs, {len(out['methods'])} method files -> {args.out}")


if __name__ == "__main__":
    main()
