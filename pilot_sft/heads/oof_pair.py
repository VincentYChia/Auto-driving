"""Two selector chains' out-of-fold candidates on the same training scenes, paired. Reporting only; no choice reads it.

    python pilot_sft/heads/oof_pair.py <chain A> <chain B> [--sets oof_half0 oof_half1]     (on the desktop; prints JSON)

A chain's half generators (R14) each turn the other half's RL scenes into 7 scored candidates, the own pick first
(``selector-<chain>/candidates_<set>.jsonl``, ``score_candidates.py``'s output). No generator of the set trained on
those scenes, so two chains' candidates for the same scenes compare their configurations and RL recipes on far more
scenes than navdev's 743 (about 80,000 RL scenes from about 100 drives), though from the training drives'
distribution and with half generators rather than the deployed ones.

Per set and pooled, over the scenes both chains scored: A minus B for the own pick, the best candidate and the mean
candidate, with R3's drive-bootstrap 95% interval (10,000 resamples, seed 0; drives from the stored features' ids,
``drive_map.drive_of_id``). Unscorable candidates count as PDMS 0, as in ``candidate_headroom.py``. Points.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drive_map import drive_of_id  # noqa: E402

KEYS = ("own_pick", "best_of_k", "mean_candidate")


def per_scene(path: str) -> Dict[str, Tuple[float, float, float]]:
    """token -> (own pick, best candidate, mean candidate) PDMS of one scored candidate file."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = np.array([0.0 if "error" in x else float(x["score"]) for x in r["results"]])
            out[r["token"]] = (float(s[r["names"].index("single")]), float(s.max()), float(s.mean()))
    return out


def token_drives(meta: str) -> Dict[str, str]:
    """token -> drive, from a stored-feature cache's ``meta.jsonl`` (ids ``openscene/<log>/<token>``)."""
    out = {}
    with open(meta, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                i = json.loads(line)["id"]
                out[i.rsplit("/", 1)[-1]] = drive_of_id(i)
    return out


def paired(a: Dict[str, tuple], b: Dict[str, tuple], drives: Dict[str, str], n: int = 10000, seed: int = 0) -> dict:
    toks = sorted(set(a) & set(b) & set(drives))
    if not toks:
        return {"scenes": 0}
    g = np.array([drives[t] for t in toks])
    u, inv = np.unique(g, return_inverse=True)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    c = np.bincount(inv)
    res = {"scenes": len(toks), "drives": int(len(u))}
    for j, k in enumerate(KEYS):
        xa = np.array([a[t][j] for t in toks])
        xb = np.array([b[t][j] for t in toks])
        s = np.bincount(inv, weights=xa - xb)
        m = s[idx].sum(1) / c[idx].sum(1)
        res[k] = {"a": round(100 * float(xa.mean()), 2), "b": round(100 * float(xb.mean()), 2),
                  "a_minus_b": round(100 * float((xa - xb).mean()), 2),
                  "ci": [round(100 * float(np.percentile(m, 2.5)), 2), round(100 * float(np.percentile(m, 97.5)), 2)]}
    return res


def main(a: str, b: str, sets: List[str], heads: str, meta: str) -> dict:
    drives = token_drives(meta)
    out, pool_a, pool_b = {"a": a, "b": b, "sets": {}}, {}, {}
    for s in sets:
        fa, fb = (os.path.join(heads, f"selector-{x}", f"candidates_{s}.jsonl") for x in (a, b))
        if not (os.path.exists(fa + ".done") and os.path.exists(fb + ".done")):
            out["sets"][s] = None  # not completely scored for both chains yet
            continue
        pa, pb = per_scene(fa), per_scene(fb)
        out["sets"][s] = paired(pa, pb, drives)
        pool_a.update(pa)
        pool_b.update(pb)
    out["pooled"] = paired(pool_a, pool_b, drives) if pool_a else None
    out["note"] = "reporting only: out-of-fold training scenes, half generators; no choice reads it (R1, R14, R16)"
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--sets", nargs="+", default=["oof_half0", "oof_half1"])
    p.add_argument("--heads", default=os.path.expanduser("~/navsim_workspace/exp/heads"))
    p.add_argument("--meta", default=os.path.expanduser("~/navsim_workspace/head_cache/h1_l12-18-24/train/meta.jsonl"))
    x = p.parse_args()
    print(json.dumps(main(x.a, x.b, x.sets, x.heads, x.meta)))
