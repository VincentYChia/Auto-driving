"""Two selectors on the same scenes, paired (reporting): the mean difference of their selected paths' true PDMS.

    python pilot_sft/heads/selector_pair.py <selector A dir> <selector B dir> --drives <drives_<split>.json> [--split navdev]

Each directory holds ``apply_selector.py``'s ``<split>/selection.jsonl`` with ``true_chosen`` per scene (the selected
path's true PDMS). Prints A minus B over the scenes both scored, with R3's drive-bootstrap 95% interval (10,000
resamples, seed 0), and how often they pick differently. Points.
"""
import argparse
import json
import os

import numpy as np


def chosen(d: str, split: str) -> dict:
    out = {}
    with open(os.path.join(d, split, "selection.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "true_chosen" in r:
                    out[r["token"]] = (float(r["true_chosen"]), r.get("chosen"))
    return out


def pair(a: str, b: str, drives: dict, split: str = "navdev", n: int = 10000, seed: int = 0) -> dict:
    ca, cb = chosen(a, split), chosen(b, split)
    toks = sorted(set(ca) & set(cb))
    d = np.array([ca[t][0] - cb[t][0] for t in toks])
    g = np.array([drives[t] for t in toks])
    u, inv = np.unique(g, return_inverse=True)
    s, c = np.bincount(inv, weights=d), np.bincount(inv)
    i = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    m = s[i].sum(1) / c[i].sum(1)
    return {"a": os.path.basename(a.rstrip("/")), "b": os.path.basename(b.rstrip("/")), "split": split, "scenes": len(toks),
            "drives": int(len(u)), "a_pdms": round(100 * float(np.mean([ca[t][0] for t in toks])), 2),
            "b_pdms": round(100 * float(np.mean([cb[t][0] for t in toks])), 2),
            "a_minus_b": round(100 * float(d.mean()), 2),
            "ci": [round(100 * float(np.percentile(m, 2.5)), 2), round(100 * float(np.percentile(m, 97.5)), 2)],
            "different_picks": round(float(np.mean([ca[t][1] != cb[t][1] for t in toks])), 4)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--drives", required=True)
    p.add_argument("--split", default="navdev")
    x = p.parse_args()
    print(json.dumps(pair(x.a, x.b, json.load(open(x.drives)), x.split)))
