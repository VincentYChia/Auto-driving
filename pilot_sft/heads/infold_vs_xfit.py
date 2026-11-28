"""In-fold against cross-fitted selector, per contender, on navdev. Exploratory, post hoc and reporting only.

    python pilot_sft/heads/infold_vs_xfit.py --heads <heads dir> --drives <cache root>/drives_navdev.json --out <json>

R14 adopts a contender's selector by the cross-fitted seed 0 (``selector-<x>-xfit-s0``), trained on candidates from
two half generators that never saw the scenes. Its in-fold control (``selector-<x>-infold-s0``) is trained on the
deployed generator's own candidates. Both pick among the same navdev candidates of the deployed generator, so their
picks compare scene by scene. After RL the in-fold control did at least as well as the cross-fitted one for every
contender, by point estimate (STATUS, 2026-11-27 23:49), and this measures it:
- per contender: in fold minus cross-fitted, the mean of the per-scene difference of the true PDMS of the two picks,
  with R3's drive interval (10,000 resamples, seed 0), and how often the two pick the same candidate;
- pooled over the contenders with RL: the per-scene mean of their differences, with the same interval.
The pattern was seen before this was written, so it is post hoc. It reads the selection files only and scores nothing.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driver import drive_interval  # noqa: E402

CONTENDERS = ("A4", "A6", "B4", "B6", "C4", "C6", "D4", "D6")
WITHOUT_RL = ("D4",)  # R19 deployed D4 at its supervised start: the one contender without RL


def picks(path: str) -> Dict[str, dict]:
    """token -> {chosen, true}: the navdev scenes NAVSIM scores (rows without ``true_chosen`` are outside its filter)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "true_chosen" in r:
                    out[r["token"]] = {"chosen": r.get("chosen"), "true": float(r["true_chosen"])}
    return out


def compare(heads: str, drives: Dict[str, str], contenders=CONTENDERS) -> dict:
    per, diffs = {}, {}
    for x in contenders:
        f = {k: os.path.join(heads, f"selector-{x.lower()}-{k}", "navdev", "selection.jsonl") for k in ("infold-s0", "xfit-s0")}
        if not all(os.path.exists(p) for p in f.values()):
            continue
        a, b = picks(f["infold-s0"]), picks(f["xfit-s0"])
        toks = sorted(set(a) & set(b))
        d = np.array([a[t]["true"] - b[t]["true"] for t in toks])
        diffs[x] = dict(zip(toks, d))
        per[x] = {"in_fold_minus_cross_fitted": round(100 * float(d.mean()), 2),
                  "ci_drives": [round(100 * v, 2) for v in drive_interval(d, [drives[t] for t in toks])],
                  "same_pick": round(float(np.mean([a[t]["chosen"] == b[t]["chosen"] for t in toks])), 4),
                  "in_fold": round(100 * float(np.mean([a[t]["true"] for t in toks])), 2),
                  "cross_fitted": round(100 * float(np.mean([b[t]["true"] for t in toks])), 2),
                  "rl": x not in WITHOUT_RL, "scenes": len(toks)}
    out = {"contenders": per, "unit": "PDMS points; intervals resample drives (R3)",
           "note": "exploratory and post hoc (the pattern was seen first); R14's adoption is unchanged"}
    rl = [x for x in diffs if x not in WITHOUT_RL]
    if rl:
        toks = sorted(set.intersection(*[set(diffs[x]) for x in rl]))
        m = np.array([np.mean([diffs[x][t] for x in rl]) for t in toks])
        out["pooled_with_rl"] = {"contenders": rl, "in_fold_minus_cross_fitted": round(100 * float(m.mean()), 2),
                                 "ci_drives": [round(100 * v, 2) for v in drive_interval(m, [drives[t] for t in toks])],
                                 "scenes": len(toks)}
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--heads", default=os.path.expanduser("~/navsim_workspace/exp/heads"))
    p.add_argument("--drives", required=True, help="drives_navdev.json")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res = compare(a.heads, json.load(open(a.drives)))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
