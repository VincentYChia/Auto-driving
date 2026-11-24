"""How optimistic is a navdev-chosen checkpoint's own-pick score? Exploratory and reporting only; no choice reads it.

    python pilot_sft/heads/choice_optimism.py <rl run> [<rl run> ...]     (on the desktop; prints JSON)

R11 deploys the checkpoint (update 0, the start, or a save) with the highest navdev single-path PDMS, and R14 then
compares the selector with that checkpoint's own pick on the same navdev scenes. The own pick's navdev score therefore
includes the luck of being the maximum. This estimates that luck by cross-fitting over navdev's drives: split the
drives in two, choose the checkpoint on one fold by R11's rule (the highest mean, a tie to the earlier update), score it
on the other fold, both ways, over many random splits. The in-sample score minus the cross-fitted one is the optimism.

It reads the per-scene score files the driver already made (``load_scores``) and scores nothing new.
"""
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cross_fit(scores: Dict[int, pd.Series], drives: Dict[str, str], n_splits: int = 500, seed: int = 0) -> dict:
    """``scores``: update -> per-scene single-path PDMS (update 0 is the start). Returns the in-sample choice and
    score, the cross-fitted score and their difference, in points."""
    ups = sorted(scores)
    toks = sorted(set.intersection(*[set(s.index) for s in scores.values()]))
    m = np.stack([scores[u].loc[toks].to_numpy(dtype=float) for u in ups])  # [checkpoint, scene]
    drv = np.array([drives[t] for t in toks])
    uniq = np.unique(drv)
    best = int(np.argmax(m.mean(1)))  # np.argmax takes the first maximum: the earlier update, as R11
    rng = np.random.default_rng(seed)
    held, picks = [], []
    for _ in range(n_splits):
        fold = np.isin(drv, rng.permutation(uniq)[: len(uniq) // 2])
        for a, b in ((fold, ~fold), (~fold, fold)):
            i = int(np.argmax(m[:, a].mean(1)))
            picks.append(ups[i])
            held.append(float(m[i, b].mean()))
    cf = float(np.mean(held))
    return {"in_sample_choice": ups[best], "in_sample": round(100 * float(m[best].mean()), 2),
            "cross_fitted": round(100 * cf, 2), "optimism": round(100 * (float(m[best].mean()) - cf), 2),
            "choice_agreement": round(float(np.mean([p == ups[best] for p in picks])), 3),
            "by_update": {int(u): round(100 * float(m[k].mean()), 2) for k, u in enumerate(ups)},
            "scenes": len(toks), "drives": len(uniq), "splits": n_splits}


def run_scores(rl: str) -> Optional[Dict[int, pd.Series]]:
    """Per-scene navdev single-path scores of an RL run at update 0 and each saved checkpoint, from its choice file."""
    import driver
    rd = os.path.join(driver.P["heads"], rl)
    cf = os.path.join(rd, "choices_navdev.json")
    if not os.path.exists(cf):
        return None
    ch = driver.read_json(cf)
    ups = sorted(int(u) for u in ch["single_by_checkpoint"])
    out = {}
    for u in ups:
        wdir = (os.path.join(driver.P["heads"], ch["start"]["run"], f"navdev_{ch['start']['weights']}") if u == 0
                else os.path.join(rd, f"navdev_rl_{u}"))
        folder = os.path.join(wdir, "steps%s" % ch["steps"], "single")
        s = driver.load_scores(driver.score_name(folder) + "_navdev_v1")
        if s is None:
            return None
        out[u] = s
    return out


def main(runs: List[str]) -> dict:
    import driver
    drives = driver.read_json(os.path.join(os.path.dirname(driver.P["cache"]), "drives_navdev.json"))
    res = {}
    for rl in runs:
        sc = run_scores(rl)
        res[rl] = cross_fit(sc, drives) if sc else None
    return {"runs": res, "note": "exploratory; reporting only, no choice reads it (R1, R11, R14 unchanged)"}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:])))
