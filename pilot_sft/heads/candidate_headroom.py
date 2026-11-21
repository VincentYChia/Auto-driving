"""Headroom of scored candidate sets (reporting): how far a generator's own pick is from the best of its candidates.

    python pilot_sft/heads/candidate_headroom.py <scores.jsonl> [<scores.jsonl> ...]

Each file is ``score_candidates.py``'s output (one line per scene: ``names`` with the own pick ``single`` first, and
``results``). Per file: the scenes, the mean own-pick PDMS, the mean best candidate (the oracle a perfect selector
reaches), the mean candidate, the share of scenes where some candidate beats the own pick, and the own pick's NC or
DAC failures (PDMS zeroed) with the share where some candidate passes both. Unscorable candidates count as PDMS 0 and
as failing. Prints JSON.
"""
import json
import sys

import numpy as np


def fails(x: dict) -> bool:
    return "error" in x or x.get("no_at_fault_collisions", 0.0) == 0.0 or x.get("drivable_area_compliance", 0.0) == 0.0


def headroom(path: str) -> dict:
    own, best, mean, beat, fail, rescued, k = [], [], [], 0, 0, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            res, i0 = r["results"], r["names"].index("single")
            s = np.array([0.0 if "error" in x else float(x["score"]) for x in res])
            own.append(s[i0])
            best.append(s.max())
            mean.append(s.mean())
            k = max(k, len(res))
            beat += int(s.max() > s[i0] + 1e-9)
            if fails(res[i0]):
                fail += 1
                rescued += int(any(not fails(x) for x in res))
    n = len(own)
    return {"scenes": n, "candidates": k, "own_pick_pdms": round(100 * float(np.mean(own)), 2),
            "best_of_k_pdms": round(100 * float(np.mean(best)), 2),
            "mean_candidate_pdms": round(100 * float(np.mean(mean)), 2),
            "share_some_candidate_beats_own": round(beat / max(n, 1), 4),
            "own_pick_fails_nc_or_dac": fail, "share_of_failures_some_candidate_passes": round(rescued / max(fail, 1), 4)}


if __name__ == "__main__":
    print(json.dumps({p: headroom(p) for p in sys.argv[1:]}))
