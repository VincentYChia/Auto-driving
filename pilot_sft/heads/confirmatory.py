"""R4's confirmatory tests on navtest, run once at the end (docs/HEADS_PLAN.md R4 and its implementation note).

    python pilot_sft/heads/confirmatory.py        (on the desktop, with the score CSVs; prints JSON)

All four use navtest's single-path PDMS and R3's hierarchical bootstrap (10,000 resamples, seed 0: drives with all
their scenes, and each method's seeds; a one-seed path generator gets the larger of A's and D's between-seed SDs as an
added normal draw). "After the RL stage" is phase 4 under the registered recipe (choices/rl_recipe.json, RL4), each RL
seed at its R11 checkpoint. "The navdev-chosen method" is R5's leader under that recipe (choices/decision_phase5.json).
- C1 (primary): A minus D after RL.
- C2: the chosen method after RL minus H1's text output (one fixed model: no seed draw for it).
- C3: the chosen method before RL (phase 3, R10's choices) minus R.
- C4: the chosen method after RL minus before RL, each RL seed paired with the supervised seed it started from.
Each p-value is two-sided, twice the smaller bootstrap tail at 0 (resolution 1e-4). Holm's step-down procedure keeps
the family-wise error at 0.05. Nothing runs until every input exists.
"""
import json
import os
from typing import Dict, Optional

import numpy as np

import method_seeds as ms

ALPHA = 0.05
H1_NAVTEST = "deskcheck-h1_navtest_v1"


def p_two_sided(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(ps: Dict[str, float], alpha: float = ALPHA) -> Dict[str, dict]:
    """Holm's step-down: the i-th smallest p (from 0) is compared with alpha / (m - i); testing stops at the first
    that fails. Adjusted p-values are the running maximum of min(1, (m - i) p)."""
    m, out, adj, going = len(ps), {}, 0.0, True
    for i, (k, p) in enumerate(sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))):
        adj = max(adj, min(1.0, (m - i) * p))
        going = going and p <= alpha / (m - i)
        out[k] = {"p": round(p, 4), "p_holm": round(adj, 4), "threshold": round(alpha / (m - i), 4), "rejected": going}
    return out


def main(exp: str = ms.EXP, cache_root: str = ms.CACHE_ROOT) -> Optional[dict]:
    heads = os.path.join(exp, "heads")
    ch = os.path.join(heads, "choices")
    need = [os.path.join(ch, f) for f in ("rl_recipe.json", "decision_phase5.json")]
    if not all(os.path.exists(f) for f in need):
        return None
    base = json.load(open(need[0]))["recipe"]
    leader = json.load(open(need[1]))["leader"]
    drives = json.load(open(os.path.join(cache_root, "drives_navtest.json")))
    lcs = {m: json.load(open(os.path.join(ch, f"{m}.json"))) for m in ("A", "B", "C", "D", "R")
           if os.path.exists(os.path.join(ch, f"{m}.json"))}
    after = {m: ms.rl_seeds(exp, heads, lcs[m], m, base, base, "navtest") for m in ("A", "D", leader) if m in lcs}
    before = {m: ms.per_seed(exp, lcs[m], "navtest") for m in (leader, "R") if m in lcs}
    h1 = ms.scores(exp, H1_NAVTEST)
    if h1 is None or any(after.get(m) is None for m in ("A", "D", leader)) or not all(before.get(m) for m in (leader, "R")):
        return None
    sd = lambda ss: float(np.std([s.mean() for s in ss], ddof=1)) if len(ss) > 1 else 0.0  # noqa: E731
    fill_after = max(sd(after["A"]["single"]), sd(after["D"]["single"]))
    fill_before = max(sd(ms.per_seed(exp, lcs[m], "navtest")["single"]) for m in ("A", "D"))
    tests, draws = {}, {}
    tests["C1"], draws["C1"] = ms.hier(after["A"]["single"], after["D"]["single"], drives, fill_after, draws=True)
    tests["C2"], draws["C2"] = ms.hier(after[leader]["single"], [h1], drives, fill_after, fill_b=0.0, draws=True)
    tests["C3"], draws["C3"] = ms.hier(before[leader]["single"], before["R"]["single"], drives, fill_before, draws=True)
    b = before[leader]["single"][:len(after[leader]["single"])]
    tests["C4"], draws["C4"] = ms.hier_paired(after[leader]["single"], b, drives, draws=True)
    what = {"C1": "A minus D after RL (primary)", "C2": f"{leader} after RL minus H1's text output",
            "C3": f"{leader} before RL minus R: does generation help?", "C4": f"{leader} after minus before RL: does RL help?"}
    h = holm({k: p_two_sided(d) for k, d in draws.items()})
    return {"tests": {k: {"what": what[k], **tests[k], **h[k]} for k in tests}, "chosen_method": leader,
            "recipe": f"RL{base}", "measure": "navtest single-path PDMS, points", "alpha_family": ALPHA,
            "seed_sd_added": {"after_rl": round(100 * fill_after, 2), "before_rl": round(100 * fill_before, 2)},
            "rule": "docs/HEADS_PLAN.md R4 (and its implementation note), R3"}


if __name__ == "__main__":
    print(json.dumps(main()))
