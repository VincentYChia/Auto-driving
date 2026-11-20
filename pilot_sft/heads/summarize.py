"""Summarise a head run's NAVSIM scores: single path, best-of-K, diversity, and paired against the text pilot.

    python pilot_sft/heads/summarize.py --run <exp>/heads/<run> --exp <NAVSIM_EXP_ROOT> [--reference deskcheck-h1]

Reads the per-scene CSVs that ``scripts/desk/score_head_run.sh`` produced (one per path folder) and writes
``<run>/summary.json``. For each weights and step count:

- ``single``: PDMS with a 95% bootstrap CI over scenes, the sub-scores and the share of zero-score scenes;
- ``vs_reference``: the paired PDMS difference to the reference run (H1, the text pilot) on the same scenes;
- ``samples``: the mean PDMS of the K sampled paths, the oracle best-of-K (each scene's best path, the room a
  selector has) for K = 1 .. k, and diversity: the mean pairwise distance between the K end points and the mean
  pairwise ADE between the K paths, in metres.

Every CSV must hold all 12,146 navtest scenes, valid. Bootstrap: 5,000 resamples, seed 0 (as ``paired.py``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

COLS = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "comfort",
        "ego_progress", "driving_direction_compliance", "score"]
N_SCENES = 12146


def load(exp: str, name: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(exp, f"{name}_navtest_v1", "*", "*.csv")))
    if not files:
        raise FileNotFoundError(name)
    d = pd.read_csv(files[-1])
    d = d[d.token != "average"].set_index("token")
    if len(d) != N_SCENES or not d.valid.astype(bool).all():
        raise ValueError(f"{name}: {len(d)} scenes, {int((~d.valid.astype(bool)).sum())} invalid")
    return d[COLS].astype(float)


def ci(x: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def read_paths(folder: str):
    with open(os.path.join(folder, "shard_000_of_001.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return {r["token"]: np.asarray(r["waypoints"], dtype=np.float64) for r in rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--reference", default="deskcheck-h1", help="score name of the text pilot H1")
    args = p.parse_args()
    run = os.path.basename(os.path.normpath(args.run))
    ref = load(args.exp, args.reference)
    out = {"run": run, "reference": args.reference, "reference_pdms": round(float(ref.score.mean()), 4), "weights": {}}
    for wdir in sorted(glob.glob(os.path.join(args.run, "navtest_*"))):
        weights = os.path.basename(wdir)[len("navtest_"):]
        for sdir in sorted(glob.glob(os.path.join(wdir, "steps*"))):
            steps = os.path.basename(sdir)
            key = f"{run}-navtest_{weights}-{steps}"
            entry = {}
            try:
                single = load(args.exp, f"heads-{key}-single")
            except (FileNotFoundError, ValueError) as e:
                entry["error"] = str(e)
                out["weights"].setdefault(weights, {})[steps] = entry
                continue
            x = single.score.values
            ids = single.index
            d = x - ref.loc[ids, "score"].values
            entry["single"] = {"pdms": round(float(x.mean()), 4), "pdms_95ci": ci(x),
                               "metrics": {c: round(float(single[c].mean()), 4) for c in COLS[:-1]},
                               "share_zero": round(float((x == 0).mean()), 4)}
            entry["vs_reference"] = {"pdms_difference": round(float(d.mean()), 4), "ci": ci(d)}
            samples = sorted(glob.glob(os.path.join(sdir, "sample_*")), key=lambda s: int(s.rsplit("_", 1)[1]))
            scored = []
            for sp in samples:
                try:
                    scored.append(load(args.exp, f"heads-{key}-{os.path.basename(sp)}").loc[ids, "score"].values)
                except (FileNotFoundError, ValueError):
                    break
            if scored:
                m = np.stack(scored, 1)  # [scenes, k]
                entry["samples"] = {
                    "k": m.shape[1], "mean_pdms": round(float(m.mean()), 4),
                    "best_of_k": {str(k): round(float(m[:, :k].max(1).mean()), 4) for k in range(1, m.shape[1] + 1)},
                    "best_of_k_95ci": ci(m.max(1))}
                paths = [read_paths(sp) for sp in samples[: m.shape[1]]]
                end, ade = [], []
                for tok in ids:
                    P = [pp[tok] for pp in paths]
                    for a, b in combinations(range(len(P)), 2):
                        end.append(float(np.linalg.norm(P[a][-1] - P[b][-1])))
                        ade.append(float(np.linalg.norm(P[a] - P[b], axis=1).mean()))
                entry["samples"]["diversity"] = {"endpoint_m": round(float(np.mean(end)), 3) if end else 0.0,
                                                 "pairwise_ade_m": round(float(np.mean(ade)), 3) if ade else 0.0}
            timing = os.path.join(wdir, "timing.json")
            if os.path.exists(timing):
                with open(timing, encoding="utf-8") as f:
                    entry["timing"] = json.load(f).get(steps)
            out["weights"].setdefault(weights, {})[steps] = entry
    with open(os.path.join(args.run, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
