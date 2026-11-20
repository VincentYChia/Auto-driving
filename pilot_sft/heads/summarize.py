"""Summarise a training run's NAVSIM scores on one split: single path, best-of-K per sampling setting, diversity,
and the paired difference to the text pilot H1, with intervals that respect drives (docs/HEADS_PLAN.md R3).

    python pilot_sft/heads/summarize.py --run <exp>/heads/<run> --exp <NAVSIM_EXP_ROOT> --split navdev

Reads the per-scene CSVs that ``scripts/desk/score_head_run.sh`` produced, one per path folder, named
``heads-<run>-<split>_<weights>-steps<S>-<folder>_<split>_v1``. Writes ``<run>/summary_<split>.json``. For each
weights and step count:

- ``single``: PDMS, its sub-scores, the share of zero-score scenes; a 95% interval resampling **drives** (all of a
  drive's scenes together; 10,000 resamples, seed 0) and, for comparison with earlier reports, one resampling
  scenes (5,000 resamples, seed 0, as ``paired.py``).
- ``vs_reference``: the paired PDMS difference to the reference (H1's text output on the same scenes), with the
  same two intervals.
- ``samples[<scale>]``: for each sampling setting, the mean PDMS of the K sampled paths, the oracle best-of-K for
  K = 1..k (each scene's best path), and diversity: the mean pairwise distance between the K end points and the
  mean pairwise ADE, in metres.

Every CSV must hold the same scenes, all valid (12,146 for navtest). Drives come from ``--drives`` (token ->
drive JSON from ``drive_map.py``); without it only scene-level intervals are given.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from itertools import combinations
from typing import Dict, Optional

import numpy as np
import pandas as pd

COLS = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "comfort",
        "ego_progress", "driving_direction_compliance", "score"]
REFERENCE = {"navtest": "deskcheck-h1", "navdev": "h1-text"}
SAMPLE = re.compile(r"^(?:scale([0-9.]+)_)?sample_(\d+)$")


def load(exp: str, name: str, split: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(exp, f"{name}_{split}_v1", "*", "*.csv")))
    if not files:
        raise FileNotFoundError(name)
    d = pd.read_csv(files[-1])
    d = d[d.token != "average"].set_index("token")
    if not d.valid.astype(bool).all():
        raise ValueError(f"{name}: {int((~d.valid.astype(bool)).sum())} invalid scenes")
    if split == "navtest" and len(d) != 12146:
        raise ValueError(f"{name}: {len(d)} scenes, navtest has 12,146")
    return d[COLS].astype(float)


def scene_ci(x: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def drive_ci(x: np.ndarray, drives: Optional[np.ndarray], n: int = 10000, seed: int = 0):
    """95% percentile interval of the mean, resampling drives with replacement (each with all its scenes)."""
    if drives is None:
        return None
    uniq, inv = np.unique(drives, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=x), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n, len(uniq)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def read_paths(folder: str) -> Dict[str, np.ndarray]:
    with open(os.path.join(folder, "shard_000_of_001.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return {r["token"]: np.asarray(r["waypoints"], dtype=np.float64) for r in rows}


def diversity(paths, ids):
    end, ade = [], []
    for tok in ids:
        P = [pp[tok] for pp in paths]
        for a, b in combinations(range(len(P)), 2):
            end.append(float(np.linalg.norm(P[a][-1] - P[b][-1])))
            ade.append(float(np.linalg.norm(P[a] - P[b], axis=1).mean()))
    return {"endpoint_m": round(float(np.mean(end)), 3) if end else 0.0,
            "pairwise_ade_m": round(float(np.mean(ade)), 3) if ade else 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--split", default="navtest", choices=("navtest", "navdev"))
    p.add_argument("--reference", default=None, help="score name of H1's text output on this split")
    p.add_argument("--drives", default=None, help="token -> drive JSON (default: <cache root>/drives_<split>.json)")
    args = p.parse_args()
    run = os.path.basename(os.path.normpath(args.run))
    ref_name = args.reference or REFERENCE[args.split]
    drives_path = args.drives or os.path.expanduser(f"~/navsim_workspace/head_cache/drives_{args.split}.json")
    drive_of = json.load(open(drives_path)) if os.path.exists(drives_path) else None
    try:
        ref = load(args.exp, ref_name, args.split)
    except (FileNotFoundError, ValueError):
        ref = None
    out = {"run": run, "split": args.split, "reference": ref_name if ref is not None else None,
           "reference_pdms": round(float(ref.score.mean()), 4) if ref is not None else None,
           "drives_file": drives_path if drive_of else None, "weights": {}}
    for wdir in sorted(glob.glob(os.path.join(args.run, f"{args.split}_*"))):
        weights = os.path.basename(wdir)[len(args.split) + 1:]
        for sdir in sorted(glob.glob(os.path.join(wdir, "steps*"))):
            steps = os.path.basename(sdir)
            key = f"{run}-{args.split}_{weights}-{steps}"
            entry = {}
            try:
                single = load(args.exp, f"heads-{key}-single", args.split)
            except (FileNotFoundError, ValueError) as e:
                entry["error"] = str(e)
                out["weights"].setdefault(weights, {})[steps] = entry
                continue
            ids = single.index
            x = single.score.values
            drives = np.array([drive_of.get(t, "unknown") for t in ids]) if drive_of else None
            entry["scenes"] = int(len(ids))
            entry["drives"] = int(len(set(drives))) if drives is not None else None
            entry["single"] = {"pdms": round(float(x.mean()), 4), "pdms_95ci_drives": drive_ci(x, drives),
                               "pdms_95ci_scenes": scene_ci(x),
                               "metrics": {c: round(float(single[c].mean()), 4) for c in COLS[:-1]},
                               "share_zero": round(float((x == 0).mean()), 4)}
            if ref is not None:
                common = ids.intersection(ref.index)
                d = single.loc[common, "score"].values - ref.loc[common, "score"].values
                dd = np.array([drive_of.get(t, "unknown") for t in common]) if drive_of else None
                entry["vs_reference"] = {"pdms_difference": round(float(d.mean()), 4), "ci_drives": drive_ci(d, dd),
                                         "ci_scenes": scene_ci(d), "scenes": int(len(common))}
            groups: Dict[float, Dict[int, str]] = {}
            for sp in glob.glob(os.path.join(sdir, "*sample_*")):
                m = SAMPLE.match(os.path.basename(sp))
                if m:
                    groups.setdefault(float(m.group(1) or 1.0), {})[int(m.group(2))] = sp
            samples = {}
            for scale, by_j in sorted(groups.items()):
                scored, folders = [], []
                for j in sorted(by_j):
                    try:
                        scored.append(load(args.exp, f"heads-{key}-{os.path.basename(by_j[j])}", args.split)
                                      .loc[ids, "score"].values)
                        folders.append(by_j[j])
                    except (FileNotFoundError, ValueError, KeyError):
                        break
                if not scored:
                    continue
                m = np.stack(scored, 1)  # [scenes, k]
                best = m.max(1)
                samples[f"{scale:g}"] = {
                    "k": m.shape[1], "mean_pdms": round(float(m.mean()), 4),
                    "best_of_k": {str(k): round(float(m[:, :k].max(1).mean()), 4) for k in range(1, m.shape[1] + 1)},
                    "best_of_k_95ci_drives": drive_ci(best, drives), "best_of_k_95ci_scenes": scene_ci(best),
                    "diversity": diversity([read_paths(f) for f in folders], ids)}
            if samples:
                entry["samples"] = samples
            timing = os.path.join(wdir, "timing.json")
            if os.path.exists(timing):
                with open(timing, encoding="utf-8") as f:
                    entry["timing"] = json.load(f).get(steps)
            out["weights"].setdefault(weights, {})[steps] = entry
    with open(os.path.join(args.run, f"summary_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
