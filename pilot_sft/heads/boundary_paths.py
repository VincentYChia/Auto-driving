"""Boundary paths for the selector's training (docs/HEADS_PLAN.md R14, an ablation): perturbations of the logged path
on both sides of the drivable-area (DAC) and collision (NC) limits, as in Versatile Samples (arXiv 2609.01799).

    python pilot_sft/heads/boundary_paths.py make --cache <cache>/train --only-tokens <rl tokens> --out <dir>/perturbed
    python pilot_sft/heads/score_candidates.py --paths <dir>/perturbed --out <dir>/perturbed_scores.jsonl
    python pilot_sft/heads/boundary_paths.py pick --paths <dir>/perturbed --scores <dir>/perturbed_scores.jsonl \\
        --out <dir>/boundary

``make`` writes, in ``predict_head.py``'s folder format, 16 perturbations of each scene's logged path (stored-feature
targets): lateral offsets that grow linearly from 0 to ``d`` at 4 s, along each waypoint's left normal, for ``d`` in
``LATERAL`` (m; left positive), and longitudinal changes that move each waypoint to ``f`` times its distance along the
path, for ``f`` in ``LONGITUDINAL``. They are generator-independent, so they need no cross-fitting.

``pick`` locates each scene's limits from the scores: on each lateral side and each longitudinal direction, the
perturbation nearest the logged path at which NC or DAC fails, and the one just inside it, which passes. It keeps the
pair of the lateral side whose failure comes first and the pair of the longitudinal direction whose failure comes
first (a tie goes to the left, or the slower, side), so 4 paths per scene, two on each side of a limit. When even the
smallest perturbation fails, the pair is the two smallest; when none fails on either side, the largest on each side. It writes ``boundary_0`` ... ``boundary_3`` with a ``score_candidates.py``-
format score file (``scores.jsonl``), for ``train_selector.py --extra-candidates``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LATERAL = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)  # m at 4 s; left positive
LONGITUDINAL = (0.6, 0.8, 1.2, 1.4)  # times the distance along the logged path


def lat_name(d: float) -> str:
    return f"lat_{'p' if d > 0 else 'm'}{abs(d):g}"


def lon_name(f: float) -> str:
    return f"lon_{f:g}"


NAMES = [lat_name(d) for d in LATERAL] + [lon_name(f) for f in LONGITUDINAL]


def directions(xy: np.ndarray) -> np.ndarray:
    """Unit direction of travel at each waypoint of ``[8, 2]`` (from the origin): the step's direction, carried over
    steps under 0.1 m (forward, +x, before the first step that moves)."""
    prev = np.vstack([np.zeros((1, 2)), xy[:-1]])
    d = xy - prev
    out, last = np.zeros_like(xy), np.array([1.0, 0.0])
    for i in range(len(xy)):
        n = np.linalg.norm(d[i])
        if n >= 0.1:
            last = d[i] / n
        out[i] = last
    return out


def lateral(xy: np.ndarray, d: float) -> np.ndarray:
    """Shift each waypoint along its left normal by ``d`` x (i + 1) / 8."""
    u = directions(xy)
    left = np.stack([-u[:, 1], u[:, 0]], axis=1)
    return xy + left * (d * np.arange(1, len(xy) + 1) / len(xy))[:, None]


def longitudinal(xy: np.ndarray, f: float) -> np.ndarray:
    """Move each waypoint to ``f`` times its distance along the path (the polyline from the origin), extended
    straight along the last direction beyond its end."""
    pts = np.vstack([np.zeros((1, 2)), xy])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # arc length at the origin and each waypoint
    out = np.empty_like(xy)
    last_dir = directions(xy)[-1]
    for i, target in enumerate(f * s[1:]):
        if target >= s[-1]:
            out[i] = pts[-1] + last_dir * (target - s[-1])
            continue
        j = int(np.searchsorted(s, target, side="right")) - 1
        j = min(max(j, 0), len(seg) - 1)
        t = 0.0 if seg[j] < 1e-9 else (target - s[j]) / seg[j]
        out[i] = pts[j] + t * (pts[j + 1] - pts[j])
    return out


def perturbations(xy: np.ndarray) -> Dict[str, np.ndarray]:
    out = {lat_name(d): lateral(xy, d) for d in LATERAL}
    out.update({lon_name(f): longitudinal(xy, f) for f in LONGITUDINAL})
    return out


def fails(result: dict) -> bool:
    """NC or DAC failed (or NAVSIM could not score the path)."""
    if "error" in result:
        return True
    return result["no_at_fault_collisions"] == 0 or result["drivable_area_compliance"] == 0


def boundary_pair(order: List[str], res: Dict[str, dict]) -> Tuple[Optional[int], List[str]]:
    """Walking ``order`` outward from the logged path: (index of the first failing perturbation or None, the pair
    kept): the last passing one and the first failing one, or the two outermost when none fails."""
    for i, n in enumerate(order):
        if fails(res[n]):
            return i, ([order[i - 1], n] if i > 0 else [n, order[i + 1] if i + 1 < len(order) else n])
    return None, order[-2:]


def pick(res: Dict[str, dict]) -> List[str]:
    """The 4 boundary paths of one scene from its perturbations' results (``NAMES`` -> result)."""
    left = [lat_name(d) for d in sorted(x for x in LATERAL if x > 0)]
    right = [lat_name(d) for d in sorted((x for x in LATERAL if x < 0), reverse=True)]
    slower = [lon_name(f) for f in sorted((x for x in LONGITUDINAL if x < 1), reverse=True)]
    faster = [lon_name(f) for f in sorted(x for x in LONGITUDINAL if x > 1)]
    chosen = []
    for a, b in ((left, right), (slower, faster)):
        ia, pa = boundary_pair(a, res)
        ib, pb = boundary_pair(b, res)
        if ia is None and ib is None:  # no limit found: the outermost on each side
            chosen += [a[-1], b[-1]]
        elif ib is None or (ia is not None and ia <= ib):  # a tie goes to the left / slower side
            chosen += pa
        else:
            chosen += pb
    return chosen


def write_folders(out: str, names: List[str], rows: Dict[str, List[np.ndarray]]) -> None:
    files = {}
    for n in names:
        os.makedirs(os.path.join(out, n), exist_ok=True)
        files[n] = open(os.path.join(out, n, "shard_000_of_001.jsonl"), "w", encoding="utf-8")
    for tok, paths in rows.items():
        for n, p in zip(names, paths):
            files[n].write(json.dumps({"token": tok, "waypoints": np.round(p, 3).tolist()}) + "\n")
    for f in files.values():
        f.close()


def make(args) -> None:
    import features
    from predict_head import bare

    store = features.Store(args.cache, allow_incomplete=False)
    if not store.has_targets:
        raise SystemExit(f"{args.cache} has no logged paths (targets)")
    keep = None
    if args.only_tokens:
        with open(args.only_tokens, encoding="utf-8") as f:
            keep = set(f.read().split())
    rows = {}
    for r, i in enumerate(store.ids):
        t = bare(i)
        if keep is None or t in keep:
            p = perturbations(store.target_xy[r].astype(np.float64))
            rows[t] = [p[n] for n in NAMES]
    write_folders(args.out, NAMES, rows)
    with open(os.path.join(args.out, "perturbations.json"), "w", encoding="utf-8") as f:
        json.dump({"lateral_m_at_4s": LATERAL, "longitudinal_factor": LONGITUDINAL, "names": NAMES, "scenes": len(rows),
                   "cache": os.path.abspath(args.cache), "rule": "docs/HEADS_PLAN.md R14 (boundary-path ablation)"}, f,
                  indent=1)
    print(json.dumps({"scenes": len(rows), "paths": len(rows) * len(NAMES)}), flush=True)


def pick_cmd(args) -> None:
    from train_selector import read_paths

    P = read_paths(args.paths, NAMES)
    out_rows, lines, counts = {}, [], {"limit_found_lateral": 0, "limit_found_longitudinal": 0, "scenes": 0}
    with open(args.scores, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            res = dict(zip(r["names"], r["results"]))
            chosen = pick(res)
            idx = [NAMES.index(n) for n in chosen]
            out_rows[r["token"]] = [P[r["token"]][i] for i in idx]
            lines.append(json.dumps({"token": r["token"], "names": [f"boundary_{j}" for j in range(4)],
                                     "results": [res[n] for n in chosen], "perturbations": chosen}) + "\n")
            counts["scenes"] += 1
            counts["limit_found_lateral"] += int(any(fails(res[n]) for n in chosen[:2]))
            counts["limit_found_longitudinal"] += int(any(fails(res[n]) for n in chosen[2:]))
    write_folders(args.out, [f"boundary_{j}" for j in range(4)], out_rows)
    with open(os.path.join(args.out, "scores.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(json.dumps(counts), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--cache", required=True, help="a stored-feature split with logged paths (train)")
    m.add_argument("--only-tokens", default=None, help="one NAVSIM token per line (the RL scenes)")
    m.add_argument("--out", required=True)
    k = sub.add_parser("pick")
    k.add_argument("--paths", required=True, help="make's output folder")
    k.add_argument("--scores", required=True, help="score_candidates.py's file for it")
    k.add_argument("--out", required=True)
    args = p.parse_args()
    make(args) if args.cmd == "make" else pick_cmd(args)


if __name__ == "__main__":
    main()
