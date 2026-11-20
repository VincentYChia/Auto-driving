"""Apply a trained selector to one candidate set and write the selected path per scene, for NAVSIM's PDM scorer.

    python pilot_sft/heads/apply_selector.py --selector <exp>/heads/<selector run> --cache <cache>/dev \\
        --paths <run>/navdev_epoch_10/steps2 --scale 2 --out <selector run>/navdev_<run> \\
        --exp <exp> --split navdev --drives <cache root>/drives_navdev.json

The candidates are the own pick (``single``) and ``--k`` samples at setting ``--scale`` (the deployed sampler's). The
selector's epoch and own-pick margin come from its ``choice.json`` (chosen on held-out training drives), unless
``--weights`` / ``--delta`` override them. Writes, under ``--out``:
- ``selected/shard_000_of_001.jsonl``: the selected path per scene, in ``predict_head.py``'s format, so NAVSIM's own
  scorer gives the reported number;
- ``selection.jsonl``: per scene, the chosen candidate and every candidate's predicted score (and sub-scores);
- ``report.json`` when true scores are given: R14's measures (``selector.report``), the selected path's PDMS from the
  candidates' own scores, with the drive interval of selected minus own pick when ``--drives`` is given.

True scores come from a ``score_candidates.py`` file (``--scores``) or from the NAVSIM score files of the candidate
folders under ``--exp`` (named as ``scripts/desk/score_head_run.sh`` and the driver name them).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import selector as sl  # noqa: E402
from predict_head import bare  # noqa: E402
from train_head import to_torch  # noqa: E402
from train_selector import candidate_names, read_paths, scores_from_jsonl  # noqa: E402


def load_selector(path: str, device: str) -> sl.Selector:
    w = torch.load(path, map_location="cpu", weights_only=False)
    m = sl.Selector(sl.SelectorConfig(**w["config"]), w["stats"])
    m.load_state_dict(w["model"])
    return m.to(device).eval()


def score_name(folder: str, heads: str, split: str) -> str:
    """The NAVSIM experiment name of a candidate folder, as the driver scores it."""
    rel = os.path.relpath(os.path.abspath(folder), os.path.abspath(heads))
    return "heads-" + rel.replace(os.sep, "-").replace("/", "-") + f"_{split}_v1"


def scores_from_exp(exp: str, paths: str, names: List[str], split: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """token -> (sub-scores [K, 5], PDMS [K], valid [K]) from the candidate folders' NAVSIM score files."""
    import pandas as pd

    frames = []
    for n in names:
        name = score_name(os.path.join(paths, n), os.path.join(exp, "heads"), split)
        files = sorted(glob.glob(os.path.join(exp, name, "*", "*.csv")))
        if not files:
            raise SystemExit(f"no NAVSIM scores for {name}")
        d = pd.read_csv(files[-1])
        frames.append(d[d.token != "average"].set_index("token"))
    tokens = sorted(set.intersection(*[set(f.index) for f in frames]))
    out = {}
    for t in tokens:
        valid = np.array([bool(f.at[t, "valid"]) for f in frames])
        sub = np.array([[float(f.at[t, s]) for s in sl.SUBS] for f in frames], dtype=np.float32)
        pdms = np.array([float(f.at[t, "score"]) for f in frames], dtype=np.float32)
        out[t] = (np.where(valid[:, None], sub, 0.0), np.where(valid, pdms, 0.0), valid)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--selector", required=True, help="a train_selector.py output folder")
    p.add_argument("--weights", default=None, help="epoch_<k> (default: choice.json's)")
    p.add_argument("--delta", type=float, default=None, help="own-pick margin (default: choice.json's)")
    p.add_argument("--cache", required=True, help="the stored-feature split of the candidate scenes")
    p.add_argument("--paths", required=True, help="a predict_head.py steps<S> folder")
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--out", required=True)
    p.add_argument("--scores", default=None, help="a score_candidates.py file for these candidates")
    p.add_argument("--exp", default=None, help="the NAVSIM experiment root holding the candidates' score files")
    p.add_argument("--split", default="navdev", choices=("navdev", "navtest", "navtrain"))
    p.add_argument("--drives", default=None, help="token -> drive map (drive_map.py) for the interval")
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    choice = {}
    if os.path.exists(os.path.join(args.selector, "choice.json")):
        with open(os.path.join(args.selector, "choice.json"), encoding="utf-8") as f:
            choice = json.load(f)
    weights = args.weights or choice.get("weights")
    delta = args.delta if args.delta is not None else choice.get("delta")
    if weights is None or delta is None:
        raise SystemExit("no choice.json: give --weights and --delta")
    model = load_selector(os.path.join(args.selector, f"{weights}.pt"), device)
    store = features.Store(args.cache, model.cfg.layers)
    names = candidate_names(args.paths, args.scale, args.k)
    P = read_paths(args.paths, names)
    row_of = {bare(i): r for r, i in enumerate(store.ids)}
    tokens = [t for t in P if t in row_of]
    if len(tokens) != len(P):
        raise SystemExit(f"{len(P) - len(tokens)} candidate scenes are not in {args.cache}")
    rows = np.array([row_of[t] for t in tokens])
    xy = np.stack([P[t] for t in tokens])
    score, pdms, sub = [], [], []
    with torch.no_grad():
        for s in range(0, len(tokens), args.batch):
            b = to_torch(store.batch(rows[s : s + args.batch]), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(b["feats"], b["mask"], b["ego"], torch.from_numpy(xy[s : s + args.batch]).to(device))
            pr = sl.predicted(logits, model.cfg)
            score.append(pr["score"].float().cpu().numpy())
            pdms.append(pr["pdms"].float().cpu().numpy())
            if "sub" in pr:
                sub.append(pr["sub"].float().cpu().numpy())
    score, pdms = np.concatenate(score), np.concatenate(pdms)
    sub = np.concatenate(sub) if sub else None
    chosen = sl.select(score, delta)
    os.makedirs(os.path.join(args.out, "selected"), exist_ok=True)
    with open(os.path.join(args.out, "selected", "shard_000_of_001.jsonl"), "w", encoding="utf-8") as f:
        for i, t in enumerate(tokens):
            f.write(json.dumps({"token": t, "waypoints": np.round(xy[i, chosen[i]], 3).tolist()}) + "\n")
    truth: Optional[Dict] = None
    if args.scores:
        truth = scores_from_jsonl(args.scores, names)
    elif args.exp:
        truth = scores_from_exp(args.exp, args.paths, names, args.split)
    true = true_sub = None
    if truth is not None:
        unscored = sum(1 for t in tokens if t not in truth)
        if unscored:
            raise SystemExit(f"{unscored} scenes have no true scores")
        true = np.stack([truth[t][1] for t in tokens])
        true_sub = np.stack([truth[t][0] for t in tokens])
    with open(os.path.join(args.out, "selection.jsonl"), "w", encoding="utf-8") as f:
        for i, t in enumerate(tokens):
            rec = {"token": t, "chosen": int(chosen[i]), "chosen_name": names[chosen[i]],
                   "score": np.round(score[i], 5).tolist(), "pdms": np.round(pdms[i], 5).tolist()}
            if sub is not None:
                rec["sub"] = np.round(sub[i], 4).tolist()
            if true is not None:  # the system path's true PDMS, and the own pick's (R16's final choice reads these)
                rec["true_chosen"], rec["true_own"] = float(true[i, chosen[i]]), float(true[i, 0])
            f.write(json.dumps(rec) + "\n")
    summary = {"selector": os.path.abspath(args.selector), "weights": weights, "delta": delta, "names": names,
               "paths": os.path.abspath(args.paths), "scenes": len(tokens), "switch_rate": float((chosen != 0).mean())}
    if truth is not None:
        drives = list(tokens)  # without a drive map, each scene is its own group
        if args.drives:
            with open(args.drives, encoding="utf-8") as f:
                dm = json.load(f)
            drives = [dm[t] for t in tokens]
        summary["report"] = sl.report(true, true_sub, score, chosen, drives, paths=xy, pred_sub=sub)
        summary["intervals_by"] = "drive" if args.drives else "scene"
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("scenes", "weights", "delta", "switch_rate")}), flush=True)
    if "report" in summary:
        r = summary["report"]
        print(json.dumps({k: r[k] for k in ("selected_pdms", "own_pick_pdms", "selected_minus_own",
                                            "selected_minus_own_ci", "best_of_k_pdms")}), flush=True)


if __name__ == "__main__":
    main()
