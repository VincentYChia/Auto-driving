"""Train the selector (docs/HEADS_PLAN.md R14) on scored candidate paths of training scenes, on one GPU.

    python pilot_sft/heads/train_selector.py --cache <cache>/train --generator <run>/epoch_10.pt --scale 2 \\
        --candidates <run-half0>/navtrain_rl_5000/steps2 <oof_half1.jsonl> \\
        --candidates <run-half1>/navtrain_rl_5000/steps2 <oof_half0.jsonl> --out <exp>/heads/<selector run>

Each ``--candidates`` pair is a ``predict_head.py`` steps folder and its ``score_candidates.py`` file. The candidates
are the deployed sampler's: the own pick (``single``) and ``--k`` samples at the chosen sampling setting ``--scale``,
7 per scene. Scenes are joined to the stored features by token. ``--generator`` gives the normalisation statistics
and, by default, the stored layers (the chosen path generator's).

``--heldout-drives`` (10) of the candidate scenes' drives (seed 0, ``drive_map.heldout_drives``) stay out of
training. After every epoch the selector picks on them at each own-pick margin in ``selector.DELTAS``; the
(epoch, margin) with the highest mean true PDMS of the selected paths is kept in ``choice.json`` (a tie goes to the
earlier epoch, then the larger margin). No setting uses navdev. Every epoch is saved as ``epoch_<k>.pt`` (weights,
config, statistics), with ``checkpoint.pt`` (plus the optimizer) for ``--resume``; ``log.jsonl`` has the loss every 50
steps and the held-out results after each epoch; ``run.json`` the settings and data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive_map  # noqa: E402
import features  # noqa: E402
import selector as sl  # noqa: E402
from predict_head import bare, sample_dir  # noqa: E402
from train_head import batches, save, to_torch  # noqa: E402


def candidate_names(paths: str, scale: float, k: int) -> List[str]:
    """The own pick, then the K samples at setting ``scale``: the deployed sampler's candidates, in this order."""
    names = ["single"] + [sample_dir(scale, j) for j in range(k)]
    missing = [n for n in names if not os.path.isfile(os.path.join(paths, n, "shard_000_of_001.jsonl"))]
    if missing:
        raise SystemExit(f"{paths} lacks candidate folders {missing}")
    return names


def read_paths(paths: str, names: List[str]) -> Dict[str, np.ndarray]:
    """token -> waypoints [K, 8, 2] of the named candidate folders."""
    out: Dict[str, list] = {}
    for j, n in enumerate(names):
        with open(os.path.join(paths, n, "shard_000_of_001.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out.setdefault(r["token"], [None] * len(names))[j] = r["waypoints"]
    bad = [t for t, v in out.items() if any(x is None for x in v)]
    if bad:
        raise SystemExit(f"{len(bad)} scenes lack a path in some candidate folder, e.g. {bad[0]}")
    return {t: np.asarray(v, dtype=np.float32) for t, v in out.items()}


def scores_from_jsonl(path: str, names: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """token -> (sub-scores [K, 5] in ``selector.SUBS`` order, PDMS [K], valid [K]) from a ``score_candidates.py`` file,
    for the named candidates (the file may hold more)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx = [r["names"].index(n) for n in names]
            res = [r["results"][i] for i in idx]
            valid = np.array(["error" not in x for x in res])
            sub = np.array([[x[s] for s in sl.SUBS] if v else [0.0] * len(sl.SUBS) for x, v in zip(res, valid)],
                           dtype=np.float32)
            pdms = np.array([x["score"] if v else 0.0 for x, v in zip(res, valid)], dtype=np.float32)
            out[r["token"]] = (sub, pdms, valid)
    return out


class CandidateSet:
    """Candidates of scenes in one stored-feature split: rows, waypoints, true sub-scores, PDMS and validity."""

    def __init__(self, store: features.Store, pairs: List[Tuple[str, str]], scale: float, k: int):
        row_of = {bare(i): r for r, i in enumerate(store.ids)}
        rows, xy, sub, pdms, valid, seen = [], [], [], [], [], set()
        self.sources = []
        for paths, scores in pairs:
            names = candidate_names(paths, scale, k)
            P = read_paths(paths, names)
            S = scores_from_jsonl(scores, names)
            toks = [t for t in P if t in S]
            missing = [t for t in toks if t not in row_of]
            if missing:
                raise SystemExit(f"{len(missing)} scenes of {paths} are not in {store.folder}, e.g. {missing[0]}")
            dup = [t for t in toks if t in seen]
            if dup:
                raise SystemExit(f"{len(dup)} scenes appear in two candidate sets, e.g. {dup[0]}")
            seen.update(toks)
            for t in toks:
                rows.append(row_of[t])
                xy.append(P[t])
                sub.append(S[t][0])
                pdms.append(S[t][1])
                valid.append(S[t][2])
            self.sources.append({"paths": os.path.abspath(paths), "scores": os.path.abspath(scores), "names": names,
                                 "scenes": len(toks), "unscored": len(P) - len(toks)})
        if not rows:
            raise SystemExit("no scored candidates")
        self.rows, self.xy = np.asarray(rows), np.stack(xy)
        self.sub, self.pdms, self.valid = np.stack(sub), np.stack(pdms), np.stack(valid)
        self.drives = np.array([drive_map.drive_of_id(store.ids[r]) for r in self.rows])
        self.ade = None
        if store.has_targets:  # the imitation ablation's target: each candidate's ADE to the logged path
            self.ade = np.linalg.norm(self.xy - store.target_xy[self.rows][:, None], axis=-1).mean(-1)

    def __len__(self) -> int:
        return len(self.rows)


@torch.no_grad()
def predict(model: sl.Selector, store: features.Store, cs: CandidateSet, idx: np.ndarray, device: str, batch: int
            ) -> Dict[str, np.ndarray]:
    """Predicted ``score``, ``pdms`` and ``sub`` (sub-score target only) for the candidate scenes ``idx``."""
    model.eval()
    out: Dict[str, list] = {}
    order = list(range(len(idx)))
    for s in range(0, len(order), batch):
        part = idx[s : s + batch]
        b = to_torch(store.batch(cs.rows[part]), device)
        paths = torch.from_numpy(cs.xy[part]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(b["feats"], b["mask"], b["ego"], paths)
        for k, v in sl.predicted(logits, model.cfg).items():
            out.setdefault(k, []).append(v.float().cpu().numpy())
    model.train()
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache", required=True, help="the stored-feature split holding the candidate scenes (train)")
    p.add_argument("--generator", required=True, help="the chosen path generator's weights (.pt): statistics, layers")
    p.add_argument("--candidates", nargs=2, action="append", required=True, metavar=("PATHS", "SCORES"))
    p.add_argument("--scale", type=float, required=True, help="the method's chosen sampling setting")
    p.add_argument("--k", type=int, default=6, help="samples per scene besides the own pick")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default=None, help="stored layers (default: the generator's)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--target", default="subscores", choices=("subscores", "pdms"))
    p.add_argument("--listwise", type=float, default=0.0, help="ablation: weight of the listwise term (0.25)")
    p.add_argument("--imitation", type=float, default=0.0, help="ablation: weight of the imitation output (0.1)")
    p.add_argument("--heldout-drives", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128, help="scenes per step (each with K + 1 candidates)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    gen = torch.load(args.generator, map_location="cpu", weights_only=False)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(gen["config"]["layers"])
    store = features.Store(args.cache, layers)
    cs = CandidateSet(store, [tuple(c) for c in args.candidates], args.scale, args.k)
    held = drive_map.heldout_drives([store.ids[r] for r in cs.rows], args.heldout_drives, seed=0)
    is_held = np.isin(cs.drives, sorted(held))
    train_idx, held_idx = np.flatnonzero(~is_held), np.flatnonzero(is_held)
    if len(train_idx) == 0 or len(held_idx) == 0:
        raise SystemExit(f"need scenes both in and out of the held-out drives ({len(train_idx)}, {len(held_idx)})")
    if args.imitation > 0 and cs.ade is None:
        raise SystemExit("the imitation ablation needs the logged paths (a split with targets)")
    cfg = sl.SelectorConfig(layers=layers, d_in=store.width, width=args.width, depth=args.depth, heads=args.heads,
                            max_tokens=max(store.tokens, gen["config"].get("max_tokens", 512)), dropout=args.dropout,
                            target=args.target, listwise=args.listwise, imitation=args.imitation)
    stats = {"mean": gen["stats"]["mean"], "std": gen["stats"]["std"]}
    model = sl.Selector(cfg, stats).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    steps_per_epoch = math.ceil(len(train_idx) / args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    epoch0, step, history = 0, 0, []
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        epoch0, step, history = ck["epoch"], ck["step"], ck["history"]
        print(f"resumed after epoch {epoch0}, step {step}", flush=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        commit = None
    run = {"args": vars(args), "config": cfg.to_dict(), "params": n_params, "commit": commit,
           "generator": os.path.abspath(args.generator), "cache": os.path.abspath(args.cache), "sources": cs.sources,
           "scenes": {"train": int(len(train_idx)), "heldout": int(len(held_idx))},
           "drives": {"train": int(len(set(cs.drives[train_idx]))), "heldout": sorted(held)},
           "candidates_per_scene": int(cs.xy.shape[1]), "unscorable_candidates": int((~cs.valid).sum()),
           "rule": "docs/HEADS_PLAN.md R14", "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    print(f"selector {n_params / 1e6:.2f}M parameters, layers {layers}, {len(train_idx)} training scenes, "
          f"{len(held_idx)} held out ({len(held)} drives), {cs.xy.shape[1]} candidates each", flush=True)

    T = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")
    start, seen = time.time(), 0
    model.train()
    for epoch in range(epoch0, args.epochs):
        order = np.asarray(random.Random(args.seed * 1000 + epoch).sample(list(train_idx), len(train_idx)))
        slices = [order[i : i + args.batch] for i in range(0, len(order), args.batch)]
        acc: Dict[str, float] = {}
        t_epoch = time.time()
        for part, b in zip(slices, batches(store, [int(r) for s in slices for r in cs.rows[s]], args.batch, args.workers)):
            t = to_torch(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(t["feats"], t["mask"], t["ego"], T(cs.xy[part]))
            out = sl.loss(logits, T(cs.sub[part]), T(cs.pdms[part]), T(cs.valid[part]), cfg,
                          T(cs.ade[part]) if cs.ade is not None else None)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(part)
            for k, v in out.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            if step % 50 == 0:
                rec = {"step": step, "epoch": round(epoch + (step - epoch * steps_per_epoch) / steps_per_epoch, 3),
                       **{k: round(v / 50, 5) for k, v in acc.items()}, "grad_norm": round(float(gn), 3),
                       "lr": sched.get_last_lr()[0], "scenes_per_s": round(seen / (time.time() - start), 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                acc = {}
        pr = predict(model, store, cs, held_idx, device, args.batch)
        true = cs.pdms[held_idx]
        ch = sl.choose_delta(pr["score"], true)
        rep = sl.report(true, cs.sub[held_idx], pr["score"], sl.select(pr["score"], ch["delta"]), cs.drives[held_idx],
                        paths=cs.xy[held_idx], pred_sub=pr.get("sub"))
        rec = {"epoch": epoch + 1, "step": step, "heldout": {"delta": ch["delta"], "selected_pdms": round(ch["selected_pdms"], 5),
               "by_delta": {d: round(v, 5) for d, v in ch["by_delta"].items()},
               **{k: rep[k] for k in ("own_pick_pdms", "best_of_k_pdms", "switch_rate", "top1_exact", "top1_tie_aware")}},
               "epoch_s": round(time.time() - t_epoch, 1)}
        history.append({"epoch": epoch + 1, "delta": ch["delta"], "selected_pdms": ch["selected_pdms"]})
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        weights = {"config": cfg.to_dict(), "stats": stats, "model": model.state_dict(), "epoch": epoch + 1,
                   "heldout": {"choice": ch, "report": rep}}
        save(os.path.join(args.out, f"epoch_{epoch + 1}.pt"), weights)
        save(ckpt_path, {**weights, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step,
                         "history": history})
        best = max(history, key=lambda h: (round(h["selected_pdms"], 10), -h["epoch"]))
        with open(os.path.join(args.out, "choice.json.part"), "w", encoding="utf-8") as f:
            json.dump({"epoch": best["epoch"], "weights": f"epoch_{best['epoch']}", "delta": best["delta"],
                       "heldout_selected_pdms": best["selected_pdms"], "epochs_done": epoch + 1, "history": history,
                       "rule": "docs/HEADS_PLAN.md R14: epoch and margin on the held-out training drives"}, f, indent=1)
        os.replace(os.path.join(args.out, "choice.json.part"), os.path.join(args.out, "choice.json"))
    run.update({"elapsed_s": round(time.time() - start, 1), "epochs_done": args.epochs,
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None})
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)


if __name__ == "__main__":
    main()
