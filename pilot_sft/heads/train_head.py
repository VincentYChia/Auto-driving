"""Train one path generator on stored pilot features (``cache_features.py``), on one GPU.

    python pilot_sft/heads/train_head.py --cache <cache> --kind masked --out <exp>/heads/<run>
    python pilot_sft/heads/train_head.py --cache <cache> --kind masked --soft-sigma 1 --l1-weight 1 --offset ...
    python pilot_sft/heads/train_head.py --cache <cache> --kind anchor --steps 2 --out ...

``<cache>`` holds ``train/`` and ``dev/`` splits. Every epoch writes ``epoch_<k>.pt`` (weights, config and the
target statistics: everything prediction needs) and ``checkpoint.pt`` (also the optimizer, for ``--resume``);
nothing is deleted, and the last epoch is copied to ``final.pt``. ``log.jsonl`` has the loss every 50 steps and,
after each epoch, the dev loss, the single-path ADE / FDE and the best-of-6 ADE on the dev split. ``run.json``
records the settings, the cache, the parameter count and the timings. Settings are [proposed].
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive_map  # noqa: E402
import features  # noqa: E402
import model as hm  # noqa: E402


def to_torch(b: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    out = {"feats": torch.from_numpy(b["feats"].view(np.int16)).view(torch.bfloat16),
           "mask": torch.from_numpy(b["mask"]), "ego": torch.from_numpy(b["ego"])}
    if "target_xy" in b:
        out["xy"] = torch.from_numpy(b["target_xy"])
    return {k: v.pin_memory().to(device, non_blocking=True) if device == "cuda" else v for k, v in out.items()}


def batches(store: features.Store, order: List[int], batch: int, workers: int) -> Iterator[Dict[str, np.ndarray]]:
    """``store.batch`` for consecutive slices of ``order``, read ahead in ``workers`` threads, in order."""
    slices = [order[i : i + batch] for i in range(0, len(order), batch)]
    ahead = 2 * workers
    with ThreadPoolExecutor(workers) as pool:
        futs = {j: pool.submit(store.batch, slices[j]) for j in range(min(ahead, len(slices)))}
        for j in range(len(slices)):
            b = futs.pop(j).result()
            if j + ahead < len(slices):
                futs[j + ahead] = pool.submit(store.batch, slices[j + ahead])
            yield b


def target_stats(store: features.Store, cfg: hm.HeadConfig, seed: int, rows: Optional[List[int]] = None
                 ) -> Dict[str, torch.Tensor]:
    """Normalisation (and D's anchors) from the training targets: every row, or only ``rows`` (a drive half)."""
    xy = torch.from_numpy(store.target_xy.reshape(len(store.target_xy), -1))
    if rows is not None:
        xy = xy[torch.as_tensor(rows, dtype=torch.long)]
    stats = {"mean": xy.mean(0), "std": xy.std(0)}
    if cfg.kind == "anchor":
        g = np.random.default_rng(seed)
        sub = xy[g.choice(len(xy), min(len(xy), 50000), replace=False)]
        z = (sub - stats["mean"]) / stats["std"].clamp_min(0.05)
        stats["anchors"] = hm.kmeans_anchors(z, cfg.anchors, seed=seed)
    return stats


@torch.no_grad()
def evaluate(head: hm.Head, dev: features.Store, device: str, batch: int, steps: int) -> Dict[str, float]:
    head.eval()
    g = torch.Generator(device=device).manual_seed(0)
    losses, ade, fde, best6, n = [], 0.0, 0.0, 0.0, 0
    for b in batches(dev, list(range(dev.rows)), batch, 2):
        t = to_torch(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            losses.append(float(head.loss(t["feats"], t["mask"], t["ego"], t["xy"])["loss"]))
            one, _ = head.generate(t["feats"], t["mask"], t["ego"], k=1, steps=steps, temperature=0.0, generator=g)
            six, _ = head.generate(t["feats"], t["mask"], t["ego"], k=6, steps=steps, temperature=1.0, generator=g)
        err = (one[:, 0].float() - t["xy"]).norm(dim=-1)  # [B, 8]
        ade += float(err.mean(-1).sum())
        fde += float(err[:, -1].sum())
        best6 += float((six.float() - t["xy"][:, None]).norm(dim=-1).mean(-1).min(-1).values.sum())
        n += len(err)
    head.train()
    return {"dev_loss": float(np.mean(losses)), "dev_ade": ade / n, "dev_fde": fde / n, "dev_best6_ade": best6 / n}


def save(path: str, obj) -> None:
    torch.save(obj, path + ".part")
    os.replace(path + ".part", path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache", required=True, help="folder with train/ and dev/ cache splits")
    p.add_argument("--out", required=True)
    p.add_argument("--kind", required=True, choices=("masked", "ddim", "flow", "anchor", "regression"))
    p.add_argument("--layers", default=None, help="stored layers to read (default: all in the cache)")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--steps", type=int, default=None, help="generation steps for the dev check (default per kind)")
    p.add_argument("--bin-m", type=float, default=0.1)
    p.add_argument("--no-metric-embed", action="store_true")
    p.add_argument("--soft-sigma", type=float, default=0.0)
    p.add_argument("--l1-weight", type=float, default=0.0)
    p.add_argument("--offset", action="store_true")
    p.add_argument("--flow-sigma", type=float, default=0.1)
    p.add_argument("--anchors", type=int, default=20)
    p.add_argument("--anchor-trunc", type=float, default=0.05,
                   help="D: the noise level the anchors start from, as a share of the schedule (R15 tunes it)")
    p.add_argument("--target", default="x0", choices=("x0", "eps", "v"), help="B: what the network predicts (R15)")
    p.add_argument("--schedule", default="cosine", choices=("cosine", "linear"), help="B: the noise schedule (R15)")
    p.add_argument("--time-sampling", default="uniform", choices=("uniform", "beta", "logit-normal"),
                   help="C: how training times are drawn (R15)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000, help="optimizer steps")
    # 8 readers (2026-11-20): with 4, the readers ran at 420% CPU while the 4090 was about 40% busy and disk wait
    # was 2-3%, so reading, not the GPU, set the pace. The data order is fixed by the seed; readers only prefetch.
    p.add_argument("--workers", type=int, default=8, help="threads reading features ahead of the GPU")
    p.add_argument("--limit", type=int, default=None, help="only the first N train rows (timing runs)")
    p.add_argument("--max-steps", type=int, default=None, help="stop early (timing runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drive-half", type=int, choices=(0, 1), default=None,
                   help="phase 6 cross-fitting: train only on the rows of this half of the training drives "
                        "(drive_map.drive_halves, seed 0); the target statistics come from that half too")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    train = features.Store(os.path.join(args.cache, "train"), layers)
    dev = features.Store(os.path.join(args.cache, "dev"), layers)
    default_steps = {"masked": 4, "ddim": 5, "flow": 2, "anchor": 2, "regression": 1}[args.kind]
    cfg = hm.HeadConfig(kind=args.kind, layers=train.layers, d_in=train.width, width=args.width, depth=args.depth,
                        heads=args.heads, max_tokens=max(train.tokens, dev.tokens), bin_m=args.bin_m,
                        metric_embed=not args.no_metric_embed, soft_sigma=args.soft_sigma, l1_weight=args.l1_weight,
                        offset=args.offset, flow_sigma=args.flow_sigma, anchors=args.anchors, anchor_trunc=args.anchor_trunc, target=args.target, schedule=args.schedule,
                        time_sampling=args.time_sampling,
                        steps=args.steps or default_steps)
    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if args.resume and os.path.exists(ckpt_path) else None
    rows = list(range(train.rows))[: args.limit]
    half_info = None
    if args.drive_half is not None:  # only this half's drives, so the other half's scenes stay unseen (phase 6)
        halves = drive_map.drive_halves(train.ids)
        rows = [r for r in rows if halves[drive_map.drive_of_id(train.ids[r])] == args.drive_half]
        half_info = {"half": args.drive_half, "split_seed": 0, "rows": len(rows),
                     "drives": sum(1 for h in halves.values() if h == args.drive_half), "drives_all": len(halves)}
        print(f"drive half {args.drive_half}: {len(rows)} rows from {half_info['drives']} of {len(halves)} drives",
              flush=True)
    stats = resume["stats"] if resume else target_stats(train, cfg, args.seed,
                                                        rows if args.drive_half is not None else None)
    if cfg.kind == "masked":
        xy = train.target_xy
        clipped = int(((xy[..., 0] < cfg.x_range[0]) | (xy[..., 0] > cfg.x_range[1]) |
                       (xy[..., 1] < cfg.y_range[0]) | (xy[..., 1] > cfg.y_range[1])).any(-1).sum())
        print(f"{clipped} of {len(xy)} training paths leave the grid and are clipped", flush=True)
    head = hm.Head(cfg, stats).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total = steps_per_epoch * args.epochs if args.max_steps is None else args.max_steps
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    epoch0, step = 0, 0
    if resume:
        head.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        epoch0, step = resume["epoch"], resume["step"]
        print(f"resumed after epoch {epoch0}, step {step}", flush=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        commit = None
    run = {"args": vars(args), "config": cfg.to_dict(), "params": n_params, "commit": commit,
           "cache": {"train": train.config, "dev": {k: dev.config.get(k) for k in ("rows", "source", "tokens")}},
           "train_rows": len(rows), "drive_half": half_info, "steps_per_epoch": steps_per_epoch,
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    print(f"{cfg.kind} head, {n_params / 1e6:.1f}M parameters, layers {cfg.layers}, {len(rows)} rows, "
          f"{steps_per_epoch} steps per epoch, {args.epochs} epochs", flush=True)

    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")
    start, seen = time.time(), 0
    head.train()
    epoch = epoch0 - 1
    for epoch in range(epoch0, args.epochs):
        order = random.Random(args.seed * 1000 + epoch).sample(rows, len(rows))
        t_epoch, acc = time.time(), {}
        for b in batches(train, order, args.batch, args.workers):
            t = to_torch(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = head.loss(t["feats"], t["mask"], t["ego"], t["xy"])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(t["xy"])
            for k, v in out.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            if step % 50 == 0:
                rec = {"step": step, "epoch": round(epoch + (step - epoch * steps_per_epoch) / steps_per_epoch, 3),
                       **{k: round(v / 50, 5) for k, v in acc.items()}, "grad_norm": round(float(gn), 3),
                       "lr": sched.get_last_lr()[0], "rows_per_s": round(seen / (time.time() - start), 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                acc = {}
            if args.max_steps and step >= args.max_steps:
                break
        ev = evaluate(head, dev, device, args.batch, cfg.steps)
        rec = {"epoch": epoch + 1, "step": step, **{k: round(v, 4) for k, v in ev.items()},
               "epoch_s": round(time.time() - t_epoch, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        weights = {"config": cfg.to_dict(), "stats": stats, "model": head.state_dict(), "epoch": epoch + 1,
                   "dev": ev}
        save(os.path.join(args.out, f"epoch_{epoch + 1}.pt"), weights)
        save(ckpt_path, {**weights, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step})
        if args.max_steps and step >= args.max_steps:
            break
    shutil.copyfile(os.path.join(args.out, f"epoch_{epoch + 1}.pt"), os.path.join(args.out, "final.pt"))
    run.update({"elapsed_s": round(time.time() - start, 1), "rows_per_s": round(seen / (time.time() - start), 1),
                "epochs_done": epoch + 1, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if device == "cuda" else None})
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    print(json.dumps({k: run[k] for k in ("elapsed_s", "rows_per_s", "epochs_done", "peak_gpu_gb")}), flush=True)


if __name__ == "__main__":
    main()
