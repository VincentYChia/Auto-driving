"""NAVSIM paths from a trained path generator on stored features, for NAVSIM's PDM scorer.

    python pilot_sft/heads/predict_head.py --run <exp>/heads/<run> --cache <cache>/navtest --steps 1,2,4,8 --k 6
    python pilot_sft/heads/predict_head.py --run <run> --split navdev --cache <cache>/dev --weights epoch_10 \\
        --steps 4 --k 6 --scales 0.5,1,1.5

For every ``--steps`` value it writes, under ``<run>/<split>_<weights>/steps<S>/``:
- ``single/``: the generator's deterministic best guess (``generate(temperature=0)``: A's most confident tokens,
  B and C from the zero-noise start, D from the clean anchors, R's one path);
- for each ``--scales`` value x, ``K`` sampled paths per scene for best-of-K, in ``sample_<j>/`` (x = 1, the
  training setting) or ``scale<x>_sample_<j>/``. x is each method's one sampling setting (docs/HEADS_PLAN.md R7).

Each folder holds ``shard_000_of_001.jsonl`` with ``{"token", "waypoints"}`` rows as ``navsim_eval.predictions``
reads them. Tokens are bare NAVSIM tokens: a dev-view id ``openscene/<log>/<token>`` becomes ``<token>``.
``timing.json`` holds the generation time per scene, from the stored features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import model as hm  # noqa: E402
from train_head import batches, to_torch  # noqa: E402


def load_head(path: str, device: str) -> hm.Head:
    w = torch.load(path, map_location="cpu", weights_only=False)
    cfg = hm.HeadConfig(**{k: (tuple(v) if k in ("x_range", "y_range") else v) for k, v in w["config"].items()})
    head = hm.Head(cfg, w["stats"])
    head.load_state_dict(w["model"])
    return head.to(device).eval()


def sample_dir(scale: float, j: int) -> str:
    return f"sample_{j}" if scale == 1.0 else f"scale{scale:g}_sample_{j}"


def bare(token: str) -> str:
    return token.rsplit("/", 1)[-1]


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--weights", default="final", help="final or epoch_<k>")
    p.add_argument("--cache", required=True, help="the stored-feature split to predict (navtest, dev, ...)")
    p.add_argument("--split", default="navtest", choices=("navtest", "navdev", "navtrain"),
                   help="names the output folder")
    p.add_argument("--only-tokens", default=None,
                   help="a file with one NAVSIM token per line: predict only those scenes (phase 6's out-of-fold "
                        "candidates, from drive_map.py halves)")
    p.add_argument("--steps", default=None, help="comma-separated step counts (default: the generator's own)")
    p.add_argument("--k", type=int, default=6, help="sampled paths per scene; 0 = single paths only")
    p.add_argument("--scales", default="1", help="comma-separated sampling settings for the K paths (R7)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = load_head(os.path.join(args.run, f"{args.weights}.pt"), device)
    store = features.Store(args.cache, head.cfg.layers)
    out_root = args.out or os.path.join(args.run, f"{args.split}_{args.weights}")
    steps_list = [int(s) for s in args.steps.split(",")] if args.steps else [head.cfg.steps]
    scales = [float(s) for s in args.scales.split(",")] if args.k > 0 else []
    tokens = [bare(t) for t in store.ids]
    rows = list(range(store.rows))
    if args.only_tokens:
        with open(args.only_tokens, encoding="utf-8") as f:
            keep = set(f.read().split())
        rows = [r for r in rows if tokens[r] in keep]
        print(f"{len(rows)} of {store.rows} rows are in {args.only_tokens} ({len(keep)} tokens)", flush=True)
    timing = {}
    for steps in steps_list:
        names = ["single"] + [sample_dir(x, j) for x in scales for j in range(args.k)]
        files = {}
        for name in names:
            d = os.path.join(out_root, f"steps{steps}", name)
            os.makedirs(d, exist_ok=True)
            files[name] = open(os.path.join(d, "shard_000_of_001.jsonl"), "w", encoding="utf-8")
        g = torch.Generator(device=device).manual_seed(0)
        t_single, t_k = 0.0, {x: 0.0 for x in scales}
        pos = 0
        for b in batches(store, rows, args.batch, 2):
            t = to_torch(b, device)
            ids = [tokens[r] for r in rows[pos : pos + len(t["ego"])]]
            pos += len(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                sync(device)
                t0 = time.time()
                one, _ = head.generate(t["feats"], t["mask"], t["ego"], k=1, steps=steps, temperature=0.0, generator=g)
                sync(device)
                t_single += time.time() - t0
                many = {}
                for x in scales:
                    t0 = time.time()
                    many[x], _ = head.generate(t["feats"], t["mask"], t["ego"], k=args.k, steps=steps,
                                               temperature=x, generator=g)
                    sync(device)
                    t_k[x] += time.time() - t0
            one = one.float().cpu()
            many = {x: m.float().cpu() for x, m in many.items()}
            for i, tok in enumerate(ids):
                files["single"].write(json.dumps({"token": tok, "waypoints": one[i, 0].round(decimals=3).tolist()}) + "\n")
                for x in scales:
                    for j in range(args.k):
                        files[sample_dir(x, j)].write(
                            json.dumps({"token": tok, "waypoints": many[x][i, j].round(decimals=3).tolist()}) + "\n")
        for f in files.values():
            f.close()
        n = max(len(rows), 1)
        timing[f"steps{steps}"] = {"single_ms_per_scene": round(1000 * t_single / n, 3),
                                   **{f"k{args.k}_scale{x:g}_ms_per_scene": round(1000 * v / n, 3)
                                      for x, v in t_k.items()},
                                   "batch": args.batch, "scenes": len(rows)}
        print(json.dumps({f"steps{steps}": timing[f"steps{steps}"]}), flush=True)
    with open(os.path.join(out_root, "timing.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": args.weights, "split": args.split, "cache": os.path.abspath(args.cache),
                   "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", **timing}, f, indent=1)


if __name__ == "__main__":
    main()
