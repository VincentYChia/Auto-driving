"""NAVSIM paths from a trained path generator on stored navtest features, for NAVSIM's PDM scorer.

    python pilot_sft/heads/predict_head.py --run <exp>/heads/<run> --cache <cache>/navtest --steps 1,2,4,8 --k 6

For every ``--steps`` value it writes ``<run>/navtest_<weights>/steps<S>/single/`` (the head's one path: greedy
for the masked head, the top-scored anchor for the anchored head, a fixed noise draw for B and C) and
``sample_<j>/`` for ``j < k`` (K paths per scene for best-of-K: sampled goals, the top-K anchors, or K noise
draws), each a ``shard_000_of_001.jsonl`` of ``{"token", "waypoints"}`` rows as ``navsim_eval.predictions``
reads them. ``timing.json`` holds the generation time per scene (the stored features are read, not computed).
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--weights", default="final", help="final or epoch_<k>")
    p.add_argument("--cache", required=True, help="the navtest cache split")
    p.add_argument("--steps", default=None, help="comma-separated step counts (default: the head's own)")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0, help="masked head: goal sampling for the K paths")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = load_head(os.path.join(args.run, f"{args.weights}.pt"), device)
    store = features.Store(args.cache, head.cfg.layers)
    out_root = args.out or os.path.join(args.run, f"navtest_{args.weights}")
    steps_list = [int(s) for s in args.steps.split(",")] if args.steps else [head.cfg.steps]
    timing = {}
    for steps in steps_list:
        dirs = {"single": os.path.join(out_root, f"steps{steps}", "single")}
        dirs.update({f"sample_{j}": os.path.join(out_root, f"steps{steps}", f"sample_{j}") for j in range(args.k)})
        files = {}
        for name, d in dirs.items():
            os.makedirs(d, exist_ok=True)
            files[name] = open(os.path.join(d, "shard_000_of_001.jsonl"), "w", encoding="utf-8")
        g = torch.Generator(device=device).manual_seed(0)
        t_single = t_k = 0.0
        row = 0
        for b in batches(store, list(range(store.rows)), args.batch, 2):
            t = to_torch(b, device)
            ids = store.ids[row : row + len(t["ego"])]
            row += len(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                one, _ = head.generate(t["feats"], t["mask"], t["ego"], k=1, steps=steps, temperature=0.0, generator=g)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                many, _ = head.generate(t["feats"], t["mask"], t["ego"], k=args.k, steps=steps,
                                        temperature=args.temperature, generator=g)
                if device == "cuda":
                    torch.cuda.synchronize()
                t2 = time.time()
            t_single, t_k = t_single + t1 - t0, t_k + t2 - t1
            one, many = one.float().cpu(), many.float().cpu()
            for i, tok in enumerate(ids):
                files["single"].write(json.dumps({"token": tok, "waypoints": one[i, 0].round(decimals=3).tolist()}) + "\n")
                for j in range(args.k):
                    files[f"sample_{j}"].write(json.dumps({"token": tok, "waypoints": many[i, j].round(decimals=3).tolist()}) + "\n")
        for f in files.values():
            f.close()
        timing[f"steps{steps}"] = {"single_ms_per_scene": round(1000 * t_single / store.rows, 3),
                                   f"k{args.k}_ms_per_scene": round(1000 * t_k / store.rows, 3),
                                   "batch": args.batch, "scenes": store.rows}
        print(json.dumps({f"steps{steps}": timing[f"steps{steps}"]}), flush=True)
    with open(os.path.join(out_root, "timing.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": args.weights, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   **timing}, f, indent=1)


if __name__ == "__main__":
    main()
