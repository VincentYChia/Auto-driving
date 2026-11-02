"""What a trained pilot relies on: its dev error with parts of the input removed or bent.

Same dev samples as ``evaluate.py`` and ``baselines.py`` (same --samples, --seed).

    python pilot_sft/probes.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke \\
        --model Qwen/Qwen3.5-0.8B --adapter runs/qwen35-0.8b-smoke-e1/adapter --out runs/qwen35-0.8b-smoke-e1/probes.json

- ``full``: the normal input (reproduces ``evaluate.py``).
- ``blank_image``: every camera image replaced by flat grey. If ADE barely moves, the image is unused.
- ``no_ego``: ego state and past positions replaced by "not available" (as ``--ego-dropout`` trains);
  what the image and the command alone give. Compare it with ``command_mean`` in ``baselines.py``.
- ``no_ego_blank_image``: both removed. The gap to ``no_ego`` is what the image carries when nothing
  else gives the speed (grey is unlike any training image, so this can fall below ``command_mean``).
- ``drift``: the past positions bent as if the car had been sliding sideways at --drift m/s, both
  directions. ``follow`` is the share of that slide carried into the 4 s prediction: 1 = continues
  it fully (pure extrapolation), 0 = ignores it. The image is not changed, so the inputs disagree:
  a rough probe of whether the model trusts the history or the picture. The no-image MLP from
  ``baselines.py`` gets the same probe for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

import numpy as np

import baselines
import evaluate
import train_lora

PAST_S = (-1.5, -1.0, -0.5)


def drift(item: Dict[str, Any], mps: float) -> Dict[str, Any]:
    """Past positions as if the car had been sliding right at ``mps`` (left if negative) until now."""
    lines = item["prompt"].split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("Past positions"))
    head, tail = lines[k].split(":", 1)
    pairs = [(float(x), float(y)) for x, y in baselines.PAIR.findall(tail)]
    moved = [(x, y - mps * t) for (x, y), t in zip(pairs, PAST_S)]  # y(t) = -mps * t: left in the past
    lines[k] = head + ": " + " ".join(f"({x:.1f}, {y:.1f})" for x, y in np.round(moved, 1) + 0.0) + "."
    return {**item, "prompt": "\n".join(lines)}


def answers(backend: evaluate.HF, items: List[Dict[str, Any]], batch: int, label: str) -> List[Any]:
    out = []
    for i in range(0, len(items), batch):
        out += [evaluate.parse_path(t) for t in backend.answer(items[i : i + batch])]
        print(f"[probe {label}] {len(out)}/{len(items)}", flush=True)
    return out


def scored(preds: List[Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [k for k, p in enumerate(preds) if p is not None]
    err = baselines.errors(np.array([preds[k] for k in ok]), [items[k] for k in ok])
    return {**baselines.summary(err, [items[k] for k in ok]), "parse_failures": len(preds) - len(ok)}


def follow(base: List[Any], bent: List[Any], mps: float) -> float:
    """Mean change of the predicted lateral position at 4 s, as a share of the slide continued (-4 * mps)."""
    pairs = [(b[-1][1], p[-1][1]) for b, p in zip(base, bent) if b is not None and p is not None]
    return round(float(np.mean([(p - b) / (-4.0 * mps) for b, p in pairs])), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--drift", type=float, default=0.5, help="m/s of sideways slide in the drift probe")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    items = baselines.dev_items(args.view, args.split, args.samples, args.seed)
    backend = evaluate.HF(args.model, args.adapter, args.view)
    full = answers(backend, items, args.batch, "full")
    result: Dict[str, Any] = {"view": args.view, "model": args.model, "adapter": args.adapter, "samples": len(items),
                              "seed": args.seed, "drift_mps": args.drift, "full": scored(full, items)}
    result["no_ego"] = scored(answers(backend, [train_lora.mask_ego(it) for it in items], args.batch, "no_ego"), items)
    backend.blank_image = True
    result["blank_image"] = scored(answers(backend, items, args.batch, "blank_image"), items)
    result["no_ego_blank_image"] = scored(answers(backend, [train_lora.mask_ego(it) for it in items], args.batch,
                                                  "no_ego_blank_image"), items)
    backend.blank_image = False
    right = answers(backend, [drift(it, args.drift) for it in items], args.batch, "drift right")
    left = answers(backend, [drift(it, -args.drift) for it in items], args.batch, "drift left")
    result["drift"] = {"follow_model": round((follow(full, right, args.drift) + follow(full, left, -args.drift)) / 2, 3)}

    mlp = baselines.fit_mlp(baselines.read_jsonl(os.path.join(args.view, "train.jsonl")))
    m0 = mlp(items).tolist()
    mr, ml = mlp([drift(it, args.drift) for it in items]).tolist(), mlp([drift(it, -args.drift) for it in items]).tolist()
    result["drift"]["follow_mlp_no_image"] = round((follow(m0, mr, args.drift) + follow(m0, ml, -args.drift)) / 2, 3)

    for key in ("full", "no_ego", "blank_image", "no_ego_blank_image"):
        print(f"{key:12s} ADE {result[key]['ade_m']:.3f}  FDE {result[key]['fde_m']:.3f}  "
              f"parse failures {result[key]['parse_failures']}", flush=True)
    print("drift follow:", result["drift"], flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
