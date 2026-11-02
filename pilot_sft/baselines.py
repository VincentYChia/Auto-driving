"""No-image baselines for the pilot, on the same dev samples ``evaluate.py`` scores (same --samples, --seed).

Each baseline sees only what the prompt gives the model besides the image: speed, acceleration and
yaw rate (as rounded in the prompt), past positions and route command. A pilot that does not beat
the learned ones here is extrapolating the car's own motion, not reading the road.

    python pilot_sft/baselines.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke --out runs/baselines-smoke.json

- constant velocity: current speed, straight ahead (the baseline ``evaluate.py`` prints);
- kinematic: constant acceleration and yaw rate, speed floored at 0;
- linear: ridge regression from the prompt's numbers to the 16 output coordinates;
- mlp: a 2 x 256 ReLU network on the same numbers, L1 loss, fixed settings, mean over 3 seeds;
- command_mean: the mean training path for the route command, no ego state either. The bar for a
  pilot whose ego state is hidden (``probes.py`` no_ego): beating it means reading the image.
The learned baselines train on ``train.jsonl`` with settings fixed in advance, not tuned on dev.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

TARGET_S = np.array([0.5 * (k + 1) for k in range(8)])
NUM = r"([-+]?\d+(?:\.\d+)?)"
STATE = re.compile(rf"speed {NUM} m/s, acceleration {NUM} m/s.*?, yaw rate {NUM} rad/s")
PAIR = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
COMMANDS = ("left", "right", "straight", "unknown")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_numbers(item: Dict[str, Any]) -> Tuple[float, float, float, List[float]]:
    """Speed, acceleration, yaw rate and the 6 past coordinates, as the prompt states them."""
    v, a, w = map(float, STATE.search(item["prompt"]).groups())
    line = next(l for l in item["prompt"].split("\n") if l.startswith("Past positions"))
    past = [float(x) for pair in PAIR.findall(line.split(":", 1)[1]) for x in pair]
    return v, a, w, past


def kinematic(v: float, a: float, w: float) -> np.ndarray:
    """Constant acceleration and yaw rate, integrated at 0.05 s; speed floored at 0."""
    out, x, y, h, s, dt = [], 0.0, 0.0, 0.0, v, 0.05
    for k in range(1, 81):
        x += s * math.cos(h) * dt
        y += s * math.sin(h) * dt
        h += w * dt
        s = max(0.0, s + a * dt)
        if k % 10 == 0:
            out.append([x, y])
    return np.array(out)


def feature_row(item: Dict[str, Any]) -> List[float]:
    v, a, w, past = prompt_numbers(item)
    return [v, a, w, v * w, v * v] + past + [1.0 if item["meta"]["command"] == c else 0.0 for c in COMMANDS]


Predictor = Callable[[List[Dict[str, Any]]], np.ndarray]


def fit_linear(train: List[Dict[str, Any]]) -> Predictor:
    X = np.array([feature_row(it) for it in train])
    Y = np.array([np.ravel(it["target_xy"]) for it in train])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    A = np.c_[(X - mu) / sd, np.ones(len(X))]
    W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ Y)
    return lambda items: (np.c_[(np.array([feature_row(it) for it in items]) - mu) / sd, np.ones(len(items))] @ W).reshape(-1, 8, 2)


def fit_mlp(train: List[Dict[str, Any]], seeds: int = 3, epochs: int = 150) -> Predictor:
    X = np.array([feature_row(it) for it in train])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(np.array([np.ravel(it["target_xy"]) for it in train]), dtype=torch.float32)
    nets = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                                  torch.nn.ReLU(), torch.nn.Linear(256, 16))
        optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 256):
                b = perm[i : i + 256]
                loss = (net(xt[b]) - yt[b]).abs().mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
        nets.append(net.eval())

    def predict(items: List[Dict[str, Any]]) -> np.ndarray:
        x = torch.tensor((np.array([feature_row(it) for it in items]) - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            return np.mean([net(x).numpy() for net in nets], axis=0).reshape(-1, 8, 2)

    return predict


def fit_command_mean(train: List[Dict[str, Any]]) -> Callable[[List[Dict[str, Any]]], np.ndarray]:
    paths: Dict[str, List[Any]] = {}
    for it in train:
        paths.setdefault(it["meta"]["command"], []).append(it["target_xy"])
    mean = {c: np.mean(v, axis=0) for c, v in paths.items()}
    overall = np.mean([p for v in paths.values() for p in v], axis=0)
    return lambda items: np.array([mean.get(it["meta"]["command"], overall) for it in items])


def errors(pred: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
    """(n, 8) distance in metres to the recorded path."""
    return np.linalg.norm(np.asarray(pred) - np.array([it["target_xy"] for it in items]), axis=-1)


def summary(err: np.ndarray, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    commands = np.array([it["meta"]["command"] for it in items])
    return {"n": len(err), "ade_m": round(float(err.mean()), 3), "fde_m": round(float(err[:, -1].mean()), 3),
            "err_at_s": {f"{t:g}": round(float(err[:, k].mean()), 3) for k, t in enumerate(TARGET_S)},
            "ade_by_command": {c: round(float(err[commands == c].mean()), 3) for c in COMMANDS if (commands == c).any()}}


def dev_items(view: str, split: str, samples: int, seed: int) -> List[Dict[str, Any]]:
    """The subset ``evaluate.py`` scores for the same arguments."""
    items = read_jsonl(os.path.join(view, f"{split}.jsonl"))
    return random.Random(seed).sample(items, min(samples, len(items))) if samples else items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--samples", type=int, default=500, help="as in evaluate.py; 0 = all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    items = dev_items(args.view, args.split, args.samples, args.seed)
    train = read_jsonl(os.path.join(args.view, "train.jsonl"))
    preds = {
        "constant_velocity": np.array([[[it["meta"]["speed"] * t, 0.0] for t in TARGET_S] for it in items]),
        "kinematic": np.array([kinematic(*prompt_numbers(it)[:3]) for it in items]),
        "linear": fit_linear(train)(items),
        "mlp": fit_mlp(train)(items),
        "command_mean": fit_command_mean(train)(items),
    }
    result = {"view": args.view, "split": args.split, "samples": len(items), "seed": args.seed,
              "baselines": {name: summary(errors(p, items), items) for name, p in preds.items()}}
    for name, s in result["baselines"].items():
        print(f"{name:18s} ADE {s['ade_m']:.3f}  FDE {s['fde_m']:.3f}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
