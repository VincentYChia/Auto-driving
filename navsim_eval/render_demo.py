"""Demonstration figures: several models' planned paths on the same navtest scenes, for showing, not scoring.

Each figure has two panels: the front camera at the planning moment, and NAVSIM's bird's-eye view (map,
other agents, ego) with the human driver's recorded 4 s path and each model's planned 4 s path, labelled
with that model's PDMS on the scene (from its NAVSIM score CSV). NAVSIM scores one 4-second plan per
scene (non-reactive), so a figure shows one plan, not continuous driving.

Scenes are chosen reproducibly from the scenes every model has a prediction and a score for: a seeded
random sample (representative), plus labelled extras (the reference model far ahead of the second, the
second ahead of the reference, the reference failing, a left-turn and a right-turn command).

    python navsim_eval/render_demo.py --out <dir> \\
        --model "0.8B pilot=<predictions dir>=<score csv>" --model "MVLAD-AD=<...>=<...>" ...

Run in the NAVSIM env on Anvil (needs OPENSCENE_DATA_ROOT with navsim_logs/test and sensor_blobs/test,
and the nuPlan maps). The images show nuPlan data: keep them out of the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim_eval.predictions import load_predictions  # noqa: E402

# model colours from the light chart palette, skipping blue, yellow and red: NAVSIM's BEV already uses them for
# vehicles, static objects and the ego car. Human path in ink, drawn on top; constant velocity in grey dots.
SERIES = ["#1baf7a", "#eb6834", "#6250d6", "#e87ba4", "#008300"]
GREY = "#52514e"
INK = "#0b0b0b"
COMMANDS = ["left", "straight", "right", "unknown"]


def load_scores(path: str):
    import pandas as pd

    d = pd.read_csv(path)
    return d[d.token != "average"].set_index("token")


def failure(row) -> str:
    """The sub-score that zeroed this scene, in words, or '' when the scene scored above 0."""
    if float(row["score"]) > 0:
        return ""
    reasons = [("no_at_fault_collisions", "collision"), ("drivable_area_compliance", "left the road"),
               ("driving_direction_compliance", "wrong direction")]
    hit = [text for col, text in reasons if col in row and float(row[col]) == 0]
    return ", ".join(hit) or "score 0"


def choose(scores: List, n_random: int, seed: int) -> List[Tuple[str, str]]:
    """(token, why) pairs; scores[0] is the reference model, scores[1] the comparison."""
    common = sorted(set.intersection(*(set(s.index) for s in scores)))
    rng = np.random.default_rng(seed)
    picked: List[Tuple[str, str]] = [(t, "random sample") for t in rng.choice(common, n_random, replace=False)]
    used = {t for t, _ in picked}
    a = scores[0].loc[common, "score"].astype(float)
    b = scores[1].loc[common, "score"].astype(float)

    def extra(pool, why, k):
        pool = [t for t in pool if t not in used]
        for t in rng.choice(pool, min(k, len(pool)), replace=False) if pool else []:
            picked.append((t, why))
            used.add(t)

    extra(list(a[(a >= 0.8) & (b == 0)].index), "reference model far ahead", 3)
    extra(list((b - a)[(b - a) > 0.3].index), "comparison model ahead", 2)
    extra(list(a[a == 0].index), "reference model fails", 2)
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--model", action="append", required=True,
                   help='"name=predictions dir or file=score csv"; the first is the reference, the second the comparison')
    p.add_argument("--cv", default=None, help="score csv of NAVSIM's constant-velocity agent: also draw its path "
                   "(the current speed held straight ahead, as that agent plans)")
    p.add_argument("--random", type=int, default=6)
    p.add_argument("--turns", type=int, default=1, help="extra scenes per turn command (left, right)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax

    models = []
    for spec in args.model:
        name, pred, csv = spec.split("=", 2)
        models.append({"name": name, "pred": load_predictions(pred), "score": load_scores(csv)})
    picked = choose([m["score"] for m in models], args.random, args.seed)
    cv = load_scores(args.cv) if args.cv else None

    data = Path(os.environ["OPENSCENE_DATA_ROOT"])
    no_cam = dict(cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False,
                  cam_b0=False, lidar_pc=False)
    # turn commands need the agent input, so pick them after loading a pool of candidate scenes
    pool_tokens = [t for t, _ in picked]
    if args.turns:
        common = sorted(set.intersection(*(set(m["score"].index) for m in models)))
        rng = np.random.default_rng(args.seed + 1)
        pool_tokens += [t for t in rng.choice(common, 400, replace=False) if t not in pool_tokens]
    loader = SceneLoader(data / "navsim_logs" / "test", data / "sensor_blobs" / "test",
                         SceneFilter(tokens=pool_tokens, frame_interval=1), SensorConfig(cam_f0=[3], **no_cam))
    available = set(loader.tokens)  # frame_interval=1: navtest tokens sit on every frame, not a 14-frame grid
    print(f"[render] {len(available)} of {len(pool_tokens)} candidate scenes loaded", flush=True)
    picked = [(t, why) for t, why in picked if t in available]

    def command_of(scene) -> str:
        c = np.asarray(scene.get_agent_input().ego_statuses[-1].driving_command).reshape(-1)
        return COMMANDS[int(np.argmax(c))] if c.size == 4 and c.max() > 0 else "unknown"

    if args.turns:
        want = {"left": args.turns, "right": args.turns}
        for t in pool_tokens[len(picked):]:
            if t not in available or t in {x for x, _ in picked} or not any(want.values()):
                continue
            cmd = command_of(loader.get_scene_from_token(t))
            if want.get(cmd, 0) > 0:
                picked.append((t, f"{cmd}-turn command"))
                want[cmd] -= 1

    os.makedirs(args.out, exist_ok=True)
    index = []
    for k, (token, why) in enumerate(picked):
        scene = loader.get_scene_from_token(token)
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.55, 1]})
        fig.patch.set_facecolor("white")
        image = frame.cameras.cam_f0.image
        if image is not None:
            ax_cam.imshow(image)
        configure_ax(ax_cam)
        ax_cam.set_title("Front camera at the planning moment", fontsize=10, color=INK, loc="left")

        add_configured_bev_on_ax(ax_bev, scene.map_api, frame)
        human = scene.get_future_trajectory(8).poses[:, :2]  # the 4 s the models plan
        line = np.vstack([[0, 0], human])
        ax_bev.plot(line[:, 1], line[:, 0], color=INK, linestyle="--", linewidth=1.6, zorder=20,
                    label="Human driver (recorded)")
        entry = {"token": token, "why": why, "command": command_of(scene), "map": scene.scene_metadata.map_name,
                 "models": {}}
        for i, m in enumerate(models):
            wp = m["pred"].get(token)
            row = m["score"].loc[token]
            pdms, why_zero = float(row["score"]), failure(row)
            entry["models"][m["name"]] = {"pdms": round(pdms, 3), "zeroed_by": why_zero or None}
            if wp is None:
                continue
            path = np.vstack([[0, 0], np.asarray(wp, dtype=float)[:8, :2]])
            label = f"{m['name']}: PDMS {pdms:.2f}" + (f" ({why_zero})" if why_zero else "")
            ax_bev.plot(path[:, 1], path[:, 0], color=SERIES[i % len(SERIES)], linewidth=3.0 if i == 0 else 2.2,
                        marker="o", markersize=4 if i == 0 else 3, zorder=15 - i, label=label)  # first model on top
        if cv is not None and token in cv.index:
            v = np.asarray(scene.get_agent_input().ego_statuses[-1].ego_velocity, dtype=float)[:2]
            t = 0.5 * np.arange(1, 9)
            path = np.vstack([[0, 0], np.stack([np.linalg.norm(v) * t, 0 * t], axis=1)])
            row = cv.loc[token]
            pdms, why_zero = float(row["score"]), failure(row)
            entry["models"]["Constant velocity"] = {"pdms": round(pdms, 3), "zeroed_by": why_zero or None}
            label = f"Constant velocity: PDMS {pdms:.2f}" + (f" ({why_zero})" if why_zero else "")
            ax_bev.plot(path[:, 1], path[:, 0], color=GREY, linewidth=1.8, linestyle=":", zorder=4, label=label)
        configure_bev_ax(ax_bev)
        configure_ax(ax_bev)
        ax_bev.set_title("Bird's-eye view, 4 s plans (ego at centre, facing up)", fontsize=10, color=INK, loc="left")
        ax_bev.legend(loc="lower left", fontsize=8, framealpha=0.92)
        fig.suptitle(f"navtest scene {token[:8]}  ·  {scene.scene_metadata.map_name}  ·  route command: "
                     f"{entry['command']}  ·  {why}", fontsize=11, color=INK, x=0.01, ha="left")
        fig.text(0.01, 0.01, "PDMS: NAVSIM's 0-1 driving score for this 4 s plan (collisions, staying on the road, "
                 "safe distance, progress, comfort); 1 is perfect.", fontsize=8, color="#52514e")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        name = f"{k + 1:02d}_{token[:8]}.png"
        fig.savefig(os.path.join(args.out, name), dpi=args.dpi, facecolor="white")
        plt.close(fig)
        entry["file"] = name
        index.append(entry)
        print(f"[render] {name}: {why}", flush=True)
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"models": [m["name"] for m in models], "scenes": index}, f, indent=1)


if __name__ == "__main__":
    main()
