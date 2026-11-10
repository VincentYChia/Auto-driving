"""Where NAVSIM scores navtest scenes zero, and how far each model's plan lies from the human driver's path.

Joins per-scene NAVSIM v1.1 score CSVs and saved waypoints with each scene's recorded human path and agent
input (driving command, speed, map), and writes aggregate tables only, never per-scene rows:

* the share of scenes scored zero, by cause (collision, left the drivable area, both);
* PDMS, zero share and distance from the human path, by driving command, ego speed, what the human did over
  the next 4 s (turned, stopped, started, shifted sideways, kept going) and map (city);
* error against the human path (ADE, FDE, along-track and cross-track error at 4 s) on zero and non-zero scenes;
* how often models fail on the same scenes, and on scenes where NAVSIM's human agent (the recorded path
  itself) also scores zero, when its score CSV is given;
* with --view, the same groups' shares in training views (pilot_v1 jsonl, read as files), next to navtest's.

    python navsim_eval/failure_analysis.py --out <json> --cache <npz> \\
        --model "0.8B pilot=<predictions>=<score csv>" --model "MVLAD-AD=<...>=<...>" \\
        [--cv <constant-velocity score csv>] [--human <human-agent score csv>]

Run in the NAVSIM env on Anvil (OPENSCENE_DATA_ROOT with navsim_logs/test, and the nuPlan maps). The first
run reads every scored scene's human path and agent input (no sensors) into --cache, which stays on Anvil
(nuPlan-derived); later runs reuse it. The JSON holds aggregates only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim_eval.predictions import load_predictions  # noqa: E402
# The maneuver labels and speed bands are shared with the training data (navsim_tools/maneuvers.py).
from navsim_tools.maneuvers import SPEED_EDGES, SPEED_NAMES, TURN_RAD, maneuver, speed_band  # noqa: E402,F401

COMMANDS = ["left", "straight", "right", "unknown"]
T = 0.5 * np.arange(1, 9)  # the 8 planned poses, 0.5 s apart


def load_scores(path: str):
    import pandas as pd

    d = pd.read_csv(path)
    return d[d.token != "average"].set_index("token")


def read_scenes(tokens: List[str], cache: str) -> Dict[str, np.ndarray]:
    """Human future poses, current speed, command and map for each token; cached as npz."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    data = Path(os.environ["OPENSCENE_DATA_ROOT"])
    loader = SceneLoader(data / "navsim_logs" / "test", data / "sensor_blobs" / "test",
                         SceneFilter(tokens=list(tokens), frame_interval=1), SensorConfig.build_no_sensors())
    got = [t for t in tokens if t in set(loader.tokens)]
    print(f"[failure] {len(got)} of {len(tokens)} scenes loaded", flush=True)
    human, speed, accel, command, maps = [], [], [], [], []
    for i, t in enumerate(got):
        scene = loader.get_scene_from_token(t)
        human.append(scene.get_future_trajectory(8).poses[:8, :3])
        status = scene.get_agent_input().ego_statuses[-1]
        speed.append(float(np.hypot(*np.asarray(status.ego_velocity, dtype=float)[:2])))
        accel.append(float(np.asarray(status.ego_acceleration, dtype=float)[0]))
        c = np.asarray(status.driving_command).reshape(-1)
        command.append(int(np.argmax(c)) if c.size == 4 and c.max() > 0 else 3)
        maps.append(scene.scene_metadata.map_name)
        if (i + 1) % 2000 == 0:
            print(f"[failure] {i + 1} scenes read", flush=True)
    out = {"tokens": np.array(got), "human": np.array(human, dtype=np.float64), "speed": np.array(speed),
           "accel": np.array(accel), "command": np.array(command, dtype=np.int64), "map": np.array(maps)}
    if cache:
        np.savez_compressed(cache, **out)
    return out


def errors(pred: np.ndarray, human: np.ndarray) -> Dict[str, np.ndarray]:
    """ADE, FDE and the 4 s endpoint error split along and across the human path's final heading."""
    d = pred[:, :8, :2] - human[:, :8, :2]
    ade = np.linalg.norm(d, axis=2).mean(axis=1)
    end = d[:, -1]
    psi = human[:, -1, 2]
    along = end[:, 0] * np.cos(psi) + end[:, 1] * np.sin(psi)  # + means the plan ends ahead of the human
    across = -end[:, 0] * np.sin(psi) + end[:, 1] * np.cos(psi)  # + means left of the human
    return {"ade": ade, "fde": np.linalg.norm(end, axis=1), "along": along, "across": across}


def read_view(path: str) -> Dict[str, np.ndarray]:
    """A training view's targets as human paths (pilot_v1 jsonl rows: target_xy, target_yaw, meta), read as a file."""
    human, speed, command, maps = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            human.append(np.column_stack([np.asarray(r["target_xy"], float), np.asarray(r["target_yaw"], float)]))
            speed.append(float(r["meta"]["speed"]))
            command.append(str(r["meta"]["command"]))
            maps.append(str(r["meta"]["city"]))
    human_a, speed_a = np.array(human, dtype=np.float64), np.array(speed)
    return {"command": np.array(command, dtype=object), "ego_speed": speed_band(speed_a),
            "human_next_4s": maneuver(human_a, speed_a), "map": np.array(maps, dtype=object)}


def shares(groups: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    return {g: {k: round(float((v == k).mean()), 4) for k in sorted(set(v))} for g, v in groups.items()}


def summary(score: np.ndarray, err: Dict[str, np.ndarray], mask: np.ndarray, nc0: np.ndarray,
            dac0: np.ndarray) -> Dict[str, float]:
    n = int(mask.sum())
    if n == 0:
        return {"scenes": 0}
    r = lambda x: round(float(x), 4)  # noqa: E731
    return {"scenes": n, "pdms": r(score[mask].mean()), "share_zero": r((score[mask] == 0).mean()),
            "share_collision": r(nc0[mask].mean()), "share_off_road": r(dac0[mask].mean()),
            "ade_m": r(err["ade"][mask].mean()), "median_ade_m": r(np.median(err["ade"][mask])),
            "fde_m": r(err["fde"][mask].mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True, help="npz of human paths and agent inputs (kept on Anvil)")
    p.add_argument("--model", action="append", required=True, help='"name=predictions dir or file=score csv"')
    p.add_argument("--cv", default=None, help="score csv of NAVSIM's constant-velocity agent (its path is rebuilt "
                   "from the current speed, as that agent plans)")
    p.add_argument("--human", default=None, help="score csv of NAVSIM's human agent")
    p.add_argument("--view", action="append", default=[],
                   help='"name=<pilot_v1 jsonl>": also give that training view\'s mix over the same groups')
    args = p.parse_args()

    models = []
    for spec in args.model:
        name, pred, csv = spec.split("=", 2)
        models.append({"name": name, "pred": load_predictions(pred), "score": load_scores(csv)})
    if args.cv:
        models.append({"name": "constant velocity", "pred": None, "score": load_scores(args.cv)})
    common = sorted(set.intersection(*(set(m["score"].index) for m in models)))
    scenes = read_scenes(common, args.cache)
    tokens = [str(t) for t in scenes["tokens"]]
    keep = np.array([t in set(common) for t in tokens])
    tokens = [t for t, k in zip(tokens, keep) if k]
    human, speed = scenes["human"][keep], scenes["speed"][keep]
    groups = {"command": np.array(COMMANDS, dtype=object)[scenes["command"][keep]],
              "ego_speed": speed_band(speed), "human_next_4s": maneuver(human, speed),
              "map": scenes["map"][keep].astype(object)}
    human_zero = None
    if args.human:
        h = load_scores(args.human)
        human_zero = np.array([float(h.loc[t, "score"]) == 0 if t in h.index else False for t in tokens])

    out = {"what": "NAVSIM v1.1 navtest: where scenes score zero and how far plans lie from the human path",
           "evidence": "V (measured on Anvil)", "scenes": len(tokens),
           "definitions": {
               "share_zero": "share of scenes with PDMS 0 (a collision or leaving the drivable area zeroes it)",
               "share_collision": "share of scenes with no_at_fault_collisions = 0",
               "share_off_road": "share of scenes with drivable_area_compliance = 0",
               "ade_m": "mean distance from the human path over the 8 poses (0.5-4 s), metres",
               "along_m / across_m": "4 s endpoint error along / across the human path's final heading: along > 0 "
                                     "ends ahead of the human (too fast), across > 0 ends left of it",
               "human_next_4s": f"from the recorded path, in priority order: turn (heading change > {TURN_RAD} rad), "
                                "stay stopped / start from a stop (speed now < 0.5 m/s; moves < or >= 1 m), brake to "
                                "a stop (>= 2 m/s now, < 0.5 m/s at 4 s), sideways shift (> 2 m lateral), keep going"},
           "group_sizes": {g: {k: int((v == k).sum()) for k in sorted(set(v))} for g, v in groups.items()},
           "models": {}}
    zero_sets = {}
    for m in models:
        s = m["score"].loc[tokens]
        score = s["score"].astype(float).values
        nc0 = s["no_at_fault_collisions"].astype(float).values == 0
        dac0 = s["drivable_area_compliance"].astype(float).values == 0
        if m["pred"] is None:  # constant velocity: the current speed held straight ahead
            pred = np.stack([np.stack([speed * t, np.zeros_like(speed)], axis=1) for t in T], axis=1)
        else:
            missing = [t for t in tokens if t not in m["pred"]]
            if missing:
                raise SystemExit(f"{m['name']}: {len(missing)} scenes without predictions")
            pred = np.stack([m["pred"][t][:8] for t in tokens])
        err = errors(pred, human)
        zero = score == 0
        zero_sets[m["name"]] = zero
        r = lambda x: round(float(x), 4)  # noqa: E731
        entry = {"all": summary(score, err, np.ones(len(tokens), bool), nc0, dac0),
                 "zero_causes": {"collision only": r((nc0 & ~dac0).mean()), "off road only": r((dac0 & ~nc0).mean()),
                                 "both": r((nc0 & dac0).mean()), "other": r((zero & ~nc0 & ~dac0).mean())},
                 "error_on_zero_vs_rest": {
                     which: {"ade_m": r(err["ade"][mask].mean()), "fde_m": r(err["fde"][mask].mean()),
                             "along_m_mean": r(err["along"][mask].mean()),
                             "along_m_abs": r(np.abs(err["along"][mask]).mean()),
                             "across_m_abs": r(np.abs(err["across"][mask]).mean()),
                             "share_ending_ahead_by_over_2m": r((err["along"][mask] > 2).mean()),
                             "share_ending_behind_by_over_2m": r((err["along"][mask] < -2).mean()),
                             "share_ending_aside_by_over_1m": r((np.abs(err["across"][mask]) > 1).mean())}
                     for which, mask in (("zero", zero), ("rest", ~zero)) if mask.any()},
                 "by": {g: {k: summary(score, err, v == k, nc0, dac0) for k in sorted(set(v))}
                        for g, v in groups.items()}}
        if human_zero is not None:
            entry["zero_where_human_agent_also_zero"] = r((zero & human_zero).sum() / max(zero.sum(), 1))
        out["models"][m["name"]] = entry
    names = list(zero_sets)
    out["shared_zero_scenes"] = {f"{a} & {b}": {"both_zero": int((zero_sets[a] & zero_sets[b]).sum()),
                                                f"share_of_{a}_zeros": round(float((zero_sets[a] & zero_sets[b]).sum()
                                                                                   / max(zero_sets[a].sum(), 1)), 4)}
                                 for i, a in enumerate(names) for b in names[i + 1:]}
    if human_zero is not None:
        out["human_agent_zero_scenes"] = int(human_zero.sum())
    if args.view:
        out["mix"] = {"navtest": shares(groups)}
        for spec in args.view:
            name, path = spec.split("=", 1)
            out["mix"][name] = shares(read_view(path))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("scenes", "group_sizes")}, indent=1))
    for name, e in out["models"].items():
        print(name, json.dumps(e["all"]), json.dumps(e["zero_causes"]))


if __name__ == "__main__":
    main()
