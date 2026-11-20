"""NAVSIM v1.1 PDM scores for many candidate paths: the RL stage's reward (docs/HEADS_PLAN.md phase 4).

Runs in the navsim env (Python 3.9; ``PYTHONPATH`` = the repo). Each candidate is scored exactly as NAVSIM's
``run_pdm_score.py`` scores a submission:
- alone, as proposal 1 next to the PDM-Closed reference (proposal 0). Scoring several candidates together would
  change the progress normalisation;
- with the simulator and scorer built from NAVSIM's own ``default_scoring_parameters.yaml``.

A scene's metric cache is loaded once per request, however many candidates it holds.

    python pilot_sft/heads/pdm_server.py serve --cache <metric cache> --workers 12 --port 47411
    python pilot_sft/heads/pdm_server.py check --cache <metric cache> --predictions <dir> --csv <official scores>

``check`` re-scores saved predictions and compares every scene with NAVSIM's own CSV, then reports how many paths
per second one process and the pool score. The client is ``pdm_client.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import asdict
from multiprocessing import Pool
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

AUTHKEY = b"heads-pdm"
_W = {}  # per-worker state: simulator, scorer, cache paths, loaded caches


def scoring_parts():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    import navsim

    cfg = OmegaConf.load(os.path.join(os.path.dirname(navsim.__file__),
                                      "planning/script/config/pdm_scoring/default_scoring_parameters.yaml"))
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling
    return simulator, scorer


def init_worker(cache_dir: str, lru: int = 64) -> None:
    from navsim.common.dataloader import MetricCacheLoader

    _W["sim"], _W["scorer"] = scoring_parts()
    _W["paths"] = MetricCacheLoader(Path(cache_dir)).metric_cache_paths
    _W["caches"], _W["lru"] = OrderedDict(), lru


def metric_cache(token: str):
    c = _W["caches"]
    if token in c:
        c.move_to_end(token)
        return c[token]
    with lzma.open(_W["paths"][token], "rb") as f:
        mc = pickle.load(f)
    c[token] = mc
    if len(c) > _W["lru"]:
        c.popitem(last=False)
    return mc


def score_item(item):
    """(token, [8 x 2 waypoints, ...]) -> one dict per candidate: score and sub-scores, or score 0 with an error."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim_eval.predictions import poses_from_waypoints

    token, paths = item
    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    try:
        mc = metric_cache(token)
    except Exception as e:  # noqa: BLE001
        return [{"score": 0.0, "error": f"metric cache: {e}"[:200]} for _ in paths]
    out = []
    for wp in paths:
        try:
            traj = Trajectory(poses_from_waypoints(np.asarray(wp, dtype=np.float64), 8), sampling)
            r = asdict(pdm_score(metric_cache=mc, model_trajectory=traj, future_sampling=_W["sim"].proposal_sampling,
                                 simulator=_W["sim"], scorer=_W["scorer"]))
            out.append({k: float(v) for k, v in r.items()})
        except Exception as e:  # noqa: BLE001
            out.append({"score": 0.0, "error": str(e)[:200]})
    return out


def serve(args) -> None:
    with Pool(args.workers, initializer=init_worker, initargs=(args.cache,)) as pool:
        with Listener(("127.0.0.1", args.port), authkey=AUTHKEY) as listener:
            print(json.dumps({"serving": args.port, "workers": args.workers, "cache": args.cache}), flush=True)
            while True:
                conn = listener.accept()
                try:
                    while True:
                        msg = conn.recv()
                        if msg.get("op") == "close":
                            break
                        t0 = time.time()
                        res = pool.map(score_item, msg["items"], chunksize=max(1, len(msg["items"]) // (4 * args.workers)))
                        conn.send({"results": res, "seconds": time.time() - t0})
                except EOFError:
                    pass
                finally:
                    conn.close()


def check(args) -> None:
    from navsim_eval.predictions import load_predictions

    preds = load_predictions(args.predictions)
    with open(args.csv, newline="") as f:
        official = {r["token"]: r for r in csv.DictReader(f) if r["token"] != "average"}
    tokens = sorted(set(preds) & set(official))[: args.limit]
    init_worker(args.cache)
    t0 = time.time()
    mine = {t: score_item((t, [preds[t]]))[0] for t in tokens}
    one = time.time() - t0
    keys = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
            "comfort", "driving_direction_compliance", "score"]
    worst = {k: max(abs(mine[t][k] - float(official[t][k])) for t in tokens) for k in keys}
    with Pool(args.workers, initializer=init_worker, initargs=(args.cache,)) as pool:
        t0 = time.time()
        pool.map(score_item, [(t, [preds[t]] * args.k) for t in tokens], chunksize=4)
        many = time.time() - t0
    print(json.dumps({"scenes": len(tokens), "max_abs_difference_vs_official": worst,
                      "errors": sum(1 for t in tokens if "error" in mine[t]),
                      "one_process_paths_per_s": round(len(tokens) / one, 1),
                      f"pool_{args.workers}_paths_per_s_at_k{args.k}": round(len(tokens) * args.k / many, 1)}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--cache", required=True)
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--port", type=int, default=47411)
    c = sub.add_parser("check")
    c.add_argument("--cache", required=True)
    c.add_argument("--predictions", required=True)
    c.add_argument("--csv", required=True)
    c.add_argument("--limit", type=int, default=500)
    c.add_argument("--workers", type=int, default=12)
    c.add_argument("--k", type=int, default=8, help="candidates per scene in the pool timing")
    args = p.parse_args()
    serve(args) if args.cmd == "serve" else check(args)


if __name__ == "__main__":
    main()
