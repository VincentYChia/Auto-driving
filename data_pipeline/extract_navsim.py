"""Extract canonical training samples from NAVSIM logs (run in the NAVSIM env; CPU only).

Needs only the log pickles (``navsim_logs/<data_split>``, ~14 GB for trainval), not maps
or sensor data: images are stored as paths. Logs are processed in batches by a process pool;
each batch becomes one gzipped JSONL shard, written atomically, so an interrupted run
resumes where it stopped.

    python -m data_pipeline.extract_navsim --split navtrain --output-dir $DATA_ROOT/canonical/navtrain --workers 16
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import json
import multiprocessing as mp
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from data_pipeline import schema
from navsim_tools import loader as nav_loader
from navsim_tools import raw_frames

EXTRACTOR_VERSION = "0.1.0"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _navsim_version() -> str:
    try:
        from importlib.metadata import version

        return version("navsim")
    except Exception:
        return "unknown"


def _check_against_navsim(scene_loader, token: str, sample: Dict[str, Any], current_idx: int) -> None:
    """Fail loudly if our geometry disagrees with the devkit on this NAVSIM version."""
    from nuplan.common.actor_state.state_representation import StateSE2

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    agent_input = scene_loader.get_agent_input_from_token(token)
    navsim_hist = np.stack([np.asarray(s.ego_pose, dtype=np.float64) for s in agent_input.ego_statuses])
    ours_hist = np.asarray(sample["ego"]["history_poses"])
    frames = scene_loader.scene_frames_dicts[token]
    globals_ = np.stack([raw_frames.global_ego_pose(f) for f in frames])
    navsim_future = convert_absolute_to_relative_se2_array(
        StateSE2(*globals_[current_idx]), globals_[current_idx + 1 : current_idx + 1 + len(sample["target"]["future_poses"])]
    )
    ours_future = np.asarray(sample["target"]["future_poses"])
    gap = max(np.abs(navsim_hist[:, :2] - ours_hist[:, :2]).max(), np.abs(navsim_future[:, :2] - ours_future[:, :2]).max())
    if gap > 2e-3:
        raise RuntimeError(f"{token}: geometry differs from NAVSIM by {gap:.4f} m")


def _process_batch(job: Dict[str, Any]) -> Dict[str, Any]:
    out_path = Path(job["out_dir"]) / f"samples-{job['batch_id']:05d}.jsonl.gz"
    if out_path.exists() and not job["overwrite"]:
        return {"batch_id": job["batch_id"], "file": out_path.name, "status": "exists"}
    scene_filter = copy.deepcopy(job["scene_filter"])
    scene_filter.log_names = job["log_names"]
    scene_loader = nav_loader.make_scene_loader(Path(job["log_path"]), Path(job["sensor_path"]), scene_filter)
    current_idx = scene_filter.num_history_frames - 1
    num_future = scene_filter.num_future_frames
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    count, errors = 0, []
    with gzip.open(tmp_path, "wt") as f:
        for i, token in enumerate(sorted(scene_loader.tokens)):
            frames = scene_loader.scene_frames_dicts[token]
            try:
                sample = schema.build_sample(
                    frames, current_idx, num_future, job["split"], job["data_split"], job["max_range_m"], job["provenance"]
                )
                schema.validate_sample(sample)
                if i == 0:
                    _check_against_navsim(scene_loader, token, sample, current_idx)
            except (ValueError, KeyError) as exc:
                errors.append({"token": token, "error": repr(exc)})
                continue
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")
            count += 1
    os.replace(tmp_path, out_path)
    return {"batch_id": job["batch_id"], "file": out_path.name, "status": "written", "samples": count,
            "logs": len(job["log_names"]), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="navtrain", help="NAVSIM split; navtrain for training data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--navsim-log-path", default=None, help="default: $OPENSCENE_DATA_ROOT/navsim_logs/<data_split>")
    parser.add_argument("--sensor-blobs-path", default=None, help="default: $OPENSCENE_DATA_ROOT/sensor_blobs/<data_split>")
    parser.add_argument("--max-range-m", type=float, default=100.0, help="keep objects that come within this range")
    parser.add_argument("--logs-per-batch", type=int, default=20)
    parser.add_argument("--max-logs", type=int, default=None, help="for quick tests")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.split in ("navtest", "navhard_two_stage", "warmup_two_stage", "private_test_hard_two_stage"):
        raise SystemExit(f"{args.split} is a NAVSIM test split; it must not be used for training data")

    scene_filter, data_split = nav_loader.load_split(args.split)
    log_path, sensor_path = nav_loader.default_paths(data_split, args.navsim_log_path, args.sensor_blobs_path)
    available = {p.name[: -len(".pkl")] for p in log_path.glob("*.pkl")}
    wanted = list(scene_filter.log_names) if scene_filter.log_names else sorted(available)
    log_names = sorted(set(wanted) & available)
    if args.max_logs:
        log_names = log_names[: args.max_logs]
    missing = len(set(wanted) - available)
    print(f"{len(log_names)} logs to process ({missing} listed by the {args.split} filter are not on disk)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "extractor": "data_pipeline.extract_navsim",
        "extractor_version": EXTRACTOR_VERSION,
        "git_commit": _git_commit(),
        "navsim_version": _navsim_version(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    jobs: List[Dict[str, Any]] = []
    for batch_id, start in enumerate(range(0, len(log_names), args.logs_per_batch)):
        jobs.append(
            {
                "batch_id": batch_id, "log_names": log_names[start : start + args.logs_per_batch],
                "scene_filter": scene_filter, "log_path": str(log_path), "sensor_path": str(sensor_path),
                "out_dir": str(out_dir), "split": args.split, "data_split": data_split,
                "max_range_m": args.max_range_m, "provenance": provenance, "overwrite": args.overwrite,
            }
        )
    results = []
    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_process_batch, jobs):
            results.append(result)
            print(f"  batch {result['batch_id']}: {result['status']} {result.get('samples', '')}", flush=True)

    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "split": args.split,
        "data_split": data_split,
        "sensor_root_hint": str(sensor_path),
        "provenance": provenance,
        "args": vars(args),
        "batches": sorted(results, key=lambda r: r["batch_id"]),
        "total_samples": sum(r.get("samples", 0) for r in results),
        "total_errors": sum(len(r.get("errors", [])) for r in results),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done: {manifest['total_samples']} samples, {manifest['total_errors']} errors -> {out_dir}")


if __name__ == "__main__":
    main()
