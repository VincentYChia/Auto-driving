"""Run a pilot on NAVSIM scenes and save its waypoints for NAVSIM's PDM scorer (pilot env, one GPU per shard).

    python pilot_sft/predict_navsim.py --inputs $NAVSIM_EXP_ROOT/mvlad/navtest_inputs.json \\
        --model Qwen/Qwen3.5-2B --adapter <run>/adapter --out <run>/navtest_predictions --shard 0 --num-shards 8

Inputs are the records ``navsim_eval/export_mvlad_inputs.py`` writes (read as a file: no import across
workstreams). Each holds NAVSIM's AgentInput for one scene: 4 ego poses, velocity, acceleration and
the driving command, plus the CAM_F0 path. From them this builds the ``pilot_v1.1`` prompt exactly as
``data_pipeline/v1/pilot_view.py`` does for training, and the image as the view does (640 px wide,
JPEG quality 90).

Each shard appends ``{"token", "waypoints", "answer"}`` lines to ``<out>/shard_XXX_of_YYY.jsonl``, the
format ``navsim_eval.predictions.load_predictions`` reads; re-running skips scenes already done. An
answer without 8 "(x, y)" pairs gets the constant-velocity path, marked ``"fallback": true`` and
counted, so every scene is scored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # data_pipeline, from pilot_sft/

from data_pipeline import labels  # noqa: E402
from data_pipeline.v1 import pilot_view  # noqa: E402


def item(record: Dict[str, Any]) -> Dict[str, Any]:
    """A pilot_v1.1 sample from one exported NAVSIM record (image relative to ``<out>``)."""
    nav = record["navsim"]
    history = nav["ego_history_local"]  # (x, y, heading) at -1.5, -1.0, -0.5, 0 s; the last is the origin
    if len(history) != 4:
        raise ValueError(f"{record['id']}: {len(history)} history poses, the prompt needs 4")
    turn = history[3][2] - history[2][2]
    yaw_rate = math.atan2(math.sin(turn), math.cos(turn)) / 0.5  # pilot_view.past_yaw_rate
    vx, vy = nav["ego_velocity"][:2]
    prompt = pilot_view.PROMPT.format(
        speed=math.hypot(vx, vy), accel=round(float(nav["ego_acceleration"][0]), 1) + 0.0,
        yaw_rate=round(yaw_rate, 2) + 0.0,
        history=pilot_view._pairs([[p[0] + 0.0, p[1] + 0.0] for p in history[:3]]),
        command=pilot_view.COMMAND_TEXT[labels.describe_command(nav["driving_command"])])
    return {"id": record["id"], "image": f"images/{record['id']}.jpg", "system": pilot_view.SYSTEM, "prompt": prompt}


def constant_velocity(record: Dict[str, Any]) -> List[List[float]]:
    speed = math.hypot(*record["navsim"]["ego_velocity"][:2])
    return [[round(speed * t, 3), 0.0] for t in pilot_view.TARGET_S]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", required=True, help="navtest_inputs.json from navsim_eval/export_mvlad_inputs.py")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None, help="cap per shard (timing runs)")
    args = parser.parse_args()

    with open(args.inputs, encoding="utf-8") as f:
        records = sorted(json.load(f), key=lambda r: r["id"])[args.shard :: args.num_shards]
    if args.max_samples:
        records = records[: args.max_samples]
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    out_path = os.path.join(args.out, f"shard_{args.shard:03d}_of_{args.num_shards:03d}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["token"] for line in f if line.strip()}
    todo = [r for r in records if r["id"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(records)} scenes, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    import evaluate  # pilot_sft/, imports torch

    backend = evaluate.HF(args.model, args.adapter, args.out)
    start, fallbacks = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            for r in chunk:  # the view's resize: same width, filter and JPEG quality as training
                pilot_view._resize((r["image"], os.path.join(args.out, "images", f"{r['id']}.jpg"), pilot_view.WIDTH))
            answers = backend.answer([item(r) for r in chunk])
            for r, text in zip(chunk, answers):
                path = pilot_view.parse_path(text)
                row = {"token": r["id"], "answer": text}
                if path is None:
                    fallbacks += 1
                    row.update(waypoints=constant_velocity(r), fallback=True)
                else:
                    row["waypoints"] = path.round(3).tolist()
                out.write(json.dumps(row) + "\n")
                os.remove(os.path.join(args.out, "images", f"{r['id']}.jpg"))  # scratch has a file quota
            out.flush()
            n = i + len(chunk)
            if fallbacks == n >= 64:  # a broken model would otherwise score as constant velocity
                sys.exit(f"first {n} answers all failed to parse, last: {answers[-1]!r}")
            print(f"{n}/{len(todo)} scenes, {n / (time.time() - start):.2f} scenes/s, {fallbacks} fallbacks", flush=True)
    print(json.dumps({"shard": args.shard, "scenes": len(todo), "fallbacks": fallbacks,
                      "elapsed_s": round(time.time() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
