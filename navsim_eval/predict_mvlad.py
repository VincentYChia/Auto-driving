"""Run the MVLAD-AD planning model on NAVSIM inputs (MVLAD env, one GPU per shard).

Inputs come from ``navsim_eval.export_mvlad_inputs``. Each shard appends one JSON line per
scene to ``<output-dir>/shard_XXX_of_YYY.jsonl``; re-running skips scenes already done, so a
job that hits the wall-clock limit can simply be resubmitted. Score the results with
``navsim_eval.precomputed_agent`` in the NAVSIM env.

    python navsim_eval/predict_mvlad.py --inputs navtest_inputs.json --output-dir predictions \
        --shard 0 --num-shards 8
"""

import argparse
import json
import os
import time
import warnings
from dataclasses import replace
from pathlib import Path

import torch

from inference.run_inference import generate_planning, load_mvlad, prepare_inputs
from train.config.nuscene_inference_vla import CONFIGS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", required=True, help="JSON written by navsim_eval.export_mvlad_inputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="cap per shard (for timing runs)")
    parser.add_argument("--lora-path", default=None, help="override the planning adapter directory")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    config = CONFIGS["planning"]
    if args.lora_path:
        config = replace(config, lora_path=args.lora_path)

    with open(args.inputs) as f:
        records = sorted(json.load(f), key=lambda r: r["id"])[args.shard :: args.num_shards]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.shard:03d}_of_{args.num_shards:03d}.jsonl"
    err_path = out_dir / f"errors_{args.shard:03d}_of_{args.num_shards:03d}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {json.loads(line)["token"] for line in f if line.strip()}
    todo = [r for r in records if r["id"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(records)} scenes, {len(done)} done, {len(todo)} to run")
    if not todo:
        return

    tokenizer, model, image_processor, point_tokenizer, ids_tensor = load_mvlad(config)
    model_seconds = []
    with open(out_path, "a") as out, open(err_path, "a") as err:
        for i, record in enumerate(todo):
            token = record["id"]
            try:
                if not record.get("image") or not Path(record["image"]).exists():
                    raise FileNotFoundError(f"front image missing: {record.get('image')}")
                input_ids, image_tensor, image_size = prepare_inputs(record, tokenizer, image_processor, model, config)
                torch.cuda.synchronize()
                start = time.time()
                points_ids, waypoints = generate_planning(
                    model, tokenizer, input_ids, image_tensor, image_size, point_tokenizer, ids_tensor, config
                )
                torch.cuda.synchronize()
                seconds = time.time() - start
            except Exception as exc:  # keep going; failed scenes are listed in the errors file
                err.write(json.dumps({"token": token, "error": repr(exc)}) + "\n")
                err.flush()
                continue
            model_seconds.append(seconds)
            out.write(
                json.dumps(
                    {
                        "token": token,
                        "waypoints": [[round(float(x), 4), round(float(y), 4)] for x, y in waypoints.tolist()],
                        "action_token_ids": [int(t) for t in points_ids.tolist()],
                        "model_seconds": round(seconds, 4),
                    }
                )
                + "\n"
            )
            out.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                mean = sum(model_seconds) / len(model_seconds)
                print(f"  [{i + 1}/{len(todo)}] mean model time {mean:.3f} s/scene")


if __name__ == "__main__":
    main()
