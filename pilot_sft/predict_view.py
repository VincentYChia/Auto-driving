"""A trained text pilot (e.g. H1) on a pilot_v1 view split, as NAVSIM paths: the reference for navdev.

    python pilot_sft/predict_view.py --view <view> --split dev --model Qwen/Qwen3.5-0.8B --adapter <H1>/adapter \\
        --out <H1 run>/navdev_predictions

The prompts and images are the view's, exactly as in training. Each answer's first 8 "(x, y)" pairs become the
path. An answer without 8 pairs gets the constant-velocity path, marked ``"fallback": true`` and counted, as in
``predict_navsim.py``. Writes ``<out>/shard_000_of_001.jsonl`` with ``{"token", "waypoints", "answer"}`` rows;
tokens are bare NAVSIM tokens (a view id ``openscene/<log>/<token>`` becomes ``<token>``). Re-running skips
scenes already written.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import evaluate  # pilot_sft/
import train_lora  # pilot_sft/


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--view", required=True)
    p.add_argument("--split", default="dev", choices=("dev", "train"))
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    items = train_lora.read_jsonl(os.path.join(args.view, f"{args.split}.jsonl"))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "shard_000_of_001.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["token"] for l in f if l.strip()}
    todo = [it for it in items if it["id"].rsplit("/", 1)[-1] not in done]
    print(f"{len(items)} scenes, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    backend = evaluate.HF(args.model, args.adapter, args.view)
    start, fallbacks = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            for it, text in zip(chunk, backend.answer(chunk)):
                path = evaluate.parse_path(text)
                row = {"token": it["id"].rsplit("/", 1)[-1], "answer": text}
                if path is None:
                    fallbacks += 1
                    row.update(waypoints=evaluate.constant_velocity(it), fallback=True)
                else:
                    row["waypoints"] = [[round(float(x), 3), round(float(y), 3)] for x, y in path]
                out.write(json.dumps(row) + "\n")
            out.flush()
            n = i + len(chunk)
            print(f"{n}/{len(todo)} scenes, {n / (time.time() - start):.2f} scenes/s, {fallbacks} fallbacks", flush=True)
    print(json.dumps({"scenes": len(todo), "fallbacks": fallbacks, "elapsed_s": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
