"""Token -> drive maps for the drive-level bootstrap (docs/HEADS_PLAN.md R3).

A drive is one nuPlan recording: the date-and-vehicle part of an OpenScene log name. For example,
``2021.06.14.18.33.41_veh-35_04905_05090`` is drive ``2021.06.14.18.33.41_veh-35``. Scenes of one drive are
correlated, so intervals resample drives, not scenes.

    python pilot_sft/heads/drive_map.py navtest --inputs <navtest_inputs.json> --out <cache root>/drives_navtest.json
    python pilot_sft/heads/drive_map.py navdev --cache <cache>/dev --out <cache root>/drives_navdev.json
    python pilot_sft/heads/drive_map.py halves --train-meta <cache>/train/meta.jsonl --rl-view <v1-nt view> \\
        --out <exp>/heads/halves

``halves`` is phase 6's cross-fitting split: the training drives in two halves (``drive_halves``), and each half's
RL-scene tokens, which the generator trained on the other half turns into out-of-fold candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Dict, Sequence

LOG = re.compile(r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)(?:_\d+_\d+)?")


def drive_of_log(log: str) -> str:
    m = LOG.search(log)
    if not m:
        raise ValueError(f"not an OpenScene log name: {log!r}")
    return m.group(1)


def drive_of_id(row_id: str) -> str:
    """The drive of a stored-feature or view row id, ``openscene/<log>/<token>``."""
    return drive_of_log(row_id.split("/")[1])


def drive_halves(ids: Sequence[str], seed: int = 0) -> Dict[str, int]:
    """Phase 6 cross-fitting (docs/HEADS_PLAN.md): the drives of ``ids`` in two halves of about equal scene counts.

    The drives, sorted, are shuffled with ``seed`` and taken in that order into half 0 until it holds at least half of
    the scenes; the remaining drives form half 1. Returns {drive: 0 or 1}."""
    counts = Counter(drive_of_id(i) for i in ids)
    order = sorted(counts)
    random.Random(seed).shuffle(order)
    half, taken, total = {}, 0, sum(counts.values())
    for d in order:
        if taken < total / 2:
            half[d], taken = 0, taken + counts[d]
        else:
            half[d] = 1
    return half


def heldout_drives(ids: Sequence[str], n: int = 10, seed: int = 0) -> set:
    """The selector's held-out training drives (docs/HEADS_PLAN.md R14): ``n`` of the drives of ``ids``, the first
    ``n`` after shuffling the sorted drives with ``seed``."""
    order = sorted({drive_of_id(i) for i in ids})
    random.Random(seed).shuffle(order)
    return set(order[:n])


def navtest_map(inputs: str) -> Dict[str, str]:
    """From the exported navtest inputs: each record's camera image path holds its log name."""
    with open(inputs, encoding="utf-8") as f:
        records = json.load(f)
    out = {}
    for r in records:
        log = (r.get("navsim") or {}).get("log_name") or r["image"]
        out[r["id"]] = drive_of_log(log)
    return out


def navdev_map(cache: str) -> Dict[str, str]:
    """From the dev split's stored-feature rows: ids are ``openscene/<log>/<token>``."""
    out = {}
    with open(os.path.join(cache, "meta.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                _, log, token = json.loads(line)["id"].split("/")
                out[token] = drive_of_log(log)
    return out


def read_ids(path: str) -> list:
    """Row ids from a stored-feature ``meta.jsonl`` or a view's ``train.jsonl``."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def write_halves(train_meta: str, rl_view: str, out: str, seed: int = 0) -> dict:
    """Phase 6: the training drives' two halves (``drive_halves``), and for each half h the NAVSIM tokens of the RL
    scenes on its drives, ``rl_tokens_half<h>.txt``: the generator trained on the other half predicts those."""
    halves = drive_halves(read_ids(train_meta), seed)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "halves.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "train_meta": train_meta, "halves": halves}, f, indent=1)
    rl = read_ids(os.path.join(rl_view, "train.jsonl"))
    summary = {"drives": [sum(1 for v in halves.values() if v == h) for h in (0, 1)], "rl_scenes": []}
    for h in (0, 1):
        toks = [i.rsplit("/", 1)[-1] for i in rl if halves[drive_of_id(i)] == h]  # every RL drive is a train drive
        with open(os.path.join(out, f"rl_tokens_half{h}.txt"), "w", encoding="utf-8") as f:
            f.writelines(t + "\n" for t in toks)
        summary["rl_scenes"].append(len(toks))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("split", choices=("navtest", "navdev", "halves"))
    p.add_argument("--inputs", help="navtest_inputs.json (navtest)")
    p.add_argument("--cache", help="the dev split of the stored features (navdev)")
    p.add_argument("--train-meta", help="halves: meta.jsonl of the stored features' train split")
    p.add_argument("--rl-view", help="halves: the RL scenes' view folder (with train.jsonl)")
    p.add_argument("--out", required=True, help="a JSON file; for halves, a folder")
    args = p.parse_args()
    if args.split == "halves":
        print(json.dumps(write_halves(args.train_meta, args.rl_view, args.out)))
        return
    m = navtest_map(args.inputs) if args.split == "navtest" else navdev_map(args.cache)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(m, f)
    counts = Counter(m.values())
    print(json.dumps({"split": args.split, "scenes": len(m), "drives": len(counts),
                      "scenes_per_drive_min_median_max": [min(counts.values()), sorted(counts.values())[len(counts) // 2],
                                                          max(counts.values())]}))


if __name__ == "__main__":
    main()
