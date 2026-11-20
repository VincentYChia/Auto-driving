"""Leakage audit for the path-generator study (docs/HEADS_PLAN.md R6): which drives and scenes the splits share.

    python pilot_sft/heads/leakage_audit.py --train <v1-hard view> --rl <v1-nt view> --dev <dev-v2 view> \\
        --navtest <navtest_inputs.json> --warmup <navsim-v1.1 .../scene_filter/warmup_test_e2e.yaml> --out <json>

**Splits:**
- train: H1's fine-tuning data, and every path generator's training data (v1-hard);
- rl: the RL scenes (v1-nt, navtrain scenes in our data);
- navdev: the validation scenes;
- navtest: the test scenes.

A drive is the date-and-vehicle part of a log name (``drive_map.py``).

**Must be 0:**
- drives shared by navdev or navtest with train or rl;
- navdev drives shared with navtest;
- RL tokens outside train;
- warm-up tokens in any split used for training.

Shared warm-up *logs* are listed but are not leaks by themselves: from v1-xl on, rows within ±7 s of a warm-up scene
are excluded, not whole logs. Only counts are written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_map import drive_of_log  # noqa: E402


def view_rows(view: str, split: str):
    out = []
    with open(os.path.join(view, f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                _, log, token = json.loads(line)["id"].split("/")
                out.append((log, token))
    return out


def yaml_list(path: str, key: str):
    lines = open(path, encoding="utf-8").read().split("\n")
    if f"{key}:" not in lines:
        return []
    i, out = lines.index(f"{key}:"), []
    for l in lines[i + 1:]:
        m = re.match(r"^\s*-\s*'?([^']+?)'?\s*$", l)
        if not m:
            break
        out.append(m.group(1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", required=True)
    p.add_argument("--rl", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--navtest", required=True)
    p.add_argument("--warmup", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    train, rl, dev = view_rows(a.train, "train"), view_rows(a.rl, "train"), view_rows(a.dev, "dev")
    with open(a.navtest, encoding="utf-8") as f:
        nt = [(r["navsim"]["log_name"], r["id"]) for r in json.load(f)]
    wu_tokens, wu_logs = set(yaml_list(a.warmup, "tokens")), set(yaml_list(a.warmup, "log_names"))
    drives = {k: {drive_of_log(l) for l, _ in v} for k, v in (("train", train), ("rl", rl), ("navdev", dev), ("navtest", nt))}
    tokens = {k: {t for _, t in v} for k, v in (("train", train), ("rl", rl), ("navdev", dev), ("navtest", nt))}
    logs = {k: {l for l, _ in v} for k, v in (("train", train), ("rl", rl), ("navdev", dev), ("navtest", nt))}
    must_be_zero = {
        "drives navdev & train": len(drives["navdev"] & drives["train"]),
        "drives navdev & rl": len(drives["navdev"] & drives["rl"]),
        "drives navtest & train": len(drives["navtest"] & drives["train"]),
        "drives navtest & rl": len(drives["navtest"] & drives["rl"]),
        "drives navdev & navtest": len(drives["navdev"] & drives["navtest"]),
        "rl tokens not in train": len(tokens["rl"] - tokens["train"]),
        "warm-up tokens in train": len(wu_tokens & tokens["train"]),
        "warm-up tokens in rl": len(wu_tokens & tokens["rl"]),
        "warm-up tokens in navdev": len(wu_tokens & tokens["navdev"]),
    }
    out = {"rule": "docs/HEADS_PLAN.md R6", "must_be_zero": must_be_zero, "pass": all(v == 0 for v in must_be_zero.values()),
           "sizes": {k: {"scenes": len(tokens[k]), "logs": len(logs[k]), "drives": len(drives[k])} for k in tokens},
           "warm_up": {"tokens": len(wu_tokens), "logs": len(wu_logs),
                       "logs shared with train (rows within 7 s excluded, not whole logs)": len(wu_logs & logs["train"]),
                       "logs shared with rl": len(wu_logs & logs["rl"])},
           "not_auditable": "the pre-training data of the Qwen3.5 base model"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
