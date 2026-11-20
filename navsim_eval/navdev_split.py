"""navdev: the held-out validation split for the path-generator study (docs/HEADS_PLAN.md, rule R1).

The 2,000 dev-v2 scenes come from 11 nuPlan drives that no training set contains. This writes them as NAVSIM v1.1
scene filters under ``navsim_eval/navsim_configs/train_test_split/``. NAVSIM finds them through
``hydra.searchpath=[pkg://navsim.planning.script.config.common,file://<repo>/navsim_eval/navsim_configs]``,
so the devkit itself is unchanged.

    python navsim_eval/navdev_split.py all --dev <view>/dev.jsonl        # navdev_all: every dev scene
    python navsim_eval/navdev_split.py filter --cv <cv csv> --human <human csv>   # navdev: navtest's filter

``navdev`` keeps the scenes that pass navtest's own difficulty filter (NAVSIM paper, read for
docs/research/sft_literature_check): constant velocity scores PDMS <= 0.8 (not near-trivial) and the human path
scores PDMS >= 0.8 (not a noisy label). Applied to navtest's scores this keeps 12,129 of 12,146 scenes (99.86%;
checked 2026-11-19), so the filter matches. Both splits use the trainval logs (``data_split: trainval``) and the
scene settings of navtest (4 history frames, 10 future frames, frame interval 1, a route required).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "navsim_configs", "train_test_split")
SCENE = """_target_: navsim.common.dataclasses.SceneFilter
_convert_: 'all'

num_history_frames: 4
num_future_frames: 10
frame_interval: 1
has_route: true

max_scenes: null
"""


def dev_pairs(dev_jsonl: str) -> List[Tuple[str, str]]:
    """(OpenScene log, token) of each dev-view row, ``openscene/<log>/<token>``."""
    out = []
    with open(dev_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                _, log, token = json.loads(line)["id"].split("/")
                out.append((log, token))
    return out


def write_split(name: str, pairs: List[Tuple[str, str]], note: str) -> None:
    logs = sorted({l for l, _ in pairs})
    tokens = sorted({t for _, t in pairs})
    os.makedirs(os.path.join(OUT, "scene_filter"), exist_ok=True)
    with open(os.path.join(OUT, f"{name}.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {note}\ndefaults:\n  - scene_filter: {name}\n\ndata_split: trainval\n")
    with open(os.path.join(OUT, "scene_filter", f"{name}.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {note}\n" + SCENE + "log_names:\n" + "".join(f"  - '{l}'\n" for l in logs)
                + "tokens:\n" + "".join(f"  - '{t}'\n" for t in tokens))
    print(f"{name}: {len(tokens)} tokens in {len(logs)} logs")


def scores(path: str) -> Dict[str, float]:
    with open(path, newline="") as f:
        return {r["token"]: float(r["score"]) for r in csv.DictReader(f) if r["token"] != "average"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all")
    a.add_argument("--dev", required=True, help="the dev-v2 view's dev.jsonl")
    fl = sub.add_parser("filter")
    fl.add_argument("--cv", required=True, help="per-scene PDMS CSV of constant velocity on navdev_all")
    fl.add_argument("--human", required=True, help="per-scene PDMS CSV of the human path on navdev_all")
    fl.add_argument("--dev", required=True, help="the dev-v2 view's dev.jsonl (for the log names)")
    args = p.parse_args()
    pairs = dev_pairs(args.dev)
    if args.cmd == "all":
        write_split("navdev_all", pairs, "navdev_all: every dev-v2 scene (2,000 rows, 11 held-out drives); "
                                         "written by navsim_eval/navdev_split.py all")
        return
    cv, hu = scores(args.cv), scores(args.human)
    keep = [(l, t) for l, t in pairs if t in cv and t in hu and cv[t] <= 0.8 and hu[t] >= 0.8]
    scored = sum(1 for _, t in pairs if t in cv and t in hu)
    write_split("navdev", keep, f"navdev: dev-v2 scenes passing navtest's filter (constant velocity PDMS <= 0.8, human "
                                f"PDMS >= 0.8): {len(keep)} of {scored} scored; written by navsim_eval/navdev_split.py filter")
    print(json.dumps({"dev_rows": len(pairs), "scored": scored, "cv_over_0.8": sum(1 for t in cv if cv[t] > 0.8),
                      "human_under_0.8": sum(1 for t in hu if hu[t] < 0.8), "navdev": len(keep)}))


if __name__ == "__main__":
    main()
