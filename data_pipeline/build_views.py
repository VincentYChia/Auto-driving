"""Build model-ready JSONL files from canonical shards (plain Python; no NAVSIM needed).

    # pilot examples for all temporal-context variants + teacher requests for reasoning labels
    python -m data_pipeline.build_views --canonical $DATA_ROOT/canonical/navtrain \
        --sensor-root $OPENSCENE_DATA_ROOT/sensor_blobs/trainval --output-dir $DATA_ROOT/views/navtrain

    # after labels exist (data_pipeline.label_client), add reasoner examples
    python -m data_pipeline.build_views ... --labels $DATA_ROOT/labels/navtrain_teacher.jsonl
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from typing import Any, Dict, Iterator

from data_pipeline import views


def iter_samples(canonical_dir: str) -> Iterator[Dict[str, Any]]:
    for path in sorted(glob.glob(os.path.join(canonical_dir, "samples-*.jsonl.gz"))):
        with gzip.open(path, "rt") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_labels(path: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("narration") and row.get("reasoning"):
                    out[row["custom_id"]] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canonical", required=True, help="directory of samples-*.jsonl.gz")
    parser.add_argument("--sensor-root", required=True, help="absolute path of sensor_blobs/<data_split>")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=list(views.PILOT_VARIANTS))
    parser.add_argument("--labels", default=None, help="teacher labels JSONL -> also write reasoner.jsonl")
    parser.add_argument("--no-teacher-requests", action="store_true")
    parser.add_argument("--max-objects", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    labels = load_labels(args.labels) if args.labels else {}
    outs = {v: open(os.path.join(args.output_dir, f"pilot_{v}.jsonl"), "w") for v in args.variants}
    teacher = None if args.no_teacher_requests else open(os.path.join(args.output_dir, "teacher_requests.jsonl"), "w")
    reasoner = open(os.path.join(args.output_dir, "reasoner.jsonl"), "w") if labels else None
    counts = {k: 0 for k in list(outs) + ["teacher_requests", "reasoner", "skipped"]}
    try:
        for sample in iter_samples(args.canonical):
            try:
                for variant, f in outs.items():
                    f.write(json.dumps(views.pilot_example(sample, args.sensor_root, variant, args.max_objects)) + "\n")
                    counts[variant] += 1
                if teacher is not None:
                    teacher.write(json.dumps(views.teacher_request(sample, args.sensor_root)) + "\n")
                    counts["teacher_requests"] += 1
                if reasoner is not None and sample["sample_id"] in labels:
                    reasoner.write(json.dumps(views.reasoner_example(sample, labels[sample["sample_id"]], args.sensor_root)) + "\n")
                    counts["reasoner"] += 1
            except (ValueError, KeyError):  # e.g. a missing camera path
                counts["skipped"] += 1
    finally:
        for f in list(outs.values()) + [teacher, reasoner]:
            if f is not None:
                f.close()
    with open(os.path.join(args.output_dir, "views_manifest.json"), "w") as f:
        json.dump({"counts": counts, "pilot_prompt_version": views.PILOT_PROMPT_VERSION,
                   "teacher_prompt_version": views.TEACHER_PROMPT_VERSION,
                   "reasoner_prompt_version": views.REASONER_PROMPT_VERSION, "args": vars(args)}, f, indent=2)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
