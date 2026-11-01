"""Fine-tune the pilot on Together.ai: check, upload, price, launch, follow, download.

Needs ``pip install together`` (SDK 3.x; checked against 3.0.0) and, except for ``check``, the
``TOGETHER_API_KEY`` environment variable. Input: the chat export of the pilot view
(``python -m data_pipeline.v1.pilot_view export``).

    python pilot_sft/together_finetune.py check  --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke
    python pilot_sft/together_finetune.py launch --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke         # uploads, prints the price, stops
    python pilot_sft/together_finetune.py launch --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke --yes   # ... and starts the job
    python pilot_sft/together_finetune.py status   JOB_ID
    python pilot_sft/together_finetune.py download JOB_ID --dest ../runs/together/JOB_ID

``launch`` without ``--yes`` costs nothing: it uploads the files and asks Together for its price
estimate. Every launch is recorded in ``../runs/together/<suffix>.json``.

Settings [proposed]: Qwen/Qwen3.5-4B (Together's smallest vision model, V 2026-11-01), LoRA, loss
on the answer only, vision encoder frozen, 2 epochs, learning rate 1e-4, 4 evaluations on dev.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

MODEL = "Qwen/Qwen3.5-4B"
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "together")


def _dump(obj: Any) -> Dict[str, Any]:
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


def check(view: str) -> bool:
    from together.lib.utils import check_file

    ok = True
    for split in ("train", "dev"):
        report = check_file(os.path.join(view, f"chat_{split}.jsonl"))
        print(split, report.get("is_check_passed"), report.get("message"), report.get("num_samples"), report.get("file_size"))
        ok &= bool(report.get("is_check_passed"))
    return ok


def _upload(client, path: str) -> str:
    print(f"[together] uploading {path} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    file_id = client.files.upload(file=path, purpose="fine-tune", check=True).id
    while True:  # Together validates the file on its side after the upload
        status = str(getattr(client.files.retrieve(file_id), "processing_status", "COMPLETED"))
        if "COMPLETED" in status.upper() or status in ("None", ""):
            return file_id
        if any(s in status.upper() for s in ("INVALID", "FAILED")):
            raise SystemExit(f"{path}: Together rejected the file ({status})")
        time.sleep(10)


def launch(args) -> None:
    from together import Together

    if not check(args.view):
        raise SystemExit("the local check failed; fix the export first")
    client = Together()
    train_id = _upload(client, os.path.join(args.view, "chat_train.jsonl"))
    dev_id = _upload(client, os.path.join(args.view, "chat_dev.jsonl"))
    price = client.fine_tuning.estimate_price(training_file=train_id, validation_file=dev_id, model=args.model,
                                              n_epochs=args.epochs, n_evals=args.evals)
    record: Dict[str, Any] = {"view": os.path.abspath(args.view), "model": args.model, "training_file": train_id,
                              "validation_file": dev_id, "price_estimate": _dump(price), "settings": vars(args)}
    print("[together] price estimate:", json.dumps(record["price_estimate"], default=str), flush=True)
    if args.yes:
        job = client.fine_tuning.create(
            training_file=train_id, validation_file=dev_id, model=args.model, n_epochs=args.epochs,
            n_evals=args.evals, n_checkpoints=1, learning_rate=args.lr, warmup_ratio=0.03, lora=True,
            lora_r=args.rank, lora_alpha=2 * args.rank, train_vision=False, train_on_inputs=False,
            suffix=args.suffix, random_seed=0)
        record["job"] = _dump(job)
        print(f"[together] job {job.id} created; follow it with: status {job.id}", flush=True)
    else:
        print("[together] not started: rerun with --yes to start the job at this price", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, f"{args.suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=str)


def status(job_id: str) -> None:
    from together import Together

    client = Together()
    job = client.fine_tuning.retrieve(job_id)
    print("status:", job.status)
    events = _dump(client.fine_tuning.list_events(job_id)).get("data", [])
    for event in events[-15:]:
        print(" ", event.get("created_at"), event.get("message"))


def download(job_id: str, dest: str) -> None:
    from together import Together

    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "adapter.tar.zst")
    Together().fine_tuning.content(ft_id=job_id, checkpoint="adapter").write_to_file(path)
    print(f"[together] {path}: unpack with  tar --zstd -xf {path} -C {dest}  then evaluate with "
          f"pilot_sft/evaluate.py --model {MODEL} --adapter <unpacked dir>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--view", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--view", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--evals", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--suffix", default=None, help="job name suffix; default pilot-<set>-<date>")
    p.add_argument("--yes", action="store_true", help="start the (billed) job after the price estimate")
    for name in ("status", "download"):
        p = sub.add_parser(name)
        p.add_argument("job_id")
        if name == "download":
            p.add_argument("--dest", required=True)
    args = parser.parse_args()
    if args.command == "check":
        sys.exit(0 if check(args.view) else 1)
    if args.command != "check" and not os.environ.get("TOGETHER_API_KEY"):
        raise SystemExit("set TOGETHER_API_KEY first")
    if args.command == "launch":
        args.suffix = args.suffix or f"pilot-{os.path.basename(os.path.normpath(args.view))}-{time.strftime('%Y%m%d')}"
        launch(args)
    elif args.command == "status":
        status(args.job_id)
    else:
        download(args.job_id, args.dest)


if __name__ == "__main__":
    main()
