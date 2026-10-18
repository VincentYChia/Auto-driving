"""Generate reasoning labels by sending teacher requests to an OpenAI-compatible VLM server.

Works with any server exposing ``/v1/chat/completions`` (e.g. ``vllm serve <model>`` on an
Anvil GPU node). Images are sent inline as base64 data URLs. Re-running skips requests that
already have a label, so it can resume after a job ends.

    python -m data_pipeline.label_client --requests views/navtrain/teacher_requests.jsonl \
        --output labels/navtrain_teacher.jsonl --base-url http://localhost:8000/v1 --model google/gemma-4-31b-it
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import datetime as dt
import json
import mimetypes
import os
import re
import urllib.request
from typing import Any, Dict, Optional


def _inline_images(body: Dict[str, Any]) -> Dict[str, Any]:
    body = json.loads(json.dumps(body))
    for message in body["messages"]:
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("file://"):
                    path = url[len("file://"):]
                    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
                    with open(path, "rb") as f:
                        part["image_url"]["url"] = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return body


def parse_label(text: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of a model reply (tolerates code fences and extra prose)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("narration"), str) or not isinstance(data.get("reasoning"), str):
        return None
    return data


def _call(request: Dict[str, Any], base_url: str, model: str, api_key: str, timeout: float) -> Dict[str, Any]:
    body = _inline_images(request["body"])
    body["model"] = model
    http = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(http, timeout=timeout) as resp:
        reply = json.loads(resp.read())
    text = reply["choices"][0]["message"]["content"]
    label = parse_label(text) or {}
    return {
        "custom_id": request["custom_id"],
        "narration": label.get("narration"),
        "reasoning": label.get("reasoning"),
        "decision": label.get("decision") or request.get("meta_action"),
        "raw": None if label else text,
        "label_source": {
            "model": model,
            "teacher_prompt_version": request.get("prompt_version"),
            "temperature": body.get("temperature"),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = {json.loads(line)["custom_id"] for line in f if line.strip()}
    with open(args.requests) as f:
        requests = [r for r in (json.loads(line) for line in f if line.strip()) if r["custom_id"] not in done]
    if args.limit is not None:
        requests = requests[: args.limit]
    print(f"{len(done)} already labelled, {len(requests)} to send")

    ok = failed = 0
    with open(args.output, "a") as out, cf.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(_call, r, args.base_url, args.model, args.api_key, args.timeout): r for r in requests}
        for fut in cf.as_completed(futures):
            try:
                row = fut.result()
            except Exception as exc:  # network/server errors: leave unlabelled so a rerun retries them
                failed += 1
                print(f"  {futures[fut]['custom_id']}: {exc!r}")
                continue
            out.write(json.dumps(row) + "\n")
            out.flush()
            ok += 1
    print(f"labelled {ok}, failed {failed}")


if __name__ == "__main__":
    main()
