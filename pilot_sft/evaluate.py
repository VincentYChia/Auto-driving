"""Score a pilot on the ``pilot_v1`` dev split: ADE / FDE of its 8 "(x, y)" positions vs the recorded path.

    python pilot_sft/evaluate.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke \\
        --model Qwen/Qwen3.5-2B [--adapter runs/qwen35-2b-smoke/adapter] --out runs/qwen35-2b-smoke/eval.json
    python pilot_sft/evaluate.py --view ... --backend ollama --model qwen3.5:2b   # zero-shot, via a local Ollama

These are open-loop imitation errors, not NAVSIM scores: they say whether the model learned
anything, cheaply. Every result is printed beside the constant-velocity baseline on the same samples.
Answers that do not hold 8 pairs count as parse failures and are left out of ADE / FDE.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import time
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional

PAIR = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
TARGET_S = [0.5 * (k + 1) for k in range(8)]


def parse_path(text: str) -> Optional[List[List[float]]]:
    pairs = PAIR.findall(text)
    return [[float(x), float(y)] for x, y in pairs[:8]] if len(pairs) >= 8 else None


def errors(pred: List[List[float]], truth: List[List[float]]) -> List[float]:
    return [math.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(pred, truth)]


def constant_velocity(item: Dict[str, Any]) -> List[List[float]]:
    return [[item["meta"]["speed"] * t, 0.0] for t in TARGET_S]


def summarise(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    ok = [r for r in rows if r[key] is not None]
    if not ok:
        return {"n": 0}
    return {"n": len(ok), "ade_m": round(sum(sum(r[key]) / 8 for r in ok) / len(ok), 3),
            "fde_m": round(sum(r[key][-1] for r in ok) / len(ok), 3),
            "err_at_s": {f"{t:g}": round(sum(r[key][k] for r in ok) / len(ok), 3) for k, t in enumerate(TARGET_S)}}


class HF:
    def __init__(self, model: str, adapter: Optional[str], view: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        import train_lora

        self.torch, self.view, self.train_lora = torch, view, train_lora
        self.blank_image = False  # probes.py: flat grey in place of every camera image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(adapter or model)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype).to(self.device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        # Stop at the token that closes the assistant turn. Qwen3.5-0.8B ships no generation_config.json, so
        # generate() otherwise stops only at the config's <|endoftext|> and runs on to max_new_tokens after the
        # answer's <|im_end|> (V, 2026-11-08: 989 of 1,000 dev answers ran past 8 pairs). The first 8 pairs are
        # the same either way under greedy decoding; stopping only removes the wasted tokens.
        tok = self.processor.tokenizer
        end = train_lora.answer_end(self.processor)
        config_eos = getattr(self.model.generation_config, "eos_token_id", None)
        candidates = [tok.convert_tokens_to_ids(end), tok.eos_token_id]
        candidates += list(config_eos) if isinstance(config_eos, (list, tuple)) else [config_eos]
        self.stop_ids = sorted({i for i in candidates if isinstance(i, int) and i >= 0})
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.stop_ids[0]

    def answer(self, items: List[Dict[str, Any]]) -> List[str]:
        from PIL import Image

        prompts = [self.train_lora.prompt_text(self.processor, it) for it in items]
        images = [Image.open(os.path.join(self.view, *it["image"].split("/"))).convert("RGB") for it in items]
        if self.blank_image:
            images = [Image.new("RGB", im.size, (128, 128, 128)) for im in images]
        batch = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=128, do_sample=False, eos_token_id=self.stop_ids,
                                      pad_token_id=self.pad_id)
        return self.processor.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)


class Ollama:
    def __init__(self, model: str, view: str, host: str = "http://localhost:11434"):
        self.model, self.view, self.host = model, view, host

    def answer(self, items: List[Dict[str, Any]]) -> List[str]:
        out = []
        for it in items:
            with open(os.path.join(self.view, *it["image"].split("/")), "rb") as f:
                image = base64.b64encode(f.read()).decode()
            body = {"model": self.model, "stream": False, "think": False, "options": {"temperature": 0},
                    "messages": [{"role": "system", "content": it["system"]},
                                 {"role": "user", "content": it["prompt"] + "\nAnswer with the 8 pairs only.",
                                  "images": [image]}]}
            request = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                out.append(json.load(response)["message"]["content"])
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--backend", default="hf", choices=("hf", "ollama"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory from train_lora.py")
    parser.add_argument("--samples", type=int, default=500, help="seeded random subset; 0 = all")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the summary (and per-sample answers) here")
    args = parser.parse_args()

    with open(os.path.join(args.view, f"{args.split}.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.samples:
        items = random.Random(args.seed).sample(items, min(args.samples, len(items)))
    backend = HF(args.model, args.adapter, args.view) if args.backend == "hf" else Ollama(args.model, args.view)
    rows = []
    start = time.time()
    for i in range(0, len(items), args.batch):
        chunk = items[i : i + args.batch]
        for it, text in zip(chunk, backend.answer(chunk)):
            pred = parse_path(text)
            rows.append({"id": it["id"], "command": it["meta"]["command"], "answer": text,
                         "model": errors(pred, it["target_xy"]) if pred else None,
                         "cv": errors(constant_velocity(it), it["target_xy"])})
        print(f"[eval] {len(rows)}/{len(items)}", flush=True)
    by_command = defaultdict(list)
    for r in rows:
        by_command[r["command"]].append(r)
    result = {
        "view": args.view, "split": args.split, "backend": args.backend, "model": args.model, "adapter": args.adapter,
        "samples": len(rows), "parse_failures": sum(r["model"] is None for r in rows),
        "seconds_per_sample": round((time.time() - start) / max(1, len(rows)), 3),
        "model_metrics": summarise(rows, "model"),
        "constant_velocity_same_samples": summarise([r for r in rows if r["model"] is not None], "cv"),
        "by_command": {c: {"model": summarise(rs, "model"), "cv": summarise(rs, "cv")} for c, rs in sorted(by_command.items())},
    }
    print(json.dumps({k: result[k] for k in ("samples", "parse_failures", "model_metrics", "constant_velocity_same_samples")}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({**result, "answers": [{k: r[k] for k in ("id", "answer")} for r in rows]}, f, indent=1)


if __name__ == "__main__":
    main()
