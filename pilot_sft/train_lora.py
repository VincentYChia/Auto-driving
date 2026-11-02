"""LoRA fine-tuning of a small Qwen VLM as the driving pilot, on the ``pilot_v1`` view.

Reads only the view written by ``data_pipeline/v1/pilot_view.py`` (``train.jsonl``, ``dev.jsonl``,
``images/``), so it runs anywhere the view is copied: this computer, Anvil or a rented GPU.

    python pilot_sft/train_lora.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke \\
        --model Qwen/Qwen3.5-2B --out runs/qwen35-2b-smoke --epochs 2

What is trained: LoRA on every linear layer of the language model; the vision encoder and its
projector stay frozen. The loss covers the answer only (the 8 "(x, y)" pairs and the end token).
Settings are [proposed] starting points for a first test, not tuned.

Writes ``<out>/adapter/`` (PEFT), ``<out>/log.jsonl`` (loss per step, dev loss per evaluation) and
``<out>/run.json`` (settings, data checksums, timings, peak GPU memory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Dict, List

import torch
from PIL import Image


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": item["system"]}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["prompt"]}]},
    ]


def prompt_text(processor, item: Dict[str, Any]) -> str:
    """The chat-templated prompt, ending where the answer starts. Thinking is off where the template has it."""
    try:
        return processor.apply_chat_template(prompt_messages(item), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return processor.apply_chat_template(prompt_messages(item), tokenize=False, add_generation_prompt=True)


def mask_ego(item: Dict[str, Any]) -> Dict[str, Any]:
    """The item with its ego-state and past-position lines replaced by "not available"; the command stays."""
    lines = [("Ego state: not available." if l.startswith("Ego state:") else
              "Past positions: not available." if l.startswith("Past positions") else l)
             for l in item["prompt"].split("\n")]
    return {**item, "prompt": "\n".join(lines)}


def answer_end(processor) -> str:
    """The token that closes an assistant turn (``<|im_end|>`` for Qwen)."""
    tok = processor.tokenizer
    return "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


class Batches:
    """Tokenised batches. The prompt part of ``labels`` is -100, so the loss covers the answer only.

    With ``ego_dropout`` > 0, training batches hide the ego state and past positions of that share of
    samples (``mask_ego``), so the answer has to come from the image and the command. Dev batches never do.
    """

    def __init__(self, processor, view: str, items: List[Dict[str, Any]], ego_dropout: float = 0.0, seed: int = 0):
        self.processor, self.view, self.items = processor, view, items
        self.end = answer_end(processor)
        self.ego_dropout, self.rng = ego_dropout, random.Random(seed + 1)

    def collate(self, items: List[Dict[str, Any]], train: bool = False) -> Dict[str, torch.Tensor]:
        if train and self.ego_dropout > 0:
            items = [mask_ego(it) if self.rng.random() < self.ego_dropout else it for it in items]
        prompts = [prompt_text(self.processor, it) for it in items]
        texts = [p + it["target"] + self.end for p, it in zip(prompts, items)]
        images = [Image.open(os.path.join(self.view, *it["image"].split("/"))).convert("RGB") for it in items]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        prompt_ids = self.processor(text=prompts, images=images, padding=False)["input_ids"]
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row, ids in enumerate(prompt_ids):
            n_prompt = len(ids)
            pad = int((batch["attention_mask"][row] == 0).sum()) if self.processor.tokenizer.padding_side == "left" else 0
            labels[row, : pad + n_prompt] = -100
        batch["labels"] = labels
        return batch


def language_linear_layers(model) -> List[str]:
    """Names of every nn.Linear in the language model: not the vision tower, projector or LM head."""
    skip = ("visual", "vision", "lm_head", "merger", "multi_modal_projector")
    return sorted({name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and not any(s in name for s in skip)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", required=True, help="pilot_v1 view directory (train.jsonl, dev.jsonl, images/)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=4, help="per-device batch")
    parser.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--warmup", type=float, default=0.03, help="share of steps")
    parser.add_argument("--eval-every", type=int, default=200, help="optimizer steps")
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (timing tests)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N train samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-checkpointing", action="store_true", help="faster, needs more memory")
    parser.add_argument("--ego-dropout", type=float, default=0.0,
                        help="share of training samples whose ego state and past positions are hidden [proposed]")
    parser.add_argument("--save-every", type=int, default=50, help="optimizer steps between checkpoints; 0 = none")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pt if it exists")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    train = read_jsonl(os.path.join(args.view, "train.jsonl"))[: args.limit]
    dev = read_jsonl(os.path.join(args.view, "dev.jsonl"))
    dev = random.Random(args.seed).sample(dev, min(args.eval_samples, len(dev)))

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(device)
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = language_linear_layers(model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    batches = Batches(processor, args.view, train, ego_dropout=args.ego_dropout, seed=args.seed)
    steps_per_epoch = math.ceil(len(train) / (args.batch * args.accum))
    steps = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))

    def dev_loss() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(dev), args.batch):
                batch = {k: v.to(device) for k, v in batches.collate(dev[i : i + args.batch]).items()}
                losses.append(float(model(**batch).loss))
        model.train()
        return sum(losses) / max(1, len(losses))

    ckpt = os.path.join(args.out, "checkpoint.pt")
    log_path = os.path.join(args.out, "log.jsonl")
    order: List[int] = []
    step = seen = tokens = 0
    done_s = 0.0  # training seconds before this process, for resumed runs
    first = None
    if args.resume and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        loaded = model.load_state_dict(state["trainable"], strict=False)
        if loaded.unexpected_keys:
            raise SystemExit(f"{ckpt} does not match this model: {loaded.unexpected_keys[:3]}")
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        step, seen, tokens, order, done_s, first = (state[k] for k in ("step", "seen", "tokens", "order", "elapsed_s",
                                                                      "dev_loss_start"))
        random.setstate(state["py_rng"])
        batches.rng.setstate(state["dropout_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        with open(log_path, encoding="utf-8") as f:  # drop what was logged after the checkpoint
            kept = [line for line in f if line.strip() and json.loads(line)["step"] <= step]
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        print(f"resumed from {ckpt} at step {step}", flush=True)

    def save_checkpoint() -> None:
        state = {"trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                 "optim": optim.state_dict(), "sched": sched.state_dict(), "step": step, "seen": seen,
                 "tokens": tokens, "order": order, "elapsed_s": time.time() - start, "dev_loss_start": first,
                 "py_rng": random.getstate(), "dropout_rng": batches.rng.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}
        torch.save(state, ckpt + ".part")
        os.replace(ckpt + ".part", ckpt)

    log = open(log_path, "a", encoding="utf-8")
    run = {"args": vars(args), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "dtype": str(dtype), "train_samples": len(train), "dev_samples_for_loss": len(dev),
           "trainable_params": trainable, "total_params": total, "lora_targets": len(targets),
           "optimizer_steps": steps, "resumed_at_step": step or None,
           "data": {f: sha256(os.path.join(args.view, f)) for f in ("train.jsonl", "dev.jsonl")}}
    print(json.dumps({k: run[k] for k in ("device", "train_samples", "trainable_params", "optimizer_steps")}), flush=True)
    if first is None:
        first = dev_loss()
        log.write(json.dumps({"step": 0, "dev_loss": first}) + "\n")
    start = time.time() - done_s
    model.train()
    while step < steps:
        for _ in range(args.accum):
            if len(order) < args.batch:
                order += random.sample(range(len(train)), len(train))
            idx, order = order[: args.batch], order[args.batch :]
            batch = {k: v.to(device) for k, v in batches.collate([train[i] for i in idx], train=True).items()}
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                loss = model(**batch).loss / args.accum
            loss.backward()
            seen += len(idx)
            tokens += int(batch["attention_mask"].sum())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        step += 1
        entry = {"step": step, "loss": round(loss.item() * args.accum, 4), "lr": sched.get_last_lr()[0],
                 "samples": seen, "elapsed_s": round(time.time() - start, 1)}
        if step % args.eval_every == 0 or step == steps:
            entry["dev_loss"] = round(dev_loss(), 4)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if step % 10 == 0 or "dev_loss" in entry:
            print(json.dumps(entry), flush=True)
        if args.save_every and step % args.save_every == 0 and step < steps:
            save_checkpoint()
    elapsed = time.time() - start
    model.save_pretrained(os.path.join(args.out, "adapter"))
    processor.save_pretrained(os.path.join(args.out, "adapter"))
    if os.path.exists(ckpt):
        os.remove(ckpt)  # the adapter is saved; the checkpoint only served a restart
    run.update({"elapsed_s": round(elapsed, 1), "samples_seen": seen, "samples_per_s": round(seen / elapsed, 3),
                "tokens_seen": tokens, "tokens_per_s": round(tokens / elapsed, 1), "dev_loss_start": first,
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None})
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    print(json.dumps({k: run[k] for k in ("elapsed_s", "samples_per_s", "tokens_per_s", "peak_gpu_gb")}), flush=True)


if __name__ == "__main__":
    main()
