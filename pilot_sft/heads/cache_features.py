"""Store a frozen pilot's hidden states for the path generators: one prefill per sample, no text decoding.

    python pilot_sft/heads/cache_features.py --model Qwen/Qwen3.5-0.8B --adapter <H1 run>/adapter \\
        --view <pilot_v1 view> --split train --out <cache>/train --layers 12,18,24
    python pilot_sft/heads/cache_features.py --model Qwen/Qwen3.5-0.8B --adapter <H1 run>/adapter \\
        --navtest <navtest_inputs.json> --out <cache>/navtest --layers 12,18,24

The model sees each sample exactly as in training and NAVSIM prediction: the chat-templated prompt (the
adapter's prompt version), ending where the answer starts, and the 640 px view image. The adapter is merged
into the base model (LoRA and the trained vision tower). For each ``--layers`` index ``l`` it keeps
``hidden_states[l]`` (0 = embeddings, 24 = the last layer of the 0.8B) for the tokens from the image on; the
layout is in ``features.py``. ``--limit`` caps the rows (timing runs). Re-running an unfinished split resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # pilot_sft/: train_lora, predict_navsim, view_images
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repo: data_pipeline

import features  # noqa: E402  pilot_sft/heads/
import predict_navsim  # noqa: E402
import train_lora  # noqa: E402
from data_pipeline.v1 import pilot_view  # noqa: E402
from view_images import ViewImages  # noqa: E402


def load_items(args) -> Tuple[List[Dict[str, Any]], str, str]:
    """The samples in row order, the prompt version, and a description of the source."""
    version = predict_navsim.run_prompt_version(args.adapter)
    if args.navtest:
        with open(args.navtest, encoding="utf-8") as f:
            records = sorted(json.load(f), key=lambda r: r["id"])
        items = []
        for r in records:
            it = predict_navsim.item(r, version)
            it["file"] = r["image"]  # the camera image on disk, resized like the view at load time
            items.append(it)
        source = f"navtest {args.navtest}"
    else:
        items = train_lora.read_jsonl(os.path.join(args.view, f"{args.split}.jsonl"))
        with open(os.path.join(args.view, "view.json"), encoding="utf-8") as f:
            view_version = json.load(f).get("prompt_version") or "pilot_v1.1"
        if view_version != version:
            raise SystemExit(f"the view has prompt {view_version}, the adapter was trained on {version}")
        source = f"view {args.view} split {args.split}"
    return items[: args.limit], version, source


class Encoder:
    """Prompt text, image and processor call for one batch; one processor and image reader per thread."""

    def __init__(self, processor_dir: str, view: str):
        from transformers import AutoProcessor

        self.processor_dir, self.view, self.local = processor_dir, view, threading.local()
        self.AutoProcessor = AutoProcessor

    def tools(self):
        if not hasattr(self.local, "processor"):
            p = self.AutoProcessor.from_pretrained(self.processor_dir)
            p.tokenizer.padding_side = "right"  # causal and linear-attention layers: padding after the tokens
            self.local.processor, self.local.images = p, (ViewImages(self.view) if self.view else None)
        return self.local.processor, self.local.images

    def image(self, it: Dict[str, Any], images):
        from PIL import Image
        import io

        if "file" in it:
            with open(it["file"], "rb") as f:
                return Image.open(io.BytesIO(pilot_view.resize_bytes(f.read()))).convert("RGB")
        return images.open(it["image"]).convert("RGB")

    def __call__(self, items: List[Dict[str, Any]]):
        processor, images = self.tools()
        texts = [train_lora.prompt_text(processor, it) for it in items]
        return processor(text=texts, images=[self.image(it, images) for it in items], padding=True,
                         return_tensors="pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="the frozen pilot's adapter (H1: <run>/adapter)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--view", help="pilot_v1 view folder (with --split)")
    src.add_argument("--navtest", help="navtest_inputs.json (paths rewritten for this machine)")
    parser.add_argument("--split", default="train", choices=("train", "dev"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="12,18,24", help="hidden_states indices to keep [proposed]")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="threads building batches ahead of the GPU")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows (timing runs)")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    torch.set_num_threads(2)
    items, version, source = load_items(args)
    encoder = Encoder(args.adapter, args.view)
    processor, _ = encoder.tools()
    tok = processor.tokenizer
    vision_start = tok.convert_tokens_to_ids("<|vision_start|>")

    # tokens per row, known before any image is read: the text's tokens from <|vision_start|> on, with the
    # image placeholder expanded to the image's visual tokens (one image size per split, checked per batch)
    probe = encoder(items[:1])
    ids0 = probe["input_ids"][0].tolist()
    n_visual = sum(1 for i in ids0 if i == tok.convert_tokens_to_ids("<|image_pad|>"))
    texts = [train_lora.prompt_text(processor, it) for it in items]
    text_ids = tok(texts, add_special_tokens=False)["input_ids"]
    starts = [ids.index(vision_start) for ids in text_ids]
    expected = np.array([len(ids) - s + n_visual - 1 for ids, s in zip(text_ids, starts)], dtype=np.int64)
    if len(set(starts)) != 1:
        raise SystemExit(f"the system prefix differs between samples: {sorted(set(starts))[:5]}")
    tokens = int(expected.max())
    print(f"{len(items)} rows from {source}; prompt {version}; {n_visual} visual tokens; "
          f"{expected.min()}-{tokens} stored tokens per row; prefix of {starts[0]} tokens dropped", flush=True)

    device = "cuda"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().to(device).eval()
    backbone = model.model  # vision tower + language model, without the LM head (no logits over the vocabulary)
    width = model.config.get_text_config().hidden_size

    settings = {"model": args.model, "adapter": os.path.abspath(args.adapter), "source": source,
                "prompt_version": version, "visual_tokens": n_visual, "prefix_tokens_dropped": starts[0],
                "hidden_states_index": "0 = embeddings, k = output of decoder layer k",
                "padding": "right", "image_width_px": pilot_view.WIDTH}
    writer = features.Writer(args.out, layers, len(items), tokens, width, settings)
    todo = list(range(writer.done, len(items), args.batch))
    print(f"{writer.done} rows already written, {len(todo)} batches to run", flush=True)

    start, done0, checked_last = time.time(), writer.done, False
    with ThreadPoolExecutor(args.workers) as pool, torch.inference_mode():
        pending = {}
        for j, s in enumerate(todo[: 2 * args.workers]):
            pending[j] = pool.submit(encoder, items[s : s + args.batch])
        for j, s in enumerate(todo):
            batch = pending.pop(j).result()
            nxt = j + 2 * args.workers
            if nxt < len(todo):
                pending[nxt] = pool.submit(encoder, items[todo[nxt] : todo[nxt] + args.batch])
            chunk = items[s : s + args.batch]
            batch = batch.to(device)
            ids = batch["input_ids"]
            row_start = (ids == vision_start).int().argmax(dim=1)
            if not bool((row_start == starts[0]).all()):
                raise SystemExit(f"batch at row {s}: image starts at {row_start.tolist()}, expected {starts[0]}")
            stored = batch["attention_mask"].sum(dim=1) - starts[0]
            if stored.tolist() != expected[s : s + len(chunk)].tolist():
                raise SystemExit(f"batch at row {s}: {stored.tolist()} tokens, expected "
                                 f"{expected[s : s + len(chunk)].tolist()}")
            out = backbone(**batch, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if not checked_last:  # is the last entry the normed output? recorded, not assumed
                settings["last_index_equals_last_hidden_state"] = bool(
                    torch.equal(hs[-1], out.last_hidden_state)) if len(hs) - 1 in layers else None
                settings["hidden_states_count"] = len(hs)
                checked_last = True
            t = int(stored.max())
            feats = {l: hs[l][:, starts[0] : starts[0] + t].contiguous().view(torch.int16).cpu().numpy()
                     .view(np.uint16) for l in layers}
            metas = []
            for it in chunk:
                m = {"id": it["id"], "ego": features.parse_ego(it["prompt"])}
                if "target_xy" in it:
                    m["target_xy"], m["target_yaw"] = it["target_xy"], it.get("target_yaw")
                metas.append(m)
            writer.write(feats, stored.tolist(), metas)
            if j % 50 == 0 or j == len(todo) - 1:
                rate = (writer.done - done0) / (time.time() - start)
                left = (len(items) - writer.done) / max(rate, 1e-9)
                print(f"{writer.done}/{len(items)} rows, {rate:.1f} rows/s, {left / 3600:.2f} h left", flush=True)
    elapsed = time.time() - start
    size = sum(os.path.getsize(os.path.join(args.out, f"feat_{l}.u16")) for l in layers)
    extra = {"elapsed_s": round(elapsed, 1), "rows_per_s": round((writer.done - done0) / max(elapsed, 1e-9), 2),
             "bytes": size, "last_index_equals_last_hidden_state": settings.get("last_index_equals_last_hidden_state"),
             "hidden_states_count": settings.get("hidden_states_count"),
             "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    writer.finish(extra)
    print(json.dumps(extra), flush=True)


if __name__ == "__main__":
    main()
