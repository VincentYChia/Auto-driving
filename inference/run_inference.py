"""MVLAD-AD inference with action-priority decoding.

Usage (run from the release root, with ``setup_env.sh`` sourced):

    python inference/run_inference.py --task planning
    python inference/run_inference.py --task nux

* ``planning``: single-stage decoding that resolves the 10 action tokens and
  recovers the future waypoints (evaluate with ``train/scripts/compute_lora_metrics_from_json.py``).
* ``nux``: two-stage action-priority decoding. Stage 1 resolves the action
  tokens; they are then fixed and the reasoning tokens are in-filled (evaluate
  with ``eval/eval_nux.py``).
"""

import argparse
import copy
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from peft import PeftModel
from PIL import Image

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.conversation import conv_templates
from llava.hooks import register_cache_LLaDA_V
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX

from train.config.nuscene_inference_vla import CONFIGS
from tokenizer.example_usage import load_point_tokenizer

MASK_ID = 126336


def generate_planning(model, tokenizer, input_ids, image_tensor, image_size,
                      point_tokenizer, ids_tensor, config):
    """Stage 1: resolve the 10 action tokens and recover waypoints."""
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        gen_length=config.generation_first_length,
        block_length=config.generation_first_block_length,
        steps=config.generation_first_steps,
        tokenizer=tokenizer,
        stopping_criteria=list(config.stopping_criteria),
    )
    points_ids = cont[0][:10]
    valid_mask = torch.isin(points_ids, ids_tensor)
    if not torch.all(valid_mask):
        invalid_ids = points_ids[~valid_mask].tolist()
        raise AssertionError(f"planning tokens not in action_token_id_list: {invalid_ids}")
    pos_in_sorted = torch.searchsorted(-ids_tensor, -points_ids)
    recovered_points = point_tokenizer.indices_to_points(pos_in_sorted)
    return points_ids, recovered_points


def generate_explanation(model, tokenizer, input_ids, image_tensor, image_size,
                         points_ids, config):
    """Stage 2: fix the action tokens, in-fill the reasoning / answer tokens."""
    num_explanation_masks = config.generation_length - 10
    mask_tokens = torch.full((num_explanation_masks,), MASK_ID, dtype=torch.long, device=config.device)
    template_ids = torch.cat([points_ids, mask_tokens]).unsqueeze(0)
    cont = model.generate_fix_mask(
        template_ids=template_ids,
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        steps=config.generation_steps,
        tokenizer=tokenizer,
    )
    explanation_ids = cont[0][10:]
    return tokenizer.decode(explanation_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(CONFIGS.keys()),
                        help="planning | nux")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of evaluated samples.")
    args = parser.parse_args()

    config = CONFIGS[args.task]
    two_stage = args.task == "nux"
    warnings.filterwarnings("ignore")

    # Load base model + merge LoRA adapter.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        config.pretrained, None, config.model_name,
        attn_implementation="sdpa", device_map=config.device_map,
    )
    model = PeftModel.from_pretrained(model, config.lora_path, adapter_name="default")
    model = model.merge_and_unload()

    # Transfer geometry-aware action-token embeddings into the vocabulary.
    weights_file = Path(config.tokenizer_weights)
    if not weights_file.exists():
        raise FileNotFoundError(f"Tokenizer weights not found at {weights_file}")
    tokenizer_state = torch.load(weights_file, map_location="cpu")
    point_embeddings = tokenizer_state["embedding.E"]
    action_token_id_list = np.load(config.unused_token_ids_path).tolist()
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        emb = point_embeddings.to(input_embeddings.weight.device, dtype=input_embeddings.weight.dtype)
        input_embeddings.weight[action_token_id_list] = emb
        if hasattr(model, "lm_head") and model.lm_head.weight.shape[0] >= len(action_token_id_list):
            model.lm_head.weight[action_token_id_list] = emb
        if hasattr(model, "config"):
            model.config.action_token_ids = action_token_id_list

    point_tokenizer = load_point_tokenizer(weights_file)
    ids_tensor = torch.from_numpy(np.load(config.unused_token_ids_path)).to(config.device)
    model.eval()

    with open(config.data_path, "r") as f:
        data_val = json.load(f)
    num_samples = len(data_val) if args.max_samples is None else min(args.max_samples, len(data_val))

    results = []
    total_time = 0.0
    print(f"Running task='{args.task}' on {num_samples} samples (two_stage={two_stage})")

    for sample_idx in range(num_samples):
        data_sample = data_val[sample_idx]
        image = Image.open(data_sample["image"])
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [t.to(dtype=torch.float16, device=config.device) for t in image_tensor]

        question = data_sample["conversations"][0]["value"]
        gt_text = data_sample["conversations"][1]["value"]

        conv = copy.deepcopy(conv_templates[config.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(config.device)

        if config.use_cache:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                prompt_interval_steps=config.prompt_interval_steps,
                gen_interval_steps=config.gen_interval_steps,
                transfer_ratio=config.transfer_ratio,
            )))
            register_cache_LLaDA_V(model, "model.layers")

        start_time = time.time()
        points_ids, recovered_points = generate_planning(
            model, tokenizer, input_ids, image_tensor, image.size,
            point_tokenizer, ids_tensor, config,
        )
        explanation = ""
        if two_stage:
            explanation = generate_explanation(
                model, tokenizer, input_ids, image_tensor, image.size,
                points_ids, config,
            )
        total_time += time.time() - start_time

        # Unified 3-slot format: [pred_trajectory, explanation_or_empty, ground_truth].
        results.append([str(recovered_points.tolist())[1:-1], explanation, gt_text])

        with open(config.results_path, "w") as f:
            json.dump(results, f, indent=2)

        if (sample_idx + 1) % 10 == 0 or sample_idx == num_samples - 1:
            print(f"  [{sample_idx + 1}/{num_samples}] {recovered_points.tolist()[:2]}...")

    print(f"\nSaved results to {config.results_path}")
    print(f"Total time: {total_time:.2f}s | avg {total_time / max(1, num_samples):.3f}s/sample")


if __name__ == "__main__":
    main()
