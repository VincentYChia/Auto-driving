"""
NuScenes inference script that computes displacement metrics for generated trajectories.

This script mirrors the generation flow in `nuscene_inference_lora_time_vla.py` but, instead of
only saving the decoded trajectories, it evaluates the predictions against the ground-truth
ego waypoints provided in the dataset (`action_targets`). It reports the Average Displacement
Error (ADE) and Final Displacement Error (FDE) aggregated across the processed samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from peft import PeftModel  # noqa: F401  # Imported to keep parity with other inference scripts.

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.conversation import conv_templates
from llava.hooks import register_cache_LLaDA_V
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX

from train.config.nuscene_inference_vla import planning_config as config
from tokenizer.example_usage import load_point_tokenizer


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NuScenes LoRA inference and compute ADE/FDE.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of data samples to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump per-sample metrics as JSON.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable KV-cache usage even if enabled in the config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log detailed metrics for each processed sample.",
    )
    return parser.parse_args()


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found at {dataset_path}")
    with dataset_path.open("r") as f:
        return json.load(f)


def setup_model() -> Tuple[Any, Any, Any, int]:
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        config.pretrained,
        config.model_base,
        config.model_name,
        attn_implementation="sdpa",
        device_map=config.device_map,
    )
    model.eval()
    return tokenizer, model, image_processor, max_length


def prepare_point_tokenizer(device: torch.device) -> Tuple[Any, torch.Tensor]:
    weights_file = Path(config.tokenizer_weights)
    if not weights_file.exists():
        raise FileNotFoundError(f"Tokenizer weights not found at {weights_file}")

    point_tokenizer = load_point_tokenizer(weights_file)

    unused_token_ids_path = Path(config.unused_token_ids_path)
    if not unused_token_ids_path.is_absolute():
        unused_token_ids_path = ROOT / unused_token_ids_path

    if not unused_token_ids_path.exists():
        raise FileNotFoundError(f"Unused token ids file not found at {unused_token_ids_path}")

    ids_tensor = torch.from_numpy(np.load(unused_token_ids_path)).to(device=device)
    return point_tokenizer, ids_tensor


def compute_displacement_errors(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Tuple[float, float, int, int]:
    """
    Compute ADE and FDE between predicted and ground-truth trajectories.

    Returns
    -------
    ade : float
        Average displacement error across the overlapping horizon.
    fde : float
        Final displacement error at the last overlapping step.
    steps_used : int
        Number of points that were compared (overlap length).
    total_pred_steps : int
        Length of the predicted trajectory tensor.
    """

    if predicted.ndim != 2 or predicted.size(-1) != 2:
        raise ValueError(f"Expected predicted points of shape (T, 2); got {tuple(predicted.shape)}")
    if ground_truth.ndim != 2 or ground_truth.size(-1) != 2:
        raise ValueError(f"Expected ground truth points of shape (T, 2); got {tuple(ground_truth.shape)}")

    pred_len = predicted.size(0)
    gt_len = ground_truth.size(0)
    horizon = min(pred_len, gt_len)
    if horizon == 0:
        raise ValueError("Predicted and ground-truth trajectories must contain at least one point.")

    pred_slice = predicted[:horizon]
    gt_slice = ground_truth[:horizon]

    diffs = pred_slice - gt_slice
    dists = torch.linalg.norm(diffs, dim=1)

    ade = float(dists.mean().item())
    fde = float(dists[-1].item())
    return ade, fde, horizon, pred_len


def main() -> None:
    args = parse_args()

    print("Loading dataset...")
    data_samples = load_dataset(config.data_path)
    if args.limit is not None:
        data_samples = data_samples[: args.limit]
    print(f"Loaded {len(data_samples)} samples from {config.data_path}")

    print("Loading model and tokenizer...")
    tokenizer, model, image_processor, _ = setup_model()

    device = torch.device(config.device)
    point_tokenizer, ids_tensor = prepare_point_tokenizer(device)

    total_time = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_steps = 0
    num_samples = 0

    per_sample_metrics: List[Dict[str, Any]] = []

    template_name = config.conv_template
    cache_enabled = config.use_cache and not args.disable_cache

    model.eval()

    for idx, sample in enumerate(data_samples):
        image = Image.open(sample["image"])
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        image_sizes = [image.size]

        conv = copy.deepcopy(conv_templates[template_name])
        question = sample["conversations"][0]["value"]
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if cache_enabled:
            dLLMCache.new_instance(
                **asdict(
                    dLLMCacheConfig(
                        prompt_interval_steps=config.prompt_interval_steps,
                        gen_interval_steps=config.gen_interval_steps,
                        transfer_ratio=config.transfer_ratio,
                    )
                )
            )
            register_cache_LLaDA_V(model, "model.layers")

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)

        maybe_synchronize(device)
        start_time = time.time()

        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                steps=config.generation_steps,
                gen_length=config.generation_length,
                block_length=config.generation_block_length,
                tokenizer=tokenizer,
                stopping_criteria=list(config.stopping_criteria),
            )

        maybe_synchronize(device)
        generation_time = time.time() - start_time
        total_time += generation_time

        flattened = generated_ids.flatten()
        positions = torch.searchsorted(-ids_tensor, -flattened)
        recovered_points = point_tokenizer.indices_to_points(positions).to(torch.float32)

        if recovered_points.dim() == 1:
            if recovered_points.numel() % 2 != 0:
                raise RuntimeError(
                    f"Recovered point tensor has odd length: {recovered_points.numel()}"
                )
            recovered_points = recovered_points.view(-1, 2)
        elif recovered_points.dim() == 2:
            if recovered_points.size(-1) != 2:
                raise RuntimeError(
                    f"Recovered point tensor last dimension mismatch: expected 2, got {recovered_points.size(-1)}"
                )
        else:
            raise RuntimeError(
                f"Recovered point tensor has unexpected shape: {tuple(recovered_points.shape)}"
            )

        gt_points = torch.tensor(sample["action_targets"], dtype=torch.float32, device=recovered_points.device)

        ade, fde, steps_used, pred_len = compute_displacement_errors(recovered_points, gt_points)

        total_ade += ade * steps_used
        total_fde += fde
        total_steps += steps_used
        num_samples += 1

        sample_metrics = {
            "id": sample.get("id", idx),
            "ade": ade,
            "fde": fde,
            "steps_used": steps_used,
            "predicted_steps": pred_len,
            "ground_truth_steps": len(sample["action_targets"]),
            "generation_time_sec": generation_time,
        }
        per_sample_metrics.append(sample_metrics)

        if args.verbose:
            print(
                f"[{idx+1}/{len(data_samples)}] "
                f"id={sample_metrics['id']} "
                f"ADE={ade:.4f} FDE={fde:.4f} "
                f"steps_used={steps_used} pred_steps={pred_len} "
                f"time={generation_time:.3f}s"
            )

    if num_samples == 0 or total_steps == 0:
        print("No samples processed; nothing to report.")
        return

    dataset_ade = total_ade / total_steps
    dataset_fde = total_fde / num_samples
    avg_time = total_time / num_samples

    print("\n=== NuScenes Trajectory Metrics ===")
    print(f"Samples processed : {num_samples}")
    print(f"Total time (s)    : {total_time:.3f}")
    print(f"Average time (s)  : {avg_time:.3f}")
    print(f"ADE (avg over pts): {dataset_ade:.4f}")
    print(f"FDE (avg over seq): {dataset_fde:.4f}")

    if args.output:
        output_path = args.output
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": {
                "job_name": config.job_name,
                "data_path": str(config.data_path),
                "pretrained": str(config.pretrained),
                "lora_path": str(config.lora_path),
            },
            "metrics": {
                "samples": num_samples,
                "total_time_sec": total_time,
                "average_time_sec": avg_time,
                "ade": dataset_ade,
                "fde": dataset_fde,
            },
            "per_sample": per_sample_metrics,
        }
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
# """
# NuScenes inference script that computes displacement metrics for generated trajectories.

# This script mirrors the generation flow in `nuscene_inference_lora_time_vla.py` but, instead of
# only saving the decoded trajectories, it evaluates the predictions against the ground-truth
# ego waypoints provided in the dataset (`action_targets`). It reports the Average Displacement
# Error (ADE) and Final Displacement Error (FDE) aggregated across the processed samples.
# """

# from __future__ import annotations

# import argparse
# import copy
# import json
# import time
# import warnings
# from dataclasses import asdict
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import torch
# from PIL import Image
# from peft import PeftModel  # noqa: F401  # Imported to keep parity with other inference scripts.

# from llava.cache import dLLMCache, dLLMCacheConfig
# from llava.conversation import conv_templates
# from llava.hooks import register_cache_LLaDA_V
# from llava.mm_utils import process_images, tokenizer_image_token
# from llava.model.builder import load_pretrained_model
# from llava.constants import IMAGE_TOKEN_INDEX

# from train.config.nuscene_inference_vla import planning_config as config
# from tokenizer.example_usage import load_point_tokenizer


# ROOT = Path(__file__).resolve().parent.parent


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Run NuScenes LoRA inference and compute ADE/FDE.")
#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=None,
#         help="Optional cap on the number of data samples to process.",
#     )
#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=None,
#         help="Optional path to dump per-sample metrics as JSON.",
#     )
#     parser.add_argument(
#         "--disable-cache",
#         action="store_true",
#         help="Disable KV-cache usage even if enabled in the config.",
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Log detailed metrics for each processed sample.",
#     )
#     return parser.parse_args()


# def maybe_synchronize(device: torch.device) -> None:
#     if device.type == "cuda" and torch.cuda.is_available():
#         torch.cuda.synchronize(device)


# def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
#     dataset_path = Path(path)
#     if not dataset_path.exists():
#         raise FileNotFoundError(f"Dataset JSON not found at {dataset_path}")
#     with dataset_path.open("r") as f:
#         return json.load(f)


# def setup_model() -> Tuple[Any, Any, Any, int]:
#     tokenizer, model, image_processor, max_length = load_pretrained_model(
#         config.pretrained,
#         config.model_base,
#         config.model_name,
#         attn_implementation="sdpa",
#         device_map=config.device_map,
#     )
#     model.eval()
#     return tokenizer, model, image_processor, max_length


# def prepare_point_tokenizer(device: torch.device) -> Tuple[Any, torch.Tensor]:
#     weights_file = Path(config.tokenizer_weights)
#     if not weights_file.exists():
#         raise FileNotFoundError(f"Tokenizer weights not found at {weights_file}")

#     point_tokenizer = load_point_tokenizer(weights_file)

#     unused_token_ids_path = Path(config.unused_token_ids_path)
#     if not unused_token_ids_path.is_absolute():
#         unused_token_ids_path = ROOT / unused_token_ids_path

#     if not unused_token_ids_path.exists():
#         raise FileNotFoundError(f"Unused token ids file not found at {unused_token_ids_path}")

#     ids_tensor = torch.from_numpy(np.load(unused_token_ids_path)).to(device=device)
#     return point_tokenizer, ids_tensor


# def compute_displacement_errors(
#     predicted: torch.Tensor,
#     ground_truth: torch.Tensor,
# ) -> Tuple[float, float, int, int]:
#     """
#     Compute ADE and FDE between predicted and ground-truth trajectories.

#     Returns
#     -------
#     ade : float
#         Average displacement error across the overlapping horizon.
#     fde : float
#         Final displacement error at the last overlapping step.
#     steps_used : int
#         Number of points that were compared (overlap length).
#     total_pred_steps : int
#         Length of the predicted trajectory tensor.
#     """

#     if predicted.ndim != 2 or predicted.size(-1) != 2:
#         raise ValueError(f"Expected predicted points of shape (T, 2); got {tuple(predicted.shape)}")
#     if ground_truth.ndim != 2 or ground_truth.size(-1) != 2:
#         raise ValueError(f"Expected ground truth points of shape (T, 2); got {tuple(ground_truth.shape)}")

#     pred_len = predicted.size(0)
#     gt_len = ground_truth.size(0)
#     horizon = min(pred_len, gt_len)
#     if horizon == 0:
#         raise ValueError("Predicted and ground-truth trajectories must contain at least one point.")

#     pred_slice = predicted[:horizon]
#     gt_slice = ground_truth[:horizon]

#     diffs = pred_slice - gt_slice
#     dists = torch.linalg.norm(diffs, dim=1)

#     ade = float(dists.mean().item())
#     fde = float(dists[-1].item())
#     return ade, fde, horizon, pred_len


# def main() -> None:
#     args = parse_args()

#     print("Loading dataset...")
#     data_samples = load_dataset(config.data_path)
#     if args.limit is not None:
#         data_samples = data_samples[: args.limit]
#     print(f"Loaded {len(data_samples)} samples from {config.data_path}")

#     print("Loading model and tokenizer...")
#     tokenizer, model, image_processor, _ = setup_model()

#     device = torch.device(config.device)
#     point_tokenizer, ids_tensor = prepare_point_tokenizer(device)

#     total_time = 0.0
#     total_ade = 0.0
#     total_fde = 0.0
#     total_steps = 0
#     num_samples = 0

#     per_sample_metrics: List[Dict[str, Any]] = []

#     template_name = config.conv_template
#     cache_enabled = config.use_cache and not args.disable_cache

#     model.eval()

#     for idx, sample in enumerate(data_samples):
#         image = Image.open(sample["image"])
#         image_tensor = process_images([image], image_processor, model.config)
#         image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
#         image_sizes = [image.size]

#         conv = copy.deepcopy(conv_templates[template_name])
#         question = sample["conversations"][0]["value"]
#         conv.append_message(conv.roles[0], question)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#         if cache_enabled:
#             dLLMCache.new_instance(
#                 **asdict(
#                     dLLMCacheConfig(
#                         prompt_interval_steps=config.prompt_interval_steps,
#                         gen_interval_steps=config.gen_interval_steps,
#                         transfer_ratio=config.transfer_ratio,
#                     )
#                 )
#             )
#             register_cache_LLaDA_V(model, "model.layers")

#         input_ids = tokenizer_image_token(
#             prompt,
#             tokenizer,
#             IMAGE_TOKEN_INDEX,
#             return_tensors="pt",
#         ).unsqueeze(0).to(device)

#         maybe_synchronize(device)
#         start_time = time.time()

#         with torch.inference_mode():
#             generated_ids = model.generate(
#                 input_ids,
#                 images=image_tensor,
#                 image_sizes=image_sizes,
#                 steps=config.generation_steps,
#                 gen_length=config.generation_length,
#                 block_length=config.generation_block_length,
#                 tokenizer=tokenizer,
#                 stopping_criteria=list(config.stopping_criteria),
#             )

#         maybe_synchronize(device)
#         generation_time = time.time() - start_time
#         total_time += generation_time

#         flattened = generated_ids.flatten()
#         positions = torch.searchsorted(-ids_tensor, -flattened)
#         recovered_points = point_tokenizer.indices_to_points(positions).to(torch.float32)

#         if recovered_points.dim() == 1:
#             if recovered_points.numel() % 2 != 0:
#                 raise RuntimeError(
#                     f"Recovered point tensor has odd length: {recovered_points.numel()}"
#                 )
#             recovered_points = recovered_points.view(-1, 2)
#         elif recovered_points.dim() == 2:
#             if recovered_points.size(-1) != 2:
#                 raise RuntimeError(
#                     f"Recovered point tensor last dimension mismatch: expected 2, got {recovered_points.size(-1)}"
#                 )
#         else:
#             raise RuntimeError(
#                 f"Recovered point tensor has unexpected shape: {tuple(recovered_points.shape)}"
#             )

#         gt_points = torch.tensor(sample["action_targets"], dtype=torch.float32, device=recovered_points.device)

#         ade, fde, steps_used, pred_len = compute_displacement_errors(recovered_points, gt_points)

#         total_ade += ade * steps_used
#         total_fde += fde
#         total_steps += steps_used
#         num_samples += 1

#         sample_metrics = {
#             "id": sample.get("id", idx),
#             "ade": ade,
#             "fde": fde,
#             "steps_used": steps_used,
#             "predicted_steps": pred_len,
#             "ground_truth_steps": len(sample["action_targets"]),
#             "generation_time_sec": generation_time,
#         }
#         per_sample_metrics.append(sample_metrics)

#         if args.verbose:
#             print(
#                 f"[{idx+1}/{len(data_samples)}] "
#                 f"id={sample_metrics['id']} "
#                 f"ADE={ade:.4f} FDE={fde:.4f} "
#                 f"steps_used={steps_used} pred_steps={pred_len} "
#                 f"time={generation_time:.3f}s"
#             )

#     if num_samples == 0 or total_steps == 0:
#         print("No samples processed; nothing to report.")
#         return

#     dataset_ade = total_ade / total_steps
#     dataset_fde = total_fde / num_samples
#     avg_time = total_time / num_samples

#     print("\n=== NuScenes Trajectory Metrics ===")
#     print(f"Samples processed : {num_samples}")
#     print(f"Total time (s)    : {total_time:.3f}")
#     print(f"Average time (s)  : {avg_time:.3f}")
#     print(f"ADE (avg over pts): {dataset_ade:.4f}")
#     print(f"FDE (avg over seq): {dataset_fde:.4f}")

#     if args.output:
#         output_path = args.output
#         if output_path.suffix.lower() != ".json":
#             output_path = output_path.with_suffix(".json")
#         output_path.parent.mkdir(parents=True, exist_ok=True)

#         payload = {
#             "config": {
#                 "job_name": config.job_name,
#                 "data_path": str(config.data_path),
#                 "pretrained": str(config.pretrained),
#                 "lora_path": str(config.lora_path),
#             },
#             "metrics": {
#                 "samples": num_samples,
#                 "total_time_sec": total_time,
#                 "average_time_sec": avg_time,
#                 "ade": dataset_ade,
#                 "fde": dataset_fde,
#             },
#             "per_sample": per_sample_metrics,
#         }
#         with output_path.open("w") as f:
#             json.dump(payload, f, indent=2)
#         print(f"Wrote metrics to {output_path}")


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()

