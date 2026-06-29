import argparse
import json
import re
from typing import Tuple

import torch

# Model action representation: absolute waypoints (False) or relative
# displacements (True). The paper uses absolute waypoints.
DELTA = False


def compute_displacement_errors(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Tuple[float, float, int, int]:
    """Compute ADE / FDE between predicted and ground-truth trajectories."""
    if predicted.ndim != 2 or predicted.size(-1) != 2:
        raise ValueError(f"Expected predicted points of shape (T, 2); got {tuple(predicted.shape)}")
    if ground_truth.ndim != 2 or ground_truth.size(-1) != 2:
        raise ValueError(f"Expected ground truth points of shape (T, 2); got {tuple(ground_truth.shape)}")

    horizon = min(predicted.size(0), ground_truth.size(0))
    if horizon == 0:
        raise ValueError("Predicted and ground-truth trajectories must contain at least one point.")

    dists = torch.linalg.norm(predicted[:horizon] - ground_truth[:horizon], dim=1)
    return float(dists.mean().item()), float(dists[-1].item()), horizon, predicted.size(0)

def extract_trajectory(text):
    """
    Extracts trajectory points from text.
    Matches patterns like [1.23, 4.56] or [1.23, 4.56], ...
    Returns a torch tensor of shape (N, 2).
    """
    if not isinstance(text, str):
        return torch.zeros((0, 2))
        
    # Pattern to match [x, y]
    # Handles optional spaces, negative signs, scientific notation
    pattern = r"\[\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*\]"
    matches = re.findall(pattern, text)
    
    traj = []
    for x, y in matches:
        traj.append([float(x), float(y)])
    
    if not traj:
        return torch.zeros((0, 2))
        
    return torch.tensor(traj, dtype=torch.float32)

def compute_metric(entries, steps=10, specific=False):
    """
    steps: number of points, assuming 2Hz (2 steps per second).
    specific: if True, compute ADE for that exact horizon only (e.g., 1s/2s/3s).
    """
    total_ade = 0.0
    total_fde = 0.0
    total_steps = 0
    num_samples = 0

    if not specific:
        min_idx = 0
        max_idx = steps
    else:
        min_idx = steps - 1
        max_idx = steps

    for item in entries:
        # item structure: [pred_traj_str, pred_text_str, gt_traj_text_str, ...]
        if len(item) < 3:
            continue

        pred_str = item[0]
        gt_str = item[2]

        pred = extract_trajectory(pred_str)
        gt = extract_trajectory(gt_str)

        if pred.size(0) == 0 or gt.size(0) == 0:
            continue

        if DELTA:
            pred = torch.cumsum(pred, dim=0)
            gt = torch.cumsum(gt, dim=0)

        horizon = min(max_idx, pred.size(0), gt.size(0))
        if horizon <= min_idx:
            continue

        pred_slice = pred[min_idx:horizon]
        gt_slice = gt[min_idx:horizon]

        try:
            ade, fde, steps_used, _ = compute_displacement_errors(pred_slice, gt_slice)
            total_ade += ade * steps_used
            total_fde += fde
            total_steps += steps_used
            num_samples += 1
        except ValueError:
            continue

    if total_steps == 0:
        return None

    dataset_ade = total_ade / total_steps
    dataset_fde = total_fde / num_samples

    num_seconds = steps * 0.5
    if specific:
        print(f"ADE@{num_seconds:.1f}s={dataset_ade:.4f}")
    else:
        print(f"ADE->{num_seconds:.1f}s={dataset_ade:.4f}, FDE={dataset_fde:.4f}")
    return dataset_ade, dataset_fde


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute nuScenes planning L2 (ADE/FDE) from a results JSON."
    )
    parser.add_argument(
        "--results", required=True,
        help="Inference results JSON: list of [pred_traj, explanation, gt_traj].",
    )
    args = parser.parse_args()

    with open(args.results) as f:
        entries = json.load(f)

    # L2 displacement error at 1s / 2s / 3s horizons (2 Hz waypoints).
    for steps in [2, 4, 6]:
        compute_metric(entries, steps=steps, specific=True)

    # Average L2 up to 3s and 5s.
    for steps in [6, 10]:
        compute_metric(entries, steps=steps, specific=False)
