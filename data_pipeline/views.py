"""Model-specific views built from canonical samples (prompt formats v0.1, all [proposed]).

* ``pilot_example``: supervised example for the fast pilot (Qwen3.5). Variants differ only in
  their temporal context, which is what claim C1 compares:
  - ``front``          current front image only
  - ``front_motion``   current front image + object-motion text from tracks
  - ``front_history``  the 4 history front images (the multi-frame baseline)
* ``teacher_request``: hindsight prompt for an offline teacher VLM that writes narration and
  reasoning labels for the slow reasoner. It contains privileged information on purpose.
* ``reasoner_example``: supervised example for the reasoner (Gemma 4) built from a teacher label.
  Its input holds only test-time information.

Examples use the common ``{"images": [...], "messages": [...]}`` layout with one ``<image>``
placeholder per image, which Qwen-VL / Gemma fine-tuning stacks accept or convert easily.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from data_pipeline import labels, motion_text

PILOT_PROMPT_VERSION = "pilot-v0.1"
TEACHER_PROMPT_VERSION = "teacher-v0.1"
REASONER_PROMPT_VERSION = "reasoner-v0.1"
PILOT_VARIANTS = ("front", "front_motion", "front_history")
FRONT_CAMERA = "CAM_F0"
NUM_TARGET_POSES = 8  # NAVSIM scores 8 poses at 0.5 s (4 s)

FRAME_NOTE = (
    "All coordinates are metres in the current ego frame (x forward, y left); "
    "headings are degrees, counter-clockwise."
)


def _xy_list(points, precision: int = 2) -> str:
    return ", ".join(f"[{x:.{precision}f}, {y:.{precision}f}]" for x, y in np.asarray(points)[:, :2])


def _image_paths(sample: Dict[str, Any], sensor_root: str, history: bool) -> List[str]:
    paths = sample["sensors"]["cameras"][FRONT_CAMERA]
    chosen = paths if history else paths[-1:]
    if any(p is None for p in chosen):
        raise ValueError(f"{sample['sample_id']}: missing {FRONT_CAMERA} image")
    return [os.path.join(sensor_root, p) for p in chosen]


def _ego_state_lines(sample: Dict[str, Any]) -> List[str]:
    ego = sample["ego"]
    vx, vy = ego["history_velocity"][-1]
    ax, ay = ego["history_acceleration"][-1]
    return [
        f"Route command: {labels.describe_command(ego['history_driving_command'][-1])}.",
        f"Ego speed: {math.hypot(vx, vy):.1f} m/s; acceleration: {ax:+.1f} m/s^2 forward, {ay:+.1f} m/s^2 left.",
        f"Ego past positions (oldest first, every {sample['timeline']['dt_s']:g} s): "
        f"{_xy_list(ego['history_poses'])}.",
    ]


def pilot_example(
    sample: Dict[str, Any],
    sensor_root: str,
    variant: str = "front_motion",
    max_objects: int = 12,
    guidance: Optional[str] = None,
) -> Dict[str, Any]:
    if variant not in PILOT_VARIANTS:
        raise ValueError(f"unknown pilot variant {variant}")
    images = _image_paths(sample, sensor_root, history=(variant == "front_history"))
    lines = ["<image>" * len(images), "You are driving the ego car. " + FRAME_NOTE]
    lines += _ego_state_lines(sample)
    if variant == "front_motion":
        lines.append("Nearby objects, tracked over the last 1.5 s:\n" + motion_text.motion_context(sample["agents"], max_objects))
    if guidance:
        lines.append(f"Planner guidance: {guidance}.")
    lines.append(f"Predict the next {NUM_TARGET_POSES} ego positions, every 0.5 s, as [x, y] pairs.")
    target = sample["target"]["future_poses"][:NUM_TARGET_POSES]
    return {
        "id": f"{sample['sample_id']}:{variant}",
        "sample_id": sample["sample_id"],
        "variant": variant,
        "prompt_version": PILOT_PROMPT_VERSION,
        "agents_source": sample["agents"]["source"],
        "images": images,
        "messages": [
            {"role": "user", "content": "\n".join(lines)},
            {"role": "assistant", "content": _xy_list(target)},
        ],
        "target_poses": target,
    }


def _future_summary(sample: Dict[str, Any]) -> str:
    poses = np.asarray(sample["target"]["future_poses"])
    picks = [i for i in (1, 3, 5, 7) if i < len(poses)]  # 1, 2, 3, 4 s
    parts = [f"t+{(i + 1) * sample['timeline']['dt_s']:g}s at ({poses[i, 0]:.1f}, {poses[i, 1]:.1f})" for i in picks]
    return "; ".join(parts)


def _agents_future_summary(sample: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    future = {t["track_token"]: t["states"] for t in sample["privileged"]["agents_future"]["tracks"]}
    out = []
    for i, row in enumerate(rows, 1):
        states = future.get(row["track_token"])
        if not states:
            out.append(f"{i}. {row['category']}: not observed later")
            continue
        seen = [(k, s) for k, s in enumerate(states) if s is not None]
        k, s = seen[-1]
        out.append(f"{i}. {row['category']}: at t+{(k + 1) * sample['timeline']['dt_s']:g}s at ({s[0]:.1f}, {s[1]:.1f})")
    return "\n".join(out) if out else "none"


def teacher_request(sample: Dict[str, Any], sensor_root: str, max_objects: int = 8, max_tokens: int = 256) -> Dict[str, Any]:
    """Prompt asking a large VLM to explain, in hindsight, the driving decision actually taken."""
    action = labels.meta_action(sample)
    rows = motion_text.summarize_tracks(sample["agents"], max_objects=max_objects)
    text = "\n".join(
        [
            "You are labelling driving data. The image is the ego car's front camera right now. " + FRAME_NOTE,
            *_ego_state_lines(sample),
            "Nearby objects now:\n" + motion_text.render(rows),
            "",
            "For your reference only (recorded future, never mention that you know it):",
            f"- The ego car then drove: {_future_summary(sample)}.",
            f"- Its manoeuvre: longitudinal={action['longitudinal']}, lateral={action['lateral']}.",
            "- Where the listed objects went:\n" + _agents_future_summary(sample, rows),
            "",
            "Write as the driver thinking at this moment. Return only JSON with keys:",
            '"narration": 1-2 sentences on what is happening around the ego car;',
            '"reasoning": 1-3 sentences on why driving this way is right, naming the specific objects or road features that matter;',
            '"decision": {"longitudinal": ..., "lateral": ...} copied from the manoeuvre above.',
        ]
    )
    image = _image_paths(sample, sensor_root, history=False)[0]
    return {
        "custom_id": sample["sample_id"],
        "prompt_version": TEACHER_PROMPT_VERSION,
        "meta_action": action,
        "image_path": image,
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{image}"}},
                        {"type": "text", "text": text},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
    }


def reasoner_example(sample: Dict[str, Any], label: Dict[str, Any], sensor_root: str, max_objects: int = 12) -> Dict[str, Any]:
    """Supervised reasoner example: test-time inputs only -> narration, reasoning, decision."""
    images = _image_paths(sample, sensor_root, history=False)
    lines = ["<image>", "You are the reasoning module of a self-driving car. " + FRAME_NOTE]
    lines += _ego_state_lines(sample)
    lines.append("Nearby objects, tracked over the last 1.5 s:\n" + motion_text.motion_context(sample["agents"], max_objects))
    lines.append("Describe the situation, explain how to drive and give the decision.")
    decision = label.get("decision") or {}
    answer = "\n".join(
        [
            f"Narration: {label['narration'].strip()}",
            f"Reasoning: {label['reasoning'].strip()}",
            f"Decision: longitudinal={decision.get('longitudinal')}, lateral={decision.get('lateral')}",
        ]
    )
    return {
        "id": sample["sample_id"],
        "prompt_version": REASONER_PROMPT_VERSION,
        "label_source": label.get("label_source"),
        "images": images,
        "messages": [{"role": "user", "content": "\n".join(lines)}, {"role": "assistant", "content": answer}],
    }
