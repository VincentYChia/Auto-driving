"""Export NAVSIM scenes as MVLAD-AD inference inputs (run in the NAVSIM env; CPU only).

Each output record holds the current front-camera image path (CAM_F0) and the planning prompt,
built only from what a NAVSIM agent may see (AgentInput: 4 ego poses, velocity, acceleration,
driving command). It does not read maps, annotations or future frames.

    python -m navsim_tools.export_mvlad_inputs --split navtest --output $NAVSIM_EXP_ROOT/mvlad/navtest_inputs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navsim_tools import loader as nav_loader
from navsim_tools import mvlad_prompt, raw_frames

FRONT_CAMERA = "CAM_F0"


def build_records(scene_loader, num_history_frames: int, sensor_path: Path, split: str, history_mode: str):
    """Return ``(records, missing_images, max_pose_gap)`` for every scene in ``scene_loader``."""
    current_idx = num_history_frames - 1
    records, missing_images, max_pose_gap = [], 0, 0.0
    for token in sorted(scene_loader.tokens):
        frames = scene_loader.scene_frames_dicts[token]
        agent_input = scene_loader.get_agent_input_from_token(token)  # exactly what a NAVSIM agent receives
        history = np.stack([np.asarray(s.ego_pose, dtype=np.float64) for s in agent_input.ego_statuses])
        ours = raw_frames.ego_history_local(frames, current_idx)
        max_pose_gap = max(max_pose_gap, float(np.abs(ours[:, :2] - history[:, :2]).max()))

        rel_image = raw_frames.camera_paths(frames[current_idx]).get(FRONT_CAMERA)
        image_path = str(sensor_path / rel_image) if rel_image else None
        if image_path is None or not Path(image_path).exists():
            missing_images += 1
        current = agent_input.ego_statuses[-1]
        records.append(
            {
                "id": token,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": mvlad_prompt.build_human_message(history[:, :2], history_mode)},
                    {"from": "gpt", "value": ""},
                ],
                "navsim": {
                    "split": split,
                    "log_name": frames[current_idx]["log_name"],
                    "history_mode": history_mode,
                    "ego_history_local": np.round(history, 4).tolist(),
                    "ego_velocity": np.asarray(current.ego_velocity, dtype=float).tolist(),
                    "ego_acceleration": np.asarray(current.ego_acceleration, dtype=float).tolist(),
                    "driving_command": [int(v) for v in np.asarray(current.driving_command).reshape(-1)],
                },
            }
        )
    return records, missing_images, max_pose_gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="navtest", help="NAVSIM train_test_split name, e.g. navtest, navmini")
    parser.add_argument("--navsim-log-path", default=None, help="default: $OPENSCENE_DATA_ROOT/navsim_logs/<data_split>")
    parser.add_argument("--sensor-blobs-path", default=None, help="default: $OPENSCENE_DATA_ROOT/sensor_blobs/<data_split>")
    parser.add_argument("--history-mode", default="pad_cv", choices=["pad_cv", "available"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scene_filter, data_split = nav_loader.load_split(args.split)
    if args.max_scenes is not None:
        scene_filter.max_scenes = args.max_scenes
    log_path, sensor_path = nav_loader.default_paths(data_split, args.navsim_log_path, args.sensor_blobs_path)
    scene_loader = nav_loader.make_scene_loader(log_path, sensor_path, scene_filter)
    records, missing_images, max_pose_gap = build_records(
        scene_loader, scene_filter.num_history_frames, sensor_path, args.split, args.history_mode
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records)} records -> {out}")
    print(f"front images missing on disk: {missing_images}")
    print(f"max |our ego history - NAVSIM AgentInput| = {max_pose_gap:.2e} m (should be ~0)")
    if max_pose_gap > 1e-3:
        raise SystemExit("ego-history mismatch with NAVSIM; check raw_frames against this NAVSIM version")


if __name__ == "__main__":
    main()
