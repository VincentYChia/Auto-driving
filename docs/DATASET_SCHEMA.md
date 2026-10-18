# Dataset schema v0.1 [proposed]

Training data for the fast pilot (Qwen3.5) and the slow reasoner (Gemma 4) is stored in one canonical format. Model inputs are derived from it.

- **Code:** `data_pipeline/schema.py` (builder + validator), `data_pipeline/extract_navsim.py`, `data_pipeline/views.py`.
- **Tests:** `tests/`. These include runs against the real NAVSIM devkit on synthetic logs.

## Why it is shaped this way

| Requirement (from `RESEARCH_DIRECTION.md`) | Design choice |
|---|---|
| C1: compare object-motion context with multi-frame images **on the same samples** | Each record stores object tracks over the history **and** paths to every camera image over the history. Variants are views, not separate datasets. |
| C1: ground-truth tracks vs a deployable tracker | `agents.source` names where tracks came from. A tracker run later writes the same `agents` block with `source: "tracker:<name>@<version>"`. |
| C2: try many action tokenizations | Targets are stored as raw continuous poses. Tokenization happens in the training loader. |
| Never leak future or map information into test-time inputs | Anything an agent cannot see at test time lives under `privileged`. Views take student inputs only from outside it; a test checks the reasoner view. |
| Reproducibility | `schema_version`, plus provenance (extractor version, git commit, NAVSIM version, time) on every record. Prompt formats are versioned separately. |
| NAVSIM rules | Extraction refuses test splits (navtest / navhard / warmup / private). |

**Conventions** (same as NAVSIM):
- Metres and radians.
- Poses are `[x, y, yaw]` in the **current ego frame** (rear axle at the current frame, x forward, y left, yaw counter-clockwise).
- Frames are 0.5 s apart; history is the 4 frames t = −1.5 … 0 s.
- Object positions from past frames are re-expressed in the current ego frame (ego-motion compensated). Object velocities are rotated into current-frame axes.

## Canonical record (one JSON object per scene token; gzipped JSONL shards)

```jsonc
{
  "schema_version": "0.1.0",
  "sample_id": "<NAVSIM token of the current frame>",
  "source":   {"dataset": "navsim", "split": "navtrain", "data_split": "trainval",
               "log_name": "...", "scene_token": "...", "map_location": "...", "timestamp_us": 0},
  "timeline": {"dt_s": 0.5, "history_offsets_s": [-1.5, -1.0, -0.5, 0.0],
               "future_offsets_s": [0.5, ..., 5.0], "history_tokens": [...], "future_tokens": [...]},
  "ego": {
    "history_poses":           [[x, y, yaw] x4],      // last one is [0, 0, 0]
    "history_velocity":        [[vx, vy] x4],          // each frame's own ego axes, as NAVSIM provides
    "history_acceleration":    [[ax, ay] x4],
    "history_driving_command": [[left, straight, right, unknown] x4]
  },
  "target": {"future_poses": [[x, y, yaw] x10]},       // NAVSIM scores the first 8 (4 s)
  "agents": {                                          // test-time-observable history
    "source": "gt_annotations",
    "frame_offsets_s": [-1.5, -1.0, -0.5, 0.0],
    "max_range_m": 100.0,
    "tracks": [{"track_token": "...", "category": "vehicle", "size_lwh": [l, w, h],
                "states": [[x, y, z, yaw, vx, vy] | null  x4]}]   // null = not annotated in that frame
  },
  "privileged": {                                      // training-only; never a student input
    "agents_future": {"frame_offsets_s": [0.5, ..., 5.0], "tracks": [... same layout ...]},
    "roadblock_ids": [...], "traffic_lights": [[lane_connector_id, is_red], ...]
  },
  "sensors": {                                         // paths relative to sensor_blobs/<data_split>
    "data_split": "trainval",
    "cameras": {"CAM_F0": [path x4], "CAM_L0": [...], "...": "8 cameras"},
    "lidar": [path x4],
    "calibration": {"CAM_F0": {"cam_intrinsic": [...], "sensor2lidar_rotation": [...],
                               "sensor2lidar_translation": [...], "distortion": [...]}}
  },
  "provenance": {"extractor": "data_pipeline.extract_navsim", "extractor_version": "0.1.0",
                 "git_commit": "...", "navsim_version": "...", "created_utc": "..."}
}
```

**Precision:** positions 3 decimals (mm), yaw 4, velocities 3. NaN in the logs becomes `null`.

**Size:** to be measured. The extraction manifest reports counts; navtrain has ≈103k tokens.

## Views (derived; prompt formats are versioned)

| File | For | Input | Target |
|---|---|---|---|
| `pilot_front.jsonl` | Qwen pilot, C1 baseline | current CAM_F0 + ego state + command + ego history | 8 future `[x, y]` |
| `pilot_front_motion.jsonl` | Qwen pilot, C1 | the above + object-motion text (nearest 12 dynamic objects within 60 m) | same |
| `pilot_front_history.jsonl` | Qwen pilot, C1 multi-frame baseline | 4 history CAM_F0 images + ego state | same |
| `teacher_requests.jsonl` | offline teacher VLM | image + state + **privileged** future (ego path, rule-based manoeuvre, where objects went) | JSON narration / reasoning / decision |
| `reasoner.jsonl` | Gemma reasoner | image + state + object motion (**no** future) | `Narration: … Reasoning: … Decision: …` |

**Layout:** examples use `{"images": [...], "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` with one `<image>` placeholder per image. Raw `target_poses` are kept for tokenized training.

**Example pilot prompt** (`pilot-v0.1`, variant `front_motion`):
```
<image>
You are driving the ego car. All coordinates are metres in the current ego frame (x forward, y left); headings are degrees, counter-clockwise.
Route command: straight.
Ego speed: 10.0 m/s; acceleration: +0.0 m/s^2 forward, +0.8 m/s^2 left.
Ego past positions (oldest first, every 0.5 s): [-14.96, 0.90], [-9.99, 0.40], [-5.00, 0.10], [0.00, 0.00].
Nearby objects, tracked over the last 1.5 s:
1. pedestrian at (4.0, 3.9) m, heading +28 deg, speed unknown, moved (+0.0, +0.0) m over 0.5 s
2. vehicle at (24.8, -8.1) m, heading -48 deg, speed 6.3 m/s, moved (+6.3, -7.1) m over 1.5 s
Predict the next 8 ego positions, every 0.5 s, as [x, y] pairs.
```

## Rule-based meta-actions (`data_pipeline/labels.py`, rules v0.1, thresholds [proposed])

These are derived from the recorded future, so they are **targets only**. They give the reasoner → pilot interface a fixed, checkable vocabulary.

- **Longitudinal**, from current speed vs mean speed over 2.5–3 s: `stopped`, `stop`, `start`, `accelerate`, `decelerate`, `keep_speed`. The change threshold is max(1 m/s, 15% of current speed).
- **Lateral**, from the pose at 4 s:
  - `turn_left/right`: heading change > 25°.
  - `lane_change_left/right`: heading change < 15° and an *unexplained* lateral offset > 1.5 m, i.e. beyond what a constant-curvature path with that heading change would give. A test caught gentle curves being mislabelled before this correction.
  - `curve_left/right`: heading change > 8°.
  - `straight`; `stationary` if forward travel is < 2 m.

## Reasoning labels (`data_pipeline/label_client.py`)

Teacher replies are stored one per line:
```jsonc
{"custom_id": "<sample_id>", "narration": "...", "reasoning": "...",
 "decision": {"longitudinal": "...", "lateral": "..."}, "raw": null,
 "label_source": {"model": "...", "teacher_prompt_version": "teacher-v0.1", "temperature": 0.2, "created_utc": "..."}}
```

- **Hindsight labelling:** the teacher sees the recorded future so its reasoning explains what the driver actually did. It is told not to mention that future.
- **Parse failures:** rows whose reply didn't parse keep `raw` and are skipped when views are built.

## Open decisions [open]
1. **Object filter:** keep objects within 100 m in the record; show the nearest 12 dynamic objects within 60 m in prompts?
2. **Motion-text format:** precision, fields, ordering. This is part of the prompt-engineering work package.
3. **Teacher model** for reasoning labels, and its licence for training use.
4. **nuScenes + Nu-X:** add a nuScenes converter to this same schema, for MVLAD-AD comparability and human-written reasoning?
5. **Tracker-derived tracks** for C1: detector + tracker choice, then the same `agents` block with a different `source`.
6. **Synthetic NAVSIM scenes** (navhard-style second stage) as extra training data? Allowed only from training splits.
