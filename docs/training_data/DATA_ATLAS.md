# Training-data atlas: what the raw data looks like, and how it becomes training samples

**Draft, 2026-10-22. Planning only [proposed]. No pipeline code.**
- **Scope [agreed]:** gather frames with their recorded driving pattern, plus reasoning text where it exists. Where it doesn't, we write our own.
- Discretisation, cleaning and balancing come after the canonical layer.

**Evidence marks** (`docs/RESEARCH_DIRECTION.md`):
- **V:** read in an official repo, doc, proto, or a real file.
- **R:** a third-party printout or a snippet.
- **D:** derived by us.
- Code blocks are **verbatim real records** unless marked *schema* (field names from official code, no real values) or *illustrative* (values invented, field names real).

**Contents:**
- §0 the pipeline;
- §1 the canonical frame;
- §2 each source's raw data;
- §3 field mapping;
- §4 one frame end to end;
- §5 labels and reasoning;
- §6 volumes;
- §7 pitfalls to audit;
- §8 decisions needed.

## 0. The pipeline on one page

```
① RAW, as downloaded, never edited    ② ADAPTER, one per source               ③ CANONICAL FRAME, one per kept image
────────────────────────────────────  ──────────────────────────────────────  ──────────────────────────────────────────
NAVSIM/OpenScene  log .pkl + JPG       pick anchor images t (≤ 2 Hz)           id       source · log · t · split · licence
nuScenes          JSON tables + JPG    front image + its calibration           image    file · W×H · lens model · HFOV
                  + CAN JSON           ego poses at t−1.5 … t+4.0 s,           path     12 poses, 0.5 s apart, in the
Waymo / WOD-E2E   parquet / TFRecord     every 0.5 s → rear-axle frame at t             rear-axle frame at t (x fwd, y left)
nuReasoning       clip folders         speed · accel · yaw rate at t           motion   speed · accel · yaw rate
PhysicalAI-AV     mp4 + parquet        command: native, or route rule          command  left/straight/right/unknown + how
L2D               LeRobot parquet+mp4  existing text, if any                   text     existing text + author
CoVLA, ZOD        mp4/jpg + json/hdf5                                          flags    quality checks
                                                                                   │
            ┌──────────────────────────────────────────────────────────────────────┘
            ▼
④ LABELS (versioned)                          ⑤ VIEWS = TRAINING SAMPLES (regenerable)
   decision  ← rules on the recorded path        pilot     image + short prompt  →  path tokens (8 poses × 0.5 s)
   reasoning ← existing text, or a teacher VLM   reasoner  image + short prompt  →  decision · reasoning · narration
```

The **driving pattern** is the path the human driver actually drove next: the 8 future poses. Every source has it; that is the core label. Reasoning text is the optional extra.

## 1. The canonical frame: what we extract from every source [proposed]

One record per kept front image. Every source is converted to this; nothing downstream reads raw formats.

| Group | Contents | Units / convention |
|---|---|---|
| **id** | source, split, log or clip id, frame id, time t, city / country, licence tag | t in µs |
| **image** | path into the raw store (for video sources, the video file + frame time, or a JPEG extracted once), camera name, width × height, lens model (pinhole / OpenCV-5 / f-theta / fisheye), intrinsics, distortion, horizontal FOV, mount position, image-to-pose time offset if known | px, degrees, m |
| **path** | 12 poses at t + {−1.5, −1.0, −0.5, 0, +0.5, …, +4.0} s, each (x, y, yaw) plus a validity flag | metres and radians, in the **rear-axle frame at t**: x forward, y left |
| **motion** | speed, longitudinal and lateral acceleration, yaw rate at t, and whether each is measured or differenced | m/s, m/s², rad/s |
| **command** | left / straight / right / unknown, and **how it was obtained** (route rule, native intent, turn signal, or derived from the path) | — |
| **text** | each existing text item: kind (narration, reasoning, QA, caption, causal chain, instruction), author (human, model name, rules), source licence | — |
| **flags** | pose jump, image–pose sync, path-projection check, standstill, near-duplicate group, traffic side (left- or right-hand) | — |

## 2. What one moment looks like in each source

### 2.1 NAVSIM / OpenScene (navtrain): the target domain
| | |
|---|---|
| **One moment** | One dict in a per-log pickle (2 Hz) + 8 JPEGs + 1 merged LiDAR `.pcd` (V) |
| **Front camera** | `CAM_F0`, 1920×1080 (D). f = 1545 px (R). OpenCV 5-coefficient distortion, k1 = −0.356 (R), so the images are distorted. ≈64° FOV pinhole, ≈72° once undistorted (D). Native 10 Hz, kept at 2 Hz (V) |
| **Driving pattern** | Rear-axle pose in the nuPlan map frame, every 0.5 s (V). `ego_dynamic_state = [vx, vy, ax, ay]` in the ego frame (V) |
| **Command** | Route-based one-hot (left, forward, right, unknown). Look 20 m ahead along the route centreline: an offset of ≥ +2 m is left, ≤ −2 m is right (V, OpenScene code) |
| **Text** | None (V). ReCogDrive's QA covers navtrain externally (§2.10) |
| **Yield** | 103,288 navtrain samples from 1,192 logs (V). ≈0.8M frames in all of trainval (D) |
| **Access** | Hugging Face; the NAVSIM scripts need no login. CC BY-NC-SA 4.0 + nuPlan non-commercial terms (V) |

On disk (V):
```
dataset/navsim_logs/trainval/2021.05.12.22.00.38_veh-35_01008_01518.pkl      # list of frame dicts
dataset/sensor_blobs/trainval/2021.05.12.22.00.38_veh-35_01008_01518/
        CAM_F0/c082c104b7ac5a71.jpg  CAM_L0/ … CAM_B0/  MergedPointCloud/0079e06969ed5625.pcd
```
One frame (keys and types exact from the official generator; the CAM_F0 calibration is a real nuPlan print (R); `<…>` = placeholder):
```python
{'token': '<16-hex>', 'timestamp': <µs, LiDAR time>, 'log_name': '2021.05.12.22.00.38_veh-35_01008_01518',
 'map_location': 'us-nv-las-vegas-strip', 'roadblock_ids': ['<id>', ...],
 'ego2global_translation': array([x, y, z]),      # rear axle, map frame, m
 'ego2global_rotation':    array([qw, qx, qy, qz]),
 'ego_dynamic_state':      [vx, vy, ax, ay],       # ego frame
 'driving_command':        array([0, 1, 0, 0]),    # left, forward, right, unknown
 'cams': {'CAM_F0': {'data_path': '<log>/CAM_F0/<token>.jpg',
                     'cam_intrinsic': [[1545, 0, 960], [0, 1545, 560], [0, 0, 1]],
                     'distortion': [-0.356123, 0.172545, -0.00213, 0.000464, -0.05231],
                     'sensor2lidar_translation': [1.655, -0.012, 1.491], ...}, 'CAM_L0': {...}, ...},
 'anns': {'gt_boxes': (N,7), 'gt_names': (N,), 'gt_velocity_3d': (N,3), 'track_tokens': [...]}}
```
**Cutting a sample:** frames i−3 … i+8. Convert to the pose of frame i (NAVSIM's own `convert_absolute_to_relative_se2_array`; our `navsim_tools/raw_frames.py` does the same) (V).

**Watch out:**
- Images are distorted.
- The image is up to 50 ms off the pose time, and that offset is not stored.
- `gt_velocity_3d` is in the LiDAR frame.
- 14-frame windows can cross nuPlan scene boundaries.

### 2.2 nuScenes
| | |
|---|---|
| **One moment** | Relational JSON rows (sample → sample_data → ego_pose, calibrated_sensor) + JPEG + per-scene CAN JSON (V) |
| **Front camera** | `CAM_FRONT` 1600×900 (R), f = 1266 px → 64.6° (D). **Undistorted** (V). ≈12 Hz (R) |
| **Driving pattern** | One global rear-axle pose per image (V / R). CAN `pose` at 50 Hz with `vel` and `accel` (V) |
| **Command** | None (V). Options: CAN route polyline (V), turn signals (V), or a path-derived rule (VAD-style, hindsight) |
| **Text** | `scene.description` (V). Nu-X: human narration + reasoning per keyframe (V) |
| **Yield** | ≈25k keyframe windows in trainval (D) |
| **Access** | Account + terms. Non-commercial (V) |

Real records (scene-0061, Singapore; printed by the official devkit, R):
```python
sample_data_CAM_FRONT = {'token': 'e3d495d4ac534d54b321f50006683844', 'timestamp': 1532402927612460,
  'filename': 'samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg',
  'width': 1600, 'height': 900, 'is_key_frame': True, 'ego_pose_token': 'e3d495d4ac534d54b321f50006683844', ...}
ego_pose = {'timestamp': 1532402927612460, 'translation': [411.4199861830012, 1181.197175631848, 0.0],
            'rotation': [0.5720063498929273, -0.0021434844534272707, 0.011564094980151613, -0.8201648693182716]}
calibrated_sensor = {'translation': [1.70079118954, 0.0159456324149, 1.51095763913],
                     'camera_intrinsic': [[1266.417203046554, 0.0, 816.2670197447984],
                                          [0.0, 1266.417203046554, 491.50706579294757], [0.0, 0.0, 1.0]]}
scene.description = 'Parked truck, construction, intersection, turn left, following a van'
can_bus/scene-0001_pose.json[0] = {'utime': 1531883530467511, 'pos': [1010.14, 610.89, 0.0],
  'vel': [4.1688763951334185, 0.0, 0.0], 'accel': [0.0525, 0.9292, 9.5138], 'rotation_rate': [0.0403, -0.0026, 0.2849]}
Nu-X['e93e98b63d3b40209056d129dc53ceee'] = {"narration": "the vehicle executes a left turn at the intersection",
                                            "reasoning": "to navigate the junction as per its predetermined route"}
```
**Cutting a sample:** anchor on a CAM_FRONT image at t. Interpolate the pose track at the 12 offsets and rotate into the pose at t. Speed and acceleration come from CAN `vel[0]` and `accel[0]` (D).

**Watch out:**
- Keyframes are 0.40–0.60 s apart, so interpolate.
- CAN `accel` includes gravity.
- Singapore drives on the left.
- The devkit flags 87 scenes with misaligned routes (V).

### 2.3 Waymo Open Dataset: Perception
| | |
|---|---|
| **One moment** | v2: one parquet per component, joined on (segment, timestamp, camera). v1: one `Frame` proto (V) |
| **Front camera** | `FRONT` 1920×1280, f = 2056 px → 50° (V values, D FOV). Distorted; rolling shutter (V) |
| **Driving pattern** | A 4×4 vehicle→global pose per frame and per image, 10 Hz (V). Velocity is given **in the global frame** (V) |
| **Command / text** | None (V) |
| **Yield** | ≈30 samples per 20 s segment, ≤61k total (D) |
| **Access** | Google sign-in + Waymo non-commercial licence (V) |

Real frame (official `tutorial/frames`, parsed, V):
```text
context.name = "11004685739714500220_2300_000_2320_000"   (location_sf, Day, sunny)
FRONT intrinsic [f_u,f_v,c_u,c_v,k1,k2,p1,p2,k3] = [2055.556, 2055.556, 939.657, 641.072, 0.0323, -0.3214, 0.00079, -0.00063, 0.0]
FRONT width 1920, height 1280, rolling_shutter LEFT_TO_RIGHT; JPEG 349,103 bytes
timestamp_micros = 1553640277206678;  image pose_timestamp = 1553640277.2595 (52.8 ms later)
pose vehicle->global = [[-0.9136, -0.4062, -0.0193, -4069.03], [0.4066, -0.9131, -0.0304, 11526.31], [-0.0053, -0.0357, 0.9994, 86.50], [0,0,0,1]]
image velocity (global) v = (-3.399, 1.509, -0.017) m/s   -> (3.719, 0.003, 0.003) m/s in the vehicle frame (D)
```
**Cutting a sample:** frames k−15, −10, −5, 0, +5, …, +40. Take relative SE(2) poses, rotate the velocity into the vehicle frame, and difference for acceleration (D).

### 2.4 Waymo WOD-E2E (long-tail end-to-end set)
| | |
|---|---|
| **One moment** | One `E2EDFrame` record: 8 camera JPEGs + calibrations + past and future ego states + intent (V) |
| **Front camera** | `FRONT` **972×1079, portrait** (R); ≈47° (D) |
| **Driving pattern** | Already in the **current rear-axle frame**, at 4 Hz: past −4 s … 0 and future (0, 5 s], positions with velocity and acceleration (V) |
| **Command** | `intent`: UNKNOWN 0 / GO_STRAIGHT 1 / GO_LEFT 2 / GO_RIGHT 3 (V) |
| **Text** | None. Up to 3 rater-scored alternative trajectories (0–10) on some validation frames (V/R) |
| **Yield** | ≈80–90k train samples at 2 Hz (D). The test split has no future (V) |

Real frame (test split, third-party printout, R):
```text
Frame name: e0ae38e2d9675d9d889eaffcf10fe392-142     Timestamp (microseconds): 0
Cameras available: 8   (972x1079 ×5, 972x587 ×2, 972x551 ×1)
PAST STATES (t=-4s to 0, @4Hz): 16 waypoints   0: pos=(-4.15,-0.24) vel=(3.00,0.87) accel=(-0.37,-0.31)  1: pos=(-3.33,-0.07) …
Intent: 3 (GO_RIGHT)
```
**Cutting a sample:** history = `past[9, 11, 13, 15]`; future = `future[1, 3, …, 15]`; yaw from position differences (D).

### 2.5 nuReasoning (Motional, 2026)
| | |
|---|---|
| **One moment** | A 20 s clip folder at 10 Hz: `metadata.json`, plus per-timestamp `ego_state/`, `annotations/` and `reasoning/` files, and 8 cameras + LiDAR (V) |
| **Front camera** | `CAM_M_F`: a different rig from nuPlan. Resolution and FOV not found (V) |
| **Driving pattern** | Map-frame pose per frame, with velocity and acceleration (V). The devkit builds the future from frames +5, +10, … rather than its precomputed future (V) |
| **Command** | `mission_goal.command`, e.g. `LANE_FOLLOW`, `LEFT_LANE_CHANGE`, `TURN_RIGHT`, plus a `route_path` polyline (V) |
| **Text** | Spatial (~1 Hz), and a Driving decision + reasoning trace + counterfactuals every 5 s (V) |
| **Yield** | ≈0.5M samples at 2 Hz; ≈51k frames with a decision (D) |
| **Access** | **Private Hugging Face repo: request access; reviewed; released progressively** (V). Non-commercial + commercial terms |

Reasoning file (*schema*, V):
```json
{"Driving": {"Scene description": "<free text>", "Critical components": {"<name>": {"Type": ""}},
             "Driving decision": {"Longitudinal": "Slow down gently", "Lateral": "No lateral action"},
             "Reasoning trace": "<free text>"},
 "Counterfactual": {"Alternative actions": [{"Longitudinal": "", "Lateral": "", "Risk level": "", "Reason": ""}]},
 "Spatial": {"object_relations": [{"track_token": "", "geometric_relations": {"euclidean_distance_m": 0}, "future_conflict": {"conflict_with_ego": false, "ttc_s": null}}]}}
```

### 2.6 NVIDIA PhysicalAI-AV
| | |
|---|---|
| **One moment** | A 20 s clip (`clip_id`). One frame of `{clip}.camera_front_wide_120fov.mp4`, plus interpolated rows of `{clip}.egomotion.parquet` and calibration parquets (V) |
| **Front camera** | `camera_front_wide_120fov`: 1920×1080, **120° f-theta** lens, ~30 fps, HEVC MP4 (V) |
| **Driving pattern** | ~100 Hz egomotion: pose, velocity, acceleration, curvature, in an anchor frame (the rig at clip start, yaw 0). Rig ≈ rear axle on the ground (R) |
| **Command** | **None in the data** (V). Must be derived from the path, or from long egomotion if it extends beyond the clip |
| **Text** | Chain-of-causation events `{event_start_timestamp, "coc": "..."}` (V format). Coverage unclear: an issue reports none in revision 26.09 (R) |
| **Yield** | ≈30 samples per clip × 306K clips ≈ 9.2M (D). 1,700+ h, 25 countries (V) |
| **Access** | Gated on Hugging Face, NVIDIA AV dataset licence (V). The full set is ~97 TB (V); the front camera alone is ~14.7 MB per clip (V) |

Real examples (official demo file, V):
```json
{"clip_id": "1ae17e2a-bc07-4e57-a04b-e9d5a372cd1e", "t0_relative": 5784920,
 "nav_text": "Turn left in 11m", "nav_maneuver": "TURN_LEFT", "distance_m": 11.2,
 "cot": "Yield to the cross-traffic vehicle since it is crossing ahead of the right turn path."}
```
Egomotion row (*schema*, V): `timestamp µs | qx qy qz qw | x y z | vx vy vz | ax ay az | curvature`

**Cutting a sample** (devkit API, V; recipe D):
```python
st = ego(ts + np.arange(-3, 9) * 500_000); xyz_t = R_t.inv().apply(st.pose.translation - p_t)
```

**Watch out:**
- The f-theta lens has to be rectified to pinhole, which loses FOV.
- The devkit needs Python ≥ 3.11.
- Pin the dataset revision.
- In 10 of the 20 demo files, the nav text contradicts the causal-chain text.

### 2.7 L2D (Yaak × Hugging Face, LeRobot format)
| | |
|---|---|
| **One moment** | One row of `data/…parquet` (10 Hz), plus 7 MP4s decoded at the matching timestamp (V) |
| **Front camera** | `observation.images.front_left`: 1920×1080 (R), ≈111° lens (R). No intrinsics published (R) |
| **Driving pattern** | **No pose.** `observation.state.vehicle` = speed (km/h), heading (compass), lat, lon, alt, acc_x, acc_y (V keys, R order). RTK GPS (V) |
| **Command** | `observation.state.waypoints`: 10 OSM map-matched points over 100 m; `task.instructions`; turn signal; a BEV map image (V / R) |
| **Text** | Driving-school instructions per episode (V) |
| **Yield** | ≈27 samples per ~19 s episode × 100k episodes ≈ 2.7M (D). Expert and student drivers are tagged (V) |
| **Access** | Hugging Face; licence not found |

Real (official blog, V):
```text
LeRobotDatasetMetadata({ Repository ID: 'yaak-ai/L2D', Total episodes: '100000', Total frames: '19042712', Features: [...] })
episode 0  task.instructions: "Drive straight up to the yield sign and take first exit from the roundabout"
episode 14 task.instructions: "Drive through the unprotected left turn yielding to through traffic"
```

**Cutting a sample** (D):
- `delta_timestamps` with rows −15 … +40.
- Convert lat/lon to UTM.
- Rotate by the compass heading at t to get x forward, y left.
- Convert km/h to m/s.

**Watch out:**
- Units are not SI.
- float32 lat/lon is quantised to ~0.4 m.
- Student episodes are sub-optimal by design.
- Pin the dataset commit.

### 2.8 CoVLA (Turing, Tokyo)
| | |
|---|---|
| **One moment** | One line of `states/<video_id>.jsonl` (20 fps) + one frame of `videos/<video_id>.mp4` + `captions/<video_id>.jsonl` (R) |
| **Front camera** | 1928×1208, 20 fps; intrinsics given per frame (R). FOV not found |
| **Driving pattern** | Each frame stores **its own next 3 s** (60 points, x forward, y left) from GNSS + IMU, plus `vEgo` and `aEgo` (R). There is no past and no 4th second: chain neighbouring frames to get them (D) |
| **Command** | Turn signals only (R) |
| **Text** | `plain_caption`, `rich_caption`, `risk`, scene attributes: rule-based + VideoLLaMA 2, with known hallucinations (R) |
| **Yield** | ≈0.49M at 2 Hz (D); the paper's 2 Hz split has 432k samples (R). 10.75% of trajectories are invalid before filtering (R) |
| **Access** | Hugging Face; no official loader on GitHub (V); licence not verified. **Left-hand traffic** |

### 2.9 ZOD (Zenseact)
| | |
|---|---|
| **One moment** | Sequences / drives folder: `camera_front_blur/*.jpg` (10.1 Hz) + `oxts.hdf5` (100 Hz) + `vehicle_data.hdf5` (V) |
| **Front camera** | 3848×2168, **120° fisheye** (Kannala–Brandt, 4 coefficients) (V) |
| **Driving pattern** | OxTS poses relative to the first pose, 100 Hz, ~0.01 m precision (V). Single Frames only have about −1 s to +10 s of OxTS, so **the 1.5 s history is missing for Frames** (V/D) |
| **Command** | Turn indicator 0 / 1 / 2 (V). A route can be generated with Zenseact's SD-RouteFusion tool (V) |
| **Text** | None (V) |
| **Yield** | ≈43k from Sequences at 2 Hz (D), plus drives |
| **Access** | Request by email. **CC BY-SA 4.0, commercial use allowed** (V) |

Real (sequence 000002, parsed, V):
```text
ego_vehicle_data @ 2022-06-13T10:50:07.39Z: lon_vel 14.3849 m/s, lat_vel 0.0078, lat_acc -0.2295, valid 1
camera_front_blur/000002_romeo_2022-06-13T10:50:07.389350Z.jpg -> 3848×2168 JPEG, 421,772 B
turn_indicator rows: off 564,143 / left 13,114 / right 10,222
```

### 2.10 Text-only sets on top of the above
- **ReCogDrive QA** (navtrain, V generator code). Per line: `{"id", "image": [".../CAM_F0/<token>.jpg"], "conversations": [...]}`.
  - Qwen2.5-VL-72B answers scene, traffic-light, sign and "most influential object" questions.
  - Rules derived from the true future answer the plan question ("KEEP, STRAIGHT"), so that answer is hindsight.
  - Ground-truth boxes answer the vulnerable-road-user question.
  - The released files were "rewritten and filtered". The image paths are the authors' local paths.
- **Nu-X** (nuScenes, V, real): `{token: {"narration": "the car changes to the left lane", "reasoning": "to avoid a green container obstructing the path ahead"}}`.
  - Human-written, about 10 words each.
  - The narration describes **what our car does**, not the scene.

## 3. One table: where each canonical field comes from

| Canonical field | NAVSIM | nuScenes | Waymo Perc. | WOD-E2E | nuReasoning | PhysicalAI-AV | L2D | CoVLA | ZOD |
|---|---|---|---|---|---|---|---|---|---|
| Anchor rate (native) | 2 Hz frames | 12 Hz images | 10 Hz | 10 Hz index | 10 Hz | 30 fps | 10 Hz | 20 fps | 10.1 Hz |
| Front image | `cams.CAM_F0` | `CAM_FRONT` | `FRONT` | `images[FRONT]` | `CAM_M_F` | `camera_front_wide_120fov` | `front_left` | `videos/` | `camera_front_blur` |
| Lens | OpenCV-5, distorted | pinhole, undistorted | OpenCV-5, rolling shutter | distorted, portrait | ? | f-theta 120° | ? (≈111°) | ? | fisheye 120° |
| Pose source | `ego2global_*` | `ego_pose` | `pose` 4×4 | already local | `ego_state.pose` | egomotion | GPS lat/lon + heading | per-frame 3 s future | OxTS `poses` |
| Speed / accel | `ego_dynamic_state` | CAN `vel` / `accel` | velocity (global) / diff | past state | ego_state | egomotion | vehicle state (km/h) | `vEgo` / `aEgo` | OxTS |
| Command (best signal) | route one-hot | CAN route + rule | rule on path | `intent` | `mission_goal` | rule on path | OSM waypoints + rule | turn signal | turn signal / SD-RouteFusion |
| Existing text | — (ReCogDrive QA) | Nu-X, description | — | — | Driving JSON | CoC (sparse) | instructions | captions | — |

**Making "command" mean the same everywhere [proposed]:**
- Where a route polyline exists (nuScenes CAN route, nuReasoning `route_path`, L2D waypoints, ZOD via SD-RouteFusion), apply **NAVSIM's exact rule** (20 m ahead, ±2 m).
- Elsewhere, apply the same rule to the driven path, and flag it as hindsight.
- Check each source's native field (WOD-E2E intent, turn signals) against the rule.

## 4. One frame, end to end (NAVSIM; *illustrative* values, real field names)

**① Raw** (the dict in §2.1). Ego at map (664,402.1, 3,997,815.6), heading 1.12 rad, `ego_dynamic_state = [7.84, 0.02, -0.41, 0.05]`, `driving_command = [0,1,0,0]`.

**③ Canonical frame:**
```yaml
id:      {source: navsim, split: train, log: 2021.05.12.22.00.38_veh-35_01008_01518, token: c082c104b7ac5a71,
          t_us: 1620857890350179, city: us-nv-las-vegas-strip, licence: cc-by-nc-sa-4.0+nuplan}
image:   {path: sensor_blobs/trainval/<log>/CAM_F0/c082c104b7ac5a71.jpg, size: [1920, 1080], lens: opencv5,
          f_px: 1545, c_px: [960, 560], dist: [-0.356, 0.173, -0.002, 0.0005, -0.052], hfov_deg: 64, skew_ms: unknown}
path:    # (x fwd m, y left m, yaw rad) in the rear-axle frame at t
  -1.5: [-11.62, 0.21, 0.01]   -1.0: [-7.78, 0.12, 0.01]   -0.5: [-3.90, 0.04, 0.00]    0.0: [0.0, 0.0, 0.0]
  +0.5: [3.89, -0.02, -0.01]   +1.0: [7.74, -0.07, -0.01]  +1.5: [11.55, -0.15, -0.02]  +2.0: [15.31, -0.27, -0.03]
  +2.5: [19.01, -0.43, -0.04]  +3.0: [22.65, -0.63, -0.06] +3.5: [26.22, -0.87, -0.07]  +4.0: [29.72, -1.15, -0.09]
motion:  {speed: 7.84, accel_lon: -0.41, accel_lat: 0.05, yaw_rate: -0.02, source: measured}
command: {value: straight, how: navsim_route_20m}
text:    []
flags:   {sync: unknown, projection: pass, standstill: false, traffic_side: right}
```

**④ Labels:**
```yaml
decision: {longitudinal: keep_speed, lateral: straight, rules: v0.1}   # speed 7.84 -> 7.0 m/s; lateral residual 0.19 m
text:     {reasoning: null, narration: null}                           # none in NAVSIM -> teacher (P5) or ReCogDrive QA
```

**⑤ Pilot sample** (what the fast model sees and predicts):
```text
input   <image: CAM_F0, rectified/resized to the visual-token budget, e.g. ~450 tokens>
        Speed 7.8 m/s, acceleration -0.4 m/s². Command: go straight.
        Past positions (m): (-11.6, 0.2) (-7.8, 0.1) (-3.9, 0.0) (0.0, 0.0)
target  8 action tokens -> (3.89, -0.02) (7.74, -0.07) … (29.72, -1.15)   # tokenizer = C2 choice; heading from path
```
**⑤ Reasoner sample:**
```text
input   <image: CAM_F0> Speed 7.8 m/s, acceleration -0.4 m/s². Command: go straight. What should the car do, and why?
target  Decision: keep speed, follow lane.
        Reasoning: the lane ahead is clear and traffic is moving steadily, so hold speed.
        Narration: the car cruises in its lane on a multi-lane road behind a white SUV.
```

## 5. Labels: the driving pattern and the reasoning

**The driving pattern is the recorded human path.** It is "correct" in the sense of what a careful human did. It isn't perfect: on navtest, the human path scores 94.5 EPDMS and passes extended comfort only 90.1% of the time (R).

**The decision label comes from rules on the path,** the same for every source. Existing vocabularies map onto ours:

| Ours (v0.1, rules) | ReCogDrive (rules) | nuReasoning (human-checked) | PhysicalAI-AV meta-actions (auto) |
|---|---|---|---|
| keep_speed / accelerate / decelerate / stop / start / stopped | KEEP / ACCELERATE / DECELERATE / STOP | Maintain speed / (Gently, Quickly) accelerate / Slow down (gently, quickly) / (Gently, Quickly) come to a stop / Remain stopped / Reverse | StrongDeceleration, … |
| straight / curve L-R / lane_change L-R / turn L-R | STRAIGHT / LEFT_CHANGE / RIGHT_CHANGE / LEFT_TURN / RIGHT_TURN | No lateral action / Slightly move L-R in lane / L-R lane change / Turn L-R | GoStraight / SteerLeft / SharpSteerLeft, … |

**Reasoning text that already exists:**

| Source | Who wrote it | Style | Covers |
|---|---|---|---|
| Nu-X | Humans | One line of ego-action narration + one line "why" | nuScenes keyframes |
| nuReasoning | Gemini + human check (R) | Scene description, critical objects, decision, reasoning trace, counterfactuals | ~51k key frames (access gated) |
| ReCogDrive | Qwen2.5-VL-72B + rules | Multi-turn QA; decisions are hindsight | navtrain |
| PhysicalAI-AV | Auto-labeller, human-refined (R) | One causal sentence per event | Sparse; coverage unclear |
| CoVLA | Rules + VideoLLaMA 2 | Captions + risk | CoVLA frames |
| L2D | Driving school | Task instruction per episode | L2D episodes |

**Where no text exists** (NAVSIM, Waymo, ZOD and most frames), we write our own with an offline teacher, as in the P5 pilot. Follow the label rules in `P1_REQUIREMENTS.md` §2:
- state only what is visible;
- support the rule-based decision;
- keep it short.

**What the closest rivals trained on** (R, `RESEARCH_DIRECTION.md` §2.2):
- WAM-Diff: nuPlan 668k trajectories (split not stated) + 800k VQA + navtrain.
- DriveFine: ReCogDrive QA + navtrain.
- ReflectDrive-2: navtrain only, plus perturbation-recovery SFT.
- None used PhysicalAI-AV, L2D, ZOD or WOD-E2E (D, from what was found).

## 6. Volumes (D unless marked)

| Source | Samples at 2 Hz | What we would store per sample |
|---|---|---|
| NAVSIM navtrain / all of trainval | 103,288 (V) / ≈0.8M | One front JPEG at model resolution (~50–120 KB) + the canonical record (~2 KB) |
| nuScenes | ≈25k | Same |
| Waymo Perception / WOD-E2E | ≤61k / ≈80–90k | Same |
| nuReasoning | ≈0.5M | Same, + reasoning JSON |
| PhysicalAI-AV | ≈9.2M (front-camera MP4 ≈ 4.5 TB raw in total) | Same |
| L2D | ≈2.7M | Same |
| CoVLA | ≈0.4–0.5M | Same, + captions |
| ZOD Sequences | ≈43k | Same |

At ~100 KB per sample, 1M samples take ~100 GB of compact views. Raw data stays in its own store, referenced and not copied.

## 7. Pitfalls to check in the P2 audit (per source, on a small sample)
1. **Ego origin.**
   - Rear axle: NAVSIM, WOD-E2E (V); nuScenes, PhysicalAI-AV (R).
   - Unknown: Waymo Perception, nuReasoning.
   - GNSS / IMU reference point: L2D, ZOD, CoVLA. There, a lever-arm offset shifts the path in turns.
2. **Lens models:** 47°–120° FOV; pinhole, OpenCV-5, f-theta and fisheye; one portrait camera (WOD-E2E). Decide whether to rectify to one pinhole FOV or keep native images and tag the camera.
3. **Frames and units:**
   - velocity in the global frame (Waymo), LiDAR frame (OpenScene boxes) or anchor frame (PhysicalAI-AV);
   - km/h and compass degrees (L2D);
   - CAN acceleration that includes gravity (nuScenes).
4. **Time:** image-to-pose skew of up to 50 ms, not stored (NAVSIM); zeroed timestamps (WOD-E2E); clip-relative time (PhysicalAI-AV).
5. **Traffic side:** left-hand in Singapore (nuScenes and part of NAVSIM) and Japan (CoVLA).
6. **Command semantics:** route intention (NAVSIM) vs hindsight from the path vs turn signals.
7. **Driver quality:** L2D student drivers; human EC is only 90%.
8. **Test-split leakage:** never touch nuPlan test logs, Waymo's or nuScenes' test splits, or Impromptu's navtest set. Check nuReasoning's logs.
9. **The check that catches most of these:** draw the 8 future poses onto the front image. The path must sit on the road where the car actually drove.

## 8. Decisions needed from you
1. **Shortlist for the first audit.** I suggest:
   - NAVSIM, nuScenes, WOD-E2E: high quality, known conventions;
   - PhysicalAI-AV: scale;
   - L2D: permissive, routes.
2. ~~**Camera handling**~~ **Answered 2026-10-27:** keep native images with a camera tag (general purpose). Rectify only if results are poor.
3. **Access you'd need to request:**
   - nuScenes and Waymo accounts;
   - PhysicalAI-AV licence acceptance;
   - nuReasoning access request (it is reviewed).
