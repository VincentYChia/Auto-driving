# Pretraining set v1-pilot: execution plan (2026-10-27)

**Status:** plan for review [proposed]. Nothing is downloaded or built until you sign off each gate (§6).

**Researcher decisions, 2026-10-27 [agreed]:**
- **Source:** OpenScene trainval only for the first build. Other sources come later, once access is confirmed.
- **Images:** native, not rectified.
  - Aim: a general-purpose model that learns every camera.
  - Rectify only if results are poor.
- **Size:** a small first test, 10k–100k samples.
- **Cameras:** the front camera only.

Everything else below is a default, marked [proposed]; change any of it. Background on the raw data is in `DATA_ATLAS.md` §2.1.

## 1. What gets built
| Item | Smoke set | v1-pilot |
|---|---|---|
| Training samples | 10,000 | 50,000 |
| Dev samples | 1,000 | 2,000 |
| Logs sampled | ~50 | ~390 |
| Camera archives streamed | 8 of 200 (~44 GB) | 60 of 200 (~330 GB) |
| Front images kept | ~28k (~4.5 GB) | ~220k (~35 GB): every CAM_F0 frame of the sampled logs, so we can resample without re-downloading |
| Canonical tables | ~5 MB | ~25 MB |

Log and archive counts are estimates (D): 1,310 trainval logs across 200 camera archives is ~6.5 logs per archive, and ~160 KB per CAM_F0 JPEG. Step 2 measures the real values.

## 2. Inputs, all public, no login (V)
- **Metadata:** `openscene-v1.1/openscene_metadata_trainval.tgz`, 6.6 GB. One pickle per log, holding every 2 Hz frame dict, no images. Use the existing `scripts/anvil/download_navsim_data.sh trainval-logs`.
- **Camera archives:** `openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{0..199}.tgz`, ~5.5 GB each, all 8 cameras.
  - Archives are streamed one at a time: download → extract `CAM_F0` only → delete.
  - At most ~12 GB of scratch is in use at any moment.
- **Splits:**
  - NAVSIM's `navtrain.yaml`: 103,288 tokens from 1,192 logs, to tag frames that are in navtrain.
  - `default_train_val_test_log_split.yaml`: navtrain's logs split 978 train / 214 val, with 0 in test.
  - `navtest.yaml`: all 136 of its logs are in the test list (V, counted 2026-10-27).

## 3. Sampling and cleaning rules [proposed]
Applied in this order. Everything except image decoding runs on the metadata, so the whole population (~0.8M frames, D) is measured before any image is downloaded.

| # | Rule | Default | Why |
|---|---|---|---|
| 1 | **Valid window** | Frames i−3 … i+8 exist in the same log, each gap 0.5 s ± 0.05 s | The pilot needs 1.5 s of history and 4 s of future |
| 2 | **Image present** | `CAM_F0` path exists and the JPEG decodes | — |
| 3 | **Pose sanity** | Drop if any 0.5 s step is > 25 m or turns > 0.5 rad. Flag if speed from the path differs from `ego_dynamic_state` by > 1 m/s | Catches localisation jumps |
| 4 | **Split** | **dev = NAVSIM's official val logs; train = its train logs.** Logs in neither go to train (count reported) | Standard NAVSIM practice; dev never leaks into training or fine-tuning |
| 5 | **Leakage guard** | Assert that no log appears in NAVSIM's test list (navtest / navhard) | NAVSIM rule |
| 6 | **Standstill thinning** | During continuous standstill (speed < 0.3 m/s, < 0.5 m of movement over the window), keep at most 1 anchor per 5 s | Standstill would otherwise dominate |
| 7 | **Spacing** | ≥ 2 s between kept anchors in a log (smoke set: ≥ 1 s) | Neighbouring 0.5 s frames are near-duplicates |
| 8 | **Balance** | Cap "keep speed + straight" at 40% of the sample. Fill the rest proportionally across the other decision classes and cities | Imitation data is mostly cruising |
| 9 | **Keep, but flag** | No route (command unknown); windows that cross a nuPlan scene boundary; frames that are in navtrain | Still valid driving; the flags let you filter later |

## 4. What each sample stores: the canonical record [proposed]
Files are Parquet. One row per kept anchor.

**`frames.parquet`** (test-time-visible inputs + the target path):

| Column | Type | Content |
|---|---|---|
| `frame_id` | str | `openscene/<log>/<token>` |
| `log_name`, `token`, `scene_token`, `t_us` | str / int64 | Identity; t is the LiDAR time |
| `nuplan_split`, `our_split` | str | train / val → train / dev |
| `city`, `traffic_side` | str | e.g. `us-nv-las-vegas-strip`, `right`; Singapore is `left` |
| `in_navtrain` | bool | Token is in navtrain |
| `image_path`, `image_sha256`, `image_w`, `image_h` | str / int | Native JPEG, copied byte-for-byte |
| `cam_name`, `lens_model` | str | `CAM_F0`, `opencv5` |
| `K` | float[4] | fx, fy, cx, cy |
| `dist` | float[5] | k1, k2, p1, p2, k3 |
| `hfov_pinhole_deg`, `hfov_undistorted_deg` | float | Computed |
| `cam_to_ego` | float[7] | Translation + quaternion. Built from `sensor2lidar_*` and `lidar2ego_*`; their meaning is verified in step 4 |
| `other_cam_paths` | str[7] | Paths to the other 7 cameras. **Not downloaded**; kept so stitching stays possible later |
| `path_offsets_s` | float[21] | −4.0 … +6.0 every 0.5 s (fixed) |
| `path_xyyaw` | float[21×3] | Poses in the rear-axle frame at t: x forward, y left, m, rad |
| `path_valid` | bool[21] | −1.5 … +4.0 is always valid; the rest only where the log allows |
| `speed`, `accel_lon`, `accel_lat` | float | From `ego_dynamic_state`, ego frame |
| `yaw_rate` | float | Differenced from the poses |
| `speed_from_path` | float | For quality checks |
| `command`, `command_how` | str | left / straight / right / unknown; `navsim_route_20m` |
| `has_route`, `crosses_scene_boundary` | bool | Flags |
| `flags` | struct | dt_ok, speed_consistent, standstill, thinned |
| `licence`, `provenance` | str / struct | CC BY-NC-SA 4.0 + nuPlan; upstream archive names, code commit, NAVSIM devkit commit |

**`privileged.parquet`** (training-only information, kept apart so a loader can't leak it into inputs):
- boxes at t: `gt_boxes`, `gt_names`, `track_tokens`;
- `gt_velocity_3d`, raw (LiDAR frame) **and** converted to the ego frame, with a flag;
- traffic lights;
- `roadblock_ids`.

It is used later for teacher labels and for quality checks.

**`labels/decision_rules_v0.1.parquet`:** longitudinal and lateral meta-actions from rules on the path. Flagged *uncalibrated* until the audit calibrates the thresholds.

**Text:** none in OpenScene. The column exists and is empty, so later sources fit the same table.

## 5. On disk after the build [proposed]
```
$DATA_PERSIST/pretrain/v1.0-pilot/
├── manifest.json            # version, rules, counts, sha256 of every file, upstream archives, code + devkit commits
├── DATA_CARD.md             # what is in it, how it was made, known issues, licence
├── frames.parquet  privileged.parquet  labels/decision_rules_v0.1.parquet
├── splits/{train,dev}_logs.txt  selected_tokens_{train,dev}.txt
├── qa/report.html           # statistics + 50 rendered samples (§6, step 4)
└── images/CAM_F0/<log>/<token>.jpg      # native JPEGs (all CAM_F0 frames of the sampled logs)
```
`$DATA_PERSIST` is persistent storage (project space). Scratch is purged after 30 days and is used only for streaming.

## 6. Execution steps and gates
| Step | What runs | Output you review | Resources (D) |
|---|---|---|---|
| **1** | Download the metadata (6.6 GB) | — | Login node or a short job; ~15 min |
| **2** | Metadata audit, no images: enumerate every valid window; distributions of speed, acceleration, curvature, command, decision classes, city; flag rates; dev-split counts | **Gate 0:** the population statistics, and the decision-class cap revisited | 1 job, ~30 min CPU |
| **3** | Stream 8 camera archives; record which logs are in each; extract CAM_F0 | The archive → log map | ~44 GB streamed, ~1 h |
| **4** | Build the smoke set (10k + 1k dev), then the QA report: **future path projected onto CAM_F0** for 50 random samples, BEV path plots, timing histogram, flag rates | **Gate 1:** look at the 50 renders; agree the conventions and the record | ~15 min |
| **5** | Stream 52 more archives, then sample the v1-pilot set (50k + 2k dev) | QA report again | ~290 GB streamed, ~4–6 h |
| **6** | Freeze the v1.0-pilot release: manifest, checksums, data card | **Gate 2:** sign off the release | Minutes |
| **7** | Write the pilot view: image path + prompt + 8 target poses (JSONL). Tokenisation happens at training time (C2) | Ready for the first pretraining run | Minutes |

**Compute note:** you have no CPU allocation, so steps 2–6 run as GPU-partition jobs on 1 GPU (8–16 cores), about 6–8 GPU-hours in total (D). Downloads may also run on a login node if Anvil's policy allows it.

## 7. Code to write after you sign off this plan
Following the repo rules (`CLAUDE.md`, `README.md`): no full run without your sign-off. Each job refuses a full run unless `ALLOW_FULL_RUN=1`.

| Piece | Location | Reuses |
|---|---|---|
| Metadata index + candidate windows | `data_pipeline/v1/openscene_index.py` | `navsim_tools/raw_frames.py` (pose transforms, already tested). **Fix its velocity-frame assumption first** (`README.md` L4) |
| Selection (rules 1–9) | `data_pipeline/v1/select.py` | `data_pipeline/labels.py` rules v0.1 |
| Archive streaming + CAM_F0 extraction | `data_pipeline/v1/openscene_images.py`, `scripts/anvil/data_pipeline/pretrain_v1_images.sbatch` | `download_navsim_data.sh` |
| Canonical rows + validation | `data_pipeline/v1/canonical.py` | — |
| QA report (path projection, plots) | `data_pipeline/v1/qa_report.py` | — |
| Release (manifest, checksums, card) | `data_pipeline/v1/release.py` | — |
| Tests on synthetic logs | `tests/data/` | `tests/shared/synthetic_navsim.py` |

## 8. How this addresses the v0.1 prototype's limitations (`README.md` §3)
| Limitation | Fix |
|---|---|
| L1 (windows stored repeatedly) | One row per anchor, plus a wide fixed window (−4 … +6 s). Images are stored once per frame |
| L2 (lossy) | Native JPEG bytes; full-precision poses; nothing filtered at storage time beyond the rules above |
| L3 (no split) | NAVSIM's official train/val logs |
| L4 (conventions unchecked) | The step-2 audit and the step-4 renders |
| L5 (privileged data in the same record) | A separate `privileged.parquet` |
| L6 (labels uncalibrated) | Rules versioned and flagged uncalibrated; calibrated at Gate 0 |
| L7 (no QA) | `qa/report.html` with renders |
| L8 (no release discipline) | Manifest, checksums, data card |

## 9. Defaults to confirm (or change)
1. **Dev split** = NAVSIM's official val logs. 214 of navtrain's logs; final fine-tuning would then use the 978 train logs.
2. **Balance:** cap "keep speed + straight" at 40%; thin standstills to 1 anchor per 5 s; space anchors ≥ 2 s apart.
3. **Wide window:** store −4 … +6 s per anchor, although the pilot uses −1.5 … +4.0.
4. **Privileged table:** keep boxes and lights for later teacher labels.
5. **Sizes:** smoke set 10k, then 50k.
6. **Storage location (question):** where on Anvil should `$DATA_PERSIST` live (project space path and quota)?
