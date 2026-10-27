# Pretraining set v1-pilot: execution plan (2026-10-27)

**Status:**
- **Plan:** [proposed]; its defaults were confirmed 2026-10-27 (§9).
- **Steps 1–2:** the code is written and tested on synthetic logs (§7), but has not run on real data yet. How to run it: §6.
- **Later steps:** each one waits for its gate (§6).

**Researcher decisions, 2026-10-27 [agreed]:**
- **Source:** OpenScene trainval only for the first build. Other sources come later, once access is confirmed.
- **Images:** native, not rectified.
  - Aim: a general-purpose model that learns every camera.
  - Rectify only if results are poor.
- **Size:** a small first test, 10k–100k samples.
- **Cameras:** the front camera only.
- **Storage:** local only, inside the repo (`data/pretrain/`). No persistent storage for now (§5).

Everything else below is a default, marked [proposed]; change any of it. Background on the raw data is in `DATA_ATLAS.md` §2.1.

## 1. What gets built
| Item | Smoke set | v1-pilot |
|---|---|---|
| Training samples | 10,000 | 50,000 |
| Dev samples | 1,000 | 2,000 |
| Logs sampled | ~50 | ~390 |
| Camera archives streamed | 8 of 200 (~44 GB) | 60 of 200 (~330 GB) |
| Front images while building | ~28k (~4.5 GB) | ~220k (~35 GB) |
| Front images after pruning | ~11k (~1.8 GB) | ~52k (~8.3 GB) |
| Canonical tables | ~5 MB | ~25 MB |
| Metadata index (step 2, all logs) | ~0.2–0.4 GB | same |

Notes:
- **Estimates (D):** the log and archive counts assume 1,310 trainval logs across 200 camera archives, i.e. ~6.5 logs per archive, and ~160 KB per CAM_F0 JPEG. Step 2 measures the real values.
- **Pruning [proposed]:**
  - while building, keep every CAM_F0 frame of the streamed logs, so samples can be re-selected after Gate 1 without downloading again;
  - after Gate 2, delete the unselected frames' images.

## 2. Inputs, all public, no login (V)
- **Metadata:** `openscene-v1.1/openscene_metadata_trainval.tgz`, 6.6 GB. One pickle per log, holding every 2 Hz frame dict, no images.
  - Either download it with the existing `scripts/anvil/download_navsim_data.sh trainval-logs`,
  - or let the index read it straight from the URL, with nothing stored.
- **Camera archives:** `openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{0..199}.tgz`, ~5.5 GB each, all 8 cameras.
  - Each archive streams from the URL straight through the extractor [proposed].
  - Only `CAM_F0` JPEGs are written. The archive itself never touches the disk.
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
| `path_t_s` | float[21] | Actual time offset of each pose. Each gap may differ from 0.5 s by up to ±0.05 s, so the stored poses are not exactly on the grid; the audit measures by how much [proposed, new] |
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

## 4b. When discretisation happens [proposed]
**Never in the stored dataset. Only at training time, from the stored floats.** The release keeps continuous poses, and each experiment tokenises them when it loads them.

| Stage | Continuous or discrete | What happens |
|---|---|---|
| Canonical record (§4) | **Continuous** (float32 poses) | Nothing is rounded except the fixed 0.5 s time grid |
| Tokenizer fit (new step 6b) | Continuous in → codebook out | Fit on the **train split only**. Versioned as `tokenizer_<kind>_<size>_v<n>` with a hash of its fit data |
| Ceiling test (step 6b) | Encode → decode → score | Dev-split human paths are run through each tokenizer and scored with NAVSIM's scorer. This measures what the representation alone costs (`docs/NAVSIM_TAILORING.md` §4.1) |
| Training data loader | **Discrete**, on the fly | Path → token indices → model vocabulary IDs. Nearest-centre lookup takes microseconds, so switching tokenizer needs no rewrite of the data |
| Inference | Discrete → continuous | Tokens are decoded back to poses before NAVSIM scores them; RL rewards are computed on the decoded path |

**Why this way:**
- **The C2 study needs continuous paths:** comparing 256 / 1,024 / 2,048 codebooks, coarse + fine tokens, separate x and y, frequency-domain coefficients.
- **Discretisation has two layers, kept apart:**
  - path → token index, which is model-agnostic;
  - index → vocabulary ID, which differs between LLaDA-V, Qwen and Gemma.
- **This is already how MVLAD-AD works:** its training code converts continuous waypoint text to tokens on load (`train/llava/train/train.py` ~L821–836, `point_tokenizer.encode_points`).
- **The prompt's numbers are also formatted at load time:** ego history, speed and acceleration, at a chosen precision. That is a view choice, not storage.

## 5. On disk [agreed 2026-10-27: local, inside the repo]
Everything is written inside the repo checkout. `.gitignore` keeps it all local except the files marked **(git)**.
```
data/pretrain/v1.0-pilot/
├── index/candidates.parquet, logs.parquet  # step 2: every valid anchor and every log
├── index/manifest.json                     # (git) inputs, parameters, counts, checksums
├── audit/summary.json, report.md           # (git) Gate 0: aggregate statistics only
├── splits/{train,dev,excluded}_logs.txt    # (git) the log-level split
├── splits/selected_tokens_{train,dev}.txt  # (git) step 4
├── frames.parquet, privileged.parquet, labels/decision_rules_v0.1.parquet   # step 4
├── images/CAM_F0/<log>/<token>.jpg         # steps 3–4: native JPEGs
├── qa/report.html                          # step 4: renders real images, so it stays local
└── manifest.json, DATA_CARD.md             # (git) step 6: release manifest and data card
data/pretrain/smoke/                        # smoke tests (MAX_LOGS); never committed
```
**What is committed, and why:**
- **Committed:** ID lists, manifests and aggregate statistics. ID lists are safe to share, because NAVSIM publishes the same kind of lists.
- **Local only:** images and per-frame tables (poses, calibration).
- **Why:** the repo is public, and nuPlan's terms on sharing derived data are not yet checked (D10 in `README.md` §5).

**Disk:** set `OUT` to a bigger disk if the checkout's disk is too small (on Anvil: scratch). The layout stays the same.

| | While building | After pruning |
|---|---|---|
| Smoke set | ~5 GB | ~2 GB |
| v1-pilot | ~35 GB | ~9 GB |

All figures are estimates (D).

## 6. Execution steps and gates
| Step | What runs | Output you review | Resources (D) |
|---|---|---|---|
| **1** | Get the metadata: download it (6.6 GB), or let step 2 stream it from the URL | — | Login node; ~15 min |
| **2** | Metadata index and audit, no images (`openscene_index.py`, `audit.py`):<br>• every valid window<br>• distributions of speed, acceleration, curvature, command, decision classes and city<br>• flag rates and dev-split counts<br>• command vs path, and calibration<br>• a selection simulation for 8 and 60 archives | **Gate 0:**<br>• the population statistics<br>• the thresholds<br>• the decision-class cap<br>• the archive counts | 1 job. ~6 min on synthetic logs of real length; real logs carry more annotations, so expect longer |
| **3** | Stream 8 camera archives; record which logs are in each; keep their CAM_F0 JPEGs | The archive → log map | ~44 GB downloaded, ~4.5 GB kept, ~1 h |
| **4** | Build the smoke set (10k + 1k dev), then the QA report: **future path projected onto CAM_F0** for 50 random samples, BEV path plots, timing histogram, flag rates | **Gate 1:** look at the 50 renders; agree the conventions and the record | ~15 min |
| **5** | Stream 52 more archives, then sample the v1-pilot set (50k + 2k dev) | QA report again | ~290 GB downloaded, ~30 GB kept, ~4–6 h |
| **6** | Freeze the v1.0-pilot release: manifest, checksums, data card | **Gate 2:** sign off the release | Minutes |
| **6b** | Fit tokenizers on the train split: MVLAD-AD's shipped 256 for reference, plus refits at 256 / 1,024 / 2,048 and coarse + fine. Run the ceiling test on the dev split, and record rounding error per horizon | One table: tokenizer × rounding error × NAVSIM ceiling | CPU, ~1–2 h |
| **7** | Write the pilot view: image path + prompt + 8 **continuous** target poses (JSONL). Tokenisation happens in the data loader (§4b) | Ready for the first pretraining run | Minutes |

All resource figures in the table are estimates (D).

**Compute note:**
- You have no CPU allocation, so steps 2–6 run as GPU-partition jobs on 1 GPU (8–16 cores), about 6–8 GPU-hours in total (D).
- Downloads may also run on a login node if Anvil's policy allows it.

### How to run steps 1–2
**On Anvil.** Run from the repo root, with `scripts/anvil/config.local.sh` set up:
```bash
bash scripts/anvil/download_navsim_data.sh trainval-logs                       # step 1, on a login node
MAX_LOGS=20 sbatch --partition=gpu-debug --time=00:30:00 \
    scripts/anvil/data_pipeline/pretrain_v1_index.sbatch                        # smoke test -> data/pretrain/smoke/
ALLOW_FULL_RUN=1 sbatch scripts/anvil/data_pipeline/pretrain_v1_index.sbatch   # step 2 -> data/pretrain/v1.0-pilot/
```
**Anywhere else.** You need Python 3.9 or later with numpy, pyarrow and PyYAML. The NAVSIM checkout is used only for its split lists:
```bash
git clone --depth 1 https://github.com/autonomousvision/navsim ~/navsim
python -m data_pipeline.v1.openscene_index --navsim-root ~/navsim --out data/pretrain/v1.0-pilot \
    --metadata https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
python -m data_pipeline.v1.audit --out data/pretrain/v1.0-pilot
```
Options:
- `--metadata` also takes a downloaded `.tgz` or a directory of `*.pkl`.
- `--max-logs 20` makes a smoke test.

**For Gate 0:**
- Read `audit/report.md`. Its first table lists the decisions.
- Commit `audit/`, `splits/` and `index/manifest.json` so they can be reviewed here.

**Safety built in:**
- **Restricted loading:** log pickles are loaded with a restricted unpickler that accepts only numpy arrays and plain containers. Anything else stops the run, and the error names the refused type.
- **Evaluation logs excluded:** a log is excluded if it is in nuPlan's test list or in any NAVSIM evaluation split (navtest, navhard, warmup, navsafe, private tests). Tokens those splits list are excluded too. The audit counts both; 0 is expected.

## 7. Code
Following the repo rules (`CLAUDE.md`, `README.md`), each job refuses a full run unless `ALLOW_FULL_RUN=1`.

| Piece | Location | Status |
|---|---|---|
| Safe pickle loading, NAVSIM split lists, Parquet helpers | `data_pipeline/v1/safe_pickle.py`, `navsim_splits.py`, `tables.py` | **Written 2026-10-27**, tested |
| Metadata index + candidate windows (steps 1–2) | `data_pipeline/v1/openscene_index.py`, job `scripts/anvil/data_pipeline/pretrain_v1_index.sbatch` | **Written**, tested on synthetic logs. Reuses `navsim_tools/raw_frames.py` and the v0.1 rules in `data_pipeline/labels.py` |
| Selection (rules 2–8) | `data_pipeline/v1/selection.py` | **Written**, tested |
| Gate 0 audit | `data_pipeline/v1/audit.py` | **Written**, tested |
| Archive streaming + CAM_F0 extraction (step 3) | `data_pipeline/v1/openscene_images.py`, `scripts/anvil/data_pipeline/pretrain_v1_images.sbatch` | After Gate 0 |
| Canonical rows + validation (step 4) | `data_pipeline/v1/canonical.py` | After Gate 0. First fix the velocity-frame assumption in `navsim_tools/raw_frames.py` (`README.md` L4). The index does not use agent velocities, so it is unaffected |
| QA report (path projection, plots) | `data_pipeline/v1/qa_report.py` | After Gate 0 |
| Release (manifest, checksums, card) | `data_pipeline/v1/release.py` | Step 6 |
| Tests | `tests/data/test_v1_index.py`, `tests/data/test_v1_select_audit.py` | Pass on Python 3.9 + numpy 1.23.4 (NAVSIM's pins) + pyarrow 16, and on Python 3.13 + numpy 2 |

**Facts checked while writing the code (V):**
- **Silent gaps:** OpenScene's generator drops every frame whose LiDAR file or any camera image is missing (`create_openscene_metadata.py`), so a log can have silent gaps.
  - NAVSIM's loader builds windows from the frame list without checking time (`navsim/common/dataloader.py`).
  - So the index checks every window, and the audit counts the navtrain samples whose window spans a gap.
- **Command order:** the one-hot order is (left, forward, right, unknown), per the docstring of OpenScene's `get_driving_command`. The prototype's reading (left, straight, right, unknown) is correct. The audit cross-checks commands against the recorded paths.
- **Evaluation splits:** every NAVSIM evaluation split with log names uses only nuPlan test logs (NAVSIM main, commit 0a380a9):

  | Split | Logs |
  |---|---|
  | navtest | 136 |
  | navhard_two_stage | 76 |
  | warmup_two_stage | 7 |
  | warmup_navsafe_two_stage_extended | 36 |
  | navsafe_two_stage | 36 |

  The private test filters list tokens only. All of them are guarded anyway.

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

## 9. Defaults (confirmed by the researcher's "Yes", 2026-10-27; change any time)
1. **Dev split** = NAVSIM's official val logs. 214 of navtrain's logs; final fine-tuning would then use the 978 train logs.
2. **Balance:** cap "keep speed + straight" at 40%; thin standstills to 1 anchor per 5 s; space anchors ≥ 2 s apart.
3. **Wide window:** store −4 … +6 s per anchor, although the pilot uses −1.5 … +4.0.
4. **Privileged table:** keep boxes and lights for later teacher labels.
5. **Sizes:** smoke set 10k, then 50k.
6. **Storage [agreed 2026-10-27]:** local, inside the repo (`data/pretrain/`). No persistent storage for now (§5).
