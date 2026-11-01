# Pretraining set v1-pilot: execution plan (2026-10-27)

**Status:**
- **Plan:** [proposed]; its defaults were confirmed 2026-10-27 (§9).
- **Steps 1–2:** ran on all 1,310 logs (2026-10-31); Gate 0 report in `data/pretrain/v1.0-pilot/audit/report.md`.
- **Gate 0:** decided 2026-10-31 (below). **Step 3:** coded; the smoke run comes first. How to run it: §6.
- **Later steps:** each one waits for its gate (§6).

**Researcher decisions, 2026-10-27 [agreed]:**
- **Source:** OpenScene trainval only for the first build. Other sources come later, once access is confirmed.
- **Images:** native, not rectified.
  - Aim: a general-purpose model that learns every camera.
  - Rectify only if results are poor.
- **Size:** a small first test, 10k–100k samples.
- **Cameras:** the front camera only.
- **Storage:** local only, inside the repo (`data/pretrain/`). No persistent storage for now (§5).
- **Compute (2026-10-29):** the data build runs on your own computer [agreed]. Fine-tuning compute is [open]: perhaps a rented cloud GPU, or Anvil (§6).

**Gate 0, 2026-10-31 [agreed]:** "more variety", and about **10 GB** of training data on the local computer.
- **v1-pilot = 45,000 train + 2,000 dev** samples, ~9.9 GB of CAM_F0 JPEGs (D, at the measured 211 KB each).
- **All 200 camera archives are streamed** (1,242 GB downloaded; only the selected JPEGs are kept), so the samples come from every log: 1,083 of 1,085 train logs, 221 of 225 dev logs.
- **The train split is drawn balanced** over city × lateral manoeuvre (rule 8). Dev keeps the natural mix.
- [proposed] parameters: balance strength α = 0.5; archives ordered by variety per GB; the smoke set from the first 20 archives.
- Why (D, simulated on the real index and archive map): 60 random archives give 31k samples at 79% Las Vegas and 16% turning. All 200 with the balanced pick give 45k at 64% Las Vegas and 28% turning.
- The other Gate 0 rules (timing tolerance, pose limits, standstill thinning, cruise cap) were not revisited; they keep their current values (§3, §9).

Everything else below is a default, marked [proposed]; change any of it. Background on the raw data is in `DATA_ATLAS.md` §2.1.

## 1. What gets built
| Item | Smoke set | v1-pilot |
|---|---|---|
| Training samples | 10,000 | 45,000 [agreed 2026-10-31] |
| Dev samples | 1,000 | 2,000 |
| Logs sampled | 98 train + 32 dev | 1,083 train + 221 dev |
| Camera archives streamed | the first 20 in plan order (83.5 GB, V) | all 200 (1,242 GB, V) |
| Front images kept | 11,000 (~2.3 GB) | 47,000 (~9.9 GB); with the smoke set's, 56,528 (~11.9 GB) until Gate 2 |
| Front images after pruning | — | 47,000 (~9.9 GB) |
| Canonical tables | ~5 MB | ~25 MB |
| Metadata index (step 2, all logs) | ~0.2–0.4 GB | same |

Notes:
- **Counts (2026-10-31):** samples and logs come from the selection on the real index (`openscene_images.py plan`); archive sizes from Hugging Face (V). Image sizes are D, at 211 KB per JPEG.
- **Original estimates (D, 2026-10-27):** the log and archive counts assumed 1,310 trainval logs across 200 camera archives, i.e. ~6.5 logs per archive, and ~160 KB per CAM_F0 JPEG. Step 2 measures the real values.
  - **Measured 2026-10-31 (V):** 1,310 logs (step 2). CAM_F0 JPEGs average **211 KB** (140–283 KB, 1,244 images from archive 70), so the image sizes in this table are ~30% low.
- **Pruning [proposed, revised 2026-10-31]:** only the selected frames are kept (v1-pilot and smoke), to stay near 10 GB. Keeping every frame of every log would take ~148 GB (D).
  - The cost: changing the selection after the full run means downloading the affected archives again. That is why Gate 1 runs on the smoke set first.
  - After Gate 2, delete the smoke-only frames' images (~2 GB).

## 2. Inputs, all public, no login (V)
- **Metadata:** `openscene-v1.1/openscene_metadata_trainval.tgz`, 6.6 GB. One pickle per log, holding every 2 Hz frame dict, no images.
  - Either download it with the existing `scripts/anvil/download_navsim_data.sh trainval-logs`,
  - or let the index read it straight from the URL, with nothing stored.
- **Camera archives:** `openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{0..199}.tgz`, all 8 cameras.
  - **Sizes (V, Hugging Face API, 2026-10-31):** 1.8–19.3 GB, median 5.3 GB, mean 6.2 GB; 1,242 GB for all 200.
  - **Which logs each holds is published (V):** `openscene-v1.1/openscene_sensor_trainval_0-199.json` maps every archive to its logs (6–7 each). It lists exactly the 1,310 logs of the step-2 index. So step 3 does not need to discover the map.
  - **Checksums (V):** each archive's size and SHA-256 are in its Hugging Face LFS record. Downloads support HTTP ranges (checked 2026-10-31).
  - **Each archive is downloaded to `images/tmp/`, resuming after a dropped connection, checked against its SHA-256, read, then deleted [proposed, revised 2026-10-31].** Streaming straight through the extractor could not resume, and a 19 GB archive can take hours at the speeds below.
  - Only the selected frames' `CAM_F0` JPEGs are written.
  - **Download speed (V, 2026-10-31, 5 samples of 25–40 s):** 1.6–16 MB/s on one connection; 2–4 parallel connections were no faster in total. Archive 70 averaged 3.5 MB/s. So the full 1,242 GB takes ~2–4 days (D).
  - **Archive order [proposed]:** greedy, most variety per GB first (`openscene_images.archive_order`). With all 200 streamed, the order only decides which come first, and so what the smoke set holds.
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
| 8 | **Balance** | Cap "keep speed + straight" at 40% of the sample. **Train [agreed 2026-10-31]:** draw anchors in a weighted random order, weight n^−α, where n counts the pool anchors with the same city and lateral decision (α = 0.5 [proposed]). Dev keeps the natural mix | Imitation data is mostly cruising, and 76% Las Vegas |
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
├── audit/examples.html                     # three example samples, with why each got its command and label (local)
├── splits/{train,dev,excluded}_logs.txt    # (git) the log-level split
├── splits/{v1-pilot,smoke}_tokens_{train,dev}.txt  # (git) step 3: the selected samples
├── splits/manifest.json                    # (git) step 3: selection reports and archive order
├── images/archives.json, archives/<n>.json # step 3: archive map + checksums; one record per fetched archive
├── frames.parquet, privileged.parquet, labels/decision_rules_v0.1.parquet   # step 4
├── images/CAM_F0/<log>/<token>.jpg         # steps 3–4: native JPEGs
├── qa/report.html                          # step 4: renders real images, so it stays local
└── manifest.json, DATA_CARD.md             # (git) step 6: release manifest and data card
data/pretrain/smoke/                        # smoke tests (first 20 logs); never committed
data/pretrain/downloads/                    # metadata archive (6.6 GB) and NAVSIM's split lists; never committed
```
**What is committed, and why:**
- **Committed:** ID lists, manifests and aggregate statistics. ID lists are safe to share, because NAVSIM publishes the same kind of lists.
- **Local only:** images and per-frame tables (poses, calibration).
- **Why:** the repo is public, and nuPlan's terms on sharing derived data are not yet checked (D10 in `README.md` §5).

**Disk:** the table below excludes the 6.6 GB metadata archive. If the checkout's disk is too small, set `OUT` to a bigger disk; the layout stays the same.

| | While building | After pruning |
|---|---|---|
| Smoke set | ~2.3 GB + one archive (≤ 19.3 GB) | ~2.3 GB |
| v1-pilot | ~11.9 GB + one archive (≤ 19.3 GB) | ~9.9 GB |

All figures are estimates (D), revised 2026-10-31.

## 6. Execution steps and gates
| Step | What runs | Output you review | Resources (D) |
|---|---|---|---|
| **1** | Download the metadata (6.6 GB, once) and NAVSIM's split lists (a few MB). The local script does both | — | ~15 min |
| **2** | Metadata index and audit, no images (`openscene_index.py`, `audit.py`):<br>• every valid window<br>• distributions of speed, acceleration, curvature, command, decision classes and city<br>• flag rates and dev-split counts<br>• command vs path, and calibration<br>• a selection simulation for 8 and 60 archives | **Gate 0:**<br>• the population statistics<br>• the thresholds<br>• the decision-class cap<br>• the archive counts | CPU only. ~6 min here for synthetic logs of real length; real logs carry more annotations, so expect longer |
| **3** | `openscene_images.py plan`: select v1-pilot and the smoke set; order the archives. Then `fetch` the first 20 archives: check their logs against OpenScene's archive → log map (§2), and keep the selected CAM_F0 JPEGs | The archive → log map; missing and undecodable JPEGs | 83.5 GB downloaded, ~2.3 GB kept for the smoke set, ~2–15 h |
| **4** | Build the smoke set (10k + 1k dev), then the QA report: **future path projected onto CAM_F0** for 50 random samples, BEV path plots, timing histogram, flag rates | **Gate 1:** look at the 50 renders; agree the conventions and the record | ~15 min |
| **5** | `fetch --full`: the other 180 archives, then build the v1-pilot set (45k + 2k dev) | QA report again | ~1,160 GB downloaded, ~9.9 GB kept, ~2–4 days |
| **6** | Freeze the v1.0-pilot release: manifest, checksums, data card | **Gate 2:** sign off the release | Minutes |
| **6b** | Fit tokenizers on the train split: MVLAD-AD's shipped 256 for reference, plus refits at 256 / 1,024 / 2,048 and coarse + fine. Run the ceiling test on the dev split, and record rounding error per horizon | One table: tokenizer × rounding error × NAVSIM ceiling | CPU, ~1–2 h |
| **7** | Write the pilot view: image path + prompt + 8 **continuous** target poses (JSONL). Tokenisation happens in the data loader (§4b) | Ready for the first pretraining run | Minutes |

All resource figures in the table are estimates (D). Download times use the speeds measured on 2026-10-31: 1.6–16 MB/s (§2).

**Where it runs (2026-10-29):**
- **Data build, steps 1–7 [agreed]:** your own computer, CPU only; no GPU is needed.
  - Disk: ~7 GB for steps 1–2, and up to ~31 GB while the v1-pilot images are being built: ~11.9 GB of images plus one archive (§5).
- **Fine-tuning [open]:** perhaps a rented cloud GPU, or Anvil.
  - The finished samples to upload are ~2 GB for the smoke set and ~9 GB for the v1-pilot.
- **Anvil job scripts:** they stay as an alternative for the data build.

### How to run steps 1–2
**On your computer (default).** macOS or Linux, with Python 3.9 or later, git and curl. Run from the repo root:
```bash
python3 -m pip install numpy pyarrow pyyaml
bash scripts/local/pretrain_v1_index.sh          # smoke test: the first 20 logs -> data/pretrain/smoke/
bash scripts/local/pretrain_v1_index.sh --full   # step 2: all logs            -> data/pretrain/v1.0-pilot/
```
The first run downloads NAVSIM's split lists and the metadata into `data/pretrain/downloads/`. If the download is interrupted, run the script again to resume it. Each run ends with two pages to open: `audit/report.md` (the statistics) and `audit/examples.html` (three samples, each with the reasons for its command and label).

**Without bash** (e.g. Windows, where the command is `curl.exe`):
```bash
git clone --depth 1 https://github.com/autonomousvision/navsim data/pretrain/downloads/navsim
curl -L -o data/pretrain/downloads/openscene_metadata_trainval.tgz https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
python -m data_pipeline.v1.openscene_index --metadata data/pretrain/downloads/openscene_metadata_trainval.tgz \
    --navsim-root data/pretrain/downloads/navsim --out data/pretrain/v1.0-pilot
python -m data_pipeline.v1.audit --out data/pretrain/v1.0-pilot
```
Options:
- `--metadata` also takes the URL itself (read as a stream, nothing stored) or a directory of `*.pkl`.
- `--max-logs 20` makes a smoke test.

**On Anvil (optional).** Run from the repo root, with `scripts/anvil/config.local.sh` set up:
```bash
bash scripts/anvil/download_navsim_data.sh trainval-logs                       # step 1, on a login node
MAX_LOGS=20 sbatch --partition=gpu-debug --time=00:30:00 \
    scripts/anvil/data_pipeline/pretrain_v1_index.sbatch                        # smoke test -> data/pretrain/smoke/
ALLOW_FULL_RUN=1 sbatch scripts/anvil/data_pipeline/pretrain_v1_index.sbatch   # step 2 -> data/pretrain/v1.0-pilot/
```

**For Gate 0:**
- Read `audit/report.md`. Its first table lists the decisions.
- Commit `audit/`, `splits/` and `index/manifest.json` so they can be reviewed here.

**Safety built in:**
- **Restricted loading:** log pickles are loaded with a restricted unpickler that accepts only numpy arrays and plain containers. Anything else stops the run, and the error names the refused type.
- **Evaluation logs excluded:** a log is excluded if it is in nuPlan's test list or in any NAVSIM evaluation split (navtest, navhard, warmup, navsafe, private tests). Tokens those splits list are excluded too. The audit counts both; 0 is expected.

### How to run step 3
From the repo root. `fetch` needs Pillow (`python -m pip install pillow`) to check that each JPEG decodes.
```bash
python -m data_pipeline.v1.openscene_images plan   --out data/pretrain/v1.0-pilot    # minutes; metadata only
python -m data_pipeline.v1.openscene_images fetch  --out data/pretrain/v1.0-pilot    # the smoke set's 20 archives
python -m data_pipeline.v1.openscene_images status --out data/pretrain/v1.0-pilot    # progress, at any time
python -m data_pipeline.v1.openscene_images fetch  --out data/pretrain/v1.0-pilot --full --archives 200   # after Gate 1
```
- **Resuming:** a finished archive leaves `images/archives/<n>.json` and is skipped next time; an unfinished download resumes from its `.part` file. After an interruption, run the same command again.
- **One bad archive doesn't stop the run:** it is reported at the end, and the command exits with an error. Run it again to retry.
- **Commit after `plan`:** `splits/*_tokens_*.txt` and `splits/manifest.json`.

### Viewing the index, with images (local)
A page for browsing the index: the front image with the future path drawn on it, a bird's-eye plot, the inputs, the 8 target poses, the label and the reasons for it. It also has the Gate 0 summary as charts. From the repo root:
```bash
python -m data_pipeline.v1.viewer fetch --out data/pretrain/v1.0-pilot   # stream camera archive 70 (3.2 GB, ~15 min); keeps ~1.8k CAM_F0 JPEGs (~0.4 GB)
python -m data_pipeline.v1.viewer build --out data/pretrain/v1.0-pilot   # then open data/pretrain/v1.0-pilot/viewer/index.html
```
- **What it shows:**
  - every frame of the logs whose images were fetched. Archive 70 holds 7 logs from all four cities, in both train and dev;
  - plus a seeded random 3,000 eligible frames from the other logs, without images.
- **More images:** run `fetch --archive N` for more archives, then `build` again.
- **Not step 3:** these images are for looking only. Like everything under `data/pretrain/`, they stay local (`.gitignore`, §5).
- **Path on the image:** projected with each log's calibration, onto flat ground (`data_pipeline/v1/projection.py`). The conventions are assumed: camera-to-LiDAR extrinsics, LiDAR = ego, the ego origin on the ground, OpenCV distortion.
  - Checked by eye on 5 frames of one log (2026-10-31): the path lies on the road and follows the lane through a straight, a left turn and a right turn.
  - That supports the conventions, but it is not the step-4 check.

## 7. Code
Following the repo rules (`CLAUDE.md`, `README.md`), nothing runs on all logs without an explicit opt-in: `--full` for the local script, `ALLOW_FULL_RUN=1` for the Anvil jobs.

| Piece | Location | Status |
|---|---|---|
| Safe pickle loading, NAVSIM split lists, Parquet helpers | `data_pipeline/v1/safe_pickle.py`, `navsim_splits.py`, `tables.py` | **Written 2026-10-27**, tested |
| Metadata index + candidate windows (steps 1–2) | `data_pipeline/v1/openscene_index.py`; runners `scripts/local/pretrain_v1_index.sh` and `scripts/anvil/data_pipeline/pretrain_v1_index.sbatch` | **Written**, tested on synthetic logs. Reuses `navsim_tools/raw_frames.py` and the v0.1 rules in `data_pipeline/labels.py` |
| Selection (rules 2–8) | `data_pipeline/v1/selection.py` | **Written**, tested. Balanced train draw added 2026-10-31 |
| Gate 0 audit | `data_pipeline/v1/audit.py` | **Written**, tested |
| Local viewer: fetch one camera archive's CAM_F0 images; page with images, projected path and Gate 0 charts | `data_pipeline/v1/viewer.py`, `viewer_template.html`, `projection.py` | **Written 2026-10-31**, tested. Not part of the build |
| Selection, archive order, CAM_F0 extraction (step 3) | `data_pipeline/v1/openscene_images.py` | **Written 2026-10-31**, tested on synthetic archives: resume, checksum, missing and undecodable JPEGs. No Anvil job yet |
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
5. **Sizes:** smoke set 10k, then 50k. **Changed at Gate 0 [agreed 2026-10-31]:** v1-pilot 45k + 2k dev (~10 GB), from all 200 archives.
6. **Storage [agreed 2026-10-27]:** local, inside the repo (`data/pretrain/`). No persistent storage for now (§5).
7. **Compute [agreed 2026-10-29]:** the data build runs on your own computer. Fine-tuning compute is still open: a cloud GPU or Anvil (§6).
