# Pretraining set v1-pilot: execution plan (2026-10-27)

**Status:**
- **Plan:** [proposed]; its defaults were confirmed 2026-10-27 (§9).
- **Steps 1–2:** ran on all 1,310 logs (2026-10-31); Gate 0 report in `data/pretrain/v1.0-pilot/audit/report.md`.
- **Gate 0:** decided 2026-10-31 (below). **Step 3:** the smoke run finished 2026-11-01 (20 archives, 0 missing or undecodable images).
- **2026-11-01, researcher: "keep going until we are completely ready to train"** (a test model on Together.ai, Ollama or Anvil, plus a cost estimate). So:
  - **Step 4 ran on the smoke set:** 11,000 frames, every check passed; the Gate 1 report is in `qa/report.html`, and Claude looked at its renders (the path lies on the road and follows the lane). **Gate 1 still needs your look** at the 50 renders.
  - **Step 5 started before that look:** the full fetch is running. The selection was fixed at Gate 0, and Gate 1 changes the record, not which images are kept.
  - **Steps 6 and 7 are coded**; step 7 ran on the smoke set. The fine-tuning code is in `pilot_sft/` (its README has the platforms and costs). Step 6b is not needed for the first test model, which writes its path as text.
- **v1-large, 2026-11-03/04:** v1-pilot's rules and seed with 100,000 train + v1-pilot's 2,000 dev, built on Anvil; every canonical check passes (V). Not released (§10).
- **v1-xl, 2026-11-08 [agreed]:** train anchors ≥ 1.0 s apart, about 2× v1-large, same rules and dev, images packed into archive files. Plan, build job and checks are written; the build runs on Anvil next. A nested subset, v1-xl-nd, is [proposed] for an A/B (§10). Gate 1 and Gate 2 are still open for both sets.

**Researcher decisions, 2026-10-27 [agreed]:**
- **Source:** OpenScene trainval only for the first build. Other sources come later, once access is confirmed.
- **Images:** native, not rectified.
  - Aim: a general-purpose model that learns every camera.
  - Rectify only if results are poor.
- **Size:** a small first test, 10k–100k samples. **Changed [agreed 2026-11-08]:** v1-large has 100k and v1-xl 200k train samples (§10).
- **Cameras:** the front camera only.
- **Storage:** local only, inside the repo (`data/pretrain/`). No persistent storage for now (§5). **Changed [agreed 2026-11-08]:** v1-large and v1-xl live on Anvil scratch, outside the repo, and v1-xl's images are packed into uncompressed archive files (§10). Their ID lists are still committed.
- **Compute (2026-10-29):** the data build runs on your own computer [agreed]. Fine-tuning compute is [open]: perhaps a rented cloud GPU, or Anvil (§6). **Changed [agreed 2026-11-08]:** v1-large and v1-xl are built on Anvil, on CPU (§10).

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
  - **Download speed (V, 2026-10-31, 5 samples of 25–40 s):** 1.6–16 MB/s on one connection; 2–4 parallel connections were no faster in total. Archive 70 averaged 3.5 MB/s. **The smoke run's 20 archives (83.5 GB, 2026-10-31/11-01) averaged 27.8 MB/s** (V), 15–43 MB/s per archive. At that rate the other 1,160 GB take ~12 h; at the slowest speeds seen, ~2–4 days (D).
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
| 7 | **Spacing** | ≥ 2 s between kept anchors in a log (smoke set: ≥ 1 s). **v1-xl train ≥ 1.0 s [agreed 2026-11-08]; dev = v1-pilot's 2 s draw, pinned by token [proposed mechanics]** (§10) | Neighbouring 0.5 s frames are near-duplicates |
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
| `image_path`, `image_sha256`, `image_w`, `image_h` | str / int | Native JPEG, copied byte-for-byte. v1-xl: the same bytes, stored in an archive file as the member named `image_path` [proposed format, §10] |
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
├── canonical.json                          # (git) step 4: counts, checks, table checksums
├── images/CAM_F0/<log>/<token>.jpg         # steps 3–4: native JPEGs
├── images/packs/<n>.zip                    # v1-xl instead (`--storage pack`): one uncompressed archive per camera archive (§10)
├── qa/report.html                          # step 4: renders real images, so it stays local
├── qa/summary.json                         # (git) step 4: the report's aggregate numbers
├── views/pilot_v1/<set>/                   # step 7: {train,dev}.jsonl, images/ at 640 px, view.json, chat_*.jsonl
│                                           #   v1-xl views (`--storage zip`): images.zip instead of images/ (§10)
└── manifest.json, DATA_CARD.md             # (git) step 6: release manifest and data card; images/SHA256SUMS stays local
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
| **4** | Build the smoke set (10k + 1k dev), then the QA report: **future path projected onto CAM_F0** for 50 random samples, BEV path plots, timing histogram, flag rates | **Gate 1:** look at the 50 renders; agree the conventions and the record | **Measured 2026-11-01 (V):** 92 s for the canonical build, 2 s for the report |
| **5** | `fetch --full`: the other 180 archives, then build the v1-pilot set (45k + 2k dev) | QA report again | ~1,160 GB downloaded, ~9.9 GB kept, ~12 h at the smoke run's speed (up to ~2–4 days) |
| **6** | Freeze the v1.0-pilot release: manifest, checksums, data card | **Gate 2:** sign off the release | Minutes |
| **6b** | Fit tokenizers on the train split: MVLAD-AD's shipped 256 for reference, plus refits at 256 / 1,024 / 2,048 and coarse + fine. Run the ceiling test on the dev split, and record rounding error per horizon | One table: tokenizer × rounding error × NAVSIM ceiling | CPU, ~1–2 h |
| **7** | Write the pilot view: image path + prompt + 8 **continuous** target poses (JSONL). Tokenisation happens in the data loader (§4b) | Ready for the first pretraining run | **Measured (V):** 32 s for the smoke set, images downscaled to 640 × 360 (51 KB each) |

All resource figures in the table are estimates (D). Download times use the speeds measured on 2026-10-31: 1.6–16 MB/s (§2).

**Where it runs (2026-10-29):**
- **Data build, steps 1–7 [agreed]:** your own computer, CPU only; no GPU is needed. **Changed [agreed 2026-11-08]:** v1-large and v1-xl are built on Anvil's CPU partitions (§10).
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
  - The lists are NAVSIM main's. NAVSIM v1.1's `warmup_test_e2e` is not among them. From v1-xl on, its scenes are kept out of train by a ±7 s window instead (§10, 2026-11-09).

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
- **Lookup fix (2026-11-01):** working out which images to keep took over 5 minutes (`np.isin` on strings); it now takes under a second.
- **Download speed (2026-11-02, V):**
  - **The cause:** Hugging Face sometimes caps a single connection at 1.5–17 MB/s while this line gets ~18 MB/s from other servers.
  - **Reconnecting:** a connection below 5 MB/s for 2 minutes is now dropped and resumed on a new one.
  - **Parallel downloads:** `--workers 3` downloads three archives at once. Extraction stays one at a time, in plan order.

### How to run steps 4–7
From the repo root. Use `--sets smoke` until the full fetch is done, then `--sets smoke v1-pilot` (a frame in both is stored once).
```bash
python -m data_pipeline.v1.canonical  --out data/pretrain/v1.0-pilot --sets smoke            # step 4: frames, privileged, labels, canonical.json
python -m data_pipeline.v1.qa_report  --out data/pretrain/v1.0-pilot --set smoke             # Gate 1: open qa/report.html
python -m data_pipeline.v1.pilot_view build  --out data/pretrain/v1.0-pilot --set smoke      # step 7: views/pilot_v1/smoke/
python -m data_pipeline.v1.pilot_view export --out data/pretrain/v1.0-pilot --set smoke      # chat_*.jsonl for Together.ai
bash scripts/local/build_v1_pilot.sh    # after the full fetch: the steps above for v1-pilot (canonical for both sets), Together's file check, Anvil tar
python -m data_pipeline.v1.release    --out data/pretrain/v1.0-pilot --set v1-pilot          # step 6, after step 5; Gate 2
```
- **Canonical checks:** unique frames; no evaluation log or token; train and dev logs disjoint, dev = nuPlan val; the core path valid and zero at the anchor; path times on the grid; the metadata's CAM_F0 path equals the index's; every other camera listed; one image size; finite calibration; lidar2ego the identity; every JPEG matches its step-3 SHA-256. A failed check makes the command exit with an error.
- **Smoke set, 2026-11-01 (V):** all checks pass. Pose times are within 1 ms of the grid at p95 (3.4 ms at p99). Of the projected 1–4 s path points of moving frames, **99.1% lie below the horizon** (52,080 points, 8,406 frames). The path is (0, 0, 0) at the anchor in every frame. Camera horizontal field of view: 63.7° pinhole, **72.2° through the lens model**.
- **Gate 1 renders, checked by Claude on 12 of the 50 (2026-11-01):** the path lies on the road and follows the lane through straights, curves and a right turn; a stationary frame shows no path. Your look at all 50 is what Gate 1 asks for.
- **`privileged.parquet`** holds the boxes as OpenScene stores them, in the LiDAR frame, which is the ego frame here (lidar2ego is the identity in every frame, checked). Object velocities are stored raw and rotated into the ego frame (L4). Traffic lights use NAVSIM's reading: `True` = red (V, `navsim_scenario.py`). It is 62 MB for the smoke set, mostly boxes (median 36 per frame, p99 314).
- **The pilot view [proposed]:** 640 × 360 images, 220 visual tokens for Qwen3.5 (D, from its image processor's 32 px steps; native would be 2,040). The prompt holds only test-time inputs: speed, acceleration, yaw rate, 1.5 s of past positions and the route command. The answer is 8 "(x, y)" pairs at 1 decimal. **Bar to beat (V, constant velocity along the current heading):** ADE 2.09 m, FDE 4.97 m on the smoke dev split (1.73 / 4.08 m on train).

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
| Canonical rows + validation (step 4) | `data_pipeline/v1/canonical.py` | **Written 2026-11-01**, tested; ran on the smoke set. Object velocities are stored raw (LiDAR frame) and rotated into the ego frame, so the L4 assumption in `navsim_tools/raw_frames.py` is not used. That file is shared with workstream A and is unchanged |
| QA report (path projection, plots) | `data_pipeline/v1/qa_report.py` | **Written 2026-11-01**, tested; ran on the smoke set |
| Release (manifest, checksums, card) | `data_pipeline/v1/release.py` | **Written 2026-11-01**, tested on synthetic data. Runs after step 5; refuses while a check fails or images are missing |
| Pilot view + chat export (step 7) | `data_pipeline/v1/pilot_view.py` | **Written 2026-11-01**, tested; ran on the smoke set |
| Fine-tuning and evaluation of the pilot | `pilot_sft/` (reads only the view), `scripts/anvil/pilot_sft/train_lora.sbatch` | **Written 2026-11-01**. See `pilot_sft/README.md` |
| v1-xl selection: superset of fixed rows; v1-xl-nd filter and label-conflict list | `selection.py` (`XL`, `select(..., fixed=)`), `data_pipeline/v1/novelty.py`, `openscene_images.py plan --extends` | **Written 2026-11-08** [proposed]. `fixed=None` keeps every earlier selection byte-identical |
| Warm-up exclusion: no train anchor within ±7 s of a NAVSIM v1.1 `warmup_test_e2e` scene | `data_pipeline/v1/scene_exclusion.py`, `select(..., train_excluded=)`, `openscene_images.py plan --exclude-scenes --exclude-window-s --exclude-metadata` | **Written 2026-11-09** [proposed], §10. `train_excluded=None` changes nothing |
| Packed images: originals and views | `openscene_images.py fetch --storage pack`, `data_pipeline/v1/image_store.py`; `pilot_view.py build --storage zip`, `pilot_sft/view_images.py` | **Written 2026-11-08** [proposed format]. Default `loose` keeps today's layout; readers take the loose file first |
| Anvil build and checks of v1-xl | `scripts/anvil/data_pipeline/build_v1_xl.{sh,sbatch}`, `verify_v1_xl.py` | **Written 2026-11-08**; not yet run on Anvil (§10) |
| Tests | `tests/data/test_v1_index.py`, `tests/data/test_v1_select_audit.py`, `test_v1_images.py`, `test_v1_canonical.py` | Pass on Python 3.9 + numpy 1.23.4 (NAVSIM's pins) + pyarrow 16, and on Python 3.13 + numpy 2 |

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
  NAVSIM v1.1 (commit 3e8291bf, on Anvil) has a `warmup_test_e2e` filter that main does not have. Its 62 logs are navmini's; the 52 in our index are nuPlan train and val logs. It is kept out of v1-xl's train by a time window instead (§10, V 2026-11-08).

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
2. **Balance:** cap "keep speed + straight" at 40%; thin standstills to 1 anchor per 5 s; space anchors ≥ 2 s apart. **Changed for v1-xl [agreed 2026-11-08]:** train anchors ≥ 1.0 s apart; dev unchanged (§10).
3. **Wide window:** store −4 … +6 s per anchor, although the pilot uses −1.5 … +4.0.
4. **Privileged table:** keep boxes and lights for later teacher labels.
5. **Sizes:** smoke set 10k, then 50k. **Changed at Gate 0 [agreed 2026-10-31]:** v1-pilot 45k + 2k dev (~10 GB), from all 200 archives. **Changed [agreed 2026-11-08]:** v1-large 100k and v1-xl 200k train, each with v1-pilot's 2k dev (§10).
6. **Storage [agreed 2026-10-27]:** local, inside the repo (`data/pretrain/`). No persistent storage for now (§5). **Changed [agreed 2026-11-08]:** v1-large and v1-xl on Anvil scratch; v1-xl's images packed (§10).
7. **Compute [agreed 2026-10-29]:** the data build runs on your own computer. Fine-tuning compute is still open: a cloud GPU or Anvil (§6). **Changed [agreed 2026-11-08]:** the v1-large and v1-xl builds run on Anvil (§10).

## 10. Anvil builds: v1-large and v1-xl
Evidence marks: **V** measured, **R** reported and not re-checked, **D** derived.

### v1-large [agreed 2026-11-03: a larger set, on Anvil; proposed: parameters]
- **Selection:** `selection.LARGE` uses v1-pilot's rules and seed with `n_train` 100,000. At 2.0 s spacing, train is capped at 103,382 (V). Dev is v1-pilot's 2,000, byte-identical (V).
- **Build, 2026-11-03/04 (V):** `build_v1_large.{sh,sbatch}` into `$TRAINING_DATA_ROOT/v1.0-large`. All 200 archives were streamed: 102,000 frames kept, 0 missing, 0 undecodable. Every canonical check passes. The view is `pilot_v1/v1-large` with prompt `pilot_v1.1`. Not released.
- **Never run `build_v1_large.sbatch` again.** Its archive loop deletes the downloaded camera archives, which v1-xl now reads (re-downloaded 2026-11-08, V).

### v1-xl [agreed 2026-11-08: 1.0 s spacing, about 2×, same rules and dev, packed images]
- **The one rule change:** rule 7 for train becomes ≥ 1.0 s (−50 ms tolerance). Rules 1–6, 8 and 9 stay as they are, and so do α = 0.5, the 40% cruise cap, 1 standstill anchor per 5 s and seed 20261027.
- **Superset and dev pinning [proposed mechanics]:**
  - `XL = SelectionConfig("v1-xl", n_train=200_000, n_dev=2_000, spacing_s=1.0, balance_alpha=0.5)`.
  - `select(..., fixed=…)` puts all 102,000 v1-large rows into the per-log picker first, in (log, time) order. They count against the quotas. A fixed row that is ineligible, thinned, too close to another or in the wrong split stops the plan.
  - The 2,000 fixed dev rows fill `n_dev`, so no dev draw happens. Dev is v1-pilot's file (SHA-256 79090629…, V). Without pinning, a 1.0 s dev draw keeps only 1,901 of the 2,000 (V).
  - Since 2026-11-09 the fixed train rows are v1-large's minus the 1,727 inside the warm-up window (next item). So v1-xl is "v1-large minus the warm-up window" plus 101,727 drawn rows.
- **Warm-up exclusion [finding V 2026-11-08; scene-level rule proposed 2026-11-09]:**
  - **Finding.** NAVSIM v1.1 has a scene filter `warmup_test_e2e`, the 2024 challenge warm-up split. It lists 563 tokens in 62 logs (the navmini logs). NAVSIM's own navtrain contains 446 of the 563 tokens (V). Our guard (`navsim_splits.py`) read NAVSIM main (commit 0a380a9), which has no such filter, so the index does not guard these tokens. Source: Anvil's devkit `/home/x-vchia/navsim-v1.1`, commit 3e8291bf…, file SHA-256 4d41f0be… (V).
  - **Why exclude although navtrain contains them:** `CLAUDE.md` says never train on warm-up splits, and that rule wins over navtrain's overlap.
  - **The rule (lead's decision):** from v1-xl on, no TRAIN anchor whose sample window could overlap a warm-up scene's window. It works per scene with a time window, not by whole log.
    - Our sample at anchor t spans [t − 1.5 s, t + 4 s].
    - A v1.1 scene at token time tw spans [tw − 1.5 s, tw + 5 s] (4 history and 10 future frames at 0.5 s).
    - So the two overlap iff t − tw is in [−5.5 s, +6.5 s].
    - We exclude a symmetric ±7.0 s in the same log, inclusive at exactly 7.0 s (`scene_exclusion.window_mask`; `--exclude-window-s`).
  - **The list:** `splits/navsim_v1.1_warmup_test_e2e_tokens.txt` (563 tokens, SHA-256 f5fdc4a1…), an ID list. The manifest's `scene_exclusion` records its source (repo, commit, path, SHA-256), the window and the counts.
  - **Placing the scenes in time (V, 2026-11-09):**
    - 448 of the 563 tokens are index rows, in 52 logs (368 train rows, 80 dev rows). The 52 logs are 43 nuPlan train and 9 val logs.
    - 6 more are frames of 3 indexed logs that fail rule 1, so they are not rows. Two of them have v1-large and v1-xl train rows within 7 s. `--exclude-metadata <openscene_metadata_trainval.tgz>` finds them in OpenScene's metadata by scanning the 52 logs (~30 s, V). It is required: `plan --exclude-scenes` refuses a list with tokens that are not index rows unless it is given (or `--allow-unlocated`, which only reports them and is not for the committed plan), since a plan without it leaves 12 v1-xl train rows within 7 s of these frames (V, 2026-11-09).
    - The other 109 are found in neither. The 10 warm-up logs without a located token are not in the index at all. A token that is not a row cannot be selected.
  - **Effect:**
    - 9,093 train rows of the index are in the window. 8,580 of them survive standstill thinning and are taken out of the train pool.
    - 1,727 v1-large train rows are in the window. 57 of them are warm-up tokens and 1,312 are within 5 s. They are dropped from the fixed rows, with a report, so they are not in v1-xl or v1-xl-nd.
    - v1-xl still reaches 200,000 train, with no shortfall. Its train rows within 7 s of a warm-up scene: 0. Warm-up tokens in it: 0 (V; `verify_v1_xl.py leakage`).
    - 167,617 of the first plan's 200,000 v1-xl train rows stay (V). Only 3,725 of the 32,383 that changed were in the window. The rest moved because the picker's seeds and quotas changed.
  - **Dev is unchanged** (evaluation only, never trained on; byte-identical, SHA-256 79090629…). It holds 3 warm-up tokens (18cf7d6f96f45847, 65f1aecfc27158a7, e6acce4cea5152e2) and 49 rows within 7 s (V). They are reported, not removed.
  - **Earlier sets are unchanged and flagged.** The smoke, v1-pilot and v1-large token files stay as committed, because results were already reported on them. Their train sets hold warm-up tokens: smoke 9, v1-pilot 36, v1-large 57. Rows within 5 s of a warm-up scene: smoke 210, v1-pilot 692, v1-large 1,312 (V, 2026-11-09, counting the 6 frames found in the metadata). Runs trained on them broke the rule for those rows; see open question 7.
- **v1-xl-nd [proposed]:** a subset of v1-xl train for an A/B against v1-xl at equal steps. It needs no extra extraction.
  - It keeps every v1-large row.
  - It drops an *added* Las Vegas row when a kept row of the same log within 1.5 s (+50 ms) is a near-duplicate: the same `decision_lon` and `decision_lat`, |Δspeed| < 0.5 m/s, |Δheading| < 2°, and a mean distance under 0.6 m between the two 4 s targets (`novelty.py`).
  - Given v1-xl, it is deterministic. The thresholds are recorded in `splits/manifest.json`.
- **Label conflicts [proposed, report only]:** `splits/v1-xl_label_conflicts.tsv` (token, set, reasons) lists the rows that fail the hard-bad label tests:
  - the command contradicts the turn;
  - speed is inconsistent;
  - acceleration disagrees with the path by > 3 m/s², or exceeds 8 m/s²;
  - slip > 20°;
  - the heading turns > 0.1 rad in 0.5 s while moving < 0.2 m.

  The plan flags 52 rows of v1-xl train and 1 of dev (V, 2026-11-09). The command contradictions alone are 38 over the three sets (V). Nothing is dropped. A view option to exclude them is [proposed] and off. The list covers the rows of all three sets; 3 of its 19 `v1-large/train` rows are among the 1,727 that v1-xl leaves out.

| (V, 2026-11-09, with the warm-up exclusion) | v1-large | v1-xl | v1-xl-nd |
|---|---|---|---|
| Train / dev | 100,000 / 2,000 | 200,000 / the same 2,000 | 176,352 / the same 2,000 |
| Contains v1-large train | yes | all but the 1,727 rows in the warm-up window | the same |
| Train rows within 7 s of a warm-up scene | 1,727 | 0 | 0 |
| Las Vegas | 76.08% | 75.59% | 72.31% |
| Cruise ("keep speed + straight") | 34.12% | 39.4% | 32.64% |
| Standstill | 16.66% | 9.71% | 10.79% |
| Turns / lane changes | 8,176 / 2,280 | 18,195 / 3,362 | 18,187 / 3,355 |
| Unknown command | 12.3% | 12.3% | 12.74% |
| Near-duplicates within 1 s / 2 s (first plan, 2026-11-08; not re-measured) | 0 / 14.6% | 42.2% / 47.0% | 23.9% / 34.1% |
| Frames to extract | 102,000 | 203,727 for the three sets with dev (101,727 not in v1-large) | 0 extra |

The v1-large set in the v1.0-xl folder keeps all its rows, so the 1,727 rows v1-xl leaves out are still extracted. The 23,648 rows v1-xl-nd drops are all Las Vegas and 89.84% cruise, and include only 8 turns and 7 lane changes (V). Train token files: v1-xl SHA-256 7f6eb976…, v1-xl-nd 5cb19408… (V, 2026-11-09). The first plan's files (2026-11-08, before the exclusion) were d29d2426… and f74bdbcd…. `plan` without `--exclude-scenes` still reproduces them byte for byte (V: `tests/data/test_v1_xl_plan.py`).

**Packed images [agreed 2026-11-08: packing; proposed: format]:**
- **Why:** loose files would add ~415K files, taking scratch to ~1.33M against the 1.0M file quota. Packed, the build adds ~0.6K (D). Scratch held 919.4K files on 2026-11-08 (V).
- **Originals:** `images/packs/<source_archive>.zip`, one per camera archive.
  - Uncompressed (ZIP_STORED); Zip64 is automatic.
  - Member name = canonical `image_path`, and the JPEG bytes are copied unchanged, so `image_path` and `image_sha256` keep their meaning.
  - Members are sorted, with fixed 1980 timestamps, so a rebuild is byte-identical.
  - Each pack is written as `.part`, read back against the frames it should hold, and renamed before the archive's record is written, so a run still resumes per archive and a pack damaged while it was written never gets a record.
  - The record gains `pack`, `pack_members`, `pack_bytes` and `pack_sha256`. Its `members` key keeps its meaning, the number of tar members streamed.
- **Views:** `<view>/images.zip`. Member = the jsonl `image` field (`images/<log>/<token>.jpg`), so `train.jsonl` and `dev.jsonl` are unchanged.
- **Readers:** `image_store.py` (originals) and `pilot_sft/view_images.py` (views) read the loose file if it exists, else the zip member. Older loose builds and views keep working. A loose view build refuses a view folder that already holds `images.zip`, since the readers would serve the zip.

**How to run (Anvil, from the repo root):**
```bash
python -m data_pipeline.v1.openscene_images plan --out data/pretrain/v1.0-xl --set v1-xl \
    --extends data/pretrain/v1.0-large/splits/v1-large \
    --exclude-scenes data/pretrain/v1.0-xl/splits/navsim_v1.1_warmup_test_e2e_tokens.txt \
    --exclude-metadata data/pretrain/downloads/openscene_metadata_trainval.tgz   # locally, ~1 min; commit splits/
bash scripts/anvil/data_pipeline/build_v1_xl.sh setup         # login node: copies index/, splits/, archives.json, SHA-checked
OUT=$TRAINING_DATA_ROOT/v1.0-xl-smoke bash scripts/anvil/data_pipeline/build_v1_xl.sh setup     # smoke test first:
OUT=$TRAINING_DATA_ROOT/v1.0-xl-smoke ARCHIVES=2 sbatch -A cis220051 -p debug --gpus-per-node=0 \
    scripts/anvil/data_pipeline/build_v1_xl.sbatch            #   2 archives, pack reproducibility, canonical, views
sbatch -A cis220051 -p debug --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_xl.sbatch  # repeat until "done"
```
- **The job:**
  - fetch with `--storage pack`, 96 processes;
  - cross-check the 102,000 shared frames' SHA-256 against v1.0-large's records;
  - canonical for `v1-xl v1-xl-nd v1-large`;
  - QA summary;
  - the two zip views;
  - the links `$TRAINING_DATA_ROOT/pilot_v1/{v1-xl,v1-xl-nd}`;
  - `verify_v1_xl.py all`.
- **Resuming across 30-minute debug slots:** each step is skipped once its output exists or its check passes. A step that would not fit in the time left stops the job cleanly ("RESUBMIT"). Fetch resumes per archive. The files a step takes as done (records, `canonical.json`, `qa/summary.json`, `view.json`) are written whole (`.part`, fsync, rename), so a job killed mid-write leaves no truncated marker.
- **One job at a time:** two jobs on one `OUT` would write the same `.part` files. `#SBATCH --dependency=singleton` holds a second `v1-xl` job until the first ends, and a `flock` on `$OUT/.build_v1_xl.lock` refuses a second copy of the job script on the same `OUT` (e.g. one run with `bash`). A job submitted while another runs waits for it; do not run the pipeline commands by hand while a job runs.
- **Quota guard:** before every step, `myquota` must show at least 20,000 files free on scratch, or the job stops. It also stops if `myquota` cannot be read. `myquota` reports a cached count, so it can lag by minutes.
- **Nothing is deleted or moved.** The archives are read in place from `v1.0-large/archives_hf/openscene-v1.1/openscene_sensor_trainval_camera`, a path set in the script, not derived from `OUT`. v1.0-large is only read.
- **What `verify_v1_xl.py` checks** (each step also writes `verify/<step>.json`):
  - dev SHA-256 79090629… in all three sets;
  - (v1-large minus the 1,727 rows in the warm-up window) ⊂ v1-xl-nd ⊂ v1-xl, with the counts and train SHA-256 above;
  - no token or log in NAVSIM's guarded sets, by the index's `our_split` (and the devkit's lists when present);
  - train and dev logs disjoint;
  - no v1-xl / v1-xl-nd train token is a warm-up token or within 7 s of one. This uses the committed list, the index and OpenScene's metadata (`--metadata`): verify locates the listed tokens that are not index rows itself, over the logs of every train row and listed row (1,094 logs, ~1 min locally, V), and they must equal the plan's `located_in_metadata` (the 6). Without `--metadata` this check fails, since the window cannot be fully checked. Every v1-large row left out of v1-xl must be inside the window. Dev's 3 warm-up tokens and v1-large's 57 are reported;
  - with `--navsim-root`: the devkit's `warmup_test_e2e` must equal the committed list and hold no v1-xl / v1-xl-nd train token. Dev hits are reported, and its logs are not guarded as a whole. Every other guarded list (nuPlan test logs, navtest, navhard, private_test_e2e, competition_private_tokens, …) must have 0 tokens and 0 logs in train and dev;
  - every pack holds exactly its record's frames, and 203,727 in total;
  - the 102,000 shared frames match v1.0-large;
  - every canonical check passes;
  - the view zips hold exactly the jsonl images;
  - each view's `dev.jsonl` is byte-identical to the v1-large view's;
  - 50 dev images are byte-identical to the v1-large view's.

  `DEEP=1` also runs `testzip` and re-hashes every zip (~55 GB read, D).
- **Sizes (D):** packed originals ~41.6 GB, the v1-xl view ~10.1 GB, the v1-xl-nd view ~8.9 GB.

**Open questions for the researcher [open]:**
1. **NAVSIM-mix tilt.** Turns are 9.4% of v1-xl train (V) against 23.6% of navtrain, and cruise 40% against 15% (R). Tilt a later view towards navtrain's mix? A tilt needs no re-extraction.
2. **Unknown-command rows:** 12.1% of v1-xl train (V), against 0% in navtrain (R). Keep them, as rule 9 does now, or leave them out of a view?
3. **Standstill depth.** v1-xl keeps rule 6 (1 anchor per 5 s); standstill is 9.66% of train (V). 60.2% of its stopped samples stay within 0.5 m over 4 s (V), against 0.9% in navtrain (R). Keep this, or thin further?
4. **Dev overlaps navtrain:** 475 of the 2,000 dev samples are navtrain tokens (R). This matters only for a fine-tune on navtrain.
5. **Gate 1 and Gate 2:** the look at the 50 renders (v1-xl's `qa/report.html`), and the release sign-off. `release.py` takes only smoke and v1-pilot today (V). Training on the unreleased views follows v1-large's precedent.
6. **After Nov 16:** scratch is flushed in the Nov 16–23 reservation (R). Where should a copy be kept?
7. **Warm-up exclusion (2026-11-09):**
   - Scene-level with ±7 s, as now, or whole warm-up logs? Whole logs would take the 52 indexed warm-up logs out of train; 43 of them are nuPlan train logs (V).
   - Retrain the runs on v1-pilot and v1-large, whose train sets contain warm-up tokens, or keep them and flag them?
   - Should NAVSIM v1.1's filters go into the index guard? `navsim_splits.py` reads NAVSIM main today. Rebuilding the index with the v1.1 devkit would guard all 62 warm-up logs as a whole, which is the log-level choice.
