# Runbook: MVLAD-AD on NAVSIM, and building training data (Anvil)

Two independent tracks:
- **A.** Score the existing MVLAD-AD model on NAVSIM (a quick baseline).
- **B.** Build training data for the Qwen pilot and the Gemma reasoner.

Both tracks share step 0.

> **Status:** every Python component here passes `tests/` against the real NAVSIM devkit (v1.1 and v2), using synthetic NAVSIM-format logs. Two things have **not** been run yet, because they need Anvil and the real data:
> - the GPU inference script (`inference/predict_navsim.py`);
> - the Slurm scripts.
>
> Expect small path fixes the first time.

## 0. One-time setup

1. **Clone code** (on Anvil, in `$HOME` or project space):
   ```bash
   git clone -b claude/stoic-darwin-ue0jm6 https://github.com/VincentYChia/Auto-driving.git ~/Auto-driving
   git clone https://github.com/autonomousvision/navsim.git ~/navsim             # main = NAVSIM v2
   git -C ~/navsim worktree add ~/navsim-v1.1 origin/v1.1                        # v1.1 = PDMS / navtest leaderboard
   ```
2. **Configure paths:**
   ```bash
   cp scripts/anvil/config.sh scripts/anvil/config.local.sh
   ```
   Then edit `config.local.sh`:
   - data roots: use project space if you have it, since scratch is purged after 30 days unused;
   - `NAVSIM_DEVKIT_ROOT`: point it at `~/navsim-v1.1` for PDMS or `~/navsim` for EPDMS. Install the matching one into the env.

   Then add `export SBATCH_ACCOUNT=<your GPU allocation account>` to `~/.bashrc` (`mybalance` lists it). **Submit every job from the repo root.**
3. **NAVSIM environment** (Python 3.9; follow NAVSIM's `docs/install.md`):
   ```bash
   module load anaconda
   conda env create --name navsim -f ~/navsim-v1.1/environment.yml
   conda activate navsim && pip install -e ~/navsim-v1.1
   ```
4. **MVLAD environment:** see the top-level README.
   - `pip install -e train` plus PyTorch.
   - `huggingface-cli login`, because LLaDA-V uses a gated LLaMA-3 tokenizer.
   - The shipped weights record `transformers` 4.39.3.
5. **Weights** (the fork only has Git LFS pointers):
   ```bash
   bash scripts/download_mvlad_weights.sh
   ```
   This downloads from the official `lan-qing/MVLAD-AD` repository and checks every file's sha256 against its pointer.
6. **Data** (login node; long-running; `screen`/`tmux` recommended):
   ```bash
   bash scripts/anvil/download_navsim_data.sh maps
   bash scripts/anvil/download_navsim_data.sh test-logs         # track A
   bash scripts/anvil/download_navsim_data.sh test-cameras      # track A (cameras only, no LiDAR)
   bash scripts/anvil/download_navsim_data.sh trainval-logs     # track B (~14 GB, enough for extraction)
   bash scripts/anvil/download_navsim_data.sh navtrain-current  # track B, only needed before training on images
   bash scripts/anvil/download_navsim_data.sh navtrain-history  # track B, only for the multi-frame variant / tracker runs
   ```

## A. MVLAD-AD baseline on navtest

| Step | Command | Runs where |
|---|---|---|
| A1. Export inputs | `sbatch scripts/anvil/navsim_export.sbatch` | NAVSIM env, CPU (debug queue) |
| A2. Timing run | `MAX_SAMPLES=20 sbatch --array=0 --partition=gpu-debug --time=00:30:00 scripts/anvil/mvlad_predict.sbatch` | MVLAD env, 1 A100 |
| A3. Full inference | `sbatch --array=0-7 scripts/anvil/mvlad_predict.sbatch` (resubmit to resume) | MVLAD env, 8× A100 |
| A4. Metric cache | `STEP=cache sbatch scripts/anvil/navsim_score.sbatch` | NAVSIM env, CPU |
| A5. Sanity score | `STEP=cv sbatch scripts/anvil/navsim_score.sbatch` | NAVSIM env, CPU |
| A6. Score MVLAD-AD | `STEP=mvlad sbatch scripts/anvil/navsim_score.sbatch` | NAVSIM env, CPU |

**What each step does:**
- **A1** writes `$NAVSIM_EXP_ROOT/mvlad/navtest_inputs.json`, with one record per scene: the CAM_F0 image path plus MVLAD-AD's training-format prompt.
  - It is built only from what a NAVSIM agent may see.
  - It aborts if our ego history differs from NAVSIM's own `AgentInput`.
- **A2** prints the mean model time per scene. GPU-hours for the full split = that × (number of exported scenes) / 3600. Pick the array size from that.
- **A3** writes `navtest_predictions/shard_*.jsonl`. Failed scenes are listed in `errors_*.jsonl`. Re-running skips finished scenes.
- **A5** runs NAVSIM's own constant-velocity agent, which checks the cache and scorer before we trust the MVLAD number. Compare it with the constant-velocity row that NAVSIM publishes.
- **A6** replays the predictions through `navsim_tools.precomputed_agent`. NAVSIM writes a CSV per run under `$NAVSIM_EXP_ROOT/<experiment_name>/`.

**How MVLAD-AD's outputs are adapted to NAVSIM:**
- **History:** MVLAD-AD was trained with 10 past waypoints (4.5 s); NAVSIM gives 4 (1.5 s). The default `--history-mode pad_cv` back-fills the older points at constant velocity to keep the trained format. `available` passes only the 4 real points; run both if the history turns out to matter.
- **Output:** MVLAD-AD emits 10 waypoints at 0.5 s (5 s). The scorer uses the first 8 (4 s) and sets each heading from the direction of travel.
- **Camera:** nuScenes CAM_FRONT (training) vs nuPlan CAM_F0. Expect a large zero-shot gap; this run mainly validates the pipeline and gives a floor.
- **Codebook:** the 256-token codebook rounds waypoints by **0.54 m on average at 1 s**, and its centres stop at **54.6 m ahead** (measured, see `RESEARCH_DIRECTION.md`). That floor applies here too.

**Versions:**
- `NAVSIM_VERSION=v1` (default; devkit `navsim-v1.1`) gives **PDMS** on navtest, comparable to most published VLA numbers.
- `NAVSIM_VERSION=v2` (devkit `navsim` main) gives **EPDMS**. navhard two-stage additionally needs predictions for the synthetic second-stage scenes, which the exporter does not produce yet.

## B. Training data (canonical samples → views → reasoning labels)

| Step | Command | Output |
|---|---|---|
| B1. Smoke test | `MAX_LOGS=5 sbatch --partition=gpu-debug --time=00:30:00 scripts/anvil/extract_navtrain.sbatch` | a few shards + views |
| B2. Full extraction + views | `sbatch scripts/anvil/extract_navtrain.sbatch` | `canonical/navtrain/samples-*.jsonl.gz`, `views/navtrain/*.jsonl` |
| B3. Teacher labels, pilot batch | `TEACHER_MODEL=<model> LIMIT=200 sbatch scripts/anvil/teacher_labels.sbatch` | `labels/navtrain_teacher.jsonl` |
| B4. Inspect labels, then the full run | same without `LIMIT` | |
| B5. Reasoner views | `python -m data_pipeline.build_views ... --labels labels/navtrain_teacher.jsonl --no-teacher-requests` | `views/navtrain/reasoner.jsonl` |

**B2 details:**
- Extraction needs only `navsim_logs/trainval`. It refuses NAVSIM test splits.
- It checks its geometry against the devkit on the first scene of every batch.
- `canonical/navtrain/manifest.json` records counts, errors and provenance.

**Views in `views/navtrain/`:**
- `pilot_front.jsonl`, `pilot_front_motion.jsonl`, `pilot_front_history.jsonl`: the three temporal-context variants compared by claim C1. Same samples and targets; only the context differs.
- `teacher_requests.jsonl`: hindsight prompts for the teacher model. These **contain privileged future information by design**.
- `reasoner.jsonl` (after B5): Gemma training examples. Their inputs contain test-time information only.

**B3 teacher model:**
- Any VLM that vLLM can serve on 4× A100 40 GB works, e.g. a ~30B model in bf16 with tensor parallelism 4.
- Check its licence allows using its outputs as training data.
- Read ~50 labels from the pilot batch before the full run. Rows whose JSON failed to parse keep the raw text in `raw`.

The record format is documented in `docs/DATASET_SCHEMA.md`.

## Troubleshooting
- **`KeyError: 'cam_l0'` or similar:** a log is missing cameras. NAVSIM expects all 8 OpenScene cameras in every frame.
- **The export says images are missing:** the camera download is incomplete, or `OPENSCENE_DATA_ROOT` points at the wrong root.
- **Ray uses all cores of the node:** the job scripts pass `worker.threads_per_node=$SLURM_CPUS_PER_TASK`.
- **Hydra cannot find `navsim_tools`:** `PYTHONPATH` must include the repo root. `common.sh` sets it.
