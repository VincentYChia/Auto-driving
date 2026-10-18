# Workstream A: score MVLAD-AD on NAVSIM (Anvil runbook)

This workstream is a **quick, deterministic baseline**: the shipped MVLAD-AD planning model, zero-shot, on navtest. It is independent of the training-data workstream (`docs/training_data/`).

- **Code:**
  - `navsim_eval/` (export → predict → replay agent);
  - shared NAVSIM access in `navsim_tools/`;
  - jobs in `scripts/anvil/navsim_eval/`.
- **Status:** the Python components pass `tests/shared` and `tests/eval` against the real NAVSIM devkit (v1.1 and v2) on synthetic logs. The GPU inference script and the Slurm jobs have **not** run on Anvil yet; expect small path fixes the first time.

## 0. One-time setup

1. **Code:**
   ```bash
   git clone -b claude/stoic-darwin-ue0jm6 https://github.com/VincentYChia/Auto-driving.git ~/Auto-driving
   git clone https://github.com/autonomousvision/navsim.git ~/navsim             # main = NAVSIM v2 (EPDMS)
   git -C ~/navsim worktree add ~/navsim-v1.1 origin/v1.1                        # v1.1 = PDMS (navtest leaderboard)
   ```
2. **Paths:**
   ```bash
   cp scripts/anvil/config.sh scripts/anvil/config.local.sh
   ```
   Then edit `config.local.sh`:
   - the data roots;
   - `NAVSIM_DEVKIT_ROOT`: `~/navsim-v1.1` for PDMS or `~/navsim` for EPDMS. Install the same one into the env.

   Then add `export SBATCH_ACCOUNT=<GPU allocation>` to `~/.bashrc` (`mybalance` lists it). Submit every job from the repo root.
3. **NAVSIM env** (Python 3.9; see NAVSIM's `docs/install.md`):
   ```bash
   conda env create --name navsim -f ~/navsim-v1.1/environment.yml && conda activate navsim && pip install -e ~/navsim-v1.1
   ```
4. **MVLAD env:** see the top-level README.
   - `pip install -e train` plus PyTorch.
   - `huggingface-cli login`, because LLaDA-V uses a gated LLaMA-3 tokenizer.
   - The weights record `transformers` 4.39.3.
5. **Weights:**
   ```bash
   bash scripts/download_mvlad_weights.sh
   ```
   This fetches from the official `lan-qing/MVLAD-AD` repo and checks each sha256 against its LFS pointer.
6. **Data** (login node; long):
   ```bash
   bash scripts/anvil/download_navsim_data.sh maps
   bash scripts/anvil/download_navsim_data.sh test-logs        # ~1 GB
   bash scripts/anvil/download_navsim_data.sh test-cameras     # cameras only, no LiDAR
   ```

## 1. Run

| Step | Command | Env / hardware |
|---|---|---|
| 1. Export inputs | `sbatch scripts/anvil/navsim_eval/navsim_export.sbatch` | NAVSIM, CPU (debug queue) |
| 2. Timing run | `MAX_SAMPLES=20 sbatch --array=0 --partition=gpu-debug --time=00:30:00 scripts/anvil/navsim_eval/mvlad_predict.sbatch` | MVLAD, 1 A100 |
| 3. Inference | `sbatch --array=0-7 scripts/anvil/navsim_eval/mvlad_predict.sbatch` (resubmit to resume) | MVLAD, 8 A100 |
| 4. Metric cache | `STEP=cache sbatch scripts/anvil/navsim_eval/navsim_score.sbatch` | NAVSIM, CPU |
| 5. Sanity score | `STEP=cv sbatch scripts/anvil/navsim_eval/navsim_score.sbatch` | NAVSIM, CPU |
| 6. Score MVLAD-AD | `STEP=mvlad sbatch scripts/anvil/navsim_eval/navsim_score.sbatch` | NAVSIM, CPU |

**What each step does:**
- **Step 1** writes `$NAVSIM_EXP_ROOT/mvlad/navtest_inputs.json`: the CAM_F0 path plus MVLAD-AD's training-format prompt, built only from AgentInput data. It aborts if our ego history differs from NAVSIM's `AgentInput`.
- **Step 2** prints the mean model time per scene. Full-split GPU-hours = that × number of scenes / 3600.
- **Step 3** writes `navtest_predictions/shard_*.jsonl`. Failed scenes go to `errors_*.jsonl`.
- **Step 5** runs NAVSIM's constant-velocity agent. Compare it with NAVSIM's published constant-velocity score before trusting step 6.
- **Step 6** writes NAVSIM's CSV under `$NAVSIM_EXP_ROOT/mvlad_navtest_<version>/`.

## 2. How MVLAD-AD is adapted (read before interpreting the score)
- **History:** MVLAD-AD was trained with 10 past waypoints (4.5 s); NAVSIM gives 4 (1.5 s).
  - `--history-mode pad_cv` (default) back-fills older points at constant velocity.
  - `available` passes only the 4 real points.
- **Output:** 10 waypoints at 0.5 s. The scorer uses the first 8 (4 s), with headings from the direction of travel.
- **Camera:** trained on nuScenes CAM_FRONT, scored on nuPlan CAM_F0. Expect a large zero-shot gap; this run mostly validates the pipeline and gives a floor.
- **Codebook:** the 256-token codebook rounds waypoints by 0.54 m on average at 1 s, and its centres stop at 54.6 m ahead (measured; `docs/RESEARCH_DIRECTION.md`).
- **Versions:** `NAVSIM_VERSION=v1` (default) gives PDMS with the v1.1 devkit; `v2` gives EPDMS with main. navhard two-stage would also need predictions for the synthetic second-stage scenes, which the exporter does not produce.

## Troubleshooting
- **`KeyError: 'cam_l0'`:** a log is missing cameras. NAVSIM expects all 8 OpenScene cameras per frame.
- **Export reports missing images:** the camera download is incomplete, or `OPENSCENE_DATA_ROOT` is wrong.
- **Ray grabs every core:** jobs pass `worker.threads_per_node=$SLURM_CPUS_PER_TASK`.
- **Hydra cannot find `navsim_eval`:** the repo root must be on `PYTHONPATH`. `scripts/anvil/common.sh` sets it.
