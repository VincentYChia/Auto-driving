#!/bin/bash
# MVLAD-AD inference on exported NAVSIM inputs: one array task per shard, one A100 each.
# Resubmitting the same command resumes (finished scenes are skipped).
#
#   # timing run first: 20 scenes on one shard
#   MAX_SAMPLES=20 sbatch --array=0 --time=00:30:00 --partition=gpu-debug scripts/anvil/mvlad_predict.sbatch
#   # full run on 8 GPUs
#   sbatch --array=0-7 scripts/anvil/mvlad_predict.sbatch
#SBATCH --job-name=mvlad-predict
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "$MVLAD_CONDA_ENV"
SPLIT="${SPLIT:-navtest}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
EXTRA=()
[ -n "${MAX_SAMPLES:-}" ] && EXTRA+=(--max-samples "$MAX_SAMPLES")
cd "$REPO_ROOT"
python inference/predict_navsim.py \
    --inputs "$NAVSIM_EXP_ROOT/mvlad/${SPLIT}_inputs.json" \
    --output-dir "$NAVSIM_EXP_ROOT/mvlad/${SPLIT}_predictions" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" ${EXTRA[@]+"${EXTRA[@]}"}
