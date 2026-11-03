#!/bin/bash
# A trained pilot on NAVSIM scenes (pilot_sft/predict_navsim.py): one array task per shard, one A100 each.
# Needs the exported inputs (scripts/anvil/navsim_eval/navsim_export.sbatch). Resubmitting resumes.
#
#   RUN_DIR=$NAVSIM_EXP_ROOT/pilot_sft/<run> MAX_SAMPLES=40 sbatch --array=0 --time=00:30:00 --partition=gpu-debug \
#       scripts/anvil/pilot_sft/predict_navsim.sbatch                                    # timing
#   RUN_DIR=$NAVSIM_EXP_ROOT/pilot_sft/<run> sbatch --array=0-3 scripts/anvil/pilot_sft/predict_navsim.sbatch
#   # then score: STEP=pilot PREDICTIONS=<run>/navtest_predictions NAME=<run> sbatch scripts/anvil/navsim_eval/navsim_score.sbatch
#SBATCH --job-name=pilot-navsim
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --time-min=01:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${PILOT_CONDA_ENV:-pilot}"
SPLIT="${SPLIT:-navtest}"
RUN="${RUN_DIR:?set RUN_DIR to a finished pilot_sft run (with adapter/)}"
MODEL="${MODEL:-$(python -c "import json; print(json.load(open('$RUN/run.json'))['args']['model'])")}"
EXTRA=()
[ -n "${MAX_SAMPLES:-}" ] && EXTRA+=(--max-samples "$MAX_SAMPLES")
cd "$REPO_ROOT"
python pilot_sft/predict_navsim.py --inputs "$NAVSIM_EXP_ROOT/mvlad/${SPLIT}_inputs.json" \
    --model "$MODEL" --adapter "$RUN/adapter" --out "$RUN/${SPLIT}_predictions" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
