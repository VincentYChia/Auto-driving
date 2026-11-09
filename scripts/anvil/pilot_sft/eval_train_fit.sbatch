#!/bin/bash
# Fit on the training data for finished pilot runs: ADE on a fixed seeded sample of each run's own train.jsonl
# (full input, no ego masking), written to <run>/eval_train.json beside eval.json (dev). One GPU for all runs.
#
#   RUNS="v1-pilot:qwen35-0.8b-v1pilot-egodrop50 v1-large:qwen35-0.8b-large-ep3" \
#       sbatch -A <account> -p <partition> scripts/anvil/pilot_sft/eval_train_fit.sbatch
#
# Each RUNS entry is <set>:<run folder under $NAVSIM_EXP_ROOT/pilot_sft/>. Runs that already have
# eval_train.json are skipped, so a resubmit after a time limit picks up where it stopped.
#SBATCH --job-name=train-fit
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --time-min=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${PILOT_CONDA_ENV:-pilot}"
cd "$REPO_ROOT"
for entry in ${RUNS:?set RUNS to "<set>:<run folder>" entries}; do
    set_name=${entry%%:*}
    run=$NAVSIM_EXP_ROOT/pilot_sft/${entry#*:}
    [ -f "$run/eval_train.json" ] && continue
    model=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['args']['model'])" "$run/run.json")
    python pilot_sft/evaluate.py --view "$TRAINING_DATA_ROOT/pilot_v1/$set_name" --split train --model "$model" \
        --adapter "$run/adapter" --samples "${EVAL_SAMPLES:-1000}" --out "$run/eval_train.json"
done
