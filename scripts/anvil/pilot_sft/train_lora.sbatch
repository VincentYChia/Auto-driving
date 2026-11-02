#!/bin/bash
# LoRA fine-tuning of the Qwen pilot on the pilot_v1 view (pilot_sft/README.md). One A100 40 GB.
# Copy the view first (plan §6, "Where it runs"): rsync -a data/pretrain/v1.0-pilot/views/pilot_v1/ anvil:$TRAINING_DATA_ROOT/pilot_v1/
#
#   # timing run first: 200 optimizer steps on the debug partition
#   SET=v1-pilot MAX_STEPS=200 EVAL_SAMPLES=40 sbatch --time=00:30:00 --partition=gpu-debug scripts/anvil/pilot_sft/train_lora.sbatch
#   # (gpu-debug stops jobs at 30 min; the small EVAL_SAMPLES keeps the scoring and probes inside it)
#   # then the run
#   SET=v1-pilot EPOCHS=2 sbatch scripts/anvil/pilot_sft/train_lora.sbatch
#   # EGO_DROPOUT=0.5 hides ego state and past positions on half the training samples [proposed]
#   # a job that hit its time limit or was preempted: resubmit with its run folder; it resumes from checkpoint.pt
#   RUN_DIR=$NAVSIM_EXP_ROOT/pilot_sft/<model>-<set>-<job> SET=v1-pilot sbatch scripts/anvil/pilot_sft/train_lora.sbatch
#SBATCH --job-name=pilot-sft
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${PILOT_CONDA_ENV:-pilot}"
SET="${SET:-smoke}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
VIEW="$TRAINING_DATA_ROOT/pilot_v1/$SET"
RUN="${RUN_DIR:-$NAVSIM_EXP_ROOT/pilot_sft/$(basename "$MODEL")-$SET-${SLURM_JOB_ID:-local}}"
EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")
cd "$REPO_ROOT"
python pilot_sft/train_lora.py --view "$VIEW" --model "$MODEL" --out "$RUN" \
    --epochs "${EPOCHS:-2}" --batch "${BATCH:-8}" --accum "${ACCUM:-2}" --ego-dropout "${EGO_DROPOUT:-0}" \
    --resume ${EXTRA[@]+"${EXTRA[@]}"}
python pilot_sft/evaluate.py --view "$VIEW" --model "$MODEL" --adapter "$RUN/adapter" \
    --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/eval.json"
# the bars to beat (no image) and what the model relies on, on the same dev samples
python pilot_sft/baselines.py --view "$VIEW" --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/baselines.json"
python pilot_sft/probes.py --view "$VIEW" --model "$MODEL" --adapter "$RUN/adapter" \
    --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/probes.json"
