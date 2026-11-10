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
#   (steps whose output exists are skipped, so resubmitting a finished or half-scored run is safe)
# --time-min lets the backfill scheduler start the job in a gap shorter than 24 h; it then resumes on resubmit.
#SBATCH --job-name=pilot-sft
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --time-min=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${PILOT_CONDA_ENV:-pilot}"
SET="${SET:-smoke}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
VIEW="$TRAINING_DATA_ROOT/pilot_v1/$SET"
RUN="${RUN_DIR:-$NAVSIM_EXP_ROOT/pilot_sft/$(basename "$MODEL")-$SET-${SLURM_JOB_ID:-local}}"
# VISION=frozen|lora|full (VISION_LR for full) trains the vision encoder too; SNAPSHOTS=1 keeps and scores
# the adapter at every epoch end; LIMIT=N uses the first N train samples and DEV_LOSS_SAMPLES the dev-loss
# sample size (tests) [proposed 2026-11-08]. WORKERS=N builds the training micro-batches in N threads ahead of
# the GPU; same batches for any N, and a resubmit may change it. TORCH_THREADS=T caps torch's CPU threads: on
# CPU, workers without a cap gave no speed-up. Until a GPU timing run (MAX_STEPS=200, WORKERS 0/1/2, with and
# without TORCH_THREADS=2) settles it, keep WORKERS at 1-2 with TORCH_THREADS=2 [proposed 2026-11-09]
EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
[ -n "${SNAPSHOTS:-}" ] && EXTRA+=(--snapshot-epochs)
[ -n "${DEV_LOSS_SAMPLES:-}" ] && EXTRA+=(--eval-samples "$DEV_LOSS_SAMPLES")
[ -n "${WORKERS:-}" ] && EXTRA+=(--workers "$WORKERS")
[ -n "${TORCH_THREADS:-}" ] && EXTRA+=(--torch-threads "$TORCH_THREADS")
cd "$REPO_ROOT"
# run.json is written once the adapter is saved; without this check a resubmit would start training over
[ -f "$RUN/run.json" ] || python pilot_sft/train_lora.py --view "$VIEW" --model "$MODEL" --out "$RUN" \
    --epochs "${EPOCHS:-2}" --batch "${BATCH:-8}" --accum "${ACCUM:-2}" --ego-dropout "${EGO_DROPOUT:-0}" \
    --lr "${LR:-2e-4}" --rank "${LORA_RANK:-32}" --alpha "${LORA_ALPHA:-64}" \
    --vision "${VISION:-frozen}" --vision-lr "${VISION_LR:-2e-5}" --resume ${EXTRA[@]+"${EXTRA[@]}"}
[ -f "$RUN/eval.json" ] || python pilot_sft/evaluate.py --view "$VIEW" --model "$MODEL" --adapter "$RUN/adapter" \
    --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/eval.json"
# fit on the training data: a fixed seeded sample of train.jsonl, full input (no ego masking). Train ADE well
# below dev ADE, with dev rising, is memorization; both low and close is learning the task
[ -f "$RUN/eval_train.json" ] || python pilot_sft/evaluate.py --view "$VIEW" --split train --model "$MODEL" \
    --adapter "$RUN/adapter" --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/eval_train.json"
# the same dev and train samples for each epoch snapshot, to see where more passes stop helping
for snap in "$RUN"/epoch_*/; do
    [ -f "$snap/adapter_config.json" ] || continue
    k=$(basename "$snap")
    [ -f "$RUN/eval_$k.json" ] || python pilot_sft/evaluate.py --view "$VIEW" --model "$MODEL" --adapter "$snap" \
        --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/eval_$k.json"
    [ -f "$RUN/eval_train_$k.json" ] || python pilot_sft/evaluate.py --view "$VIEW" --split train --model "$MODEL" \
        --adapter "$snap" --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/eval_train_$k.json"
done
# the bars to beat (no image) and what the model relies on, on the same dev samples
[ -f "$RUN/baselines.json" ] || python pilot_sft/baselines.py --view "$VIEW" --samples "${EVAL_SAMPLES:-1000}" \
    --out "$RUN/baselines.json"
[ -f "$RUN/probes.json" ] || python pilot_sft/probes.py --view "$VIEW" --model "$MODEL" --adapter "$RUN/adapter" \
    --samples "${EVAL_SAMPLES:-1000}" --out "$RUN/probes.json"
