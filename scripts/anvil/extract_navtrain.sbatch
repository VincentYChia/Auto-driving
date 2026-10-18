#!/bin/bash
# Build the canonical training samples from navtrain logs, then the model views.
# Needs only navsim_logs/trainval (~14 GB); images are referenced by path.
#
#   sbatch scripts/anvil/extract_navtrain.sbatch
#   MAX_LOGS=5 sbatch --partition=gpu-debug --time=00:30:00 scripts/anvil/extract_navtrain.sbatch   # smoke test
#SBATCH --job-name=extract-navtrain
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "$NAVSIM_CONDA_ENV"
OUT="$TRAINING_DATA_ROOT/canonical/navtrain"
EXTRA=()
[ -n "${MAX_LOGS:-}" ] && EXTRA+=(--max-logs "$MAX_LOGS")
cd "$REPO_ROOT"
python -m data_pipeline.extract_navsim --split navtrain --output-dir "$OUT" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${EXTRA[@]+"${EXTRA[@]}"}
python -m data_pipeline.build_views --canonical "$OUT" \
    --sensor-root "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
    --output-dir "$TRAINING_DATA_ROOT/views/navtrain"
