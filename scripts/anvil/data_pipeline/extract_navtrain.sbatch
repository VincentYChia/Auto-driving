#!/bin/bash
# PROTOTYPE (v0.1): smoke tests only until the training-data v1 design is agreed.
# Build the canonical training samples from navtrain logs, then the model views.
# Needs only navsim_logs/trainval (~14 GB); images are referenced by path.
#
#   sbatch scripts/anvil/data_pipeline/extract_navtrain.sbatch
#   MAX_LOGS=5 sbatch --partition=gpu-debug --time=00:30:00 scripts/anvil/data_pipeline/extract_navtrain.sbatch   # smoke test
#SBATCH --job-name=extract-navtrain
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
# PROTOTYPE guard: the training-data design is not agreed yet (docs/training_data/README.md).
if [ -z "${MAX_LOGS:-}" ] && [ "${ALLOW_FULL_RUN:-0}" != 1 ]; then
    echo "Prototype pipeline: run a 5-log smoke test: MAX_LOGS=5, or set ALLOW_FULL_RUN=1 once the v1 design is signed off." >&2
    exit 1
fi
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
