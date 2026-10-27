#!/bin/bash
# Pretraining set v1, steps 1-2 (docs/training_data/PRETRAIN_V1_PLAN.md §6): index the OpenScene trainval
# metadata and write the Gate 0 audit. Reads no images.
# Input: the logs from `bash scripts/anvil/download_navsim_data.sh trainval-logs`, or
#   METADATA=<path or URL of openscene_metadata_trainval.tgz>, read as a stream without extracting it.
# Output, in the repo: data/pretrain/v1.0-pilot/ (smoke tests: data/pretrain/smoke/). index/ stays
#   local; audit/ and splits/ can be committed (.gitignore).
#
#   MAX_LOGS=20 sbatch --partition=gpu-debug --time=00:30:00 scripts/anvil/data_pipeline/pretrain_v1_index.sbatch   # smoke test
#   ALLOW_FULL_RUN=1 sbatch scripts/anvil/data_pipeline/pretrain_v1_index.sbatch                                   # step 2: all logs
#SBATCH --job-name=pretrain-v1-index
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
if [ -z "${MAX_LOGS:-}" ] && [ "${ALLOW_FULL_RUN:-0}" != 1 ]; then
    echo "Run a smoke test first (MAX_LOGS=20), then set ALLOW_FULL_RUN=1 to index all logs." >&2
    exit 1
fi
METADATA="${METADATA:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}"
if [[ "$METADATA" != http* ]] && [ ! -e "$METADATA" ]; then
    echo "No metadata at $METADATA: run 'bash scripts/anvil/download_navsim_data.sh trainval-logs' or set METADATA." >&2
    exit 1
fi
EXTRA=()
if [ -n "${MAX_LOGS:-}" ]; then
    OUT="${OUT:-$REPO_ROOT/data/pretrain/smoke}"
    EXTRA+=(--max-logs "$MAX_LOGS")
else
    OUT="${OUT:-$REPO_ROOT/data/pretrain/v1.0-pilot}"
fi
activate_env "$NAVSIM_CONDA_ENV"
cd "$REPO_ROOT"
python -m data_pipeline.v1.openscene_index --metadata "$METADATA" --navsim-root "$NAVSIM_DEVKIT_ROOT" \
    --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
python -m data_pipeline.v1.audit --out "$OUT"
