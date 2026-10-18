#!/bin/bash
# Export navtest scenes as MVLAD-AD inputs (NAVSIM env; CPU work, run on the debug queue).
#   sbatch scripts/anvil/navsim_export.sbatch            # SPLIT=navtest by default
#SBATCH --job-name=navsim-export
#SBATCH --partition=gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "$NAVSIM_CONDA_ENV"
SPLIT="${SPLIT:-navtest}"
cd "$REPO_ROOT"
python -m navsim_tools.export_mvlad_inputs --split "$SPLIT" \
    --output "$NAVSIM_EXP_ROOT/mvlad/${SPLIT}_inputs.json"
