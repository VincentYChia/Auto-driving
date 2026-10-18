#!/bin/bash
# NAVSIM metric caching and PDM scoring (CPU-bound; uses the cores of one GPU-node slot).
#
#   STEP=cache sbatch scripts/anvil/navsim_score.sbatch     # once per split (needs maps + logs)
#   STEP=cv    sbatch scripts/anvil/navsim_score.sbatch     # sanity check: NAVSIM's constant-velocity agent
#   STEP=mvlad sbatch scripts/anvil/navsim_score.sbatch     # score MVLAD-AD predictions
#
# NAVSIM_VERSION=v1 (default) gives PDMS with the v1.1 devkit; v2 gives EPDMS with the main branch.
#SBATCH --job-name=navsim-score
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "$NAVSIM_CONDA_ENV"
SPLIT="${SPLIT:-navtest}"
STEP="${STEP:?set STEP=cache|cv|mvlad}"
NAVSIM_VERSION="${NAVSIM_VERSION:-v1}"
CACHE="$NAVSIM_EXP_ROOT/metric_cache_${SPLIT}_${NAVSIM_VERSION}"
THREADS="worker.threads_per_node=${SLURM_CPUS_PER_TASK:-8}"
SCRIPTS="$NAVSIM_DEVKIT_ROOT/navsim/planning/script"
cd "$REPO_ROOT"

if [ "$NAVSIM_VERSION" = "v1" ]; then CACHE_ARG="cache.cache_path=$CACHE"; else CACHE_ARG="metric_cache_path=$CACHE"; fi

case "$STEP" in
cache)
    python "$SCRIPTS/run_metric_caching.py" train_test_split="$SPLIT" "$CACHE_ARG" "$THREADS"
    ;;
cv)
    python "$SCRIPTS/run_pdm_score.py" train_test_split="$SPLIT" metric_cache_path="$CACHE" "$THREADS" \
        experiment_name="cv_${SPLIT}_${NAVSIM_VERSION}"
    ;;
mvlad)
    python "$SCRIPTS/run_pdm_score.py" train_test_split="$SPLIT" metric_cache_path="$CACHE" "$THREADS" \
        experiment_name="mvlad_${SPLIT}_${NAVSIM_VERSION}" \
        agent._target_=navsim_tools.precomputed_agent.PrecomputedTrajectoryAgent \
        +agent.predictions_path="$NAVSIM_EXP_ROOT/mvlad/${SPLIT}_predictions"
    ;;
*)
    echo "unknown STEP=$STEP" >&2; exit 1 ;;
esac
echo "results: $NAVSIM_EXP_ROOT/<experiment_name>/<timestamp>.csv"
