#!/bin/bash
# pilot_v1.2 views [proposed 2026-11-09]: the same samples and images, with the past given as 4 poses with heading
# (-1.5, -1.0, -0.5 and 0 s; ReCogDrive's VLM history, in our prompt's wording; data_pipeline/v1/pilot_view.py).
# Written next to the pilot_v1.1 views (<build>/views/pilot_v1.2/<set>), from builds whose canonical step is done, and
# linked as $TRAINING_DATA_ROOT/pilot_v1/<set>-p12, so training takes SET=<set>-p12. Reads frames and packs only:
# nothing is deleted or moved. A view whose view.json exists is skipped, so resubmitting is safe.
#
#   sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_views_v12.sbatch
#   VIEWS="v1.0-pool:v1-nt" sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_views_v12.sbatch
#SBATCH --job-name=views-v12
#SBATCH --dependency=singleton
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${DATA_CONDA_ENV:-data}"
cd "$REPO_ROOT"
# build:set pairs; the default is the four v1.0-pool sets plus v1-large and v1-xl-nd (for A/B against v1.1 runs)
VIEWS="${VIEWS:-v1.0-pool:v1-nt v1.0-pool:v1-nt-mix v1.0-pool:v1-xl-mix v1.0-pool:v1-all v1.0-large:v1-large v1.0-xl:v1-xl-nd}"
for spec in $VIEWS; do
    build="${spec%%:*}"
    set_name="${spec#*:}"
    out="$TRAINING_DATA_ROOT/$build"
    dir="$out/views/pilot_v1.2/$set_name"
    if [ -f "$dir/view.json" ]; then
        echo "[views-v12] $set_name: exists, skipped"
    else
        [ -f "$out/frames.parquet" ] || { echo "[views-v12] $out/frames.parquet missing (canonical step not done)" >&2; exit 1; }
        echo "[views-v12] $set_name from $build $(date +%T)"
        python -m data_pipeline.v1.pilot_view build --out "$out" --set "$set_name" --storage zip \
            --workers "${WORKERS:-16}" --prompt-version pilot_v1.2
    fi
    ln -sfn "$dir" "$TRAINING_DATA_ROOT/pilot_v1/$set_name-p12"
done
echo "[views-v12] done"
