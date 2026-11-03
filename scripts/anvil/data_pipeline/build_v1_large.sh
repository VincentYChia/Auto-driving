#!/bin/bash
# v1-large (selection.LARGE: 100k train + v1-pilot's 2k dev) built on Anvil [proposed 2026-11-03].
# Its plan is committed in data/pretrain/v1.0-large/splits/; the build lives in $TRAINING_DATA_ROOT/v1.0-large.
#
#   bash scripts/anvil/data_pipeline/build_v1_large.sh setup      # login node: plan files into the build folder
#   bash scripts/anvil/data_pipeline/build_v1_large.sh download   # login node: metadata + 200 camera archives, 1.24 TB
#   sbatch -A <account> scripts/anvil/data_pipeline/build_v1_large.sbatch   # once download says ALL DONE
#
# setup needs index/ (candidates.parquet, logs.parquet) and images/archives.json from the machine that ran
# `openscene_images plan --set v1-large`, copied to the build folder first:
#   scp -r data/pretrain/v1.0-large/index data/pretrain/v1.0-large/images anvil:<build folder>/
# download uses hf_xet (the `hf` CLI in the pilot env): ~20-100 MB/s here, against ~1-3 MB/s for wget (V, 2026-11-03).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-large}"
ARCHIVES="$OUT/archives_hf"

case "${1:-}" in
setup)
    for f in index/candidates.parquet index/logs.parquet images/archives.json; do
        [ -f "$OUT/$f" ] || { echo "missing $OUT/$f: copy it from the planning machine (see the header)" >&2; exit 1; }
    done
    mkdir -p "$OUT/splits"
    cp "$REPO_ROOT"/data/pretrain/v1.0-large/splits/* "$OUT/splits/"
    echo "ready: $OUT"
    ;;
download)
    activate_env "${PILOT_CONDA_ENV:-pilot}"
    export HF_XET_HIGH_PERFORMANCE=1
    hf download OpenDriveLab/OpenScene openscene-v1.1/openscene_metadata_trainval.tgz \
        --repo-type dataset --local-dir "$ARCHIVES"
    hf download OpenDriveLab/OpenScene --include "openscene-v1.1/openscene_sensor_trainval_camera/*.tgz" \
        --repo-type dataset --local-dir "$ARCHIVES" --max-workers 8
    echo "ALL DONE $(date)"
    ;;
*)
    echo "usage: $0 {setup|download}" >&2; exit 1 ;;
esac
