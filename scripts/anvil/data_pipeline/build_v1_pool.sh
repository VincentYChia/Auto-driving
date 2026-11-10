#!/bin/bash
# v1.0-pool: the whole 0.5 s train pool extracted once (v1-all: 426,694 train) and four views of it with v1-xl's
# pinned 2,000 dev: v1-nt (navtrain's rows, 80,626), v1-xl-mix (238,034), v1-nt-mix (259,929) and v1-all, built on
# Anvil with packed images in a fresh folder (PRETRAIN_V1_PLAN.md §11) [proposed mechanics]. The plan is committed in
# data/pretrain/v1.0-pool/splits/; the build lives in $TRAINING_DATA_ROOT/v1.0-pool.
#
#   bash scripts/anvil/data_pipeline/build_v1_pool.sh setup     # login node: index, splits and archive map, sha-checked
#   bash scripts/anvil/data_pipeline/build_v1_pool.sh quota     # login node: files free on scratch (the guard's view)
#   sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_pool.sbatch   # until "done"
#
# setup copies, never moves: index/{candidates,logs}.parquet, index/manifest.json and images/archives.json from FROM
# (default: the v1.0-large build; v1-xl's copies are the same files, V 2026-11-09), and splits/ from the repo checkout
# (git pull after the plan is committed). Each copy must match its expected SHA-256 (build_v1_xl.sh's V1XL_SHA_*,
# or the source's own for splits/). Then verify_v1_pool.py checks the splits (the four sets' counts and SHA-256, dev =
# v1-xl's pinned 2,000 in all four, the navtrain and warm-up lists, the base v1-xl-nd), the rules (the pool, v1-nt
# and v1-xl-mix recomputed from the index) and leakage, against the index and, when the folders exist, both NAVSIM
# devkits: NAVSIM_DEVKIT_ROOT (v1.1, from config.local.sh) and NAVSIM_MAIN_ROOT (default ~/navsim, main).
# Each writes $OUT/verify/<check>.json; the sbatch starts only when all three say "ok": true and are newer than
# every file in splits/ and index/, so after a failed check fix the cause and run setup again.
# The camera archives are read in place from V1XL_SRC (build_v1_xl.sh); nothing here or in the sbatch deletes or moves
# them. NEVER run build_v1_large.sbatch: its archive loop deletes the downloaded archives.
#
# Sourced (by build_v1_pool.sbatch, or tests) it only defines build_v1_xl.sh's helpers and V1XL_* constants.

BUILD_TAG=v1-pool  # the prefix of the shared helpers' messages
# shellcheck source=build_v1_xl.sh
source "$(dirname "${BASH_SOURCE[0]}")/build_v1_xl.sh"  # helpers only: quota_guard, need_minutes, copy_checked, V1XL_*

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0  # sourced: functions only
fi

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-pool}"
LARGE="${LARGE:-$TRAINING_DATA_ROOT/v1.0-large}"
XL="${XL:-$TRAINING_DATA_ROOT/v1.0-xl}"
FROM="${FROM:-$LARGE}"
SPLITS="${SPLITS:-$REPO_ROOT/data/pretrain/v1.0-pool/splits}"
VERIFY="$REPO_ROOT/scripts/anvil/data_pipeline/verify_v1_pool.py"

case "${1:-}" in
setup)
    [ -f "$SPLITS/manifest.json" ] || { echo "no plan in $SPLITS: commit and pull data/pretrain/v1.0-pool/splits first" >&2; exit 1; }
    n_tgz="$(find "$V1XL_SRC" -maxdepth 1 -name 'openscene_sensor_trainval_camera_*.tgz' | wc -l)"
    [ "$n_tgz" -eq 200 ] || { echo "$V1XL_SRC holds $n_tgz of 200 camera archives" >&2; exit 1; }
    [ -f "$V1XL_METADATA" ] || { echo "missing $V1XL_METADATA" >&2; exit 1; }
    [ -f "$XL/views/pilot_v1/v1-xl/view.json" ] || { echo "no v1-xl view in $XL: the views are checked against it" >&2; exit 1; }
    [ -d "$LARGE" ] || { echo "missing $LARGE" >&2; exit 1; }
    out_real="$(cd "$(dirname "$OUT")" && pwd -P)/$(basename "$OUT")"
    for d in "$LARGE" "$XL"; do  # a fresh folder: the earlier builds are only read
        case "$out_real/" in "$(cd "$d" && pwd -P)/"*) echo "OUT must not be inside $d" >&2; exit 1 ;; esac
    done
    quota_guard setup
    mkdir -p "$OUT/index" "$OUT/splits" "$OUT/images"
    copy_checked "$FROM/index/candidates.parquet" "$OUT/index/candidates.parquet" "$V1XL_SHA_CANDIDATES"
    copy_checked "$FROM/index/logs.parquet" "$OUT/index/logs.parquet" "$V1XL_SHA_LOGS"
    copy_checked "$FROM/index/manifest.json" "$OUT/index/manifest.json" "$V1XL_SHA_INDEX_MANIFEST"
    copy_checked "$FROM/images/archives.json" "$OUT/images/archives.json" "$V1XL_SHA_ARCHIVES_JSON"
    for f in "$SPLITS"/*; do
        copy_checked "$f" "$OUT/splits/$(basename "$f")" ""
    done
    activate_env "${DATA_CONDA_ENV:-data}"
    cd "$REPO_ROOT"
    navsim=()  # also check against NAVSIM's own lists, in every devkit that is here
    for root in "${NAVSIM_DEVKIT_ROOT:-}" "${NAVSIM_MAIN_ROOT:-$HOME/navsim}"; do
        if [ -n "$root" ] && [ -d "$root" ]; then
            navsim+=(--navsim-root "$root")
        else
            echo "[v1-pool] NAVSIM devkit '$root' is not a folder: not checked against it"
        fi
    done
    python "$VERIFY" splits --out "$OUT" --xl "$XL"
    # --metadata (required): places the warm-up frames that are not index rows, independently of the plan (~1 min, D)
    python "$VERIFY" rules --out "$OUT" --metadata "$V1XL_METADATA"
    python "$VERIFY" leakage --out "$OUT" --metadata "$V1XL_METADATA" ${navsim[@]+"${navsim[@]}"}
    echo "ready: $OUT (next: sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_pool.sbatch)"
    ;;
quota)
    echo "files free on scratch: $(quota_free_files || echo unknown) (guard stops below $V1XL_MIN_FREE_FILES)"
    ;;
*)
    echo "usage: $0 {setup|quota}" >&2; exit 1 ;;
esac
