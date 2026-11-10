#!/bin/bash
# v1.0-full: every CAM_F0 frame of the index that passes the quality rules outside the ±7 s warm-up window, extracted
# in one pass over the 200 archives (690,529 frames), with dev re-split: dev-v2 = 2,000 anchors from 40 held-out nuPlan
# val logs (13 whole drives, navtest's city mix); v1-full = every such row of the TRAIN logs and the 185 released val
# logs (675,701 train, nothing thinned) with dev-v2; v1-full-devlogs = the held-out logs' 14,828 rows, in no view
# (PRETRAIN_V1_PLAN.md §12) [proposed mechanics]. The plan is committed in data/pretrain/v1.0-full/splits/; the build
# lives in $TRAINING_DATA_ROOT/v1.0-full.
#
#   bash scripts/anvil/data_pipeline/build_v1_full.sh setup     # login node: index, splits and archive map, sha-checked
#   bash scripts/anvil/data_pipeline/build_v1_full.sh quota     # login node: files free on scratch (the guard's view)
#   sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_full.sbatch   # until "done"
#
# setup copies, never moves: index/{candidates,logs}.parquet, index/manifest.json and images/archives.json from FROM
# (default: the v1.0-large build; v1-xl's and v1.0-pool's copies are the same files), and splits/ from the repo
# checkout (git pull after the plan is committed). Each copy must match its expected SHA-256 (build_v1_xl.sh's
# V1XL_SHA_*, or the source's own for splits/). Then verify_v1_full.py checks the splits (both sets' counts and
# SHA-256, dev-v2, dev-v1 = the old pinned dev, the held-out log list, v1.0-pool's v1-all inside v1-full), the rules
# (the held-out logs, the sets and dev-v2's draw recomputed from the index) and leakage, against the index and, when
# the folders exist, both NAVSIM devkits: NAVSIM_DEVKIT_ROOT (v1.1, from config.local.sh) and NAVSIM_MAIN_ROOT
# (default ~/navsim, main). Each writes $OUT/verify/<check>.json; the sbatch starts only when all three say "ok": true
# and are newer than every file in splits/ and index/, so after a failed check fix the cause and run setup again.
# The camera archives are read in place from V1XL_SRC (build_v1_xl.sh); nothing here or in the sbatch deletes or moves
# them. NEVER run build_v1_large.sbatch: its archive loop deletes the downloaded archives.
#
# Sourced (by build_v1_full.sbatch, or tests) it only defines build_v1_xl.sh's helpers and V1XL_* constants.

BUILD_TAG=v1-full  # the prefix of the shared helpers' messages
# shellcheck source=build_v1_xl.sh
source "$(dirname "${BASH_SOURCE[0]}")/build_v1_xl.sh"  # helpers only: quota_guard, need_minutes, copy_checked, V1XL_*

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0  # sourced: functions only
fi

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-full}"
LARGE="${LARGE:-$TRAINING_DATA_ROOT/v1.0-large}"
XL="${XL:-$TRAINING_DATA_ROOT/v1.0-xl}"
POOL="${POOL:-$TRAINING_DATA_ROOT/v1.0-pool}"
FROM="${FROM:-$LARGE}"
SPLITS="${SPLITS:-$REPO_ROOT/data/pretrain/v1.0-full/splits}"
VERIFY="$REPO_ROOT/scripts/anvil/data_pipeline/verify_v1_full.py"

case "${1:-}" in
setup)
    [ -f "$SPLITS/manifest.json" ] || { echo "no plan in $SPLITS: commit and pull data/pretrain/v1.0-full/splits first" >&2; exit 1; }
    n_tgz="$(find "$V1XL_SRC" -maxdepth 1 -name 'openscene_sensor_trainval_camera_*.tgz' | wc -l)"
    [ "$n_tgz" -eq 200 ] || { echo "$V1XL_SRC holds $n_tgz of 200 camera archives" >&2; exit 1; }
    [ -f "$V1XL_METADATA" ] || { echo "missing $V1XL_METADATA" >&2; exit 1; }
    [ -f "$POOL/views/pilot_v1/v1-all/view.json" ] || { echo "no v1-all view in $POOL: the views are checked against it" >&2; exit 1; }
    [ -d "$POOL/images/archives" ] || { echo "no fetch records in $POOL/images/archives: the cross-check reads them" >&2; exit 1; }
    [ -d "$LARGE" ] || { echo "missing $LARGE" >&2; exit 1; }
    out_real="$(cd "$(dirname "$OUT")" && pwd -P)/$(basename "$OUT")"
    for d in "$LARGE" "$XL" "$POOL"; do  # a fresh folder: the earlier builds are only read
        [ -d "$d" ] || continue
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
            echo "[v1-full] NAVSIM devkit '$root' is not a folder: not checked against it"
        fi
    done
    python "$VERIFY" splits --out "$OUT" --pool "$POOL"
    # --metadata (required): places the warm-up frames that are not index rows, independently of the plan (~1 min, D)
    python "$VERIFY" rules --out "$OUT" --metadata "$V1XL_METADATA"
    python "$VERIFY" leakage --out "$OUT" --metadata "$V1XL_METADATA" ${navsim[@]+"${navsim[@]}"}
    echo "ready: $OUT (next: sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_full.sbatch)"
    ;;
quota)
    echo "files free on scratch: $(quota_free_files || echo unknown) (guard stops below $V1XL_MIN_FREE_FILES)"
    ;;
*)
    echo "usage: $0 {setup|quota}" >&2; exit 1 ;;
esac
