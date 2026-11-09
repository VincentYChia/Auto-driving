#!/bin/bash
# v1-xl (selection.XL: 200k train at >= 1.0 s, a superset of v1-large, v1-pilot's 2k dev pinned) and its nested
# subset v1-xl-nd, built on Anvil with packed images (PRETRAIN_V1_PLAN.md §10) [proposed mechanics].
# The plan is committed in data/pretrain/v1.0-xl/splits/; the build lives in $TRAINING_DATA_ROOT/v1.0-xl.
#
#   bash scripts/anvil/data_pipeline/build_v1_xl.sh setup     # login node: index, splits and archive map, sha-checked
#   bash scripts/anvil/data_pipeline/build_v1_xl.sh quota     # login node: files free on scratch (the guard's view)
#   sbatch -A cis220051 -p debug --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_xl.sbatch   # repeat until "done"
#
# setup copies, never moves: index/{candidates,logs}.parquet, index/manifest.json and images/archives.json from
# FROM (default: the v1.0-large build, whose files are identical, V 2026-11-08), and splits/ from the repo checkout
# (git pull after the plan is committed). Each copy must match its expected SHA-256, then verify_v1_xl.py checks the
# splits (dev = v1-pilot's, v1-large inside v1-xl-nd inside v1-xl, counts) and leakage against the index.
# The camera archives are read in place from SRC (below); nothing here or in the sbatch deletes or moves them.
# NEVER run build_v1_large.sbatch: its archive loop deletes the downloaded archives.
#
# Sourced (by build_v1_xl.sbatch, or tests) it only defines the helper functions below.

V1XL_SRC=/anvil/scratch/x-vchia/navsim_workspace/training_data/v1.0-large/archives_hf/openscene-v1.1/openscene_sensor_trainval_camera
V1XL_METADATA=/anvil/scratch/x-vchia/navsim_workspace/training_data/v1.0-large/archives_hf/openscene-v1.1/openscene_metadata_trainval.tgz
V1XL_MIN_FREE_FILES=20000  # keep at least this many files free on scratch at every step (spec §6)
# Expected SHA-256 of the copied inputs (V: local plan machine and the Anvil v1.0-large copies, 2026-11-08).
V1XL_SHA_CANDIDATES=7483a39c6003e5c5b5a3f28d615e85caa8aa415c3e0cb2d16500bd3c820d8349
V1XL_SHA_LOGS=940212fa856a17fefd0b649e407862cb46093cce074556ffc0293c0f787a201c
V1XL_SHA_INDEX_MANIFEST=8bbc1af7ccb65ad3590b5f26deade7805e3a046a0e132859fa16a64dc051da62
V1XL_SHA_ARCHIVES_JSON=0043a6fd1dafed7fde4834cf771eea34712022e5d8893c347e04b50aadb0cce6

quota_free_files() {
    # Files free on scratch, from `myquota` (Anvil's cached report; it can lag the live count by minutes).
    # Prints nothing and fails when the scratch line cannot be read.
    local report
    report="$(myquota 2>/dev/null)" || return 1
    printf '%s\n' "$report" | awk '
        function count(s,  n, unit) {
            unit = substr(s, length(s)); n = s + 0
            if (unit == "K") return n * 1000
            if (unit == "M") return n * 1000000
            if (unit == "B" || unit == "G") return n * 1000000000
            return (s ~ /^[0-9.]+$/) ? n : -1
        }
        $1 == "scratch" { used = count($6); limit = count($7)
                          if (used >= 0 && limit > 0) { printf "%d\n", limit - used; found = 1 } }
        END { exit found ? 0 : 1 }'
}

quota_guard() {
    # Abort before step "$1" when fewer than V1XL_MIN_FREE_FILES files are free; fail closed if myquota is unreadable.
    local free
    if ! free="$(quota_free_files)" || [ -z "$free" ]; then
        echo "quota guard ($1): cannot read the scratch file count from myquota; stopping" >&2
        exit 1
    fi
    if [ "$free" -lt "$V1XL_MIN_FREE_FILES" ]; then
        echo "quota guard ($1): only $free files free on scratch (< $V1XL_MIN_FREE_FILES); stopping" >&2
        exit 1
    fi
    echo "[v1-xl] quota guard ($1): $free files free"
}

minutes_left() {
    # Minutes left in this Slurm job (9999 outside Slurm or when squeue cannot tell).
    local left d=0 h=0 m=0
    left="$(squeue -h -j "${SLURM_JOB_ID:-}" -o %L 2>/dev/null)" || { echo 9999; return; }
    [ -n "${SLURM_JOB_ID:-}" ] && [ -n "$left" ] || { echo 9999; return; }
    if [[ "$left" == *-* ]]; then d="${left%%-*}"; left="${left#*-}"; fi
    IFS=: read -r -a parts <<< "$left"
    case "${#parts[@]}" in
        3) h="${parts[0]}"; m="${parts[1]}" ;;
        2) m="${parts[0]}" ;;
        *) echo 9999; return ;;
    esac
    [[ "$d$h$m" =~ ^[0-9]+$ ]] || { echo 9999; return; }
    echo $(( 10#$d * 1440 + 10#$h * 60 + 10#$m ))
}

need_minutes() {
    # Stop cleanly (exit 0) when step "$2" needs more than the minutes left; the next submission resumes there.
    local left
    left="$(minutes_left)"
    if [ "$left" -lt "$1" ]; then
        echo "[v1-xl] RESUBMIT: $2 needs ~$1 min, $left left; submit the same command again"
        exit 0
    fi
}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

copy_checked() {
    # Copy $1 to $2 unless $2 exists; then $2 must have SHA-256 $3 (or $1's, when $3 is empty).
    local src="$1" dst="$2" want="${3:-}"
    [ -f "$src" ] || { echo "missing $src" >&2; exit 1; }
    [ -n "$want" ] || want="$(sha_of "$src")"
    [ -e "$dst" ] || cp -- "$src" "$dst"
    if [ "$(sha_of "$dst")" != "$want" ]; then
        echo "SHA-256 of $dst is not $want: fix it by hand (nothing is deleted here), then run setup again" >&2
        exit 1
    fi
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0  # sourced: functions only
fi

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-xl}"
LARGE="${LARGE:-$TRAINING_DATA_ROOT/v1.0-large}"
FROM="${FROM:-$LARGE}"
SPLITS="${SPLITS:-$REPO_ROOT/data/pretrain/v1.0-xl/splits}"
VERIFY="$REPO_ROOT/scripts/anvil/data_pipeline/verify_v1_xl.py"

case "${1:-}" in
setup)
    [ -f "$SPLITS/manifest.json" ] || { echo "no plan in $SPLITS: commit and pull data/pretrain/v1.0-xl/splits first" >&2; exit 1; }
    n_tgz="$(find "$V1XL_SRC" -maxdepth 1 -name 'openscene_sensor_trainval_camera_*.tgz' | wc -l)"
    [ "$n_tgz" -eq 200 ] || { echo "$V1XL_SRC holds $n_tgz of 200 camera archives" >&2; exit 1; }
    [ -f "$V1XL_METADATA" ] || { echo "missing $V1XL_METADATA" >&2; exit 1; }
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
    python "$VERIFY" splits --out "$OUT" --large "$LARGE"
    navsim=()  # also check against NAVSIM's own split lists when the devkit is there
    if [ -d "${NAVSIM_DEVKIT_ROOT:-}" ]; then
        navsim=(--navsim-root "$NAVSIM_DEVKIT_ROOT")
    else
        echo "[v1-xl] NAVSIM_DEVKIT_ROOT is not a folder: leakage is checked against the index's splits only"
    fi
    python "$VERIFY" leakage --out "$OUT" ${navsim[@]+"${navsim[@]}"}
    echo "ready: $OUT (next: sbatch -A cis220051 -p debug --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_xl.sbatch)"
    ;;
quota)
    echo "files free on scratch: $(quota_free_files || echo unknown) (guard stops below $V1XL_MIN_FREE_FILES)"
    ;;
*)
    echo "usage: $0 {setup|quota}" >&2; exit 1 ;;
esac
