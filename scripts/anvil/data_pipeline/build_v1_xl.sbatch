#!/bin/bash
# v1-xl steps 4-9 on Anvil (PRETRAIN_V1_PLAN.md §10), after `build_v1_xl.sh setup`: pack the selected CAM_F0 JPEGs
# of all 200 archives (202,000 frames), cross-check the 102,000 shared with v1-large, canonical tables for
# v1-xl, v1-xl-nd and v1-large, QA summary, the two zip views and their pilot_v1 links, then verify_v1_xl.py.
# CPU only. Account and partition come from the command line; a 30-minute debug slot does part of the work and
# the next submission resumes (each step is skipped once its output exists or its check passes):
#   sbatch -A cis220051 -p debug --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_xl.sbatch
#   sbatch -A cis220051 -p shared --gpus-per-node=0 --time=03:00:00 scripts/anvil/data_pipeline/build_v1_xl.sbatch
# Smoke test first (spec §6 step 3): its own folder, the first ARCHIVES archives, plus a pack-reproducibility check:
#   OUT=$TRAINING_DATA_ROOT/v1.0-xl-smoke bash scripts/anvil/data_pipeline/build_v1_xl.sh setup
#   OUT=$TRAINING_DATA_ROOT/v1.0-xl-smoke ARCHIVES=2 sbatch -A cis220051 -p debug --gpus-per-node=0 \
#       scripts/anvil/data_pipeline/build_v1_xl.sbatch
# Safety: SRC is set explicitly (not derived from OUT) and only read; nothing under archives_hf or v1.0-large is
# deleted, moved or written. A quota guard runs before every step (stop under 20,000 files free on scratch).
# One job at a time: two jobs on one OUT would write the same .part files (a pack truncated under its writer).
# --dependency=singleton holds a second v1-xl job of this user until the first ends, and a flock on
# $OUT/.build_v1_xl.lock refuses a second copy of this script on the same OUT (another job, or one run with bash).
#SBATCH --job-name=v1-xl
#SBATCH --dependency=singleton
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
source "$REPO_ROOT/scripts/anvil/data_pipeline/build_v1_xl.sh"  # helpers only: quota_guard, need_minutes, V1XL_*
case "${SLURM_JOB_PARTITION:-}" in
    gpu|gpu-debug|ai) echo "this build is CPU-only: submit with -A cis220051 -p debug (or shared) --gpus-per-node=0" >&2
                      exit 1 ;;
esac
activate_env "${DATA_CONDA_ENV:-data}"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-xl}"
LARGE="$TRAINING_DATA_ROOT/v1.0-large"
SRC="$V1XL_SRC"  # /anvil/scratch/x-vchia/navsim_workspace/training_data/v1.0-large/archives_hf/openscene-v1.1/openscene_sensor_trainval_camera
METADATA="$V1XL_METADATA"
ARCHIVES="${ARCHIVES:-200}"
SHARDS="${SHARDS:-96}"
WORKERS="${WORKERS:-64}"
SETS=(v1-xl v1-xl-nd v1-large)
VIEWS=(v1-xl v1-xl-nd)
VERIFY="$REPO_ROOT/scripts/anvil/data_pipeline/verify_v1_xl.py"
SMOKE=()
cd "$REPO_ROOT"

[ -f "$OUT/splits/manifest.json" ] && [ -f "$OUT/index/candidates.parquet" ] && [ -f "$OUT/images/archives.json" ] \
    || { echo "run 'OUT=$OUT bash scripts/anvil/data_pipeline/build_v1_xl.sh setup' first" >&2; exit 1; }
case "$(cd "$OUT" && pwd -P)/" in
    "$(cd "$LARGE" && pwd -P)/"*) echo "OUT must not be inside $LARGE" >&2; exit 1 ;;
esac
if [ "$ARCHIVES" -lt 200 ]; then
    [ "$(basename "$OUT")" != v1.0-xl ] || { echo "a smoke run (ARCHIVES=$ARCHIVES) needs its own OUT folder" >&2; exit 1; }
    SMOKE=(--smoke)
    [ "$SHARDS" -le "$ARCHIVES" ] || SHARDS="$ARCHIVES"
fi
if command -v flock > /dev/null; then
    exec 9>> "$OUT/.build_v1_xl.lock"  # one empty file, kept; the lock ends with this job
    flock -n 9 || { echo "another v1-xl build holds $OUT/.build_v1_xl.lock: wait for it to end" >&2; exit 1; }
else
    echo "[v1-xl] no flock here: relying on --dependency=singleton alone" >&2
fi

# --- step 4: fetch into images/packs/<n>.zip, 96 processes; resumes per archive record --------------------
rc=0; python "$VERIFY" fetched --out "$OUT" --archives "$ARCHIVES" --quiet || rc=$?
if [ "$rc" -eq 2 ]; then
    quota_guard fetch
    need_minutes 5 fetch
    mkdir -p "$OUT/images/logs"
    pids=()
    for s in $(seq 0 $((SHARDS - 1))); do
        python -m data_pipeline.v1.openscene_images fetch --out "$OUT" --full --archives "$ARCHIVES" \
            --source-dir "$SRC" --storage pack --shard "$s" --num-shards "$SHARDS" \
            >> "$OUT/images/logs/fetch_shard_$s.log" 2>&1 &
        pids+=($!)
    done
    failed=0
    for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
    [ "$failed" -eq 0 ] || grep -h "failed" "$OUT"/images/logs/fetch_shard_*.log | tail -n 20 || true
    python "$VERIFY" fetched --out "$OUT" --archives "$ARCHIVES"
elif [ "$rc" -ne 0 ]; then
    python "$VERIFY" fetched --out "$OUT" --archives "$ARCHIVES"  # prints the failing checks
    exit 1
fi

# --- smoke only: a second pack of the first archive must have the same bytes (fetch skips it once packed) ---
if [ "${#SMOKE[@]}" -gt 0 ]; then
    quota_guard repro
    REPRO="$OUT/repro"
    mkdir -p "$REPRO/images"
    [ -e "$REPRO/index" ] || ln -s "$OUT/index" "$REPRO/index"
    [ -e "$REPRO/splits" ] || ln -s "$OUT/splits" "$REPRO/splits"
    [ -e "$REPRO/images/archives.json" ] || ln -s "$OUT/images/archives.json" "$REPRO/images/archives.json"
    python -m data_pipeline.v1.openscene_images fetch --out "$REPRO" --full --archives 1 \
        --source-dir "$SRC" --storage pack
    python "$VERIFY" repro --out "$OUT" --other "$REPRO"
fi

# --- step 5: the shared frames' SHA-256 against v1.0-large's records (seconds: reads the records only) -----
quota_guard crosscheck
python "$VERIFY" crosscheck --out "$OUT" --large "$LARGE" ${SMOKE[@]+"${SMOKE[@]}"}

# --- step 6: canonical tables for the three sets (~11 min, D) ----------------------------------------------
rc=0; python "$VERIFY" canonical --out "$OUT" --quiet ${SMOKE[@]+"${SMOKE[@]}"} || rc=$?
if [ "$rc" -eq 2 ]; then
    quota_guard canonical
    need_minutes 15 canonical
    python -m data_pipeline.v1.canonical --out "$OUT" --sets "${SETS[@]}" --metadata "$METADATA"
    python "$VERIFY" canonical --out "$OUT" ${SMOKE[@]+"${SMOKE[@]}"}
elif [ "$rc" -ne 0 ]; then
    python "$VERIFY" canonical --out "$OUT" ${SMOKE[@]+"${SMOKE[@]}"}  # prints the failing checks
    exit 1
fi

# --- step 7: QA summary (Gate 1 renders of v1-xl) -----------------------------------------------------------
if [ ! -f "$OUT/qa/summary.json" ]; then
    quota_guard qa
    need_minutes 5 qa
    python -m data_pipeline.v1.qa_report --out "$OUT" --set v1-xl
fi

# --- step 8: the two zip views, then the pilot_v1 links train_lora.sbatch reads -----------------------------
for view in "${VIEWS[@]}"; do
    if [ ! -f "$OUT/views/pilot_v1/$view/view.json" ]; then
        quota_guard "view $view"
        need_minutes 12 "view $view"
        python -m data_pipeline.v1.pilot_view build --out "$OUT" --set "$view" --storage zip --workers "$WORKERS"
    fi
done
if [ "${#SMOKE[@]}" -eq 0 ]; then
    mkdir -p "$TRAINING_DATA_ROOT/pilot_v1"
    for view in "${VIEWS[@]}"; do
        link="$TRAINING_DATA_ROOT/pilot_v1/$view" target="$OUT/views/pilot_v1/$view"
        if [ -L "$link" ]; then
            [ "$(readlink "$link")" = "$target" ] || { echo "$link points to $(readlink "$link"), not $target: left as is" >&2; exit 1; }
        elif [ -e "$link" ]; then
            echo "$link exists and is not a link: left as is" >&2; exit 1
        else
            ln -s "$target" "$link"
        fi
    done
fi

# --- step 9: every check of spec §8 (writes verify/all.json) ------------------------------------------------
quota_guard verify
need_minutes 10 verify
extra=(--archives "$ARCHIVES" ${SMOKE[@]+"${SMOKE[@]}"})
[ "${#SMOKE[@]}" -gt 0 ] || extra+=(--links "$TRAINING_DATA_ROOT/pilot_v1")
[ "${DEEP:-0}" != 1 ] || extra+=(--deep)  # DEEP=1: also testzip and re-hash every pack and view zip (~55 GB read)
python "$VERIFY" all --out "$OUT" --large "$LARGE" ${extra[@]+"${extra[@]}"}
echo "done: SET=v1-xl (or v1-xl-nd) sbatch scripts/anvil/pilot_sft/train_lora.sbatch"
