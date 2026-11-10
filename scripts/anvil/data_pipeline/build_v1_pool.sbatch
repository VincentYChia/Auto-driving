#!/bin/bash
# v1.0-pool steps 4-9 on Anvil (PRETRAIN_V1_PLAN.md §11), after `build_v1_pool.sh setup`: pack the CAM_F0 JPEGs of
# v1-all and dev (428,694 frames, all 200 archives) into a fresh images/packs/, cross-check the 202,000 frames that
# v1.0-xl also packed against its records, canonical tables for v1-nt, v1-xl-mix, v1-nt-mix and v1-all, QA summary,
# the four zip views and their pilot_v1 links, then verify_v1_pool.py.
# It starts only after setup's checks passed: $OUT/verify/{splits,rules,leakage}.json must say "ok": true and be
# newer than every file in splits/ and index/ (setup copies them first, so a failed setup leaves them in place).
# CPU only, on the shared partition; account and partition come from the command line:
#   sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_pool.sbatch   # until "done"
# One 4 h slot holds the whole build (~1.5 h of work, D below). A shorter one does part of it and the next submission
# resumes: each step is skipped once its output exists or its check passes, and a step that would not fit in the time
# left stops the job cleanly ("RESUBMIT"). It must end before the Nov 16 04:00 maintenance of the CPU nodes.
# Time per step (D, scaled from v1-xl's build, V 2026-11-09: fetch ~5 min with 64 processes, canonical 20.6 min for
# 203,727 frames, views 6.6 min for 202,000 samples at WORKERS=16, verify 1.7 min): fetch ~5-10 min (every archive
# is streamed whatever is kept), canonical ~43 min for 428,694 frames, QA ~2 min, views ~33 min for the four,
# verify ~6 min. The need_minutes guards below are those times plus a margin.
# Memory (--mem=200G): fetch runs SHARDS=64 processes, each holding the frame map (~0.65 GB, V locally) and the
# frames of its current archive (at most 7,447, ~1.5 GB); at most 207,387 frames are held at once (42.7 GB, D: the
# plan's frames per archive), so ~85 GB in all. Canonical holds every frame's privileged data and its columns:
# ~167 KB per frame (V: 30,000 real frames, locally), so ~75 GB for 428,694 (D). Views: v1-xl's peaked at 41.6 GiB
# with WORKERS=16 (V) and 64 workers ran out at 120 GB (V), so WORKERS stays 16; twice the samples gives ~90 GB (D).
# Safety: SRC is set explicitly (V1XL_SRC, not derived from OUT) and only read; nothing under archives_hf, v1.0-large or
# v1.0-xl is deleted, moved or written. A quota guard runs before every step (stop under 20,000 files free on scratch).
# One job at a time: two jobs on one OUT would write the same .part files (a pack truncated under its writer).
# --dependency=singleton holds a second v1-pool job of this user until the first ends, and a flock on
# $OUT/.build_v1_pool.lock refuses a second copy of this script on the same OUT (another job, or one run with bash).
#SBATCH --job-name=v1-pool
#SBATCH --dependency=singleton
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
source "$REPO_ROOT/scripts/anvil/data_pipeline/build_v1_pool.sh"  # helpers only: quota_guard, need_minutes, V1XL_*
case "${SLURM_JOB_PARTITION:-}" in
    gpu|gpu-debug|ai) echo "this build is CPU-only: submit with -A cis220051 -p shared --gpus-per-node=0" >&2
                      exit 1 ;;
esac
activate_env "${DATA_CONDA_ENV:-data}"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-pool}"
LARGE="$TRAINING_DATA_ROOT/v1.0-large"
XL="$TRAINING_DATA_ROOT/v1.0-xl"
SRC="$V1XL_SRC"  # /anvil/scratch/x-vchia/navsim_workspace/training_data/v1.0-large/archives_hf/openscene-v1.1/openscene_sensor_trainval_camera
METADATA="$V1XL_METADATA"
SHARDS="${SHARDS:-64}"
WORKERS="${WORKERS:-16}"
SETS=(v1-nt v1-xl-mix v1-nt-mix v1-all)
VIEWS=(v1-nt v1-xl-mix v1-nt-mix v1-all)  # smallest first: each view is usable once its view.json exists
VERIFY="$REPO_ROOT/scripts/anvil/data_pipeline/verify_v1_pool.py"
cd "$REPO_ROOT"

view_minutes() {
    # Minutes a view needs: ~2.0 ms per sample at WORKERS=16 (v1-xl: 6.6 min for 202,000, V), x1.5 and +2 (D).
    case "$1" in
        v1-nt) echo 6 ;;        # 82,626 samples, ~3 min
        v1-xl-mix) echo 14 ;;   # 240,034, ~8 min
        v1-nt-mix) echo 15 ;;   # 261,929, ~9 min
        *) echo 24 ;;           # v1-all: 428,694, ~14 min
    esac
}

[ -f "$OUT/splits/manifest.json" ] && [ -f "$OUT/index/candidates.parquet" ] && [ -f "$OUT/images/archives.json" ] \
    || { echo "run 'OUT=$OUT bash scripts/anvil/data_pipeline/build_v1_pool.sh setup' first" >&2; exit 1; }
setup_passed() {
    # setup's `verify $1` wrote "ok": true after the last change to splits/ and index/ (setup copies them before it
    # verifies, so a setup whose verify failed or stopped leaves the copies but no passing report)
    local report="$OUT/verify/$1.json"
    [ -f "$report" ] \
        && python -c 'import json, sys; sys.exit(json.load(open(sys.argv[1], encoding="utf-8")).get("ok") is not True)' \
            "$report" \
        && [ -z "$(find "$OUT/splits" "$OUT/index" -type f -newer "$report" -print -quit)" ]
}
for check in splits rules leakage; do
    setup_passed "$check" || { echo "setup's 'verify $check' has not passed on these files ($OUT/verify/$check.json):" \
        "run 'OUT=$OUT bash scripts/anvil/data_pipeline/build_v1_pool.sh setup' first" >&2; exit 1; }
done
for d in "$LARGE" "$XL"; do  # a fresh folder: the earlier builds are only read
    case "$(cd "$OUT" && pwd -P)/" in
        "$(cd "$d" && pwd -P)/"*) echo "OUT must not be inside $d" >&2; exit 1 ;;
    esac
done
if command -v flock > /dev/null; then
    exec 9>> "$OUT/.build_v1_pool.lock"  # one empty file, kept; the lock ends with this job
    flock -n 9 || { echo "another v1-pool build holds $OUT/.build_v1_pool.lock: wait for it to end" >&2; exit 1; }
else
    echo "[v1-pool] no flock here: relying on --dependency=singleton alone" >&2
fi

# --- step 4: fetch into images/packs/<n>.zip, SHARDS processes; resumes per archive record ----------------
rc=0; python "$VERIFY" fetched --out "$OUT" --quiet || rc=$?
if [ "$rc" -eq 2 ]; then
    quota_guard fetch
    need_minutes 12 fetch
    mkdir -p "$OUT/images/logs"
    pids=()
    for s in $(seq 0 $((SHARDS - 1))); do
        python -m data_pipeline.v1.openscene_images fetch --out "$OUT" --full --archives 200 \
            --source-dir "$SRC" --storage pack --shard "$s" --num-shards "$SHARDS" \
            >> "$OUT/images/logs/fetch_shard_$s.log" 2>&1 &
        pids+=($!)
    done
    failed=0
    for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
    [ "$failed" -eq 0 ] || grep -h "failed" "$OUT"/images/logs/fetch_shard_*.log | tail -n 20 || true
    python "$VERIFY" fetched --out "$OUT"
elif [ "$rc" -ne 0 ]; then
    python "$VERIFY" fetched --out "$OUT"  # prints the failing checks
    exit 1
fi

# --- step 5: the frames v1.0-xl also packed, against its records (seconds: reads the records only) --------------
quota_guard crosscheck
python "$VERIFY" crosscheck --out "$OUT" --xl "$XL"

# --- step 6: canonical tables for the four sets (~43 min, D) ---------------------------------------------------
rc=0; python "$VERIFY" canonical --out "$OUT" --quiet || rc=$?
if [ "$rc" -eq 2 ]; then
    quota_guard canonical
    need_minutes 50 canonical
    python -m data_pipeline.v1.canonical --out "$OUT" --sets "${SETS[@]}" --metadata "$METADATA"
    python "$VERIFY" canonical --out "$OUT"
elif [ "$rc" -ne 0 ]; then
    python "$VERIFY" canonical --out "$OUT"  # prints the failing checks
    exit 1
fi

# --- step 7: QA summary (Gate 1 renders of v1-all, which holds every other set) ----------------------------------
if [ ! -f "$OUT/qa/summary.json" ]; then
    quota_guard qa
    need_minutes 8 qa
    python -m data_pipeline.v1.qa_report --out "$OUT" --set v1-all
fi

# --- step 8: the four zip views, then the pilot_v1 links train_lora.sbatch reads ----------------------------------
for view in "${VIEWS[@]}"; do
    if [ ! -f "$OUT/views/pilot_v1/$view/view.json" ]; then
        quota_guard "view $view"
        need_minutes "$(view_minutes "$view")" "view $view"
        python -m data_pipeline.v1.pilot_view build --out "$OUT" --set "$view" --storage zip --workers "$WORKERS"
    fi
done
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

# --- step 9: every check (writes verify/all.json) ----------------------------------------------------------------
quota_guard verify
need_minutes 15 verify
extra=(--links "$TRAINING_DATA_ROOT/pilot_v1")
[ "${DEEP:-0}" != 1 ] || extra+=(--deep)  # DEEP=1: also testzip and re-hash every pack and view zip (~140 GB read, D)
navsim=()
for root in "${NAVSIM_DEVKIT_ROOT:-}" "${NAVSIM_MAIN_ROOT:-$HOME/navsim}"; do
    [ -z "$root" ] || [ ! -d "$root" ] || navsim+=(--navsim-root "$root")
done
python "$VERIFY" all --out "$OUT" --xl "$XL" --metadata "$METADATA" "${extra[@]}" ${navsim[@]+"${navsim[@]}"}
echo "done: SET=v1-nt (or v1-xl-mix, v1-nt-mix, v1-all) sbatch scripts/anvil/pilot_sft/train_lora.sbatch"
