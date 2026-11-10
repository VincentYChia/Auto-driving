#!/bin/bash
# v1.0-full steps 4-9 on Anvil (PRETRAIN_V1_PLAN.md §12), after `build_v1_full.sh setup`: pack the CAM_F0 JPEGs of
# v1-full's train, dev-v2 and the held-out rows (690,529 frames, all 200 archives) into a fresh images/packs/,
# cross-check the 428,645 frames that v1.0-pool also packed against its records, canonical tables for v1-full and
# v1-full-devlogs, QA summary of v1-full, the v1-full zip views in both prompt versions (pilot_v1.1 and pilot_v1.2)
# and their links pilot_v1/v1-full and pilot_v1/v1-full-p12, then verify_v1_full.py.
# It starts only after setup's checks passed: $OUT/verify/{splits,rules,leakage}.json must say "ok": true and be
# newer than every file in splits/ and index/ (setup copies them first, so a failed setup leaves them in place).
# CPU only, on the shared partition; account and partition come from the command line:
#   sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_full.sbatch   # until "done"
# One 4 h slot should hold the whole build (~3.0 h of work, D below). A shorter one does part of it and the next
# submission resumes: each step is skipped once its output exists or its check passes, and a step that would not fit
# in the time left stops the job cleanly ("RESUBMIT"). It must end before the Nov 16 04:00 maintenance of the CPU nodes.
# Time per step (D, scaled from v1.0-pool's build: fetch ~5 min with 64 processes, canonical ~50 min for 428,694
# frames, views 6-24 min each at WORKERS=16; v1-xl's verify 1.7 min, V): fetch ~5-10 min (every archive is streamed
# whatever is kept), canonical ~81 min for 690,529 frames, QA ~3 min, each view ~38 min for 677,701 samples (two
# views), verify ~10-15 min. The need_minutes guards below are those times plus a margin.
# Memory (--mem=200G): fetch runs SHARDS=64 processes, each holding the frame map (~0.7 GB peak, V locally for the
# 690,529 frames) and the frames of its current archive (at most 11,650 in archive 124, ~2.4 GB at 205.7 KB a frame);
# the 64 largest archives hold 352,482 frames (72.5 GB, D: the plan's frames per archive), so ~120 GB in all.
# Canonical holds every frame's privileged data and its columns: ~167 KB per frame (V: 30,000 real frames, locally),
# so ~115 GB for 690,529 (D). Views: v1.0-pool's peaked at ~44 GB with WORKERS=16; 1.6x the frames gives ~71 GB (D);
# 64 workers ran out at 120 GB on v1-xl (V), so WORKERS stays 16.
# Safety: SRC is set explicitly (V1XL_SRC, not derived from OUT) and only read; nothing under archives_hf, v1.0-large,
# v1.0-xl or v1.0-pool is deleted, moved or written. A quota guard runs before every step (stop under 20,000 files
# free on scratch). One job at a time: two jobs on one OUT would write the same .part files (a pack truncated under its
# writer). --dependency=singleton holds a second v1-full job of this user until the first ends, and a flock on
# $OUT/.build_v1_full.lock refuses a second copy of this script on the same OUT (another job, or one run with bash).
#SBATCH --job-name=v1-full
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
source "$REPO_ROOT/scripts/anvil/data_pipeline/build_v1_full.sh"  # helpers only: quota_guard, need_minutes, V1XL_*
case "${SLURM_JOB_PARTITION:-}" in
    gpu|gpu-debug|ai) echo "this build is CPU-only: submit with -A cis220051 -p shared --gpus-per-node=0" >&2
                      exit 1 ;;
esac
activate_env "${DATA_CONDA_ENV:-data}"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-full}"
LARGE="$TRAINING_DATA_ROOT/v1.0-large"
XL="$TRAINING_DATA_ROOT/v1.0-xl"
POOL="$TRAINING_DATA_ROOT/v1.0-pool"
SRC="$V1XL_SRC"  # /anvil/scratch/x-vchia/navsim_workspace/training_data/v1.0-large/archives_hf/openscene-v1.1/openscene_sensor_trainval_camera
METADATA="$V1XL_METADATA"
SHARDS="${SHARDS:-64}"
WORKERS="${WORKERS:-16}"
SETS=(v1-full v1-full-devlogs)
VIEW=v1-full  # one set, two prompt versions; v1-full-devlogs gets no view (its rows are dev-v2's held-out logs)
VERSIONS=(pilot_v1.1 pilot_v1.2)
VERIFY="$REPO_ROOT/scripts/anvil/data_pipeline/verify_v1_full.py"
cd "$REPO_ROOT"

view_root() {  # pilot_view.VIEW_ROOTS
    case "$1" in
        pilot_v1.1) echo pilot_v1 ;;
        pilot_v1.2) echo pilot_v1.2 ;;
        *) echo "unknown prompt version $1" >&2; exit 1 ;;
    esac
}
link_suffix() {  # pilot_v1/<set> for pilot_v1.1, pilot_v1/<set>-p12 for pilot_v1.2 (build_views_v12.sbatch)
    case "$1" in
        pilot_v1.1) echo "" ;;
        pilot_v1.2) echo "-p12" ;;
    esac
}
VIEW_MINUTES=60  # each view: ~3.4 ms per sample at WORKERS=16 (v1.0-pool's 24 min for 428,694), 677,701 samples ~38 min, x1.5 + 2 (D)

[ -f "$OUT/splits/manifest.json" ] && [ -f "$OUT/index/candidates.parquet" ] && [ -f "$OUT/images/archives.json" ] \
    || { echo "run 'OUT=$OUT bash scripts/anvil/data_pipeline/build_v1_full.sh setup' first" >&2; exit 1; }
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
        "run 'OUT=$OUT bash scripts/anvil/data_pipeline/build_v1_full.sh setup' first" >&2; exit 1; }
done
for d in "$LARGE" "$XL" "$POOL"; do  # a fresh folder: the earlier builds are only read
    [ -d "$d" ] || continue
    case "$(cd "$OUT" && pwd -P)/" in
        "$(cd "$d" && pwd -P)/"*) echo "OUT must not be inside $d" >&2; exit 1 ;;
    esac
done
if command -v flock > /dev/null; then
    exec 9>> "$OUT/.build_v1_full.lock"  # one empty file, kept; the lock ends with this job
    flock -n 9 || { echo "another v1-full build holds $OUT/.build_v1_full.lock: wait for it to end" >&2; exit 1; }
else
    echo "[v1-full] no flock here: relying on --dependency=singleton alone" >&2
fi

# --- step 4: fetch into images/packs/<n>.zip, SHARDS processes; resumes per archive record ----------------
rc=0; python "$VERIFY" fetched --out "$OUT" --quiet || rc=$?
if [ "$rc" -eq 2 ]; then
    quota_guard fetch
    need_minutes 15 fetch
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

# --- step 5: the frames v1.0-pool also packed, against its records (seconds: reads the records only) ------------
quota_guard crosscheck
python "$VERIFY" crosscheck --out "$OUT" --pool "$POOL"

# --- step 6: canonical tables for the two sets (~81 min, D) -------------------------------------------------------
rc=0; python "$VERIFY" canonical --out "$OUT" --quiet || rc=$?
if [ "$rc" -eq 2 ]; then
    quota_guard canonical
    need_minutes 110 canonical
    python -m data_pipeline.v1.canonical --out "$OUT" --sets "${SETS[@]}" --metadata "$METADATA"
    python "$VERIFY" canonical --out "$OUT"
elif [ "$rc" -ne 0 ]; then
    python "$VERIFY" canonical --out "$OUT"  # prints the failing checks
    exit 1
fi

# --- step 7: QA summary (Gate 1 renders of v1-full: its train and dev-v2) ------------------------------------------
if [ ! -f "$OUT/qa/summary.json" ]; then
    quota_guard qa
    need_minutes 8 qa
    python -m data_pipeline.v1.qa_report --out "$OUT" --set "$VIEW"
fi

# --- step 8: the v1-full zip view in both prompt versions, then the pilot_v1 links train_lora.sbatch reads --------
for version in "${VERSIONS[@]}"; do
    if [ ! -f "$OUT/views/$(view_root "$version")/$VIEW/view.json" ]; then
        quota_guard "view $version"
        need_minutes "$VIEW_MINUTES" "view $version"
        python -m data_pipeline.v1.pilot_view build --out "$OUT" --set "$VIEW" --storage zip --workers "$WORKERS" \
            --prompt-version "$version"
    fi
done
mkdir -p "$TRAINING_DATA_ROOT/pilot_v1"
for version in "${VERSIONS[@]}"; do
    link="$TRAINING_DATA_ROOT/pilot_v1/$VIEW$(link_suffix "$version")" target="$OUT/views/$(view_root "$version")/$VIEW"
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
need_minutes 25 verify
extra=(--links "$TRAINING_DATA_ROOT/pilot_v1")
[ "${DEEP:-0}" != 1 ] || extra+=(--deep)  # DEEP=1: also testzip and re-hash every pack and view zip (~210 GB read, D)
navsim=()
for root in "${NAVSIM_DEVKIT_ROOT:-}" "${NAVSIM_MAIN_ROOT:-$HOME/navsim}"; do
    [ -z "$root" ] || [ ! -d "$root" ] || navsim+=(--navsim-root "$root")
done
python "$VERIFY" all --out "$OUT" --pool "$POOL" --metadata "$METADATA" "${extra[@]}" ${navsim[@]+"${navsim[@]}"}
echo "done: SET=v1-full (or v1-full-p12) sbatch scripts/anvil/pilot_sft/train_lora.sbatch"
