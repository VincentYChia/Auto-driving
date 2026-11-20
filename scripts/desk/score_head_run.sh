#!/bin/bash
# NAVSIM v1.1 PDM scores, on the lab 4090 desktop's CPUs, for every path folder a head run wrote
# (pilot_sft/heads/predict_head.py: <run>/navtest_<weights>/steps<S>/{single,sample_<j>}).
# The same scorer command as scripts/anvil/navsim_eval/navsim_score.sbatch STEP=pilot, with the navsim env that
# mirrors Anvil's (~/setup_navsim_env.sh) and a copy of Anvil's navtest metric cache; checked 2026-11-19 by
# re-scoring H1. Folders already scored are skipped. Then pilot_sft/heads/summarize.py writes summary.json.
#
#   SAMPLE_STEPS="4" bash scripts/desk/score_head_run.sh <run dir> [threads]
# SAMPLE_STEPS (space-separated) limits the K sampled paths to those step counts; single paths are always scored.
set -uo pipefail
RUN="${1:?run dir}"; THREADS="${2:-12}"; SAMPLE_STEPS="${SAMPLE_STEPS:-}"
export NAVSIM_DEVKIT_ROOT=$HOME/navsim-v1.1 OPENSCENE_DATA_ROOT=$HOME/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=$HOME/navsim_workspace/dataset/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=$HOME/navsim_workspace/exp
export PYTHONPATH=$HOME/Auto-driving${PYTHONPATH:+:$PYTHONPATH}
PY=~/miniforge3/envs/navsim/bin/python
cd ~/Auto-driving
run=$(basename "$RUN")
for d in "$RUN"/navtest_*/steps*/*/; do
  d=${d%/}; [ -f "$d/shard_000_of_001.jsonl" ] || continue
  rel=${d#$RUN/}; name="heads-$run-$(echo "$rel" | tr '/' '-')"
  if [ -n "$SAMPLE_STEPS" ] && [ "$(basename "$d")" != single ]; then
    s=$(basename "$(dirname "$d")"); s=${s#steps}
    case " $SAMPLE_STEPS " in *" $s "*) ;; *) continue ;; esac
  fi
  if ls "$NAVSIM_EXP_ROOT/${name}_navtest_v1"/*/*.csv >/dev/null 2>&1; then continue; fi
  t0=$(date +%s)
  taskset -c 16-31 nice -n 5 $PY $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_navtest_v1 \
    worker.threads_per_node=$THREADS experiment_name="${name}_navtest_v1" \
    agent._target_=navsim_eval.precomputed_agent.PrecomputedTrajectoryAgent +agent.predictions_path="$d" \
    > "$d/score.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') $rel scored in $(( $(date +%s) - t0 )) s (exit $rc)"
done
$PY pilot_sft/heads/summarize.py --run "$RUN" --exp "$NAVSIM_EXP_ROOT"
