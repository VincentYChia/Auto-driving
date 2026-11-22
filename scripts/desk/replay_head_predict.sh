#!/bin/bash
# The 20 s replay page for a path generator, step 1 of 2 (desktop): its paths for the 410 replay frames (the 10 replay-v2
# drives, 41 re-plans each), read from H1's stored features (head_cache/h1_l12-18-24/replay_v2). CPU only, about a
# minute, so it never waits for the GPU queue.
#   bash scripts/desk/replay_head_predict.sh <run> <weights> <steps>
#   e.g. bash scripts/desk/replay_head_predict.sh a1-l12-18-24-rl rl_1250 8
# Writes exp/heads/<run>/replay_v2_<weights>/steps<steps>/single/. Step 2 runs on Anvil: copy that folder and the run's
# navtest score folder there (from_desk.sh), then scripts/anvil/helpers/run_replay_v2_head.sh. The page shows nuPlan
# camera images, so it stays local.
set -euo pipefail
cd ~/Auto-driving
RUN=$1; W=$2; S=$3
H=$HOME/navsim_workspace/exp/heads
C=$HOME/navsim_workspace/head_cache/h1_l12-18-24
[ -f $C/replay_v2/meta.jsonl ] || { echo "no replay_v2 feature cache"; exit 1; }
export CUDA_VISIBLE_DEVICES=
taskset -c 24-31 nice -n 5 ~/miniforge3/envs/pilot/bin/python pilot_sft/heads/predict_head.py --run $H/$RUN --weights $W \
  --split navtest --cache $C/replay_v2 --steps $S --k 0 --out $H/$RUN/replay_v2_$W
echo "rows: $(wc -l < $H/$RUN/replay_v2_$W/steps$S/single/shard_000_of_001.jsonl) (expect 410)"
