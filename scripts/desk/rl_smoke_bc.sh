#!/bin/bash
# RL smoke test for B and C on real data, before phase 4 runs them for 1,500 updates (R16): 20 updates each under
# RL4 and RL6, from each method's R11 starting epoch at its R10 steps and sampling setting. Outputs go to
# exp/heads_smoke/, outside the study's run folders, and are used for no choice. A GPU job (~/queue, one at a time).
cd ~/Auto-driving
source scripts/anvil/config.local.sh
PY=~/miniforge3/envs/pilot/bin/python
H=$HOME/navsim_workspace/exp/heads
O=$HOME/navsim_workspace/exp/heads_smoke
NC=$HOME/navsim_workspace/exp/metric_cache_navtrain_v1
export NAVSIM_DEVKIT_ROOT=$HOME/navsim-v1.1 OPENSCENE_DATA_ROOT=$HOME/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=$HOME/navsim_workspace/dataset/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$HOME/navsim_workspace/exp
mkdir -p $O
PYTHONPATH=$HOME/Auto-driving setsid nohup taskset -c 16-31 nice -n 5 ~/miniforge3/envs/navsim/bin/python \
  pilot_sft/heads/pdm_server.py serve --cache $NC --workers 12 --port 47411 > $O/server.log 2>&1 < /dev/null &
for i in $(seq 120); do grep -q serving $O/server.log && break; sleep 5; done
# run, R11 start epoch, R10 steps, R10 sampling setting (from choices/B.json, C.json and *_rl_start.json)
for spec in "b-l12-18-24 10 10 1.5" "c-l12-18-24 10 2 2"; do
  set -- $spec
  for recipe in 4 6; do
    bc=0; [ $recipe = 6 ] && bc=1
    echo "$(date '+%F %T') $1 RL$recipe"
    taskset -c 16-31 $PY pilot_sft/heads/rl_head.py --run $H/$1 --weights epoch_$2 --out $O/$1-rl$recipe \
      --rl-scenes $HOME/navsim_workspace/training_data/pilot_v1/v1-nt-p12 --steps $3 --scale $4 \
      --lr 1e-05 --kl-beta 0 --g 8 --scenes-per-update 32 --adv dr --bc-balance $bc --lp-sd-floor 0.1 \
      --seed 0 --updates 20 --save-every 20 || echo "FAILED $1 RL$recipe"
    tail -2 $O/$1-rl$recipe/log.jsonl | cut -c1-400
  done
done
pkill -f "pdm_server.py serve --cache $NC --workers 12 --port 47411"
echo "$(date '+%F %T') done"
