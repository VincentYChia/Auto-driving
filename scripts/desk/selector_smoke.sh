#!/bin/bash
# Smoke test of phase 6's selector pipeline (R14) on real data, CPU only (~/queue2), about 1,000 RL scenes.
# It checks the joins between candidates, PDM scores, stored features, held-out drives and NAVSIM's navdev score
# files before phase 6 depends on them. Its numbers are used for no choice (R14: no selector setting uses navdev).
set -euo pipefail
cd ~/Auto-driving
source scripts/anvil/config.local.sh
export CUDA_VISIBLE_DEVICES=
C=$HOME/navsim_workspace/head_cache/h1_l12-18-24
PY=~/miniforge3/envs/pilot/bin/python
H=$HOME/navsim_workspace/exp/heads
S=$H/selector-smoke
NC=$HOME/navsim_workspace/exp/metric_cache_navtrain_v1
mkdir -p $S
echo "$(date '+%F %T') 1. tokens"
python3 -c "
import json
ids = [json.loads(l)['id'] for l in open('$HOME/navsim_workspace/training_data/pilot_v1/v1-nt-p12/train.jsonl') if l.strip()]
open('$S/tokens.txt', 'w').write(''.join(i.rsplit('/', 1)[-1] + chr(10) for i in ids[::80][:1000]))
print(len(ids), 'RL scenes; every 80th taken')"
echo "$(date '+%F %T') 2. candidates (A1, epoch 15, 8 steps, 6 samples at 1.5), CPU"
taskset -c 24-31 $PY pilot_sft/heads/predict_head.py --run $H/a1-l12-18-24 --weights epoch_15 --split navtrain \
  --cache $C/train --only-tokens $S/tokens.txt --steps 8 --k 6 --scales 1.5 --out $S/cands --batch 32
echo "$(date '+%F %T') 3. PDM scores (port 47412)"
export NAVSIM_DEVKIT_ROOT=$HOME/navsim-v1.1 OPENSCENE_DATA_ROOT=$HOME/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=$HOME/navsim_workspace/dataset/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$HOME/navsim_workspace/exp
PYTHONPATH=$HOME/Auto-driving setsid nohup taskset -c 24-31 nice -n 5 ~/miniforge3/envs/navsim/bin/python \
  pilot_sft/heads/pdm_server.py serve --cache $NC --workers 8 --port 47412 > $S/server.log 2>&1 < /dev/null &
for i in $(seq 120); do grep -q serving $S/server.log && break; sleep 5; done
rc=0
taskset -c 24-31 $PY pilot_sft/heads/score_candidates.py --paths $S/cands/steps8 --out $S/scores.jsonl --port 47412 || rc=$?
pkill -f "pdm_server.py serve --cache $NC --workers 8 --port 47412" || true
[ $rc -eq 0 ] || exit $rc
echo "$(date '+%F %T') 4. selector, 2 epochs on CPU"
taskset -c 24-31 $PY pilot_sft/heads/train_selector.py --cache $C/train --generator $H/a1-l12-18-24/epoch_15.pt \
  --candidates $S/cands/steps8 $S/scores.jsonl --scale 1.5 --out $S/sel --epochs 2 --batch 32 --warmup 10 \
  --heldout-drives 2 --workers 4
echo "$(date '+%F %T') 5. applied to navdev's own candidates, true scores from NAVSIM's score files"
taskset -c 24-31 $PY pilot_sft/heads/apply_selector.py --selector $S/sel --cache $C/dev \
  --paths $H/a1-l12-18-24/navdev_epoch_15/steps8 --scale 1.5 --out $S/navdev --exp $HOME/navsim_workspace/exp \
  --split navdev --drives $HOME/navsim_workspace/head_cache/drives_navdev.json --batch 32
echo "$(date '+%F %T') done"
