#!/bin/bash
# navdev paths of one phase-1 training run, for the pre-registered choices (docs/HEADS_PLAN.md R3, R7). Epochs 5, 10,
# 15 and 20, each at the method's step grid, with the single path (the deterministic best guess) and 6 sampled paths
# at each of the method's 3 sampling settings. GPU, a few minutes per run. Scoring happens on the CPU in the decision
# order (pilot_sft/heads/driver.py): epoch, then steps, then the sampling setting. So of the ~300 path sets, only
# ~26 are scored.
#
#   bash scripts/desk/navdev_grid.sh <run dir> <kind: masked|ddim|flow|anchor|regression>
set -euo pipefail
RUN=${1:?run dir}; KIND=${2:?kind}
C=$HOME/navsim_workspace/head_cache/h1_l12-18-24
PY=~/miniforge3/envs/pilot/bin/python
case $KIND in
  masked)     STEPS=1,2,4,8;  SCALES=0.5,1,1.5; K=6 ;;
  ddim)       STEPS=1,2,5,10; SCALES=0.5,1,1.5; K=6 ;;
  flow)       STEPS=1,2,5;    SCALES=0.5,1,2;   K=6 ;;
  anchor)     STEPS=1,2;      SCALES=0.5,1,2;   K=6 ;;
  regression) STEPS=1;        SCALES=1;         K=0 ;;
  *) echo "unknown kind $KIND"; exit 1 ;;
esac
cd ~/Auto-driving
for e in 5 10 15 20; do
  [ -f "$RUN/navdev_epoch_$e/timing.json" ] && continue
  taskset -c 16-31 $PY pilot_sft/heads/predict_head.py --run "$RUN" --split navdev --cache $C/dev --weights epoch_$e \
    --steps $STEPS --k $K --scales $SCALES
done
echo "navdev grid done for $(basename "$RUN") ($KIND)"
