#!/bin/bash
# Backup copy of navtrain_cache.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Heads step 6 prerequisite (user 2026-11-19, diffusion plan approved): NAVSIM metric cache for navtrain, the
# PDM-score reward for RL. Unpacks the OpenScene trainval logs already on scratch (from the data builds), then
# runs NAVSIM v1.1 run_metric_caching on navtrain. Resumable: caching skips finished scenes.
set -euo pipefail
cd ~/Auto-driving && source scripts/anvil/common.sh
D=$OPENSCENE_DATA_ROOT
if [ ! -f $D/navsim_logs/trainval/.unpacked ]; then
  T=$D/tmp_trainval_unpack; rm -rf $T; mkdir -p $T
  tar -xzf /anvil/scratch/x-vchia/navsim_workspace/training_data/v1.0-large/archives_hf/openscene-v1.1/openscene_metadata_trainval.tgz -C $T
  src=$(find $T -type d -name trainval | head -1); mkdir -p $D/navsim_logs/trainval
  rsync -a "$src/" $D/navsim_logs/trainval/ && rm -rf $T && touch $D/navsim_logs/trainval/.unpacked
  echo "trainval logs: $(ls $D/navsim_logs/trainval | wc -l) files, $(du -sh $D/navsim_logs/trainval | cut -f1)"
fi
STEP=cache SPLIT=navtrain bash scripts/anvil/navsim_eval/navsim_score.sbatch
echo "navtrain cache: $(find $NAVSIM_EXP_ROOT/metric_cache_navtrain_v1 -name "*.pkl*" | wc -l) files, $(du -sh $NAVSIM_EXP_ROOT/metric_cache_navtrain_v1 | cut -f1)"
