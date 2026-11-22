#!/bin/bash
# The 20 s replay page for a path generator against H1, step 2 of 2 (Anvil login node, about 2 min): the standard
# combined format the researcher confirmed on 2026-11-14 (the 20 s re-planning replay plus NAVSIM's scored moment at
# t = 0), on the 10 replay-v2 drives. Step 1 is scripts/desk/replay_head_predict.sh on the desktop; copy its
# replay_v2_<weights> folder and the run's navtest score folder here first (from_desk.sh).
#   RUN=a1-l12-18-24-rl W=rl_1250 S=8 TAG=a1rl4 PDMS=0.8727 NAME="A1 after RL4 (seed 0)" \
#     bash ~/Auto-driving/scripts/anvil/helpers/run_replay_v2_head.sh
# The page holds nuPlan camera images, so it stays local (copy it home with scp).
set -euo pipefail
source ~/Auto-driving/scripts/anvil/common.sh
activate_env "$NAVSIM_CONDA_ENV"
E=/anvil/scratch/x-vchia/navsim_workspace/exp
H1=qwen35-0.8b-v1hard-p12-visfull-ep4
H1CSV=$E/${H1}_navtest_v1/2026.11.13.19.25.25/2026.11.13.19.27.05.csv
HUMAN=$E/human_navtest_v1/2026.11.09.12.02.29/2026.11.09.12.04.18.csv
: "${RUN:?set RUN}" "${W:?set W}" "${S:?set S}" "${TAG:?set TAG}" "${PDMS:?set PDMS}" "${NAME:?set NAME}"
CSV=$(ls $E/heads-$RUN-navtest_$W-steps$S-single_navtest_v1/*/*.csv | tail -1)
REPLAY=$E/heads/$RUN/replay_v2_$W/steps$S/single
[ "$(wc -l < $REPLAY/shard_000_of_001.jsonl)" -eq 410 ] || { echo "replay paths incomplete"; exit 1; }
cd ~/Auto-driving
python navsim_eval/replay_demo.py scored --drives $E/replay/v2 --metric-cache $E/metric_cache_navtest_v1 --human "$HUMAN" \
  --model "$TAG=$E/heads/$RUN/navtest_$W/steps$S/single=$CSV" --model "H1=$E/pilot_sft/$H1/navtest_predictions=$H1CSV"
cp $E/replay/v2/navsim_t0.json $E/replay/v2/navsim_t0_${TAG}_vs_h1.json  # scored overwrites navsim_t0.json: keep each pair's
python navsim_eval/replay_demo.py page --drives $E/replay/v2 \
  --predictions $REPLAY --name "$NAME" --navtest-pdms $PDMS \
  --compare-predictions $E/pilot_sft/$H1/replay_v2_predictions \
  --compare-name "H1: 0.8B, v1-hard, vision trained, 4 epochs (text output)" --compare-pdms 0.856 \
  --navsim $E/replay/v2/navsim_t0_${TAG}_vs_h1.json --html $E/demo_renders/replay_v2_${TAG}_vs_h1.html
ls -la $E/demo_renders/replay_v2_${TAG}_vs_h1.html
