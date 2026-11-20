#!/bin/bash
# Backup copy of failure.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# failure analysis over the finished full NAVSIM scores (navsim env, CPU); MORE="name:run_dir ..." adds pilot runs
set -euo pipefail
cd ~/Auto-driving; source scripts/anvil/common.sh; activate_env "$NAVSIM_CONDA_ENV"
E=$NAVSIM_EXP_ROOT; P=$E/pilot_sft; V=$TRAINING_DATA_ROOT/pilot_v1
csv() { ls $E/$1_navtest_v1/*/*.csv | tail -1; }
EXTRA=()
for spec in ${MORE:-}; do n=${spec%%:*}; r=${spec#*:}; s=${r##*=}; r=${r%%=*}; EXTRA+=(--model "$n=$P/$r/navtest_predictions=$(csv $s)"); done
python navsim_eval/failure_analysis.py --out $E/failure_analysis/navtest_v1.json --cache $E/failure_analysis/navtest_scenes.npz \
  --model "0.8B pilot=$P/qwen35-0.8b-v1pilot-egodrop50/navtest_predictions=$(csv qwen35-0.8b-v1pilot-egodrop50)" \
  --model "2B pilot=$P/qwen35-2b-v1pilot-egodrop50/navtest_predictions=$(csv qwen35-2b-v1pilot-egodrop50)" \
  --model "MVLAD-AD=$E/mvlad/navtest_predictions=$(csv mvlad)" \
  ${EXTRA[@]+"${EXTRA[@]}"} --cv "$(csv cv)" --human "$(csv human)" \
  --view "v1-pilot train=$V/v1-pilot/train.jsonl" --view "v1-large train=$V/v1-large/train.jsonl" \
  --view "v1-xl train=$V/v1-xl/train.jsonl" --view "v1-xl-nd train=$V/v1-xl-nd/train.jsonl" \
  --view "dev=$V/v1-xl-nd/dev.jsonl"
