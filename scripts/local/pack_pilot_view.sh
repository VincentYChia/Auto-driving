#!/bin/bash
# Pack a pilot_v1 view into one file to copy to Anvil or a rented GPU (images + jsonl + view.json;
# the chat_*.jsonl exports for Together.ai are left out). From the repo root, in Git Bash or any shell:
#   bash scripts/local/pack_pilot_view.sh smoke      # -> data/pretrain/upload/pilot_v1_smoke.tar (~0.6 GB)
#   bash scripts/local/pack_pilot_view.sh v1-pilot   # after step 5 (~2.4 GB)
set -euo pipefail
SET="${1:-smoke}"
OUT="${OUT:-data/pretrain/v1.0-pilot}"
VIEW="$OUT/views/pilot_v1/$SET"
[ -f "$VIEW/train.jsonl" ] || { echo "no view at $VIEW: run python -m data_pipeline.v1.pilot_view build --set $SET first" >&2; exit 1; }
mkdir -p data/pretrain/upload
TAR="data/pretrain/upload/pilot_v1_$SET.tar"
tar -cf "$TAR" -C "$OUT/views/pilot_v1" --exclude='chat_*.jsonl' "$SET"
echo "$TAR: $(du -h "$TAR" | cut -f1)"
echo "copy:   scp $TAR <user>@anvil.rcac.purdue.edu:    # to your home; newer scp does not expand \$SCRATCH"
echo "unpack, on Anvil from the repo root:"
echo "        source scripts/anvil/common.sh && mkdir -p \$TRAINING_DATA_ROOT/pilot_v1 && tar -xf ~/pilot_v1_$SET.tar -C \$TRAINING_DATA_ROOT/pilot_v1"
