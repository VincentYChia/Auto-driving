#!/bin/bash
# Steps 4 and 7 for v1-pilot, plus the upload files, once the full fetch (step 5) is complete.
# From the repo root, in Git Bash:
#   bash scripts/local/build_v1_pilot.sh
# Refuses to start while any selected image is missing. Gate 2 (release.py) stays a separate, signed-off step.
set -euo pipefail
OUT="${OUT:-data/pretrain/v1.0-pilot}"
PYTHON="${PYTHON:-python}"                                # data builds: numpy, pyarrow, Pillow
TRAIN_PYTHON="${TRAIN_PYTHON:-../train-venv/Scripts/python.exe}"   # has the together SDK, for its file check

"$PYTHON" - "$OUT" <<'EOF'
import json, sys
from data_pipeline.v1.openscene_images import status
s = status(sys.argv[1], quiet=True)
print(json.dumps({k: s[k] for k in ("archives_done", "archives_total", "images_kept", "images_to_keep", "missing", "undecodable")}))
if s["archives_done"] < s["archives_total"] or s["images_kept"] < s["images_to_keep"] or s["missing"] or s["undecodable"]:
    sys.exit("fetch not complete: rerun  python -m data_pipeline.v1.openscene_images fetch --out %s --full --archives 200" % sys.argv[1])
EOF
"$PYTHON" -m data_pipeline.v1.canonical --out "$OUT" --sets smoke v1-pilot     # step 4, both sets
"$PYTHON" -m data_pipeline.v1.qa_report --out "$OUT" --set v1-pilot            # Gate 1 page for v1-pilot
"$PYTHON" -m data_pipeline.v1.pilot_view build --out "$OUT" --set v1-pilot     # step 7
"$PYTHON" -m data_pipeline.v1.pilot_view export --out "$OUT" --set v1-pilot    # chat_*.jsonl for Together.ai
if [ -x "$TRAIN_PYTHON" ]; then
    "$TRAIN_PYTHON" pilot_sft/together_finetune.py check --view "$OUT/views/pilot_v1/v1-pilot"
else
    echo "skipped Together's file check: no $TRAIN_PYTHON (pip install together, then run pilot_sft/together_finetune.py check)"
fi
bash scripts/local/pack_pilot_view.sh v1-pilot                                 # tar for Anvil or a rented GPU
