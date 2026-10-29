#!/usr/bin/env bash
# Pretraining set v1, steps 1-2, on your own computer (docs/training_data/PRETRAIN_V1_PLAN.md §6):
# index the OpenScene trainval metadata and write the Gate 0 audit. Reads no images.
#
#   bash scripts/local/pretrain_v1_index.sh          # smoke test: the first 20 logs -> data/pretrain/smoke/
#   bash scripts/local/pretrain_v1_index.sh --full   # all logs (step 2)            -> data/pretrain/v1.0-pilot/
#
# Needs Python 3.9+ with numpy, pyarrow and PyYAML (set PYTHON to choose the interpreter), git and curl.
# Downloads go to data/pretrain/downloads/ (git-ignored): NAVSIM's split lists (a sparse clone, a few MB)
# and the metadata archive (6.6 GB, once; rerun the script to resume an interrupted download).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-python3}"
DOWNLOADS=data/pretrain/downloads
NAVSIM_ROOT="${NAVSIM_ROOT:-$DOWNLOADS/navsim}"
METADATA="$DOWNLOADS/openscene_metadata_trainval.tgz"
URL=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz

if ! "$PYTHON" -c "import numpy, pyarrow, yaml" 2>/dev/null; then
    echo "Missing packages: run '$PYTHON -m pip install numpy pyarrow pyyaml' first." >&2
    exit 1
fi
mkdir -p "$DOWNLOADS"
if [ ! -d "$NAVSIM_ROOT/.git" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/autonomousvision/navsim "$NAVSIM_ROOT"
    git -C "$NAVSIM_ROOT" sparse-checkout set navsim/planning/script/config/training \
        navsim/planning/script/config/common/train_test_split
fi
if [ ! -f "$METADATA" ]; then
    echo "Downloading the OpenScene trainval metadata (6.6 GB) to $METADATA"
    curl -L --fail -C - -o "$METADATA.part" "$URL"
    mv "$METADATA.part" "$METADATA"
fi

if [ "${1:-}" = "--full" ]; then
    OUT=data/pretrain/v1.0-pilot
    EXTRA=()
else
    OUT=data/pretrain/smoke
    EXTRA=(--max-logs 20)
fi
"$PYTHON" -m data_pipeline.v1.openscene_index --metadata "$METADATA" --navsim-root "$NAVSIM_ROOT" \
    --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
"$PYTHON" -m data_pipeline.v1.audit --out "$OUT"
echo "Gate 0 report: $OUT/audit/report.md"
