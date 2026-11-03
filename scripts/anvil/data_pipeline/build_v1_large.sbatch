#!/bin/bash
# v1-large steps 3-7 on Anvil, after `build_v1_large.sh download` (see its header): keep the selected CAM_F0
# JPEGs from the 200 downloaded archives, then canonical tables, QA summary and the pilot view.
# CPU work; the GPU partition is the default only because it is the approved account. Rerunning resumes:
# archives with a record are skipped, and each one is deleted once its record is written.
#   sbatch -A cis251316-gpu scripts/anvil/data_pipeline/build_v1_large.sbatch
#SBATCH --job-name=v1-large
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --time-min=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${DATA_CONDA_ENV:-data}"
OUT="${OUT:-$TRAINING_DATA_ROOT/v1.0-large}"
SRC="$OUT/archives_hf/openscene-v1.1/openscene_sensor_trainval_camera"
cd "$REPO_ROOT"
python -m data_pipeline.v1.openscene_images fetch --out "$OUT" --full --archives 200 --source-dir "$SRC" || true
for record in "$OUT"/images/archives/*.json; do  # an archive whose images are kept is no longer needed
    rm -f "$SRC/openscene_sensor_trainval_camera_$(basename "$record" .json).tgz"
done
python - "$OUT" <<'EOF'
import json, sys
from data_pipeline.v1.openscene_images import status
s = status(sys.argv[1], quiet=True)
print(json.dumps({k: s[k] for k in ("archives_done", "archives_total", "images_kept", "images_to_keep", "missing", "undecodable")}))
if s["archives_done"] < s["archives_total"]:
    sys.exit("fetch not complete: rerun this job once the download has finished")
EOF
python -m data_pipeline.v1.canonical --out "$OUT" --sets v1-large \
    --metadata "$OUT/archives_hf/openscene-v1.1/openscene_metadata_trainval.tgz"
python -m data_pipeline.v1.qa_report --out "$OUT" --set v1-large
python -m data_pipeline.v1.pilot_view build --out "$OUT" --set v1-large
mkdir -p "$TRAINING_DATA_ROOT/pilot_v1"
ln -sfn "$OUT/views/pilot_v1/v1-large" "$TRAINING_DATA_ROOT/pilot_v1/v1-large"  # where train_lora.sbatch looks
echo "done: SET=v1-large sbatch scripts/anvil/pilot_sft/train_lora.sbatch"
