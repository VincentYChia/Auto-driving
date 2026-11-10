#!/bin/bash
# Views of the v1.0-full derived sets [proposed 2026-11-10] (data_pipeline/v1/derived_sets.py): v1-full-thin (v1-full's
# train standstill-thinned as v1-all, plus every navtrain row; dev-v2) and dev-v2 alone (re-evaluating earlier adapters).
# No fetch and no canonical step: the token files are committed (data/pretrain/v1.0-full/splits) and copied into the
# built folder after their SHA-256 is checked against derived.json; the views read the folder's frames.parquet.
# Links $TRAINING_DATA_ROOT/pilot_v1/<set> (pilot_v1.1) and <set>-p12 (pilot_v1.2). A view whose view.json exists is
# skipped, so resubmitting is safe.
#
#   sbatch -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_derived_views.sbatch
#   VIEWS="dev-v2:pilot_v1.1" sbatch ...   # <set>:<prompt version> pairs
#SBATCH --job-name=views-derived
#SBATCH --dependency=singleton
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${DATA_CONDA_ENV:-data}"
cd "$REPO_ROOT"
out="$TRAINING_DATA_ROOT/v1.0-full"
[ -f "$out/frames.parquet" ] || { echo "[views-derived] $out/frames.parquet missing (build v1.0-full first)" >&2; exit 1; }
python - "$REPO_ROOT/data/pretrain/v1.0-full/splits" "$out/splits" <<'EOF'
import hashlib, json, os, shutil, sys
src, dst = sys.argv[1:]
report = json.load(open(os.path.join(src, "derived.json")))
for name, s in report["sets"].items():
    for split, sha in s["sha256"].items():
        f = f"{name}_tokens_{split}.txt"
        if hashlib.sha256(open(os.path.join(src, f), "rb").read()).hexdigest() != sha:
            sys.exit(f"[views-derived] {f}: SHA-256 differs from derived.json (a CRLF checkout?)")
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
shutil.copyfile(os.path.join(src, "derived.json"), os.path.join(dst, "derived.json"))
print("[views-derived] token files checked and copied:", {n: (s["train"], s["dev"]) for n, s in report["sets"].items()})
EOF
VIEWS="${VIEWS:-dev-v2:pilot_v1.1 dev-v2:pilot_v1.2 v1-full-thin:pilot_v1.2}"
for spec in $VIEWS; do
    set_name="${spec%%:*}"
    version="${spec#*:}"
    case "$version" in
        pilot_v1.1) dir="$out/views/pilot_v1/$set_name"; link="$set_name" ;;
        pilot_v1.2) dir="$out/views/pilot_v1.2/$set_name"; link="$set_name-p12" ;;
        *) echo "[views-derived] unknown prompt version $version" >&2; exit 1 ;;
    esac
    if [ -f "$dir/view.json" ]; then
        echo "[views-derived] $set_name ($version): exists, skipped"
    else
        echo "[views-derived] $set_name ($version) $(date +%T)"
        python -m data_pipeline.v1.pilot_view build --out "$out" --set "$set_name" --storage zip \
            --workers "${WORKERS:-16}" --prompt-version "$version"
    fi
    ln -sfn "$dir" "$TRAINING_DATA_ROOT/pilot_v1/$link"
done
echo "[views-derived] done"
