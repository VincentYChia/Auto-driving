#!/usr/bin/env bash
# Download NAVSIM data into $OPENSCENE_DATA_ROOT, in the layout NAVSIM expects
# (navsim_logs/<split>/*.pkl, sensor_blobs/<split>/<log>/CAM_F0/...). Run on a login /
# data-transfer node; the camera downloads take hours.
#
#   bash scripts/anvil/download_navsim_data.sh maps              # nuPlan maps (needed for scoring)
#   bash scripts/anvil/download_navsim_data.sh test-logs         # navtest logs, ~1 GB
#   bash scripts/anvil/download_navsim_data.sh test-cameras      # navtest camera images only (no LiDAR)
#   bash scripts/anvil/download_navsim_data.sh trainval-logs     # navtrain logs + annotations, ~14 GB
#   bash scripts/anvil/download_navsim_data.sh navtrain-current  # navtrain current-frame sensors
#   bash scripts/anvil/download_navsim_data.sh navtrain-history  # navtrain past-frame sensors
#
# URLs are the ones in NAVSIM's download/*.sh scripts.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
HF="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main"
mkdir -p "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs"
cd "$OPENSCENE_DATA_ROOT"

fetch_untar() {  # url
    local name; name="$(basename "$1")"
    wget -q --show-progress -c "$1" -O "$name"
    tar -xzf "$name"
    rm -f "$name"
}

merge_split() {  # extracted_dir split dest_parent: copy <dir>/<split> (or <dir> itself) into <dest_parent>/<split>
    local src="$1" split="$2" dest="$3/$2"
    [ -d "$src/$split" ] && src="$src/$split"
    mkdir -p "$dest"
    rsync -a "$src/" "$dest/"
}

case "${1:-}" in
maps)
    wget -q --show-progress -c https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
    unzip -q nuplan-maps-v1.1.zip && rm nuplan-maps-v1.1.zip
    mv nuplan-maps-v1.0 maps
    ;;
test-logs)
    fetch_untar "$HF/openscene-v1.1/openscene_metadata_test.tgz"
    merge_split openscene-v1.1/meta_datas test navsim_logs && rm -r openscene-v1.1
    ;;
test-cameras)
    for split in $(seq 0 31); do
        fetch_untar "$HF/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz"
        merge_split openscene-v1.1/sensor_blobs test sensor_blobs && rm -r openscene-v1.1
    done
    ;;
trainval-logs)
    fetch_untar "$HF/openscene-v1.1/openscene_metadata_trainval.tgz"
    merge_split openscene-v1.1/meta_datas trainval navsim_logs && rm -r openscene-v1.1
    ;;
navtrain-current)
    for split in $(seq 1 32); do
        fetch_untar "$HF/navsim/navtrain_current_${split}.tgz"
        merge_split "navtrain_current_${split}" trainval sensor_blobs && rm -r "navtrain_current_${split}"
    done
    ;;
navtrain-history)  # past-frame sensors; needed for the multi-frame pilot variant and for tracker runs
    for split in $(seq 1 32); do
        fetch_untar "$HF/navsim/navtrain_history_${split}.tgz"
        merge_split "navtrain_history_${split}" trainval sensor_blobs && rm -r "navtrain_history_${split}"
    done
    ;;
*)
    echo "usage: $0 {maps|test-logs|test-cameras|trainval-logs|navtrain-current|navtrain-history}" >&2
    exit 1
    ;;
esac
echo "done: $1 -> $OPENSCENE_DATA_ROOT"
ls "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs" 2>/dev/null || true
