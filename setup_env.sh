# Source this file from the release root before running anything:
#   source setup_env.sh
#
# It sets PYTHONPATH so that `llava`, `tokenizer`, and `train.config` import
# correctly, and declares the machine-specific paths used by the pipeline.

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${RELEASE_ROOT}:${RELEASE_ROOT}/train:${PYTHONPATH}"

# --- Conda environment (this project) ----------------------------------------
# conda activate /depot/ziran/apps/jiaru/envs/vladiffusion

# --- Foundation models (HuggingFace repo id or local snapshot path) ----------
export LLADA_V_PATH="${LLADA_V_PATH:-GSAI-ML/LLaDA-V}"
export SIGLIP_PATH="${SIGLIP_PATH:-google/siglip2-so400m-patch14-384}"

# --- nuScenes dataset root (v1.0-trainval; REQUIRED for data prep / inference)
# Must contain metadata JSONs and samples/CAM_FRONT/*.jpg.
export NUSCENES_ROOT="${NUSCENES_ROOT:-/path/to/nuscenes/full}"

# --- Release-internal directories (override only if relocating) --------------
export VLAD_DATA_DIR="${VLAD_DATA_DIR:-${RELEASE_ROOT}/data}"
export VLAD_WEIGHTS_DIR="${VLAD_WEIGHTS_DIR:-${RELEASE_ROOT}/weights}"
export VLAD_TOKENIZER_DIR="${VLAD_TOKENIZER_DIR:-${RELEASE_ROOT}/tokenizer}"
export VLAD_RESULTS_DIR="${VLAD_RESULTS_DIR:-${RELEASE_ROOT}/results}"

echo "[setup_env] RELEASE_ROOT=${RELEASE_ROOT}"
echo "[setup_env] PYTHONPATH set; LLADA_V_PATH=${LLADA_V_PATH}"
echo "[setup_env] NUSCENES_ROOT=${NUSCENES_ROOT}"
