#!/usr/bin/env bash
# Stage 1: Action-Centric Warm-up (paper Sec. III-E).
# Trains the masked VLA diffusion model on the planning task only
# (visual + instruction -> action tokens), establishing a motion prior.
#
# Run from the release root, after sourcing setup_env.sh:
#   bash train/scripts/stage1_action_warmup.sh
set -x
export OMP_NUM_THREADS=2

# Resolve release root (this script lives at release/train/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_ROOT}"
export PYTHONPATH="${RELEASE_ROOT}:${RELEASE_ROOT}/train:${PYTHONPATH}"

# Foundation models (override via env; default to HuggingFace repos).
LLM_VERSION="${LLADA_V_PATH:-GSAI-ML/LLaDA-V}"
VISION_MODEL_VERSION="${SIGLIP_PATH:-google/siglip2-so400m-patch14-384}"

# Data (planning JSON produced by data_prep/gen_data_planning.py).
DATA_DIR="${VLAD_DATA_DIR:-${RELEASE_ROOT}/data}"
TRAIN_JSON="${DATA_DIR}/nuscenes_waypoint_text_long_prompt_train.json"

DATE=$(date +%Y%m%d_%H%M)
RUN_NAME="stage1_action_warmup_${DATE}"
OUTPUT_DIR="${RELEASE_ROOT}/exp/${RUN_NAME}"

python -m llava.train.train_mem \
    --model_name_or_path "${LLM_VERSION}" \
    --version llava_llada \
    --data_path "${TRAIN_JSON}" \
    --image_folder "/" \
    --video_folder "/" \
    --lora_enable True \
    --lora_r 256 \
    --mm_tunable_parts="" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bits 16 \
    --bf16 True \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile False \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False
