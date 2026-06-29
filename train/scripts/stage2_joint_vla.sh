#!/usr/bin/env bash
# Stage 2: Joint VLA Fine-tuning (paper Sec. III-E).
# Initializes from the Stage-1 LoRA and jointly learns action + reasoning
# tokens on the Nu-X (or QA) data, aligning physical actions with semantics.
#
# Run from the release root, after sourcing setup_env.sh:
#   INIT_LORA_DIR=exp/stage1_action_warmup_XXXX bash train/scripts/stage2_joint_vla.sh
set -x
export OMP_NUM_THREADS=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_ROOT}"
export PYTHONPATH="${RELEASE_ROOT}:${RELEASE_ROOT}/train:${PYTHONPATH}"

LLM_VERSION="${LLADA_V_PATH:-GSAI-ML/LLaDA-V}"
VISION_MODEL_VERSION="${SIGLIP_PATH:-google/siglip2-so400m-patch14-384}"

# Stage-1 LoRA checkpoint to initialize from (required).
INIT_LORA_DIR="${INIT_LORA_DIR:-${RELEASE_ROOT}/weights/planning}"

# Data: Nu-X explanation JSON.
#   Nu-X : nuscenes_waypoint_text_long_prompt_train_Nu_X.json
DATA_DIR="${VLAD_DATA_DIR:-${RELEASE_ROOT}/data}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_DIR}/nuscenes_waypoint_text_long_prompt_train_Nu_X.json}"

DATE=$(date +%Y%m%d_%H%M)
RUN_NAME="stage2_joint_vla_${DATE}"
OUTPUT_DIR="${RELEASE_ROOT}/exp/${RUN_NAME}"

python -m llava.train.train_mem \
    --model_name_or_path "${LLM_VERSION}" \
    --version llava_llada \
    --data_path "${TRAIN_JSON}" \
    --lora_init_path "${INIT_LORA_DIR}" \
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
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 5e-4 \
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
