#!/bin/bash
# One-time setup of the `mvlad` conda env for MVLAD-AD inference on Anvil. Run on a login node from the repo root:
#   bash scripts/anvil/navsim_eval/setup_mvlad_env.sh
# Versions follow the weights (transformers 4.39.3) and train/requirements.txt; only what inference imports.
# LLaDA-V uses a gated LLaMA-3 tokenizer: run `huggingface-cli login` in this env afterwards (access granted).
set -euo pipefail
source "$(dirname "$0")/../common.sh"
ENV="${MVLAD_CONDA_ENV:-mvlad}"
module load anaconda 2>/dev/null || module load conda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV " || conda create -y -n "$ENV" python=3.10
conda activate "$ENV"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.39.3 tokenizers==0.15.2 accelerate==0.29.3 peft==0.9.0 numpy==1.26.4 timm==0.9.16 \
    einops==0.6.1 einops-exts==0.0.4 sentencepiece open_clip_torch pillow tqdm scipy
# imported on the way to inference/run_inference (tokenizer/, train/); hub and datasets pinned for transformers 4.39
pip install scikit-learn matplotlib nuscenes-devkit python-dotenv pyquaternion wandb \
    huggingface_hub==0.23.4 datasets==2.18.0 "fsspec<=2024.2.0" numpy==1.26.4
pip install -e train --no-deps
python -c "import torch, transformers, peft, llava; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
python -c "import navsim_eval.predict_mvlad"  # the whole import chain of the NAVSIM predictor
# weights the jobs load, cached on scratch from the login node (compute jobs then need no download)
HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}" huggingface-cli download GSAI-ML/LLaDA-V
HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}" huggingface-cli download google/siglip2-so400m-patch14-384
