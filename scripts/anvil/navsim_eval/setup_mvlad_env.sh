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
    einops==0.6.1 einops-exts==0.0.4 sentencepiece open_clip_torch pillow tqdm huggingface_hub scipy
pip install -e train --no-deps
python -c "import torch, transformers, peft, llava; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
