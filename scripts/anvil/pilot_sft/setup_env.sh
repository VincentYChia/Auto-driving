#!/bin/bash
# One-time setup of the `pilot` conda env for pilot_sft/ on Anvil. Run on a login node from the repo root:
#   bash scripts/anvil/pilot_sft/setup_env.sh
# Then check the GPU from a debug job:
#   srun --partition=gpu-debug --gpus-per-node=1 --time=00:10:00 bash -c 'source scripts/anvil/common.sh; activate_env pilot; python -c "import torch; print(torch.cuda.get_device_name(0))"'
set -euo pipefail
source "$(dirname "$0")/../common.sh"
ENV="${PILOT_CONDA_ENV:-pilot}"
module load anaconda 2>/dev/null || module load conda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV " || conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
# CUDA 12.6 wheels run on any driver from 525 up (CUDA minor-version compatibility).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r pilot_sft/requirements.txt
# Fast kernels for Qwen3.5's linear-attention layers; without them transformers falls back to slow
# reference code (seen on Windows, 2026-11-01). causal-conv1d compiles, so it may take a while or fail;
# training still works without it.
pip install flash-linear-attention || echo "flash-linear-attention not installed: training will be slower"
pip install causal-conv1d --no-build-isolation || echo "causal-conv1d not installed: training will be slower"
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
