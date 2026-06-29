"""Central path resolution for the MVLAD-AD release.

All absolute, machine-specific paths are resolved here from environment
variables with sensible release-relative defaults, so the repository is
portable. Override any of these by exporting the corresponding environment
variable (see ``setup_env.sh``).
"""

import os
from pathlib import Path

# Release root = .../release  (this file lives at release/train/config/paths.py)
RELEASE_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else default


# --- Base (foundation) models -------------------------------------------------
# Downloaded from HuggingFace; can also point to a local snapshot.
LLADA_V_PATH = os.environ.get("LLADA_V_PATH", "GSAI-ML/LLaDA-V")
SIGLIP_PATH = os.environ.get("SIGLIP_PATH", "google/siglip2-so400m-patch14-384")

# --- Dataset root -------------------------------------------------------------
# nuScenes v1.0-trainval root (must contain the metadata + samples/CAM_FRONT).
NUSCENES_ROOT = os.environ.get("NUSCENES_ROOT", "")

# --- Release-internal directories --------------------------------------------
TOKENIZER_DIR = _env_path("VLAD_TOKENIZER_DIR", RELEASE_ROOT / "tokenizer")
DATA_DIR = _env_path("VLAD_DATA_DIR", RELEASE_ROOT / "data")
WEIGHTS_DIR = _env_path("VLAD_WEIGHTS_DIR", RELEASE_ROOT / "weights")
RESULTS_DIR = _env_path("VLAD_RESULTS_DIR", RELEASE_ROOT / "results")

# Tokenizer artifacts (geometry-aware embedding + codebook).
TOKENIZER_WEIGHTS = TOKENIZER_DIR / "tokenizer_model.pth"
UNUSED_TOKEN_IDS = TOKENIZER_DIR / "unused_token_ids.npy"

# Per-task LoRA adapters.
PLANNING_LORA = WEIGHTS_DIR / "planning"
NUX_LORA = WEIGHTS_DIR / "nu_x"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
