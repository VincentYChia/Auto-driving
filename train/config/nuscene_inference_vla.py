"""Inference configuration for MVLAD-AD (action-priority two-stage decoding).

Two task configs are provided, all derived from portable paths in
``paths.py``:

* ``planning_config`` - nuScenes planning (waypoint prediction, L2 metric).
* ``nux_config``      - Nu-X driving explanation (two-stage decoding).

The action-priority decoding first resolves the ``generation_first_*`` action
tokens, then fixes them and in-fills the reasoning/answer tokens.
"""

from dataclasses import dataclass, field
from typing import Tuple

from train.config import paths


@dataclass(frozen=True)
class InferenceConfig:
    job_name: str

    # Model / weights
    pretrained: str = paths.LLADA_V_PATH
    model_base: str = "GSAI-ML/LLaDA-V"
    model_name: str = "llava_llada"
    lora_path: str = ""

    # Data / output
    data_path: str = ""
    results_path: str = ""

    # Tokenizer artifacts (geometry-aware embeddings + codebook).
    tokenizer_weights: str = str(paths.TOKENIZER_WEIGHTS)
    unused_token_ids_path: str = str(paths.UNUSED_TOKEN_IDS)

    # Device
    device: str = "cuda:0"
    device_map: str = "cuda:0"
    conv_template: str = "llava_llada"

    # dLLM cache (parallel-decoding acceleration).
    use_cache: bool = True
    prompt_interval_steps: int = 25
    gen_interval_steps: int = 7
    transfer_ratio: float = 0.25

    # Stage 1: action-priority decoding (10 action tokens).
    generation_first_steps: int = 2
    generation_first_length: int = 10
    generation_first_block_length: int = 10

    # Stage 2: reasoning / answer in-filling.
    generation_steps: int = 128
    generation_length: int = 128
    generation_block_length: int = 128

    stopping_criteria: Tuple[str, ...] = ("<|eot_id|>",)


# --- nuScenes planning (single-stage action generation) ----------------------
planning_config = InferenceConfig(
    job_name="planning",
    model_name="llava_llada_lora",
    lora_path=str(paths.PLANNING_LORA),
    data_path=str(paths.DATA_DIR / "nuscenes_waypoint_text_long_prompt_val.json"),
    results_path=str(paths.RESULTS_DIR / "planning_result.json"),
    generation_first_steps=10,
    generation_steps=10,
    generation_length=10,
    generation_block_length=10,
)

# --- Nu-X explanation (two-stage action-priority decoding) -------------------
nux_config = InferenceConfig(
    job_name="nu_x",
    lora_path=str(paths.NUX_LORA),
    data_path=str(paths.DATA_DIR / "nuscenes_waypoint_text_long_prompt_val_Nu_X.json"),
    results_path=str(paths.RESULTS_DIR / "nu_x_result.json"),
    generation_first_steps=2,
    generation_steps=128,
    generation_length=128,
    generation_block_length=128,
)

CONFIGS = {
    "planning": planning_config,
    "nux": nux_config,
}
