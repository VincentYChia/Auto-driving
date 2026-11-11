"""``train_lora --init-adapter``: a staged run starts only from an adapter with the same LoRA settings."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft"))
import train_lora  # noqa: E402


def test_a_matching_adapter_config_passes():
    train_lora.check_adapter_config({"r": 32, "lora_alpha": 64, "modules_to_save": None}, 32, 64, "frozen")
    train_lora.check_adapter_config({"r": 32, "lora_alpha": 64, "modules_to_save": ["visual"]}, 32, 64, "full")


@pytest.mark.parametrize("config, args, why", [
    ({"r": 64, "lora_alpha": 64}, (32, 64, "frozen"), "rank"),
    ({"r": 32, "lora_alpha": 128}, (32, 64, "frozen"), "alpha"),
    ({"r": 32, "lora_alpha": 64, "modules_to_save": ["visual"]}, (32, 64, "frozen"), "vision"),
    ({"r": 32, "lora_alpha": 64}, (32, 64, "full"), "vision"),
])
def test_a_mismatched_adapter_is_refused(config, args, why):
    with pytest.raises(SystemExit, match=why):
        train_lora.check_adapter_config(config, *args)
