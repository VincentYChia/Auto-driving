"""Version-agnostic access to NAVSIM's SceneLoader and split configs (needs the devkit)."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Optional, Tuple


def navsim_config_dir() -> Path:
    import navsim  # noqa: F401  (devkit must be importable)

    return Path(navsim.__file__).parent / "planning" / "script" / "config" / "common" / "train_test_split"


def load_split(split: str) -> Tuple[Any, str]:
    """Return ``(SceneFilter, data_split)`` for a NAVSIM split name such as ``navtest`` or ``navtrain``."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg_dir = navsim_config_dir()
    split_cfg = OmegaConf.load(cfg_dir / f"{split}.yaml")
    filter_name = split
    for entry in split_cfg.get("defaults", []):
        if isinstance(entry, dict) or OmegaConf.is_dict(entry):
            filter_name = entry.get("scene_filter", filter_name)
    scene_filter = instantiate(OmegaConf.load(cfg_dir / "scene_filter" / f"{filter_name}.yaml"))
    return scene_filter, str(split_cfg["data_split"])


def default_paths(data_split: str, log_path: Optional[str], sensor_path: Optional[str]) -> Tuple[Path, Path]:
    root = os.environ.get("OPENSCENE_DATA_ROOT")
    if log_path is None or sensor_path is None:
        if not root:
            raise SystemExit("Set OPENSCENE_DATA_ROOT or pass --navsim-log-path and --sensor-blobs-path.")
    logs = Path(log_path) if log_path else Path(root) / "navsim_logs" / data_split
    sensors = Path(sensor_path) if sensor_path else Path(root) / "sensor_blobs" / data_split
    return logs, sensors


def make_scene_loader(log_path: Path, sensor_path: Path, scene_filter: Any):
    """Build a SceneLoader without sensors, for NAVSIM v1.1 (``sensor_blobs_path``) or v2 (``original_sensor_path``)."""
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    params = inspect.signature(SceneLoader.__init__).parameters
    kwargs = dict(data_path=log_path, scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors())
    if "original_sensor_path" in params:
        kwargs["original_sensor_path"] = sensor_path
    else:
        kwargs["sensor_blobs_path"] = sensor_path
    return SceneLoader(**kwargs)
