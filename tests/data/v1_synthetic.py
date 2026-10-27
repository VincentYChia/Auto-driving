"""Synthetic OpenScene logs and a minimal NAVSIM split tree, for the data_pipeline/v1 tests."""

import io
import pickle
import tarfile

import yaml

from data_pipeline.v1 import navsim_splits
from tests.shared import synthetic_navsim as syn


def make_log(name, num_frames=30):
    """``syn.make_frames`` renamed to log ``name``: tokens ``<name>_tokNN``, image paths under ``<name>/``."""
    frames = syn.make_frames(num_frames)
    for frame in frames:
        frame["log_name"] = name
        frame["token"] = f"{name}_{frame['token']}"
        for cam in frame["cams"].values():
            cam["data_path"] = cam["data_path"].replace("synthetic_log", name)
    return frames


def make_standstill(frames):
    for frame in frames:
        frame["ego2global_translation"] = [1000.0, 500.0, 0.0]
        frame["ego2global_rotation"] = [1.0, 0.0, 0.0, 0.0]
        frame["ego_dynamic_state"] = [0.0, 0.0, 0.0, 0.0]
    return frames


def write_logs_dir(path, logs):
    path.mkdir(parents=True, exist_ok=True)
    for frames in logs:
        with open(path / f"{frames[0]['log_name']}.pkl", "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def write_logs_tgz(path, logs):
    """The layout of openscene_metadata_trainval.tgz: openscene-v1.1/meta_datas/trainval/<log>.pkl."""
    with tarfile.open(path, "w:gz") as tar:
        for frames in logs:
            data = pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL)
            info = tarfile.TarInfo(f"openscene-v1.1/meta_datas/trainval/{frames[0]['log_name']}.pkl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def write_navsim_root(root, train=(), val=(), test=(), navtrain_logs=(), navtrain_tokens=(), filters=None):
    """NAVSIM's split files as ``navsim_splits`` reads them. ``filters``: extra scene filters by name."""
    log_split = root / navsim_splits.LOG_SPLIT_YAML
    log_split.parent.mkdir(parents=True, exist_ok=True)
    log_split.write_text(yaml.safe_dump({"train_logs": list(train), "val_logs": list(val), "test_logs": list(test)}))
    scene_filters = root / navsim_splits.SCENE_FILTER_DIR
    scene_filters.mkdir(parents=True, exist_ok=True)
    specs = {"navtrain": {"log_names": list(navtrain_logs), "tokens": list(navtrain_tokens)}, **(filters or {})}
    for name, spec in specs.items():
        (scene_filters / f"{name}.yaml").write_text(yaml.safe_dump({"num_history_frames": 4, **spec}))
    return root
