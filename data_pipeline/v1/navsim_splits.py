"""NAVSIM's official split lists, read as YAML from a NAVSIM devkit checkout (no devkit import).

Files as in NAVSIM main (V, commit 0a380a9, 2025-10-27):
* ``LOG_SPLIT_YAML``: nuPlan's log split, ``train_logs`` / ``val_logs`` / ``test_logs``
  (13,180 / 1,381 / 1,349 logs).
* ``SCENE_FILTER_DIR/*.yaml``: scene filters with ``log_names`` and ``tokens``. navtrain has 1,192 logs
  (978 train, 214 val) and 103,288 tokens.

Every scene filter except ``TRAINING_FILTERS`` is an evaluation split: navtest, navhard, warmup,
navsafe, the private tests. Their logs and tokens are guarded, i.e. never trained on or used for
selection (``CLAUDE.md``). A new evaluation filter is guarded automatically. On the commit above all
their logs are nuPlan test logs, which are guarded too.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

import yaml

from data_pipeline.v1 import provenance

LOG_SPLIT_YAML = "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
SCENE_FILTER_DIR = "navsim/planning/script/config/common/train_test_split/scene_filter"
TRAINING_FILTERS = ("navtrain", "navmini", "all_scenes")
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


@dataclass(frozen=True)
class NavsimSplits:
    train_logs: FrozenSet[str]
    val_logs: FrozenSet[str]
    test_logs: FrozenSet[str]
    navtrain_logs: FrozenSet[str]
    navtrain_tokens: FrozenSet[str]
    guarded_logs: Dict[str, str]  # log -> why, e.g. "nuplan_test+navtest"
    guarded_tokens: Dict[str, str]  # token -> why
    devkit_commit: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)  # file read -> sha256

    def nuplan_split(self, log_name: str) -> str:
        if log_name in self.train_logs:
            return "train"
        if log_name in self.val_logs:
            return "val"
        if log_name in self.test_logs:
            return "test"
        return "none"


def load(navsim_root: str) -> NavsimSplits:
    files: Dict[str, str] = {}

    def read(rel_path: str) -> dict:
        with open(os.path.join(navsim_root, rel_path), "rb") as f:
            data = f.read()
        files[rel_path] = hashlib.sha256(data).hexdigest()
        return yaml.load(data, Loader=_LOADER) or {}

    def names(spec: dict, key: str) -> List[str]:
        # NAVSIM quotes its tokens; str() only guards against a stray unquoted value.
        return [str(v) for v in spec.get(key) or ()]

    log_split = read(LOG_SPLIT_YAML)
    train, val, test = (frozenset(names(log_split, k)) for k in ("train_logs", "val_logs", "test_logs"))
    guarded_logs: Dict[str, List[str]] = {log: ["nuplan_test"] for log in test}
    guarded_tokens: Dict[str, List[str]] = {}
    navtrain = None
    for fname in sorted(os.listdir(os.path.join(navsim_root, SCENE_FILTER_DIR))):
        if not fname.endswith(".yaml"):
            continue
        name = fname[: -len(".yaml")]
        spec = read(f"{SCENE_FILTER_DIR}/{fname}")
        if name == "navtrain":
            navtrain = spec
        if name in TRAINING_FILTERS:
            continue
        for log in names(spec, "log_names"):
            guarded_logs.setdefault(log, []).append(name)
        for token in names(spec, "tokens"):
            guarded_tokens.setdefault(token, []).append(name)
    if navtrain is None:
        raise FileNotFoundError(f"no navtrain.yaml in {os.path.join(navsim_root, SCENE_FILTER_DIR)}")
    return NavsimSplits(
        train_logs=train,
        val_logs=val,
        test_logs=test,
        navtrain_logs=frozenset(names(navtrain, "log_names")),
        navtrain_tokens=frozenset(names(navtrain, "tokens")),
        guarded_logs={k: "+".join(v) for k, v in guarded_logs.items()},
        guarded_tokens={k: "+".join(v) for k, v in guarded_tokens.items()},
        devkit_commit=provenance.git_commit(navsim_root),
        files=files,
    )
