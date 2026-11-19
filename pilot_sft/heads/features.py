"""Stored backbone features for the path generators ("heads"). Numpy only, so the tests run anywhere.

The frozen pilot reads each sample once (one prefill, no text decoding) and ``cache_features.py`` keeps a few
of its layers. One cache split lives in one folder:

- ``cache.json``: the settings (model, adapter, layers, tokens per row ``T``, width ``d``, rows, source,
  prompt version) and, once finished, ``"complete": true`` with timings.
- ``feat_<layer>.u16``: ``np.memmap`` uint16 ``[rows, T, d]``, the bfloat16 bits of that layer's hidden states
  for the sample's tokens from the image on. The system prefix before the image is the same for every sample
  (right padding, causal model), so it carries no information and is dropped. Rows shorter than ``T`` are
  zero-padded.
- ``lengths.i32``: ``np.memmap`` int32 ``[rows]``, the stored tokens of each row.
- ``meta.jsonl``: one line per row, in row order: ``id``, ``ego`` (``EGO_DIM`` numbers read back from the
  prompt the backbone saw), and for training splits ``target_xy`` (8 x 2) and ``target_yaw`` (8).
- ``progress.json``: rows written so far, so an interrupted run resumes where it stopped.

``Writer`` fills a split in row order; ``Store`` reads it for training and prediction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

COMMANDS = ("turn left", "go straight", "turn right", "none given")  # pilot_view.COMMAND_TEXT values
EGO_DIM = 3 + 4 * 3 + len(COMMANDS)  # speed, acceleration, yaw rate; 4 past poses (x, y, heading); command

_EGO = re.compile(r"Ego state: speed (-?[\d.]+) m/s, acceleration ([-+]?[\d.]+) m/s², yaw rate ([-+]?[\d.]+) rad/s\.")
_TRIPLE = re.compile(r"\((-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\)")
_COMMAND = re.compile(r"Route command: (.+?)\.\s*$")


def parse_ego(prompt: str) -> List[float]:
    """The ego vector of a ``pilot_v1.2`` prompt, read back from its text so it is exactly what the model saw:
    speed (m/s), acceleration (m/s²), yaw rate (rad/s), the 4 past poses (x, y, heading) at -1.5, -1.0, -0.5
    and 0 s, then a one-hot route command."""
    lines = prompt.split("\n")
    ego = next((_EGO.match(l) for l in lines if l.startswith("Ego state:")), None)
    if ego is None:
        raise ValueError(f"no ego state in prompt: {prompt[:200]!r}")
    pose_line = next((l for l in lines if l.startswith("Past poses")), None)
    if pose_line is None:
        raise ValueError("no 'Past poses' line: the heads need prompt pilot_v1.2")
    poses = _TRIPLE.findall(pose_line.split(":", 1)[1])
    if len(poses) != 4:
        raise ValueError(f"{len(poses)} past poses, need 4: {pose_line!r}")
    command = next((_COMMAND.match(l) for l in lines if l.startswith("Route command:")), None)
    if command is None or command.group(1) not in COMMANDS:
        raise ValueError(f"unknown route command in prompt: {prompt[-200:]!r}")
    onehot = [1.0 if c == command.group(1) else 0.0 for c in COMMANDS]
    return [float(v) for v in ego.groups()] + [float(v) for p in poses for v in p] + onehot


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: str, obj: Any) -> None:
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".part", path)


class Writer:
    """Fills one cache split in row order; re-opening an unfinished split resumes after its last full batch."""

    def __init__(self, folder: str, layers: Sequence[int], rows: int, tokens: int, width: int,
                 settings: Dict[str, Any]):
        self.folder, self.layers, self.rows, self.tokens, self.width = folder, list(layers), rows, tokens, width
        os.makedirs(folder, exist_ok=True)
        cfg_path = os.path.join(folder, "cache.json")
        shape = {"layers": self.layers, "rows": rows, "tokens": tokens, "width": width}
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                old = json.load(f)
            if {k: old.get(k) for k in shape} != shape:
                raise SystemExit(f"{folder} holds a different cache {({k: old.get(k) for k in shape})}, "
                                 f"wanted {shape}: use a new folder")
            if old.get("complete"):
                raise SystemExit(f"{folder} is already complete")
            mode = "r+"
        else:
            _write_json(cfg_path, {**settings, **shape, "dtype": "bfloat16 bits as uint16", "complete": False})
            mode = "w+"
        self.feats = {l: np.memmap(os.path.join(folder, f"feat_{l}.u16"), dtype=np.uint16, mode=mode,
                                   shape=(rows, tokens, width)) for l in self.layers}
        self.lengths = np.memmap(os.path.join(folder, "lengths.i32"), dtype=np.int32, mode=mode, shape=(rows,))
        self.progress_path = os.path.join(folder, "progress.json")
        self.done = 0
        if mode == "r+" and os.path.exists(self.progress_path):
            with open(self.progress_path, encoding="utf-8") as f:
                self.done = int(json.load(f)["rows_done"])
        meta_path = os.path.join(folder, "meta.jsonl")
        kept = []
        if os.path.exists(meta_path):  # only the rows progress.json counts: a crash can leave a torn last line
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    if len(kept) == self.done:
                        break
                    kept.append(json.loads(line))
        if len(kept) != self.done:
            raise SystemExit(f"{meta_path} has {len(kept)} rows but progress says {self.done}")
        with open(meta_path, "w", encoding="utf-8") as f:  # drop any lines written after the last full batch
            f.writelines(json.dumps(m) + "\n" for m in kept)
        self.meta = open(meta_path, "a", encoding="utf-8")

    def write(self, feats: Dict[int, np.ndarray], lengths: Sequence[int], metas: Sequence[Dict[str, Any]]) -> None:
        """One batch: ``feats[layer]`` uint16 ``[b, t, d]`` with ``t <= T``, its rows' stored ``lengths``."""
        b, start = len(metas), self.done
        if start + b > self.rows:
            raise ValueError(f"{start} + {b} rows exceed the split's {self.rows}")
        if set(feats) != set(self.layers):
            raise ValueError(f"got layers {sorted(feats)}, the cache holds {self.layers}")
        for layer, arr in feats.items():
            if arr.dtype != np.uint16 or arr.ndim != 3 or arr.shape[0] != b or arr.shape[2] != self.width:
                raise ValueError(f"layer {layer}: need uint16 [{b}, t, {self.width}], got {arr.dtype} {arr.shape}")
            if arr.shape[1] > self.tokens:
                raise ValueError(f"layer {layer}: {arr.shape[1]} tokens exceed the cache's {self.tokens}")
            mm = self.feats[layer]
            mm[start : start + b, : arr.shape[1]] = arr
            mm[start : start + b, arr.shape[1] :] = 0
        self.lengths[start : start + b] = np.asarray(lengths, dtype=np.int32)
        for mm in list(self.feats.values()) + [self.lengths]:
            mm.flush()
        self.meta.writelines(json.dumps(m) + "\n" for m in metas)
        self.meta.flush()
        os.fsync(self.meta.fileno())
        self.done += b
        _write_json(self.progress_path, {"rows_done": self.done})

    def finish(self, extra: Dict[str, Any]) -> None:
        if self.done != self.rows:
            raise ValueError(f"only {self.done} of {self.rows} rows written")
        self.meta.close()
        cfg_path = os.path.join(self.folder, "cache.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        _write_json(cfg_path, {**cfg, **extra, "complete": True})


class Store:
    """A finished cache split, read-only: memmapped features, lengths, ego vectors and targets."""

    def __init__(self, folder: str, layers: Optional[Iterable[int]] = None, allow_incomplete: bool = False):
        with open(os.path.join(folder, "cache.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        if not self.config.get("complete") and not allow_incomplete:
            raise SystemExit(f"{folder} is not finished (cache.json complete=false)")
        rows, tokens, width = self.config["rows"], self.config["tokens"], self.config["width"]
        self.layers = list(self.config["layers"] if layers is None else layers)
        missing = [l for l in self.layers if l not in self.config["layers"]]
        if missing:
            raise SystemExit(f"layers {missing} are not in {folder} (it holds {self.config['layers']})")
        self.feats = {l: np.memmap(os.path.join(folder, f"feat_{l}.u16"), dtype=np.uint16, mode="r",
                                   shape=(rows, tokens, width)) for l in self.layers}
        self.lengths = np.array(np.memmap(os.path.join(folder, "lengths.i32"), dtype=np.int32, mode="r",
                                          shape=(rows,)))
        meta = _read_jsonl(os.path.join(folder, "meta.jsonl"))
        if allow_incomplete:
            rows = len(meta)
            self.lengths = self.lengths[:rows]
        elif len(meta) != rows:
            raise SystemExit(f"{folder}: meta.jsonl has {len(meta)} rows, cache.json says {rows}")
        self.rows, self.tokens, self.width = rows, tokens, width
        self.ids = [m["id"] for m in meta]
        self.ego = np.array([m["ego"] for m in meta], dtype=np.float32)
        self.has_targets = all("target_xy" in m for m in meta)
        if self.has_targets:
            self.target_xy = np.array([m["target_xy"] for m in meta], dtype=np.float32)
            self.target_yaw = np.array([m.get("target_yaw", [0.0] * 8) for m in meta], dtype=np.float32)

    def batch(self, idx: Sequence[int]) -> Dict[str, np.ndarray]:
        """Rows ``idx``: ``feats`` uint16 ``[b, layers, t, d]`` cut to the batch's longest row, ``mask`` bool
        ``[b, t]`` (True = a real token), ``ego`` ``[b, EGO_DIM]``, and the targets when the split has them."""
        idx = np.asarray(idx)
        lengths = self.lengths[idx]
        t = int(lengths.max())
        order = np.argsort(idx)  # memmap reads in file order, then back to the requested order
        sorted_idx = idx[order]
        feats = np.stack([self.feats[l][sorted_idx, :t] for l in self.layers], axis=1)
        feats = feats[np.argsort(order)]
        out = {"feats": feats, "mask": np.arange(t)[None, :] < lengths[:, None], "ego": self.ego[idx]}
        if self.has_targets:
            out["target_xy"], out["target_yaw"] = self.target_xy[idx], self.target_yaw[idx]
        return out
