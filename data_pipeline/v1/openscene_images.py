"""Step 3 of ``docs/training_data/PRETRAIN_V1_PLAN.md``: the CAM_F0 JPEG of every selected sample,
from OpenScene's camera archives.

    python -m data_pipeline.v1.openscene_images plan   --out data/pretrain/v1.0-pilot
    python -m data_pipeline.v1.openscene_images fetch  --out data/pretrain/v1.0-pilot          # the smoke set's 20 archives
    python -m data_pipeline.v1.openscene_images fetch  --out data/pretrain/v1.0-pilot --full   # all 200, after Gate 1
    python -m data_pipeline.v1.openscene_images status --out data/pretrain/v1.0-pilot
    python -m data_pipeline.v1.openscene_images plan   --out <dir> --set v1-large   # 100k train, own folder (Anvil)
    python -m data_pipeline.v1.openscene_images fetch  --out <dir> --full --archives 200 --source-dir <hf download dir>
    python -m data_pipeline.v1.openscene_images plan   --out data/pretrain/v1.0-xl --set v1-xl \
        --extends data/pretrain/v1.0-large/splits/v1-large \
        --exclude-scenes data/pretrain/v1.0-xl/splits/navsim_v1.1_warmup_test_e2e_tokens.txt \
        --exclude-metadata data/pretrain/downloads/openscene_metadata_trainval.tgz
        # 200k train ⊇ v1-large minus the warm-up window (±7 s, train only), plus v1-xl-nd (PRETRAIN_V1_PLAN.md §10)
    python -m data_pipeline.v1.openscene_images plan   --out data/pretrain/v1.0-pool --set v1-all \
        --extends data/pretrain/v1.0-xl/splits/v1-xl-nd \
        --navtrain data/pretrain/v1.0-pool/splits/navsim_navtrain_tokens.txt \
        --exclude-scenes data/pretrain/v1.0-xl/splits/navsim_v1.1_warmup_test_e2e_tokens.txt \
        --exclude-metadata data/pretrain/downloads/openscene_metadata_trainval.tgz
        # the 0.5 s train pool (v1-all) and v1-nt, v1-xl-mix, v1-nt-mix, v1-xl's dev (token_sets.py, §11)

``plan`` reads only the metadata index (minutes, no images) [agreed 2026-10-31]:
* selects v1-pilot (``selection.PILOT``: 45k train + 2k dev, train balanced) from every log;
* orders the 200 archives so that each next one adds the most variety per GB to the selected
  frames. The smoke set (``selection.SMOKE``) is selected from the logs of the first 20;
* writes ``splits/<set>_tokens_<split>.txt`` and ``splits/manifest.json`` (both committed: IDs and
  counts only) and ``images/archives.json`` (local: OpenScene's archive → log map, with sizes and
  SHA-256 from Hugging Face).

``fetch`` works through the archives in plan order. For each one it:
* downloads it to ``images/tmp/``, resuming after a dropped connection, and checks its size and
  SHA-256 against Hugging Face's;
* checks the logs inside against OpenScene's archive → log map;
* keeps the CAM_F0 JPEG of every selected frame (smoke or v1-pilot), byte for byte, as
  ``images/CAM_F0/<log>/<token>.jpg``, once it has checked the JPEG decodes (rule 2);
* writes ``images/archives/<n>.json`` and deletes the archive. Archives with that file are skipped,
  so an interrupted run resumes when run again. Disk use peaks at the largest archive (19.3 GB)
  plus the kept images (~10 GB).

With ``--source-dir`` the archives are read where they are and never deleted. ``--storage pack``
[proposed format, v1-xl spec §4; packing agreed 2026-11-08 for Anvil's file quota] keeps the same
JPEG bytes as members of one uncompressed ``images/packs/<n>.zip`` per archive instead of one file
per frame (``image_store.py``: format, and ``read_image`` for either layout); the pack is complete
before the record is written. ``--shard i/k`` and ``--extract-workers N`` spread the archives over
processes; each archive is still one pack and one record, so a run stopped by a job's time limit
resumes at the first archive without a record:

    python -m data_pipeline.v1.openscene_images fetch --out <dir> --full --archives 200 \\
        --source-dir <hf download dir> --storage pack --shard 3/96
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import shutil
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data_pipeline.v1 import image_store, novelty, provenance, scene_exclusion, selection, stats, tables, token_sets
from data_pipeline.v1.openscene_index import CAMERA
from navsim_tools import maneuvers

HF_ROOT = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/"
ARCHIVE_MAP_URL = HF_ROOT + "openscene_sensor_trainval_0-199.json"
ARCHIVE_TREE_URL = ("https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/"
                    "openscene-v1.1/openscene_sensor_trainval_camera")
ARCHIVE_NAME = "openscene_sensor_trainval_camera_{n}.tgz"
SMOKE_ARCHIVES = 20  # the first 20 in plan order reach the smoke set's 10k + 1k (8 reach 3.8k)
SETS = (selection.PILOT, selection.SMOKE)
KB_PER_IMAGE = 211  # mean CAM_F0 JPEG, measured on archive 70 (V); estimates from it are D
CHUNK = 1 << 20
MAX_ATTEMPTS = 30  # per archive; a dropped connection resumes where it stopped
# A connection that stays below SLOW_MB_S for SLOW_WINDOW_S is dropped and resumed on a new one: the
# server sometimes leaves one connection crawling at ~1.6 MB/s while a fresh one gets ~36 (V, 2026-11-02).
SLOW_MB_S = 5.0
SLOW_WINDOW_S = 120.0
MAX_RECONNECTS = 50  # per archive; after that a slow server is simply waited for


class _SlowConnection(Exception):
    pass


# --- plan ----------------------------------------------------------------------------------------


def _load_json(source: str) -> Any:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=120) as response:
            return json.load(response)
    with open(source, encoding="utf-8") as f:
        return json.load(f)


def load_archives(archive_map: str = ARCHIVE_MAP_URL, archive_tree: str = ARCHIVE_TREE_URL) -> Dict[int, Dict[str, Any]]:
    """Archive number → logs (OpenScene's map), bytes and SHA-256 (Hugging Face's LFS record)."""
    logs = {int(k.rsplit("_", 1)[1]): list(v) for k, v in _load_json(archive_map).items()}
    files = {posixpath.basename(f["path"]): f for f in _load_json(archive_tree) if f.get("type") == "file"}
    archives = {}
    for n in sorted(logs):
        name = ARCHIVE_NAME.format(n=n)
        lfs = files[name]["lfs"]
        archives[n] = {"name": name, "url": HF_ROOT + "openscene_sensor_trainval_camera/" + name,
                       "bytes": int(lfs["size"]), "sha256": lfs["oid"], "logs": logs[n]}
    return archives


def archive_order(cands: tables.Columns, rows: np.ndarray, archives: Dict[int, Dict[str, Any]]) -> List[int]:
    """Greedy: next, the archive whose selected ``rows`` add the most variety per GB.

    Variety = sum over (split, city, lateral decision) of sqrt(selected frames), dev counted twice.
    The square root makes the 100th Singapore lane change worth more than the 10,000th Las Vegas
    straight. With every archive streamed the order changes nothing but which come first.
    """
    archive_of = {log: n for n, a in archives.items() for log in a["logs"]}
    cells: Dict[int, Counter] = {n: Counter() for n in archives}
    for r in rows:
        n = archive_of.get(cands["log_name"][r])
        if n is not None:
            cells[n][(cands["our_split"][r], cands["city"][r], cands["decision_lat"][r])] += 1

    def value(c: Counter) -> float:
        return sum(np.sqrt(v) * (2.0 if k[0] == "dev" else 1.0) for k, v in c.items())

    order: List[int] = []
    total: Counter = Counter()
    left = set(archives)
    while left:
        base = value(total)
        best = max(sorted(left), key=lambda n: (value(total + cells[n]) - base) / max(archives[n]["bytes"], 1))
        order.append(best)
        total += cells[best]
        left.remove(best)
    return order


def _write_tokens(path: str, tokens: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{t}\n" for t in sorted(tokens))


PLAN_COLUMNS = ["token", "log_name", "t_us", "our_split", "city", "decision_lon", "decision_lat",
                "standstill", "has_cam_f0", "pose_ok"]
ND_SUFFIX = "-nd"  # v1-xl-nd: v1-xl train without added Las Vegas near-duplicates [proposed 2026-11-08]


def _read_tokens(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _base_rows(cands: tables.Columns, extends: str) -> Dict[str, np.ndarray]:
    """Rows of the base set's token files ``<extends>_tokens_{train,dev}.txt``, per split.

    Every base token must be in the index, once, with the split its file names. The files must have
    LF line ends: the plan checks the copies and the dev files by their bytes' SHA-256, so a CRLF
    checkout (``core.autocrlf``) would fail those checks for no data reason.
    """
    row_of = {t: i for i, t in enumerate(cands["token"].tolist())}  # np.isin on strings takes minutes
    rows = {}
    for split in ("train", "dev"):
        path = f"{extends}_tokens_{split}.txt"
        with open(path, "rb") as f:
            if b"\r" in f.read():
                raise ValueError(f"{path} has CRLF line ends, the committed file LF; the plan compares the files' "
                                 "bytes, so use an LF copy (core.autocrlf rewrites them on a Windows checkout)")
        tokens = _read_tokens(path)
        missing = [t for t in tokens if t not in row_of]
        if missing:
            raise ValueError(f"{len(missing)} base {split} tokens are not in the index, e.g. {missing[0]}")
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"the base {split} file lists a token twice")
        r = np.array(sorted(row_of[t] for t in tokens), dtype=np.int64)
        wrong = int((cands["our_split"][r] != split).sum())
        if wrong:
            raise ValueError(f"{wrong} base {split} tokens are in another split of the index")
        rows[split] = r
    return rows


def _describe(cands: tables.Columns, rows: np.ndarray) -> Dict[str, Any]:
    """Per split: the counts a set derived from a selection reports (its rows are not drawn, so no target)."""
    out = {}
    cruise = selection.is_cruise(cands)
    for split in ("train", "dev"):
        r = rows[cands["our_split"][rows] == split]
        out[split] = {"selected": len(r), "logs": len(set(cands["log_name"][r].tolist())),
                      "cruise_share": round(float(cruise[r].mean()), 4) if len(r) else None,
                      "city": stats.value_counts(cands["city"][r]),
                      "decision_lon": stats.value_counts(cands["decision_lon"][r]),
                      "decision_lat": stats.value_counts(cands["decision_lat"][r]),
                      "standstill_share": round(float(cands["standstill"][r].mean()), 4) if len(r) else None}
    return out


def _repo_relative(path: str) -> str:
    """``path`` relative to the repo when it is inside it (manifests are committed), else as given; ``/`` separators."""
    full, root = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(provenance.REPO_ROOT))
    try:
        inside = os.path.commonpath([full, root]) == root
    except ValueError:  # another drive
        inside = False
    return (os.path.relpath(os.path.abspath(path), provenance.REPO_ROOT) if inside else path).replace(os.sep, "/")


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def plan(out_dir: str, archive_map: str = ARCHIVE_MAP_URL, archive_tree: str = ARCHIVE_TREE_URL,
         set_name: str = selection.PILOT.name, extends: Optional[str] = None,
         exclude_scenes: Optional[str] = None, exclude_window_s: float = scene_exclusion.WINDOW_S,
         exclude_metadata: Optional[str] = None, allow_unlocated: bool = False,
         navtrain: Optional[str] = None) -> Dict[str, Any]:
    """``set_name`` v1-pilot selects v1-pilot and smoke; v1-large selects only v1-large (its own dataset folder).

    v1-all (``token_sets.POOL``) plans the v1.0-pool folder's four sets instead, with no draw: ``_plan_pool``.

    v1-xl [agreed 2026-11-08; superset mechanics proposed] needs ``extends``, the prefix of the base
    set's token files (``data/pretrain/v1.0-large/splits/v1-large``). Every base row is fixed in the
    draw (``selection.select(fixed=...)``), so v1-xl contains the base and its dev set is the base's.
    The plan also writes, as [proposed] extras that need no further extraction:
    * ``v1-xl-nd``: v1-xl train through ``novelty.nested_filter`` (base ⊂ v1-xl-nd ⊂ v1-xl), same dev;
    * a copy of the base token files, so the folder holds all three sets;
    * ``v1-xl_label_conflicts.tsv``: the rows of the three sets that fail a hard-bad test (report only).

    ``exclude_scenes`` (v1-xl only) [proposed 2026-11-09]: a token list of NAVSIM scenes, e.g.
    ``splits/navsim_v1.1_warmup_test_e2e_tokens.txt``. No train row within ``exclude_window_s`` of a
    listed scene in its log is selected (``scene_exclusion.window_mask``, ``select(train_excluded=)``).
    Base train rows inside the window are dropped from the fixed rows, so v1-xl and v1-xl-nd contain the
    base minus those rows; the base's own token files are copied unchanged. Dev is unchanged. The
    manifest's ``scene_exclusion`` records the list (file, SHA-256, source), the window and the counts,
    and the list is copied into ``splits/`` when it is not there yet. ``exclude_metadata`` (OpenScene's
    ``openscene_metadata_trainval.tgz``) also places listed tokens that are frames but not index rows
    (``scene_exclusion.locate_in_metadata``). It is required when the list has tokens that are not index rows:
    such a token can be a frame of an indexed log with train rows around it (6 warm-up tokens are, V), so a
    plan without it fails open. ``allow_unlocated`` plans anyway and only reports them (tests, other lists).
    """
    if set_name == token_sets.POOL:
        if allow_unlocated:
            raise ValueError(f"allow_unlocated is not for {token_sets.POOL}: its plan places every listed warm-up frame")
        return _plan_pool(out_dir, archive_map, archive_tree, extends, navtrain, exclude_scenes, exclude_window_s,
                          exclude_metadata)
    if navtrain is not None:
        raise ValueError(f"navtrain is for {token_sets.POOL} only")
    xl = set_name == selection.XL.name
    if xl and extends is None:
        raise ValueError("v1-xl extends a base set: pass extends=<splits dir>/<base set name>, e.g. .../splits/v1-large")
    if extends is not None and not xl:
        raise ValueError(f"extends is for {selection.XL.name} only")
    if exclude_metadata is not None and exclude_scenes is None:
        raise ValueError("exclude_metadata needs exclude_scenes")
    if allow_unlocated and exclude_scenes is None:
        raise ValueError("allow_unlocated needs exclude_scenes")
    if exclude_scenes is not None and not xl:
        raise ValueError(f"exclude_scenes is for {selection.XL.name} only: earlier sets stay as committed")
    split_dir = os.path.join(out_dir, "splits")
    if xl and _same_dir(os.path.dirname(extends) or ".", split_dir):
        raise ValueError(f"--out {out_dir} is the base set's own folder: its splits/manifest.json and "
                         "images/archives.json would be replaced; plan v1-xl into a folder of its own")
    list_copy = None  # where the exclusion list goes in splits/, when it is not there already
    if exclude_scenes is not None:
        list_copy = _list_target(exclude_scenes, split_dir)
    columns = list(PLAN_COLUMNS)
    if xl:  # the near-duplicate and label-conflict tests need the path and the prompt values
        columns += [c for c in dict.fromkeys(novelty.NEAR_DUP_COLUMNS + novelty.CONFLICT_COLUMNS) if c not in columns]
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"), columns)
    archives = load_archives(archive_map, archive_tree)
    in_index = set(cands["log_name"].tolist())
    in_map = {log for a in archives.values() for log in a["logs"]}
    main = {c.name: c for c in (selection.PILOT, selection.LARGE, selection.XL)}[set_name]
    base = _base_rows(cands, extends) if xl else None
    fixed, train_excluded, exclusion = base, None, None
    if exclude_scenes is not None:
        train_excluded, exclusion = _scene_exclusion(cands, exclude_scenes, exclude_window_s, base, exclude_metadata,
                                                     allow_unlocated)
        fixed = {"train": base["train"][~train_excluded[base["train"]]], "dev": base["dev"]}
    main_rows, main_report = selection.select(cands, main, available_logs=in_map, fixed=fixed,
                                              train_excluded=train_excluded)
    if exclusion is not None:
        exclusion["train_pool_excluded"] = main_report["splits"]["train"]["train_excluded"]["after_thinning"]
    order = archive_order(cands, main_rows, archives)
    sets = [(main.name, main_rows, main_report)]
    smoke_archives: List[int] = []
    if main is selection.PILOT:
        smoke_archives = order[:SMOKE_ARCHIVES]
        smoke_logs = [log for n in smoke_archives for log in archives[n]["logs"]]
        sets.append((selection.SMOKE.name, *selection.select(cands, selection.SMOKE, available_logs=smoke_logs)))
    extra: Dict[str, Any] = {}
    if xl:
        nested, extra = _nested_sets(cands, main.name, main_rows, base, extends, fixed)
        sets += nested
        if exclusion is not None:
            extra["scene_exclusion"] = exclusion

    os.makedirs(split_dir, exist_ok=True)
    staged: List[str] = []  # each file is written as <path>.part and moved into place only once every check passed

    def part(path: str) -> str:
        staged.append(path)
        return path + ".part"

    reports = {}
    try:
        for name, rows, report in sets:
            for split in ("train", "dev"):
                path = os.path.join(split_dir, f"{name}_tokens_{split}.txt")
                if xl and name == extra["base"]["set"]:  # a byte copy of the base files
                    shutil.copyfile(f"{extends}_tokens_{split}.txt", part(path))
                else:
                    _write_tokens(part(path), cands["token"][rows][cands["our_split"][rows] == split])
            report["images_gb_estimate"] = round(len(rows) * KB_PER_IMAGE / 1e6, 2)  # D
            if xl:  # interface I1 of the v1-xl spec: counts and file hashes per selection
                for split in ("train", "dev"):
                    report[split] = int((cands["our_split"][rows] == split).sum())
                report["sha256"] = {split: _sha256(os.path.join(split_dir, f"{name}_tokens_{split}.txt.part"))
                                    for split in ("train", "dev")}
            reports[name] = report
        if xl:
            for name, report in reports.items():  # violations are bugs, so they raise
                if report["sha256"]["dev"] != extra["base"]["sha256"]["dev"]:
                    raise RuntimeError(f"{name}'s dev tokens differ from the base's")
            if reports[extra["base"]["set"]]["sha256"] != extra["base"]["sha256"]:
                raise RuntimeError("the copy of the base token files differs from the base")
            extra["label_conflicts"] = _write_label_conflicts(cands, sets, split_dir, main.name, part)
            if list_copy is not None:
                shutil.copyfile(exclude_scenes, part(list_copy))
    except BaseException:  # leave the folder as it was: remove only this call's unfinished .part files
        for path in staged:
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
        raise
    for path in staged:
        os.replace(path + ".part", path)
    wanted = set().union(*(set(rows.tolist()) for _, rows, _ in sets))
    per_archive = Counter({n: 0 for n in archives})
    archive_of = {log: n for n, a in archives.items() for log in a["logs"]}
    per_archive.update(archive_of[cands["log_name"][r]] for r in wanted)

    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    provenance.write_text(os.path.join(image_dir, "archives.json"),
                          json.dumps({str(n): a for n, a in archives.items()}, indent=1))
    manifest = {
        "step": "3: selection and archive order (docs/training_data/PRETRAIN_V1_PLAN.md)",
        "created_utc": provenance.utc_now(),
        "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
        "repo_dirty": provenance.git_dirty(provenance.REPO_ROOT),
        "archive_map_url": ARCHIVE_MAP_URL,
        "archive_map_logs": len(in_map),
        "index_logs_missing_from_map": len(in_index - in_map),
        "map_logs_missing_from_index": len(in_map - in_index),
        "archive_order": order,
        "smoke_archives": smoke_archives,
        "download_gb": {"smoke": round(sum(archives[n]["bytes"] for n in smoke_archives) / 1e9, 1),
                        "all": round(sum(a["bytes"] for a in archives.values()) / 1e9, 1)},
        "frames_to_keep": len(wanted),
        "frames_to_keep_per_archive": {str(n): per_archive[n] for n in order},
        "selections": reports,
        **extra,
    }
    provenance.write_text(os.path.join(split_dir, "manifest.json"), json.dumps(manifest, indent=1))  # last: a whole plan
    return manifest


def _same_dir(a: str, b: str) -> bool:
    """``a`` and ``b`` name one folder (``samefile`` when both exist, which also sees links)."""
    if os.path.isdir(a) and os.path.isdir(b):
        return os.path.samefile(a, b)
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _list_target(path: str, split_dir: str, name: Optional[str] = None) -> Optional[str]:
    """Where the token list ``path`` is to be copied in ``split_dir`` (None: it is there already, byte for byte).

    The copy is named ``name`` (default: the list's own file name). Refuses a CRLF list (the manifest records its
    SHA-256) and a different file of that name in ``split_dir``.
    """
    with open(path, "rb") as f:
        data = f.read()
    if b"\r" in data:
        raise ValueError(f"{path} has CRLF line ends; the manifest records its SHA-256, so use an LF copy")
    target = os.path.join(split_dir, name or os.path.basename(path))
    if os.path.exists(target) and os.path.samefile(target, path):
        return None
    if os.path.exists(target) and _sha256(target) != hashlib.sha256(data).hexdigest():
        raise ValueError(f"{target} exists and differs from {path}")
    return None if os.path.exists(target) else target


XL_APPLIES_TO = ("train only: v1-xl and v1-xl-nd (base train rows in the window dropped); dev and the base's own "
                 "token files unchanged")
POOL_APPLIES_TO = ("train only: no train row of v1-all, v1-nt-mix, v1-xl-mix or v1-nt is in the window (the pool, "
                   "the navtrain rows and the mix are cut by it); dev and the base v1-xl-nd unchanged")


def _scene_exclusion(cands: tables.Columns, path: str, window_s: float, base: Dict[str, np.ndarray],
                     metadata: Optional[str] = None, allow_unlocated: bool = False,
                     applies_to: str = XL_APPLIES_TO) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The train exclusion mask of a scene token list, and its manifest entry (``plan(exclude_scenes=)``).

    ``metadata`` (OpenScene's metadata tgz): listed tokens that are not index rows are looked up in the
    pickles of the logs that hold a listed row, and windows are centred on them too. Without ``metadata``, a
    list with tokens that are not index rows raises unless ``allow_unlocated`` (fail closed: plan docstring).
    """
    listed = scene_exclusion.read_tokens(path)
    if listed != sorted(set(listed)):
        raise ValueError(f"{path} is not a sorted list of unique tokens")
    listed_set = set(listed)
    located, scanned = None, None
    if metadata is None and not allow_unlocated:
        index_tokens = set(cands["token"].tolist())
        unplaced = sum(t not in index_tokens for t in listed)
        if unplaced:
            raise ValueError(f"{unplaced} tokens of {path} are not index rows and may be frames near train rows: "
                             "pass exclude_metadata (--exclude-metadata <openscene_metadata_trainval.tgz>) to place "
                             "them, or allow_unlocated (--allow-unlocated) to only report them")
    if metadata is not None:
        in_index = np.array([t in listed_set for t in cands["token"].tolist()], dtype=bool)
        logs = sorted(set(cands["log_name"][in_index].tolist()))
        missing = listed_set - set(cands["token"][in_index].tolist())
        located = scene_exclusion.locate_in_metadata(metadata, missing, logs)
        scanned = len(logs)
    mask, report = scene_exclusion.window_mask(cands, listed, window_s, located)

    def exact(rows: np.ndarray) -> List[str]:
        return sorted(t for t in cands["token"][rows].tolist() if t in listed_set)

    dropped = base["train"][mask[base["train"]]]
    entry = {
        "file": _repo_relative(path), "name": os.path.basename(path), "sha256": _sha256(path),
        "source": scene_exclusion.source_of(path),
        "applies_to": applies_to,
        **report,
        "metadata": None if metadata is None else {"file": os.path.basename(metadata), "logs_scanned": scanned},
        **({"unlocated_allowed": True} if metadata is None and allow_unlocated else {}),
        "base_train_dropped": len(dropped),
        "base_train_listed_tokens": len(exact(base["train"])),
        "base_dev_in_window": int(mask[base["dev"]].sum()),
        "dev_listed_tokens": exact(base["dev"]),  # kept: dev is evaluation only (reported, never trained on)
    }
    print(f"[plan] scene exclusion {entry['name']} +/- {window_s} s: {report['tokens_in_index']} of {report['tokens']} "
          f"tokens in the index, {len(report['located_in_metadata'])} more found in the metadata, "
          f"{report['tokens_not_located']} not located; "
          f"{len(dropped)} base train rows are inside the window (left out of the train sets); "
          f"{entry['base_dev_in_window']} base dev rows inside it stay (dev unchanged)", flush=True)
    return mask, entry


def _nested_sets(cands: tables.Columns, main_name: str, main_rows: np.ndarray, base: Dict[str, np.ndarray],
                 extends: str, fixed: Optional[Dict[str, np.ndarray]] = None
                 ) -> Tuple[List[Tuple[str, np.ndarray, Dict[str, Any]]], Dict[str, Any]]:
    """The two sets nested in v1-xl, base ⊂ v1-xl-nd ⊂ v1-xl, as (name, rows, report), and manifest keys.

    ``fixed``: the base rows that were fixed in the draw, when a scene exclusion dropped some (default:
    all of ``base``); then fixed ⊂ v1-xl-nd ⊂ v1-xl, and the base rows outside v1-xl are exactly the dropped ones.
    """
    fixed = base if fixed is None else fixed
    base_name, nd_name = os.path.basename(extends), main_name + ND_SUFFIX
    if base_name in (main_name, nd_name):
        raise ValueError(f"the base set cannot be {base_name}")
    base_rows = np.sort(np.concatenate([base["train"], base["dev"]]))
    fixed_rows = np.sort(np.concatenate([fixed["train"], fixed["dev"]]))
    train = main_rows[cands["our_split"][main_rows] == "train"]
    dev = main_rows[cands["our_split"][main_rows] == "dev"]
    if not np.array_equal(dev, base["dev"]):  # checked before any file is written
        raise RuntimeError(f"{main_name} dev is not {base_name}'s: {len(dev)} rows vs {len(base['dev'])}; "
                           f"its n_dev must equal the base's dev count so that no dev row is drawn")
    nd_train = novelty.nested_filter(cands, train, fixed["train"])
    nd_rows = np.sort(np.concatenate([nd_train, dev]))
    held = "base" if fixed is base else "the fixed base rows"
    if not (np.isin(fixed_rows, nd_rows).all() and np.isin(nd_rows, main_rows).all()):  # a bug, so it raises
        raise RuntimeError(f"{held} ⊂ {nd_name} ⊂ {main_name} does not hold")
    outside = np.setdiff1d(base_rows, main_rows)
    if not np.array_equal(outside, np.setdiff1d(base_rows, fixed_rows)):  # a bug, so it raises
        raise RuntimeError(f"{len(outside)} {base_name} rows are not in {main_name}, but "
                           f"{len(base_rows) - len(fixed_rows)} were dropped from the fixed rows")
    nd_report = {"derived_from": main_name, "filter": "novelty.nested_filter on train; dev unchanged",
                 "dropped_train": int(len(train) - len(nd_train)), "splits": _describe(cands, nd_rows)}
    base_report = {"copied_from": f"{base_name}_tokens_{{train,dev}}.txt", "splits": _describe(cands, base_rows)}
    if fixed is not base:
        base_report["train_not_in"] = {main_name: len(outside), nd_name: int((~np.isin(base["train"], nd_rows)).sum())}
    extra = {
        "base": {"set": base_name, "prefix": _repo_relative(extends),
                 "sha256": {s: _sha256(f"{extends}_tokens_{s}.txt") for s in ("train", "dev")},
                 "train": len(base["train"]), "dev": len(base["dev"])},
        "novelty": {"set": nd_name, **novelty.parameters()},
    }
    return [(nd_name, nd_rows, nd_report), (base_name, base_rows, base_report)], extra


def _write_label_conflicts(cands: tables.Columns, sets: Sequence[Tuple[str, np.ndarray, Any]], split_dir: str,
                           main_name: str, target: Callable[[str], str] = lambda path: path,
                           nested: bool = True) -> Dict[str, Any]:
    """``splits/<main>_label_conflicts.tsv``: token, set, reasons for each flagged row (report only).

    ``nested`` (v1-xl: each of ``sets``, largest first, holds the next): ``set`` is the smallest set that holds
    the row, with its split, e.g. ``v1-large/train`` (so it is in all three sets) or ``v1-xl/train`` (only in v1-xl).
    With a scene exclusion, base train rows in its window are in the base only, but are listed as
    ``v1-large/train`` too; the manifest's ``scene_exclusion.base_train_dropped`` counts them.
    Not ``nested`` (the pool: v1-nt and v1-xl-mix overlap and neither holds the other): ``set`` names every set
    that holds the row, comma-separated in ``sets`` order, then its split, e.g.
    ``v1-all,v1-nt-mix,v1-xl-mix,v1-nt/train`` (in all four) or ``v1-all,v1-nt-mix,v1-nt/train`` (not in v1-xl-mix).
    The rows tested are the union of the sets (= v1-xl's rows without an exclusion).
    ``target`` maps the file's path to the path actually written (``plan`` writes ``.part`` first).
    """
    holders: Dict[int, List[str]] = {}
    for name, rows, _ in sets:  # largest first
        for r in rows.tolist():
            if nested:  # each smaller set overwrites
                holders[int(r)] = [name]
            else:
                holders.setdefault(int(r), []).append(name)
    conflicts = novelty.label_conflicts(cands, np.unique(np.concatenate([rows for _, rows, _ in sets])))
    path = os.path.join(split_dir, f"{main_name}_label_conflicts.tsv")
    with open(target(path), "w", encoding="utf-8", newline="\n") as f:
        f.write("token\tset\treasons\n")
        for r, reasons in sorted(conflicts, key=lambda item: cands["token"][item[0]]):
            f.write(f"{cands['token'][r]}\t{','.join(holders[r])}/{cands['our_split'][r]}\t{','.join(reasons)}\n")
    flagged = {r for r, _ in conflicts}
    return {
        "file": os.path.basename(path), "report_only": True, "tests": list(novelty.REASONS),
        "rows": {name: {s: sum(1 for r in rows.tolist() if r in flagged and cands["our_split"][r] == s)
                        for s in ("train", "dev")} for name, rows, _ in sets},
        "by_reason": {k: sum(k in reasons for _, reasons in conflicts) for k in novelty.REASONS},
    }


POOL_COLUMNS = list(dict.fromkeys(PLAN_COLUMNS + list(novelty.CONFLICT_COLUMNS) + ["speed", "in_navtrain"]))


def _plan_pool(out_dir: str, archive_map: str, archive_tree: str, extends: Optional[str], navtrain: Optional[str],
               exclude_scenes: Optional[str], window_s: float, metadata: Optional[str]) -> Dict[str, Any]:
    """``plan(set_name="v1-all")``: the v1.0-pool folder's four sets (``token_sets``) [proposed 2026-11-09].

    Every input is required: ``extends``, the prefix of v1-xl-nd's token files (the mix's base, and its dev is
    every set's dev); ``navtrain``, NAVSIM's navtrain token list, which must be a known training list
    (``scene_exclusion.KNOWN_LISTS``: a set trained on a list must name its source); ``exclude_scenes``, the
    warm-up list; ``metadata``, OpenScene's metadata (it places the warm-up frames that are not index rows and
    explains the navtrain tokens that are not).
    Writes, in ``splits/``: ``<set>_tokens_{train,dev}.txt`` for v1-all, v1-nt-mix, v1-xl-mix and v1-nt (every dev
    file is the base's, byte for byte), byte copies of the base's files and of both lists,
    ``v1-all_label_conflicts.tsv`` (report only; the sets are not nested, so its ``set`` column names every set
    that holds the row: ``_write_label_conflicts(nested=False)``) and ``manifest.json``, staged as ``.part`` like
    ``plan``. The base
    is recorded under ``base``, not ``selections``: ``fetch`` extracts the selections' frames, which already
    hold every base row the mix keeps (v1-all holds them all).
    """
    missing = [name for name, value in (("extends", extends), ("navtrain", navtrain),
                                        ("exclude_scenes", exclude_scenes), ("exclude_metadata", metadata))
               if value is None]
    if missing:
        raise ValueError(f"{token_sets.POOL} needs {', '.join(missing)}: v1-xl-nd's token file prefix, NAVSIM's "
                         f"navtrain list ({scene_exclusion.NAVTRAIN_FILE}), the warm-up list and OpenScene's metadata")
    split_dir = os.path.join(out_dir, "splits")
    if _same_dir(os.path.dirname(extends) or ".", split_dir):
        raise ValueError(f"--out {out_dir} is the base set's own folder: plan {token_sets.POOL} into a folder of its own")
    with open(navtrain, "rb") as f:
        if b"\r" in f.read():
            raise ValueError(f"{navtrain} has CRLF line ends; the manifest records its SHA-256, so use an LF copy")
    source = scene_exclusion.source_of(navtrain)
    if source is None or source.get("filter") != "navtrain":
        raise ValueError(f"{navtrain} (SHA-256 {_sha256(navtrain)[:8]}…) is not a known navtrain token list "
                         "(scene_exclusion.KNOWN_LISTS): a set trained on a list must name its source")
    copies = [(navtrain, _list_target(navtrain, split_dir, scene_exclusion.NAVTRAIN_FILE)),
              (exclude_scenes, _list_target(exclude_scenes, split_dir))]
    listed = scene_exclusion.read_tokens(navtrain)
    if listed != sorted(set(listed)):
        raise ValueError(f"{navtrain} is not a sorted list of unique tokens")
    base_name = os.path.basename(extends)
    if base_name in token_sets.SETS:
        raise ValueError(f"the base set cannot be {base_name}")

    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"), POOL_COLUMNS)
    archives = load_archives(archive_map, archive_tree)
    in_index = set(cands["log_name"].tolist())
    in_map = {log for a in archives.values() for log in a["logs"]}
    base = _base_rows(cands, extends)
    window, exclusion = _scene_exclusion(cands, exclude_scenes, window_s, base, metadata, applies_to=POOL_APPLIES_TO)
    pool, pool_report = token_sets.train_pool(cands, window)
    exclusion["train_pool_excluded"] = pool_report["after_standstill_thinning"] - pool_report["after_window"]
    nt = token_sets.listed_train_rows(cands, listed, window)
    labels = token_sets.maneuver_labels(cands)
    mix, mix_report = token_sets.mix_rows(cands, base["train"], pool, labels, token_sets.XL_MIX)
    mix_report["numpy"] = np.__version__  # the draw is numpy's Generator.choice (token_sets.DRAW)
    train_sets = token_sets.union_sets(pool, nt.selected, mix)
    token_sets.check_sets(cands, train_sets, base["dev"], window)

    # navtrain: what each rule removes (aggregates only)
    listed_rows = np.sort(np.concatenate([nt.selected, *nt.dropped.values()]))
    dev_logs = set(cands["log_name"][base["dev"]].tolist())
    dev_log_rows = nt.dropped["train_log"]
    in_pinned = np.array([log in dev_logs for log in cands["log_name"][dev_log_rows]], dtype=bool)
    warm = set(scene_exclusion.read_tokens(exclude_scenes))
    plain, _ = scene_exclusion.window_mask(cands, warm, window_s)  # without the frames placed from the metadata
    in_window = nt.dropped["outside_window"]
    listed_set = set(listed)
    navtrain_entry = {
        "file": _repo_relative(navtrain), "name": scene_exclusion.NAVTRAIN_FILE, "sha256": _sha256(navtrain),
        "source": source, "used_as": f"{token_sets.NT}'s token list (trained on): its rows that pass every rule",
        **nt.report,
        "train_log_failures": {
            "in_pinned_dev_logs": {"rows": int(in_pinned.sum()), "pinned_dev_logs": len(dev_logs),
                                   "logs": len(set(cands["log_name"][dev_log_rows[in_pinned]].tolist()))},
            "in_other_dev_logs": {"rows": int((~in_pinned).sum()),
                                  "logs": len(set(cands["log_name"][dev_log_rows[~in_pinned]].tolist()))},
            "pinned_dev_rows_listed": sum(t in listed_set for t in cands["token"][base["dev"]].tolist())},
        "window_failures": {"warmup_tokens": sum(t in warm for t in cands["token"][in_window].tolist()),
                            "only_through_frames_placed_from_the_metadata": int((~plain[in_window]).sum())},
        "not_index_rows": token_sets.explain_unindexed(metadata, nt.unindexed),
        "index_in_navtrain_matches": bool(np.array_equal(listed_rows, np.flatnonzero(cands["in_navtrain"]))),
    }
    rules = {
        token_sets.POOL: "the pool (manifest pool) ∪ v1-nt",
        token_sets.NT_MIX: "v1-nt ∪ v1-xl-mix",
        token_sets.MIX: "base v1-xl-nd ∪ pool rows by city and maneuver, stay-stopped thinned to 1% (manifest mix)",
        token_sets.NT: "the navtrain list's rows that pass every rule, in order (manifest navtrain)",
    }
    reference = {base_name: base["train"], "navtrain": listed_rows}
    sets = []
    for name in token_sets.SETS:
        rows = np.sort(np.concatenate([train_sets[name], base["dev"]]))
        sets.append((name, rows, {"rule": rules[name], "splits": _describe(cands, rows),
                                  "train_detail": token_sets.describe_train(cands, labels, train_sets[name], reference)}))

    os.makedirs(split_dir, exist_ok=True)
    staged: List[str] = []

    def part(path: str) -> str:
        staged.append(path)
        return path + ".part"

    base_sha = {split: _sha256(f"{extends}_tokens_{split}.txt") for split in ("train", "dev")}
    reports = {}
    extra: Dict[str, Any] = {}
    try:
        for name, rows, report in sets:
            for split in ("train", "dev"):
                path = os.path.join(split_dir, f"{name}_tokens_{split}.txt")
                _write_tokens(part(path), cands["token"][rows][cands["our_split"][rows] == split])
                report[split] = int((cands["our_split"][rows] == split).sum())
            report["sha256"] = {split: _sha256(os.path.join(split_dir, f"{name}_tokens_{split}.txt.part"))
                                for split in ("train", "dev")}
            report["images_gb_estimate"] = round(len(rows) * KB_PER_IMAGE / 1e6, 2)  # D
            if report["sha256"]["dev"] != base_sha["dev"]:  # a bug, so it raises
                raise RuntimeError(f"{name}'s dev tokens differ from the base's")
            reports[name] = report
        for split in ("train", "dev"):  # byte copies of the base, so the folder holds what the mix was built from
            src = f"{extends}_tokens_{split}.txt"
            dst = os.path.join(split_dir, f"{base_name}_tokens_{split}.txt")
            shutil.copyfile(src, part(dst))
            if _sha256(dst + ".part") != base_sha[split]:
                raise RuntimeError(f"the copy of {src} differs from it")
        for src, dst in copies:
            if dst is not None:
                shutil.copyfile(src, part(dst))
        extra["label_conflicts"] = _write_label_conflicts(cands, sets, split_dir, token_sets.POOL, part,
                                                           nested=False)  # v1-nt and v1-xl-mix overlap
    except BaseException:  # leave the folder as it was: remove only this call's unfinished .part files
        for path in staged:
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
        raise
    for path in staged:
        os.replace(path + ".part", path)

    all_rows = sets[0][1]  # v1-all with dev: every other set is inside it (check_sets)
    order = archive_order(cands, all_rows, archives)
    archive_of = {log: n for n, a in archives.items() for log in a["logs"]}
    per_archive = Counter({n: 0 for n in archives})
    per_archive.update(archive_of[cands["log_name"][r]] for r in all_rows.tolist())
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    provenance.write_text(os.path.join(image_dir, "archives.json"),
                          json.dumps({str(n): a for n, a in archives.items()}, indent=1))
    manifest = {
        "step": "3: selection and archive order (docs/training_data/PRETRAIN_V1_PLAN.md)",
        "created_utc": provenance.utc_now(),
        "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
        "repo_dirty": provenance.git_dirty(provenance.REPO_ROOT),
        "archive_map_url": ARCHIVE_MAP_URL,
        "archive_map_logs": len(in_map),
        "index_logs_missing_from_map": len(in_index - in_map),
        "map_logs_missing_from_index": len(in_map - in_index),
        "archive_order": order,
        "smoke_archives": [],
        "download_gb": {"smoke": 0.0, "all": round(sum(a["bytes"] for a in archives.values()) / 1e9, 1)},
        "frames_to_keep": len(all_rows),
        "frames_to_keep_per_archive": {str(n): per_archive[n] for n in order},
        "selections": reports,
        "base": {"set": base_name, "prefix": _repo_relative(extends), "sha256": base_sha,
                 "train": len(base["train"]), "dev": len(base["dev"]),
                 "role": f"{token_sets.MIX}'s base; its dev is every set's dev; not a selection (not fetched)"},
        "pool": pool_report,
        "navtrain": navtrain_entry,
        "mix": mix_report,
        "maneuver_rules": {"module": "navsim_tools.maneuvers", "labels": list(maneuvers.LABELS),
                           "turn_rad": maneuvers.TURN_RAD, "horizon_s": list(maneuvers.HORIZON_S),
                           "input": "the index's path_xyyaw at +0.5 ... +4.0 s (float32, read as float64) and speed",
                           "priority": "turn > stay stopped / start from a stop > brake to a stop > sideways shift > "
                                       "keep going (navsim_tools/maneuvers.py)"},
        "scene_exclusion": exclusion,
        **extra,
    }
    provenance.write_text(os.path.join(split_dir, "manifest.json"), json.dumps(manifest, indent=1))  # last: a whole plan
    return manifest


# --- fetch ---------------------------------------------------------------------------------------


def _download(archive: Dict[str, Any], dest: str) -> float:
    """Download to ``dest``, resuming a partial ``dest.part``; check size and SHA-256. Returns seconds."""
    start = time.time()
    part = dest + ".part"
    attempt = reconnects = 0
    while attempt < MAX_ATTEMPTS:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= archive["bytes"]:
            break
        try:
            request = urllib.request.Request(archive["url"], headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the range: start again
                next_report = have + (1 << 30)
                window_t, window_have = time.time(), have
                with open(part, "ab" if have else "wb") as f:
                    for block in iter(lambda: response.read(CHUNK), b""):
                        f.write(block)
                        have += len(block)
                        if have >= next_report:
                            rate = have / max(time.time() - start, 1e-6) / 1e6
                            print(f"[fetch]   {archive['name']}: {have / 1e9:.1f} / {archive['bytes'] / 1e9:.1f} GB "
                                  f"({rate:.1f} MB/s)", flush=True)
                            next_report += 1 << 30
                        now = time.time()
                        if now - window_t >= SLOW_WINDOW_S:
                            rate = (have - window_have) / (now - window_t) / 1e6
                            if rate < SLOW_MB_S and reconnects < MAX_RECONNECTS:
                                raise _SlowConnection(f"{rate:.1f} MB/s")
                            window_t, window_have = now, have
        except _SlowConnection as slow:
            reconnects += 1
            print(f"[fetch]   {archive['name']}: slow connection ({slow}) at {have / 1e9:.2f} GB, resuming on a new one",
                  flush=True)
        except (OSError, ValueError) as error:  # URLError, timeouts and dropped connections are OSErrors
            attempt += 1
            wait = min(300, 10 * attempt)
            print(f"[fetch]   attempt {attempt} stopped at {have / 1e9:.2f} GB: {error!r}; resuming in {wait} s", flush=True)
            time.sleep(wait)
    size = os.path.getsize(part) if os.path.exists(part) else 0
    if size != archive["bytes"]:
        raise RuntimeError(f"{archive['name']}: {size} of {archive['bytes']} bytes after {MAX_ATTEMPTS} attempts")
    if provenance.sha256_file(part) != archive["sha256"]:
        os.remove(part)
        raise RuntimeError(f"{archive['name']}: SHA-256 differs from Hugging Face's; deleted, will download again")
    os.replace(part, dest)
    return time.time() - start


def _decode(data: bytes) -> List[int]:
    """Rule 2: the JPEG decodes. Returns [width, height]."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError(f"format {image.format}")
        image.load()
        return list(image.size)


def _member_name(log: str, token: str) -> str:
    """The kept JPEG's path relative to the dataset directory: ``canonical.image_path``, and its pack member name."""
    return f"images/{CAMERA}/{log}/{token}.jpg"


def _extract(tgz: str, archive: Dict[str, Any], wanted: Dict[str, Dict[str, str]], image_root: str,
             pack: Optional[str] = None) -> Dict[str, Any]:
    """Keep the wanted CAM_F0 JPEGs of one archive. ``wanted``: cam_f0_path → {token, log}.

    ``pack`` None writes each JPEG as a loose file under ``image_root``. A path writes them all, byte
    for byte, as the members of one pack (``image_store.write_pack``: ``.part``, fsync, ``os.replace``),
    with no file per frame; the frames of one archive are held in memory until then, because the members are
    sorted and the tar stream is not: at most 3,543 in v1-xl and 7,447 in v1.0-pool (V: the plans'
    ``frames_to_keep_per_archive``), about 0.7 and 1.5 GB at v1-xl's 205.7 KB per packed frame (D).
    """
    start = time.time()
    logs_seen, images, undecodable = set(), [], []
    members = 0
    found = set()
    packed: List[Tuple[str, bytes]] = []
    with tarfile.open(tgz, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            members += 1
            parts = member.name.split("/")
            if len(parts) >= 3:
                logs_seen.add(parts[-3])
            rel = posixpath.join(*parts[-3:]) if len(parts) >= 3 else ""
            if parts[-2:-1] != [CAMERA] or rel not in wanted:
                continue
            frame = wanted[rel]
            data = tar.extractfile(member).read()
            found.add(rel)
            try:
                size = _decode(data)
            except Exception as error:  # any decoder error means rule 2 fails
                undecodable.append({"token": frame["token"], "error": repr(error)[:200]})
                continue
            # The path is built from the index's log name and token, never from the member name.
            if pack is not None:
                packed.append((_member_name(frame["log"], frame["token"]), data))
            else:
                path = os.path.join(image_root, CAMERA, frame["log"], frame["token"] + ".jpg")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".part", "wb") as f:
                    f.write(data)
                os.replace(path + ".part", path)
            images.append({"token": frame["token"], "log": frame["log"], "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest(), "width": size[0], "height": size[1]})
    in_archive = {rel: f for rel, f in wanted.items() if f["log"] in set(archive["logs"])}
    pack_fields: Dict[str, Any] = {}
    if pack is not None:  # written before the record, so a record always means a finished pack
        written = image_store.write_pack(pack, packed)
        pack_fields = {"pack": os.path.relpath(pack, os.path.dirname(image_root)).replace(os.sep, "/"),
                       "pack_members": written["members"], "pack_bytes": written["bytes"],
                       "pack_sha256": written["sha256"]}
    return {
        "logs_in_map": sorted(archive["logs"]),
        "logs_in_archive": sorted(logs_seen),
        "logs_match_map": logs_seen == set(archive["logs"]),
        "members": members,
        "wanted": len(in_archive),
        "kept": len(images),
        "missing": sorted(f["token"] for rel, f in in_archive.items() if rel not in found),
        "undecodable": undecodable,
        "image_bytes": sum(i["bytes"] for i in images),
        "extract_s": round(time.time() - start, 1),
        "images": images,
        **pack_fields,
    }


def _wanted(out_dir: str, manifest: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    tokens = set()
    for name in manifest["selections"]:
        for split in ("train", "dev"):
            with open(os.path.join(out_dir, "splits", f"{name}_tokens_{split}.txt"), encoding="utf-8") as f:
                tokens.update(line.strip() for line in f if line.strip())
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"), ["token", "log_name", "cam_f0_path"])
    keep = np.array([t in tokens for t in cands["token"]], dtype=bool)  # np.isin on strings takes minutes
    return {str(p): {"token": str(t), "log": str(log)}
            for t, log, p in zip(cands["token"][keep], cands["log_name"][keep], cands["cam_f0_path"][keep])}


STORAGE = ("loose", "pack")  # how fetch keeps the JPEGs; see image_store.py


def _write_record(record_path: str, record: Dict[str, Any]) -> None:
    record["finished_utc"] = provenance.utc_now()
    provenance.write_text(record_path, json.dumps(record))  # synced before the rename: a record is never empty


def _print_kept(n: int, record: Dict[str, Any]) -> None:
    print(f"[fetch] archive {n}: kept {record['kept']} of {record['wanted']}, {len(record['missing'])} missing, "
          f"{len(record['undecodable'])} undecodable, logs match map: {record['logs_match_map']}"
          + (f", pack {record['pack_bytes'] / 1e6:.0f} MB" if "pack" in record else ""), flush=True)


def _source_record(n: int, archive: Dict[str, Any], tgz: str, wanted: Dict[str, Dict[str, str]], image_root: str,
                   pack: Optional[str]) -> Dict[str, Any]:
    """The record of one archive in a source directory: check its SHA-256, then extract. ``tgz`` is only read."""
    if provenance.sha256_file(tgz) != archive["sha256"]:
        raise RuntimeError(f"{archive['name']}: SHA-256 differs")
    record = {"archive": n, "name": archive["name"], "bytes": archive["bytes"], "sha256": archive["sha256"],
              "sha256_ok": True, "download_s": 0.0}
    record.update(_extract(tgz, archive, wanted, image_root, pack))
    return record


_worker_wanted: Dict[str, Dict[str, str]] = {}


def _init_extract_worker(wanted: Dict[str, Dict[str, str]]) -> None:
    global _worker_wanted
    _worker_wanted = wanted


def _extract_task(n: int, archive: Dict[str, Any], tgz: str, image_root: str, pack: Optional[str],
                  record_path: str) -> Dict[str, Any]:
    """One archive in an extraction worker: its record is written there, so a stopped job keeps it."""
    print(f"[fetch] archive {n}, {archive['bytes'] / 1e9:.1f} GB (worker {os.getpid()})", flush=True)
    record = _source_record(n, archive, tgz, _worker_wanted, image_root, pack)
    _write_record(record_path, record)
    return record


def parse_shard(shard: Any, num_shards: Optional[int] = None) -> Tuple[int, int]:
    """``"3"`` with ``num_shards`` 96, or ``"3/96"``, → (3, 96). The shard must be in 0 … num_shards − 1."""
    text = str(shard)
    if "/" in text:
        i, k = (int(v) for v in text.split("/", 1))
        if num_shards not in (None, 1, k):
            raise ValueError(f"shard {text} and num_shards {num_shards} disagree")
    else:
        i, k = int(text), 1 if num_shards is None else int(num_shards)
    if not (k >= 1 and 0 <= i < k):
        raise ValueError(f"shard {i} of {k}: needs 0 <= shard < num_shards")
    return i, k


def fetch(out_dir: str, n_archives: int = SMOKE_ARCHIVES, source_dir: Optional[str] = None,
          workers: int = 1, shard: Any = 0, num_shards: int = 1, storage: str = "loose",
          extract_workers: int = 1) -> Dict[str, Any]:
    """Fetch the first ``n_archives`` archives of the plan.

    ``workers`` > 1 downloads that many archives at once (the server caps each connection at
    ~15 MB/s, V 2026-11-02); extraction and records stay one archive at a time, in plan order.
    ``source_dir`` reads archives already downloaded there instead (tests, or ``hf download`` on
    Anvil); archives not there yet are skipped, so running again picks them up. An archive in
    ``source_dir`` is only read, never deleted; only one this function downloaded itself (to
    ``images/tmp/``) is deleted, after its record is written.
    ``shard`` of ``num_shards`` takes every ``num_shards``-th archive of the plan, so several processes
    can extract at once: each archive has its own record and its frames their own files (or pack).
    ``shard`` may also be given as ``"i/k"``.
    ``extract_workers`` > 1 (``source_dir`` only) extracts that many archives at once in a process
    pool, within this process's shard. Each worker writes its archive's pack and record itself, so a
    job stopped by its time limit keeps every finished archive and the next run resumes.
    ``storage`` "loose" (default) keeps ``images/CAM_F0/<log>/<token>.jpg``. "pack" [proposed format,
    v1-xl spec §4] keeps the same bytes as the members ``images/CAM_F0/<log>/<token>.jpg`` of
    ``images/packs/<archive>.zip`` (``image_store``) and adds ``pack``, ``pack_members``, ``pack_bytes``
    and ``pack_sha256`` to the record; ``members`` stays the number of tar members streamed.
    """
    if storage not in STORAGE:
        raise ValueError(f"storage must be one of {STORAGE}, not {storage!r}")
    if extract_workers > 1 and not source_dir:
        raise ValueError("extract_workers > 1 needs source_dir (downloaded archives extract one at a time)")
    shard, num_shards = parse_shard(shard, num_shards)
    try:  # PIL checks each JPEG: fail before the first download, not after it
        import PIL  # noqa: F401
    except ImportError as error:
        raise SystemExit("fetch needs Pillow to check that each JPEG decodes: python -m pip install pillow") from error
    with open(os.path.join(out_dir, "splits", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(out_dir, "images", "archives.json"), encoding="utf-8") as f:
        archives = {int(k): v for k, v in json.load(f).items()}
    wanted = _wanted(out_dir, manifest)
    image_root = os.path.join(out_dir, "images")
    record_dir = os.path.join(image_root, "archives")
    tmp_dir = os.path.join(image_root, "tmp")
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    def pack_of(n: int) -> Optional[str]:
        return image_store.pack_path(out_dir, n) if storage == "pack" else None

    done, failed = [], []
    order = manifest["archive_order"][:n_archives]
    pending = [n for k, n in enumerate(order) if k % num_shards == shard
               and not os.path.exists(os.path.join(record_dir, f"{n}.json"))]
    not_yet = []
    if source_dir:
        not_yet = [n for n in pending if not os.path.exists(os.path.join(source_dir, archives[n]["name"]))]
        pending = [n for n in pending if n not in not_yet]
    if source_dir and extract_workers > 1 and pending:
        with ProcessPoolExecutor(max_workers=extract_workers, initializer=_init_extract_worker,
                                 initargs=(wanted,)) as extractors:
            futures = {extractors.submit(_extract_task, n, archives[n], os.path.join(source_dir, archives[n]["name"]),
                                         image_root, pack_of(n), os.path.join(record_dir, f"{n}.json")): n
                       for n in pending}
            for future in as_completed(futures):
                n = futures[future]
                try:
                    record = future.result()
                except Exception as error:  # keep going, as below; the next run retries this archive
                    print(f"[fetch] archive {n} failed: {error!r}", flush=True)
                    failed.append({"archive": n, "error": repr(error)[:500]})
                    continue
                done.append(n)
                _print_kept(n, record)
        done.sort(key=order.index)
        failed.sort(key=lambda f: order.index(f["archive"]))
        pending = []
    pool = ThreadPoolExecutor(max_workers=workers) if not source_dir else None
    downloads: Dict[int, Future] = {}
    for i, n in enumerate(pending):
        k = order.index(n) + 1
        record_path = os.path.join(record_dir, f"{n}.json")
        archive = archives[n]
        if pool:  # keep the next `workers` downloads running while this one is extracted
            for m in pending[i : i + workers]:
                if m not in downloads:
                    downloads[m] = pool.submit(_download, archives[m], os.path.join(tmp_dir, archives[m]["name"]))
        print(f"[fetch] archive {n} ({k}/{len(order)}), {archive['bytes'] / 1e9:.1f} GB", flush=True)
        try:
            if source_dir:
                tgz = os.path.join(source_dir, archive["name"])
                record = _source_record(n, archive, tgz, wanted, image_root, pack_of(n))
            else:
                tgz = os.path.join(tmp_dir, archive["name"])
                download_s = downloads.pop(n).result()
                record = {"archive": n, "name": archive["name"], "bytes": archive["bytes"],
                          "sha256": archive["sha256"], "sha256_ok": True, "download_s": round(download_s, 1)}
                record.update(_extract(tgz, archive, wanted, image_root, pack_of(n)))
        except Exception as error:  # keep going: one bad archive should not stop a multi-day run
            print(f"[fetch] archive {n} failed: {error!r}", flush=True)
            failed.append({"archive": n, "error": repr(error)[:500]})
            continue
        _write_record(record_path, record)
        if not source_dir:  # only the copy downloaded to images/tmp/ above; a source_dir archive is never touched
            os.remove(tgz)
        done.append(n)
        _print_kept(n, record)
    if pool:
        pool.shutdown()
    result = {"done_now": done, "failed": failed, "not_in_source_dir": not_yet, **status(out_dir, quiet=True)}
    print(json.dumps({**{k: v for k, v in result.items() if k not in ("archives_left", "not_in_source_dir")},
                      "not_in_source_dir": len(not_yet)}), flush=True)
    return result


def status(out_dir: str, quiet: bool = False) -> Dict[str, Any]:
    with open(os.path.join(out_dir, "splits", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    record_dir = os.path.join(out_dir, "images", "archives")
    records = []
    for n in manifest["archive_order"]:
        path = os.path.join(record_dir, f"{n}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
    result = {
        "archives_done": len(records),
        "archives_total": len(manifest["archive_order"]),
        "smoke_archives_done": sum(r["archive"] in manifest["smoke_archives"] for r in records),
        "images_kept": sum(r["kept"] for r in records),
        "images_to_keep": manifest["frames_to_keep"],
        "image_gb": round(sum(r["image_bytes"] for r in records) / 1e9, 2),
        "downloaded_gb": round(sum(r["bytes"] for r in records) / 1e9, 1),
        "missing": sum(len(r["missing"]) for r in records),
        "undecodable": sum(len(r["undecodable"]) for r in records),
        "logs_not_matching_map": [r["archive"] for r in records if not r["logs_match_map"]],
        "archives_left": [n for n in manifest["archive_order"] if n not in {r["archive"] for r in records}],
    }
    packed = [r for r in records if "pack" in r]
    if packed:  # only for --storage pack, so a loose build's status is unchanged
        result["archives_packed"] = len(packed)
        result["pack_gb"] = round(sum(r["pack_bytes"] for r in packed) / 1e9, 2)
    if not quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "archives_left"}, indent=1))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step 3: CAM_F0 JPEGs of the selected samples.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "select the samples and order the archives (metadata only)"),
                       ("fetch", "download archives in plan order and keep the selected CAM_F0 JPEGs"),
                       ("status", "what has been fetched so far")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--out", required=True, help="dataset directory holding index/, e.g. data/pretrain/v1.0-pilot")
        if name == "plan":
            p.add_argument("--set", default=selection.PILOT.name,
                           choices=(selection.PILOT.name, selection.LARGE.name, selection.XL.name, token_sets.POOL),
                           help="v1-pilot (with smoke), v1-large, v1-xl (with v1-xl-nd; needs --extends), or "
                                "v1-all (the v1.0-pool folder: v1-all, v1-nt-mix, v1-xl-mix, v1-nt; needs --extends, "
                                "--navtrain, --exclude-scenes, --exclude-metadata), each in its own dataset directory")
            p.add_argument("--extends", help="v1-xl: prefix of the base token files, "
                                             "e.g. data/pretrain/v1.0-large/splits/v1-large; v1-all: v1-xl-nd's, "
                                             "data/pretrain/v1.0-xl/splits/v1-xl-nd")
            p.add_argument("--navtrain", help="v1-all: NAVSIM's navtrain token list, e.g. data/pretrain/v1.0-pool/splits/"
                                              + scene_exclusion.NAVTRAIN_FILE)
            p.add_argument("--exclude-scenes", help="v1-xl, v1-all: token list of NAVSIM scenes no train sample may come "
                                                    "near, e.g. data/pretrain/v1.0-xl/splits/" + scene_exclusion.WARMUP_FILE)
            p.add_argument("--exclude-window-s", type=float, default=scene_exclusion.WINDOW_S,
                           help="with --exclude-scenes: same-log window around each listed scene, seconds, inclusive")
            p.add_argument("--exclude-metadata", help="with --exclude-scenes: openscene_metadata_trainval.tgz, to place "
                                                      "listed tokens that are frames but not index rows (required "
                                                      "when the list has tokens that are not index rows)")
            p.add_argument("--allow-unlocated", action="store_true",
                           help="with --exclude-scenes and no --exclude-metadata: plan anyway, only reporting the "
                                "listed tokens that are not index rows (fails open; not for the committed plan)")
            p.add_argument("--map", default=ARCHIVE_MAP_URL, help="OpenScene's archive → log map: URL or local copy")
            p.add_argument("--tree", default=ARCHIVE_TREE_URL,
                           help="Hugging Face's file listing of the camera archives: URL or local copy")
        if name == "fetch":
            p.add_argument("--archives", type=int, default=SMOKE_ARCHIVES, help="the first N archives of the plan")
            p.add_argument("--full", action="store_true", help="allow more than the smoke set's archives (after Gate 1)")
            p.add_argument("--workers", type=int, default=1, help="archives downloading at once")
            p.add_argument("--source-dir", help="read archives already downloaded here; skip those not there yet")
            p.add_argument("--shard", default="0",
                           help="with --num-shards: this process's share of the archives; or i/k for both")
            p.add_argument("--num-shards", type=int, default=None)
            p.add_argument("--storage", default="loose", choices=STORAGE,
                           help="loose: one file per JPEG (default); pack: one uncompressed zip per archive, "
                                "images/packs/<n>.zip [proposed format]")
            p.add_argument("--extract-workers", type=int, default=1,
                           help="with --source-dir: archives extracted at once by a process pool")
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.set == token_sets.POOL:
            need = [flag for flag, value in (("--extends", args.extends), ("--navtrain", args.navtrain),
                                             ("--exclude-scenes", args.exclude_scenes),
                                             ("--exclude-metadata", args.exclude_metadata)) if not value]
            if need:
                parser.error(f"--set {token_sets.POOL} needs {' '.join(need)}")
            if args.allow_unlocated:
                parser.error(f"--allow-unlocated is not for --set {token_sets.POOL}")
        else:
            if args.navtrain:
                parser.error(f"--navtrain is for --set {token_sets.POOL} only")
            if args.set == selection.XL.name and not args.extends:
                parser.error(f"--set {selection.XL.name} needs --extends <base token file prefix>")
            if args.extends and args.set != selection.XL.name:
                parser.error(f"--extends is for --set {selection.XL.name} only")
            if args.exclude_scenes and args.set != selection.XL.name:
                parser.error(f"--exclude-scenes is for --set {selection.XL.name} only")
        manifest = plan(args.out, args.map, args.tree, set_name=args.set, extends=args.extends,
                        exclude_scenes=args.exclude_scenes, exclude_window_s=args.exclude_window_s,
                        exclude_metadata=args.exclude_metadata, allow_unlocated=args.allow_unlocated,
                        navtrain=args.navtrain)
        print(json.dumps({k: manifest[k] for k in ("frames_to_keep", "smoke_archives", "download_gb")}))
        for name, report in manifest["selections"].items():
            print(name, {s: (r["selected"], r.get("target"), r["logs"]) for s, r in report["splits"].items()},
                  f"~{report['images_gb_estimate']} GB (D)")
        if "label_conflicts" in manifest:
            print("label conflicts (report only):", json.dumps(manifest["label_conflicts"]["rows"]))
        if "scene_exclusion" in manifest:
            e = manifest["scene_exclusion"]
            print("scene exclusion:", json.dumps({k: e[k] for k in ("name", "window_s", "tokens_in_index",
                                                                     "tokens_not_in_index", "tokens_not_located", "train_pool_excluded",
                                                                     "base_train_dropped", "base_dev_in_window")}))
        if "navtrain" in manifest:
            nt = manifest["navtrain"]
            print("navtrain:", json.dumps({k: nt[k] for k in ("tokens", "left_after", "excluded_by_first_failing_rule",
                                                              "also_fail_a_later_rule")}))
            print("mix:", json.dumps({k: manifest["mix"][k] for k in ("base", "added_outside_gated", "added_gated",
                                                                      "union", "stay_in_union", "stay_kept", "selected")}))
    elif args.command == "fetch":
        if args.archives > SMOKE_ARCHIVES and not args.full:
            parser.error(f"more than {SMOKE_ARCHIVES} archives needs --full: the full run waits for Gate 1 (plan §6)")
        try:
            shard, num_shards = parse_shard(args.shard, args.num_shards)
        except ValueError as error:
            parser.error(str(error))
        if args.extract_workers > 1 and not args.source_dir:
            parser.error("--extract-workers needs --source-dir")
        result = fetch(args.out, args.archives, source_dir=args.source_dir, workers=args.workers,
                       shard=shard, num_shards=num_shards, storage=args.storage,
                       extract_workers=args.extract_workers)
        sys.exit(1 if result["failed"] else 0)
    else:
        status(args.out)


if __name__ == "__main__":
    main()
