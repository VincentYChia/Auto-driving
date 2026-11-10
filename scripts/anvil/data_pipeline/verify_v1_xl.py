"""Checks on the v1-xl build (``docs/training_data/PRETRAIN_V1_PLAN.md`` §10; the v1-xl spec's §8) [proposed].

Run from the repo root; ``build_v1_xl.sh setup`` and ``build_v1_xl.sbatch`` call it between steps:

    python scripts/anvil/data_pipeline/verify_v1_xl.py splits     --out <v1.0-xl> --large <v1.0-large>
    python scripts/anvil/data_pipeline/verify_v1_xl.py leakage    --out <v1.0-xl> [--navsim-root <devkit>] [--metadata <tgz>]
    python scripts/anvil/data_pipeline/verify_v1_xl.py fetched    --out <v1.0-xl> [--archives 200]
    python scripts/anvil/data_pipeline/verify_v1_xl.py repro      --out <smoke dir> --other <smoke dir>/repro
    python scripts/anvil/data_pipeline/verify_v1_xl.py crosscheck --out <v1.0-xl> --large <v1.0-large>
    python scripts/anvil/data_pipeline/verify_v1_xl.py canonical  --out <v1.0-xl>
    python scripts/anvil/data_pipeline/verify_v1_xl.py all        --out <v1.0-xl> --large <v1.0-large> \
        [--links $TRAINING_DATA_ROOT/pilot_v1] [--deep]

What is checked:
* ``splits``: the token files are sorted, unique, LF-terminated; dev is v1-pilot's (SHA-256 79090629…) in all
  three sets and equals v1.0-large's; (v1-large minus the warm-up window) ⊂ v1-xl-nd ⊂ v1-xl (train), with
  exactly ``BASE_DROPPED`` v1-large train rows outside v1-xl; the counts and train SHA-256 (§10, V); the copies
  of v1-large's files are byte-identical; the manifest's checksums and frame count; the committed warm-up
  token list (``WARMUP_FILE``); the label-conflict list (report only).
* ``leakage``: every selected token is in the index with the same split, no NAVSIM guard reason, no excluded
  log (the index's ``our_split``); train and dev logs disjoint; dev logs are nuPlan val. The warm-up
  exclusion [proposed 2026-11-09]: no v1-xl / v1-xl-nd TRAIN token is a ``warmup_test_e2e`` token or within
  the manifest's window (±7.0 s, same log) of one, located by the index and, for listed tokens that are not
  index rows, by OpenScene's metadata (``--metadata``: scanned here over the logs of every train row and listed
  row, and the result must equal the plan's ``located_in_metadata``; without ``--metadata`` that check fails,
  since the window cannot be fully checked); every v1-large train row left out of
  v1-xl is inside the window; dev's warm-up tokens (3) and v1-large's (57) are reported, not failed (dev is
  evaluation only and pinned; v1-large is an earlier set, kept as committed). With ``--navsim-root``, also
  against NAVSIM's own lists: ``warmup_test_e2e`` (NAVSIM v1.1) is held to the same scene-level rule (train
  clean, dev hits reported, its logs not guarded as a whole: an information check reports the train rows in
  them) and must equal the committed list; every other
  guarded list (nuPlan test logs, navtest, navhard, private_test_e2e, competition_private_tokens, …) must have
  0 tokens and 0 logs in train and dev of all three sets.
* ``fetched``: a clean record for each of the first ``--archives`` archives of the plan (no missing or
  undecodable frame, logs match the map) and its pack ``images/packs/<n>.zip`` holding exactly the record's
  frames, named by canonical ``image_path``, in sorted order.
* ``repro`` (smoke test): a second pack of the first archive has the same bytes.
* ``crosscheck``: every v1-large frame has the SHA-256 and archive it has in v1.0-large's records.
* ``canonical``: ``canonical.json`` covers the three sets, every check passed, the image check ran.
* ``all``: the above plus the two views: ``storage`` zip, jsonl counts and ids = the token files, zip members
  = the jsonl ``image`` fields, dev.jsonl byte-identical to the v1-large view's, and a 50-sample byte check of
  dev images against the v1-large view (and of the packed originals against their recorded SHA-256).
  ``--deep`` also runs ``ZipFile.testzip`` and re-hashes every pack and view zip (reads ~55 GB, D).

``--smoke`` (a build of the first few archives) turns the full-build counts into subset checks.
Each command writes ``<out>/verify/<command>.json`` and exits 0 when every check passes, 1 when one fails, and
2 when the step's output is not there yet (``fetched``, ``canonical``: the job then runs that step).
It only reads the build; it deletes, moves and rewrites nothing.

What a build holds and expects is a ``Build`` (``XL`` here); every check takes one, so ``verify_v1_pool.py``
(the v1.0-pool build, PRETRAIN_V1_PLAN.md §11) reuses them with its own.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import sys
import zipfile
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SETS = ("v1-xl", "v1-xl-nd", "v1-large")
VIEWS = ("v1-xl", "v1-xl-nd")
SPLITS = ("train", "dev")
DEV_SHA256 = "79090629e278705a066890efbfc8271acd9204e2085c3f3ddd8d668316ab7205"  # v1-pilot's = v1-large's dev (V)
EXPECTED = {  # V: the plan with the warm-up exclusion, 2026-11-09 (PRETRAIN_V1_PLAN.md §10); v1-large as committed
    "v1-xl": {"train": 200_000, "dev": 2_000,
              "train_sha256": "7f6eb97653c690c54a6a0e0a6101e8123efcea9a62f13989bec4c71fedec7643"},
    "v1-xl-nd": {"train": 176_352, "dev": 2_000,
                 "train_sha256": "5cb194088089ae1b997ef499cb58377e87fe778558ad5cd0e1cd9d9b455a0923"},
    "v1-large": {"train": 100_000, "dev": 2_000,
                 "train_sha256": "e6e6e7d320298ee71d8ae8bea03aa525042884cfa4e0bdc92541acf2ed6601ab"},
}
FRAMES_TO_KEEP = 203_727  # union of the three sets: v1-xl's 202,000 and the 1,727 v1-large train rows it leaves out
LARGE_FRAMES = 102_000
BASE_DROPPED = 1_727  # v1-large train rows inside the warm-up window, so not in v1-xl or v1-xl-nd (V, 2026-11-09)
WARMUP_FILE = "navsim_v1.1_warmup_test_e2e_tokens.txt"  # committed ID list of NAVSIM v1.1 warmup_test_e2e
WARMUP_SHA256 = "f5fdc4a169e53f0369d41f946125d7113cd208e4b8fac60a2d65a21353f0cea4"
WARMUP_TOKENS = 563
WARMUP_FILTER = "warmup_test_e2e"  # the devkit's scene filter held to the scene-level rule, not guarded by log
SCENE_LEVEL_FILTERS = (WARMUP_FILTER,)
WINDOW_S = 7.0  # default when the manifest does not say
MIN_WINDOW_S = 6.5  # our window [t - 1.5, t + 4] overlaps a v1.1 scene's [tw - 1.5, tw + 5] iff t - tw in [-5.5, 6.5]
TRAINED = ("v1-xl", "v1-xl-nd")  # the sets built here to be trained on
CAMERA = "CAM_F0"
VIEW = "pilot_v1"
NOT_YET = 2  # exit code: the step's output is not there yet

Check = Dict[str, Any]
Splits = Dict[str, Set[str]]  # set name -> its tokens of one split


class Build(NamedTuple):
    """What one build folder holds and what its checks expect.

    ``ref`` is the earlier build it is checked against: ``ref_sets``' frames it also packed must carry the same
    SHA-256 and archive (``crosscheck``; ``shared_frames`` of them in a full build), ``copied_set``'s token files
    are byte copies of ``ref``'s, and ``ref_view``'s dev.jsonl and dev images must equal each view's.
    ``relations(out_dir, train, dev, manifest)`` adds the build's own split checks (nesting, base, lists).
    ``left_out`` = (earlier set, set): every train row of the earlier set that the set leaves out is inside the
    warm-up window. ``reported`` sets are earlier sets whose warm-up hits are reported, not failed.
    """

    name: str
    sets: Tuple[str, ...]
    views: Tuple[str, ...]
    trained: Tuple[str, ...]
    expected: Dict[str, Dict[str, Any]]
    frames_to_keep: int
    conflicts_file: str
    ref: str
    ref_sets: Tuple[str, ...]
    ref_view: str
    copied_set: str
    shared_frames: int
    shared_check: str
    relations: Callable[[str, Splits, Splits, Dict[str, Any]], List[Check]]
    left_out: Optional[Tuple[str, str]] = None
    reported: Tuple[str, ...] = ()
    warmup_note: str = "dev is evaluation only and pinned"


def _words(n: int) -> str:
    return {2: "two", 3: "three", 4: "four"}.get(n, str(n))


# --- helpers -------------------------------------------------------------------------------------


def _check(ok: bool, what: str, value: Any = None) -> Check:
    """The form of ``canonical.json``'s checks."""
    return {"check": what, "ok": bool(ok), **({"value": value} if value is not None else {})}


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def token_file(out_dir: str, name: str, split: str) -> str:
    return os.path.join(out_dir, "splits", f"{name}_tokens_{split}.txt")


def read_tokens(path: str) -> List[str]:
    with open(path, encoding="utf-8", newline="") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def image_path(log: str, token: str) -> str:
    """Canonical ``image_path`` (``canonical.image_path``): the pack member name of an original."""
    return f"images/{CAMERA}/{log}/{token}.jpg"


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest(out_dir: str) -> Dict[str, Any]:
    return load_json(os.path.join(out_dir, "splits", "manifest.json"))


def records(out_dir: str, archives: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, Any]]:
    """Archive number → its step-3 record, for those that have one."""
    folder = os.path.join(out_dir, "images", "archives")
    if archives is None:
        archives = [int(f[:-5]) for f in os.listdir(folder) if f.endswith(".json")] if os.path.isdir(folder) else []
    out = {}
    for n in archives:
        path = os.path.join(folder, f"{n}.json")
        if os.path.exists(path):
            out[int(n)] = load_json(path)
    return out


def frame_records(out_dir: str) -> Dict[str, Dict[str, Any]]:
    """Token → {sha256, bytes, log, archive} over every record."""
    out = {}
    for n, record in records(out_dir).items():
        for image in record["images"]:
            out[image["token"]] = {**image, "archive": n}
    return out


def pack_path(out_dir: str, n: int) -> str:
    return os.path.join(out_dir, "images", "packs", f"{n}.zip")


def view_dir(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, "views", VIEW, name)


class ViewReader:
    """A view's images: the ``images.zip`` member if the view has one, else the loose file."""

    def __init__(self, folder: str):
        self.folder = folder
        zpath = os.path.join(folder, "images.zip")
        self.zip = zipfile.ZipFile(zpath) if os.path.exists(zpath) else None

    def read(self, rel: str) -> bytes:
        if self.zip is not None:
            return self.zip.read(rel)
        with open(os.path.join(self.folder, *rel.split("/")), "rb") as f:
            return f.read()

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()


def jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def token_of(item: Dict[str, Any]) -> str:
    return str(item["id"]).rsplit("/", 1)[-1]


# --- the v1-xl build -----------------------------------------------------------------------------


def _xl_relations(out_dir: str, train: Splits, dev: Splits, m: Dict[str, Any]) -> List[Check]:
    """v1-xl's own split checks: (v1-large minus the warm-up window) ⊂ v1-xl-nd ⊂ v1-xl, ``BASE_DROPPED`` left out."""
    exclusion = m.get("scene_exclusion") if isinstance(m.get("scene_exclusion"), dict) else {}
    kept_large = train["v1-large"] & train["v1-xl"]
    out = [_check(kept_large <= train["v1-xl-nd"] <= train["v1-xl"],
                  "train: (v1-large minus the warm-up window) ⊂ v1-xl-nd ⊂ v1-xl",
                  {"v1-large in v1-xl, not in v1-xl-nd": len(kept_large - train["v1-xl-nd"]),
                   "v1-xl-nd not in v1-xl": len(train["v1-xl-nd"] - train["v1-xl"])})]
    left_out = len(train["v1-large"] - train["v1-xl"])
    out.append(_check(left_out == BASE_DROPPED == exclusion.get("base_train_dropped"),
                      f"{BASE_DROPPED:,} v1-large train rows are left out of v1-xl (= manifest scene_exclusion), "
                      "the same in v1-xl-nd", {"left_out": left_out, "manifest": exclusion.get("base_train_dropped"),
                                               "left_out_of_nd": len(train["v1-large"] - train["v1-xl-nd"])}))
    out.append(_check(train["v1-large"] - train["v1-xl"] == train["v1-large"] - train["v1-xl-nd"],
                      "v1-xl-nd leaves out the same v1-large rows as v1-xl"))
    return out


XL = Build(name="v1-xl", sets=SETS, views=VIEWS, trained=TRAINED, expected=EXPECTED, frames_to_keep=FRAMES_TO_KEEP,
           conflicts_file="v1-xl_label_conflicts.tsv", ref="v1.0-large", ref_sets=("v1-large",), ref_view="v1-large",
           copied_set="v1-large", shared_frames=LARGE_FRAMES, shared_check="all {n:,} v1-large frames are packed",
           relations=_xl_relations, left_out=("v1-large", "v1-xl"), reported=("v1-large",),
           warmup_note="dev is evaluation only and pinned; v1-large is an earlier set kept as committed")


# --- splits and leakage --------------------------------------------------------------------------


def _manifest_shas(entry: Any) -> Dict[str, str]:
    """Split → SHA-256 from a manifest selection entry, in any of the shapes WP1 may write."""
    found: Dict[str, str] = {}
    if not isinstance(entry, dict):
        return found
    sha = entry.get("sha256")
    if isinstance(sha, dict):
        found.update({s: str(v) for s, v in sha.items() if s in SPLITS})
    for split in SPLITS:
        sub = entry.get(split)
        if isinstance(sub, dict) and isinstance(sub.get("sha256"), str):
            found[split] = sub["sha256"]
    return found


def warmup_path(out_dir: str) -> str:
    return os.path.join(out_dir, "splits", WARMUP_FILE)


def check_warmup_list(out_dir: str, exclusion: Dict[str, Any]) -> List[Check]:
    """The committed warm-up token list and the manifest's record of the exclusion."""
    path = warmup_path(out_dir)
    if not os.path.exists(path):
        return [_check(False, f"warm-up token list splits/{WARMUP_FILE} exists")]
    with open(path, "rb") as f:
        data = f.read()
    lines = read_tokens(path)
    sha = hashlib.sha256(data).hexdigest()
    window = exclusion.get("window_s")
    return [
        _check(b"\r" not in data and data.endswith(b"\n") and lines == sorted(set(lines)) and len(lines) == WARMUP_TOKENS
               and sha == WARMUP_SHA256, f"{WARMUP_FILE}: {WARMUP_TOKENS} tokens, sorted, unique, LF, "
               f"SHA-256 {WARMUP_SHA256[:8]}…", {"tokens": len(lines), "sha256": sha}),
        _check(exclusion.get("name") == WARMUP_FILE and exclusion.get("sha256") == sha
               and (exclusion.get("source") or {}).get("filter") == WARMUP_FILTER,
               "manifest scene_exclusion records this list and its source", exclusion.get("source")),
        _check(isinstance(window, (int, float)) and window >= MIN_WINDOW_S,
               f"exclusion window >= {MIN_WINDOW_S} s (the two sample windows overlap up to 6.5 s apart)", window),
    ]


def check_splits(out_dir: str, large_dir: Optional[str], build: Build = XL) -> List[Check]:
    """``large_dir``: the reference build's folder (``build.ref``), whose ``build.copied_set`` files ours copy."""
    out: List[Check] = []
    tokens: Dict[Tuple[str, str], List[str]] = {}
    for name in build.sets:
        for split in SPLITS:
            path = token_file(out_dir, name, split)
            if not os.path.exists(path):
                out.append(_check(False, f"{name} {split}: token file exists", path))
                continue
            with open(path, "rb") as f:
                data = f.read()
            lines = read_tokens(path)
            tokens[(name, split)] = lines
            out.append(_check(b"\r" not in data and data.endswith(b"\n") and lines == sorted(set(lines)),
                              f"{name} {split}: sorted, unique, LF, trailing newline"))
            want = build.expected[name][split]
            out.append(_check(len(lines) == want, f"{name} {split}: {want:,} tokens", len(lines)))
            sha = hashlib.sha256(data).hexdigest()
            want_sha = DEV_SHA256 if split == "dev" else build.expected[name]["train_sha256"]
            out.append(_check(sha == want_sha, f"{name} {split}: SHA-256 {want_sha[:8]}… (spec §3)", sha))
    if len(tokens) < 2 * len(build.sets):
        return out
    train = {name: set(tokens[(name, "train")]) for name in build.sets}
    dev = {name: set(tokens[(name, "dev")]) for name in build.sets}
    m = manifest(out_dir)
    exclusion = m.get("scene_exclusion") if isinstance(m.get("scene_exclusion"), dict) else {}
    out.extend(build.relations(out_dir, train, dev, m))
    words = _words(len(build.sets))
    out.append(_check(all(dev[name] == dev[build.sets[0]] for name in build.sets),
                      f"dev is the same 2,000 in all {words} sets"))
    out.append(_check(not any(train[name] & dev[name] for name in build.sets), "no token is in both train and dev"))
    union = set().union(*train.values(), *dev.values())
    out.append(_check(len(union) == build.frames_to_keep, f"the {words} sets hold {build.frames_to_keep:,} frames",
                      len(union)))

    if large_dir:
        same = {}
        for split in SPLITS:
            theirs, ours = token_file(large_dir, build.copied_set, split), token_file(out_dir, build.copied_set, split)
            same[split] = os.path.exists(theirs) and os.path.exists(ours) and sha256_file(theirs) == sha256_file(ours)
        out.append(_check(all(same.values()), f"{build.copied_set} token files are byte-identical to {build.ref}'s",
                          same))

    out.extend(check_warmup_list(out_dir, exclusion))
    out.append(_check(set(build.sets) <= set(m.get("selections", {})),
                      f"manifest selections list {', '.join(build.sets)}", sorted(m.get("selections", {}))))
    out.append(_check(m.get("frames_to_keep") == build.frames_to_keep,
                      f"manifest frames_to_keep = {build.frames_to_keep:,}", m.get("frames_to_keep")))
    order = m.get("archive_order", [])
    out.append(_check(len(order) == 200 and len(set(order)) == 200, "archive order lists 200 distinct archives",
                      len(order)))
    found, bad = 0, []
    for name in build.sets:
        for split, sha in _manifest_shas(m.get("selections", {}).get(name)).items():
            found += 1
            if sha != sha256_file(token_file(out_dir, name, split)):
                bad.append(f"{name} {split}")
    base = m.get("base") if isinstance(m.get("base"), dict) else {}
    for split, sha in _manifest_shas(base).items():
        found += 1
        if sha != sha256_file(token_file(out_dir, str(base.get("set", "v1-large")), split)):
            bad.append(f"base {split}")
    out.append(_check(found > 0 and not bad, "manifest SHA-256 entries match the token files",
                      {"entries": found, "mismatched": bad}))

    tsv = os.path.join(out_dir, "splits", build.conflicts_file)
    if os.path.exists(tsv):
        with open(tsv, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        by_set: Dict[str, int] = {}
        for row in rows:
            by_set[row[1] if len(row) > 1 else "?"] = by_set.get(row[1] if len(row) > 1 else "?", 0) + 1
        out.append(_check(header[:3] == ["token", "set", "reasons"] and all(r[0] in union for r in rows),
                          "label-conflict list: columns token, set, reasons; every token selected (report only)",
                          {"rows": len(rows), "by_set": by_set}))
    else:
        out.append(_check(False, f"label-conflict list splits/{build.conflicts_file} exists"))
    return out


def selected(out_dir: str, build: Build = XL) -> Dict[str, Dict[str, str]]:
    """Set → token → split."""
    return {name: {t: split for split in SPLITS for t in read_tokens(token_file(out_dir, name, split))}
            for name in build.sets}


def _nearest_s(times: Dict[str, List[int]], log: str, t_us: int) -> Optional[float]:
    """Seconds from ``t_us`` to the nearest warm-up scene of ``log`` (None: the log has none)."""
    centre = times.get(log)
    if not centre:
        return None
    k = bisect.bisect_left(centre, t_us)
    return min(abs(t_us - centre[j]) for j in (k - 1, k) if 0 <= j < len(centre)) / 1e6


def check_warmup(out_dir: str, sel: Dict[str, Dict[str, str]], cands: Dict[str, Any], row_of: Dict[str, int],
                 metadata: Optional[str] = None, build: Build = XL) -> List[Check]:
    """The scene-level warm-up exclusion, from the committed list and the index (module docstring)."""
    path = warmup_path(out_dir)
    if not os.path.exists(path):
        return [_check(False, f"warm-up token list splits/{WARMUP_FILE} exists")]
    listed = set(read_tokens(path))
    exclusion = manifest(out_dir).get("scene_exclusion") or {}
    window_s = float(exclusion.get("window_s", WINDOW_S))
    times: Dict[str, List[int]] = {}
    for t in listed:
        r = row_of.get(t)
        if r is not None:
            times.setdefault(str(cands["log_name"][r]), []).append(int(cands["t_us"][r]))
    by_metadata = exclusion.get("located_in_metadata") or {}  # token -> log, placed by the plan from the metadata
    missing = sorted(listed - set(row_of))  # listed tokens that are not index rows
    placed: Dict[str, Tuple[str, int]] = {}
    scanned = 0
    if metadata and missing:
        # Located here, not taken from the manifest: every log with a train row of any set or a listed row
        # (a frame in a log with no train row cannot be near one). ~1,100 logs, ~1 min locally (V, 2026-11-09).
        from data_pipeline.v1 import scene_exclusion

        logs = {str(cands["log_name"][row_of[t]]) for n in build.sets for t, s in sel[n].items()
                if s == "train" and t in row_of}
        logs |= {str(cands["log_name"][row_of[t]]) for t in listed if t in row_of}
        scanned = len(logs)
        placed = scene_exclusion.locate_in_metadata(metadata, missing, logs)
        for t, (log, ts) in placed.items():
            times.setdefault(log, []).append(ts)
    for v in times.values():
        v.sort()

    def near(tokens: Iterable[str]) -> List[str]:
        out = []
        for t in tokens:
            r = row_of.get(t)
            if r is None:
                continue
            d = _nearest_s(times, str(cands["log_name"][r]), int(cands["t_us"][r]))
            if d is not None and d <= window_s + 1e-9:
                out.append(t)
        return out

    split_tokens = {(n, s): [t for t, sp in sel[n].items() if sp == s] for n in build.sets for s in SPLITS}
    exact = {f"{n} {s}": sum(t in listed for t in ts) for (n, s), ts in split_tokens.items()}
    inside = {f"{n} {s}": len(near(ts)) for (n, s), ts in split_tokens.items()}
    trained = {n: inside[f"{n} train"] for n in build.trained}
    names = " / ".join(build.trained)
    out = [
        _check(all(exact[f"{n} train"] == 0 for n in build.trained),
               f"no {names} train token is a {WARMUP_FILTER} token (CLAUDE.md: never train on warmup)",
               {n: exact[f"{n} train"] for n in build.trained}),
        _check(all(v == 0 for v in trained.values()),
               f"no {names} train anchor within ±{window_s} s of a {WARMUP_FILTER} scene in its log",
               {n: inside[f"{n} train"] for n in build.trained}),
    ]
    if build.left_out:
        earlier, later = build.left_out
        left_out = set(sel[earlier]) - set(sel[later])
        left_out_inside = set(near(left_out))
        unplaced_logs = set() if placed else set(by_metadata.values())  # windows this run cannot place
        unchecked = {t for t in left_out - left_out_inside if str(cands["log_name"][row_of[t]]) in unplaced_logs}
        out.append(_check(len(left_out_inside) + len(unchecked) == len(left_out),
                          f"every {earlier} train row left out of {later} is inside the window",
                          {"left_out": len(left_out), "inside": len(left_out_inside),
                           "near_frames_placed_only_with_metadata": len(unchecked)}))
    out.append(_check(True, f"information: {WARMUP_FILTER} tokens and rows within ±{window_s} s per set and split "
                            f"({build.warmup_note})",
                      {"tokens": exact, "within_window": inside, "expected_dev_tokens": 3}))
    if missing:  # without --metadata they cannot be placed, so the window checks above are incomplete: fail
        found = {t: log for t, (log, _) in placed.items()}
        out.append(_check(bool(metadata) and found == dict(by_metadata),
                          f"{len(missing)} listed tokens that are not index rows: "
                          + ("located independently in the metadata (logs of every train row and listed row), "
                             "included above, and = the plan's located_in_metadata" if metadata else
                             "NOT located, so the window is not fully checked (pass --metadata)"),
                          {"not_in_index": len(missing), "logs_scanned": scanned, "placed": len(found),
                           "listed_by_plan": len(by_metadata),
                           "differ": sorted(set(found.items()) ^ set(by_metadata.items()))}))
    return out


def check_leakage(out_dir: str, navsim_root: Union[None, str, Sequence[str]] = None, metadata: Optional[str] = None,
                  build: Build = XL) -> List[Check]:
    """``navsim_root``: one devkit or several (each is checked; with several, each check names its devkit)."""
    from data_pipeline.v1 import tables

    sel = selected(out_dir, build)
    split_of: Dict[str, str] = {}
    for name in build.sets:
        split_of.update(sel[name])
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"),
                        ["token", "log_name", "our_split", "guard_reason", "t_us"])
    row_of = {str(t): k for k, t in enumerate(cands["token"])}
    rows = [row_of.get(t) for t in split_of]
    absent = sum(r is None for r in rows)
    wrong_split = guarded = 0
    log_split: Dict[str, Set[str]] = {"train": set(), "dev": set()}
    for t, r in zip(split_of, rows):
        if r is None:
            continue
        if str(cands["our_split"][r]) != split_of[t]:
            wrong_split += 1
        if str(cands["guard_reason"][r]):
            guarded += 1
        log_split[split_of[t]].add(str(cands["log_name"][r]))
    logs = tables.read(os.path.join(out_dir, "index", "logs.parquet"),
                       ["log_name", "our_split", "guard_reason", "nuplan_split"])
    log_info = {str(l): (str(s), str(g), str(n)) for l, s, g, n in
                zip(logs["log_name"], logs["our_split"], logs["guard_reason"], logs["nuplan_split"])}
    all_logs = log_split["train"] | log_split["dev"]
    # A log missing from logs.parquet counts as excluded too.
    excluded = sorted(l for l in all_logs if l not in log_info or log_info[l][0] == "excluded" or log_info[l][1])
    out = [
        _check(absent == 0, "every selected token is in the index", absent),
        _check(wrong_split == 0, "every token's split is the index's our_split (rule 4)", wrong_split),
        _check(guarded == 0, "no token has a NAVSIM guard reason (rule 5)", guarded),
        _check(not excluded, "no selected log is excluded in the index (NAVSIM evaluation logs, rule 5)", len(excluded)),
        _check(not (log_split["train"] & log_split["dev"]), "no log is in both train and dev",
               len(log_split["train"] & log_split["dev"])),
        _check(all(log_info.get(l, ("", "", ""))[2] == "val" for l in log_split["dev"]), "dev logs are nuPlan val logs",
               {"train_logs": len(log_split["train"]), "dev_logs": len(log_split["dev"])}),
    ]
    out += check_warmup(out_dir, sel, cands, row_of, metadata, build)
    roots = [navsim_root] if isinstance(navsim_root, str) else [r for r in (navsim_root or []) if r]
    for root in roots:
        checks = check_devkit(out_dir, sel, log_split, cands, row_of, root, build)
        if len(roots) > 1:  # say which devkit each check is about
            label = os.path.basename(os.path.normpath(root))
            checks = [{**c, "check": f"{c['check']} [{label}]"} for c in checks]
        out += checks
    return out


def check_devkit(out_dir: str, sel: Dict[str, Dict[str, str]], log_split: Dict[str, Set[str]],
                 cands: Dict[str, Any], row_of: Dict[str, int], navsim_root: str, build: Build = XL) -> List[Check]:
    """NAVSIM's own lists: scene-level filters (warm-up) by the window rule, every other guarded list at 0 hits.

    ``SCENE_LEVEL_FILTERS`` are exempt from the log-level guard: their logs hold train rows by design (outside the
    window of every listed scene), so those rows and logs are reported, not failed, and the 0-hit check is over the
    other lists only. ``navsim_splits.load`` itself guards a scene-level filter's logs as a whole.
    """
    from data_pipeline.v1 import navsim_splits

    guard = navsim_splits.load(navsim_root)

    def strict(reason: str) -> bool:  # guarded by something other than a scene-level filter
        return any(r not in SCENE_LEVEL_FILTERS for r in reason.split("+"))

    strict_tokens = {t for t, r in guard.guarded_tokens.items() if strict(r)}
    strict_logs = {l for l, r in guard.guarded_logs.items() if strict(r)}
    all_tokens = {t for name in build.sets for t in sel[name]}
    all_logs = log_split["train"] | log_split["dev"]
    hits: Dict[str, int] = {}
    for t in all_tokens & strict_tokens:
        for r in guard.guarded_tokens[t].split("+"):
            hits[f"token:{r}"] = hits.get(f"token:{r}", 0) + 1
    for l in all_logs & strict_logs:
        for r in guard.guarded_logs[l].split("+"):
            hits[f"log:{r}"] = hits.get(f"log:{r}", 0) + 1
    scene_level = ", ".join(SCENE_LEVEL_FILTERS)
    out = [_check(not hits, f"no token or log of any set is in NAVSIM's other guarded lists (every list but the "
                            f"scene-level {scene_level}: nuPlan test, navtest, navhard, private tests, ...), train or dev",
                  {"guarded_logs": len(strict_logs), "guarded_tokens": len(strict_tokens), "hits": hits,
                   "exempt_scene_level": list(SCENE_LEVEL_FILTERS),
                   "guarded_logs_with_scene_level": len(guard.guarded_logs),
                   "guarded_tokens_with_scene_level": len(guard.guarded_tokens),
                   "devkit_commit": guard.devkit_commit})]
    warm = {t for t, r in guard.guarded_tokens.items() if WARMUP_FILTER in r.split("+")}
    if warm:
        committed = set(read_tokens(warmup_path(out_dir))) if os.path.exists(warmup_path(out_dir)) else set()
        train_hits = {n: sum(t in warm for t, s in sel[n].items() if s == "train") for n in build.trained}
        reported = {f"{n} train": sum(t in warm for t, s in sel[n].items() if s == "train") for n in build.reported}
        out.append(_check(warm == committed, f"the devkit's {WARMUP_FILTER} tokens = the committed list",
                          {"devkit": len(warm), "committed": len(committed)}))
        out.append(_check(not any(train_hits.values()),
                          f"devkit {WARMUP_FILTER}: no {' / '.join(build.trained)} train token "
                          f"(dev{''.join(' and ' + n for n in build.reported)} reported)",
                          {"train": train_hits, **reported,
                           "dev": sum(t in warm for t, s in sel[build.sets[0]].items() if s == "dev"),
                           "logs_listed_not_guarded_by_log": sum(WARMUP_FILTER in r.split("+")
                                                                 for r in guard.guarded_logs.values())}))
        scene_logs = {l for l, r in guard.guarded_logs.items() if not strict(r)}  # only scene-level reasons

        def in_scene_logs(name: str, split: str) -> List[str]:
            return [str(cands["log_name"][row_of[t]]) for t, s in sel[name].items()
                    if s == split and t in row_of and str(cands["log_name"][row_of[t]]) in scene_logs]

        rows = {name: in_scene_logs(name, "train") for name in build.trained}
        out.append(_check(True, f"information: {scene_level} is held to the scene-level rule (SCENE_LEVEL_FILTERS: no "
                                "train anchor within the window of a listed scene), not by log, so train rows in its "
                                "logs are reported here, not failed (the log-level choice is open)",
                          {"scene_level_logs": len(scene_logs),
                           "train_rows_in_them": {n: len(v) for n, v in rows.items()},
                           "train_logs_in_them": {n: len(set(v)) for n, v in rows.items()},
                           "dev_rows_in_them": len(in_scene_logs(build.sets[0], "dev"))}))
    return out


# --- packs ---------------------------------------------------------------------------------------


def check_pack(out_dir: str, n: int, record: Dict[str, Any], deep: bool) -> List[Check]:
    path = pack_path(out_dir, n)
    if not os.path.exists(path):
        return [_check(False, f"archive {n}: pack images/packs/{n}.zip exists")]
    want = [image_path(i["log"], i["token"]) for i in record["images"]]
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        ok = names == sorted(want) and all(i.compress_type == zipfile.ZIP_STORED for i in infos)
        bad_zip = z.testzip() if deep else None
    count = record.get("pack_members", record.get("kept"))
    out = [_check(ok and len(names) == count, f"archive {n}: pack members = the record's frames, sorted, stored",
                  {"members": len(names), "kept": record.get("kept")})]
    if "pack_bytes" in record:
        out.append(_check(os.path.getsize(path) == record["pack_bytes"], f"archive {n}: pack size = record"))
    if deep:
        out.append(_check(bad_zip is None, f"archive {n}: testzip", bad_zip))
        if "pack_sha256" in record:
            out.append(_check(sha256_file(path) == record["pack_sha256"], f"archive {n}: pack SHA-256 = record"))
    return out


def check_fetched(out_dir: str, n_archives: int, full: bool, deep: bool = False,
                  build: Build = XL) -> Tuple[List[Check], bool]:
    """Checks, and whether every record is there yet."""
    order = manifest(out_dir)["archive_order"][:n_archives]
    recs = records(out_dir, order)
    left = [n for n in order if n not in recs]
    out = [_check(not left, f"a record for each of the first {len(order)} archives", {"left": left[:20], "n_left": len(left)})]
    problems = []
    for n in order:
        if n not in recs:
            continue
        r = recs[n]
        missing = r["missing"] if isinstance(r["missing"], list) else [None] * int(r["missing"])
        undecodable = r["undecodable"] if isinstance(r["undecodable"], list) else [None] * int(r["undecodable"])
        if missing or undecodable or not r.get("logs_match_map") or not r.get("sha256_ok") \
                or r["kept"] != r["wanted"] or r["kept"] != len(r["images"]):
            problems.append(n)
        out.extend(c for c in check_pack(out_dir, n, r, deep) if not c["ok"] or deep)
    out.append(_check(not problems, "records: 0 missing, 0 undecodable, logs match the map, kept = wanted",
                      problems[:20]))
    kept = sum(r["kept"] for r in recs.values())
    packs_ok = all(c["ok"] for c in out if c["check"].startswith("archive "))
    out.append(_check(packs_ok, "every pack holds exactly its record's frames", {"packs": len(recs), "frames": kept}))
    if full and not left:
        out.append(_check(kept == build.frames_to_keep, f"{build.frames_to_keep:,} frames kept", kept))
        sel = selected(out_dir, build)
        union = set().union(*(set(s) for s in sel.values()))
        got = {i["token"] for r in recs.values() for i in r["images"]}
        out.append(_check(got == union, f"the packed frames are exactly the {_words(len(build.sets))} sets' frames",
                          {"not packed": len(union - got), "not selected": len(got - union)}))
    return out, not left


def check_repro(out_dir: str, other_dir: str) -> List[Check]:
    n = manifest(out_dir)["archive_order"][0]
    a, b = records(out_dir, [n]).get(n), records(other_dir, [n]).get(n)
    if a is None or b is None:
        return [_check(False, f"archive {n}: packed in both folders")]
    same_frames = [(i["token"], i["sha256"]) for i in a["images"]] == [(i["token"], i["sha256"]) for i in b["images"]]
    sha_a, sha_b = sha256_file(pack_path(out_dir, n)), sha256_file(pack_path(other_dir, n))
    return [_check(same_frames, f"archive {n}: same frames and SHA-256 in both records"),
            _check(sha_a == sha_b, f"archive {n}: the two packs are byte-identical (reproducible)", sha_a)]


def check_crosscheck(out_dir: str, large_dir: str, full: bool, build: Build = XL) -> List[Check]:
    """``large_dir``: the reference build's folder (``build.ref``): the frames both packed must agree."""
    ours = frame_records(out_dir)
    theirs = frame_records(large_dir)
    ref_tokens = {t for name in build.ref_sets for split in SPLITS
                  for t in read_tokens(token_file(large_dir, name, split))}
    shared = [t for t in sorted(ref_tokens) if t in ours]
    absent = [t for t in ref_tokens if t not in theirs]
    sha_bad = [t for t in shared if t in theirs and ours[t]["sha256"] != theirs[t]["sha256"]]
    archive_bad = [t for t in shared if t in theirs and ours[t]["archive"] != theirs[t]["archive"]]
    ref_sets = " / ".join(build.ref_sets)
    out = [
        _check(not absent, f"{build.ref} has a record for every {ref_sets} frame", len(absent)),
        _check(not sha_bad, f"shared frames: SHA-256 = {build.ref}'s record",
               {"compared": len(shared), "mismatched": len(sha_bad)}),
        _check(not archive_bad, f"shared frames: same source archive as in {build.ref}", len(archive_bad)),
    ]
    if full:
        out.append(_check(len(shared) == build.shared_frames, build.shared_check.format(n=build.shared_frames),
                          len(shared)))
    else:
        out.append(_check(len(shared) > 0, f"some {ref_sets} frames are packed (smoke)", len(shared)))
    return out


def check_canonical(out_dir: str, full: bool, build: Build = XL) -> Tuple[List[Check], bool]:
    """Checks, and whether canonical.json is there for the build's sets."""
    path = os.path.join(out_dir, "canonical.json")
    if not os.path.exists(path):
        return [_check(False, "canonical.json exists")], False
    report = load_json(path)
    if set(report.get("sets", {})) != set(build.sets):
        return [_check(False, f"canonical.json covers {', '.join(build.sets)}", sorted(report.get("sets", {})))], False
    failed = [c["check"] for c in report["checks"] if not c["ok"]]
    out = [_check(not failed, "every canonical check passed", failed),
           _check(any(c["check"].startswith("every image file matches") for c in report["checks"]),
                  "the image SHA-256 check ran (not --no-verify-images)")]
    if full:
        counts = {name: report["sets"][name] for name in build.sets}
        out.append(_check(all(c["missing_image"] == 0 for c in counts.values()), "no selected frame lacks its image",
                          {n: c["missing_image"] for n, c in counts.items()}))
        out.append(_check(all(counts[n]["by_split"] == {"train": build.expected[n]["train"],
                                                       "dev": build.expected[n]["dev"]} for n in build.sets),
                          "canonical counts per set and split = spec §3", {n: c["by_split"] for n, c in counts.items()}))
        out.append(_check(report.get("frames") == build.frames_to_keep, f"{build.frames_to_keep:,} canonical frames",
                          report.get("frames")))
    return out, True


# --- views ---------------------------------------------------------------------------------------


def check_view(out_dir: str, name: str, large_view: str, full: bool, samples: int, seed: int,
               deep: bool, build: Build = XL) -> List[Check]:
    """``large_view``: the reference view (``build.ref_view``), whose dev.jsonl and dev images ours must equal."""
    ref = build.ref_view
    folder = view_dir(out_dir, name)
    spec_path = os.path.join(folder, "view.json")
    if not os.path.exists(spec_path):
        return [_check(False, f"view {name}: view.json exists")]
    spec = load_json(spec_path)
    large_spec = load_json(os.path.join(large_view, "view.json"))
    out = [_check(spec.get("storage") == "zip" and spec.get("set") == name, f"view {name}: storage zip, set {name}",
                  spec.get("storage")),
           _check(spec.get("prompt_version") == large_spec.get("prompt_version"),
                  f"view {name}: prompt version = the {ref} view's", spec.get("prompt_version"))]
    items = {split: jsonl(os.path.join(folder, f"{split}.jsonl")) for split in SPLITS}
    for split in SPLITS:
        ids = {token_of(i) for i in items[split]}
        want = set(read_tokens(token_file(out_dir, name, split)))
        ok = ids == want if full else ids <= want
        out.append(_check(ok and len(ids) == len(items[split]),
                          f"view {name} {split}.jsonl: one line per token of the set", len(items[split])))
    images = {i["image"] for split in SPLITS for i in items[split]}
    zpath = os.path.join(folder, "images.zip")
    if not os.path.exists(zpath):
        out.append(_check(False, f"view {name}: images.zip exists"))
        return out
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        bad_zip = z.testzip() if deep else None
    meta = spec.get("images_zip", {})
    out.append(_check(set(names) == images and len(names) == len(images) == meta.get("members"),
                      f"view {name}: zip members = the jsonl images = view.json", {"members": len(names)}))
    out.append(_check(os.path.getsize(zpath) == meta.get("bytes"), f"view {name}: zip size = view.json"))
    if deep:
        out.append(_check(bad_zip is None, f"view {name}: testzip", bad_zip))
        out.append(_check(sha256_file(zpath) == meta.get("sha256"), f"view {name}: zip SHA-256 = view.json"))

    large_dev = os.path.join(large_view, "dev.jsonl")
    ours_dev = os.path.join(folder, "dev.jsonl")
    if full:
        out.append(_check(sha256_file(ours_dev) == sha256_file(large_dev),
                          f"view {name}: dev.jsonl byte-identical to the {ref} view's", sha256_file(ours_dev)))
    else:
        with open(large_dev, encoding="utf-8") as f:
            theirs = set(f)
        with open(ours_dev, encoding="utf-8") as f:
            ours = list(f)
        out.append(_check(all(line in theirs for line in ours), f"view {name}: every dev.jsonl line is in the {ref} view's",
                          len(ours)))

    dev = sorted(items["dev"], key=lambda i: i["id"])
    pick = random.Random(seed).sample(dev, min(samples, len(dev)))
    ours_r, theirs_r = ViewReader(folder), ViewReader(large_view)
    try:
        differ = [i["id"] for i in pick if ours_r.read(i["image"]) != theirs_r.read(i["image"])]
    finally:
        ours_r.close()
        theirs_r.close()
    out.append(_check(bool(pick) and not differ, f"view {name}: {len(pick)} dev images byte-identical to the {ref} view's",
                      {"differ": differ[:5]}))

    frames = frame_records(out_dir)
    bad = []
    for i in pick:
        t = token_of(i)
        f = frames.get(t)
        if f is None:
            bad.append(t)
            continue
        with zipfile.ZipFile(pack_path(out_dir, f["archive"])) as z:
            if hashlib.sha256(z.read(image_path(f["log"], t))).hexdigest() != f["sha256"]:
                bad.append(t)
    out.append(_check(bool(pick) and not bad, f"packs: the same {len(pick)} originals read back with their SHA-256",
                      bad[:5]))
    return out


def check_links(out_dir: str, links_root: str, build: Build = XL) -> List[Check]:
    out = []
    for name in build.views:
        link = os.path.join(links_root, name)
        ok = os.path.islink(link) and os.path.realpath(link) == os.path.realpath(view_dir(out_dir, name))
        out.append(_check(ok, f"{link} → the {name} view"))
    return out


# --- main ----------------------------------------------------------------------------------------


def write_report(out_dir: str, command: str, checks: List[Check]) -> str:
    folder = os.path.join(out_dir, "verify")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{command}.json")
    from data_pipeline.v1 import provenance

    report = {"command": command, "created_utc": provenance.utc_now(),
              "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
              "ok": all(c["ok"] for c in checks), "checks": checks}
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(path + ".part", path)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Checks on the v1-xl build (PRETRAIN_V1_PLAN.md §10).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("splits", "leakage", "fetched", "repro", "crosscheck", "canonical", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the v1.0-xl dataset directory")
        p.add_argument("--smoke", action="store_true", help="a build of the first few archives: subset checks")
        p.add_argument("--quiet", action="store_true", help="print only the summary line")
        if name in ("splits", "crosscheck", "all"):
            p.add_argument("--large", required=name != "splits", help="the v1.0-large dataset directory")
        if name in ("leakage", "all"):
            p.add_argument("--navsim-root", help="NAVSIM devkit: also check against its own split lists")
            p.add_argument("--metadata", help="openscene_metadata_trainval.tgz: locate the listed warm-up tokens that "
                                              "are not index rows (required when there are any, else leakage fails)")
        if name in ("fetched", "all"):
            p.add_argument("--archives", type=int, default=200, help="the first N archives of the plan")
            p.add_argument("--deep", action="store_true", help="testzip and re-hash every pack (and view zip)")
        if name == "repro":
            p.add_argument("--other", required=True, help="a second build folder that packed the first archive")
        if name == "all":
            p.add_argument("--large-view", help="default: <large>/views/pilot_v1/v1-large")
            p.add_argument("--links", help="folder of the pilot_v1/<set> links, e.g. $TRAINING_DATA_ROOT/pilot_v1")
            p.add_argument("--samples", type=int, default=50)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    full = not args.smoke
    status = 0
    if args.command == "splits":
        checks = check_splits(args.out, args.large)
    elif args.command == "leakage":
        checks = check_leakage(args.out, args.navsim_root, args.metadata)
    elif args.command == "fetched":
        checks, complete = check_fetched(args.out, args.archives, full, args.deep)
        status = 0 if complete else NOT_YET
    elif args.command == "repro":
        checks = check_repro(args.out, args.other)
    elif args.command == "crosscheck":
        checks = check_crosscheck(args.out, args.large, full)
    elif args.command == "canonical":
        checks, present = check_canonical(args.out, full)
        status = 0 if present else NOT_YET
    else:
        large_view = args.large_view or view_dir(args.large, "v1-large")
        checks = check_splits(args.out, args.large) + check_leakage(args.out, args.navsim_root, args.metadata)
        checks += check_fetched(args.out, args.archives, full, args.deep)[0]
        checks += check_crosscheck(args.out, args.large, full) + check_canonical(args.out, full)[0]
        for name in VIEWS:
            checks += check_view(args.out, name, large_view, full, args.samples, args.seed, args.deep)
        if args.links:
            checks += check_links(args.out, args.links)
    path = write_report(args.out, args.command, checks)
    failed = [c for c in checks if not c["ok"]]
    if not args.quiet or (failed and status != NOT_YET):
        for c in failed:
            print(f"[verify] FAILED: {c['check']}" + (f": {json.dumps(c['value'])}" if "value" in c else ""))
    print(f"[verify] {args.command}: {len(checks) - len(failed)} of {len(checks)} checks pass ({path})")
    if status == NOT_YET:
        sys.exit(NOT_YET)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
