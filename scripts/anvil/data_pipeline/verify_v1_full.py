"""Checks on the v1.0-full build (``docs/training_data/PRETRAIN_V1_PLAN.md`` §12) [proposed].

Run from the repo root; ``build_v1_full.sh setup`` and ``build_v1_full.sbatch`` call it between steps:

    python scripts/anvil/data_pipeline/verify_v1_full.py splits     --out <v1.0-full> [--pool <v1.0-pool>]
    python scripts/anvil/data_pipeline/verify_v1_full.py rules      --out <v1.0-full> --metadata <tgz>
    python scripts/anvil/data_pipeline/verify_v1_full.py leakage    --out <v1.0-full> --metadata <tgz> \
        [--navsim-root <devkit>]...
    python scripts/anvil/data_pipeline/verify_v1_full.py fetched    --out <v1.0-full>
    python scripts/anvil/data_pipeline/verify_v1_full.py crosscheck --out <v1.0-full> --pool <v1.0-pool>
    python scripts/anvil/data_pipeline/verify_v1_full.py canonical  --out <v1.0-full>
    python scripts/anvil/data_pipeline/verify_v1_full.py all        --out <v1.0-full> --pool <v1.0-pool> --metadata <tgz> \
        [--navsim-root <devkit>]... [--links $TRAINING_DATA_ROOT/pilot_v1] [--deep]

The checks of ``verify_v1_xl.py`` with ``FULL`` (its ``resplit``: a token's split is its file's, since dev was
re-split and 185 val logs are released to training) on the two sets: v1-full (675,701 train, dev = dev-v2's 2,000)
and v1-full-devlogs (every row of dev-v2's 40 held-out val logs that passes the rules, 14,828, as dev; no train
rows); 690,529 packed frames, of which 428,645 v1.0-pool also packed. Plus:
* ``splits`` (``_relations``): the copy of v1.0-pool's v1-all (SHA-256 92081c07…) lies inside v1-full's train;
  ``dev-v1_tokens.txt`` is the old pinned dev byte for byte (SHA-256 79090629…, kept for the record); dev-v2 lies
  inside v1-full-devlogs; ``dev-v2_logs.txt`` lists the 40 held-out logs (SHA-256 eff9eb2e…), as the manifest does.
* ``rules``: everything recomputed from the index, the warm-up list and OpenScene's metadata: the held-out logs
  are whole nuPlan val drives with navtest's city mix per log count, and ``token_sets.choose_dev_logs`` picks
  exactly them; v1-full's train = every eligible row outside ±7 s of every warm-up scene in the TRAIN logs and the
  released val logs (nothing thinned), v1-full-devlogs = those of the held-out logs; dev-v2's anchors are >= 1.0 s
  apart, per city navtest's share, each a navtrain token or a row with a known route command, and the draw pinned
  in ``token_sets.DEV_DRAW`` reproduces them; the counts per rule = the manifest's.
* ``leakage``: ``verify_v1_xl``'s over both sets against every devkit given (``--navsim-root``, repeatable: NAVSIM
  v1.1 and main): no token or log of either set in a guarded list except the scene-level warm-up filter
  (``SCENE_LEVEL_FILTERS``: no warm-up token and no train anchor within ±7 s of a warm-up scene; the warm-up logs that
  hold train rows are reported, not failed, as for v1.0-pool); and no held-out log or drive in any train set.
* ``all`` also checks both views of v1-full: pilot_v1.1 (``views/pilot_v1/v1-full``: the token files' samples with
  their split, images re-made from the packs, sampled train lines identical to v1.0-pool's v1-all view) and
  pilot_v1.2 (``views/pilot_v1.2/v1-full``: line for line the pilot_v1.1 samples, only the prompt differs); that no
  view's train.jsonl holds a held-out row, and the links ``pilot_v1/v1-full`` and ``pilot_v1/v1-full-p12``.

Each command writes ``<out>/verify/<command>.json`` and exits 0 when every check passes, 1 when one fails, and 2 when
the step's output is not there yet (``fetched``, ``canonical``). It only reads; it deletes, moves and rewrites nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import zipfile
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import verify_v1_xl as xl  # noqa: E402  (the shared checks; it puts the repo root on sys.path)

Check = xl.Check
_check = xl._check
FULL, DEVLOGS = "v1-full", "v1-full-devlogs"
SETS = (FULL, DEVLOGS)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED = {  # V: the committed plan, 2026-11-10 (PRETRAIN_V1_PLAN.md §12)
    FULL: {"train": 675_701, "dev": 2_000,
           "train_sha256": "759fb537d3df10c4dfafa06445de0c74f5c5131147e36fa002e5e7d76d724115",
           "dev_sha256": "0661d4815bd0cae50f1d3495ed1b024d2e40711965197603b9b1d593d36eb951"},  # dev-v2
    DEVLOGS: {"train": 0, "dev": 14_828, "train_sha256": EMPTY_SHA256,
              "dev_sha256": "2e9203dd3ea86034aa4ea84e42164a17de4373b32de16ce0d5ebd9fce9be44e0"},
}
FRAMES_TO_KEEP = 690_529  # v1-full's train and the held-out rows (dev-v2 is inside them)
SHARED_FRAMES = 428_645  # v1.0-pool's v1-all train and dev-v1, but dev-v1's 49 rows inside the warm-up window
BASE = "v1-all"  # v1.0-pool's, copied into splits/ (not a selection)
BASE_TRAIN = 426_694
BASE_TRAIN_SHA256 = "92081c07454b747fb7a120dfec15c813f8b8b4a2969c28fd10fb5bf7826da683"
DEV_V1_FILE = "dev-v1_tokens.txt"  # openscene_images.DEV_V1_FILE: the old pinned dev, = xl.DEV_SHA256
DEV_V2_LOGS_FILE = "dev-v2_logs.txt"  # openscene_images.DEV_V2_LOGS_FILE
DEV_V2_LOGS = 40
DEV_V2_LOGS_SHA256 = "eff9eb2e884858bcc602b2461eb3017738fa04b2267293690da2743232815ed4"
VIEWS = {"pilot_v1.1": ("pilot_v1", ""), "pilot_v1.2": ("pilot_v1.2", "-p12")}  # version -> (views/ root, link suffix)
SAME_FIELDS = ("id", "image", "system", "target", "target_xy", "target_yaw", "meta")  # all but the prompt


def held_out_logs(out_dir: str) -> List[str]:
    path = os.path.join(out_dir, "splits", DEV_V2_LOGS_FILE)
    return [log.strip() for log in xl.read_tokens(path)] if os.path.exists(path) else []  # splits fails a CRLF file


def _relations(out_dir: str, train: xl.Splits, dev: xl.Splits, m: Dict[str, Any]) -> List[Check]:
    """v1.0-full's own split checks: the base copy, dev-v1 and dev-v2's files, the held-out rows apart from training."""
    shas, base = {}, {}
    for split in xl.SPLITS:
        path = xl.token_file(out_dir, BASE, split)
        shas[split] = xl.sha256_file(path) if os.path.exists(path) else None
        base[split] = set(xl.read_tokens(path)) if shas[split] else set()
    out = [
        _check(shas == {"train": BASE_TRAIN_SHA256, "dev": xl.DEV_SHA256} and len(base["train"]) == BASE_TRAIN,
               f"base {BASE}: the copy of v1.0-pool's token files ({BASE_TRAIN:,} train, SHA-256 {BASE_TRAIN_SHA256[:8]}…; "
               "the old pinned dev)", {"sha256": shas, "train": len(base["train"])}),
        _check(bool(base["train"]) and base["train"] <= train[FULL],
               f"every {BASE} train row is in {FULL}'s train (every train frame v1.0-pool extracted is extracted again)",
               {"not in v1-full": len(base["train"] - train[FULL]), "new train rows": len(train[FULL] - base["train"])}),
        _check(not train[DEVLOGS] and not (dev[DEVLOGS] & train[FULL]) and dev[FULL] <= dev[DEVLOGS],
               f"{DEVLOGS} has no train rows and no token of {FULL}'s train; dev-v2 ({FULL}'s dev) lies inside it",
               {"train": len(train[DEVLOGS]), "in v1-full train": len(dev[DEVLOGS] & train[FULL]),
                "dev-v2 not in it": len(dev[FULL] - dev[DEVLOGS])}),
    ]
    path = os.path.join(out_dir, "splits", DEV_V1_FILE)
    sha = xl.sha256_file(path) if os.path.exists(path) else None
    v1 = set(xl.read_tokens(path)) if sha else set()
    entry = m.get("dev_v1") if isinstance(m.get("dev_v1"), dict) else {}
    out.append(_check(sha == xl.DEV_SHA256 == entry.get("sha256") and v1 == base["dev"],
                      f"{DEV_V1_FILE}: the old pinned dev, byte for byte (SHA-256 {xl.DEV_SHA256[:8]}…), kept for the "
                      "record", {"sha256": sha, f"in {FULL} train": len(v1 & train[FULL]),
                                 "manifest": {k: entry.get(k) for k in (f"in_{FULL}_train", "in_held_out_logs")}}))
    path = os.path.join(out_dir, "splits", DEV_V2_LOGS_FILE)
    data = open(path, "rb").read() if os.path.exists(path) else b""
    lines = held_out_logs(out_dir)
    recorded = ((m.get("dev_v2") or {}).get("logs") or {}).get("logs")
    out.append(_check(b"\r" not in data and data.endswith(b"\n") and lines == sorted(set(lines))
                      and len(lines) == DEV_V2_LOGS == recorded and hashlib.sha256(data).hexdigest() == DEV_V2_LOGS_SHA256,
                      f"{DEV_V2_LOGS_FILE}: {DEV_V2_LOGS} held-out val logs, sorted, unique, LF, SHA-256 "
                      f"{DEV_V2_LOGS_SHA256[:8]}…, = the manifest's", {"logs": len(lines), "manifest": recorded,
                                                                     "sha256": hashlib.sha256(data).hexdigest()}))
    dev_v2 = m.get("dev_v2") if isinstance(m.get("dev_v2"), dict) else {}
    out.append(_check(dev_v2.get("file") == f"{FULL}_tokens_dev.txt"
                      and dev_v2.get("sha256") == xl.sha256_file(xl.token_file(out_dir, FULL, "dev")),
                      f"manifest dev_v2: {FULL}'s dev file and its SHA-256"))
    base_entry = m.get("base") if isinstance(m.get("base"), dict) else {}
    out.append(_check(base_entry.get("set") == BASE and BASE not in m.get("selections", {}),
                      f"manifest: {BASE} is the base, not a selection (fetch extracts the selections' frames only)"))
    logs = m.get("dev_logs") if isinstance(m.get("dev_logs"), dict) else {}
    out.append(_check(logs.get("in_training_views") == 0 and logs.get(DEVLOGS) == len(dev[DEVLOGS])
                      and logs.get("held_out_logs") == len(lines),
                      f"manifest dev_logs: the held-out rows ({DEVLOGS}) are extracted, in no training view",
                      {DEVLOGS: len(dev[DEVLOGS]), "held_out_logs": logs.get("held_out_logs")}))
    covered = m.get("covered_by_base") if isinstance(m.get("covered_by_base"), dict) else {}
    union = train[FULL] | dev[FULL] | dev[DEVLOGS]
    shared = len((base["train"] | base["dev"]) & union)
    out.append(_check(covered.get("frames") == SHARED_FRAMES == shared,
                      f"manifest covered_by_base: {SHARED_FRAMES:,} frames v1.0-pool packed are planned here",
                      {k: covered.get(k) for k in ("frames", "new_frames", "base_frames_not_kept")}))
    return out


FULL_BUILD = xl.Build(name="v1-full", sets=SETS, views=(FULL,), trained=(FULL,), expected=EXPECTED,
                      frames_to_keep=FRAMES_TO_KEEP, conflicts_file="v1-full_label_conflicts.tsv", ref="v1.0-pool",
                      ref_sets=(BASE,), ref_view=BASE, copied_set=BASE, shared_frames=SHARED_FRAMES,
                      shared_check="the {n:,} frames v1.0-pool packed for v1-all (train and dev-v1) and kept here are "
                                   "packed here",
                      relations=_relations, resplit=True,
                      warmup_note="dev-v2 and the held-out rows are evaluation only, in no training view")


def _index(out_dir: str, columns: Sequence[str]) -> Dict[str, Any]:
    from data_pipeline.v1 import tables

    return tables.read(os.path.join(out_dir, "index", "candidates.parquet"), list(columns))


def check_rules(out_dir: str, metadata: Optional[str]) -> List[Check]:
    """The sets, the held-out logs and dev-v2 recomputed from the index and the warm-up list (module docstring)."""
    import numpy as np

    from data_pipeline.v1 import scene_exclusion, selection, token_sets

    m = xl.manifest(out_dir)
    exclusion = m.get("scene_exclusion") if isinstance(m.get("scene_exclusion"), dict) else {}
    window_s = float(exclusion.get("window_s", xl.WINDOW_S))
    cands = _index(out_dir, ["token", "log_name", "t_us", "our_split", "city", "has_cam_f0", "pose_ok", "standstill",
                             "in_navtrain", "command"])
    row_of = {str(t): i for i, t in enumerate(cands["token"].tolist())}
    sel = xl.selected(out_dir, FULL_BUILD)
    absent = sum(t not in row_of for name in SETS for t in sel[name])
    if absent:
        return [_check(False, "every token of the two sets is an index row", absent)]

    def rows(name: str, split: str) -> np.ndarray:
        return np.array(sorted(row_of[t] for t, s in sel[name].items() if s == split), dtype=np.int64)

    warm = xl.read_tokens(xl.warmup_path(out_dir))
    missing = sorted(set(warm) - set(row_of))
    located: Dict[str, Any] = {}
    out: List[Check] = []
    if missing:  # placed from the metadata over the logs that hold a listed row, as the plan placed them
        if metadata:
            logs = sorted({str(cands["log_name"][row_of[t]]) for t in warm if t in row_of})
            located = scene_exclusion.locate_in_metadata(metadata, missing, logs)
        found = {t: log for t, (log, _) in located.items()}
        recorded = dict(exclusion.get("located_in_metadata") or {})
        out.append(_check(bool(metadata) and found == recorded,
                          f"{len(missing)} listed warm-up tokens that are not index rows: "
                          + ("placed from the metadata as the plan placed them (= located_in_metadata)" if metadata else
                             "NOT placed, so the window and the sets cannot be recomputed (pass --metadata)"),
                          {"placed": len(found), "plan": len(recorded),
                           "differ": sorted(set(found.items()) ^ set(recorded.items()))}))
        if not metadata:
            return out
    window, _ = scene_exclusion.window_mask(cands, warm, window_s, located)
    passing = selection.eligible(cands) & ~window
    rule = token_sets.DEV_V2

    # the held-out logs: whole val drives, navtest's city mix by log count, the fewest navtrain rows (the DP)
    held_logs = held_out_logs(out_dir)
    held = token_sets.in_logs(cands, held_logs)
    drives = {token_sets.drive_of(log) for log in held_logs}
    index_logs = sorted(set(cands["log_name"].tolist()))
    split_of_log = dict(zip(cands["log_name"].tolist(), cands["our_split"].tolist()))
    sharing = [log for log in index_logs if log not in set(held_logs) and token_sets.drive_of(log) in drives]
    city_of_log = dict(zip(cands["log_name"].tolist(), cands["city"].tolist()))
    logs_per_city = {city: sum(city_of_log[log] == city for log in held_logs) for city, _ in rule.city_share}
    chosen, _ = token_sets.choose_dev_logs(cands, window, rule)
    out += [
        _check(bool(held_logs) and all(split_of_log.get(log) == "dev" for log in held_logs) and not sharing,
               "the held-out logs are whole nuPlan val drives: every one a val log of the index, no other log of their "
               "drives (token_sets.drive_of)", {"logs": len(held_logs), "drives": len(drives), "sharing": sharing[:5]}),
        _check(logs_per_city == token_sets.quotas(rule.n_logs, rule.city_share),
               f"held-out logs per city = navtest's city mix of {rule.n_logs} logs (largest remainder)", logs_per_city),
        _check(chosen == held_logs, "token_sets.choose_dev_logs picks exactly these logs (" + token_sets.DEV_LOGS_RULE + ")",
               {"differ": sorted(set(chosen) ^ set(held_logs))}),
    ]

    # v1-full's train and the held-out rows: every row that passes the rules, nothing thinned
    is_train, is_val = cands["our_split"] == "train", cands["our_split"] == "dev"
    for name, split, mask, what in ((FULL, "train", passing & (is_train | (is_val & ~held)),
                                     "row of the TRAIN logs and the released val logs"),
                                    (DEVLOGS, "dev", passing & held, "row of the held-out logs")):
        want, got = np.flatnonzero(mask), rows(name, split)
        out.append(_check(np.array_equal(got, want),
                          f"{name} {split} = every eligible {what} outside the ±{window_s} s warm-up window "
                          "(no standstill thinning, no spacing)",
                          {"recomputed": len(want), name: len(got), "differ": int(len(np.setxor1d(got, want))),
                           "standstill_rows": int(cands["standstill"][want].sum())}))
    eligible = selection.eligible(cands)
    recorded_rules = (m.get("every_row") or {}).get("by_group") or {}
    groups = {"train_logs": is_train, "released_val_logs": is_val & ~held, "held_out_val_logs": held}
    counts = {g: {"left_after": {"index_row": int(mk.sum()), "eligible": int((mk & eligible).sum()),
                                 "outside_window": int((mk & passing).sum())},
                  "excluded_by_first_failing_rule": {"eligible": int((mk & ~eligible).sum()),
                                                     "outside_window": int((mk & eligible & window).sum())}}
              for g, mk in groups.items()}
    mine = {g: {k: (recorded_rules.get(g) or {}).get(k) for k in c} for g, c in counts.items()}
    out.append(_check(counts == mine, "the rows left after each rule (token_sets.FULL_RULES order), per group of logs, "
                                      "= the manifest's every_row", {"recomputed": counts, "manifest": mine}))

    # dev-v2: its properties, then the pinned draw
    dev = rows(FULL, "dev")
    order = dev[np.lexsort((cands["t_us"][dev], cands["log_name"][dev]))]
    same = cands["log_name"][order][1:] == cands["log_name"][order][:-1]
    gaps = np.diff(cands["t_us"][order])[same]
    per_city = {city: int((cands["city"][dev] == city).sum()) for city, _ in rule.city_share}
    known = cands["in_navtrain"][dev] | (cands["command"][dev] != token_sets.UNKNOWN_COMMAND)
    out += [
        _check(bool(held[dev].all()) and bool(passing[dev].all()) and (not gaps.size or gaps.min() >= rule.spacing_s * 1e6
                                                                       - 50_000),
               f"dev-v2: every anchor a held-out row that passes the rules, >= {rule.spacing_s} s from the next in its log",
               {"anchors": len(dev), "min_gap_s": round(float(gaps.min()) / 1e6, 3) if gaps.size else None}),
        _check(per_city == token_sets.quotas(rule.n_anchors, rule.city_share),
               f"dev-v2 per city = navtest's city mix of {rule.n_anchors:,} anchors", per_city),
        _check(bool(known.all()), "dev-v2: every anchor a navtrain token or a row with a known route command "
                                  "(fill_needs_command)", {"navtrain": int(cands["in_navtrain"][dev].sum()),
                                                           "unknown_command": int((~known).sum())}),
    ]
    drawn, _ = token_sets.draw_dev(cands, window, held_logs, rule)
    out.append(_check(np.array_equal(drawn, dev), "the draw pinned in token_sets.DEV_DRAW reproduces dev-v2 ("
                      + token_sets.DEV_DRAW + ")",
                      {"numpy": np.__version__, "plan_numpy": ((m.get("dev_v2") or {}).get("draw") or {}).get("numpy"),
                       "differ": int(len(np.setxor1d(drawn, dev)))}))
    v1 = np.array(sorted(row_of[t] for t in xl.read_tokens(os.path.join(out_dir, "splits", DEV_V1_FILE))
                         if t in row_of), dtype=np.int64)
    in_train = np.zeros(len(cands["token"]), dtype=bool)
    in_train[rows(FULL, "train")] = True
    out.append(_check(True, "information: dev-v1 (the old pinned dev; never for a model trained on v1-full)",
                      {"rows": len(v1), "in_v1-full_train": int(in_train[v1].sum()), "in_held_out_logs": int(held[v1].sum()),
                       "inside_window": int(window[v1].sum())}))
    return out


def check_held_out_apart(out_dir: str) -> List[Check]:
    """dev-v2's held-out logs and drives are in no train set; every dev token is in a held-out log."""
    from data_pipeline.v1 import token_sets

    held = set(held_out_logs(out_dir))
    drives = {token_sets.drive_of(log) for log in held}
    cands = _index(out_dir, ["token", "log_name"])
    log_of = {str(t): str(l) for t, l in zip(cands["token"].tolist(), cands["log_name"].tolist())}
    sel = xl.selected(out_dir, FULL_BUILD)
    train = [t for name in SETS for t, s in sel[name].items() if s == "train"]
    dev = [t for name in SETS for t, s in sel[name].items() if s == "dev"]
    in_held = sum(log_of.get(t) in held for t in train)
    in_drive = sum(token_sets.drive_of(log_of[t]) in drives for t in train if t in log_of)
    return [_check(bool(held) and in_held == 0 == in_drive and all(log_of.get(t) in held for t in dev),
                   "dev-v2's held-out logs are in no train set: no train token in a held-out log or in a held-out drive, "
                   "and every dev token (dev-v2, the held-out rows) in a held-out log",
                   {"train_in_held_out_logs": in_held, "train_in_held_out_drives": in_drive,
                    "dev_outside_held_out_logs": sum(log_of.get(t) not in held for t in dev),
                    "held_out_logs": len(held), "held_out_drives": len(drives)})]


def check_leakage(out_dir: str, roots: Sequence[str], metadata: Optional[str]) -> List[Check]:
    return xl.check_leakage(out_dir, list(roots), metadata, FULL_BUILD) + check_held_out_apart(out_dir)


# --- views ---------------------------------------------------------------------------------------


def view_dir(out_dir: str, version: str, name: str = FULL) -> str:
    return os.path.join(out_dir, "views", VIEWS[version][0], name)


def _zip_checks(label: str, folder: str, spec: Dict[str, Any], images: set, deep: bool) -> List[Check]:
    zpath = os.path.join(folder, "images.zip")
    if not os.path.exists(zpath):
        return [_check(False, f"{label}: images.zip exists")]
    meta = spec.get("images_zip") or {}
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        bad_zip = z.testzip() if deep else None
    out = [_check(set(names) == images and len(names) == len(images) == meta.get("members"),
                  f"{label}: zip members = the jsonl images = view.json", {"members": len(names)}),
           _check(os.path.getsize(zpath) == meta.get("bytes"), f"{label}: zip size = view.json")]
    if deep:
        out.append(_check(bad_zip is None, f"{label}: testzip", bad_zip))
        out.append(_check(xl.sha256_file(zpath) == meta.get("sha256"), f"{label}: zip SHA-256 = view.json"))
    return out


def check_view(out_dir: str, pool_view: Optional[str], samples: int = 50, seed: int = 0,
               deep: bool = False, name: str = FULL) -> List[Check]:
    """The pilot_v1.1 view: the token files' samples with their split, images re-made from the packs, and sampled
    train lines and images identical to v1.0-pool's v1-all view (``pool_view``) where the sample is in both."""
    from data_pipeline.v1 import pilot_view

    folder = view_dir(out_dir, "pilot_v1.1", name)
    label = f"view {name} (pilot_v1.1)"
    if not os.path.exists(os.path.join(folder, "view.json")):
        return [_check(False, f"{label}: view.json exists")]
    spec = xl.load_json(os.path.join(folder, "view.json"))
    out = [_check(spec.get("storage") == "zip" and spec.get("set") == name and spec.get("prompt_version") == "pilot_v1.1",
                  f"{label}: storage zip, set {name}, prompt version pilot_v1.1",
                  {k: spec.get(k) for k in ("storage", "set", "prompt_version")})]
    items: Dict[str, List[Dict[str, Any]]] = {}
    for split in xl.SPLITS:
        items[split] = xl.jsonl(os.path.join(folder, f"{split}.jsonl"))
        ids = [xl.token_of(i) for i in items[split]]
        want = set(xl.read_tokens(xl.token_file(out_dir, name, split)))
        wrong = sum((i.get("meta") or {}).get("split") != split for i in items[split])
        out.append(_check(set(ids) == want and len(ids) == len(want) and not wrong,
                          f"{label} {split}.jsonl: one line per token of the set's {split} file, meta.split {split}",
                          {"lines": len(ids), "wrong_split": wrong}))
    out += _zip_checks(label, folder, spec, {i["image"] for s in xl.SPLITS for i in items[s]}, deep)

    rng = random.Random(seed)
    pick = rng.sample(sorted(items["dev"], key=lambda i: i["id"]), min(samples, len(items["dev"])))
    pick += rng.sample(sorted(items["train"], key=lambda i: i["id"]), min(samples, len(items["train"])))
    frames = xl.frame_records(out_dir)
    bad, remade = [], []
    reader = xl.ViewReader(folder)
    try:
        for i in pick:
            t = xl.token_of(i)
            f = frames.get(t)
            if f is None:
                bad.append(t)
                continue
            with zipfile.ZipFile(xl.pack_path(out_dir, f["archive"])) as z:
                original = z.read(xl.image_path(f["log"], t))
            if hashlib.sha256(original).hexdigest() != f["sha256"]:
                bad.append(t)
            elif reader.read(i["image"]) != pilot_view.resize_bytes(original):
                remade.append(t)
    finally:
        reader.close()
    out.append(_check(bool(pick) and not bad and not remade,
                      f"{label}: {len(pick)} sampled images (dev and train) = their packed originals resized again, the "
                      "originals with their recorded SHA-256", {"bad_original": bad[:5], "differ": remade[:5]}))
    if pool_view:
        base_train = set(xl.read_tokens(xl.token_file(out_dir, BASE, "train")))
        shared = [i for i in sorted(items["train"], key=lambda i: i["id"]) if xl.token_of(i) in base_train]
        chosen = {i["id"]: i for i in rng.sample(shared, min(samples, len(shared)))}
        theirs: Dict[str, Dict[str, Any]] = {}
        path = os.path.join(pool_view, "train.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip() and json.loads(line)["id"] in chosen:
                        theirs[json.loads(line)["id"]] = json.loads(line)
        differ = [k for k, i in chosen.items() if theirs.get(k) != i]
        ours_r, theirs_r = xl.ViewReader(folder), xl.ViewReader(pool_view)
        try:
            image_differ = [k for k, i in chosen.items() if k in theirs and ours_r.read(i["image"]) != theirs_r.read(i["image"])]
        finally:
            ours_r.close()
            theirs_r.close()
        out.append(_check(bool(chosen) and not differ and not image_differ,
                          f"{label}: {len(chosen)} sampled train samples also in v1.0-pool's {BASE} view have the same "
                          "line and image there", {"compared": len(chosen), "line_differs": differ[:5],
                                                   "image_differs": image_differ[:5]}))
    return out


def check_p12_view(out_dir: str, name: str = FULL, deep: bool = False) -> List[Check]:
    """The pilot_v1.2 view against the pilot_v1.1 view of the same build: the same samples, images and targets."""
    folder, ref = view_dir(out_dir, "pilot_v1.2", name), view_dir(out_dir, "pilot_v1.1", name)
    label = f"view {name} (pilot_v1.2)"
    for path in (folder, ref):
        if not os.path.exists(os.path.join(path, "view.json")):
            return [_check(False, f"{label}: {os.path.relpath(path, out_dir)}/view.json exists")]
    spec, ref_spec = xl.load_json(os.path.join(folder, "view.json")), xl.load_json(os.path.join(ref, "view.json"))
    out = [_check(spec.get("storage") == "zip" and spec.get("set") == name and spec.get("prompt_version") == "pilot_v1.2",
                  f"{label}: storage zip, set {name}, prompt version pilot_v1.2",
                  {k: spec.get(k) for k in ("storage", "set", "prompt_version")})]
    images = set()
    for split in xl.SPLITS:
        items, ref_items = xl.jsonl(os.path.join(folder, f"{split}.jsonl")), xl.jsonl(os.path.join(ref, f"{split}.jsonl"))
        want = set(xl.read_tokens(xl.token_file(out_dir, name, split)))
        ids = [xl.token_of(i) for i in items]
        out.append(_check(set(ids) == want and len(ids) == len(want),
                          f"{label} {split}.jsonl: one line per token of the set", len(items)))
        differ = sum(any(a.get(k) != b.get(k) for k in SAME_FIELDS) for a, b in zip(items, ref_items))
        prompts = sum("Past poses" in str(a.get("prompt")) for a in items)
        out.append(_check(len(items) == len(ref_items) and not differ and prompts == len(items),
                          f"{label} {split}.jsonl: line for line the pilot_v1.1 view's samples ({', '.join(SAME_FIELDS)}); "
                          "only the prompt differs, with the 4 past poses",
                          {"lines": len(items), "pilot_v1.1_lines": len(ref_items), "differ": differ,
                           "v1.2_prompts": prompts}))
        images |= {i["image"] for i in items}
    out += _zip_checks(label, folder, spec, images, deep)
    meta, ref_meta = spec.get("images_zip") or {}, ref_spec.get("images_zip") or {}
    out.append(_check(meta.get("sha256") == ref_meta.get("sha256") and meta.get("bytes") == ref_meta.get("bytes"),
                      f"{label}: images.zip = the pilot_v1.1 view's (size and recorded SHA-256: the same images)",
                      {"sha256": meta.get("sha256"), "pilot_v1.1": ref_meta.get("sha256")}))
    return out


def check_views_train_only(out_dir: str) -> List[Check]:
    """No view's train.jsonl holds a held-out row, and no view of v1-full-devlogs exists (it is in no view)."""
    cands = _index(out_dir, ["token", "log_name"])
    log_of = {str(t): str(l) for t, l in zip(cands["token"].tolist(), cands["log_name"].tolist())}
    held = set(held_out_logs(out_dir))
    out, seen = [], {}
    for version in VIEWS:
        path = os.path.join(view_dir(out_dir, version), "train.jsonl")
        if not os.path.exists(path):
            out.append(_check(False, f"view {FULL} ({version}): train.jsonl exists"))
            continue
        ids = [xl.token_of(i) for i in xl.jsonl(path)]
        seen[version] = {"train": len(ids), "in_held_out_logs": sum(log_of.get(t) in held for t in ids),
                         "not_in_index": sum(t not in log_of for t in ids)}
    out.append(_check(bool(seen) and all(v["in_held_out_logs"] == 0 == v["not_in_index"] for v in seen.values()),
                      "no held-out row in any training view: no train.jsonl sample is in one of dev-v2's logs", seen))
    stray = [os.path.join("views", root, DEVLOGS) for root, _ in VIEWS.values()
             if os.path.exists(os.path.join(out_dir, "views", root, DEVLOGS))]
    out.append(_check(not stray, f"no view of {DEVLOGS} (the held-out rows are in no view)", stray))
    return out


def check_links(out_dir: str, links_root: str) -> List[Check]:
    out = []
    for version, (_, suffix) in VIEWS.items():
        link = os.path.join(links_root, FULL + suffix)
        ok = os.path.islink(link) and os.path.realpath(link) == os.path.realpath(view_dir(out_dir, version))
        out.append(_check(ok, f"{link} → the {FULL} view ({version})"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Checks on the v1.0-full build (PRETRAIN_V1_PLAN.md §12).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("splits", "rules", "leakage", "fetched", "crosscheck", "canonical", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the v1.0-full dataset directory")
        p.add_argument("--quiet", action="store_true", help="print only the summary line")
        if name in ("splits", "crosscheck", "all"):
            p.add_argument("--pool", required=name != "splits", help="the v1.0-pool dataset directory")
        if name in ("rules", "leakage", "all"):
            p.add_argument("--metadata", help="openscene_metadata_trainval.tgz: places the listed warm-up tokens that "
                                              "are not index rows (required when there are any, else the check fails)")
        if name in ("leakage", "all"):
            p.add_argument("--navsim-root", action="append", default=[],
                           help="a NAVSIM devkit: also check against its own lists; repeat for several")
        if name in ("fetched", "all"):
            p.add_argument("--deep", action="store_true", help="testzip and re-hash every pack (and view zip)")
        if name == "all":
            p.add_argument("--pool-view", help=f"default: <pool>/views/pilot_v1/{BASE}")
            p.add_argument("--links", help="folder of the pilot_v1/<set> links, e.g. $TRAINING_DATA_ROOT/pilot_v1")
            p.add_argument("--samples", type=int, default=50)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    status = 0
    if args.command == "splits":
        checks = xl.check_splits(args.out, args.pool, FULL_BUILD)
    elif args.command == "rules":
        checks = check_rules(args.out, args.metadata)
    elif args.command == "leakage":
        checks = check_leakage(args.out, args.navsim_root, args.metadata)
    elif args.command == "fetched":
        checks, complete = xl.check_fetched(args.out, 200, True, args.deep, FULL_BUILD)
        status = 0 if complete else xl.NOT_YET
    elif args.command == "crosscheck":
        checks = xl.check_crosscheck(args.out, args.pool, True, FULL_BUILD)
    elif args.command == "canonical":
        checks, present = xl.check_canonical(args.out, True, FULL_BUILD)
        status = 0 if present else xl.NOT_YET
    else:
        pool_view = args.pool_view or xl.view_dir(args.pool, BASE)
        checks = xl.check_splits(args.out, args.pool, FULL_BUILD) + check_rules(args.out, args.metadata)
        checks += check_leakage(args.out, args.navsim_root, args.metadata)
        checks += xl.check_fetched(args.out, 200, True, args.deep, FULL_BUILD)[0]
        checks += xl.check_crosscheck(args.out, args.pool, True, FULL_BUILD)
        checks += xl.check_canonical(args.out, True, FULL_BUILD)[0]
        checks += check_view(args.out, pool_view, args.samples, args.seed, args.deep)
        checks += check_p12_view(args.out, FULL, args.deep) + check_views_train_only(args.out)
        if args.links:
            checks += check_links(args.out, args.links)
    path = xl.write_report(args.out, args.command, checks)
    failed = [c for c in checks if not c["ok"]]
    if not args.quiet or (failed and status != xl.NOT_YET):
        for c in failed:
            print(f"[verify] FAILED: {c['check']}" + (f": {json.dumps(c['value'])}" if "value" in c else ""))
    print(f"[verify] {args.command}: {len(checks) - len(failed)} of {len(checks)} checks pass ({path})")
    if status == xl.NOT_YET:
        sys.exit(xl.NOT_YET)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
