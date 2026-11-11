"""Sets cut from a built folder's own set, with no new fetch and no canonical step [proposed 2026-11-10].

Every frame of a derived set is a frame of its parent, so ``frames.parquet`` already holds it and its image;
``pilot_view`` takes a derived set's membership from its token files (``pilot_view.member_rows``), not from an
``in_<set>`` column. In the v1.0-full folder (PRETRAIN_V1_PLAN.md §12):

* ``v1-full-thin``: v1-full's train, standstill-thinned as v1-all is (rule 6: at most one standstill anchor per
  5 s in each log, thinned before the warm-up window, so v1-all's train lies inside it), plus every navtrain row
  of v1-full's train (NAVSIM's own scenes, not thinned, as in v1-nt). Dev: v1-full's (dev-v2).
* ``dev-v2``: v1-full's dev alone, train empty: a 2,000-sample view to re-evaluate earlier adapters on dev-v2.
* ``v1-hard`` [proposed 2026-11-11]: the hard part of v1-full-thin (``HARD_RULE``). Researcher, 2026-11-11: "the
  best results come from showing a variety of harder data. This is crucial because we can save compute", and on
  standstill frames: "you should thin them so we don't waste compute but they should not be omitted". So: every
  navtrain row, every moving row about as hard as navtrain by the constant-velocity endpoint error (``HARD_QUANTILE``
  of navtrain's own), and standstill anchors thinned further (``HARD_STANDSTILL_PERIOD_S``). Dev: dev-v2.

    python -m data_pipeline.v1.derived_sets --out data/pretrain/v1.0-full              # v1-full-thin, dev-v2
    python -m data_pipeline.v1.derived_sets --out data/pretrain/v1.0-full --set v1-hard  # after v1-full-thin

Writes ``splits/<set>_tokens_{train,dev}.txt`` and a report: ``splits/derived.json`` (v1-full-thin, dev-v2) or
``splits/hard.json`` (v1-hard), with rules, counts and SHA-256. A derived set's view can be cut from its parent's
built view (``data_pipeline/v1/subset_view.py``) instead of re-packing the images.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from data_pipeline.v1 import openscene_images, provenance, selection, tables, token_sets
from data_pipeline.v1.openscene_index import DT_TOL_US, PATH_OFFSETS_S
from navsim_tools import maneuvers

THIN, DEV_ONLY, HARD = "v1-full-thin", "dev-v2", "v1-hard"
PARENT = token_sets.FULL
DERIVED = {THIN: PARENT, DEV_ONLY: PARENT, HARD: THIN}  # set -> the set whose frames it is cut from
HARD_QUANTILE = 0.25  # a moving non-navtrain row must be at least as hard as navtrain's easiest quarter [proposed]
HARD_STANDSTILL_PERIOD_S = 20.0  # v1-hard's standstill anchors: at most one per 20 s in each log (thin: 5 s) [proposed]
HARD_RULE = ("v1-full-thin's train rows that are (a) navtrain scenes; (b) not standstill, where driving on at the "
             "anchor's speed and heading for {horizon:g} s ends at least {cut:.3f} m from the human's pose then (the "
             "{q:g} quantile of that distance over v1-full-thin's navtrain rows: a proxy for NAVSIM's filter, which "
             "scores that constant-velocity path with the PDM score); or (c) standstill anchors thinned to at most "
             "one per {period:g} s in each log (rule 6 again, over v1-full-thin's standstill rows); "
             "dev = v1-full's dev (dev-v2)")
BASE = "v1-all"  # v1.0-pool's, copied into v1.0-full/splits by its plan: its train must lie inside v1-full-thin
THIN_RULE = ("v1-full's train rows that rule 6 keeps when it thins the eligible rows of v1-full's train logs "
             "(at most one standstill anchor per {period:g} s in each log; before the warm-up window, as v1-all), "
             "plus every navtrain row of v1-full's train; dev = v1-full's dev (dev-v2)")


def _rows(cands: tables.Columns, path: str) -> np.ndarray:
    row_of = {t: i for i, t in enumerate(cands["token"].tolist())}  # np.isin on strings takes minutes
    tokens = openscene_images._read_tokens(path)
    missing = [t for t in tokens if t not in row_of]
    if missing:
        raise ValueError(f"{len(missing)} tokens of {path} are not in the index, e.g. {missing[0]}")
    return np.sort(np.array([row_of[t] for t in tokens], dtype=np.int64))


def thin_rows(cands: tables.Columns, full_train: np.ndarray,
              period_s: float = token_sets.STANDSTILL_PERIOD_S) -> np.ndarray:
    """``v1-full-thin``'s train rows (``THIN_RULE``), ascending."""
    n = len(cands["t_us"])
    in_full = np.zeros(n, dtype=bool)
    in_full[full_train] = True
    logs = token_sets.in_logs(cands, set(cands["log_name"][full_train].tolist()))
    thinned = selection.thin_standstill(cands, selection.eligible(cands) & logs, period_s)
    return np.flatnonzero(in_full & (thinned | np.asarray(cands["in_navtrain"], dtype=bool)))


def check_thin(cands: tables.Columns, thin: np.ndarray, full_train: np.ndarray, base_train: Optional[np.ndarray],
               period_s: float = token_sets.STANDSTILL_PERIOD_S) -> None:
    """Post-conditions of ``thin_rows``; violations are bugs, so they raise even under ``python -O``."""
    if not np.isin(thin, full_train).all():
        raise RuntimeError("a v1-full-thin row is not a v1-full train row")
    navtrain = full_train[np.asarray(cands["in_navtrain"], dtype=bool)[full_train]]
    if not np.isin(navtrain, thin).all():
        raise RuntimeError("a navtrain row of v1-full's train is not in v1-full-thin")
    if base_train is not None and not np.isin(base_train, thin).all():
        raise RuntimeError(f"{int((~np.isin(base_train, thin)).sum())} {BASE} train rows are not in v1-full-thin")
    still = thin[np.asarray(cands["standstill"], dtype=bool)[thin] & ~np.asarray(cands["in_navtrain"], dtype=bool)[thin]]
    order = still[np.lexsort((cands["t_us"][still], cands["log_name"][still]))]
    same = cands["log_name"][order][1:] == cands["log_name"][order][:-1]
    gaps = np.diff(cands["t_us"][order])[same]
    if gaps.size and gaps.min() < period_s * 1e6 - DT_TOL_US:
        raise RuntimeError(f"two thinned standstill anchors in one log are closer than {period_s:g} s")


def derive(out_dir: str, period_s: float = token_sets.STANDSTILL_PERIOD_S) -> Dict[str, Any]:
    """Write both derived sets' token files and ``splits/derived.json``; returns the report."""
    split_dir = os.path.join(out_dir, "splits")
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"), openscene_images.POOL_COLUMNS)
    full = {split: _rows(cands, os.path.join(split_dir, f"{PARENT}_tokens_{split}.txt")) for split in ("train", "dev")}
    base_path = os.path.join(split_dir, f"{BASE}_tokens_train.txt")
    base_train = _rows(cands, base_path) if os.path.exists(base_path) else None
    thin = thin_rows(cands, full["train"], period_s)
    check_thin(cands, thin, full["train"], base_train, period_s)
    sets = {THIN: {"train": thin, "dev": full["dev"]},
            DEV_ONLY: {"train": np.zeros(0, dtype=np.int64), "dev": full["dev"]}}
    labels = token_sets.maneuver_labels(cands)
    nav = np.flatnonzero(np.asarray(cands["in_navtrain"], dtype=bool))
    report: Dict[str, Any] = {"step": "derived_sets", "created_utc": provenance.utc_now(),
                              "repo_commit": provenance.git_commit(provenance.REPO_ROOT), "sets": {}}
    for name, splits in sets.items():
        sha = {}
        for split, rows in splits.items():
            path = os.path.join(split_dir, f"{name}_tokens_{split}.txt")
            openscene_images._write_tokens(path, cands["token"][rows].tolist())
            sha[split] = openscene_images._sha256(path)
        reference = {PARENT: full["train"], "navtrain": nav}
        if base_train is not None:
            reference[BASE] = base_train
        report["sets"][name] = {
            "parent": DERIVED[name],
            "rule": THIN_RULE.format(period=period_s) if name == THIN else f"train empty; dev = {PARENT}'s dev (dev-v2)",
            "train": int(len(splits["train"])), "dev": int(len(splits["dev"])), "sha256": sha,
            "splits": openscene_images._describe(cands, np.union1d(splits["train"], splits["dev"]), splits),
            "train_detail": token_sets.describe_train(cands, labels, splits["train"], reference),
        }
    report["sets"][THIN]["removed_from_parent_train"] = int(len(full["train"]) - len(thin))
    report["standstill_period_s"] = period_s
    provenance.write_text(os.path.join(split_dir, "derived.json"), json.dumps(report, indent=1) + "\n")
    return report


def cv_endpoint_error(cands: tables.Columns, rows: np.ndarray) -> np.ndarray:
    """Metres between each row's human pose at the horizon (+4 s) and constant velocity: the anchor's speed held
    straight ahead (+x of the anchor's ego frame), as NAVSIM's constant-velocity agent drives."""
    rows = np.asarray(rows, dtype=np.int64)
    human = maneuvers.future_poses(np.asarray(cands["path_xyyaw"])[rows], PATH_OFFSETS_S)
    end = human[:, -1, :2]
    ahead = np.asarray(cands["speed"], dtype=np.float64)[rows] * maneuvers.HORIZON_S[-1]
    return np.hypot(end[:, 0] - ahead, end[:, 1])


def hard_rows(cands: tables.Columns, thin: np.ndarray, quantile: float = HARD_QUANTILE,
              period_s: float = HARD_STANDSTILL_PERIOD_S) -> Tuple[np.ndarray, float, Dict[str, int]]:
    """``v1-hard``'s train rows (``HARD_RULE``), ascending; the cut in metres; the rows of each part."""
    thin = np.unique(np.asarray(thin, dtype=np.int64))
    nav = np.asarray(cands["in_navtrain"], dtype=bool)[thin]
    still = np.asarray(cands["standstill"], dtype=bool)[thin]
    err = cv_endpoint_error(cands, thin)
    if not nav.any():
        raise ValueError("v1-hard's cut is a quantile of the navtrain rows, and the parent has none")
    cut = float(np.quantile(err[nav], quantile))
    moving = thin[~nav & ~still & (err >= cut)]
    standstill = np.zeros(len(cands["t_us"]), dtype=bool)
    standstill[thin[~nav & still]] = True
    kept_still = np.flatnonzero(selection.thin_standstill(cands, standstill, period_s))
    rows = np.union1d(np.union1d(thin[nav], moving), kept_still)
    parts = {"navtrain": int(nav.sum()), "hard_moving": int(len(moving)), "standstill": int(len(kept_still)),
             "dropped_moving_below_cut": int((~nav & ~still & (err < cut)).sum()),
             "dropped_standstill": int((~nav & still).sum() - len(kept_still))}
    return rows, cut, parts


def check_hard(cands: tables.Columns, hard: np.ndarray, thin: np.ndarray, cut: float,
               period_s: float = HARD_STANDSTILL_PERIOD_S) -> None:
    """Post-conditions of ``hard_rows``; violations are bugs, so they raise even under ``python -O``."""
    if not np.isin(hard, thin).all():
        raise RuntimeError("a v1-hard row is not a v1-full-thin train row")
    nav = np.asarray(cands["in_navtrain"], dtype=bool)
    still = np.asarray(cands["standstill"], dtype=bool)
    if not np.isin(thin[nav[thin]], hard).all():
        raise RuntimeError("a navtrain row of v1-full-thin is not in v1-hard")
    moving = hard[~nav[hard] & ~still[hard]]
    if len(moving) and cv_endpoint_error(cands, moving).min() < cut:
        raise RuntimeError("a moving non-navtrain row of v1-hard is below the cut")
    out = thin[~nav[thin] & ~still[thin] & ~np.isin(thin, hard)]
    if len(out) and cv_endpoint_error(cands, out).max() >= cut:
        raise RuntimeError("a moving v1-full-thin row at or above the cut is not in v1-hard")
    kept = hard[still[hard] & ~nav[hard]]
    if not len(kept) and (still[thin] & ~nav[thin]).any():
        raise RuntimeError("v1-hard has no standstill anchor: they are thinned, never removed")
    order = kept[np.lexsort((cands["t_us"][kept], cands["log_name"][kept]))]
    same = cands["log_name"][order][1:] == cands["log_name"][order][:-1]
    gaps = np.diff(cands["t_us"][order])[same]
    if gaps.size and gaps.min() < period_s * 1e6 - DT_TOL_US:
        raise RuntimeError(f"two v1-hard standstill anchors in one log are closer than {period_s:g} s")


def derive_hard(out_dir: str, quantile: float = HARD_QUANTILE,
                period_s: float = HARD_STANDSTILL_PERIOD_S) -> Dict[str, Any]:
    """Write ``v1-hard``'s token files and ``splits/hard.json`` from v1-full-thin's (``derive`` first)."""
    split_dir = os.path.join(out_dir, "splits")
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"), openscene_images.POOL_COLUMNS)
    thin = _rows(cands, os.path.join(split_dir, f"{THIN}_tokens_train.txt"))
    dev = _rows(cands, os.path.join(split_dir, f"{THIN}_tokens_dev.txt"))
    hard, cut, parts = hard_rows(cands, thin, quantile, period_s)
    check_hard(cands, hard, thin, cut, period_s)
    splits = {"train": hard, "dev": dev}
    sha = {}
    for split, rows in splits.items():
        path = os.path.join(split_dir, f"{HARD}_tokens_{split}.txt")
        openscene_images._write_tokens(path, cands["token"][rows].tolist())
        sha[split] = openscene_images._sha256(path)
    labels = token_sets.maneuver_labels(cands)
    nav = np.flatnonzero(np.asarray(cands["in_navtrain"], dtype=bool))
    report = {"step": "derived_sets", "created_utc": provenance.utc_now(),
              "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
              "sets": {HARD: {
                  "parent": THIN,
                  "rule": HARD_RULE.format(horizon=maneuvers.HORIZON_S[-1], cut=cut, q=quantile, period=period_s),
                  "cut_m": round(cut, 4), "quantile": quantile, "standstill_period_s": period_s, "parts": parts,
                  "train": int(len(hard)), "dev": int(len(dev)), "sha256": sha,
                  "splits": openscene_images._describe(cands, np.union1d(hard, dev), splits),
                  "train_detail": token_sets.describe_train(cands, labels, hard, {THIN: thin, "navtrain": nav}),
                  "removed_from_parent_train": int(len(thin) - len(hard))}}}
    provenance.write_text(os.path.join(split_dir, "hard.json"), json.dumps(report, indent=1) + "\n")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Derived sets of a built folder (no new fetch): v1-full-thin, "
                                                 "dev-v2; or v1-hard (--set v1-hard, after them).")
    parser.add_argument("--out", required=True, help="the v1.0-full folder, e.g. data/pretrain/v1.0-full")
    parser.add_argument("--set", choices=(THIN, HARD), default=THIN,
                        help=f"{THIN} (with dev-v2, the default) or {HARD}")
    args = parser.parse_args(argv)
    report = derive_hard(args.out) if args.set == HARD else derive(args.out)
    print(json.dumps({name: {k: s[k] for k in ("train", "dev", "sha256")} for name, s in report["sets"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
