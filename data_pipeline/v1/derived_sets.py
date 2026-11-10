"""Sets cut from a built folder's own set, with no new fetch and no canonical step [proposed 2026-11-10].

Every frame of a derived set is a frame of its parent, so ``frames.parquet`` already holds it and its image;
``pilot_view`` takes a derived set's membership from its token files (``pilot_view.member_rows``), not from an
``in_<set>`` column. In the v1.0-full folder (PRETRAIN_V1_PLAN.md §12):

* ``v1-full-thin``: v1-full's train, standstill-thinned as v1-all is (rule 6: at most one standstill anchor per
  5 s in each log, thinned before the warm-up window, so v1-all's train lies inside it), plus every navtrain row
  of v1-full's train (NAVSIM's own scenes, not thinned, as in v1-nt). Dev: v1-full's (dev-v2).
* ``dev-v2``: v1-full's dev alone, train empty: a 2,000-sample view to re-evaluate earlier adapters on dev-v2.

    python -m data_pipeline.v1.derived_sets --out data/pretrain/v1.0-full

Writes ``splits/<set>_tokens_{train,dev}.txt`` for both sets and ``splits/derived.json`` (rules, counts, SHA-256).
"""

import argparse
import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

from data_pipeline.v1 import openscene_images, provenance, selection, tables, token_sets
from data_pipeline.v1.openscene_index import DT_TOL_US

THIN, DEV_ONLY = "v1-full-thin", "dev-v2"
PARENT = token_sets.FULL
DERIVED = {THIN: PARENT, DEV_ONLY: PARENT}  # set -> the set whose frames it is cut from
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Derived sets of a built folder (no new fetch): v1-full-thin, dev-v2.")
    parser.add_argument("--out", required=True, help="the v1.0-full folder, e.g. data/pretrain/v1.0-full")
    args = parser.parse_args(argv)
    report = derive(args.out)
    print(json.dumps({name: {k: s[k] for k in ("train", "dev", "sha256")} for name, s in report["sets"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
