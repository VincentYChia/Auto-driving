"""Rules 2–8 of ``docs/training_data/PRETRAIN_V1_PLAN.md`` §3: which candidate anchors become samples.

Input: the candidate columns of ``openscene_index`` (a dict of numpy arrays). The index has already
applied rule 1 (valid window) and assigned each log its split (rule 4). Rule 9 (no route, scene
boundary, in navtrain) only flags rows, so it is not applied here. The "JPEG decodes" half of
rule 2 is checked when the images are extracted (step 3).

Order:
1. eligibility: CAM_F0 path present (rule 2), pose sane (rule 3), not guarded (rule 5) and, from
   step 3 on, the log's images are available;
2. standstill thinning (rule 6);
3. per split, a seeded random pick that keeps anchors at least ``spacing_s`` apart in each log
   (rule 7) and caps keep_speed + straight at ``cruise_cap`` of the sample (rule 8). The other
   classes are drawn first, in random order, so they keep their natural proportions; cruise anchors
   fill the rest. If the other classes run out, cruise fills the gap and the cap is reported as
   exceeded. Random order alone reaches 83 of the 100 anchors an unbroken 200 s log allows at 2 s
   (V, the synthetic log in the tests) and 84–88% of the per-log maximum on the real index at 2.0,
   1.5 and 1.0 s (V, 2026-11-08). A last pass then offers the remaining rows log by log in time
   order, but it runs only after every pool row of the split has been offered and the picker never
   frees a slot, so it adds 0 rows (V: tests, 200 random synthetic setups, the real index). It is
   kept as a harmless guard; the report counts what it adds (``stages.fill``).

``fixed`` rows (v1-xl: every v1-large row) are accepted before the draw [proposed 2026-11-08]. They
seed the picker in (log, time) order, count against the target and the cruise quota, and are left
out of the shuffled lists; the balance weights are still computed over the full pool. A fixed row
that is ineligible, thinned, in the wrong split or closer than the spacing to another raises.
The first draw of other classes is capped at the room the fixed rows leave, so fixed rows that are
more than ``cruise_cap`` cruise never push a split past its target. With ``fixed=None`` the selected rows are exactly those of the draw without it.

``train_excluded`` (v1-xl: the rows near a NAVSIM warm-up scene, ``scene_exclusion.window_mask``)
[proposed 2026-11-09] is a boolean mask over the rows; a masked row is never selected for TRAIN. Dev
ignores it (dev stays pinned, evaluation only). It is applied after standstill thinning, so thinning
and therefore every unmasked fixed row stay valid, and the draw order and balance weights are those
of the full pool, with masked rows skipped as they come up: an all-False mask selects exactly what
``train_excluded=None`` selects. A fixed train row inside the mask raises; the caller drops such rows
from ``fixed`` and reports them (``openscene_images.plan --exclude-scenes``).

With ``balance_alpha`` > 0 the train split is drawn in a weighted random order instead: an anchor's
weight is ``n ** -balance_alpha``, where n counts the split's pool anchors with its city and lateral
decision. Rare combinations (a Singapore lane change) come up earlier than common ones (a Las Vegas
straight), so they make up more of the sample when the pool is larger than the target. Dev keeps
the natural mix, like NAVSIM's test splits. Agreed 2026-10-31 for v1-pilot (plan §3, rule 8).
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from data_pipeline.v1 import stats, tables
from data_pipeline.v1.openscene_index import DT_TOL_US

CRUISE = ("keep_speed", "straight")
ARCHIVES_TOTAL = 200  # openscene_sensor_trainval_camera_{0..199}.tgz


@dataclass(frozen=True)
class SelectionConfig:
    name: str
    n_train: int
    n_dev: int
    spacing_s: float = 2.0  # rule 7
    standstill_period_s: float = 5.0  # rule 6
    cruise_cap: float = 0.40  # rule 8
    balance_alpha: float = 0.0  # rule 8, train only; 0 = random order
    seed: int = 20261027


# ~10 GB of CAM_F0 JPEGs at 211 KB each (V, archive 70), drawn from every log [agreed 2026-10-31].
SMOKE = SelectionConfig("smoke", n_train=10_000, n_dev=1_000, spacing_s=1.0, balance_alpha=0.5)
PILOT = SelectionConfig("v1-pilot", n_train=45_000, n_dev=2_000, balance_alpha=0.5)
# Built on Anvil in its own dataset folder [proposed 2026-11-03]. The 2.0 s spacing caps train at 103,382 (V,
# 2026-11-03); the dev draw does not depend on n_train, so its 2k dev set is v1-pilot's.
LARGE = SelectionConfig("v1-large", n_train=100_000, n_dev=2_000, balance_alpha=0.5)
# v1-large's rules with train anchors >= 1.0 s apart, about 2x [agreed 2026-11-08]. Planned with every v1-large
# row fixed, so it is a superset of v1-large with v1-pilot's dev set [proposed mechanics]: 200,000 train (V).
# Since 2026-11-09 the 1,727 v1-large train rows within 7 s of a NAVSIM v1.1 warm-up scene are not fixed and no
# train row in that window is drawn (train_excluded; PRETRAIN_V1_PLAN.md §10) [proposed]: still 200,000 (V).
XL = SelectionConfig("v1-xl", n_train=200_000, n_dev=2_000, spacing_s=1.0, balance_alpha=0.5)


def log_codes(cands: tables.Columns) -> np.ndarray:
    return np.unique(cands["log_name"], return_inverse=True)[1].reshape(-1)


def is_cruise(cands: tables.Columns) -> np.ndarray:
    return (cands["decision_lon"] == CRUISE[0]) & (cands["decision_lat"] == CRUISE[1])


def eligible(cands: tables.Columns, available_logs: Optional[Iterable[str]] = None) -> np.ndarray:
    """Rules 2, 3 and 5, plus the logs whose images are available (all, if ``None``)."""
    ok = cands["has_cam_f0"] & cands["pose_ok"] & (cands["our_split"] != "excluded")
    if available_logs is not None:
        available = set(available_logs)
        ok &= np.array([log in available for log in cands["log_name"]], dtype=bool)
    return ok


def thin_standstill(
    cands: tables.Columns, mask: np.ndarray, period_s: float, codes: Optional[np.ndarray] = None
) -> np.ndarray:
    """Rule 6: of the standstill anchors in ``mask``, keep at most one per ``period_s`` in each log."""
    codes = log_codes(cands) if codes is None else codes
    keep = mask.copy()
    rows = np.flatnonzero(mask & cands["standstill"])
    rows = rows[np.lexsort((cands["t_us"][rows], codes[rows]))]
    min_gap_us = period_s * 1e6 - DT_TOL_US
    last_code, last_t = None, 0
    for r in rows:
        if codes[r] == last_code and cands["t_us"][r] - last_t < min_gap_us:
            keep[r] = False
        else:
            last_code, last_t = codes[r], cands["t_us"][r]
    return keep


def _draw_order(rng: np.random.Generator, rows: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """A random order of ``rows``; with ``weights``, a weighted one (Efraimidis–Spirakis keys)."""
    if weights is None:
        return rng.permutation(rows)
    keys = np.log(rng.random(len(rows))) / weights  # log of u ** (1 / w): same order, no underflow
    return rows[np.argsort(-keys, kind="stable")]


def balance_weights(cands: tables.Columns, rows: np.ndarray, alpha: float) -> np.ndarray:
    """``n ** -alpha`` per row, n = how many of ``rows`` share its city and lateral decision."""
    if len(rows) == 0:
        return np.zeros(0)
    cell = [f"{c}|{d}" for c, d in zip(cands["city"][rows], cands["decision_lat"][rows])]
    _, inverse, counts = np.unique(cell, return_inverse=True, return_counts=True)
    return counts[inverse.reshape(-1)].astype(np.float64) ** -alpha


class _SpacedPicker:
    """Accepts rows whose anchor is at least ``spacing_s`` from every accepted anchor of its log."""

    def __init__(self, codes: np.ndarray, t_us: np.ndarray, spacing_s: float):
        self.codes, self.t_us = codes, t_us
        self.min_gap_us = spacing_s * 1e6 - DT_TOL_US
        self.taken: Dict[int, List[int]] = {}

    def take(self, rows: Iterator[int], n: int) -> List[int]:
        out: List[int] = []
        while len(out) < n:
            r = next(rows, None)
            if r is None:
                break
            times = self.taken.setdefault(int(self.codes[r]), [])
            t = int(self.t_us[r])
            k = bisect.bisect_left(times, t)
            if (k < len(times) and times[k] - t < self.min_gap_us) or (k > 0 and t - times[k - 1] < self.min_gap_us):
                continue
            times.insert(k, t)
            out.append(int(r))
        return out


SPLITS = ("train", "dev")
Fixed = Union[Sequence[int], np.ndarray, Mapping[str, Sequence[int]]]


def _fixed_rows(cands: tables.Columns, fixed: Optional[Fixed]) -> np.ndarray:
    """Sorted unique row indices of ``fixed``. A mapping split → rows also checks each row's split."""
    if fixed is None:
        return np.zeros(0, dtype=np.int64)
    if isinstance(fixed, Mapping):
        unknown = set(fixed) - set(SPLITS)
        if unknown:
            raise ValueError(f"fixed rows given for unknown splits {sorted(unknown)}")
        parts = []
        for split, rows in fixed.items():
            rows = np.asarray(rows, dtype=np.int64).reshape(-1)
            wrong = int((cands["our_split"][rows] != split).sum())
            if wrong:
                raise RuntimeError(f"{wrong} fixed {split} rows are not in the index's {split} split")
            parts.append(rows)
        rows = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    else:
        rows = np.asarray(fixed, dtype=np.int64).reshape(-1)
    unique = np.unique(rows)
    if len(unique) != len(rows):
        raise ValueError(f"{len(rows) - len(unique)} fixed rows are given twice")
    if len(unique) and (unique[0] < 0 or unique[-1] >= len(cands["t_us"])):
        raise ValueError("a fixed row is outside the index")
    return unique


def select(
    cands: tables.Columns,
    config: SelectionConfig,
    available_logs: Optional[Iterable[str]] = None,
    fixed: Optional[Fixed] = None,
    train_excluded: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sorted indices of the selected rows, and statistics per split.

    ``fixed``: rows that must be selected (row indices, or a mapping split → row indices, which also
    checks the split). They are accepted first and count against each split's target and cruise
    quota (see the module docstring). ``None`` selects exactly what the draw alone selects.
    ``train_excluded``: boolean mask over the rows that may not be selected for train (module
    docstring); a fixed train row inside it raises. ``None`` changes nothing.
    """
    codes = log_codes(cands)
    n_logs = int(codes.max()) + 1 if len(codes) else 0
    base = eligible(cands, available_logs)
    pool = thin_standstill(cands, base, config.standstill_period_s, codes)
    cruise = is_cruise(cands)
    picker = _SpacedPicker(codes, cands["t_us"], config.spacing_s)
    fixed_rows = _fixed_rows(cands, fixed)
    excluded = None
    if train_excluded is not None:
        excluded = np.asarray(train_excluded)
        if excluded.dtype != bool or excluded.shape != (len(codes),):
            raise ValueError(f"train_excluded must be a boolean mask of {len(codes)} rows")
        masked_fixed = int((excluded[fixed_rows] & (cands["our_split"][fixed_rows] == "train")).sum())
        if masked_fixed:
            raise RuntimeError(f"{masked_fixed} fixed train rows are in train_excluded: drop them from fixed first")
    if not base[fixed_rows].all():
        raise RuntimeError(f"{int((~base[fixed_rows]).sum())} fixed rows fail rule 2, 3 or 5, or their log is unavailable")
    if not pool[fixed_rows].all():
        raise RuntimeError(f"{int((~pool[fixed_rows]).sum())} fixed rows are removed by standstill thinning (rule 6)")
    if not np.isin(cands["our_split"][fixed_rows], SPLITS).all():
        raise RuntimeError("a fixed row is in neither train nor dev")
    is_fixed = np.zeros(len(codes), dtype=bool)
    is_fixed[fixed_rows] = True
    seeded = picker.take(iter(fixed_rows[np.lexsort((cands["t_us"][fixed_rows], codes[fixed_rows]))]), len(fixed_rows))
    if len(seeded) != len(fixed_rows):
        raise RuntimeError(f"{len(fixed_rows) - len(seeded)} fixed rows are closer than {config.spacing_s} s "
                           "to another fixed row of their log (rule 7)")
    report: Dict[str, Any] = {
        "config": asdict(config),
        "available_logs": None if available_logs is None else len(set(available_logs)),
        "splits": {},
    }
    chosen: List[int] = []
    for split_id, (split, target) in enumerate((("train", config.n_train), ("dev", config.n_dev))):
        in_split = cands["our_split"] == split
        rng = np.random.default_rng([config.seed, split_id])
        weights = None
        if split == "train" and config.balance_alpha > 0:
            in_pool = np.flatnonzero(pool & in_split)  # fixed rows included: the weights stay those of the full pool
            weights = np.zeros(len(codes))
            weights[in_pool] = balance_weights(cands, in_pool, config.balance_alpha)
        picked = [int(r) for r in fixed_rows[in_split[fixed_rows]]]
        n_fixed = len(picked)
        if n_fixed > target:
            raise RuntimeError(f"{n_fixed} fixed {split} rows exceed the target of {target}")
        n_fixed_other = int((~cruise[picked]).sum()) if picked else 0
        rows_other = np.flatnonzero(pool & in_split & ~cruise & ~is_fixed)
        rows_cruise = np.flatnonzero(pool & in_split & cruise & ~is_fixed)
        others = iter(_draw_order(rng, rows_other, None if weights is None else weights[rows_other]))
        cruisers = iter(_draw_order(rng, rows_cruise, None if weights is None else weights[rows_cruise]))
        skip = excluded if split == "train" and excluded is not None else None
        if skip is not None:  # the same draw order, masked rows skipped as they come up
            others = (r for r in others if not skip[r])
            cruisers = (r for r in cruisers if not skip[r])
        cruise_quota = int(np.floor(config.cruise_cap * target))
        stages: Dict[str, int] = {}
        # others_1 is capped at the room left: fixed rows that are mostly cruise would otherwise push it past the
        # target (a no-op without fixed rows, where target - quota <= target)
        n_others_1 = min(target - n_fixed, max(0, target - cruise_quota - n_fixed_other))
        for stage, rows_in, n in (("others_1", others, n_others_1),
                                  ("cruise", cruisers, None), ("others_2", others, None)):  # others_2: only if cruise ran out
            new = picker.take(rows_in, target - len(picked) if n is None else n)
            stages[stage] = len(new)
            picked += new
        n_before_fill = len(picked)
        if len(picked) < target:  # offer what is left log by log in time order (adds 0 rows, see the module docstring)
            left = pool & in_split if skip is None else pool & in_split & ~skip
            left[picked] = False
            log_order = rng.permutation(n_logs)
            for group in (left & ~cruise, left & cruise):
                rows = np.flatnonzero(group)
                rows = rows[np.lexsort((cands["t_us"][rows], log_order[codes[rows]]))]
                picked += picker.take(iter(rows), target - len(picked))
        stages["fill"] = len(picked) - n_before_fill
        rows = np.array(picked, dtype=np.int64)
        if len(rows) > target:  # a bug, so it raises
            raise RuntimeError(f"selected {len(rows)} {split} rows, more than the target of {target}")
        n_cruise = int(cruise[rows].sum())
        report["splits"][split] = {
            "candidates": int(in_split.sum()),
            "eligible": int((base & in_split).sum()),
            "after_thinning": int((pool & in_split).sum()),
            "pool_cruise": int((pool & in_split & cruise).sum()),
            "target": target,
            "selected": len(rows),
            "shortfall": target - len(rows),
            "fixed": n_fixed,
            "added": len(rows) - n_fixed,
            "stages": stages,
            "cruise_share": round(n_cruise / len(rows), 4) if len(rows) else None,
            "cruise_cap_exceeded": n_cruise > cruise_quota,
            "logs": len(set(codes[rows].tolist())),
            "city": stats.value_counts(cands["city"][rows]) if "city" in cands else None,
            "decision_lon": stats.value_counts(cands["decision_lon"][rows]),
            "decision_lat": stats.value_counts(cands["decision_lat"][rows]),
            "standstill_share": round(float(cands["standstill"][rows].mean()), 4) if len(rows) else None,
        }
        if skip is not None:  # only with a mask, so the reports of earlier selections are unchanged
            report["splits"][split]["train_excluded"] = {
                "candidates": int((in_split & skip).sum()), "eligible": int((base & in_split & skip).sum()),
                "after_thinning": int((pool & in_split & skip).sum()),
                "pool_left": int((pool & in_split & ~skip).sum())}
        chosen += picked
    selected = np.sort(np.array(chosen, dtype=np.int64))
    _check(cands, selected, base, codes, config.spacing_s, fixed=fixed_rows)
    if excluded is not None and (excluded[selected] & (cands["our_split"][selected] == "train")).any():  # a bug
        raise RuntimeError("selected a train row that is in train_excluded")
    return selected, report


def _check(cands: tables.Columns, selected: np.ndarray, base: np.ndarray, codes: np.ndarray, spacing_s: float,
           fixed: Optional[np.ndarray] = None) -> None:
    """Post-conditions; violations are bugs, so they raise even under ``python -O``."""
    if fixed is not None and not np.isin(fixed, selected).all():
        raise RuntimeError("a fixed row is not selected")
    if not base[selected].all():
        raise RuntimeError("selected a row that fails rule 2, 3 or 5")
    splits = cands["our_split"][selected]
    train_logs = set(codes[selected][splits == "train"].tolist())
    dev_logs = set(codes[selected][splits == "dev"].tolist())
    if train_logs & dev_logs:
        raise RuntimeError("a log has samples in both train and dev")
    order = selected[np.lexsort((cands["t_us"][selected], codes[selected]))]
    same_log = codes[order][1:] == codes[order][:-1]
    gaps = np.diff(cands["t_us"][order])[same_log]
    if gaps.size and gaps.min() < spacing_s * 1e6 - DT_TOL_US:
        raise RuntimeError("two selected anchors in one log are closer than the spacing")


def simulate(
    cands: tables.Columns, config: SelectionConfig, all_logs: Sequence[str], n_archives: int
) -> Dict[str, Any]:
    """Selection as if only the logs in ``n_archives`` camera archives had been downloaded (step 3).

    Which logs an archive holds is unknown until it is streamed, so the available logs are a random
    sample of the expected size, ``n_archives / 200`` of all logs (D).
    """
    logs = np.array(sorted(set(all_logs)), dtype=object)
    k = min(len(logs), max(1, int(round(len(logs) * n_archives / ARCHIVES_TOTAL))))
    available = np.random.default_rng([config.seed, 99]).choice(logs, size=k, replace=False).tolist()
    _, report = select(cands, config, available_logs=available)
    report["simulated_archives"] = n_archives
    return report
