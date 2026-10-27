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
   exceeded. Random order packs anchors ~25% below what the spacing allows, so if the target is
   still not met, a last pass fills the remaining gaps log by log in time order. Together they reach
   ~83% of the maximum on an unbroken log (D, measured on synthetic logs in the tests).
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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
    seed: int = 20261027


SMOKE = SelectionConfig("smoke", n_train=10_000, n_dev=1_000, spacing_s=1.0)
PILOT = SelectionConfig("v1-pilot", n_train=50_000, n_dev=2_000)


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


def select(
    cands: tables.Columns, config: SelectionConfig, available_logs: Optional[Iterable[str]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sorted indices of the selected rows, and statistics per split."""
    codes = log_codes(cands)
    n_logs = int(codes.max()) + 1 if len(codes) else 0
    base = eligible(cands, available_logs)
    pool = thin_standstill(cands, base, config.standstill_period_s, codes)
    cruise = is_cruise(cands)
    picker = _SpacedPicker(codes, cands["t_us"], config.spacing_s)
    report: Dict[str, Any] = {
        "config": asdict(config),
        "available_logs": None if available_logs is None else len(set(available_logs)),
        "splits": {},
    }
    chosen: List[int] = []
    for split_id, (split, target) in enumerate((("train", config.n_train), ("dev", config.n_dev))):
        in_split = cands["our_split"] == split
        rng = np.random.default_rng([config.seed, split_id])
        others = iter(rng.permutation(np.flatnonzero(pool & in_split & ~cruise)))
        cruisers = iter(rng.permutation(np.flatnonzero(pool & in_split & cruise)))
        cruise_quota = int(np.floor(config.cruise_cap * target))
        picked = picker.take(others, target - cruise_quota)
        picked += picker.take(cruisers, target - len(picked))
        picked += picker.take(others, target - len(picked))  # only if cruise anchors ran out
        if len(picked) < target:  # fill the gaps random order left: logs in random order, each in time order
            left = pool & in_split
            left[picked] = False
            log_order = rng.permutation(n_logs)
            for group in (left & ~cruise, left & cruise):
                rows = np.flatnonzero(group)
                rows = rows[np.lexsort((cands["t_us"][rows], log_order[codes[rows]]))]
                picked += picker.take(iter(rows), target - len(picked))
        rows = np.array(picked, dtype=np.int64)
        n_cruise = int(cruise[rows].sum())
        report["splits"][split] = {
            "candidates": int(in_split.sum()),
            "eligible": int((base & in_split).sum()),
            "after_thinning": int((pool & in_split).sum()),
            "pool_cruise": int((pool & in_split & cruise).sum()),
            "target": target,
            "selected": len(rows),
            "shortfall": target - len(rows),
            "cruise_share": round(n_cruise / len(rows), 4) if len(rows) else None,
            "cruise_cap_exceeded": n_cruise > cruise_quota,
            "logs": len(set(codes[rows].tolist())),
            "decision_lon": stats.value_counts(cands["decision_lon"][rows]),
            "decision_lat": stats.value_counts(cands["decision_lat"][rows]),
            "standstill_share": round(float(cands["standstill"][rows].mean()), 4) if len(rows) else None,
        }
        chosen += picked
    selected = np.sort(np.array(chosen, dtype=np.int64))
    _check(cands, selected, base, codes, config.spacing_s)
    return selected, report


def _check(cands: tables.Columns, selected: np.ndarray, base: np.ndarray, codes: np.ndarray, spacing_s: float) -> None:
    """Post-conditions; violations are bugs, so they raise even under ``python -O``."""
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
