"""Near-duplicate anchors and label conflicts among selected rows: v1-xl-nd and the label-conflict list.

Both are [proposed 2026-11-08] (v1-xl spec §2.2, §2.3); v1-xl itself is unchanged by them.

``near_duplicate``: two anchors of one log carry nearly the same lesson. All of:
* the same ``decision_lon`` and the same ``decision_lat``;
* |Δspeed| < ``dspeed`` (0.5 m/s);
* |Δheading| < ``dheading_deg`` (2°), read from the earlier anchor's ``path_xyyaw`` at the later
  anchor's offset k (0.5 s steps). ``frame_index`` must step by exactly k, otherwise the pair is not
  a duplicate;
* the two 8-point 4 s targets, each in its own anchor frame, are on average < ``target_ade_m``
  (0.6 m) apart (ADE).
This is the measure survey's definition (``DATA_ATLAS.md``) plus the target condition, which keeps
a start-up next to a standstill apart (V: 2026-11-08 prototype ``judge_nested.py``).

``nested_filter`` (v1-xl-nd): keep every base row (v1-large); visit the other rows in (log, time)
order and drop one if its city is gated (Las Vegas) and a kept row of its log within ``window_s``
(+50 ms) on either side is a near-duplicate of it. Kept rows are the earlier rows already kept and
the base rows ahead in time. It depends on nothing random, so v1-xl fixes it. On v1-xl train it
keeps 176,352 of 200,000 rows (V, 2026-11-09, with the warm-up exclusion; 176,232 in the first plan).
With the warm-up exclusion the base rows are v1-large's minus those in the window (``scene_exclusion``).

``label_conflicts``: the quality proposal's hard-bad tests over the rule-1 window (-1.5 … +4.0 s),
reported only; nothing is dropped. 52 rows of v1-xl train and 1 of dev (V, 2026-11-09; 54 in the first
plan); 38 command contradictions over the three sets (V).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from data_pipeline.v1 import selection, tables
from data_pipeline.v1.openscene_index import ANCHOR, DT_TOL_US, DT_US
from navsim_tools import geometry

GATED_CITIES = ("us-nv-las-vegas-strip",)
WINDOW_S = 1.5
DSPEED_MPS = 0.5
DHEADING_DEG = 2.0
TARGET_ADE_M = 0.6
TARGET = slice(ANCHOR + 1, ANCHOR + 9)  # the 8 target poses, 0.5 … 4.0 s
NEAR_DUP_COLUMNS = ("log_name", "t_us", "frame_index", "city", "decision_lon", "decision_lat", "speed", "path_xyyaw")

# Hard-bad tests (quality proposal, 2026-11-08)
CONFLICT_COLUMNS = ("decision_lat", "command", "speed_consistent", "accel_lon", "path_xyyaw", "path_t_s")
CORE = slice(ANCHOR - 3, ANCHOR + 9)  # the rule-1 window, -1.5 … +4.0 s
ACCEL_ERR_MAX = 3.0  # m/s², prompt acceleration vs the path's at the anchor
ACCEL_MAX = 8.0  # m/s², over the window
SLIP_MAX_DEG = 20.0  # direction of motion vs heading, on steps longer than 1 m
YAW_STILL_STEP_M, YAW_STILL_DPSI = 0.2, 0.10  # heading change > 0.1 rad while moving < 0.2 m in 0.5 s
REASONS = ("command_contradicts_turn", "speed_inconsistent", "accel_mismatch", "accel_gt_8", "slip_gt_20deg",
           "yaw_in_place")


def parameters() -> dict:
    """The thresholds, as recorded in the plan manifest."""
    return {"cities": list(GATED_CITIES), "window_s": WINDOW_S, "window_tolerance_s": DT_TOL_US / 1e6,
            "dspeed_mps": DSPEED_MPS, "dheading_deg": DHEADING_DEG, "target_ade_m": TARGET_ADE_M,
            "keeps": "every base row; earlier kept rows and later base rows count as neighbours"}


def near_duplicate(cands: tables.Columns, a: int, b: int, dspeed: float = DSPEED_MPS,
                   dheading_deg: float = DHEADING_DEG, target_ade_m: float = TARGET_ADE_M) -> bool:
    """Whether rows ``a`` and ``b`` (either order) are near-duplicates (module docstring)."""
    a, b = int(a), int(b)
    if cands["t_us"][a] > cands["t_us"][b]:
        a, b = b, a
    if cands["log_name"][a] != cands["log_name"][b]:
        return False
    path = cands["path_xyyaw"]
    k = int(round((int(cands["t_us"][b]) - int(cands["t_us"][a])) / DT_US))
    if not (0 < k < path.shape[1] - ANCHOR and int(cands["frame_index"][b]) - int(cands["frame_index"][a]) == k):
        return False
    dheading = float(path[a, ANCHOR + k, 2])
    if not np.isfinite(dheading) or abs(np.degrees(dheading)) >= dheading_deg:
        return False
    if cands["decision_lon"][a] != cands["decision_lon"][b] or cands["decision_lat"][a] != cands["decision_lat"][b]:
        return False
    if abs(float(cands["speed"][a]) - float(cands["speed"][b])) >= dspeed:
        return False
    d = path[a, TARGET, :2].astype(np.float64) - path[b, TARGET, :2].astype(np.float64)
    return bool(np.sqrt((d ** 2).sum(axis=1)).mean() < target_ade_m)


def nested_filter(cands: tables.Columns, rows: Iterable[int], base_rows: Iterable[int],
                  cities: Optional[Iterable[str]] = GATED_CITIES, window_s: float = WINDOW_S,
                  **thresholds: float) -> np.ndarray:
    """Sorted rows of ``rows`` that are kept (module docstring); ``base_rows`` ⊂ ``rows`` are always kept.

    ``cities=None`` gates every city. ``thresholds`` go to ``near_duplicate``.
    """
    rows = np.unique(np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64))
    base_rows = np.unique(np.asarray(list(base_rows) if not isinstance(base_rows, np.ndarray) else base_rows,
                                     dtype=np.int64))
    if not np.isin(base_rows, rows).all():
        raise ValueError("every base row must be one of the rows")
    codes = selection.log_codes(cands)
    t_us = cands["t_us"].astype(np.int64)
    gated = None if cities is None else set(cities)
    window_us = window_s * 1e6 + DT_TOL_US
    order = rows[np.lexsort((t_us[rows], codes[rows]))]
    is_base = np.isin(order, base_rows)
    keep = is_base.copy()
    n = len(order)
    for i in range(n):
        r = order[i]
        if keep[i] or (gated is not None and cands["city"][r] not in gated):
            keep[i] = True
            continue
        duplicate = False
        j = i - 1  # earlier rows that were kept
        while not duplicate and j >= 0 and codes[order[j]] == codes[r] and t_us[r] - t_us[order[j]] <= window_us:
            duplicate = keep[j] and near_duplicate(cands, order[j], r, **thresholds)
            j -= 1
        j = i + 1  # later rows are not decided yet, so only base rows count
        while not duplicate and j < n and codes[order[j]] == codes[r] and t_us[order[j]] - t_us[r] <= window_us:
            duplicate = is_base[j] and near_duplicate(cands, r, order[j], **thresholds)
            j += 1
        keep[i] = not duplicate
    return np.sort(order[keep])


def conflict_flags(cands: tables.Columns, rows: Iterable[int]) -> np.ndarray:
    """(len(rows), len(REASONS)) booleans: which hard-bad tests each row fails."""
    rows = np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64)
    p = cands["path_xyyaw"][rows][:, CORE].astype(np.float64)
    t = cands["path_t_s"][rows][:, CORE].astype(np.float64)
    d = np.diff(p[..., :2], axis=1)
    dt = np.diff(t, axis=1)
    step = np.linalg.norm(d, axis=2)
    vel = d / dt[..., None]
    tm = (t[:, 1:] + t[:, :-1]) / 2
    acc = np.diff(vel, axis=1) / np.diff(tm, axis=1)[..., None]
    amax = np.linalg.norm(acc, axis=2).max(axis=1)
    heading = p[:, :-1, 2] + geometry.normalize_angle(p[:, 1:, 2] - p[:, :-1, 2]) / 2
    slip = np.abs(geometry.normalize_angle(np.arctan2(d[..., 1], d[..., 0]) - heading))
    slip_fwd_deg = np.degrees(np.where((step > 1.0) & (slip < np.pi / 2), slip, 0).max(axis=1))
    dpsi = np.abs(geometry.normalize_angle(np.diff(p[..., 2], axis=1)))
    yaw_in_place = ((step < YAW_STILL_STEP_M) & (dpsi > YAW_STILL_DPSI)).any(axis=1)
    a = ANCHOR - CORE.start  # the anchor's column inside the window
    acc_path = (step[:, a] / dt[:, a] - step[:, a - 1] / dt[:, a - 1]) / (tm[:, a] - tm[:, a - 1])  # signed speed change
    command, lat = cands["command"][rows], cands["decision_lat"][rows]
    contra = ((command == "left") & (lat == "turn_right")) | ((command == "right") & (lat == "turn_left"))
    return np.stack([
        contra,
        ~cands["speed_consistent"][rows].astype(bool),
        np.abs(cands["accel_lon"][rows].astype(np.float64) - acc_path) > ACCEL_ERR_MAX,
        amax > ACCEL_MAX,
        slip_fwd_deg > SLIP_MAX_DEG,
        yaw_in_place,
    ], axis=1).reshape(len(rows), len(REASONS))


def label_conflicts(cands: tables.Columns, rows: Iterable[int]) -> List[Tuple[int, List[str]]]:
    """``(row, reasons)`` for each of ``rows`` (in the given order) that fails at least one hard-bad test."""
    rows = np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64)
    flags = conflict_flags(cands, rows)
    return [(int(r), [REASONS[k] for k in np.flatnonzero(f)]) for r, f in zip(rows, flags) if f.any()]
