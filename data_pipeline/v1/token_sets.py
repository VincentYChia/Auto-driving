"""Training sets defined by a token list or by rules, not by a draw: the v1.0-pool plan (PRETRAIN_V1_PLAN.md §11).

Researcher, 2026-11-09: "I like the navtrain data. I mean any extra data will always boost performance. So we
should aim to collect lots of data" [agreed]; 2026-11-08: curate for quality and variety, no overly repetitive
data [agreed]. Everything below is the mechanics, [proposed].

The four sets share v1-xl's pinned 2,000 dev rows and come from one train pool:

* ``train_pool`` (the set **v1-all**): every eligible TRAIN anchor of the index (rules 2, 3 and 5),
  standstill-thinned (rule 6: one anchor per 5 s in each log), outside the ±7 s window of every NAVSIM v1.1
  warm-up scene (``scene_exclusion``), at the index's own 0.5 s spacing. Rules 7 (spacing) and 8 (cruise cap)
  are not applied, so neighbours 0.5 s apart stay. 426,694 rows (V, 2026-11-09).
* ``listed_train_rows`` (**v1-nt**): the rows of NAVSIM's navtrain token list that pass ``LIST_RULES`` in order.
  An excluded token's reason is the FIRST rule it fails, so each has exactly one; ``also_fail_a_later_rule``
  counts the ones a later rule would have excluded too (800 navtrain rows in dev logs are also inside the
  warm-up window, V). 80,626 rows (V). Its rows are not standstill-thinned (NAVSIM's own scenes).
* ``mix_rows`` with ``XL_MIX`` (**v1-xl-mix**, option A of the 2026-11-09 proposal): v1-xl-nd's train rows,
  plus the pool rows not already in v1-xl-nd that are (a) outside Las Vegas and not stay-stopped, or (b) in Las
  Vegas and a turn, sideways shift or brake to a stop; then stay-stopped is thinned to 1% of the set by the
  draw pinned in ``MixRule`` (``DRAW``). 238,034 rows in 1,084 logs (V). The maneuver labels are
  ``navsim_tools.maneuvers`` on the index path (+0.5 … +4.0 s) and speed (``maneuver_labels``).
* **v1-nt-mix** = v1-nt ∪ v1-xl-mix (259,929, V). **v1-all** = the pool ∪ v1-nt, so it holds every other set
  and its frames (with dev) are all the folder extracts; on the real index v1-nt adds nothing to the pool
  (every v1-nt row is a pool row, V), so v1-all is the pool.

``check_sets`` is the post-condition (``selection._check`` at the index's 0.5 s spacing): every train row is an
eligible train-log row outside the window, and no log has rows in both train and dev.

**v1.0-full** (PRETRAIN_V1_PLAN.md §12). Researcher, 2026-11-10: "actively building the dataset in as large steps
as possible. Its a safe assumption to think that more data will do better, so long as it meets quality standards.
Additionally its much faster to trim data then it is to get new data. Try to max out the available data first then
train." [agreed]; and on re-splitting dev, "If it does not harm the integrity of training go for it" [agreed]. The
mechanics are [proposed]. The old pinned dev (dev-v1) spans 221 of the 225 nuPlan val logs, so the log-level split
kept every row of those logs out of training. The new dev stays log-level, from few whole drives:
* ``choose_dev_logs`` (``DEV_V2``): about 40 nuPlan val logs, in whole nuPlan drives (``drive_of``: the log name
  without its two segment fields; nuPlan's own train/val split is by drive, so no val drive holds a train log, V),
  per city as many logs as navtest's city mix gives (``NAVTEST_CITY_SHARE``), and among those the drives holding
  the fewest navtrain rows that pass the rules (so most navtrain rows are released to training). Exact: a
  dynamic program per city; no randomness.
* ``draw_dev`` (**dev-v2**): 2,000 anchors from those logs, per city navtest's share; first the logs' navtrain
  tokens, then the other rows with a known route command (``fill_needs_command``: no NAVSIM scene lacks one, V),
  standstill-thinned (rule 6), all >= 1.0 s apart, in the order pinned in ``DEV_DRAW``.
* ``every_row`` keeps every index anchor that passes the quality rules (rules 2, 3, 5: ``selection.eligible``) and
  lies outside the warm-up window, with no standstill thinning (rule 6), spacing (rule 7) or cruise cap (rule 8):
  trimming is left to the views. **v1-full** (``FULL``): those rows of the TRAIN logs and of the released val logs
  (every val log but dev-v2's), with dev-v2 as dev. **v1-full-devlogs** (``FULL_DEVLOGS``): those rows of dev-v2's
  logs, as dev (no train rows): extracted, in no training view.
A row's split in these sets is the set's token files', not the index's ``our_split`` (which stays nuPlan's: val
logs are ``dev``). ``check_full`` is the post-condition.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data_pipeline.v1 import selection, stats, tables
from data_pipeline.v1.openscene_index import DT_TOL_US, DT_US, FUTURE_REQUIRED, PAST_REQUIRED, PATH_OFFSETS_S
from navsim_tools import maneuvers

POOL, NT_MIX, MIX, NT = "v1-all", "v1-nt-mix", "v1-xl-mix", "v1-nt"
SETS = (POOL, NT_MIX, MIX, NT)  # largest first; not nested: v1-nt and v1-xl-mix overlap, neither holds the other
STANDSTILL_PERIOD_S = 5.0  # rule 6, as in every v1 set (SelectionConfig.standstill_period_s)
SPACING_S = 0.5  # the index's frame spacing: only the post-condition's minimum gap (rule 7 is not applied)
LIST_RULES = (  # (name, what a row must be), in the order an excluded listed token is explained
    ("index_row", "a row of the index (rule 1: the -1.5 ... +4.0 s window exists with no timing gap)"),
    ("train_log", "in a log the index assigns to TRAIN (our_split: dev = nuPlan val logs, excluded = NAVSIM-guarded)"),
    ("eligible", "eligible: CAM_F0 path, sane pose, not guarded (rules 2, 3, 5: selection.eligible)"),
    ("outside_window", "outside the window of every warm-up scene in its log (scene_exclusion.window_mask)"),
)


# --- the pool -----------------------------------------------------------------------------------


def _bool_mask(mask: Any, n: int, what: str) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (n,):
        raise ValueError(f"{what} must be a boolean mask of {n} rows")
    return mask


def train_pool(cands: tables.Columns, train_excluded: np.ndarray,
               period_s: float = STANDSTILL_PERIOD_S) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The pool mask (eligible train rows, thinned, outside ``train_excluded``) and its counts per rule.

    Thinning comes before the window, as in ``selection.select``: the window only removes rows, so every row it
    keeps is a row the thinning kept (the order of v1-xl's draw, PRETRAIN_V1_PLAN.md §10).
    """
    excluded = _bool_mask(train_excluded, len(cands["t_us"]), "train_excluded")
    train = cands["our_split"] == "train"
    eligible = selection.eligible(cands) & train
    thinned = selection.thin_standstill(cands, eligible, period_s)
    pool = thinned & ~excluded
    report = {
        "rules": "eligible TRAIN anchors (rules 2, 3, 5), standstill-thinned to one per "
                 f"{period_s:g} s per log (rule 6), outside the warm-up window; the index's 0.5 s spacing "
                 "(no rule 7 spacing, no rule 8 cruise cap)",
        "train_rows": int(train.sum()), "eligible": int(eligible.sum()),
        "after_standstill_thinning": int(thinned.sum()), "after_window": int(pool.sum()),
        "standstill_period_s": period_s,
    }
    return pool, report


# --- a token list: v1-nt ------------------------------------------------------------------------


@dataclass
class ListedRows:
    """``listed_train_rows``' result. ``dropped``: rule → the index rows that rule excludes first."""

    selected: np.ndarray
    unindexed: List[str]
    dropped: Dict[str, np.ndarray]
    report: Dict[str, Any]


def listed_train_rows(cands: tables.Columns, tokens: Iterable[str], train_excluded: np.ndarray) -> ListedRows:
    """The TRAIN rows of a token list (NAVSIM's navtrain: v1-nt) that pass every rule of ``LIST_RULES``.

    Every listed token is tested against the rules in order. An excluded token's reason is the first rule it
    fails, so the reasons partition the excluded tokens; ``also_fail_a_later_rule`` says how many of them a
    later rule would have excluded as well. ``train_excluded`` is the warm-up window mask over all rows (dev
    rows included, so a dev-log row can also be counted as inside it).
    """
    tokens = list(tokens)
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{len(tokens) - len(set(tokens))} tokens are listed twice")
    excluded = _bool_mask(train_excluded, len(cands["t_us"]), "train_excluded")
    row_of = {t: i for i, t in enumerate(cands["token"].tolist())}  # np.isin on strings takes minutes
    unindexed = sorted(t for t in tokens if t not in row_of)
    rows = np.array(sorted(row_of[t] for t in tokens if t in row_of), dtype=np.int64)
    tests = [("train_log", cands["our_split"][rows] == "train"),
             ("eligible", selection.eligible(cands)[rows]),
             ("outside_window", ~excluded[rows])]
    alive = np.ones(len(rows), dtype=bool)
    dropped: Dict[str, np.ndarray] = {}
    left = {"index_row": len(rows)}
    later: Dict[str, Dict[str, int]] = {}
    for k, (name, ok) in enumerate(tests):
        first = alive & ~ok
        dropped[name] = rows[first]
        if k + 1 < len(tests):
            later[name] = {other: int((first & ~ok_other).sum()) for other, ok_other in tests[k + 1:]}
        alive &= ok
        left[name] = int(alive.sum())
    selected = rows[alive]
    report = {
        "tokens": len(tokens),
        "rules": [f"{name}: {text}" for name, text in LIST_RULES],
        "left_after": left,
        "excluded_by_first_failing_rule": {"index_row": len(unindexed), **{n: len(r) for n, r in dropped.items()}},
        "also_fail_a_later_rule": later,
        "train_log_failures_by_split": stats.value_counts(cands["our_split"][dropped["train_log"]]),
        "selected": len(selected),
    }
    if len(unindexed) + sum(len(r) for r in dropped.values()) + len(selected) != len(tokens):  # a bug, so it raises
        raise RuntimeError("the reasons do not partition the listed tokens")
    return ListedRows(selected=selected, unindexed=unindexed, dropped=dropped, report=report)


def explain_unindexed(source: str, tokens: Iterable[str]) -> Dict[str, Any]:
    """Why listed tokens are not index rows, from one scan of OpenScene's metadata (every log, ~70 s, V).

    For each token found as a frame: too close to its log's start or end for the window, or which part of the
    window (-1.5 … 0 s history, 0 … +4.0 s future) has a link that is not 0.5 s ± 0.05 s (rule 1, as
    ``openscene_index.index_log`` checks it, frames in time order). Aggregates only: counts, no tokens.
    """
    from data_pipeline.v1 import openscene_index, safe_pickle

    wanted = set(tokens)
    why: Counter = Counter()
    largest: Counter = Counter()
    logs: Counter = Counter()
    seen = set()
    scanned = 0
    for _, data in (openscene_index.iter_log_pickles(source) if wanted else ()):
        frames = safe_pickle.loads(data)
        scanned += 1
        hits = [k for k, f in enumerate(frames) if str(f["token"]) in wanted]
        if not hits:
            continue
        order = sorted(range(len(frames)), key=lambda k: int(frames[k]["timestamp"]))
        position = {k: j for j, k in enumerate(order)}
        t_us = np.array([int(frames[k]["timestamp"]) for k in order], dtype=np.int64)
        for k in hits:
            seen.add(str(frames[k]["token"]))
            logs[str(frames[k]["log_name"])] += 1
            j = position[k]
            if j < PAST_REQUIRED or j + FUTURE_REQUIRED > len(order) - 1:
                why["too close to the log's start or end"] += 1
                continue
            links = np.arange(j - PAST_REQUIRED, j + FUTURE_REQUIRED)  # frame a to a + 1
            dt = t_us[links + 1] - t_us[links]
            bad = np.abs(dt - DT_US) > DT_TOL_US
            past, future = bool(bad[links < j].any()), bool(bad[links >= j].any())
            why[{(True, False): "timing gap in the history only", (False, True): "timing gap in the future only",
                 (True, True): "timing gaps in the history and the future",
                 (False, False): "no gap found (unexplained)"}[(past, future)]] += 1
            if bad.any():
                largest[f"{dt[bad].max() / 1e6:.1f} s"] += 1
    missing = len(wanted - seen)
    if missing:
        why["not a frame of the metadata"] += missing
    return {"tokens": len(wanted), "logs_scanned": scanned, "frames_found": len(seen), "in_logs": len(logs),
            "why": stats.value_counts(why.elements()), "largest_gap": stats.value_counts(largest.elements())}


# --- a mix rule: v1-xl-mix ----------------------------------------------------------------------


@dataclass(frozen=True)
class MixRule:
    """A base set plus pool rows by city and maneuver, with one maneuver thinned to a share (``mix_rows``).

    Outside ``gated_cities`` every maneuver but ``stay`` is added; inside them only ``gated_add``. ``stay`` is
    never added, and in the union it is thinned to ``stay_share`` of the set: ``keep`` rows are drawn by
    ``DRAW``, pinned here so the token files reproduce byte for byte (the seed alone does not fix them: other
    row orders or draw methods give other files, and floor instead of round another count, V 2026-11-09).
    """

    name: str
    gated_cities: Tuple[str, ...]
    gated_add: Tuple[str, ...]
    stay: str
    stay_share: float
    seed: Tuple[int, ...]

    def keep(self, n_other: int) -> int:
        """Rows of ``stay`` kept with ``n_other`` other rows: round(n_other * share / (1 - share)); n/99 at 1%."""
        return int(round(n_other * self.stay_share / (1.0 - self.stay_share)))


DRAW = ("numpy.random.default_rng(list(seed)).choice(<the union's stay rows in ascending index-row order>, "
        "size=min(keep(n_other), n_stay), replace=False)")
XL_MIX = MixRule(name=MIX, gated_cities=("us-nv-las-vegas-strip",),
                 gated_add=(maneuvers.TURN_LEFT, maneuvers.TURN_RIGHT, maneuvers.SIDEWAYS, maneuvers.BRAKE),
                 stay=maneuvers.STAY, stay_share=0.01, seed=(20261027, 31))


def maneuver_labels(cands: tables.Columns) -> np.ndarray:
    """``navsim_tools.maneuvers.maneuver`` of every row: its stored path at +0.5 … +4.0 s and its speed."""
    human = maneuvers.future_poses(cands["path_xyyaw"], PATH_OFFSETS_S)
    return maneuvers.maneuver(human, np.asarray(cands["speed"], dtype=np.float64))


def mix_rows(cands: tables.Columns, base_rows: Iterable[int], pool: np.ndarray, labels: np.ndarray,
             rule: MixRule = XL_MIX) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sorted rows of ``rule``'s mix (class docstring) and its report. Every base row must be a pool row."""
    n = len(cands["t_us"])
    pool = _bool_mask(pool, n, "pool")
    labels = np.asarray(labels, dtype=object)
    if labels.shape != (n,):
        raise ValueError(f"labels must give one label for each of the {n} rows")
    base_rows = np.unique(np.asarray(list(base_rows) if not isinstance(base_rows, np.ndarray) else base_rows,
                                     dtype=np.int64))
    if len(base_rows) and not pool[base_rows].all():
        raise ValueError(f"{int((~pool[base_rows]).sum())} base rows are not pool rows")
    in_base = np.zeros(n, dtype=bool)
    in_base[base_rows] = True
    gated = np.zeros(n, dtype=bool)
    for city in rule.gated_cities:
        gated |= cands["city"] == city
    stay = labels == rule.stay
    gated_add = np.zeros(n, dtype=bool)
    for label in rule.gated_add:
        gated_add |= labels == label
    candidates = pool & ~in_base
    add_other = candidates & ~gated & ~stay
    add_gated = candidates & gated & gated_add
    union = in_base | add_other | add_gated
    stay_rows = np.flatnonzero(union & stay)  # ascending index-row order: the draw's input (DRAW)
    n_other = int(union.sum()) - len(stay_rows)
    k = min(rule.keep(n_other), len(stay_rows))
    kept = np.zeros(0, dtype=np.int64)
    if k:
        kept = np.sort(np.random.default_rng(list(rule.seed)).choice(stay_rows, size=k, replace=False))
    out = union & ~stay
    out[kept] = True
    rows = np.flatnonzero(out)

    def counts(mask: np.ndarray) -> Dict[str, int]:
        return stats.value_counts(labels[mask])

    left_out = in_base & ~out
    report = {
        "rule": {**asdict(rule), "draw": DRAW,
                 "keep": "round(n_other * stay_share / (1 - stay_share)), n_other = union rows that are not stay"},
        "base": len(base_rows), "pool": int(pool.sum()), "pool_not_in_base": int(candidates.sum()),
        "added_outside_gated": int(add_other.sum()), "added_outside_gated_by_maneuver": counts(add_other),
        "left_out_outside_gated": counts(candidates & ~gated & ~add_other),
        "added_gated": int(add_gated.sum()), "added_gated_by_maneuver": counts(add_gated),
        "left_out_gated": counts(candidates & gated & ~add_gated),
        "union": int(union.sum()), "stay_in_union": len(stay_rows),
        "stay_in_union_from_base": int(in_base[stay_rows].sum()), "other_in_union": n_other,
        "stay_kept": int(k), "selected": len(rows),
        "base_left_out": int(left_out.sum()), "base_left_out_are_all_stay": bool(stay[left_out].all()),
        "stay_share": round(len(kept) / len(rows), 5) if len(rows) else None,
    }
    return rows, report


# --- the four sets, and their post-conditions ---------------------------------------------------


def union_sets(pool: np.ndarray, nt_rows: np.ndarray, mix: np.ndarray) -> Dict[str, np.ndarray]:
    """Train rows of the four sets, largest first (``SETS``): v1-all = pool ∪ v1-nt, v1-nt-mix = v1-nt ∪ mix."""
    pool_rows = np.flatnonzero(pool)
    return {POOL: np.union1d(pool_rows, nt_rows).astype(np.int64), NT_MIX: np.union1d(nt_rows, mix).astype(np.int64),
            MIX: np.asarray(mix, dtype=np.int64), NT: np.asarray(nt_rows, dtype=np.int64)}


def check_sets(cands: tables.Columns, train_sets: Mapping[str, np.ndarray], dev_rows: np.ndarray,
               train_excluded: np.ndarray) -> None:
    """Post-conditions; violations are bugs, so they raise even under ``python -O``.

    Each set's train rows are TRAIN-log rows outside ``train_excluded``, and with ``dev_rows`` they pass
    ``selection._check`` (eligible, no log in both train and dev, distinct frames ≥ 0.45 s apart). v1-nt and
    v1-xl-mix are inside v1-nt-mix, which is inside v1-all.
    """
    excluded = _bool_mask(train_excluded, len(cands["t_us"]), "train_excluded")
    codes = selection.log_codes(cands)
    eligible = selection.eligible(cands)
    for name, rows in train_sets.items():
        rows = np.asarray(rows, dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"{name}: a train row is listed twice")
        if (cands["our_split"][rows] != "train").any():
            raise RuntimeError(f"{name}: a train row is not in a TRAIN log")
        if excluded[rows].any():
            raise RuntimeError(f"{name}: a train row is inside the warm-up window")
        selection._check(cands, np.sort(np.concatenate([rows, dev_rows])), eligible, codes, SPACING_S)
    if set(train_sets) == set(SETS):
        inside = {(NT, NT_MIX), (MIX, NT_MIX), (NT_MIX, POOL)}
        for small, large in sorted(inside):
            if not np.isin(train_sets[small], train_sets[large]).all():
                raise RuntimeError(f"{small} is not inside {large}")


# --- every row and a new dev: v1.0-full ---------------------------------------------------------

FULL, FULL_DEVLOGS = "v1-full", "v1-full-devlogs"
FULL_SETS = (FULL, FULL_DEVLOGS)
FULL_RULES = (  # (name, what a row must be), in the order an excluded row is explained
    ("index_row", "a row of the index (rule 1: the -1.5 ... +4.0 s window exists with no timing gap)"),
    ("eligible", "eligible: CAM_F0 path, sane pose, not guarded (rules 2, 3, 5: selection.eligible)"),
    ("outside_window", "outside the window of every warm-up scene in its log (scene_exclusion.window_mask)"),
)
GROUPS = ("train_logs", "released_val_logs", "held_out_val_logs")  # every_row's report, by where a row's log is
# navtest's city mix, % of its 12,146 scenes (the committed failure analysis, PRETRAIN_V1_PLAN.md §11), in this order
NAVTEST_CITY_SHARE = (("us-nv-las-vegas-strip", 0.335), ("us-ma-boston", 0.307),
                      ("us-pa-pittsburgh-hazelwood", 0.201), ("sg-one-north", 0.158))
DRIVE = re.compile(r"^(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)_\d+_\d+$")


@dataclass(frozen=True)
class DevRule:
    """A log-level dev from few whole nuPlan val drives (``choose_dev_logs``, ``draw_dev``)."""

    name: str
    city_share: Tuple[Tuple[str, float], ...]
    n_logs: int
    n_anchors: int
    spacing_s: float
    standstill_period_s: float
    seed: Tuple[int, ...]
    fill_needs_command: bool = True  # the fill skips rows whose route command is unknown (no NAVSIM scene has one)


DEV_V2 = DevRule(name="dev-v2", city_share=NAVTEST_CITY_SHARE, n_logs=40, n_anchors=2_000, spacing_s=1.0,
                 standstill_period_s=STANDSTILL_PERIOD_S, seed=(20261110, 2))
UNKNOWN_COMMAND = "unknown"  # the index's command when OpenScene's one-hot gives none (0 of navtrain's rows, V)
DEV_LOGS_RULE = ("per city, logs = largest-remainder share of n_logs by city_share; among the nuPlan val drives of the "
                 "city whose every index log is a val log, the whole drives whose logs sum to that count with the "
                 "fewest navtrain rows that pass the rules (eligible, outside the warm-up window); ties: more rows "
                 "that pass the rules, then the sorted drive names (exact dynamic program, no randomness)")
DEV_DRAW = ("one numpy.random.default_rng(list(seed)); per city in city_share order, quota = largest-remainder share "
            "of n_anchors: rng.permutation(<the city's navtrain rows that pass the rules, ascending index-row order>), "
            "then rng.permutation(<its other rows that pass the rules and (fill_needs_command) have a known route "
            "command, standstill-thinned (rule 6), ascending>), each "
            "row taken in that order when >= spacing_s - 0.05 s from every row taken in its log, until the quota")


def drive_of(log_name: str) -> str:
    """The nuPlan drive of a log: its name without the two segment fields (``<date>.<time>_<vehicle>``)."""
    m = DRIVE.match(str(log_name))
    if m is None:
        raise ValueError(f"{log_name} is not a nuPlan log name (<date>.<time>_veh-<n>_<start>_<end>)")
    return m.group(1)


def quotas(total: int, shares: Sequence[Tuple[str, float]]) -> Dict[str, int]:
    """``total`` split by ``shares`` (normalised), largest remainder; ties go to the earlier name."""
    weight = sum(s for _, s in shares)
    exact = [(name, total * s / weight) for name, s in shares]
    out = {name: int(np.floor(x)) for name, x in exact}
    order = sorted(range(len(exact)), key=lambda k: (-(exact[k][1] - np.floor(exact[k][1])), k))
    for k in order[: total - sum(out.values())]:
        out[exact[k][0]] += 1
    return out


def _passing(cands: tables.Columns, window: np.ndarray) -> np.ndarray:
    return selection.eligible(cands) & ~_bool_mask(window, len(cands["t_us"]), "window")


def in_logs(cands: tables.Columns, logs: Iterable[str]) -> np.ndarray:
    """Rows whose log is one of ``logs`` (a set lookup: ``np.isin`` on strings takes minutes)."""
    wanted = set(logs)
    return np.array([log in wanted for log in cands["log_name"].tolist()], dtype=bool)


def choose_dev_logs(cands: tables.Columns, window: np.ndarray, rule: DevRule = DEV_V2
                    ) -> Tuple[List[str], Dict[str, Any]]:
    """dev-v2's held-out logs (``DEV_LOGS_RULE``) and a report. Needs ``in_navtrain`` and ``city`` columns."""
    ok = _passing(cands, window)
    logs = cands["log_name"]
    split_of: Dict[str, set] = {}
    for log, split in zip(logs.tolist(), cands["our_split"].tolist()):
        split_of.setdefault(log, set()).add(split)
    drives: Dict[str, Dict[str, Any]] = {}
    for log in sorted(split_of):
        d = drives.setdefault(drive_of(log), {"logs": [], "splits": set()})
        d["logs"].append(log)
        d["splits"] |= split_of[log]
    log_codes = {log: k for k, log in enumerate(sorted(split_of))}
    code = np.array([log_codes[log] for log in logs.tolist()], dtype=np.int64)
    rows_ok = np.bincount(code[ok], minlength=len(log_codes))
    navtrain_ok = np.bincount(code[ok & cands["in_navtrain"]], minlength=len(log_codes))
    city_of: Dict[str, set] = {}
    for log, city in zip(logs.tolist(), cands["city"].tolist()):
        city_of.setdefault(log, set()).add(city)
    want = quotas(rule.n_logs, rule.city_share)
    chosen: List[str] = []
    per_city: Dict[str, Any] = {}
    mixed = sorted(name for name, d in drives.items() if len(d["splits"]) > 1)
    for city, _ in rule.city_share:
        units = []
        for name in sorted(drives):
            d = drives[name]
            cities = set().union(*(city_of[log] for log in d["logs"]))
            if d["splits"] != {"dev"} or cities != {city}:
                continue
            units.append((name, len(d["logs"]), int(sum(navtrain_ok[log_codes[l]] for l in d["logs"])),
                          int(sum(rows_ok[log_codes[l]] for l in d["logs"]))))
        # best[k] = (navtrain rows, -rows, drive names) of the best set of whole drives holding k logs
        best: Dict[int, Tuple[int, int, Tuple[str, ...]]] = {0: (0, 0, ())}
        for name, n, nav, rows in units:
            for k, (b_nav, b_rows, b_names) in sorted(best.items(), reverse=True):
                cand = (b_nav + nav, b_rows - rows, tuple(sorted(b_names + (name,))))
                if k + n not in best or cand < best[k + n]:
                    best[k + n] = cand
        k = min(best, key=lambda j: (abs(j - want[city]), j))  # the exact count when it can be reached
        nav, neg_rows, names = best[k]
        city_logs = sorted(l for name in names for l in drives[name]["logs"])
        chosen += city_logs
        per_city[city] = {"logs_wanted": want[city], "logs": len(city_logs), "drives": list(names),
                          "navtrain_rows_held_out": nav, "rows_held_out": -neg_rows,
                          "val_drives_of_the_city": len(units),
                          "navtrain_rows_of_the_city": sum(u[2] for u in units)}
    report = {"rule": DEV_LOGS_RULE, "per_city": per_city, "logs": len(chosen),
              "drives": sum(len(c["drives"]) for c in per_city.values()),
              "drives_with_logs_in_two_splits": mixed,
              "navtrain_rows_held_out": sum(c["navtrain_rows_held_out"] for c in per_city.values()),
              "navtrain_rows_in_val_logs": int((ok & cands["in_navtrain"] & (cands["our_split"] == "dev")).sum())}
    return sorted(chosen), report


def draw_dev(cands: tables.Columns, window: np.ndarray, logs: Iterable[str], rule: DevRule = DEV_V2
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """dev-v2's anchors from ``logs`` (``DEV_DRAW``), sorted, and a report; raises when a city's quota is not met."""
    ok = _passing(cands, window)
    held = in_logs(cands, logs)
    codes = selection.log_codes(cands)
    picker = selection._SpacedPicker(codes, cands["t_us"], rule.spacing_s)
    rng = np.random.default_rng(list(rule.seed))
    want = quotas(rule.n_anchors, rule.city_share)
    out: List[int] = []
    per_city: Dict[str, Any] = {}
    for city, _ in rule.city_share:
        pool = ok & held & (cands["city"] == city)
        listed = np.flatnonzero(pool & cands["in_navtrain"])
        fill = pool & ~cands["in_navtrain"]
        if rule.fill_needs_command:
            fill &= cands["command"] != UNKNOWN_COMMAND
        other = np.flatnonzero(selection.thin_standstill(cands, fill, rule.standstill_period_s, codes))
        first = picker.take(iter(rng.permutation(listed).tolist()), want[city])
        rest = picker.take(iter(rng.permutation(other).tolist()), want[city] - len(first))
        if len(first) + len(rest) < want[city]:
            raise ValueError(f"{rule.name}: {city} gives {len(first) + len(rest)} of its {want[city]} anchors "
                             f">= {rule.spacing_s} s apart; the logs chosen hold too few rows")
        out += first + rest
        per_city[city] = {"anchors": want[city], "navtrain_tokens": len(first), "other_rows": len(rest),
                          "navtrain_rows_offered": len(listed), "other_rows_offered": len(other)}
    rows = np.sort(np.array(out, dtype=np.int64))
    report = {"rule": {**asdict(rule), "draw": DEV_DRAW, "logs_rule": DEV_LOGS_RULE}, "per_city": per_city,
              "anchors": len(rows), "navtrain_tokens": sum(c["navtrain_tokens"] for c in per_city.values())}
    return rows, report


def every_row(cands: tables.Columns, window: np.ndarray, held_out_logs: Iterable[str]
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Every index row that passes ``FULL_RULES``: ``train`` (TRAIN logs and the released val logs) and ``held_out``
    (``held_out_logs``, which must be val logs), with the counts per rule in each of ``GROUPS``.

    No standstill thinning, spacing or cruise cap. ``window`` is the warm-up window mask over all rows. An excluded
    row's reason is the first rule it fails. Rows of ``excluded`` (NAVSIM-guarded) logs fail ``eligible``.
    """
    n = len(cands["t_us"])
    window = _bool_mask(window, n, "window")
    held_out_logs = sorted(set(held_out_logs))
    held = in_logs(cands, held_out_logs)
    if (cands["our_split"][held] != "dev").any():
        raise ValueError("a held-out log is not a val (dev) log of the index")
    eligible = selection.eligible(cands)
    keep = eligible & ~window
    is_train, is_val = cands["our_split"] == "train", cands["our_split"] == "dev"
    groups = {"train_logs": is_train, "released_val_logs": is_val & ~held, "held_out_val_logs": held}
    by_group: Dict[str, Any] = {}
    for name, mask in groups.items():
        kept = mask & keep
        by_group[name] = {
            "left_after": {"index_row": int(mask.sum()), "eligible": int((mask & eligible).sum()),
                           "outside_window": int(kept.sum())},
            "excluded_by_first_failing_rule": {"eligible": int((mask & ~eligible).sum()),
                                               "outside_window": int((mask & eligible & window).sum())},
            "standstill_rows_kept": int((kept & cands["standstill"]).sum()),
            "logs": len(set(cands["log_name"][kept].tolist())),
        }
    rows = {"train": np.flatnonzero((is_train | (is_val & ~held)) & keep), "held_out": np.flatnonzero(held & keep)}
    report = {
        "rules": [f"{name}: {text}" for name, text in FULL_RULES],
        "not_applied": "standstill thinning (rule 6), spacing (rule 7), cruise cap (rule 8): trimming is a view choice",
        "by_group": by_group,
        "held_out_logs": len(held_out_logs),
        "rows_in_other_splits": int((~(is_train | is_val)).sum()),
    }
    return rows, report


def check_full(cands: tables.Columns, train: np.ndarray, held_out: np.ndarray, dev: np.ndarray, window: np.ndarray,
               held_out_logs: Iterable[str], base_train: Optional[np.ndarray] = None,
               spacing_s: float = DEV_V2.spacing_s) -> None:
    """Post-conditions of the v1.0-full sets; violations are bugs, so they raise even under ``python -O``.

    Every train and held-out row passes the rules; the held-out logs are whole val drives, and no train row is in
    one of their drives; dev (dev-v2) lies inside the held-out rows, its anchors ``spacing_s`` apart in each log;
    ``base_train`` (v1.0-pool's v1-all) lies inside the train rows.
    """
    excluded = _bool_mask(window, len(cands["t_us"]), "window")
    ok = selection.eligible(cands) & ~excluded
    held_out_logs = set(held_out_logs)
    for name, rows in (("train", train), ("held-out", held_out), ("dev", dev)):
        rows = np.asarray(rows, dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"{name}: a row is listed twice")
        if not ok[rows].all():
            raise RuntimeError(f"{name}: a row is ineligible or inside the warm-up window")
    if (cands["our_split"][in_logs(cands, held_out_logs)] != "dev").any():
        raise RuntimeError("a held-out log is not a val log")
    held_drives = {drive_of(log) for log in held_out_logs}
    other_logs = set(cands["log_name"].tolist()) - held_out_logs
    if any(drive_of(log) in held_drives for log in other_logs):
        raise RuntimeError("a held-out drive has a log that is not held out")
    train_logs = set(cands["log_name"][train].tolist())
    if train_logs & held_out_logs or {drive_of(log) for log in train_logs} & held_drives:
        raise RuntimeError("a train row is in a held-out log or drive")
    if set(cands["log_name"][held_out].tolist()) - held_out_logs:
        raise RuntimeError("a held-out row is not in a held-out log")
    if not np.isin(dev, held_out).all():
        raise RuntimeError("a dev row is not a held-out row")
    order = np.asarray(dev)[np.lexsort((cands["t_us"][dev], cands["log_name"][dev]))]
    same = cands["log_name"][order][1:] == cands["log_name"][order][:-1]
    gaps = np.diff(cands["t_us"][order])[same]
    if gaps.size and gaps.min() < spacing_s * 1e6 - DT_TOL_US:
        raise RuntimeError(f"two dev anchors in one log are closer than {spacing_s} s")
    if base_train is not None and not np.isin(base_train, train).all():
        raise RuntimeError(f"{int((~np.isin(base_train, train)).sum())} base train rows are not train rows")


def maneuver_counts(labels: np.ndarray, rows: np.ndarray) -> Dict[str, int]:
    """Maneuver label counts of ``rows``, in ``maneuvers.LABELS`` order (0 where absent)."""
    found = Counter(np.asarray(labels, dtype=object)[rows].tolist())
    return {label: int(found.get(label, 0)) for label in maneuvers.LABELS}


def describe_train(cands: tables.Columns, labels: np.ndarray, rows: np.ndarray,
                   reference: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, Any]:
    """Shares of one train set that the selection reports do not give: maneuvers, and overlaps with ``reference``."""
    rows = np.asarray(rows, dtype=np.int64)
    out: Dict[str, Any] = {"maneuver": maneuver_counts(labels, rows),
                           "logs": len(set(cands["log_name"][rows].tolist()))}
    if reference:
        out["shared_with"] = {name: int(np.isin(rows, ref).sum()) for name, ref in reference.items()}
    return out
