"""Checks on the v1.0-pool build (``docs/training_data/PRETRAIN_V1_PLAN.md`` §11) [proposed].

Run from the repo root; ``build_v1_pool.sh setup`` and ``build_v1_pool.sbatch`` call it between steps:

    python scripts/anvil/data_pipeline/verify_v1_pool.py splits     --out <v1.0-pool> [--xl <v1.0-xl>]
    python scripts/anvil/data_pipeline/verify_v1_pool.py rules      --out <v1.0-pool> --metadata <tgz>
    python scripts/anvil/data_pipeline/verify_v1_pool.py leakage    --out <v1.0-pool> --metadata <tgz> \
        [--navsim-root <devkit>]...
    python scripts/anvil/data_pipeline/verify_v1_pool.py fetched    --out <v1.0-pool>
    python scripts/anvil/data_pipeline/verify_v1_pool.py crosscheck --out <v1.0-pool> --xl <v1.0-xl>
    python scripts/anvil/data_pipeline/verify_v1_pool.py canonical  --out <v1.0-pool>
    python scripts/anvil/data_pipeline/verify_v1_pool.py all        --out <v1.0-pool> --xl <v1.0-xl> --metadata <tgz> \
        [--navsim-root <devkit>]... [--links $TRAINING_DATA_ROOT/pilot_v1] [--deep]

The checks of ``verify_v1_xl.py`` with ``POOL`` (the four sets' counts and SHA-256, dev = the pinned 2,000 in all
four, 428,694 packed frames, the 202,000 frames shared with v1.0-xl, the v1-xl view as the dev reference), plus:
* ``splits`` (``_relations``): v1-nt ⊂ v1-nt-mix ⊂ v1-all, v1-xl-mix ⊂ v1-nt-mix and v1-nt-mix = v1-nt ∪ v1-xl-mix;
  the copy of the base v1-xl-nd (SHA-256 5cb19408…) lies inside v1-all and is not a selection; the committed
  navtrain list (103,288 tokens, SHA-256 5614a8a3…, recorded with its source) holds every v1-nt token.
* ``rules``: the sets recomputed from the index, the lists and OpenScene's metadata (``--metadata`` places the
  warm-up frames that are not index rows, as the plan did): v1-all = the pool (eligible train rows,
  standstill-thinned, outside ±7 s of every warm-up scene) ∪ v1-nt; v1-nt = the navtrain rows that pass every rule,
  each excluded token having exactly one reason, the FIRST rule it fails in ``token_sets.LIST_RULES`` order (800
  tokens fail two, V), with the manifest's counts; v1-xl-mix follows its rule (every pool row outside Las Vegas
  that is not stay-stopped, every Las Vegas pool turn, sideways shift and brake to a stop, and every v1-xl-nd row
  that is not stay-stopped is in it, and nothing else; its stay-stopped rows all come from v1-xl-nd and number
  round(others / 99), at most 1% + 1 row) and the draw pinned in ``token_sets.XL_MIX`` reproduces it.
* ``leakage``: ``verify_v1_xl``'s over the four sets, all trained on, against every devkit given (``--navsim-root``,
  repeatable: NAVSIM v1.1 and main): no token or log in a guarded list except the scene-level warm-up filter
  (``SCENE_LEVEL_FILTERS``), no warm-up token and no train anchor within ±7 s of a warm-up scene. The warm-up logs
  that hold train rows are reported, not failed: the scene-level rule is [proposed], whole logs are the log-level
  choice (open question 7 of §10). Each devkit's navtrain must equal the committed list.

Each command writes ``<out>/verify/<command>.json`` and exits 0 when every check passes, 1 when one fails, and 2 when
the step's output is not there yet (``fetched``, ``canonical``). It only reads; it deletes, moves and rewrites nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import verify_v1_xl as xl  # noqa: E402  (the shared checks; it puts the repo root on sys.path)

Check = xl.Check
_check = xl._check
SETS = ("v1-nt", "v1-xl-mix", "v1-nt-mix", "v1-all")
EXPECTED = {  # V: the committed plan, 2026-11-09 (PRETRAIN_V1_PLAN.md §11)
    "v1-nt": {"train": 80_626, "dev": 2_000,
              "train_sha256": "ecd1b53885ffda4503ffd58edc6f8b5c08f4080eda43d9682c3767457044cc73"},
    "v1-xl-mix": {"train": 238_034, "dev": 2_000,
                  "train_sha256": "05a88a8c176da079dcec93ef7fcaef7a593142e073d7e68ae6ac425fe4e2a5d8"},
    "v1-nt-mix": {"train": 259_929, "dev": 2_000,
                  "train_sha256": "00ed5a88bc61229f7e94e2e3f59d5b41c266a93a1baaf99e5157b8c2cf047bb4"},
    "v1-all": {"train": 426_694, "dev": 2_000,
               "train_sha256": "92081c07454b747fb7a120dfec15c813f8b8b4a2969c28fd10fb5bf7826da683"},
}
FRAMES_TO_KEEP = 428_694  # v1-all train and the pinned dev: every other set is inside it
SHARED_FRAMES = 202_000  # v1-xl's train and dev, also packed in v1.0-xl (the pool holds no v1-large row in the window)
BASE = "v1-xl-nd"  # the mix's base, copied into splits/ (not a selection)
BASE_TRAIN = 176_352
BASE_TRAIN_SHA256 = "5cb194088089ae1b997ef499cb58377e87fe778558ad5cd0e1cd9d9b455a0923"
NAVTRAIN_FILE = "navsim_navtrain_tokens.txt"  # scene_exclusion.NAVTRAIN_FILE
NAVTRAIN_SHA256 = "5614a8a32a7030bda7cc71696e085f59116df442ba6c0b2604bb0649fea7f083"
NAVTRAIN_TOKENS = 103_288


def _relations(out_dir: str, train: xl.Splits, dev: xl.Splits, m: Dict[str, Any]) -> List[Check]:
    """The pool's own split checks: nesting, the base copy, the navtrain list."""
    nt, mix, nt_mix, pool = (train[n] for n in ("v1-nt", "v1-xl-mix", "v1-nt-mix", "v1-all"))
    out = [
        _check(nt <= nt_mix <= pool and mix <= nt_mix, "train: v1-nt ⊂ v1-nt-mix ⊂ v1-all and v1-xl-mix ⊂ v1-nt-mix",
               {"v1-nt not in v1-nt-mix": len(nt - nt_mix), "v1-xl-mix not in v1-nt-mix": len(mix - nt_mix),
                "v1-nt-mix not in v1-all": len(nt_mix - pool)}),
        _check(nt_mix == nt | mix, "train: v1-nt-mix = v1-nt ∪ v1-xl-mix", {"in both": len(nt & mix)}),
    ]
    shas, base_train = {}, set()
    for split in xl.SPLITS:
        path = xl.token_file(out_dir, BASE, split)
        shas[split] = xl.sha256_file(path) if os.path.exists(path) else None
    if shas["train"]:
        base_train = set(xl.read_tokens(xl.token_file(out_dir, BASE, "train")))
    out.append(_check(shas == {"train": BASE_TRAIN_SHA256, "dev": xl.DEV_SHA256} and len(base_train) == BASE_TRAIN,
                      f"base {BASE}: the copy of its token files ({BASE_TRAIN:,} train, SHA-256 {BASE_TRAIN_SHA256[:8]}…; "
                      "the pinned dev)", {"sha256": shas, "train": len(base_train)}))
    out.append(_check(bool(base_train) and base_train <= pool, f"every {BASE} train row is in v1-all",
                      len(base_train - pool)))
    base = m.get("base") if isinstance(m.get("base"), dict) else {}
    out.append(_check(base.get("set") == BASE and BASE not in m.get("selections", {}),
                      f"manifest: {BASE} is the base, not a selection (fetch extracts the selections' frames only)"))
    path = os.path.join(out_dir, "splits", NAVTRAIN_FILE)
    if not os.path.exists(path):
        return out + [_check(False, f"navtrain token list splits/{NAVTRAIN_FILE} exists")]
    with open(path, "rb") as f:
        data = f.read()
    lines = xl.read_tokens(path)
    sha = hashlib.sha256(data).hexdigest()
    entry = m.get("navtrain") if isinstance(m.get("navtrain"), dict) else {}
    out += [
        _check(b"\r" not in data and data.endswith(b"\n") and lines == sorted(set(lines)) and len(lines) == NAVTRAIN_TOKENS
               and sha == NAVTRAIN_SHA256, f"{NAVTRAIN_FILE}: {NAVTRAIN_TOKENS:,} tokens, sorted, unique, LF, "
               f"SHA-256 {NAVTRAIN_SHA256[:8]}…", {"tokens": len(lines), "sha256": sha}),
        _check(entry.get("name") == NAVTRAIN_FILE and entry.get("sha256") == sha
               and (entry.get("source") or {}).get("filter") == "navtrain",
               "manifest navtrain records this list and its source", entry.get("source")),
        _check(nt <= set(lines), "every v1-nt train token is a navtrain token", len(nt - set(lines))),
    ]
    return out


POOL = xl.Build(name="v1-pool", sets=SETS, views=SETS, trained=SETS, expected=EXPECTED, frames_to_keep=FRAMES_TO_KEEP,
                conflicts_file="v1-all_label_conflicts.tsv", ref="v1.0-xl", ref_sets=("v1-xl", "v1-xl-nd", "v1-large"),
                ref_view="v1-xl", copied_set=BASE, shared_frames=SHARED_FRAMES,
                shared_check="the {n:,} frames v1.0-xl also packed (v1-xl's train and dev) are packed here",
                relations=_relations)


def check_rules(out_dir: str, metadata: Optional[str]) -> List[Check]:
    """The sets recomputed from the index and the lists (module docstring)."""
    import numpy as np

    from data_pipeline.v1 import scene_exclusion, selection, tables, token_sets
    from navsim_tools import maneuvers

    m = xl.manifest(out_dir)
    exclusion = m.get("scene_exclusion") if isinstance(m.get("scene_exclusion"), dict) else {}
    window_s = float(exclusion.get("window_s", xl.WINDOW_S))
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"),
                        ["token", "log_name", "t_us", "our_split", "city", "has_cam_f0", "pose_ok", "standstill",
                         "speed", "path_xyyaw"])
    n = len(cands["token"])
    row_of = {str(t): i for i, t in enumerate(cands["token"].tolist())}
    sel = xl.selected(out_dir, POOL)
    train = {name: [t for t, s in sel[name].items() if s == "train"] for name in SETS}
    base_tokens = xl.read_tokens(xl.token_file(out_dir, BASE, "train"))
    absent = sum(t not in row_of for ts in [*train.values(), base_tokens] for t in ts)
    if absent:
        return [_check(False, "every train token of the four sets and of the base is an index row", absent)]

    def rows(tokens: Sequence[str]) -> np.ndarray:
        return np.array(sorted(row_of[t] for t in tokens), dtype=np.int64)

    def mask(r: np.ndarray) -> np.ndarray:
        out_ = np.zeros(n, dtype=bool)
        out_[r] = True
        return out_

    # the warm-up window, its listed frames that are not rows placed from the metadata (logs holding a listed row)
    warm = xl.read_tokens(xl.warmup_path(out_dir))
    missing = sorted(set(warm) - set(row_of))
    located: Dict[str, Any] = {}
    out: List[Check] = []
    if missing:
        if metadata:
            logs = sorted({str(cands["log_name"][row_of[t]]) for t in warm if t in row_of})
            located = scene_exclusion.locate_in_metadata(metadata, missing, logs)
        found = {t: log for t, (log, _) in located.items()}
        recorded = dict(exclusion.get("located_in_metadata") or {})
        out.append(_check(bool(metadata) and found == recorded,
                          f"{len(missing)} listed warm-up tokens that are not index rows: "
                          + ("placed from the metadata as the plan placed them (= located_in_metadata)" if metadata else
                             "NOT placed, so the window and the sets cannot be recomputed (pass --metadata)"),
                          {"placed": len(found), "plan": len(recorded), "differ": sorted(set(found.items()) ^ set(recorded.items()))}))
        if not metadata:
            return out
    window, _ = scene_exclusion.window_mask(cands, warm, window_s, located)
    sets = {name: rows(train[name]) for name in SETS}

    # the pool and v1-all
    eligible = selection.eligible(cands)
    is_train = cands["our_split"] == "train"
    pool = selection.thin_standstill(cands, eligible & is_train, token_sets.STANDSTILL_PERIOD_S) & ~window
    pool_rows = np.flatnonzero(pool)
    out.append(_check(np.array_equal(np.union1d(pool_rows, sets["v1-nt"]), sets["v1-all"]),
                      "v1-all = the pool (eligible train rows, one standstill per 5 s, outside the warm-up window) ∪ v1-nt",
                      {"pool": len(pool_rows), "v1-all": len(sets["v1-all"]),
                       "v1-nt rows outside the pool": int((~pool[sets["v1-nt"]]).sum())}))

    # v1-nt: each listed token's first failing rule, tested here rule by rule
    navtrain = xl.read_tokens(os.path.join(out_dir, "splits", NAVTRAIN_FILE))
    listed = np.array([row_of.get(t, -1) for t in navtrain], dtype=np.int64)
    r = listed[listed >= 0]
    passes = [("index_row", listed >= 0)]
    for name, ok in (("train_log", is_train), ("eligible", eligible), ("outside_window", ~window)):
        full_ok = np.ones(len(listed), dtype=bool)  # a token that is not a row: the later rules do not apply
        full_ok[listed >= 0] = ok[r]
        passes.append((name, full_ok))
    failing = np.stack([~ok for _, ok in passes], axis=1)
    first = np.where(failing.any(axis=1), failing.argmax(axis=1), -1)
    counts = {name: int((first == k).sum()) for k, (name, _) in enumerate(passes)}
    two_or_more = int((failing.sum(axis=1) >= 2).sum())
    kept = np.sort(listed[first == -1])
    recorded = (m.get("navtrain") or {}).get("excluded_by_first_failing_rule")
    out.append(_check(np.array_equal(kept, sets["v1-nt"]),
                      "v1-nt = the navtrain rows that pass every rule: index row, TRAIN log, eligible, outside the window",
                      {"recomputed": len(kept), "v1-nt": len(sets["v1-nt"])}))
    out.append(_check(counts == recorded and sum(counts.values()) + len(kept) == len(navtrain),
                      "every excluded navtrain token has exactly one reason, the first rule it fails (in "
                      "token_sets.LIST_RULES order); the counts = the manifest's",
                      {"first_failing_rule": counts, "manifest": recorded, "tokens_failing_two_or_more_rules": two_or_more,
                       "rules": [name for name, _ in token_sets.LIST_RULES]}))

    # v1-xl-mix: its rule, as properties, then the pinned draw
    rule = token_sets.XL_MIX
    labels = token_sets.maneuver_labels(cands)
    gated = np.zeros(n, dtype=bool)
    for city in rule.gated_cities:
        gated |= cands["city"] == city
    stay = labels == rule.stay
    gated_add = np.zeros(n, dtype=bool)
    for label in rule.gated_add:
        gated_add |= labels == label
    base, in_mix = mask(rows(base_tokens)), mask(sets["v1-xl-mix"])
    add_other, add_gated = pool & ~gated & ~stay, pool & gated & gated_add
    n_mix, n_stay = int(in_mix.sum()), int((in_mix & stay).sum())
    n_other = n_mix - n_stay
    want_stay = min(rule.keep(n_other), int((base & stay).sum()))
    out += [
        _check(not (add_other & ~in_mix).any(), "v1-xl-mix holds every pool row outside Las Vegas that is not stay-stopped",
               int((add_other & ~in_mix).sum())),
        _check(not (add_gated & ~in_mix).any(), "v1-xl-mix holds every Las Vegas pool turn, sideways shift and brake to a stop",
               int((add_gated & ~in_mix).sum())),
        _check(not (base & ~stay & ~in_mix).any(), f"v1-xl-mix holds every {BASE} train row that is not stay-stopped",
               int((base & ~stay & ~in_mix).sum())),
        _check(not (in_mix & ~(base | add_other | add_gated)).any(), f"v1-xl-mix holds nothing else ({BASE} and the two "
               "added groups)", int((in_mix & ~(base | add_other | add_gated)).sum())),
        _check(not (in_mix & stay & ~base).any() and n_stay == want_stay and n_stay <= rule.stay_share * n_mix + 1,
               f"v1-xl-mix's stay-stopped rows all come from {BASE}: round(others / 99), at most 1% + 1 row",
               {"stay_stopped": n_stay, "others": n_other, "want": want_stay,
                "not_from_base": int((in_mix & stay & ~base).sum())}),
    ]
    drawn, _ = token_sets.mix_rows(cands, rows(base_tokens), pool, labels, rule)
    out.append(_check(np.array_equal(drawn, sets["v1-xl-mix"]), "the draw pinned in token_sets.XL_MIX reproduces "
                      "v1-xl-mix (" + token_sets.DRAW + ")",
                      {"numpy": np.__version__, "plan_numpy": (m.get("mix") or {}).get("numpy"),
                       "differ": int(len(np.setxor1d(drawn, sets["v1-xl-mix"])))}))
    return out


def check_navtrain_devkits(out_dir: str, roots: Sequence[str]) -> List[Check]:
    """Each devkit's navtrain = the committed list (both Anvil devkits give it byte for byte, V 2026-11-09)."""
    from data_pipeline.v1 import navsim_splits

    path = os.path.join(out_dir, "splits", NAVTRAIN_FILE)
    committed = set(xl.read_tokens(path)) if os.path.exists(path) else set()
    out = []
    for root in roots:
        guard = navsim_splits.load(root)
        label = os.path.basename(os.path.normpath(root))
        out.append(_check(set(guard.navtrain_tokens) == committed, f"the devkit's navtrain tokens = the committed list [{label}]",
                          {"devkit": len(guard.navtrain_tokens), "committed": len(committed),
                           "devkit_commit": guard.devkit_commit}))
    return out


def check_leakage(out_dir: str, roots: Sequence[str], metadata: Optional[str]) -> List[Check]:
    return xl.check_leakage(out_dir, list(roots), metadata, POOL) + check_navtrain_devkits(out_dir, roots)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Checks on the v1.0-pool build (PRETRAIN_V1_PLAN.md §11).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("splits", "rules", "leakage", "fetched", "crosscheck", "canonical", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the v1.0-pool dataset directory")
        p.add_argument("--quiet", action="store_true", help="print only the summary line")
        if name in ("splits", "crosscheck", "all"):
            p.add_argument("--xl", required=name != "splits", help="the v1.0-xl dataset directory")
        if name in ("rules", "leakage", "all"):
            p.add_argument("--metadata", help="openscene_metadata_trainval.tgz: places the listed warm-up tokens that "
                                              "are not index rows (required when there are any, else the check fails)")
        if name in ("leakage", "all"):
            p.add_argument("--navsim-root", action="append", default=[],
                           help="a NAVSIM devkit: also check against its own lists; repeat for several")
        if name in ("fetched", "all"):
            p.add_argument("--deep", action="store_true", help="testzip and re-hash every pack (and view zip)")
        if name == "all":
            p.add_argument("--xl-view", help="default: <xl>/views/pilot_v1/v1-xl")
            p.add_argument("--links", help="folder of the pilot_v1/<set> links, e.g. $TRAINING_DATA_ROOT/pilot_v1")
            p.add_argument("--samples", type=int, default=50)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    status = 0
    if args.command == "splits":
        checks = xl.check_splits(args.out, args.xl, POOL)
    elif args.command == "rules":
        checks = check_rules(args.out, args.metadata)
    elif args.command == "leakage":
        checks = check_leakage(args.out, args.navsim_root, args.metadata)
    elif args.command == "fetched":
        checks, complete = xl.check_fetched(args.out, 200, True, args.deep, POOL)
        status = 0 if complete else xl.NOT_YET
    elif args.command == "crosscheck":
        checks = xl.check_crosscheck(args.out, args.xl, True, POOL)
    elif args.command == "canonical":
        checks, present = xl.check_canonical(args.out, True, POOL)
        status = 0 if present else xl.NOT_YET
    else:
        xl_view = args.xl_view or xl.view_dir(args.xl, POOL.ref_view)
        checks = xl.check_splits(args.out, args.xl, POOL) + check_rules(args.out, args.metadata)
        checks += check_leakage(args.out, args.navsim_root, args.metadata)
        checks += xl.check_fetched(args.out, 200, True, args.deep, POOL)[0]
        checks += xl.check_crosscheck(args.out, args.xl, True, POOL) + xl.check_canonical(args.out, True, POOL)[0]
        for name in POOL.views:
            checks += xl.check_view(args.out, name, xl_view, True, args.samples, args.seed, args.deep, POOL)
        if args.links:
            checks += xl.check_links(args.out, args.links, POOL)
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
