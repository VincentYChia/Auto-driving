"""Training exclusion around the scenes of a NAVSIM evaluation filter, by time window [proposed 2026-11-09].

Why: NAVSIM v1.1's ``warmup_test_e2e`` (the 2024 challenge warm-up split) lists 563 scene tokens in 62 logs,
all navmini logs; NAVSIM's own navtrain holds 446 of the tokens (V, 2026-11-08). ``navsim_splits`` read
NAVSIM main, which has no such filter, so the index does not guard them, and v1-pilot, v1-large and the
first v1-xl plan train on some of them. ``CLAUDE.md``: never train on warmup splits. Decision (lead): from
v1-xl on, no TRAIN anchor whose sample window could overlap a listed scene's window; dev is unchanged.

The window: our sample at anchor t spans [t - 1.5 s, t + 4.0 s]; a NAVSIM v1.1 scene at token time tw
spans [tw - 1.5 s, tw + 5.0 s] (4 history and 10 future frames at 0.5 s). They overlap iff
t is in [tw - 5.5 s, tw + 6.5 s]. ``window_mask`` takes the symmetric, more cautious |t - tw| <= ``window_s``
(default 7.0 s), in the same log, **inclusive** at exactly ``window_s`` and with no frame tolerance
(``DT_TOL_US``): a row 7.0 s away is excluded, 7.000001 s away is not (V: the tests).

A listed token that is not in the index is not itself a row, so no selection can contain it. It may still
be a frame of an indexed log that failed rule 1 (a gap, say), with candidate rows around it: 6 of the 115
warm-up tokens missing from the index are such frames, in 3 indexed logs, and two of them have v1-large
and v1-xl train rows within 7 s (V, 2026-11-09). ``locate_in_metadata`` finds them in OpenScene's metadata
pickles (the logs that already hold a listed candidate, ~30 s for warm-up, V) and ``window_mask`` centres
windows on them too (``located``). Tokens found nowhere (the other 109: no frame in the indexed logs
scanned; the 10 warm-up logs without a located token are not in the index at all, V) are reported
(``not_located``).

    python -m data_pipeline.v1.scene_exclusion from-filter <scene filter .yaml> --out <tokens .txt>

The same command wrote navtrain's list (``NAVTRAIN_FILE``), which the v1.0-pool plan trains on instead of
excluding (``token_sets.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data_pipeline.v1 import tables

WINDOW_S = 7.0
OVERLAP_S = (-5.5, 6.5)  # anchor offsets from tw at which the two windows overlap (module docstring)
WARMUP_FILE = "navsim_v1.1_warmup_test_e2e_tokens.txt"
WARMUP_SHA256 = "f5fdc4a169e53f0369d41f946125d7113cd208e4b8fac60a2d65a21353f0cea4"  # the list file (V, 2026-11-09)
WARMUP_SOURCE = {  # V: the file on Anvil and its local copy, 2026-11-08
    "filter": "warmup_test_e2e",
    "repo": "https://github.com/autonomousvision/navsim.git",
    "devkit": "NAVSIM v1.1 (Anvil checkout /home/x-vchia/navsim-v1.1)",
    "commit": "3e8291bfa89ff247231e0227778840cd0a036896",
    "path": "navsim/planning/script/config/common/train_test_split/scene_filter/warmup_test_e2e.yaml",
    "sha256": "4d41f0bec7eb97e678a237d2d974f3173aa812ff01d983c4874a4d65e4ec03c1",
    "tokens": 563,
    "log_names": 62,
}
# NAVSIM's navtrain, the token list of the v1-nt training set (``token_sets.listed_train_rows``) [proposed 2026-11-09].
# A training filter, not an exclusion: it is recorded here because the plan records every list's source the same way.
NAVTRAIN_FILE = "navsim_navtrain_tokens.txt"
NAVTRAIN_SHA256 = "5614a8a32a7030bda7cc71696e085f59116df442ba6c0b2604bb0649fea7f083"  # 103,288 tokens (V, 2026-11-09)
NAVTRAIN_SOURCE = {  # V: copies of both Anvil devkits' files, 2026-11-09; both give this list byte for byte
    "filter": "navtrain",
    "repo": "https://github.com/autonomousvision/navsim.git",
    "path": "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
    "devkits": [
        {"devkit": "NAVSIM main (Anvil checkout /home/x-vchia/navsim)",
         "commit": "0a380a9063d7162ec93d0f51e9990ebac585f720",
         "sha256": "5f3c406a06c961e75d69d33bc127acd5704a3a440a7186bfa2212493ecf9fc01"},
        {"devkit": "NAVSIM v1.1 (Anvil checkout /home/x-vchia/navsim-v1.1)",
         "commit": "3e8291bfa89ff247231e0227778840cd0a036896",
         "sha256": "c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f"},
    ],
    "tokens": 103_288,
    "log_names": 1_192,  # 978 nuPlan train and 214 val logs
}
# Token lists whose source is known, by the list file's SHA-256 (the plan records the source in its manifest).
KNOWN_LISTS: Dict[str, Dict[str, Any]] = {WARMUP_SHA256: WARMUP_SOURCE, NAVTRAIN_SHA256: NAVTRAIN_SOURCE}


def read_tokens(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tokens_from_scene_filter(yaml_path: str) -> List[str]:
    """The sorted, unique ``tokens`` of a NAVSIM scene filter YAML."""
    import yaml

    with open(yaml_path, "rb") as f:
        spec = yaml.load(f.read(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
    return sorted({str(t) for t in spec.get("tokens") or ()})


def write_tokens(path: str, tokens: Iterable[str]) -> None:
    """Sorted, LF, one token per line with a trailing newline (the form of every committed ID list)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{t}\n" for t in sorted(set(tokens)))


def locate_in_metadata(source: str, tokens: Iterable[str], logs: Iterable[str]) -> Dict[str, Tuple[str, int]]:
    """Token → (log, timestamp µs) for the ``tokens`` that are frames of ``logs`` in OpenScene's metadata.

    ``source``: ``openscene_metadata_trainval.tgz`` or a folder of log pickles (``openscene_index``'s
    reader). Only the pickles of ``logs`` (by file name) are unpickled, with the restricted unpickler.
    """
    from data_pipeline.v1 import openscene_index, safe_pickle

    wanted, logs = set(tokens), set(logs)
    found: Dict[str, Tuple[str, int]] = {}
    if not wanted or not logs:
        return found
    for name, data in openscene_index.iter_log_pickles(source):
        if name[: -len(".pkl")] not in logs:
            continue
        for frame in safe_pickle.loads(data):
            token = str(frame["token"])
            if token in wanted:
                found[token] = (str(frame["log_name"]), int(frame["timestamp"]))
    return found


def window_mask(cands: tables.Columns, tokens: Iterable[str], window_s: float = WINDOW_S,
                located: Optional[Dict[str, Tuple[str, int]]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rows of the same log whose ``t_us`` is within ``window_s`` (inclusive) of a listed token's, and a report.

    ``cands`` needs ``token``, ``log_name`` and ``t_us`` (``our_split`` too, for the per-split counts).
    The listed tokens' own rows are in the mask. Rows of other logs are never in it. ``located``
    (``locate_in_metadata``) places listed tokens that are not rows of the index; it is not used for
    a token that is a row.
    """
    if window_s < 0:
        raise ValueError("window_s must be >= 0")
    tokens = sorted(set(tokens))
    located = located or {}
    row_of = {t: i for i, t in enumerate(cands["token"].tolist())}  # np.isin on strings takes minutes
    found = np.array(sorted(row_of[t] for t in tokens if t in row_of), dtype=np.int64)
    not_found = [t for t in tokens if t not in row_of]
    by_metadata = [t for t in not_found if t in located]
    t_us = cands["t_us"].astype(np.int64)
    logs = cands["log_name"]
    mask = np.zeros(len(t_us), dtype=bool)
    window_us = int(round(window_s * 1e6))
    centres: Dict[str, List[int]] = {}
    for r in found:
        centres.setdefault(logs[r], []).append(int(t_us[r]))
    for t in by_metadata:
        centres.setdefault(located[t][0], []).append(int(located[t][1]))
    for log, times in centres.items():
        rows = np.flatnonzero(logs == log)
        if not len(rows):
            continue
        centre = np.sort(np.array(times, dtype=np.int64))
        t = t_us[rows]
        k = np.searchsorted(centre, t)  # nearest centre on either side
        before = np.abs(t - centre[np.clip(k - 1, 0, len(centre) - 1)])
        after = np.abs(centre[np.clip(k, 0, len(centre) - 1)] - t)
        mask[rows] = np.minimum(before, after) <= window_us
    report: Dict[str, Any] = {
        "window_s": window_s,
        "window": "same log, |t_anchor - t_token| <= window_s (inclusive)",
        "overlap_s": list(OVERLAP_S),
        "tokens": len(tokens),
        "tokens_in_index": len(found),
        "tokens_not_in_index": len(not_found),
        "located_in_metadata": {t: located[t][0] for t in by_metadata},  # token → log (no times: IDs only)
        "tokens_not_located": len(not_found) - len(by_metadata),
        "not_located": [t for t in not_found if t not in located],
        "logs_with_tokens_in_index": len({logs[r] for r in found}),
    }
    if "our_split" in cands:
        report["tokens_in_index_by_split"] = {s: int((cands["our_split"][found] == s).sum()) for s in ("train", "dev")}
        report["rows_in_window_by_split"] = {s: int((mask & (cands["our_split"] == s)).sum()) for s in ("train", "dev")}
    return mask, report


def source_of(path: str) -> Optional[Dict[str, Any]]:
    """The recorded source of a known token list (by its SHA-256), else ``None``."""
    return KNOWN_LISTS.get(sha256_file(path))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Token lists of NAVSIM scene filters, for the training exclusion.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("from-filter", help="write a scene filter's tokens as a sorted LF ID list")
    p.add_argument("yaml", help="a NAVSIM scene filter, e.g. .../scene_filter/warmup_test_e2e.yaml")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    tokens = tokens_from_scene_filter(args.yaml)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_tokens(args.out, tokens)
    print(f"{len(tokens)} tokens -> {args.out} (sha256 {sha256_file(args.out)}; filter sha256 {sha256_file(args.yaml)})")


if __name__ == "__main__":
    main()
