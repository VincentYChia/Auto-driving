"""Checks on the v1-xl build (``docs/training_data/PRETRAIN_V1_PLAN.md`` §10; the v1-xl spec's §8) [proposed].

Run from the repo root; ``build_v1_xl.sh setup`` and ``build_v1_xl.sbatch`` call it between steps:

    python scripts/anvil/data_pipeline/verify_v1_xl.py splits     --out <v1.0-xl> --large <v1.0-large>
    python scripts/anvil/data_pipeline/verify_v1_xl.py leakage    --out <v1.0-xl> [--navsim-root <devkit>]
    python scripts/anvil/data_pipeline/verify_v1_xl.py fetched    --out <v1.0-xl> [--archives 200]
    python scripts/anvil/data_pipeline/verify_v1_xl.py repro      --out <smoke dir> --other <smoke dir>/repro
    python scripts/anvil/data_pipeline/verify_v1_xl.py crosscheck --out <v1.0-xl> --large <v1.0-large>
    python scripts/anvil/data_pipeline/verify_v1_xl.py canonical  --out <v1.0-xl>
    python scripts/anvil/data_pipeline/verify_v1_xl.py all        --out <v1.0-xl> --large <v1.0-large> \
        [--links $TRAINING_DATA_ROOT/pilot_v1] [--deep]

What is checked:
* ``splits``: the token files are sorted, unique, LF-terminated; dev is v1-pilot's (SHA-256 79090629…) in all
  three sets and equals v1.0-large's; v1-large ⊂ v1-xl-nd ⊂ v1-xl (train); the counts and train SHA-256 of the
  spec (§3, V); the copies of v1-large's files are byte-identical; the manifest's checksums and frame count;
  the label-conflict list (report only).
* ``leakage``: every selected token is in the index with the same split, no NAVSIM guard reason, no excluded
  log (the index's ``our_split``); train and dev logs disjoint; dev logs are nuPlan val. With
  ``--navsim-root``, also against NAVSIM's own lists.
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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SETS = ("v1-xl", "v1-xl-nd", "v1-large")
VIEWS = ("v1-xl", "v1-xl-nd")
SPLITS = ("train", "dev")
DEV_SHA256 = "79090629e278705a066890efbfc8271acd9204e2085c3f3ddd8d668316ab7205"  # v1-pilot's = v1-large's dev (V)
EXPECTED = {  # spec §3 (V: design_faithful.py and judge_nested.py, 2026-11-08; v1-large as committed)
    "v1-xl": {"train": 200_000, "dev": 2_000,
              "train_sha256": "d29d242611447c67c7a03e1305f4fa9458f45404efa68e5d54818a1ad7e2cdf4"},
    "v1-xl-nd": {"train": 176_232, "dev": 2_000,
                 "train_sha256": "f74bdbcded2d3c98f56e53663326f77886601e70eda021211274e6a1d6bdf94e"},
    "v1-large": {"train": 100_000, "dev": 2_000,
                 "train_sha256": "e6e6e7d320298ee71d8ae8bea03aa525042884cfa4e0bdc92541acf2ed6601ab"},
}
FRAMES_TO_KEEP = 202_000  # union of the three sets
LARGE_FRAMES = 102_000
CAMERA = "CAM_F0"
VIEW = "pilot_v1"
NOT_YET = 2  # exit code: the step's output is not there yet

Check = Dict[str, Any]


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


def check_splits(out_dir: str, large_dir: Optional[str]) -> List[Check]:
    out: List[Check] = []
    tokens: Dict[Tuple[str, str], List[str]] = {}
    for name in SETS:
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
            want = EXPECTED[name][split]
            out.append(_check(len(lines) == want, f"{name} {split}: {want:,} tokens", len(lines)))
            sha = hashlib.sha256(data).hexdigest()
            want_sha = DEV_SHA256 if split == "dev" else EXPECTED[name]["train_sha256"]
            out.append(_check(sha == want_sha, f"{name} {split}: SHA-256 {want_sha[:8]}… (spec §3)", sha))
    if len(tokens) < 2 * len(SETS):
        return out
    train = {name: set(tokens[(name, "train")]) for name in SETS}
    dev = {name: set(tokens[(name, "dev")]) for name in SETS}
    out.append(_check(train["v1-large"] <= train["v1-xl-nd"] <= train["v1-xl"],
                      "train: v1-large ⊂ v1-xl-nd ⊂ v1-xl",
                      {"v1-large not in v1-xl-nd": len(train["v1-large"] - train["v1-xl-nd"]),
                       "v1-xl-nd not in v1-xl": len(train["v1-xl-nd"] - train["v1-xl"])}))
    out.append(_check(dev["v1-xl"] == dev["v1-xl-nd"] == dev["v1-large"], "dev is the same 2,000 in all three sets"))
    out.append(_check(not (train["v1-xl"] & dev["v1-xl"]), "no token is in both train and dev"))
    union = set().union(*train.values(), *dev.values())
    out.append(_check(len(union) == FRAMES_TO_KEEP, f"the three sets hold {FRAMES_TO_KEEP:,} frames", len(union)))

    if large_dir:
        same = {}
        for split in SPLITS:
            theirs = token_file(large_dir, "v1-large", split)
            same[split] = os.path.exists(theirs) and sha256_file(theirs) == sha256_file(token_file(out_dir, "v1-large", split))
        out.append(_check(all(same.values()), "v1-large token files are byte-identical to v1.0-large's", same))

    m = manifest(out_dir)
    out.append(_check(set(SETS) <= set(m.get("selections", {})), "manifest selections list v1-xl, v1-xl-nd, v1-large",
                      sorted(m.get("selections", {}))))
    out.append(_check(m.get("frames_to_keep") == FRAMES_TO_KEEP, f"manifest frames_to_keep = {FRAMES_TO_KEEP:,}",
                      m.get("frames_to_keep")))
    order = m.get("archive_order", [])
    out.append(_check(len(order) == 200 and len(set(order)) == 200, "archive order lists 200 distinct archives",
                      len(order)))
    found, bad = 0, []
    for name in SETS:
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

    tsv = os.path.join(out_dir, "splits", "v1-xl_label_conflicts.tsv")
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
        out.append(_check(False, "label-conflict list splits/v1-xl_label_conflicts.tsv exists"))
    return out


def selected(out_dir: str) -> Dict[str, Dict[str, str]]:
    """Set → token → split."""
    return {name: {t: split for split in SPLITS for t in read_tokens(token_file(out_dir, name, split))}
            for name in SETS}


def check_leakage(out_dir: str, navsim_root: Optional[str] = None) -> List[Check]:
    from data_pipeline.v1 import tables

    sel = selected(out_dir)
    split_of: Dict[str, str] = {}
    for name in SETS:
        split_of.update(sel[name])
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"),
                        ["token", "log_name", "our_split", "guard_reason"])
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
    if navsim_root:
        from data_pipeline.v1 import navsim_splits

        guard = navsim_splits.load(navsim_root)
        out.append(_check(not (set(split_of) & set(guard.guarded_tokens)) and not (all_logs & set(guard.guarded_logs)),
                          "no token or log is in NAVSIM's guarded lists (devkit)",
                          {"guarded_logs": len(guard.guarded_logs), "guarded_tokens": len(guard.guarded_tokens),
                           "tokens_hit": len(set(split_of) & set(guard.guarded_tokens)),
                           "logs_hit": len(all_logs & set(guard.guarded_logs))}))
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


def check_fetched(out_dir: str, n_archives: int, full: bool, deep: bool = False) -> Tuple[List[Check], bool]:
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
        out.append(_check(kept == FRAMES_TO_KEEP, f"{FRAMES_TO_KEEP:,} frames kept", kept))
        sel = selected(out_dir)
        union = set().union(*(set(s) for s in sel.values()))
        got = {i["token"] for r in recs.values() for i in r["images"]}
        out.append(_check(got == union, "the packed frames are exactly the three sets' frames",
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


def check_crosscheck(out_dir: str, large_dir: str, full: bool) -> List[Check]:
    ours = frame_records(out_dir)
    theirs = frame_records(large_dir)
    large_tokens = {t for split in SPLITS for t in read_tokens(token_file(large_dir, "v1-large", split))}
    shared = [t for t in sorted(large_tokens) if t in ours]
    absent_large = [t for t in large_tokens if t not in theirs]
    sha_bad = [t for t in shared if t in theirs and ours[t]["sha256"] != theirs[t]["sha256"]]
    archive_bad = [t for t in shared if t in theirs and ours[t]["archive"] != theirs[t]["archive"]]
    out = [
        _check(not absent_large, "v1.0-large has a record for every v1-large frame", len(absent_large)),
        _check(not sha_bad, "shared frames: SHA-256 = v1.0-large's record", {"compared": len(shared), "mismatched": len(sha_bad)}),
        _check(not archive_bad, "shared frames: same source archive as in v1.0-large", len(archive_bad)),
    ]
    if full:
        out.append(_check(len(shared) == LARGE_FRAMES, f"all {LARGE_FRAMES:,} v1-large frames are packed", len(shared)))
    else:
        out.append(_check(len(shared) > 0, "some v1-large frames are packed (smoke)", len(shared)))
    return out


def check_canonical(out_dir: str, full: bool) -> Tuple[List[Check], bool]:
    """Checks, and whether canonical.json is there for these three sets."""
    path = os.path.join(out_dir, "canonical.json")
    if not os.path.exists(path):
        return [_check(False, "canonical.json exists")], False
    report = load_json(path)
    if set(report.get("sets", {})) != set(SETS):
        return [_check(False, "canonical.json covers v1-xl, v1-xl-nd, v1-large", sorted(report.get("sets", {})))], False
    failed = [c["check"] for c in report["checks"] if not c["ok"]]
    out = [_check(not failed, "every canonical check passed", failed),
           _check(any(c["check"].startswith("every image file matches") for c in report["checks"]),
                  "the image SHA-256 check ran (not --no-verify-images)")]
    if full:
        counts = {name: report["sets"][name] for name in SETS}
        out.append(_check(all(c["missing_image"] == 0 for c in counts.values()), "no selected frame lacks its image",
                          {n: c["missing_image"] for n, c in counts.items()}))
        out.append(_check(all(counts[n]["by_split"] == {"train": EXPECTED[n]["train"], "dev": EXPECTED[n]["dev"]}
                              for n in SETS), "canonical counts per set and split = spec §3",
                          {n: c["by_split"] for n, c in counts.items()}))
        out.append(_check(report.get("frames") == FRAMES_TO_KEEP, f"{FRAMES_TO_KEEP:,} canonical frames", report.get("frames")))
    return out, True


# --- views ---------------------------------------------------------------------------------------


def check_view(out_dir: str, name: str, large_view: str, full: bool, samples: int, seed: int,
               deep: bool) -> List[Check]:
    folder = view_dir(out_dir, name)
    spec_path = os.path.join(folder, "view.json")
    if not os.path.exists(spec_path):
        return [_check(False, f"view {name}: view.json exists")]
    spec = load_json(spec_path)
    large_spec = load_json(os.path.join(large_view, "view.json"))
    out = [_check(spec.get("storage") == "zip" and spec.get("set") == name, f"view {name}: storage zip, set {name}",
                  spec.get("storage")),
           _check(spec.get("prompt_version") == large_spec.get("prompt_version"),
                  f"view {name}: prompt version = the v1-large view's", spec.get("prompt_version"))]
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
                          f"view {name}: dev.jsonl byte-identical to the v1-large view's", sha256_file(ours_dev)))
    else:
        with open(large_dev, encoding="utf-8") as f:
            theirs = set(f)
        with open(ours_dev, encoding="utf-8") as f:
            ours = list(f)
        out.append(_check(all(line in theirs for line in ours), f"view {name}: every dev.jsonl line is in the v1-large view's",
                          len(ours)))

    dev = sorted(items["dev"], key=lambda i: i["id"])
    pick = random.Random(seed).sample(dev, min(samples, len(dev)))
    ours_r, theirs_r = ViewReader(folder), ViewReader(large_view)
    try:
        differ = [i["id"] for i in pick if ours_r.read(i["image"]) != theirs_r.read(i["image"])]
    finally:
        ours_r.close()
        theirs_r.close()
    out.append(_check(bool(pick) and not differ, f"view {name}: {len(pick)} dev images byte-identical to the v1-large view's",
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


def check_links(out_dir: str, links_root: str) -> List[Check]:
    out = []
    for name in VIEWS:
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
        checks = check_leakage(args.out, args.navsim_root)
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
        checks = check_splits(args.out, args.large) + check_leakage(args.out, args.navsim_root)
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
