"""Step 3 of ``docs/training_data/PRETRAIN_V1_PLAN.md``: the CAM_F0 JPEG of every selected sample,
from OpenScene's camera archives.

    python -m data_pipeline.v1.openscene_images plan   --out data/pretrain/v1.0-pilot
    python -m data_pipeline.v1.openscene_images fetch  --out data/pretrain/v1.0-pilot          # the smoke set's 20 archives
    python -m data_pipeline.v1.openscene_images fetch  --out data/pretrain/v1.0-pilot --full   # all 200, after Gate 1
    python -m data_pipeline.v1.openscene_images status --out data/pretrain/v1.0-pilot
    python -m data_pipeline.v1.openscene_images plan   --out <dir> --set v1-large   # 100k train, own folder (Anvil)
    python -m data_pipeline.v1.openscene_images fetch  --out <dir> --full --archives 200 --source-dir <hf download dir>

``plan`` reads only the metadata index (minutes, no images) [agreed 2026-10-31]:
* selects v1-pilot (``selection.PILOT``: 45k train + 2k dev, train balanced) from every log;
* orders the 200 archives so that each next one adds the most variety per GB to the selected
  frames. The smoke set (``selection.SMOKE``) is selected from the logs of the first 20;
* writes ``splits/<set>_tokens_<split>.txt`` and ``splits/manifest.json`` (both committed: IDs and
  counts only) and ``images/archives.json`` (local: OpenScene's archive → log map, with sizes and
  SHA-256 from Hugging Face).

``fetch`` works through the archives in plan order. For each one it:
* downloads it to ``images/tmp/``, resuming after a dropped connection, and checks its size and
  SHA-256 against Hugging Face's;
* checks the logs inside against OpenScene's archive → log map;
* keeps the CAM_F0 JPEG of every selected frame (smoke or v1-pilot), byte for byte, as
  ``images/CAM_F0/<log>/<token>.jpg``, once it has checked the JPEG decodes (rule 2);
* writes ``images/archives/<n>.json`` and deletes the archive. Archives with that file are skipped,
  so an interrupted run resumes when run again. Disk use peaks at the largest archive (19.3 GB)
  plus the kept images (~10 GB).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from data_pipeline.v1 import provenance, selection, tables
from data_pipeline.v1.openscene_index import CAMERA

HF_ROOT = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/"
ARCHIVE_MAP_URL = HF_ROOT + "openscene_sensor_trainval_0-199.json"
ARCHIVE_TREE_URL = ("https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/"
                    "openscene-v1.1/openscene_sensor_trainval_camera")
ARCHIVE_NAME = "openscene_sensor_trainval_camera_{n}.tgz"
SMOKE_ARCHIVES = 20  # the first 20 in plan order reach the smoke set's 10k + 1k (8 reach 3.8k)
SETS = (selection.PILOT, selection.SMOKE)
KB_PER_IMAGE = 211  # mean CAM_F0 JPEG, measured on archive 70 (V); estimates from it are D
CHUNK = 1 << 20
MAX_ATTEMPTS = 30  # per archive; a dropped connection resumes where it stopped
# A connection that stays below SLOW_MB_S for SLOW_WINDOW_S is dropped and resumed on a new one: the
# server sometimes leaves one connection crawling at ~1.6 MB/s while a fresh one gets ~36 (V, 2026-11-02).
SLOW_MB_S = 5.0
SLOW_WINDOW_S = 120.0
MAX_RECONNECTS = 50  # per archive; after that a slow server is simply waited for


class _SlowConnection(Exception):
    pass


# --- plan ----------------------------------------------------------------------------------------


def _load_json(source: str) -> Any:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=120) as response:
            return json.load(response)
    with open(source, encoding="utf-8") as f:
        return json.load(f)


def load_archives(archive_map: str = ARCHIVE_MAP_URL, archive_tree: str = ARCHIVE_TREE_URL) -> Dict[int, Dict[str, Any]]:
    """Archive number → logs (OpenScene's map), bytes and SHA-256 (Hugging Face's LFS record)."""
    logs = {int(k.rsplit("_", 1)[1]): list(v) for k, v in _load_json(archive_map).items()}
    files = {posixpath.basename(f["path"]): f for f in _load_json(archive_tree) if f.get("type") == "file"}
    archives = {}
    for n in sorted(logs):
        name = ARCHIVE_NAME.format(n=n)
        lfs = files[name]["lfs"]
        archives[n] = {"name": name, "url": HF_ROOT + "openscene_sensor_trainval_camera/" + name,
                       "bytes": int(lfs["size"]), "sha256": lfs["oid"], "logs": logs[n]}
    return archives


def archive_order(cands: tables.Columns, rows: np.ndarray, archives: Dict[int, Dict[str, Any]]) -> List[int]:
    """Greedy: next, the archive whose selected ``rows`` add the most variety per GB.

    Variety = sum over (split, city, lateral decision) of sqrt(selected frames), dev counted twice.
    The square root makes the 100th Singapore lane change worth more than the 10,000th Las Vegas
    straight. With every archive streamed the order changes nothing but which come first.
    """
    archive_of = {log: n for n, a in archives.items() for log in a["logs"]}
    cells: Dict[int, Counter] = {n: Counter() for n in archives}
    for r in rows:
        n = archive_of.get(cands["log_name"][r])
        if n is not None:
            cells[n][(cands["our_split"][r], cands["city"][r], cands["decision_lat"][r])] += 1

    def value(c: Counter) -> float:
        return sum(np.sqrt(v) * (2.0 if k[0] == "dev" else 1.0) for k, v in c.items())

    order: List[int] = []
    total: Counter = Counter()
    left = set(archives)
    while left:
        base = value(total)
        best = max(sorted(left), key=lambda n: (value(total + cells[n]) - base) / max(archives[n]["bytes"], 1))
        order.append(best)
        total += cells[best]
        left.remove(best)
    return order


def _write_tokens(path: str, tokens: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{t}\n" for t in sorted(tokens))


def plan(out_dir: str, archive_map: str = ARCHIVE_MAP_URL, archive_tree: str = ARCHIVE_TREE_URL,
         set_name: str = selection.PILOT.name) -> Dict[str, Any]:
    """``set_name`` v1-pilot selects v1-pilot and smoke; v1-large selects only v1-large (its own dataset folder)."""
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"),
                        ["token", "log_name", "t_us", "our_split", "city", "decision_lon", "decision_lat",
                         "standstill", "has_cam_f0", "pose_ok"])
    archives = load_archives(archive_map, archive_tree)
    in_index = set(cands["log_name"].tolist())
    in_map = {log for a in archives.values() for log in a["logs"]}
    main = {c.name: c for c in (selection.PILOT, selection.LARGE)}[set_name]
    main_rows, main_report = selection.select(cands, main, available_logs=in_map)
    order = archive_order(cands, main_rows, archives)
    sets = [(main, main_rows, main_report)]
    smoke_archives: List[int] = []
    if main is selection.PILOT:
        smoke_archives = order[:SMOKE_ARCHIVES]
        smoke_logs = [log for n in smoke_archives for log in archives[n]["logs"]]
        sets.append((selection.SMOKE, *selection.select(cands, selection.SMOKE, available_logs=smoke_logs)))

    split_dir = os.path.join(out_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    reports = {}
    for config, rows, report in sets:
        for split in ("train", "dev"):
            _write_tokens(os.path.join(split_dir, f"{config.name}_tokens_{split}.txt"),
                          cands["token"][rows][cands["our_split"][rows] == split])
        report["images_gb_estimate"] = round(len(rows) * KB_PER_IMAGE / 1e6, 2)  # D
        reports[config.name] = report
    wanted = set().union(*(set(rows.tolist()) for _, rows, _ in sets))
    per_archive = Counter({n: 0 for n in archives})
    archive_of = {log: n for n, a in archives.items() for log in a["logs"]}
    per_archive.update(archive_of[cands["log_name"][r]] for r in wanted)

    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    with open(os.path.join(image_dir, "archives.json"), "w", encoding="utf-8") as f:
        json.dump({str(n): a for n, a in archives.items()}, f, indent=1)
    manifest = {
        "step": "3: selection and archive order (docs/training_data/PRETRAIN_V1_PLAN.md)",
        "created_utc": provenance.utc_now(),
        "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
        "repo_dirty": provenance.git_dirty(provenance.REPO_ROOT),
        "archive_map_url": ARCHIVE_MAP_URL,
        "archive_map_logs": len(in_map),
        "index_logs_missing_from_map": len(in_index - in_map),
        "map_logs_missing_from_index": len(in_map - in_index),
        "archive_order": order,
        "smoke_archives": smoke_archives,
        "download_gb": {"smoke": round(sum(archives[n]["bytes"] for n in smoke_archives) / 1e9, 1),
                        "all": round(sum(a["bytes"] for a in archives.values()) / 1e9, 1)},
        "frames_to_keep": len(wanted),
        "frames_to_keep_per_archive": {str(n): per_archive[n] for n in order},
        "selections": reports,
    }
    with open(os.path.join(split_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


# --- fetch ---------------------------------------------------------------------------------------


def _download(archive: Dict[str, Any], dest: str) -> float:
    """Download to ``dest``, resuming a partial ``dest.part``; check size and SHA-256. Returns seconds."""
    start = time.time()
    part = dest + ".part"
    attempt = reconnects = 0
    while attempt < MAX_ATTEMPTS:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= archive["bytes"]:
            break
        try:
            request = urllib.request.Request(archive["url"], headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the range: start again
                next_report = have + (1 << 30)
                window_t, window_have = time.time(), have
                with open(part, "ab" if have else "wb") as f:
                    for block in iter(lambda: response.read(CHUNK), b""):
                        f.write(block)
                        have += len(block)
                        if have >= next_report:
                            rate = have / max(time.time() - start, 1e-6) / 1e6
                            print(f"[fetch]   {archive['name']}: {have / 1e9:.1f} / {archive['bytes'] / 1e9:.1f} GB "
                                  f"({rate:.1f} MB/s)", flush=True)
                            next_report += 1 << 30
                        now = time.time()
                        if now - window_t >= SLOW_WINDOW_S:
                            rate = (have - window_have) / (now - window_t) / 1e6
                            if rate < SLOW_MB_S and reconnects < MAX_RECONNECTS:
                                raise _SlowConnection(f"{rate:.1f} MB/s")
                            window_t, window_have = now, have
        except _SlowConnection as slow:
            reconnects += 1
            print(f"[fetch]   {archive['name']}: slow connection ({slow}) at {have / 1e9:.2f} GB, resuming on a new one",
                  flush=True)
        except (OSError, ValueError) as error:  # URLError, timeouts and dropped connections are OSErrors
            attempt += 1
            wait = min(300, 10 * attempt)
            print(f"[fetch]   attempt {attempt} stopped at {have / 1e9:.2f} GB: {error!r}; resuming in {wait} s", flush=True)
            time.sleep(wait)
    size = os.path.getsize(part) if os.path.exists(part) else 0
    if size != archive["bytes"]:
        raise RuntimeError(f"{archive['name']}: {size} of {archive['bytes']} bytes after {MAX_ATTEMPTS} attempts")
    if provenance.sha256_file(part) != archive["sha256"]:
        os.remove(part)
        raise RuntimeError(f"{archive['name']}: SHA-256 differs from Hugging Face's; deleted, will download again")
    os.replace(part, dest)
    return time.time() - start


def _decode(data: bytes) -> List[int]:
    """Rule 2: the JPEG decodes. Returns [width, height]."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError(f"format {image.format}")
        image.load()
        return list(image.size)


def _extract(tgz: str, archive: Dict[str, Any], wanted: Dict[str, Dict[str, str]], image_root: str) -> Dict[str, Any]:
    """Keep the wanted CAM_F0 JPEGs of one archive. ``wanted``: cam_f0_path → {token, log}."""
    start = time.time()
    logs_seen, images, undecodable = set(), [], []
    members = 0
    found = set()
    with tarfile.open(tgz, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            members += 1
            parts = member.name.split("/")
            if len(parts) >= 3:
                logs_seen.add(parts[-3])
            rel = posixpath.join(*parts[-3:]) if len(parts) >= 3 else ""
            if parts[-2:-1] != [CAMERA] or rel not in wanted:
                continue
            frame = wanted[rel]
            data = tar.extractfile(member).read()
            found.add(rel)
            try:
                size = _decode(data)
            except Exception as error:  # any decoder error means rule 2 fails
                undecodable.append({"token": frame["token"], "error": repr(error)[:200]})
                continue
            # The path is built from the index's log name and token, never from the member name.
            path = os.path.join(image_root, CAMERA, frame["log"], frame["token"] + ".jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            images.append({"token": frame["token"], "log": frame["log"], "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest(), "width": size[0], "height": size[1]})
    in_archive = {rel: f for rel, f in wanted.items() if f["log"] in set(archive["logs"])}
    return {
        "logs_in_map": sorted(archive["logs"]),
        "logs_in_archive": sorted(logs_seen),
        "logs_match_map": logs_seen == set(archive["logs"]),
        "members": members,
        "wanted": len(in_archive),
        "kept": len(images),
        "missing": sorted(f["token"] for rel, f in in_archive.items() if rel not in found),
        "undecodable": undecodable,
        "image_bytes": sum(i["bytes"] for i in images),
        "extract_s": round(time.time() - start, 1),
        "images": images,
    }


def _wanted(out_dir: str, manifest: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    tokens = set()
    for name in manifest["selections"]:
        for split in ("train", "dev"):
            with open(os.path.join(out_dir, "splits", f"{name}_tokens_{split}.txt"), encoding="utf-8") as f:
                tokens.update(line.strip() for line in f if line.strip())
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"), ["token", "log_name", "cam_f0_path"])
    keep = np.array([t in tokens for t in cands["token"]], dtype=bool)  # np.isin on strings takes minutes
    return {str(p): {"token": str(t), "log": str(log)}
            for t, log, p in zip(cands["token"][keep], cands["log_name"][keep], cands["cam_f0_path"][keep])}


def fetch(out_dir: str, n_archives: int = SMOKE_ARCHIVES, source_dir: Optional[str] = None,
          workers: int = 1, shard: int = 0, num_shards: int = 1) -> Dict[str, Any]:
    """Fetch the first ``n_archives`` archives of the plan.

    ``workers`` > 1 downloads that many archives at once (the server caps each connection at
    ~15 MB/s, V 2026-11-02); extraction and records stay one archive at a time, in plan order.
    ``source_dir`` reads archives already downloaded there instead (tests, or ``hf download`` on
    Anvil); archives not there yet are skipped, so running again picks them up.
    ``shard`` of ``num_shards`` takes every ``num_shards``-th archive of the plan, so several processes
    can extract at once: each archive has its own record and its frames their own files.
    """
    try:  # PIL checks each JPEG: fail before the first download, not after it
        import PIL  # noqa: F401
    except ImportError as error:
        raise SystemExit("fetch needs Pillow to check that each JPEG decodes: python -m pip install pillow") from error
    with open(os.path.join(out_dir, "splits", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(out_dir, "images", "archives.json"), encoding="utf-8") as f:
        archives = {int(k): v for k, v in json.load(f).items()}
    wanted = _wanted(out_dir, manifest)
    image_root = os.path.join(out_dir, "images")
    record_dir = os.path.join(image_root, "archives")
    tmp_dir = os.path.join(image_root, "tmp")
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    done, failed = [], []
    order = manifest["archive_order"][:n_archives]
    pending = [n for k, n in enumerate(order) if k % num_shards == shard
               and not os.path.exists(os.path.join(record_dir, f"{n}.json"))]
    not_yet = []
    if source_dir:
        not_yet = [n for n in pending if not os.path.exists(os.path.join(source_dir, archives[n]["name"]))]
        pending = [n for n in pending if n not in not_yet]
    pool = ThreadPoolExecutor(max_workers=workers) if not source_dir else None
    downloads: Dict[int, Future] = {}
    for i, n in enumerate(pending):
        k = order.index(n) + 1
        record_path = os.path.join(record_dir, f"{n}.json")
        archive = archives[n]
        if pool:  # keep the next `workers` downloads running while this one is extracted
            for m in pending[i : i + workers]:
                if m not in downloads:
                    downloads[m] = pool.submit(_download, archives[m], os.path.join(tmp_dir, archives[m]["name"]))
        print(f"[fetch] archive {n} ({k}/{len(order)}), {archive['bytes'] / 1e9:.1f} GB", flush=True)
        try:
            if source_dir:
                tgz, download_s = os.path.join(source_dir, archive["name"]), 0.0
                if provenance.sha256_file(tgz) != archive["sha256"]:
                    raise RuntimeError(f"{archive['name']}: SHA-256 differs")
            else:
                tgz = os.path.join(tmp_dir, archive["name"])
                download_s = downloads.pop(n).result()
            record = {"archive": n, "name": archive["name"], "bytes": archive["bytes"], "sha256": archive["sha256"],
                      "sha256_ok": True, "download_s": round(download_s, 1)}
            record.update(_extract(tgz, archive, wanted, image_root))
        except Exception as error:  # keep going: one bad archive should not stop a multi-day run
            print(f"[fetch] archive {n} failed: {error!r}", flush=True)
            failed.append({"archive": n, "error": repr(error)[:500]})
            continue
        record["finished_utc"] = provenance.utc_now()
        with open(record_path + ".part", "w", encoding="utf-8") as f:
            json.dump(record, f)
        os.replace(record_path + ".part", record_path)
        if not source_dir:
            os.remove(tgz)
        done.append(n)
        print(f"[fetch] archive {n}: kept {record['kept']} of {record['wanted']}, {len(record['missing'])} missing, "
              f"{len(record['undecodable'])} undecodable, logs match map: {record['logs_match_map']}", flush=True)
    if pool:
        pool.shutdown()
    result = {"done_now": done, "failed": failed, "not_in_source_dir": not_yet, **status(out_dir, quiet=True)}
    print(json.dumps({**{k: v for k, v in result.items() if k not in ("archives_left", "not_in_source_dir")},
                      "not_in_source_dir": len(not_yet)}), flush=True)
    return result


def status(out_dir: str, quiet: bool = False) -> Dict[str, Any]:
    with open(os.path.join(out_dir, "splits", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    record_dir = os.path.join(out_dir, "images", "archives")
    records = []
    for n in manifest["archive_order"]:
        path = os.path.join(record_dir, f"{n}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
    result = {
        "archives_done": len(records),
        "archives_total": len(manifest["archive_order"]),
        "smoke_archives_done": sum(r["archive"] in manifest["smoke_archives"] for r in records),
        "images_kept": sum(r["kept"] for r in records),
        "images_to_keep": manifest["frames_to_keep"],
        "image_gb": round(sum(r["image_bytes"] for r in records) / 1e9, 2),
        "downloaded_gb": round(sum(r["bytes"] for r in records) / 1e9, 1),
        "missing": sum(len(r["missing"]) for r in records),
        "undecodable": sum(len(r["undecodable"]) for r in records),
        "logs_not_matching_map": [r["archive"] for r in records if not r["logs_match_map"]],
        "archives_left": [n for n in manifest["archive_order"] if n not in {r["archive"] for r in records}],
    }
    if not quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "archives_left"}, indent=1))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step 3: CAM_F0 JPEGs of the selected samples.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "select the samples and order the archives (metadata only)"),
                       ("fetch", "download archives in plan order and keep the selected CAM_F0 JPEGs"),
                       ("status", "what has been fetched so far")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--out", required=True, help="dataset directory holding index/, e.g. data/pretrain/v1.0-pilot")
        if name == "plan":
            p.add_argument("--set", default=selection.PILOT.name, choices=(selection.PILOT.name, selection.LARGE.name),
                           help="v1-pilot (with smoke) or v1-large, each in its own dataset directory")
        if name == "fetch":
            p.add_argument("--archives", type=int, default=SMOKE_ARCHIVES, help="the first N archives of the plan")
            p.add_argument("--full", action="store_true", help="allow more than the smoke set's archives (after Gate 1)")
            p.add_argument("--workers", type=int, default=1, help="archives downloading at once")
            p.add_argument("--source-dir", help="read archives already downloaded here; skip those not there yet")
            p.add_argument("--shard", type=int, default=0, help="with --num-shards: this process's share of the archives")
            p.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "plan":
        manifest = plan(args.out, set_name=args.set)
        print(json.dumps({k: manifest[k] for k in ("frames_to_keep", "smoke_archives", "download_gb")}))
        for name, report in manifest["selections"].items():
            print(name, {s: (r["selected"], r["target"], r["logs"]) for s, r in report["splits"].items()},
                  f"~{report['images_gb_estimate']} GB (D)")
    elif args.command == "fetch":
        if args.archives > SMOKE_ARCHIVES and not args.full:
            parser.error(f"more than {SMOKE_ARCHIVES} archives needs --full: the full run waits for Gate 1 (plan §6)")
        result = fetch(args.out, args.archives, source_dir=args.source_dir, workers=args.workers,
                       shard=args.shard, num_shards=args.num_shards)
        sys.exit(1 if result["failed"] else 0)
    else:
        status(args.out)


if __name__ == "__main__":
    main()
