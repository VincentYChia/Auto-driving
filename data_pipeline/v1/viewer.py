"""A local viewer for the metadata index, with real CAM_F0 images: ``<out>/viewer/index.html``.

Two steps, both local; nothing here is committed (``.gitignore``), because the images and per-frame
rows come from nuPlan and the terms for sharing them are not checked yet (D10):

    python -m data_pipeline.v1.viewer fetch --out data/pretrain/v1.0-pilot [--archive 70]
    python -m data_pipeline.v1.viewer build --out data/pretrain/v1.0-pilot

``fetch`` streams one OpenScene camera archive (1.8–19 GB each; nothing but the kept JPEGs is
stored) and keeps the CAM_F0 image of every indexed frame in it, under
``<out>/viewer/sensor_blobs/<log>/CAM_F0/``. OpenScene publishes which logs each archive holds
(``ARCHIVE_MAP_URL``). These images are for looking only: they are not step 3 of the plan.

``build`` writes the page. It shows every candidate frame that has an image, plus a seeded random
sample of the rest of the index (bird's-eye view only), and the Gate 0 summary as charts. The
future path is drawn on the image with ``projection.py``, whose calibration conventions are
assumed; whether the path lands on the road is the check.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import tarfile
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from data_pipeline.v1 import examples, projection, selection, tables
from data_pipeline.v1.openscene_index import ANCHOR, CAMERA, PATH_OFFSETS_S

HF_ROOT = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/"
ARCHIVE_MAP_URL = HF_ROOT + "openscene_sensor_trainval_0-199.json"
ARCHIVE_URL = HF_ROOT + "openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{n}.tgz"
DEFAULT_ARCHIVE = 70  # 3.2 GB; 7 logs from all four cities, train and dev
IMAGE_SIZE = (1920, 1080)
TEMPLATE = os.path.join(os.path.dirname(__file__), "viewer_template.html")


def blobs_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "viewer", "sensor_blobs")


# --- fetch ---------------------------------------------------------------------------------------


def archive_logs(archive: int) -> List[str]:
    with urllib.request.urlopen(ARCHIVE_MAP_URL, timeout=120) as response:
        return list(json.load(response)[f"openscene_sensor_trainval_{archive}"])


def fetch(out_dir: str, archive: int = DEFAULT_ARCHIVE, source: Optional[str] = None) -> Dict[str, Any]:
    """Keep the CAM_F0 JPEG of every indexed frame in one camera archive.

    ``source`` (a local ``.tgz``) replaces the download, for tests.
    """
    wanted = set(tables.read(os.path.join(out_dir, "index", "candidates.parquet"), ["cam_f0_path"])["cam_f0_path"])
    wanted -= {None, ""}  # "" where a frame has no CAM_F0
    root = blobs_dir(out_dir)
    kept = seen = 0
    start = time.time()
    url = source or ARCHIVE_URL.format(n=archive)
    print(f"[fetch] streaming {url}", flush=True)
    opener = open(url, "rb") if source else urllib.request.urlopen(url, timeout=120)
    with opener as stream, tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            seen += 1
            parts = member.name.split("/")
            if len(parts) < 3 or parts[-2] != CAMERA:
                continue
            rel = posixpath.join(*parts[-3:])  # <log>/CAM_F0/<file>.jpg, as in cam_f0_path
            if rel not in wanted:
                continue
            # The name is matched against the index, so it cannot escape ``root``.
            path = os.path.join(root, *parts[-3:])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(tar.extractfile(member).read())
            os.replace(path + ".part", path)
            kept += 1
            if kept % 500 == 0:
                print(f"[fetch] {kept} images kept, {time.time() - start:.0f} s", flush=True)
    result = {"archive": archive, "files_seen": seen, "images_kept": kept, "runtime_s": round(time.time() - start, 1)}
    print(json.dumps(result), flush=True)
    return result


# --- build ---------------------------------------------------------------------------------------


def _round(values: np.ndarray, digits: int) -> List[Optional[float]]:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=np.float64)]


def _log_calibration(logs: tables.Columns) -> Dict[str, Dict[str, np.ndarray]]:
    keys = ("cam_f0_intrinsic", "cam_f0_distortion", "cam_f0_sensor2lidar_rotation",
            "cam_f0_sensor2lidar_translation", "lidar2ego_rotation", "lidar2ego_translation")
    return {str(name): {k: logs[k][i] for k in keys} for i, name in enumerate(logs["log_name"])}


def _row(cands: tables.Columns, r: int, flags: Dict[str, np.ndarray], log_start: Dict[str, int],
         image: Optional[str], calib: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    ex = examples.describe(cands, r)
    log = str(cands["log_name"][r])
    path = cands["path_xyyaw"][r].astype(np.float64)
    valid = cands["path_valid"][r]
    row = {
        "id": ex["frame_id"],
        "log": log,
        "t": round((int(cands["t_us"][r]) - log_start[log]) / 1e6, 1),
        "split": ex["split"],
        "city": ex["city"],
        "cmd": ex["command"],
        "lon": ex["lon"],
        "lat": ex["lat"],
        "v": round(ex["speed"], 2),
        "al": round(ex["accel_lon"], 2),
        "at": round(ex["accel_lat"], 2),
        "yr": round(ex["yaw_rate"], 3),
        "route": ex["has_route"],
        "eligible": bool(flags["eligible"][r]),
        "kept": bool(flags["kept"][r]),
        "flags": ex["flags"],
        "why": [ex["why_command"], ex["why_lon"], ex["why_lat"]],
        "path": [[*_round(p, 2)[:2], round(float(p[2]), 4)] if ok else None for p, ok in zip(path, valid)],
        "img": image,
    }
    if image and calib is not None:
        px = projection.path_pixels(path, valid, calib, IMAGE_SIZE)
        row["px"] = [None if np.isnan(u) else [round(float(u), 1), round(float(v), 1)] for u, v in px]
    return row


def build(out_dir: str, sample: int = 3000, seed: int = 0) -> str:
    """Write ``<out>/viewer/index.html`` and ``data.js``. Returns the page's path."""
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"))
    logs = tables.read(os.path.join(out_dir, "index", "logs.parquet"))
    with open(os.path.join(out_dir, "audit", "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    eligible = selection.eligible(cands)
    flags = {"eligible": eligible, "kept": selection.thin_standstill(cands, eligible, selection.PILOT.standstill_period_s)}
    log_start = {str(n): int(t) for n, t in zip(logs["log_name"], logs["t_start_us"])}
    calib = _log_calibration(logs)

    root = blobs_dir(out_dir)
    image_logs = sorted(d for d in (os.listdir(root) if os.path.isdir(root) else []) if d in calib)
    in_image_logs = np.isin(cands["log_name"], image_logs)
    with_image = [r for r in np.flatnonzero(in_image_logs)
                  if os.path.exists(os.path.join(root, *str(cands["cam_f0_path"][r]).split("/")))]
    rest = np.flatnonzero(eligible & ~in_image_logs)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(rest, size=min(sample, len(rest)), replace=False) if len(rest) else np.array([], int)

    def ordered(rows):
        rows = np.asarray(rows, dtype=np.int64)
        return rows[np.lexsort((cands["t_us"][rows], cands["log_name"][rows]))] if len(rows) else rows

    rows = [_row(cands, int(r), flags, log_start, str(cands["cam_f0_path"][r]), calib[str(cands["log_name"][r])])
            for r in ordered(with_image)]
    rows += [_row(cands, int(r), flags, log_start, None, None) for r in ordered(sampled)]

    data = {
        "meta": {
            "out_dir": out_dir,
            "offsets_s": PATH_OFFSETS_S.tolist(),
            "anchor": ANCHOR,
            "image_size": IMAGE_SIZE,
            "image_logs": image_logs,
            "rows_with_image": len(with_image),
            "rows_sampled": int(len(sampled)),
            "index_rows": int(len(cands["frame_id"])),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": summary,
        "rows": rows,
    }
    viewer = os.path.join(out_dir, "viewer")
    os.makedirs(viewer, exist_ok=True)
    # A script, not JSON, so the page also works opened straight from disk (file://).
    with open(os.path.join(viewer, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.VIEWER_DATA = ")
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
        f.write(";\n")
    page = os.path.join(viewer, "index.html")
    with open(TEMPLATE, encoding="utf-8") as src, open(page, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    return page


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Local viewer of the metadata index, with CAM_F0 images.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch", help="stream one camera archive and keep the indexed CAM_F0 images")
    p_fetch.add_argument("--out", required=True, help="dataset directory holding index/")
    p_fetch.add_argument("--archive", type=int, default=DEFAULT_ARCHIVE, help="camera archive number, 0-199")
    p_build = sub.add_parser("build", help="write <out>/viewer/index.html")
    p_build.add_argument("--out", required=True, help="dataset directory holding index/ and audit/")
    p_build.add_argument("--sample", type=int, default=3000, help="rows without images to include")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.out, args.archive)
    else:
        print(build(args.out, sample=args.sample))


if __name__ == "__main__":
    main()
