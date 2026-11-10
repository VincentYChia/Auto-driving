"""Step 4 of ``docs/training_data/PRETRAIN_V1_PLAN.md``: the canonical record (§4) of every selected
frame whose CAM_F0 image was kept in step 3, and the checks on it.

    python -m data_pipeline.v1.canonical --out data/pretrain/v1.0-pilot [--sets smoke v1-pilot]

Inputs: ``index/`` (step 2), ``splits/<set>_tokens_<split>.txt`` and ``images/`` (step 3: loose JPEGs
or their packs ``images/packs/<n>.zip``, read through ``image_store``; ``image_path`` names the same
bytes either way), and the
metadata archive (``--metadata``, default ``data/pretrain/downloads/openscene_metadata_trainval.tgz``;
only the selected logs are unpickled) for the other cameras' paths and the privileged data.

Outputs, in ``<out>/``:
* ``frames.parquet``: test-time-visible inputs and the target path, one row per frame. A frame in
  both sets is stored once; ``in_<set>`` says which sets hold it.
* ``privileged.parquet``: training-only data (boxes, object velocities, traffic lights, route),
  kept apart so a loader can't leak it into inputs.
* ``labels/decision_rules_v0.1.parquet``: the rule-based meta-actions, flagged uncalibrated.
* ``canonical.json``: counts, checks and table checksums. Aggregate only, so it can be committed.

Frames of a selected set whose image is not (yet) on disk are left out and counted, so the build
can run on the smoke set while the full fetch is still going.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from data_pipeline import labels
from data_pipeline.v1 import image_store, openscene_index, projection, provenance, safe_pickle, stats, tables
from data_pipeline.v1.openscene_index import ANCHOR, CAMERA, CORE, PATH_OFFSETS_S

SETS = ("smoke", "v1-pilot", "v1-large", "v1-xl", "v1-xl-nd",  # v1-xl-nd [proposed]: v1-xl spec §2.2
        "v1-nt", "v1-xl-mix", "v1-nt-mix", "v1-all")  # the v1.0-pool sets [proposed]: token_sets.py, plan §11
METADATA = os.path.join("data", "pretrain", "downloads", "openscene_metadata_trainval.tgz")
OTHER_CAMERAS = ("CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")
LICENCE = "CC BY-NC-SA 4.0 (nuPlan, OpenScene)"
COMMAND_HOW = "navsim_route_20m"  # OpenScene's one-hot, from the route 20 m ahead (V)
LENS_MODEL = "opencv5"  # k1, k2, p1, p2, k3
LABELS_FILE = f"decision_rules_v{labels.RULES_VERSION.rsplit('.', 1)[0]}.parquet"
INDEX_COLUMNS = (
    "frame_id", "log_name", "token", "scene_token", "t_us", "nuplan_split", "our_split", "guard_reason", "city",
    "traffic_side", "in_navtrain", "cam_f0_path", "speed", "accel_lon", "accel_lat", "yaw_rate", "speed_from_path",
    "command", "command_onehot_ok", "has_route", "crosses_scene_boundary", "pose_ok", "speed_consistent",
    "standstill", "max_time_err_s", "decision_lon", "decision_lat", "path_xyyaw", "path_t_s", "path_valid",
)


# --- inputs --------------------------------------------------------------------------------------


def set_tokens(out_dir: str, name: str) -> Dict[str, str]:
    """Token → split of one selected set."""
    out = {}
    for split in ("train", "dev"):
        with open(os.path.join(out_dir, "splits", f"{name}_tokens_{split}.txt"), encoding="utf-8") as f:
            out.update((line.strip(), split) for line in f if line.strip())
    return out


def image_records(out_dir: str) -> Dict[str, Dict[str, Any]]:
    """Token → the step-3 record of its kept JPEG (bytes, sha256, width, height)."""
    out = {}
    for path in sorted(glob(os.path.join(out_dir, "images", "archives", "*.json"))):
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        for image in record["images"]:
            out[image["token"]] = {**image, "archive": record["archive"]}
    return out


def image_path(log: str, token: str) -> str:
    """Relative to the dataset directory, with "/" on every OS."""
    return f"images/{CAMERA}/{log}/{token}.jpg"


def _extras(frame: Dict[str, Any]) -> Dict[str, Any]:
    """What the index does not keep: the other cameras' paths and the privileged data."""
    cams = frame.get("cams") or {}
    anns = frame.get("anns") or {}
    boxes = np.asarray(anns.get("gt_boxes", np.zeros((0, 7))), dtype=np.float32).reshape(-1, 7)
    velocity = np.asarray(anns.get("gt_velocity_3d", np.zeros((0, 3))), dtype=np.float64).reshape(len(boxes), 3)
    to_ego = projection.quaternion_matrix(frame.get("lidar2ego_rotation") or (1.0, 0.0, 0.0, 0.0))
    lights = list(frame.get("traffic_lights") or [])
    return {
        "cam_f0_path": str((cams.get(CAMERA) or {}).get("data_path") or ""),
        "other_cam_paths": [str((cams.get(c) or {}).get("data_path") or "") for c in OTHER_CAMERAS],
        "gt_boxes": boxes,
        "gt_names": [str(n) for n in anns.get("gt_names", [])],
        "track_tokens": [str(t) for t in anns.get("track_tokens", [])],
        "gt_velocity_lidar": velocity.astype(np.float32),
        "gt_velocity_ego": (velocity @ to_ego.T).astype(np.float32),
        "lidar2ego_identity": bool(np.allclose(to_ego, np.eye(3))
                                   and np.allclose(np.asarray(frame.get("lidar2ego_translation", (0, 0, 0)), float), 0)),
        "tl_lane_connector_ids": [int(lane) for lane, _ in lights],
        "tl_is_red": [bool(red) for _, red in lights],
        "roadblock_ids": [str(r) for r in frame.get("roadblock_ids") or []],
    }


def metadata_extras(source: str, wanted: Dict[str, Set[str]]) -> Dict[str, Dict[str, Any]]:
    """Token → ``_extras`` for the wanted tokens of each wanted log. Other logs are not unpickled."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, data in openscene_index.iter_log_pickles(source):
        log = name[: -len(".pkl")]
        if log not in wanted:
            continue
        for frame in safe_pickle.loads(data):
            token = str(frame["token"])
            if token in wanted[log]:
                out[token] = _extras(frame)
    return out


# --- calibration ---------------------------------------------------------------------------------


def hfov_undistorted_deg(intrinsic, distortion, width: float) -> float:
    """Horizontal field of view of the native image, through the lens model.

    Finds the undistorted normalised x that the distortion maps to each image edge, along the row of
    the principal point. NaN if an edge lies beyond the range where the model is monotonic.
    """
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    xn = np.linspace(0.0, 3.0, 30001)
    angles = []
    for sign, edge in ((1.0, width - k[0, 2]), (-1.0, k[0, 2])):
        xd, _ = projection.distort(sign * xn, np.zeros_like(xn), distortion)
        xd = sign * xd
        rising = np.flatnonzero(np.diff(xd) <= 0)
        stop = rising[0] + 1 if len(rising) else len(xn)
        target = edge / k[0, 0]
        if target > xd[stop - 1]:
            return float("nan")
        angles.append(math.atan(float(np.interp(target, xd[:stop], xn[:stop]))))
    return math.degrees(sum(angles))


def _log_calibration(logs: tables.Columns, width: int) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, log in enumerate(logs["log_name"]):
        kmat = np.asarray(logs["cam_f0_intrinsic"][k], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(logs["cam_f0_distortion"][k], dtype=np.float64)
        r_cam = np.asarray(logs["cam_f0_sensor2lidar_rotation"][k], dtype=np.float64).reshape(3, 3)
        t_cam = np.asarray(logs["cam_f0_sensor2lidar_translation"][k], dtype=np.float64)
        r_lid = projection.quaternion_matrix(logs["lidar2ego_rotation"][k])
        t_lid = np.asarray(logs["lidar2ego_translation"][k], dtype=np.float64)
        rotation = r_lid @ r_cam  # camera → ego
        out[str(log)] = {
            "K": np.array([kmat[0, 0], kmat[1, 1], kmat[0, 2], kmat[1, 2]]),
            "dist": dist,
            "hfov_pinhole_deg": math.degrees(2 * math.atan(width / 2 / kmat[0, 0])),
            "hfov_undistorted_deg": hfov_undistorted_deg(kmat, dist, width),
            "cam_to_ego": np.concatenate([r_lid @ t_cam + t_lid, _quaternion(rotation)]),
        }
    return out


def _quaternion(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix, w ≥ 0."""
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# --- build ---------------------------------------------------------------------------------------


def _object_column(values: List[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def build(out_dir: str, sets: Sequence[str] = ("smoke",), metadata: str = METADATA,
          verify_images: bool = True) -> Dict[str, Any]:
    started = time.time()
    members = {name: set_tokens(out_dir, name) for name in sets}
    selected = set().union(*members.values())
    images = image_records(out_dir)
    have = sorted(t for t in selected if t in images)

    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"), INDEX_COLUMNS)
    row_of = {t: k for k, t in enumerate(cands["token"])}
    rows = np.array([row_of[t] for t in have], dtype=np.int64)
    c = {k: v[rows] for k, v in cands.items()}
    n = len(rows)
    logs = tables.read(os.path.join(out_dir, "index", "logs.parquet"))
    widths = Counter(images[t]["width"] for t in have)
    width = widths.most_common(1)[0][0] if widths else 1920
    calib = _log_calibration(logs, width)

    wanted: Dict[str, Set[str]] = {}
    for log, token in zip(c["log_name"], c["token"]):
        wanted.setdefault(str(log), set()).add(str(token))
    print(f"[canonical] {n:,} frames with images; reading the metadata of {len(wanted)} logs", file=sys.stderr, flush=True)
    extras = metadata_extras(metadata, wanted)
    no_metadata = [t for t in have if t not in extras]
    if no_metadata:
        raise RuntimeError(f"{len(no_metadata)} selected frames are not in the metadata, e.g. {no_metadata[:3]}")
    x = [extras[t] for t in have]

    def per_log(key: str) -> np.ndarray:
        return np.stack([calib[log][key] for log in c["log_name"]]) if n else np.zeros((0,))

    frames: tables.Columns = {
        "frame_id": c["frame_id"], "log_name": c["log_name"], "token": c["token"], "scene_token": c["scene_token"],
        "t_us": c["t_us"], "nuplan_split": c["nuplan_split"], "our_split": c["our_split"],
        "city": c["city"], "traffic_side": c["traffic_side"], "in_navtrain": c["in_navtrain"],
        **{f"in_{name.replace('-', '_')}": np.array([t in members[name] for t in have], dtype=bool) for name in sets},
        "image_path": np.array([image_path(log, t) for log, t in zip(c["log_name"], c["token"])], dtype=object),
        "image_sha256": np.array([images[t]["sha256"] for t in have], dtype=object),
        "image_bytes": np.array([images[t]["bytes"] for t in have], dtype=np.int64),
        "image_w": np.array([images[t]["width"] for t in have], dtype=np.int32),
        "image_h": np.array([images[t]["height"] for t in have], dtype=np.int32),
        "source_archive": np.array([images[t]["archive"] for t in have], dtype=np.int32),
        "cam_name": np.full(n, CAMERA, dtype=object),
        "lens_model": np.full(n, LENS_MODEL, dtype=object),
        "K": per_log("K").astype(np.float32),
        "dist": per_log("dist").astype(np.float32),
        "hfov_pinhole_deg": per_log("hfov_pinhole_deg").astype(np.float32),
        "hfov_undistorted_deg": per_log("hfov_undistorted_deg").astype(np.float32),
        "cam_to_ego": per_log("cam_to_ego").astype(np.float32),
        "other_cam_paths": _object_column([e["other_cam_paths"] for e in x]),
        "path_offsets_s": np.tile(PATH_OFFSETS_S.astype(np.float32), (n, 1)),
        "path_xyyaw": c["path_xyyaw"],
        "path_valid": c["path_valid"],
        "path_t_s": c["path_t_s"],
        "speed": c["speed"], "accel_lon": c["accel_lon"], "accel_lat": c["accel_lat"],
        "yaw_rate": c["yaw_rate"], "speed_from_path": c["speed_from_path"],
        "command": c["command"],
        "command_how": np.full(n, COMMAND_HOW, dtype=object),
        "has_route": c["has_route"], "crosses_scene_boundary": c["crosses_scene_boundary"],
        "flag_speed_consistent": c["speed_consistent"], "flag_standstill": c["standstill"],
        "flag_pose_ok": c["pose_ok"], "flag_command_onehot_ok": c["command_onehot_ok"],
        "max_time_err_s": c["max_time_err_s"],
        "text": np.full(n, "", dtype=object),  # none in OpenScene; later sources fill it
        "licence": np.full(n, LICENCE, dtype=object),
    }
    privileged: tables.Columns = {
        "frame_id": c["frame_id"],
        "n_boxes": np.array([len(e["gt_boxes"]) for e in x], dtype=np.int32),
        "gt_boxes": _object_column([e["gt_boxes"].tolist() for e in x]),
        "gt_names": _object_column([e["gt_names"] for e in x]),
        "track_tokens": _object_column([e["track_tokens"] for e in x]),
        "gt_velocity_lidar": _object_column([e["gt_velocity_lidar"].tolist() for e in x]),
        "gt_velocity_ego": _object_column([e["gt_velocity_ego"].tolist() for e in x]),
        "lidar2ego_identity": np.array([e["lidar2ego_identity"] for e in x], dtype=bool),
        "tl_lane_connector_ids": _object_column([e["tl_lane_connector_ids"] for e in x]),
        "tl_is_red": _object_column([e["tl_is_red"] for e in x]),
        "roadblock_ids": _object_column([e["roadblock_ids"] for e in x]),
    }
    decisions: tables.Columns = {
        "frame_id": c["frame_id"],
        "decision_lon": c["decision_lon"],
        "decision_lat": c["decision_lat"],
        "rules_version": np.full(n, labels.RULES_VERSION, dtype=object),
        "calibrated": np.zeros(n, dtype=bool),
    }
    checks = run_checks(out_dir, frames, c, x, logs, verify_images)

    code = {"module": __name__, "repo_commit": provenance.git_commit(provenance.REPO_ROOT),
            "repo_dirty": provenance.git_dirty(provenance.REPO_ROOT)}
    meta = {
        "frames.parquet": {"path_frame": openscene_index.parameters()["path_frame"],
                           "cam_to_ego": "translation (m) + quaternion (w, x, y, z): camera (OpenCV axes) to ego",
                           "K": "fx, fy, cx, cy (px), native image", "dist": "OpenCV k1, k2, p1, p2, k3",
                           "other_cam_paths": list(OTHER_CAMERAS), "image_path": "relative to the dataset directory",
                           "code": code},
        "privileged.parquet": {"gt_boxes": "x, y, z, length, width, height, yaw in the LiDAR frame at the anchor",
                               "gt_velocity_lidar": "as stored by OpenScene: LiDAR frame (L4 in README.md)",
                               "gt_velocity_ego": "rotated by lidar2ego (the identity in every trainval log)",
                               "tl_is_red": "NAVSIM reads it as is_red (navsim_scenario.py, V)",
                               "use": "training-only; never a model input", "code": code},
        LABELS_FILE: {"rules_version": labels.RULES_VERSION, "calibrated": False, "code": code},
    }
    outputs = {"frames.parquet": frames, "privileged.parquet": privileged, f"labels/{LABELS_FILE}": decisions}
    report: Dict[str, Any] = {
        "step": "4: canonical record (docs/training_data/PRETRAIN_V1_PLAN.md §4)",
        "created_utc": provenance.utc_now(),
        "code": code,
        "sets": {name: {"selected": len(tok), "with_image": sum(t in images for t in tok),
                        "missing_image": sum(t not in images for t in tok),
                        "by_split": stats.value_counts(s for t, s in tok.items() if t in images)}
                 for name, tok in members.items()},
        "frames": n,
        "logs": len(wanted),
        "checks": checks,
        "outputs": {},
    }
    for name, columns in outputs.items():
        path = os.path.join(out_dir, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tables.write(path, columns, metadata=meta[os.path.basename(name)])
        report["outputs"][name] = {"rows": n, "bytes": os.path.getsize(path), "sha256": provenance.sha256_file(path)}
    report["runtime_s"] = round(time.time() - started, 1)
    provenance.write_text(os.path.join(out_dir, "canonical.json"), json.dumps(report, indent=1) + "\n")  # last: done
    return report


# --- checks --------------------------------------------------------------------------------------


def _check(ok: bool, what: str, value: Any = None) -> Dict[str, Any]:
    return {"check": what, "ok": bool(ok), **({"value": value} if value is not None else {})}


def run_checks(out_dir: str, frames: tables.Columns, c: tables.Columns, x: List[Dict[str, Any]],
               logs: tables.Columns, verify_images: bool) -> List[Dict[str, Any]]:
    n = len(frames["token"])
    split_logs = {s: set(frames["log_name"][frames["our_split"] == s].tolist()) for s in ("train", "dev")}
    nuplan_of = dict(zip(logs["log_name"], logs["nuplan_split"]))
    origin = c["path_xyyaw"][:, ANCHOR] if n else np.zeros((0, 3))
    core_t = c["path_t_s"][:, CORE] if n else np.zeros((0, 1))
    out = [
        _check(len(set(frames["frame_id"].tolist())) == n, "frame ids are unique"),
        _check(not (c["guard_reason"] != "").any(), "no frame from a NAVSIM evaluation log or token (rule 5)"),
        _check(set(frames["our_split"].tolist()) <= {"train", "dev"}, "every frame is train or dev"),
        _check(not (split_logs["train"] & split_logs["dev"]), "no log is in both train and dev"),
        _check(all(nuplan_of[log] == "val" for log in split_logs["dev"]), "dev logs are nuPlan val logs (rule 4)"),
        _check(bool(c["path_valid"][:, CORE].all()) if n else True, "the -1.5 … +4.0 s path is valid in every frame"),
        _check(bool(np.abs(origin).max() < 1e-4) if n else True, "the path is (0, 0, 0) at the anchor",
               float(np.abs(origin).max()) if n else None),
        _check(bool(np.nanmax(np.abs(core_t - PATH_OFFSETS_S[CORE])) <= 0.05 * 8 + 1e-6) if n else True,
               "path times stay within 8 × 0.05 s of the 0.5 s grid", stats.percentiles(np.abs(core_t - PATH_OFFSETS_S[CORE]))),
        _check(all(e["cam_f0_path"] == p for e, p in zip(x, c["cam_f0_path"])), "metadata CAM_F0 path = index path"),
        _check(all(len(p) == len(OTHER_CAMERAS) and all(p) for p in frames["other_cam_paths"]),
               "every frame lists the other 7 cameras' paths",
               int(sum(not all(p) for p in frames["other_cam_paths"]))),
        _check(len(set(zip(frames["image_w"].tolist(), frames["image_h"].tolist()))) <= 1, "one image size",
               stats.value_counts(f"{w}x{h}" for w, h in zip(frames["image_w"], frames["image_h"]))),
        _check(bool(np.isfinite(frames["K"]).all() and np.isfinite(frames["cam_to_ego"]).all()),
               "calibration is finite"),
        _check(all(e["lidar2ego_identity"] for e in x), "lidar2ego is the identity (so LiDAR frame = ego frame)",
               int(sum(not e["lidar2ego_identity"] for e in x))),
        _check(all(len(e["gt_names"]) == len(e["gt_boxes"]) == len(e["track_tokens"]) for e in x),
               "box, name and track lists have equal lengths"),
    ]
    out.append(_check(True, "rates (information, not pass/fail)", {
        "speed_consistent": round(float(frames["flag_speed_consistent"].mean()), 4) if n else None,
        "standstill": round(float(frames["flag_standstill"].mean()), 4) if n else None,
        "has_route": round(float(frames["has_route"].mean()), 4) if n else None,
        "crosses_scene_boundary": round(float(frames["crosses_scene_boundary"].mean()), 4) if n else None,
        "in_navtrain": round(float(frames["in_navtrain"].mean()), 4) if n else None,
        "boxes_per_frame": stats.percentiles([len(e["gt_boxes"]) for e in x]),
    }))
    if verify_images:
        def ok(p: str, sha: str, archive: int) -> bool:  # the loose file, else the pack member (image_store)
            try:
                data = image_store.read_image(out_dir, p, int(archive))
            except (OSError, KeyError, zipfile.BadZipFile):  # absent, or a damaged member (CRC-32)
                return False
            return hashlib.sha256(data).hexdigest() == sha

        bad = [p for p, sha, a in zip(frames["image_path"], frames["image_sha256"], frames["source_archive"])
               if not ok(p, sha, a)]
        out.append(_check(not bad, "every image file matches its step-3 SHA-256", len(bad)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Step 4: the canonical record of the selected frames.")
    parser.add_argument("--out", required=True, help="dataset directory, e.g. data/pretrain/v1.0-pilot")
    parser.add_argument("--sets", nargs="+", default=["smoke"], choices=SETS, help="selected sets to include")
    parser.add_argument("--metadata", default=METADATA, help="metadata .tgz, URL or directory of *.pkl")
    parser.add_argument("--no-verify-images", action="store_true", help="skip re-hashing every JPEG")
    args = parser.parse_args(argv)
    report = build(args.out, args.sets, args.metadata, verify_images=not args.no_verify_images)
    failed = [ch["check"] for ch in report["checks"] if not ch["ok"]]
    print(json.dumps({"frames": report["frames"], "sets": report["sets"], "failed_checks": failed,
                      "runtime_s": report["runtime_s"]}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
