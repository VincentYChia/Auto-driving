"""Gate 0 audit (``docs/training_data/PRETRAIN_V1_PLAN.md``, step 2): statistics of the metadata index.

Reads ``<out>/index/`` and writes:
* ``audit/summary.json`` and ``audit/report.md``: aggregate statistics only, no per-frame rows, so
  both can be committed while the index itself stays local;
* ``splits/{train,dev}_logs.txt`` and ``splits/excluded_logs.txt``: the log-level split (log names
  are public IDs; NAVSIM's own split files list them).

    python -m data_pipeline.v1.audit --out data/pretrain/v1.0-pilot
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from data_pipeline.v1 import openscene_index as oi
from data_pipeline.v1 import selection, tables
from data_pipeline.v1.stats import INF, histogram, percentiles, value_counts
from navsim_tools import geometry

COL_4S = oi.ANCHOR + 8  # path column at +4.0 s
SPEED_EDGES = (0, 0.3, 1, 2, 5, 8, 11, 14, 17, 20, 25, 30, INF)
ACCEL_EDGES = (-INF, -4, -2, -1, -0.5, 0.5, 1, 2, 4, INF)
YAW_RATE_EDGES = (-INF, -0.5, -0.2, -0.1, -0.03, 0.03, 0.1, 0.2, 0.5, INF)
HEADING_EDGES_DEG = (-INF, -90, -45, -25, -8, -2, 2, 8, 25, 45, 90, INF)
LATERAL_EDGES_M = (-INF, -10, -3.5, -1.5, -0.5, 0.5, 1.5, 3.5, 10, INF)
SPEED_DIFF_EDGES = (-INF, -2, -1, -0.5, -0.2, 0.2, 0.5, 1, 2, INF)
STEP_EDGES_M = (0, 5, 10, 15, 20, 25, 50, INF)
TURN_EDGES_RAD = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, INF)
TIME_ERR_EDGES_S = (0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, INF)
SIMULATIONS = (("smoke, 8 archives", selection.SMOKE, 8), ("v1-pilot, 60 archives", selection.PILOT, 60))


def _share(mask) -> Optional[float]:
    mask = np.asarray(mask, dtype=bool)
    return round(float(mask.mean()), 4) if mask.size else None


def _distributions(cands: tables.Columns, mask: np.ndarray) -> Dict[str, Any]:
    path = cands["path_xyyaw"][mask]
    heading_4s = np.degrees(path[:, COL_4S, 2])
    lateral_4s = path[:, COL_4S, 1]
    lon, lat = cands["decision_lon"][mask], cands["decision_lat"][mask]

    def numeric(values, edges):
        return {"hist": histogram(values, edges), "pct": percentiles(values)}

    return {
        "rows": int(mask.sum()),
        "speed_mps": numeric(cands["speed"][mask], SPEED_EDGES),
        "accel_lon_mps2": numeric(cands["accel_lon"][mask], ACCEL_EDGES),
        "accel_lat_mps2": numeric(cands["accel_lat"][mask], ACCEL_EDGES),
        "yaw_rate_radps": numeric(cands["yaw_rate"][mask], YAW_RATE_EDGES),
        "heading_change_4s_deg": numeric(heading_4s, HEADING_EDGES_DEG),
        "lateral_offset_4s_m": numeric(lateral_4s, LATERAL_EDGES_M),
        "speed_minus_path_speed_mps": numeric(cands["speed"][mask] - cands["speed_from_path"][mask], SPEED_DIFF_EDGES),
        "command": value_counts(cands["command"][mask]),
        "decision_lon": value_counts(lon),
        "decision_lat": value_counts(lat),
        "decision_pair": value_counts(f"{a}+{b}" for a, b in zip(lon, lat)),
        "city": value_counts(cands["city"][mask]),
        "share": {
            "cruise (keep_speed+straight)": _share(selection.is_cruise(cands)[mask]),
            "standstill": _share(cands["standstill"][mask]),
            "reversing (vx < -0.3 m/s)": _share(cands["vx"][mask] < -oi.STANDSTILL_SPEED),
            "has_route": _share(cands["has_route"][mask]),
            "command_onehot_ok": _share(cands["command_onehot_ok"][mask]),
            "speed_consistent": _share(cands["speed_consistent"][mask]),
            "crosses_scene_boundary": _share(cands["crosses_scene_boundary"][mask]),
            "in_navtrain": _share(cands["in_navtrain"][mask]),
            "path_valid_at_-4s": _share(cands["path_valid"][mask][:, 0]),
            "path_valid_at_+6s": _share(cands["path_valid"][mask][:, -1]),
        },
    }


def _command_vs_path(cands: tables.Columns, mask: np.ndarray) -> Dict[str, Any]:
    """Does the command agree with where the car went? Checks the one-hot order (left, forward, right, unknown)."""
    path = cands["path_xyyaw"]
    out = {}
    for command in ("left", "straight", "right", "unknown"):
        m = mask & (cands["command"] == command)
        out[command] = {
            "rows": int(m.sum()),
            "decision_lat": value_counts(cands["decision_lat"][m]),
            "heading_change_4s_deg": percentiles(np.degrees(path[m, COL_4S, 2]), (10, 50, 90)),
            "lateral_offset_4s_m": percentiles(path[m, COL_4S, 1], (10, 50, 90)),
        }
    return out


def _yaw_pitch_roll_deg(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz.T
    yaw = np.array([geometry.yaw_from_quaternion(q) for q in wxyz]) if len(wxyz) else np.zeros(0)
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.degrees(np.stack([yaw, pitch, roll], axis=1))


def _calibration(logs: tables.Columns) -> Dict[str, Any]:
    """CAM_F0 and LiDAR calibration across logs. Input to the frame-convention checks of step 4."""
    k = logs["cam_f0_intrinsic"].reshape(-1, 3, 3)
    fx, fy, cx, cy = k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]
    rotation = logs["cam_f0_sensor2lidar_rotation"].reshape(-1, 3, 3)
    axis = rotation[:, :, 2]  # the camera's optical axis (z) in the parent frame, if R maps camera -> parent
    present = np.isfinite(fx)
    distinct = {tuple(np.round(row, 1)) for row in logs["cam_f0_intrinsic"][present]}
    ypr = _yaw_pitch_roll_deg(logs["lidar2ego_rotation"][np.isfinite(logs["lidar2ego_rotation"]).all(axis=1)])
    minmax = (0, 50, 100)
    return {
        "logs_with_cam_f0": int(present.sum()),
        "distinct_intrinsics_0.1px": len(distinct),
        "logs_with_varying_calibration": int(logs["calib_varies"].sum()),
        "fx": percentiles(fx, minmax),
        "fy": percentiles(fy, minmax),
        "cx": percentiles(cx, minmax),
        "cy": percentiles(cy, minmax),
        "hfov_pinhole_deg (width = 2 cx)": percentiles(np.degrees(2 * np.arctan2(cx, fx)), minmax),
        "distortion": {
            name: percentiles(logs["cam_f0_distortion"][:, i], minmax) for i, name in enumerate(("k1", "k2", "p1", "p2", "k3"))
        },
        "cam_translation_m": {a: percentiles(logs["cam_f0_sensor2lidar_translation"][:, i], minmax) for i, a in enumerate("xyz")},
        "cam_optical_axis": {a: percentiles(axis[:, i], minmax) for i, a in enumerate("xyz")},
        "lidar2ego_translation_m": {a: percentiles(logs["lidar2ego_translation"][:, i], minmax) for i, a in enumerate("xyz")},
        "lidar2ego_yaw_pitch_roll_deg": {a: percentiles(ypr[:, i], minmax) for i, a in enumerate(("yaw", "pitch", "roll"))},
    }


def summarize(out_dir: str) -> Dict[str, Any]:
    index_dir = os.path.join(out_dir, "index")
    cands = tables.read(os.path.join(index_dir, "candidates.parquet"))
    logs = tables.read(os.path.join(index_dir, "logs.parquet"))
    with open(os.path.join(index_dir, "manifest.json")) as f:
        manifest = json.load(f)
    dt_edges = tables.read_metadata(os.path.join(index_dir, "logs.parquet"))["dt_hist_edges_s"]

    rule2 = cands["has_cam_f0"]
    rule3 = rule2 & cands["pose_ok"]
    rule5 = rule3 & (cands["our_split"] != "excluded")
    thinned = selection.thin_standstill(cands, rule5, selection.PILOT.standstill_period_s)
    pose_dropped = rule2 & ~cands["pose_ok"]
    duration_h = (logs["t_end_us"] - logs["t_start_us"]) / 3.6e9
    excluded_logs = logs["our_split"] == "excluded"

    summary: Dict[str, Any] = {
        "input": {
            "index_created_utc": manifest["created_utc"],
            "source": manifest["input"],
            "partial_run": manifest["input"]["max_logs"] is not None,
            "repo_commit": manifest["code"]["repo_commit"],
            "devkit_commit": manifest["navsim"]["devkit_commit"],
            "decision_rules_version": manifest["parameters"]["decision_rules_version"],
            "parameters": manifest["parameters"],
        },
        "logs": {
            "total": len(logs["log_name"]),
            "by_nuplan_split": value_counts(logs["nuplan_split"]),
            "by_our_split": value_counts(logs["our_split"]),
            "excluded_by_reason": value_counts(logs["guard_reason"][excluded_logs]),
            "by_city": value_counts(logs["city"]),
            "vehicles": len(set(logs["vehicle_name"].tolist())),
            "hours": round(float(duration_h.sum()), 2),
            "minutes_per_log": percentiles(duration_h * 60, (0, 50, 100)),
            "not_time_sorted": int((~logs["was_sorted"]).sum()),
            "name_differs_from_file": int((~logs["name_matches_file"]).sum()),
            "city_varies": int(logs["city_varies"].sum()),
            "with_duplicate_timestamps": int((logs["n_dup_timestamps"] > 0).sum()),
            "with_duplicate_tokens": int((logs["n_dup_tokens"] > 0).sum()),
            "with_timing_gaps": int((logs["n_gaps"] > 0).sum()),
        },
        "frames": {
            "total": int(logs["n_frames"].sum()),
            "valid_windows (rule 1)": len(cands["token"]),
            "lost: too close to a log's start or end": int(logs["n_no_room"].sum()),
            "lost: timing gap inside the window": int(logs["n_gap_windows"].sum()),
            "without CAM_F0 path": int(logs["n_no_cam_f0"].sum()),
            "without route": int(logs["n_no_route"].sum()),
            "command not one-hot": int(logs["n_bad_command"].sum()),
            "navtrain tokens in NAVSIM": manifest["navsim"]["navtrain_tokens"],
            "navtrain tokens found in this metadata": int(logs["n_navtrain_frames"].sum()),
            "navtrain tokens with a valid window": int(logs["n_navtrain_candidates"].sum()),
        },
        "timing": {
            "frame_gap_s": {"edges": dt_edges, "counts": logs["dt_hist"].sum(axis=0).tolist()},
            "gaps_outside_tolerance": int(logs["n_gaps"].sum()),
            "pose_jump_links": int(logs["n_pose_jumps"].sum()),
            "window_max_time_error_s": histogram(cands["max_time_err_s"], TIME_ERR_EDGES_S),
        },
        "rules": {
            "1 valid window": len(cands["token"]),
            "2 CAM_F0 path present": int(rule2.sum()),
            "3 pose sane": int(rule3.sum()),
            "3 dropped: step > max_step_m": int((pose_dropped & (cands["max_step_m"] > oi.MAX_STEP_M)).sum()),
            "3 dropped: turn > max_turn_rad": int((pose_dropped & (cands["max_turn_rad"] > oi.MAX_TURN_RAD)).sum()),
            "3 flag: speed inconsistent": int((rule3 & ~cands["speed_consistent"]).sum()),
            "5 not guarded": int(rule5.sum()),
            "6 after standstill thinning": int(thinned.sum()),
            "by split": {
                split: {
                    "eligible (rules 1-5)": int((rule5 & (cands["our_split"] == split)).sum()),
                    "after thinning (rule 6)": int((thinned & (cands["our_split"] == split)).sum()),
                }
                for split in ("train", "dev")
            },
            "pose": {
                "max_step_m": histogram(cands["max_step_m"], STEP_EDGES_M),
                "max_turn_rad": histogram(cands["max_turn_rad"], TURN_EDGES_RAD),
            },
        },
        "distributions": {split: _distributions(cands, rule5 & (cands["our_split"] == split)) for split in ("train", "dev")},
        "command_vs_path": _command_vs_path(cands, rule5),
        "calibration": _calibration(logs),
        "selection_simulation": {
            name: selection.simulate(cands, config, logs["log_name"].tolist(), n_archives)
            for name, config, n_archives in SIMULATIONS
        },
    }
    summary["checks"] = _checks(summary, cands)
    return summary


def _checks(summary: Dict[str, Any], cands: tables.Columns) -> List[Dict[str, Any]]:
    logs, frames = summary["logs"], summary["frames"]
    unexpected_guarded = sum(v for k, v in logs["excluded_by_reason"].items())
    checks = [
        ("no nuPlan test or NAVSIM evaluation log in the trainval metadata", unexpected_guarded == 0, f"{unexpected_guarded} logs (excluded)"),
        ("frame ids are unique", len(set(cands["frame_id"].tolist())) == len(cands["frame_id"]), ""),
        ("logs are time-sorted (NAVSIM relies on pickle order)", logs["not_time_sorted"] == 0, f"{logs['not_time_sorted']} logs"),
        ("no duplicate timestamps or tokens", logs["with_duplicate_timestamps"] + logs["with_duplicate_tokens"] == 0, ""),
        ("every command is one-hot", frames["command not one-hot"] == 0, f"{frames['command not one-hot']} frames"),
        ("every frame has a CAM_F0 path", frames["without CAM_F0 path"] == 0, f"{frames['without CAM_F0 path']} frames"),
        (
            "every navtrain token found has a valid window",
            frames["navtrain tokens found in this metadata"] == frames["navtrain tokens with a valid window"],
            f"{frames['navtrain tokens found in this metadata'] - frames['navtrain tokens with a valid window']} without",
        ),
    ]
    return [{"check": name, "ok": bool(ok), "detail": detail} for name, ok, detail in checks]


# --- report --------------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def _pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _bins(edges: Sequence[Any]) -> List[str]:
    return [f"[{a}, {b})" for a, b in zip(edges[:-1], edges[1:])]


def _hist_table(title: str, hists: Dict[str, Dict[str, Any]]) -> List[str]:
    names = list(hists)
    edges = hists[names[0]]["edges"]
    rows = []
    for k, label in enumerate(_bins(edges)):
        row: List[Any] = [label]
        for name in names:
            counts = hists[name]["counts"]
            total = sum(counts)
            row.append(f"{counts[k]:,} ({100 * counts[k] / total:.1f}%)" if total else "0")
        rows.append(row)
    return [f"**{title}**", ""] + _table(["bin"] + names, rows)


def _counts_table(title: str, counts: Dict[str, Dict[str, int]]) -> List[str]:
    names = list(counts)
    keys = sorted({k for c in counts.values() for k in c}, key=lambda k: -sum(c.get(k, 0) for c in counts.values()))
    rows = []
    for key in keys:
        row: List[Any] = [key]
        for name in names:
            total = sum(counts[name].values())
            n = counts[name].get(key, 0)
            row.append(f"{n:,} ({100 * n / total:.1f}%)" if total else "0")
        rows.append(row)
    return [f"**{title}**", ""] + _table(["value"] + names, rows)


def render(summary: Dict[str, Any]) -> str:
    inp, logs, frames, rules = summary["input"], summary["logs"], summary["frames"], summary["rules"]
    dist, sims = summary["distributions"], summary["selection_simulation"]
    params = inp["parameters"]
    out: List[str] = [
        "# Gate 0 audit: OpenScene trainval metadata",
        "",
        f"Generated by `data_pipeline/v1/audit.py` from the index of {inp['index_created_utc']}. "
        f"Repo commit `{inp['repo_commit']}`, NAVSIM devkit `{inp['devkit_commit']}`, "
        f"decision rules v{inp['decision_rules_version']}. Plan: `docs/training_data/PRETRAIN_V1_PLAN.md`.",
        "",
        "Aggregate statistics only. All numbers are measured on the metadata (V) unless marked D.",
        "",
    ]
    if inp["partial_run"]:
        out += [f"> **Partial run:** only the first {inp['source']['max_logs']} logs were indexed. Not for Gate 0.", ""]

    train, dev = dist["train"], dist["dev"]
    turn_left = summary["command_vs_path"]["left"]["decision_lat"]
    turn_right = summary["command_vs_path"]["right"]["decision_lat"]

    def lat_share(counts, prefix):
        total = sum(counts.values())
        return sum(v for k, v in counts.items() if k.endswith(prefix)) / total if total else None

    out += ["## Decisions for Gate 0", ""]
    out += _table(
        ["Question", "Current rule", "What the data says"],
        [
            ["Timing tolerance", f"each gap {params['dt_s']} ± {params['dt_tolerance_s']} s",
             f"{summary['timing']['gaps_outside_tolerance']:,} gaps outside it; {frames['lost: timing gap inside the window']:,} windows lost"],
            ["Pose-jump thresholds", f"step ≤ {params['max_step_m']} m, turn ≤ {params['max_turn_rad']} rad per 0.5 s",
             f"{rules['3 dropped: step > max_step_m']:,} dropped by step, {rules['3 dropped: turn > max_turn_rad']:,} by turn"],
            ["Standstill thinning", f"< {params['standstill_speed_mps']} m/s and < {params['standstill_move_m']} m; 1 per 5 s",
             f"standstill share {_pct(train['share']['standstill'])} (train); thinning keeps {rules['6 after standstill thinning']:,} of {rules['5 not guarded']:,}"],
            ["Cruise cap", "keep_speed + straight ≤ 40% of the sample",
             f"natural share {_pct(train['share']['cruise (keep_speed+straight)'])} (train), {_pct(dev['share']['cruise (keep_speed+straight)'])} (dev)"],
            ["Command encoding", "one-hot (left, forward, right, unknown)",
             f"'left' → *_left lateral decision {_pct(lat_share(turn_left, '_left'))}; 'right' → *_right {_pct(lat_share(turn_right, '_right'))}"],
            ["Dev split", "NAVSIM's val logs", f"{logs['by_our_split'].get('dev', 0):,} logs, {dev['rows']:,} eligible anchors"],
        ]
        + [
            [f"Archives for {name.split(',')[0]}", f"{name.split(', ')[1]} (D: ~{logs['total'] / 200:.1f} logs per archive)",
             f"train {s['splits']['train']['selected']:,} / {s['splits']['train']['target']:,}, dev {s['splits']['dev']['selected']:,} / {s['splits']['dev']['target']:,}"]
            for name, s in sims.items()
        ],
    )

    out += ["## Logs and frames", ""]
    out += _table(["", "Count"], [[f"logs: {k}", v] for k, v in logs.items() if not isinstance(v, dict)]
                  + [[f"logs in our {k} split", v] for k, v in logs["by_our_split"].items()]
                  + [[f"logs in nuPlan {k}", v] for k, v in logs["by_nuplan_split"].items()]
                  + [[f"logs excluded as {k}", v] for k, v in logs["excluded_by_reason"].items()]
                  + [[f"frames: {k}", v] for k, v in frames.items()])
    out += _counts_table("Logs by city", {"logs": logs["by_city"]})

    out += ["## Rules (plan §3)", ""]
    out += _table(["Rule", "Rows"], [[k, v] for k, v in rules.items() if not isinstance(v, dict)]
                  + [[f"{split}: {k}", v] for split, d in rules["by split"].items() for k, v in d.items()])
    out += _hist_table("Largest 0.5 s step in the window (m)", {"candidates": rules["pose"]["max_step_m"]})
    out += _hist_table("Largest 0.5 s turn in the window (rad)", {"candidates": rules["pose"]["max_turn_rad"]})

    out += ["## Timing", ""]
    out += _hist_table("Gap between consecutive frames (s)", {"frame pairs": summary["timing"]["frame_gap_s"]})
    out += _hist_table("Largest |actual − nominal| time offset in a window (s)", {"candidates": summary["timing"]["window_max_time_error_s"]})

    out += ["## Distributions (eligible anchors, rules 1–5)", ""]
    for key, title in (
        ("speed_mps", "Speed (m/s)"),
        ("accel_lon_mps2", "Longitudinal acceleration (m/s²)"),
        ("accel_lat_mps2", "Lateral acceleration (m/s²)"),
        ("yaw_rate_radps", "Yaw rate (rad/s)"),
        ("heading_change_4s_deg", "Heading change at +4 s (deg)"),
        ("lateral_offset_4s_m", "Lateral offset at +4 s (m)"),
        ("speed_minus_path_speed_mps", "Logged speed − speed from the path (m/s)"),
    ):
        out += _hist_table(title, {split: dist[split][key]["hist"] for split in dist})
    for key, title in (("command", "Command"), ("decision_lon", "Longitudinal decision (rules v0.1, uncalibrated)"),
                       ("decision_lat", "Lateral decision (rules v0.1, uncalibrated)"), ("city", "City")):
        out += _counts_table(title, {split: dist[split][key] for split in dist})
    out += _table(["Share"] + list(dist), [[k] + [_pct(dist[s]["share"][k]) for s in dist] for k in train["share"]])

    out += ["## Command vs path (checks the one-hot order)", ""]
    cvp = summary["command_vs_path"]
    out += _counts_table("Lateral decision by command", {c: cvp[c]["decision_lat"] for c in cvp})
    out += _table(["Command", "Rows", "Heading change at +4 s, p10 / p50 / p90 (deg)", "Lateral offset at +4 s, p10 / p50 / p90 (m)"],
                  [[c, v["rows"], " / ".join(_fmt(x) for x in v["heading_change_4s_deg"].values()),
                    " / ".join(_fmt(x) for x in v["lateral_offset_4s_m"].values())] for c, v in cvp.items()])

    out += ["## Calibration (min / median / max over logs)", ""]
    cal = summary["calibration"]
    rows: List[List[Any]] = []
    for key, value in cal.items():
        if isinstance(value, dict) and value and isinstance(next(iter(value.values())), dict):
            rows += [[f"{key}.{sub}", " / ".join(_fmt(x) for x in p.values())] for sub, p in value.items()]
        elif isinstance(value, dict):
            rows.append([key, " / ".join(_fmt(x) for x in value.values())])
        else:
            rows.append([key, value])
    out += _table(["Field", "Value"], rows)

    out += ["## Selection simulation (D: random logs stand in for the archives' unknown contents)", ""]
    sim_rows = []
    for name, s in sims.items():
        for split, r in s["splits"].items():
            sim_rows.append([name, split, r["eligible"], r["after_thinning"], f"{r['selected']:,} / {r['target']:,}",
                             _pct(r["cruise_share"]), r["cruise_cap_exceeded"], _pct(r["standstill_share"]), r["logs"]])
    out += _table(["Run", "Split", "Eligible", "After thinning", "Selected / target", "Cruise", "Cap exceeded", "Standstill", "Logs"], sim_rows)

    out += ["## Checks", ""]
    out += _table(["Check", "OK", "Detail"], [[c["check"], c["ok"], c["detail"]] for c in summary["checks"]])
    return "\n".join(out)


def write_splits(out_dir: str) -> Dict[str, int]:
    logs = tables.read(os.path.join(out_dir, "index", "logs.parquet"), columns=["log_name", "our_split", "guard_reason"])
    split_dir = os.path.join(out_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    counts = {}
    for split in ("train", "dev", "excluded"):
        mask = logs["our_split"] == split
        lines = sorted(
            f"{name}\t{reason}" if split == "excluded" else name
            for name, reason in zip(logs["log_name"][mask], logs["guard_reason"][mask])
        )
        with open(os.path.join(split_dir, f"{split}_logs.txt"), "w") as f:
            f.write("".join(line + "\n" for line in lines))
        counts[split] = len(lines)
    return counts


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Gate 0 audit of the metadata index (plan step 2).")
    parser.add_argument("--out", required=True, help="dataset directory holding index/, e.g. data/pretrain/v1.0-pilot")
    args = parser.parse_args(argv)
    summary = summarize(args.out)
    audit_dir = os.path.join(args.out, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    with open(os.path.join(audit_dir, "report.md"), "w") as f:
        f.write(render(summary) + "\n")
    split_counts = write_splits(args.out)
    failed = [c["check"] for c in summary["checks"] if not c["ok"]]
    print(f"audit written to {audit_dir}; splits {split_counts}; checks failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
