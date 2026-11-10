"""The Gate 1 QA report of ``docs/training_data/PRETRAIN_V1_PLAN.md`` (step 4): ``<out>/qa/report.html``.

    python -m data_pipeline.v1.qa_report --out data/pretrain/v1.0-pilot [--set smoke] [--n 50]

It reads only the canonical record (``canonical.py``), so it checks what training will see:
* **renders:** for ``--n`` seeded random frames of the set, the future path projected onto the
  native CAM_F0 image with the frame's own ``K``, ``dist`` and ``cam_to_ego``, beside a bird's-eye
  plot of the stored path (−4 … +6 s);
* **a numeric convention check:** a car on the ground ahead must project below the horizon. The
  share of valid future points (1–4 s, moving frames) that land below the principal point's row;
* **distributions:** timing error, speed, command, decision classes, city, flag rates.

The page embeds real images, so it stays local (``.gitignore``). ``qa/summary.json`` holds the
aggregate numbers only and can be committed.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

from data_pipeline.v1 import image_store, projection, provenance, stats, tables
from data_pipeline.v1.openscene_index import PATH_OFFSETS_S

RENDER_WIDTH = 960
FUTURE = (PATH_OFFSETS_S > 0) & (PATH_OFFSETS_S <= 4.0)  # the 8 target poses
LATER = PATH_OFFSETS_S > 4.0  # stored, not a target
HORIZON_CHECK = (PATH_OFFSETS_S >= 1.0) & (PATH_OFFSETS_S <= 4.0)
MOVING_MPS = 2.0
TIME_ERR_EDGES = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, stats.INF)
SPEED_EDGES = (0.0, 0.3, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, stats.INF)
COLUMNS = ("frame_id", "log_name", "token", "our_split", "city", "image_path", "source_archive", "image_w", "image_h",
           "K", "dist", "cam_to_ego", "path_xyyaw", "path_valid", "path_t_s", "speed", "accel_lon", "command",
           "has_route", "crosses_scene_boundary", "flag_speed_consistent", "flag_standstill", "max_time_err_s",
           "in_navtrain")
SETS = ("smoke", "v1-pilot", "v1-large", "v1-xl", "v1-xl-nd", "v1-nt", "v1-xl-mix", "v1-nt-mix", "v1-all")  # canonical.SETS


def pixels(frames: tables.Columns, k: int) -> np.ndarray:
    """Pixels of row ``k``'s stored path on its native image, through its own calibration."""
    path = np.asarray(frames["path_xyyaw"][k], dtype=np.float64)
    ground = np.column_stack([path[:, 0], path[:, 1], np.zeros(len(path))])
    pose = np.asarray(frames["cam_to_ego"][k], dtype=np.float64)
    cam = projection.ego_to_camera(ground, projection.quaternion_matrix(pose[3:]), pose[:3])
    fx, fy, cx, cy = (float(v) for v in frames["K"][k])
    kmat = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    px = projection.project(cam, kmat, frames["dist"][k], (int(frames["image_w"][k]), int(frames["image_h"][k])))
    px[~np.asarray(frames["path_valid"][k], dtype=bool)] = np.nan
    return px


def horizon_check(frames: tables.Columns, rows: Sequence[int]) -> Dict[str, Any]:
    """Share of projected 1–4 s points below the principal point's row, over moving frames."""
    below = total = outside = 0
    for k in rows:
        if frames["speed"][k] < MOVING_MPS:
            continue
        px = pixels(frames, k)[HORIZON_CHECK]
        seen = px[np.isfinite(px[:, 1])]
        outside += int(len(px) - len(seen))
        total += len(seen)
        below += int((seen[:, 1] > float(frames["K"][k][3])).sum())
    return {"frames_checked": int(sum(frames["speed"][k] >= MOVING_MPS for k in rows)), "points": total,
            "below_horizon": round(below / total, 4) if total else None, "points_off_image": outside}


def render(out_dir: str, frames: tables.Columns, k: int) -> str:
    """The native JPEG, downscaled, with the projected path drawn on it, as a data URL."""
    from PIL import Image, ImageDraw

    data = image_store.read_image(out_dir, str(frames["image_path"][k]), int(frames["source_archive"][k]))
    with Image.open(io.BytesIO(data)) as image:  # the loose original, else its pack member
        image = image.convert("RGB")
        scale = RENDER_WIDTH / image.width
        image = image.resize((RENDER_WIDTH, round(image.height * scale)))
    draw = ImageDraw.Draw(image)
    px = pixels(frames, k) * scale
    for mask, colour in ((LATER, (255, 210, 60)), (FUTURE | (PATH_OFFSETS_S == 0), (0, 230, 255))):
        pts = [tuple(p) for p in px[mask] if np.isfinite(p).all()]
        if len(pts) > 1:
            draw.line(pts, fill=colour, width=3)
        for x, y in pts:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 0, 0), fill=colour)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def bev_svg(path: np.ndarray, valid: np.ndarray, size: int = 220) -> str:
    """Bird's-eye plot: x forward is up, y left is left. 4 m grid lines every 10 m."""
    p = np.asarray(path, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    span = max(20.0, float(np.abs(p[:, :2]).max()) * 1.1 if len(p) else 20.0)

    def xy(x: float, y: float) -> str:
        return f"{size / 2 - y / span * size / 2:.1f},{size / 2 - x / span * size / 2:.1f}"

    t = PATH_OFFSETS_S[np.asarray(valid, dtype=bool)]
    parts = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
             f'aria-label="bird\'s-eye path"><rect width="100%" height="100%" class="bev"/>']
    for g in np.arange(-np.floor(span / 10) * 10, span + 1e-6, 10):
        a, b = xy(g, -span), xy(g, span)
        c, d = xy(-span, g), xy(span, g)
        parts.append(f'<polyline points="{a} {b}" class="grid"/><polyline points="{c} {d}" class="grid"/>')
    for mask, cls in ((t <= 0, "past"), ((t >= 0) & (t <= 4.0), "future"), (t >= 4.0, "later")):
        pts = " ".join(xy(x, y) for x, y in p[mask, :2])
        if pts:
            parts.append(f'<polyline points="{pts}" class="{cls}"/>')
    parts.append(f'<circle cx="{size / 2}" cy="{size / 2}" r="3" class="ego"/>'
                 f'<text x="4" y="{size - 6}" class="scale">grid 10 m</text></svg>')
    return "".join(parts)


def bars(counts: Dict[str, int], title: str) -> str:
    total = max(sum(counts.values()), 1)
    peak = max(counts.values()) if counts else 1
    rows = "".join(f'<tr><td>{html.escape(k)}</td><td class="bar"><span style="width:{100 * v / peak:.1f}%"></span></td>'
                   f'<td class="num">{v:,}</td><td class="num">{100 * v / total:.1f}%</td></tr>' for k, v in counts.items())
    return f'<section class="chart"><h3>{html.escape(title)}</h3><table class="bars">{rows}</table></section>'


def _hist(values, edges, unit: str) -> Dict[str, int]:
    counts = stats.bin_counts(values, edges)
    labels = [f"{a:g}–{b:g} {unit}" if np.isfinite(b) else f"≥ {a:g} {unit}" for a, b in zip(edges[:-1], edges[1:])]
    return dict(zip(labels, counts))


def build(out_dir: str, set_name: str = "smoke", n: int = 50, seed: int = 0) -> Dict[str, Any]:
    frames = tables.read(os.path.join(out_dir, "frames.parquet"), list(COLUMNS) + [f"in_{set_name.replace('-', '_')}"])
    rows = np.flatnonzero(frames[f"in_{set_name.replace('-', '_')}"])
    with open(os.path.join(out_dir, "canonical.json"), encoding="utf-8") as f:
        canonical = json.load(f)
    pick = np.sort(np.random.default_rng(seed).choice(rows, size=min(n, len(rows)), replace=False))
    core = (PATH_OFFSETS_S >= -1.5) & (PATH_OFFSETS_S <= 4.0)
    time_err = np.abs(frames["path_t_s"][rows][:, core] - PATH_OFFSETS_S[core])
    summary = {
        "step": "4: Gate 1 QA (docs/training_data/PRETRAIN_V1_PLAN.md §6)",
        "created_utc": provenance.utc_now(),
        "set": set_name,
        "frames": int(len(rows)),
        "renders": {"n": int(len(pick)), "seed": seed},
        "horizon_check_all_frames": horizon_check(frames, rows),
        "horizon_check_renders": horizon_check(frames, pick),
        "time_error_s": {"per_pose": stats.percentiles(time_err), "histogram": _hist(time_err, TIME_ERR_EDGES, "s")},
        "speed_mps": _hist(frames["speed"][rows], SPEED_EDGES, "m/s"),
        "split": stats.value_counts(frames["our_split"][rows]),
        "city": stats.value_counts(frames["city"][rows]),
        "command": stats.value_counts(frames["command"][rows]),
        "flags": {name: round(float(frames[name][rows].mean()), 4) for name in
                  ("has_route", "crosses_scene_boundary", "flag_speed_consistent", "flag_standstill", "in_navtrain")},
        "canonical_checks_failed": [c["check"] for c in canonical["checks"] if not c["ok"]],
    }
    labels_path = os.path.join(out_dir, "labels", "decision_rules_v0.1.parquet")
    decisions = tables.read(labels_path)
    summary["decision_lon"] = stats.value_counts(decisions["decision_lon"][rows])
    summary["decision_lat"] = stats.value_counts(decisions["decision_lat"][rows])

    cards = []
    for k in pick:
        facts = (f"{html.escape(str(frames['city'][k]))} · {frames['our_split'][k]} · speed {frames['speed'][k]:.1f} m/s · "
                 f"accel {frames['accel_lon'][k]:+.1f} m/s² · command <b>{frames['command'][k]}</b> · "
                 f"decision {decisions['decision_lon'][k]}, {decisions['decision_lat'][k]}")
        end = np.asarray(frames["path_xyyaw"][k])[np.flatnonzero(PATH_OFFSETS_S == 4.0)[0]]
        cards.append(
            f'<figure class="card"><img src="{render(out_dir, frames, k)}" alt="CAM_F0 with projected path" loading="lazy">'
            f'{bev_svg(frames["path_xyyaw"][k], frames["path_valid"][k])}'
            f'<figcaption>{facts}<br>at +4 s: x {end[0]:.1f} m, y {end[1]:+.1f} m, yaw {np.degrees(end[2]):+.0f}° · '
            f'max time error {frames["max_time_err_s"][k] * 1000:.0f} ms<br><code>{html.escape(str(frames["frame_id"][k]))}</code>'
            f'</figcaption></figure>')
    checks = "".join(f'<tr><td>{"✔" if c["ok"] else "✘ FAILED"}</td><td>{html.escape(c["check"])}</td>'
                     f'<td><code>{html.escape(json.dumps(c.get("value", "")))[:200]}</code></td></tr>'
                     for c in canonical["checks"])
    hc = summary["horizon_check_all_frames"]
    charts = "".join([
        bars(summary["time_error_s"]["histogram"], "Pose time vs the 0.5 s grid, −1.5 … +4 s (per pose)"),
        bars(summary["speed_mps"], "Speed at the anchor"),
        bars(summary["command"], "Route command"),
        bars(summary["decision_lon"], "Decision, longitudinal (rules v0.1, uncalibrated)"),
        bars(summary["decision_lat"], "Decision, lateral (rules v0.1, uncalibrated)"),
        bars(summary["city"], "City"),
        bars(summary["split"], "Split"),
        bars({k: round(v * 10000) for k, v in summary["flags"].items()}, "Flag rates (per 10,000 frames)"),
    ])
    page = TEMPLATE.format(
        title=f"Gate 1 QA: {html.escape(set_name)}", n=len(pick), frames=f"{len(rows):,}", seed=seed,
        below=f"{100 * hc['below_horizon']:.2f}%" if hc["below_horizon"] is not None else "n/a",
        points=f"{hc['points']:,}", checks=checks, charts=charts, cards="".join(cards))
    qa_dir = os.path.join(out_dir, "qa")
    os.makedirs(qa_dir, exist_ok=True)
    provenance.write_text(os.path.join(qa_dir, "report.html"), page)
    provenance.write_text(os.path.join(qa_dir, "summary.json"), json.dumps(summary, indent=1) + "\n")  # last: done
    return summary


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>
:root {{ --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#d9d9d4; --card:#fff; --bar:#2a6f97;
  --future:#0aa5c2; --later:#c9a227; --past:#8a8a85; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ececea; --muted:#a3a39e; --line:#3a3a37;
  --card:#20201e; --bar:#5fa8d3; --future:#3cc8e0; --later:#e0bb45; --past:#8f8f8a; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:28px 0 8px; }} h3 {{ font-size:14px; margin:0 0 6px; }}
p, figcaption {{ color:var(--muted); }} code {{ font-size:12px; }}
table {{ border-collapse:collapse; }} td {{ padding:3px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:16px; }}
.chart {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.bars {{ width:100%; }} .bars td {{ border:0; padding:2px 4px; }} .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.bar {{ width:45%; }} .bar span {{ display:block; height:10px; background:var(--bar); border-radius:0 4px 4px 0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 700px), 1fr)); gap:16px; }}
.card {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px;
  display:grid; grid-template-columns:1fr 220px; gap:8px; }}
.card img {{ width:100%; height:auto; border-radius:4px; }} .card figcaption {{ grid-column:1 / -1; }}
@media (max-width:640px) {{ .card {{ grid-template-columns:1fr; }} }}
svg .bev {{ fill:var(--bg); }} svg .grid {{ stroke:var(--line); stroke-width:1; fill:none; }}
svg .past {{ stroke:var(--past); stroke-width:2; fill:none; }} svg .future {{ stroke:var(--future); stroke-width:3; fill:none; }}
svg .later {{ stroke:var(--later); stroke-width:2; fill:none; stroke-dasharray:4 3; }} svg .ego {{ fill:var(--fg); }}
svg .scale {{ fill:var(--muted); font-size:10px; }}
</style></head><body>
<h1>{title}</h1>
<p>{frames} frames in the set. {n} renders, seeded random (seed {seed}). Cyan: the 8 target poses (0.5 … 4 s), projected
onto the native image with each frame's own K, distortion and camera-to-ego; yellow: the stored +4.5 … +6 s.
The bird's-eye plot shows −4 … +6 s (grey: past), x forward up.</p>
<h2>What to look for</h2>
<p>The cyan path should lie on the road and follow the lane the car actually takes. A path floating above the road,
mirrored left/right, or bent the wrong way means a convention is wrong. Numeric check, every moving frame of the set:
<b>{below}</b> of {points} projected 1–4 s points lie below the horizon (the principal point's row).</p>
<h2>Canonical checks</h2><table>{checks}</table>
<h2>Distributions</h2><div class="charts">{charts}</div>
<h2>Renders</h2><div class="cards">{cards}</div>
</body></html>
"""


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Gate 1 QA report: renders and distributions of the canonical record.")
    parser.add_argument("--out", required=True, help="dataset directory, e.g. data/pretrain/v1.0-pilot")
    parser.add_argument("--set", default="smoke", choices=SETS)
    parser.add_argument("--n", type=int, default=50, help="number of renders")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = build(args.out, args.set, args.n, args.seed)
    print(json.dumps({k: summary[k] for k in ("frames", "horizon_check_all_frames", "canonical_checks_failed")}, indent=1))
    print(f"open {os.path.join(args.out, 'qa', 'report.html')}")


if __name__ == "__main__":
    main()
