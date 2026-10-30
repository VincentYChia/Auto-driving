"""A few samples from the metadata index, as a page to open in a browser: ``audit/examples.html``.

For each one it shows what the model gets as input, what it must output, and how the route
command and the decision label were decided. The page holds per-frame data (poses, tokens), so
unlike the audit report it stays local (``.gitignore``). ``audit.py`` writes it on every run.

How the two are decided:
* **Command** is an input, copied unchanged from OpenScene's ``driving_command``. OpenScene's
  ``get_driving_command`` (V, ``DriveEngine/process_data/helpers/driving_command.py``) follows the
  route's lane centreline to the point 20 m past the car's projection onto it. A sideways offset
  there of at least +2 m is "left", at most -2 m is "right", and anything between is "forward"
  (our name: "straight"). With no usable route it is "unknown". The route is the nuPlan scene's
  roadblock list.
* **Label** is a target, computed from the recorded future path by ``data_pipeline/labels.py``
  (rules v0.1; the thresholds are [proposed] and uncalibrated). Longitudinal: the speed now vs
  the mean speed 2-3 s ahead. Lateral: the heading change and sideways offset at +4 s.

    python -m data_pipeline.v1.examples --out data/pretrain/v1.0-pilot [--n 3]
"""

from __future__ import annotations

import argparse
import html
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from data_pipeline import labels
from data_pipeline.v1 import selection, tables
from data_pipeline.v1.openscene_index import ANCHOR, PATH_OFFSETS_S

FUTURE = slice(ANCHOR + 1, ANCHOR + 9)  # +0.5 .. +4.0 s: the 8 poses the pilot predicts
TURNS = ("turn_left", "turn_right")
SPEED_CHANGES = ("stop", "decelerate", "start", "accelerate")
BENDS = ("lane_change_left", "lane_change_right", "curve_left", "curve_right")


# --- why each label -------------------------------------------------------------------------------


def longitudinal_evidence(speed: float, future_xy: np.ndarray) -> Dict[str, float]:
    """The numbers ``labels.longitudinal_action`` compares (path at 0.5 s steps, 3 s horizon)."""
    points = np.vstack([np.zeros((1, 2)), np.asarray(future_xy, dtype=np.float64)])
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / 0.5
    return {
        "speed_now": float(speed),
        "speed_ahead": float(np.mean(speeds[4:6])),  # mean over 2.0-3.0 s
        "change_needed": max(labels.SPEED_CHANGE_ABS, labels.SPEED_CHANGE_REL * float(speed)),
    }


def lateral_evidence(future: np.ndarray) -> Dict[str, float]:
    """The numbers ``labels.lateral_action`` compares (pose at +4.0 s)."""
    x, y, yaw = (float(v) for v in np.asarray(future, dtype=np.float64)[7])
    return {"ahead_m": x, "left_m": y, "heading_deg": math.degrees(yaw), "shift_m": y - x * math.tan(yaw / 2.0)}


def why_longitudinal(ev: Dict[str, float], label: str) -> str:
    now, ahead, need = ev["speed_now"], ev["speed_ahead"], ev["change_needed"]
    stop = labels.STOP_SPEED
    if label == "stopped":
        return f"Below {stop} m/s now ({now:.1f}) and 2–3 s ahead ({ahead:.1f})."
    if label == "stop":
        return f"{now:.1f} m/s now, below {stop} m/s 2–3 s ahead ({ahead:.1f})."
    if label == "start":
        return f"Below {stop} m/s now ({now:.1f}), {ahead:.1f} m/s 2–3 s ahead."
    change = ahead - now
    verdict = {"accelerate": "more than", "decelerate": "more than", "keep_speed": "within"}[label]
    return (f"{now:.1f} m/s now, {ahead:.1f} m/s 2–3 s ahead: a change of {_signed(change, 1)} m/s, {verdict} "
            f"±{need:.1f} (the larger of {labels.SPEED_CHANGE_ABS:g} m/s and {labels.SPEED_CHANGE_REL:.0%} of the speed).")


def _signed(value: float, digits: int, plus: bool = True) -> str:
    """``+1.5`` / ``-1.5`` (``1.5`` without ``plus``), and ``0.0`` rather than ``-0.0``."""
    value = round(value, digits) + 0.0
    return f"{value:{'+' if plus else ''}.{digits}f}" if value else f"{0:.{digits}f}"


def why_lateral(ev: Dict[str, float], label: str) -> str:
    x, y, deg, shift = ev["ahead_m"], ev["left_m"], ev["heading_deg"], ev["shift_m"]
    where = f"At +4 s the car is {x:.1f} m ahead and {_signed(y, 1)} m to the left, heading {_signed(deg, 0)}°"
    if label == "stationary":
        return f"{where}: under {labels.MIN_FORWARD_M:g} m of forward travel."
    if label in TURNS:
        return f"{where}: more than ±{labels.TURN_DEG:g}° is a turn."
    if label.startswith("lane_change"):
        return (f"{where}. The sideways shift that a steady curve would not explain is {_signed(shift, 1)} m: "
                f"beyond ±{labels.LANE_CHANGE_LATERAL_M:g} m with under {labels.LANE_CHANGE_MAX_DEG:g}° of heading change is a lane change.")
    if label.startswith("curve"):
        return f"{where}: between ±{labels.CURVE_DEG:g}° and ±{labels.TURN_DEG:g}° is a curve."
    return f"{where}: within ±{labels.CURVE_DEG:g}° and no lane-change shift (shift {_signed(shift, 1)} m)."


def why_command(command: str, has_route: bool, one_hot_ok: bool = True) -> str:
    if not one_hot_ok:
        return "The log's command is not a valid one-hot, so it is read as unknown."
    if not has_route:
        return "No route in this frame, so OpenScene's command is unknown."
    rule = {
        "left": "the route's centreline 20 m ahead is at least 2 m to the left",
        "right": "the route's centreline 20 m ahead is at least 2 m to the right",
        "straight": "the route's centreline 20 m ahead is within ±2 m (OpenScene: forward)",
        "unknown": "OpenScene could not match the route to the map",
    }[command]
    return f"Copied from OpenScene: {rule}."


# --- picking and describing rows ------------------------------------------------------------------


def pick(cands: tables.Columns, n: int = 3, seed: int = 0) -> List[int]:
    """Up to ``n`` eligible rows: a turn, a speed change, a curve or lane change, then new label pairs."""
    ok = selection.eligible(cands)
    lon, lat = cands["decision_lon"], cands["decision_lat"]
    turn = np.isin(lat, TURNS)
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for group in (ok & turn, ok & np.isin(lon, SPEED_CHANGES) & ~turn, ok & np.isin(lat, BENDS)):
        rows = np.flatnonzero(group)
        if len(chosen) < n and len(rows):
            chosen.append(int(rng.choice(rows)))
    pairs = np.array([f"{a}+{b}" for a, b in zip(lon, lat)], dtype=object)
    while len(chosen) < n:
        unseen = ok & ~np.isin(pairs, pairs[chosen])
        rows = np.setdiff1d(np.flatnonzero(unseen if unseen.any() else ok), chosen)
        if not len(rows):
            break
        chosen.append(int(rng.choice(rows)))
    return chosen


def describe(cands: tables.Columns, row: int, title: str = "") -> Dict[str, Any]:
    """One example: identity, inputs, outputs and the reasons, from a row of the index."""
    path = cands["path_xyyaw"][row].astype(np.float64)
    valid = cands["path_valid"][row]
    future = path[FUTURE]
    lon, lat = str(cands["decision_lon"][row]), str(cands["decision_lat"][row])
    command, has_route = str(cands["command"][row]), bool(cands["has_route"][row])
    lon_ev = longitudinal_evidence(float(cands["speed"][row]), future[:, :2])
    lat_ev = lateral_evidence(future)
    return {
        "title": title or f"{lon.replace('_', ' ')} + {lat.replace('_', ' ')}",
        "frame_id": str(cands["frame_id"][row]),
        "city": str(cands["city"][row]),
        "split": str(cands["our_split"][row]),
        "image": str(cands["cam_f0_path"][row]),
        "speed": float(cands["speed"][row]),
        "accel_lon": float(cands["accel_lon"][row]),
        "accel_lat": float(cands["accel_lat"][row]),
        "yaw_rate": float(cands["yaw_rate"][row]),
        "command": command,
        "has_route": has_route,
        "history": [(float(PATH_OFFSETS_S[k]), path[k]) for k in range(ANCHOR - 3, ANCHOR)],
        "future": [(float(PATH_OFFSETS_S[k]), path[k]) for k in range(FUTURE.start, FUTURE.stop)],
        "path": path,
        "valid": valid,
        "lon": lon,
        "lat": lat,
        "why_command": why_command(command, has_route, bool(cands["command_onehot_ok"][row])),
        "why_lon": why_longitudinal(lon_ev, lon),
        "why_lat": why_lateral(lat_ev, lat),
        "flags": [name for name, on in (
            ("in navtrain", bool(cands["in_navtrain"][row])),
            ("window crosses a scene boundary", bool(cands["crosses_scene_boundary"][row])),
            ("logged speed disagrees with the path", not bool(cands["speed_consistent"][row])),
        ) if on],
    }


# --- page -----------------------------------------------------------------------------------------

STYLE = """
/* One column of examples: the bird's-eye plot beside the input/output tables, stacked on phones. */
:root {
  --bg: #f3f4f5; --surface: #ffffff; --fg: #1b1f24; --muted: #5d6671; --line: #d6dadf;
  --target: #2052b8; --history: #8b939d; --route: #b98a00; --flag: #a4321f;
  --display: "Overpass", "Helvetica Neue", Arial, sans-serif;
  --body: "Overpass", "Helvetica Neue", Arial, sans-serif;
  --mono: "Overpass Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121417; --surface: #1a1d21; --fg: #e5e7ea; --muted: #9aa3ad; --line: #2d3238;
  --target: #79a3ff; --history: #6e7781; --route: #e2b33a; --flag: #ef8a74; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121417; --surface: #1a1d21; --fg: #e5e7ea; --muted: #9aa3ad; --line: #2d3238;
  --target: #79a3ff; --history: #6e7781; --route: #e2b33a; --flag: #ef8a74; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1040px; margin: 0 auto; padding-block: 28px 56px; display: grid; gap: 28px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.2; }
h1 { font-size: 1.9rem; font-weight: 800; }
h2 { font-size: 1.3rem; font-weight: 700; }
h3 { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
p { margin: 0; max-width: 68ch; }
.lede { display: grid; gap: 10px; }
.note { color: var(--muted); }
.rules { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.rule { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 16px; display: grid; gap: 8px; min-width: 0; }
.rule ul { margin: 0; padding-left: 1.1em; display: grid; gap: 4px; }
.example { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 18px; display: grid; gap: 14px; }
.example header { display: grid; gap: 2px; }
.id { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); overflow-wrap: anywhere; }
.body { display: grid; grid-template-columns: minmax(0, 340px) minmax(0, 1fr); gap: 20px; align-items: start; }
@media (max-width: 720px) { .body { grid-template-columns: minmax(0, 1fr); } }
.plot { width: 100%; max-width: 340px; height: auto; display: block; }
.plot .grid { stroke: var(--line); stroke-width: 1; fill: none; }
.plot .tick { fill: var(--muted); font: 10px var(--mono); }
.plot .history { stroke: var(--history); stroke-width: 2.5; fill: none; }
.plot .target { stroke: var(--target); stroke-width: 3; fill: none; }
.plot .beyond { stroke: var(--target); stroke-width: 2; fill: none; stroke-dasharray: 4 4; opacity: 0.6; }
.plot .dot { fill: var(--target); }
.plot .route { stroke: var(--route); stroke-width: 2; fill: none; stroke-dasharray: 7 5; }
.plot .look { fill: var(--route); }
.plot .ego { fill: var(--surface); stroke: var(--fg); stroke-width: 1.5; }
.plot .label { fill: var(--fg); font: 11px var(--mono); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 0.8rem; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 14px; height: 3px; margin-right: 6px; vertical-align: middle; background: var(--c); }
.cols { display: grid; gap: 14px; min-width: 0; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 14px; margin: 0; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.mono { font-family: var(--mono); font-size: 0.85rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-family: var(--mono); font-size: 0.82rem; font-variant-numeric: tabular-nums; }
th, td { padding: 3px 10px 3px 0; text-align: right; white-space: nowrap; }
th { color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--line); }
.label-out { font-family: var(--display); font-size: 1.15rem; font-weight: 700; }
.why { color: var(--muted); }
.flag { color: var(--flag); }
"""


def _plot(ex: Dict[str, Any]) -> str:
    """Bird's-eye view in the anchor's frame: forward is up, left is left."""
    path, valid = ex["path"], ex["valid"]
    points = [path[k, :2] for k in range(len(path)) if valid[k]]
    extra = [np.asarray(p, dtype=np.float64) for p in ex.get("route", [])]
    if ex.get("lookahead") is not None:
        extra.append(np.asarray(ex["lookahead"][:2], dtype=np.float64))
    xy = np.array(points + extra + [np.array([-1.0, 0.0]), np.array([4.0, 0.0])])
    u, v = -xy[:, 1], -xy[:, 0]  # screen right = -left, screen down = -forward
    span = max(u.max() - u.min(), v.max() - v.min(), 20.0) + 8.0
    cu, cv = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    size, pad = 340.0, 22.0
    scale = (size - 2 * pad) / span

    def px(x: float, y: float) -> str:
        return f"{pad + (-y - cu + span / 2) * scale:.1f},{pad + (-x - cv + span / 2) * scale:.1f}"

    step = 5.0 if span <= 40 else 10.0 if span <= 90 else 20.0
    out = [f'<svg class="plot" viewBox="0 0 {size:.0f} {size:.0f}" role="img" aria-label="Bird\'s-eye view of the path">']
    lo_x, hi_x = -(cv + span / 2), -(cv - span / 2)
    lo_y, hi_y = -(cu + span / 2), -(cu - span / 2)
    for g in np.arange(math.ceil(lo_x / step) * step, hi_x, step):
        a, b = px(g, hi_y).split(","), px(g, lo_y).split(",")
        out.append(f'<line class="grid" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>')
        out.append(f'<text class="tick" x="{float(b[0]) - 2:.1f}" y="{float(b[1]) - 3:.1f}" text-anchor="end">{g:+.0f} m</text>')
    for g in np.arange(math.ceil(lo_y / step) * step, hi_y, step):
        a, b = px(hi_x, g).split(","), px(lo_x, g).split(",")
        out.append(f'<line class="grid" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>')
        out.append(f'<text class="tick" x="{b[0]}" y="{float(b[1]) + 13:.1f}" text-anchor="middle">{g:+.0f}</text>')
    if ex.get("route"):
        out.append(f'<polyline class="route" points="{" ".join(px(x, y) for x, y in ex["route"])}"/>')
    history = [path[k] for k in range(0, ANCHOR + 1) if valid[k]]
    target = [path[k] for k in range(ANCHOR, FUTURE.stop) if valid[k]]
    beyond = [path[k] for k in range(FUTURE.stop - 1, len(path)) if valid[k]]
    for cls, pts in (("history", history), ("beyond", beyond), ("target", target)):
        if len(pts) > 1:
            out.append(f'<polyline class="{cls}" points="{" ".join(px(p[0], p[1]) for p in pts)}"/>')
    for p in target[1:]:
        c = px(p[0], p[1]).split(",")
        out.append(f'<circle class="dot" cx="{c[0]}" cy="{c[1]}" r="3"/>')
    if ex.get("lookahead") is not None:
        lx, ly, offset = ex["lookahead"]
        c = px(lx, ly).split(",")
        out.append(f'<circle class="look" cx="{c[0]}" cy="{c[1]}" r="5"/>')
        right = float(c[0]) > size * 0.55  # keep the label inside the plot
        out.append(f'<text class="label" x="{float(c[0]) + (-8 if right else 8):.1f}" y="{float(c[1]) + 4:.1f}" '
                   f'text-anchor="{"end" if right else "start"}">20 m: {_signed(offset, 1)} m</text>')
    corners = [(-1.0, -0.95), (3.9, -0.95), (3.9, 0.95), (-1.0, 0.95)]  # car outline, origin at the rear axle
    out.append(f'<polygon class="ego" points="{" ".join(px(x, y) for x, y in corners)}"/>')
    out.append("</svg>")
    return "".join(out)


def _card(ex: Dict[str, Any]) -> str:
    e = html.escape
    history = " ".join(f"({_signed(p[0], 1, False)}, {_signed(p[1], 1, False)})" for _, p in ex["history"])
    rows = "".join(
        f"<tr><td>{t:+.1f} s</td><td>{p[0]:.2f}</td><td>{_signed(p[1], 2)}</td><td>{_signed(math.degrees(p[2]), 1)}°</td></tr>"
        for t, p in ex["future"]
    )
    legend = ['<span style="--c: var(--history)">past 4 s</span>', '<span style="--c: var(--target)">target: next 4 s</span>']
    if ex.get("route"):
        legend.append('<span style="--c: var(--route)">route centreline</span>')
    flags = "".join(f'<p class="flag">Flag: {e(f)}</p>' for f in ex["flags"])
    return f"""
<section class="example">
  <header><h2>{e(ex["title"])}</h2><div class="id">{e(ex["frame_id"])} · {e(ex["city"])} · split: {e(ex["split"])}</div></header>
  <div class="body">
    <div class="cols">{_plot(ex)}<div class="legend">{"".join(legend)}</div></div>
    <div class="cols">
      <h3>Input at time t</h3>
      <dl class="kv">
        <dt>Front image</dt><dd class="mono">{e(ex["image"] or "none")}</dd>
        <dt>Speed</dt><dd>{ex["speed"]:.1f} m/s</dd>
        <dt>Acceleration</dt><dd>{_signed(ex["accel_lon"], 2)} m/s² forward, {_signed(ex["accel_lat"], 2)} m/s² left</dd>
        <dt>Command</dt><dd><strong>{e(ex["command"])}</strong></dd>
        <dt>Past positions</dt><dd class="mono">{history} m at −1.5, −1.0, −0.5 s</dd>
      </dl>
      <p class="why">{e(ex["why_command"])}</p>
      <h3>Output</h3>
      <div class="scroll"><table><thead><tr><th>time</th><th>ahead (m)</th><th>left (m)</th><th>heading</th></tr></thead><tbody>{rows}</tbody></table></div>
      <p class="label-out">{e(ex["lon"])} + {e(ex["lat"])}</p>
      <p class="why">{e(ex["why_lon"])}</p>
      <p class="why">{e(ex["why_lat"])}</p>
      {flags}
    </div>
  </div>
</section>"""


RULES = f"""
<section class="rules">
  <div class="rule"><h3>Command (an input)</h3>
    <p>Copied unchanged from OpenScene, which derives it from the route, not from where the car went next:</p>
    <ul><li>Follow the route's lane centreline to 20 m past the car.</li>
    <li>Offset of +2 m or more there: <strong>left</strong>; −2 m or less: <strong>right</strong>; between: <strong>straight</strong> (OpenScene: forward).</li>
    <li>No usable route: <strong>unknown</strong>.</li></ul></div>
  <div class="rule"><h3>Label (a target, rules v{labels.RULES_VERSION}, uncalibrated)</h3>
    <p>Computed from the recorded future path:</p>
    <ul><li><strong>Speed:</strong> now vs the mean 2–3 s ahead. Stop below {labels.STOP_SPEED} m/s; otherwise a change beyond the larger of {labels.SPEED_CHANGE_ABS:g} m/s and {labels.SPEED_CHANGE_REL:.0%} counts.</li>
    <li><strong>Direction</strong> at +4 s: turn beyond ±{labels.TURN_DEG:g}°, curve beyond ±{labels.CURVE_DEG:g}°, lane change for a sideways shift beyond {labels.LANE_CHANGE_LATERAL_M:g} m with under {labels.LANE_CHANGE_MAX_DEG:g}° of turning, stationary under {labels.MIN_FORWARD_M:g} m of travel.</li></ul></div>
</section>"""


def render(examples: Sequence[Dict[str, Any]], heading: str, intro: str, standalone: bool = True) -> str:
    fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Overpass:wght@400;600;700;800'
             '&family=Overpass+Mono:wght@400;600&display=swap">')
    head = (f'<title>{html.escape(heading)}</title><meta name="claude:viewport-fit-sides" content="cover">'
            f'{fonts}<style>{STYLE}</style>')
    body = (f'<main><div class="lede"><h1>{html.escape(heading)}</h1>{intro}</div>{RULES}'
            + "".join(_card(ex) for ex in examples) + "</main>")
    if not standalone:  # an Artifact page: the publisher adds the document skeleton
        return head + body
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            f'{head}</head><body>{body}</body></html>')


def write(out_dir: str, n: int = 3, seed: int = 0) -> Optional[str]:
    """Write ``<out_dir>/audit/examples.html`` from the index. Returns its path (None if no eligible rows)."""
    cands = tables.read(os.path.join(out_dir, "index", "candidates.parquet"))
    rows = pick(cands, n=n, seed=seed)
    if not rows:
        return None
    intro = (f'<p class="note">{len(rows)} rows from the index in <span class="mono">{html.escape(out_dir)}</span>, picked to show a turn, '
             "a speed change and a curve or lane change where the data has them. The images themselves come in step 3. "
             "This page holds per-frame data, so it stays on this computer.</p>")
    path = os.path.join(out_dir, "audit", "examples.html")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render([describe(cands, r) for r in rows], "Example samples", intro))
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Write example samples from the metadata index as an HTML page.")
    parser.add_argument("--out", required=True, help="dataset directory holding index/, e.g. data/pretrain/v1.0-pilot")
    parser.add_argument("--n", type=int, default=3, help="number of examples")
    args = parser.parse_args(argv)
    print(write(args.out, n=args.n) or "no eligible rows in the index")


if __name__ == "__main__":
    main()
