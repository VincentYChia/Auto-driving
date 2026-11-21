"""Build the path-generator study's HTML overview report from the aggregate JSON (``collect_results.py``).

    python pilot_sft/heads/report_html.py --aggregate agg.json --notes notes.json --out report.html

The page has no camera images, only aggregate numbers. ``notes.json`` holds the words that change with the results:
``asof``, ``headline``, ``lede``, and lists (HTML strings) ``done``, ``meaning``, ``problems``, ``next``,
``decisions``, ``queue``. Every chart has a table view and a hover tooltip. Methods keep fixed colours (A blue,
B orange, C aqua, D yellow, R magenta), validated with the dataviz palette checker in light and dark mode.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from typing import Dict, Optional

METHOD_NAMES = {"A": "A, masked diffusion", "B": "B, DDIM", "C": "C, flow", "D": "D, anchored", "R": "R, regression"}
ORDER = ["A", "B", "C", "D", "R"]


def method_of(run: str) -> str:
    m = re.match(r"selector-([a-z])\d?-", run) or re.match(r"([a-z])\d?-", run)
    return m.group(1).upper() if m else "?"


def seed_of(run: str) -> int:
    """The supervised seed a run descends from: ``-s1`` / ``-s2`` in its name, else 0."""
    m = re.search(r"-s(\d)(?:-|$)", run)
    return int(m.group(1)) if m else 0


def layers_of(run: str) -> str:
    return "12/18/24" if "-l12-18-24" in run else ("24" if "-l24" in run else "?")


def recipe_of(run: str) -> Optional[str]:
    m = re.search(r"-rlrecipe(\d)", run)
    if m:
        return f"RL{m.group(1)}"
    if run.endswith("-rl-trial"):
        return "trial"
    if "-gen-half" in run:  # phase 6's half-data path generators (R14), before the "-rl" suffix test
        return "phase 6, half generators"
    m = re.search(r"-rlfull(\d)$", run)
    if m:
        return f"phase 6, full RL (RL{m.group(1)})"
    m = re.search(r"-rl(\d)$", run)
    if m:
        return f"phase 4, RL{m.group(1)}"
    if run.endswith("-rl"):  # phase 4's registered recipe (R11, R13: RL4)
        return "phase 4, RL4"
    return None


def iv(x: Optional[dict], scale: float = 100.0) -> Optional[dict]:
    """{pdms, ci_drives, vs_h1_text, vs_start} -> rounded points, in PDMS points (x100)."""
    if not x:
        return None
    out = {"x": round(x["pdms"] * scale, 2), "lo": round(x["ci_drives"][0] * scale, 2),
           "hi": round(x["ci_drives"][1] * scale, 2)}
    for k, n in (("vs_h1_text", "vsH1"), ("vs_start", "vsStart")):
        if x.get(k):
            out[n] = diff(x[k])
    if x.get("diversity"):
        out["div"] = x["diversity"]
    if x.get("subscores"):
        out["subs"] = x["subscores"]
    return out


def diff(d: dict) -> dict:
    return {"x": round(d["difference"] * 100, 2), "lo": round(d["ci_drives"][0] * 100, 2),
            "hi": round(d["ci_drives"][1] * 100, 2)}


def page_data(agg: dict, ms: Optional[dict] = None) -> dict:
    refs = {k: iv(v) for k, v in agg.get("references", {}).items()}
    runs = []
    for run, r in agg.get("runs", {}).items():
        runs.append({"run": run, "method": method_of(run), "layers": layers_of(run), "choice": r.get("choice", {}),
                     "dev_ade": r.get("dev_ade"),
                     **{f"{sp}_{k}": iv(r.get(f"{sp}_{k}")) for sp in ("navdev", "navtest")
                        for k in ("single", "best_of_6")}})
    runs.sort(key=lambda r: (ORDER.index(r["method"]) if r["method"] in ORDER else 9, r["layers"] != "24", r["run"]))
    layer = [{"method": m, "runs": t["runs"], "chosen": t.get("chosen"),
              **{k: diff(t[k]) for k in t if k.startswith("nav")}} for m, t in agg.get("layer_tests", {}).items()]
    pairs = [{"pair": p, "runs": c["runs"], **{k: diff(c[k]) for k in c if k.startswith("nav")}}
             for p, c in agg.get("comparisons", {}).items()]
    rl = []
    for run, r in agg.get("rl", {}).items():
        cps = [{"u": int(u), "single": iv(c.get("single")), "b6": iv(c.get("best_of_6"))}
               for u, c in sorted(r.get("checkpoints", {}).items(), key=lambda kv: int(kv[0]))]
        rl.append({"run": run, "method": method_of(run), "seed": seed_of(run), "recipe": recipe_of(run),
                   "settings": r.get("settings", {}),
                   "bc_weight": r.get("bc_weight"), "rl_scenes": r.get("rl_scenes"), "done": r.get("updates_done"),
                   "checkpoints": cps,
                   "curve": r.get("curve", []), "choice": (r.get("choice") or {}).get("checkpoint")})
    rl.sort(key=lambda r: (r["recipe"] or "", ORDER.index(r["method"]) if r["method"] in ORDER else 9, r["seed"]))
    seeds = []
    for m in ORDER:
        lc = agg.get("methods", {}).get(m)
        if not isinstance(lc, dict) or "run" not in lc:
            continue
        r0, extra = agg.get("runs", {}).get(lc["run"], {}), agg.get("methods", {}).get(f"{m}_seeds") or {}
        vals = {}
        for sp in ("navdev", "navtest"):
            for k in ("single", "best_of_6"):
                first = (r0.get(f"{sp}_{k}") or {}).get("pdms")
                rest = [extra.get(f"{lc['run']}-s{sd}", {}).get(f"{sp}_{k}" if k == "best_of_6" else f"{sp}_single_pdms")
                        for sd in (1, 2)]
                vals[f"{sp}_{k}"] = [None if v is None else round(100 * v, 2) for v in [first] + rest]
        seeds.append({"method": m, "run": lc["run"], "values": vals})
    return {"refs": refs, "runs": runs, "layer": layer, "pairs": pairs, "rl": rl, "seeds": seeds, "methods": agg.get("methods", {}),
            "seedPairs": seed_pairs(ms), "series": series_rows(agg), "phase4": phase4_rows(ms), "selectors": selector_rows(agg), "seriesBest": {sp: r["series_best_vs_a1"] for sp, r in
                                                                                    (ms or {}).items() if r.get("series_best_vs_a1")},
            "generated": agg.get("generated"), "names": METHOD_NAMES}


SEL_ORDER = ("xfit-s0", "xfit-s1", "infold-s0", "xfit-pdms", "xfit-listwise", "xfit-imitation", "xfit-boundary")


def selector_rows(agg: dict) -> list:
    """Phase 6 (R14): per chain (a contender, or the rehearsal last), each selector variant's navdev measures in points
    (selected minus own pick with its drive interval) and the chain's adoption."""
    pct = lambda x: None if x is None else round(100 * x, 2)  # noqa: E731
    chains: Dict[str, dict] = {}
    for name, r in agg.get("selectors", {}).items():
        row = {"name": name, "variant": r["variant"], "heldout": r.get("heldout", {}), "train": r.get("train", {})}
        for sp in ("navdev", "navtest"):
            n = r.get(sp)
            if n and "selected_minus_own" in n:
                ci = n.get("selected_minus_own_ci") or [None, None]
                row[sp] = {"selected": pct(n.get("selected_pdms")), "own": pct(n.get("own_pick_pdms")),
                           "best": pct(n.get("best_of_k_pdms")), "gain": {"x": pct(n["selected_minus_own"]),
                                                                          "lo": pct(ci[0]), "hi": pct(ci[1])},
                           "share": n.get("share_of_best_of_k_gap_closed"), "switch": n.get("switch_rate"),
                           "switches": n.get("switches"), "fail": n.get("picks_failing_nc_or_dac"),
                           "top1": n.get("top1_tie_aware"), "scenes": n.get("scenes"), "candidates": n.get("candidates")}
        chains.setdefault(r["chain"], {"chain": r["chain"], "variants": []})["variants"].append(row)
    out = []
    for x in sorted(chains, key=lambda c: (c == "rehearsal", c)):
        ch = chains[x]
        ch["variants"].sort(key=lambda v: SEL_ORDER.index(v["variant"]) if v["variant"] in SEL_ORDER else 99)
        ad = agg.get("methods", {}).get(f"selector_{x}_adoption")
        if isinstance(ad, dict):
            ch["adoption"] = {k: ad.get(k) for k in ("adopted", "gain_claimed", "role")}
        out.append(ch)
    return out


def phase4_rows(ms: Optional[dict]) -> dict:
    """Phase 4 with every seed (method_seeds.py's ``phase4``), per split and recipe: each method after RL, its change
    from phase 3 (seed-paired), and the method pairs after RL; differences in points with their intervals."""
    iv_ = lambda c: {"x": c["difference"], "lo": c["ci"][0], "hi": c["ci"][1], "seeds": c["seeds"]}  # noqa: E731
    out = {}
    for sp, r in (ms or {}).items():
        for rec, v in sorted((r.get("phase4") or {}).items()):
            out.setdefault(sp, {})[rec] = {
                "methods": [{"method": m, **mv} for m, mv in sorted(v.get("methods", {}).items())],
                "change": [{"method": m, **{k: iv_(c) for k, c in cv.items()}}
                           for m, cv in sorted(v.get("change_from_phase3", {}).items())],
                "pairs": [{"pair": p, **{k: iv_(c) for k, c in pv.items()}} for p, pv in v.get("pairs", {}).items()]}
    return out


SERIES_LABELS = {"a0": "A0: plain bins, no notion of distance", "a2": "A2: A1 + soft labels",
                 "a3": "A3: A2 + position-error loss and averaged readout", "a4": "A4: A3 + continuous offset in the bin"}
FEATURE_NAMES = {"metric": "distance-aware bin embeddings", "soft": "soft labels", "l1": "the position-error loss",
                 "offset": "the continuous offset"}


def series_label(config: str, variant: str) -> str:
    if not variant:
        return SERIES_LABELS.get(config, config.upper())
    if variant.startswith("no-"):
        f = variant[3:]
        return f"{config.upper()} without {FEATURE_NAMES.get(f, f)}"
    if variant.startswith("bin"):
        return f"{config.upper()} with {variant[3]}.{variant[4:]} m bins"
    return f"{config.upper()} {variant}"


def series_rows(agg: dict) -> list:
    """Phase 2 (R17): each series run minus A1, in points, with R17's test per measure."""
    rows = []
    for run, r in agg.get("series", {}).items():
        v = r.get("variant", "")
        short = r["config"].upper() + ("" if not v else f" {v}" if v.startswith("no-") else f" {v[3]}.{v[4:]} m bins")
        row = {"run": run, "variant": v, "label": series_label(r["config"], v), "short": short,
               "measured": r.get("measured_effect")}
        for k, v in r.items():
            if k.startswith("nav") and isinstance(v, dict):
                row[k] = {**diff(v), "beyond": v.get("beyond_seed_noise")}
        rows.append(row)
    rows.sort(key=lambda x: (x["variant"] != "", x["run"]))  # the series first, then leave-one-out and bin sizes
    return rows


def seed_pairs(ms: Optional[dict]) -> dict:
    """Method pairs with every seed, from method_seeds.py (points; R5's hierarchical bootstrap), per split."""
    out = {}
    for sp, r in (ms or {}).items():
        rows = []
        for p, v in r.get("pairs", {}).items():
            row = {"pair": p}
            for k in ("single", "best_of_6"):
                if k in v:
                    row[k] = {"x": v[k]["difference"], "lo": v[k]["ci"][0], "hi": v[k]["ci"][1]}
                    row["seeds"] = v[k]["seeds"]
            rows.append(row)
        out[sp] = {"pairs": rows, "fill": r.get("seed_sd_added_for_single_seed", {})}
    return out


CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --band: rgba(137,135,129,0.14);
  --mA: #2a78d6; --mB: #eb6834; --mC: #1baf7a; --mD: #eda100; --mR: #e87ba4; --ref: #52514e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --band: rgba(137,135,129,0.18);
    --mA: #3987e5; --mB: #d95926; --mC: #199e70; --mD: #c98500; --mR: #d55181; --ref: #c3c2b7;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --band: rgba(137,135,129,0.18);
  --mA: #3987e5; --mB: #d95926; --mC: #199e70; --mD: #c98500; --mR: #d55181; --ref: #c3c2b7;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 32px 16px 64px; }
.kicker { color: var(--muted); font-size: 13px; letter-spacing: .02em; text-transform: uppercase; }
h1 { font-size: 30px; line-height: 1.2; margin: 6px 0 10px; font-weight: 650; }
h2 { font-size: 21px; line-height: 1.3; margin: 0 0 6px; font-weight: 620; }
h3 { font-size: 16px; margin: 16px 0 4px; font-weight: 620; }
p { margin: 0 0 12px; }
.lede { color: var(--ink-2); font-size: 17px; max-width: 780px; }
section { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 22px 22px 18px; margin: 18px 0; }
.sub { color: var(--ink-2); margin-bottom: 12px; }
ul.points { margin: 8px 0 4px; padding-left: 20px; } ul.points li { margin: 5px 0; }
.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0 6px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 14px 16px; }
.tile .label { color: var(--ink-2); font-size: 13px; } .tile .value { font-size: 28px; font-weight: 620; line-height: 1.2; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 13px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 4px 0 10px; font-size: 14px; color: var(--ink-2); }
.seg { display: inline-flex; border: 1px solid var(--ring); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; font-size: 14px; padding: 6px 12px; border: 0; background: var(--page); color: var(--ink-2); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.chart { width: 100%; position: relative; } .chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--ink-2); font-size: 13px; margin: 2px 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.key.hollow { background: var(--surface); border: 2px solid var(--ink-2); }
.key.solid { background: var(--ink-2); }
.key.line { width: 16px; height: 0; border-top: 2px dashed var(--ref); border-radius: 0; }
details { margin-top: 10px; color: var(--ink-2); font-size: 14px; } summary { cursor: pointer; color: var(--muted); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 8px; }
th, td { text-align: right; padding: 5px 7px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; } th { color: var(--ink-2); font-weight: 560; }
.small { font-size: 13px; color: var(--muted); }
.mult { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--ring);
  box-shadow: 0 4px 14px rgba(0,0,0,.12); border-radius: 8px; padding: 8px 10px; font-size: 13px; line-height: 1.4; max-width: 300px; opacity: 0; z-index: 10; }
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
@media (max-width: 720px) { .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } .mult { grid-template-columns: 1fr; } h1 { font-size: 25px; } section { padding: 18px 14px 14px; } }
"""

JS = r"""
const THEME = new URLSearchParams(location.search).get('theme');  // ?theme=light or ?theme=dark overrides the OS
if (THEME === 'light' || THEME === 'dark') document.documentElement.dataset.theme = THEME;
const D = JSON.parse(document.getElementById('data').textContent);
const NOTES = JSON.parse(document.getElementById('notes').textContent);
const C = m => `var(--m${m})`;
const f1 = v => (v == null ? '–' : v.toFixed(1));
const sgn = v => (v == null ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1));
const ivs = o => o ? `${f1(o.x)} (${f1(o.lo)} to ${f1(o.hi)})` : '–';
const dvs = o => o ? `${sgn(o.x)} (${sgn(o.lo)} to ${sgn(o.hi)})` : '–';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const tip = document.getElementById('tip');
function showTip(e, h) { tip.innerHTML = h; tip.style.opacity = 1; const x = Math.min(e.clientX + 14, innerWidth - 310); tip.style.left = x + 'px'; tip.style.top = (e.clientY + 14) + 'px'; }
function hideTip() { tip.style.opacity = 0; }
function niceTicks(a, b, n) {
  const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => span / s <= n) || 10 * mag;
  const t = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) t.push(+v.toFixed(6)); return t;
}
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
// A forest plot: rows of marks {x, lo, hi, kind: 'solid'|'hollow', tip}, optional reference lines and a zero line.
function forest(box, rows, o) {
  box.innerHTML = '';
  const W = Math.max(300, box.clientWidth || 800), narrow = W < 560;
  const left = narrow ? 118 : 210, right = 14, top = 26, rowH = 28, bottom = 30;
  const H = top + rows.length * rowH + bottom;
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.aria || ''}, box);
  const xs = []; rows.forEach(r => r.marks.forEach(m => { if (m.x != null) xs.push(m.lo ?? m.x, m.hi ?? m.x); }));
  (o.refs || []).forEach(r => xs.push(r.x)); if (o.zero) xs.push(0);
  if (!xs.length) { box.innerHTML = '<p class="small">No scores yet.</p>'; return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs); const pad = (x1 - x0) * 0.06 || 1; x0 -= pad; x1 += pad;
  const sx = v => left + (v - x0) / (x1 - x0) * (W - left - right);
  const T = niceTicks(x0, x1, narrow ? 4 : 8), st = T.length > 1 ? T[1] - T[0] : 1;
  let dec = 0; while (dec < 3 && Math.abs(st * 10 ** dec - Math.round(st * 10 ** dec)) > 1e-6) dec++;
  const fmt = v => (o.signed ? (v > 1e-9 ? '+' : v < -1e-9 ? '−' : '') + Math.abs(v).toFixed(dec) : v.toFixed(dec));
  for (const t of T) {
    el('line', {x1: sx(t), x2: sx(t), y1: top - 6, y2: H - bottom, stroke: 'var(--grid)', 'stroke-width': 1}, svg);
    const tx = el('text', {x: sx(t), y: H - bottom + 16, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)'}, svg); tx.textContent = fmt(t);
  }
  if (o.zero) el('line', {x1: sx(0), x2: sx(0), y1: top - 8, y2: H - bottom, stroke: 'var(--axis)', 'stroke-width': 2}, svg);
  (o.refs || []).forEach(r => {
    el('line', {x1: sx(r.x), x2: sx(r.x), y1: top - 8, y2: H - bottom, stroke: 'var(--ref)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3'}, svg);
    const t = el('text', {x: sx(r.x) + 4, y: top - 12, 'font-size': 12, fill: 'var(--ink-2)'}, svg); t.textContent = r.label;
  });
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    if (i % 2 === 1) el('rect', {x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'var(--band)', opacity: 0.35}, svg);
    const lab = el('text', {x: 0, y: y + 4, 'font-size': narrow ? 12 : 13, fill: 'var(--ink)'}, svg); lab.textContent = r.label;
    if (r.sub && !narrow) { const s = el('tspan', {fill: 'var(--muted)', 'font-size': 12}, lab); s.textContent = '  ' + r.sub; }
    r.marks.forEach((m, j) => {
      if (m.x == null) return;
      const dy = r.marks.length > 1 ? (j - (r.marks.length - 1) / 2) * 8 : 0;
      if (m.lo != null) el('line', {x1: sx(m.lo), x2: sx(m.hi), y1: y + dy, y2: y + dy, stroke: r.color, 'stroke-width': 2, 'stroke-linecap': 'round'}, svg);
      el('circle', {cx: sx(m.x), cy: y + dy, r: 5, fill: m.kind === 'hollow' ? 'var(--surface)' : r.color, stroke: m.kind === 'hollow' ? r.color : 'var(--surface)', 'stroke-width': 2}, svg);
      const hit = el('rect', {x: sx(m.lo ?? m.x) - 6, y: y + dy - 8, width: Math.max(12, sx(m.hi ?? m.x) - sx(m.lo ?? m.x) + 12), height: 16, fill: 'transparent'}, svg);
      hit.addEventListener('mousemove', e => showTip(e, m.tip)); hit.addEventListener('mouseleave', hideTip);
    });
  });
}
function table(box, head, body) {
  box.innerHTML = '<div class="tablewrap"><table><thead><tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>' +
    body.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
}
const runLabel = r => `${r.method} · layer${r.layers === '24' ? '' : 's'} ${r.layers}`;
const choiceText = c => `epoch ${c.epoch}, ${c.steps} steps${c.scale != null ? ', setting ' + c.scale : ''}`;
let SPLIT = 'navtest';
function drawPhase1() {
  const ref = D.refs['h1_text_' + SPLIT];
  const rows = D.runs.map(r => {
    const s = r[SPLIT + '_single'], b = r[SPLIT + '_best_of_6'];
    const t = (k, o) => o ? `<b>${esc(r.run)}</b> · ${k}<br>${SPLIT}: ${ivs(o)}<br>vs H1 text: ${dvs(o.vsH1)}${o.div ? `<br>diversity: end points ${o.div.endpoint_m} m apart` : ''}<br>choices: ${choiceText(r.choice)}` : '';
    return {label: runLabel(r), sub: r.run, color: C(r.method), marks: [
      {...(s || {}), kind: 'solid', tip: t('single path', s)}, {...(b || {}), kind: 'hollow', tip: t('best-of-6', b)}]};
  });
  forest(document.getElementById('ch-phase1'), rows, {refs: ref ? [{x: ref.x, label: `H1 text ${f1(ref.x)}`}] : [], aria: 'Single path and best-of-6 PDMS per training run'});
  table(document.getElementById('tb-phase1'), ['Training run', 'Choices (navdev)', 'Single path', 'vs H1', 'Best-of-6', 'vs H1', 'Diversity (m)', 'Dev ADE (m)'],
    D.runs.map(r => { const s = r[SPLIT + '_single'], b = r[SPLIT + '_best_of_6'];
      return [esc(r.run), choiceText(r.choice), ivs(s), s ? dvs(s.vsH1) : '–', ivs(b), b ? dvs(b.vsH1) : '–', b && b.div ? b.div.endpoint_m : '–', r.dev_ade != null ? r.dev_ade.toFixed(3) : '–']; }));
  const rowsL = [];
  D.layer.forEach(t => [['single', 'single path', 'solid'], ['best_of_6', 'best-of-6', 'hollow']].forEach(([k, name, kind]) => {
    const o = t[SPLIT + '_' + k]; if (!o) return;
    rowsL.push({label: `${t.method} · ${name}`, color: C(t.method), marks: [{...o, kind, tip: `<b>${t.method}</b>, ${name}, ${SPLIT}<br>layers 12/18/24 − layer 24: ${dvs(o)}<br>layer choice: ${esc(t.chosen || '–')}`}]});
  }));
  forest(document.getElementById('ch-layer'), rowsL, {zero: true, signed: true, aria: 'Layer test, paired differences'});
  table(document.getElementById('tb-layer'), ['Method', 'Single path, 12/18/24 − 24', 'Best-of-6, 12/18/24 − 24', 'Layer choice'],
    D.layer.map(t => [t.method, dvs(t[SPLIT + '_single']), dvs(t[SPLIT + '_best_of_6']), esc(t.chosen || '–')]));
  const rowsP = [];
  D.pairs.forEach(p => [['single', 'single path', 'solid'], ['best_of_6', 'best-of-6', 'hollow']].forEach(([k, name, kind]) => {
    const o = p[SPLIT + '_' + k]; if (!o) return;
    rowsP.push({label: `${p.pair.replace('-', ' − ')} · ${name}`, color: 'var(--ink-2)', marks: [{...o, kind, tip: `<b>${esc(p.runs[0])} − ${esc(p.runs[1])}</b><br>${name}, ${SPLIT}: ${dvs(o)}`}]});
  }));
  forest(document.getElementById('ch-pairs'), rowsP, {zero: true, signed: true, aria: 'Method pairs, paired differences'});
  table(document.getElementById('tb-pairs'), ['Pair (each at its navdev choices)', 'Single path', 'Best-of-6'],
    D.pairs.map(p => [`${esc(p.runs[0])} − ${esc(p.runs[1])}`, dvs(p[SPLIT + '_single']), dvs(p[SPLIT + '_best_of_6'])]));
  const cell = v => { const x = v.filter(y => y != null); if (!x.length) return '–';
    const m = x.reduce((a, b) => a + b, 0) / x.length;
    const sd = x.length > 1 ? Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1)) : null;
    return v.map(y => y == null ? '–' : y.toFixed(2)).join(' · ') + ` <span class="small">(mean ${m.toFixed(2)}${sd != null ? ', SD ' + sd.toFixed(2) : ''})</span>`; };
  table(document.getElementById('tb-seeds'), ['Method (layer choice)', 'Single path: seeds 0 · 1 · 2', 'Best-of-6: seeds 0 · 1 · 2'],
    (D.seeds || []).map(r => [`${r.method} <span class="small">${esc(r.run)}</span>`, cell(r.values[SPLIT + '_single']), cell(r.values[SPLIT + '_best_of_6'])]));
  const sp = ((D.seedPairs || {})[SPLIT] || {}).pairs || [], rowsS = [];
  sp.forEach(p => [['single', 'single path', 'solid'], ['best_of_6', 'best-of-6', 'hollow']].forEach(([k, name, kind]) => {
    const o = p[k]; if (!o) return;
    rowsS.push({label: `${p.pair.replace('-', ' − ')} · ${name}`, color: 'var(--ink-2)', marks: [{...o, kind,
      tip: `<b>${esc(p.pair.replace('-', ' − '))}</b>, ${p.seeds[0]} and ${p.seeds[1]} seeds<br>${name}, ${SPLIT}: ${dvs(o)}`}]});
  }));
  const spBox = document.getElementById('ch-seedpairs');
  if (rowsS.length) forest(spBox, rowsS, {zero: true, signed: true, aria: 'Method pairs with every seed'}); else spBox.innerHTML = '';
  table(document.getElementById('tb-seedpairs'), ['Pair (seed-averaged means)', 'Seeds', 'Single path', 'Best-of-6'],
    sp.map(p => [esc(p.pair.replace('-', ' − ')), `${p.seeds[0]} vs ${p.seeds[1]}`, p.single ? dvs(p.single) : '–', p.best_of_6 ? dvs(p.best_of_6) : '–']));
  const ser = D.series || [], rowsSer = [];
  ser.forEach(r => [['single', 'single path', 'solid'], ['best_of_6', 'best-of-6', 'hollow']].forEach(([k, name, kind]) => {
    const o = r[SPLIT + '_' + k]; if (!o) return;
    rowsSer.push({label: `${r.short} · ${name}`, color: C('A'), marks: [{...o, kind,
      tip: `<b>${esc(r.label)}</b>, minus A1<br>${name}, ${SPLIT}: ${dvs(o)}<br>${o.beyond ? 'beyond seed noise (R17)' : 'within seed noise (R17)'}`}]});
  }));
  const serBox = document.getElementById('ch-series');
  if (rowsSer.length) forest(serBox, rowsSer, {zero: true, signed: true, aria: 'Continuity series minus A1'});
  else serBox.innerHTML = '<p class="small">No continuity-series run is scored yet.</p>';
  const verdict = o => !o ? '–' : (o.beyond ? '<b>measured effect</b>' : 'within seed noise');
  table(document.getElementById('tb-series'), ['Configuration (minus A1, each at its navdev choices)', 'Single path', 'Best-of-6', 'R17 on this split, single path'],
    ser.map(r => [esc(r.label), r[SPLIT + '_single'] ? dvs(r[SPLIT + '_single']) : '–', r[SPLIT + '_best_of_6'] ? dvs(r[SPLIT + '_best_of_6']) : '–',
      verdict(r[SPLIT + '_single'])]));
  const p4box = document.getElementById('ph4'), p4 = (D.phase4 || {})[SPLIT] || {};
  p4box.innerHTML = Object.keys(p4).length ? '' : '<p class="small">No method has all its phase-4 RL seeds chosen and scored yet.</p>';
  const meas = [['single', 'single path', 'solid'], ['best_of_6', 'best-of-6', 'hollow']];
  Object.keys(p4).forEach(rec => {
    const v = p4[rec], h = document.createElement('h3'); h.textContent = `${rec}: each method after RL, at the checkpoint R11 chose`; p4box.appendChild(h);
    const tb = document.createElement('div'); p4box.appendChild(tb);
    const seedCell = (o, k) => !o[k] ? '–' : o[k].seeds.map(x => x.toFixed(2)).join(' · ') +
      ` <span class="small">(mean ${o[k].mean.toFixed(2)}${o[k].sd != null ? ', SD ' + o[k].sd.toFixed(2) : ''})</span>`;
    table(tb, ['Method', 'Single path, by seed', 'Best-of-6, by seed', 'Chosen checkpoints'],
      v.methods.map(o => [o.method, seedCell(o, 'single'), seedCell(o, 'best_of_6'), esc(((o.single || o.best_of_6 || {}).checkpoints || []).join(', '))]));
    const h2 = document.createElement('h3'); h2.textContent = `${rec}: change from phase 3 (after RL minus before, each RL seed against its own supervised seed)`; p4box.appendChild(h2);
    const c1 = document.createElement('div'); c1.className = 'chart'; p4box.appendChild(c1);
    const rowsC = [];
    v.change.forEach(o => meas.forEach(([k, name, kind]) => { if (o[k]) rowsC.push({label: `${o.method} · ${name}`, color: C(o.method),
      marks: [{...o[k], kind, tip: `<b>${o.method}, ${rec}</b>: ${name} after minus before, ${SPLIT}<br>${dvs(o[k])} (${o[k].seeds} seed${o[k].seeds > 1 ? 's' : ''})`}]}); }));
    if (rowsC.length) forest(c1, rowsC, {zero: true, signed: true, aria: `${rec}: change from phase 3`});
    const h3 = document.createElement('h3'); h3.textContent = `${rec}: method pairs after RL, seed-averaged, drives and seeds resampled`; p4box.appendChild(h3);
    const c2 = document.createElement('div'); c2.className = 'chart'; p4box.appendChild(c2);
    const rowsP4 = [];
    v.pairs.forEach(o => meas.forEach(([k, name, kind]) => { if (o[k]) rowsP4.push({label: `${o.pair.replace('-', ' − ')} · ${name}`, color: 'var(--ink-2)',
      marks: [{...o[k], kind, tip: `<b>${esc(o.pair.replace('-', ' − '))}, ${rec}</b><br>${name}, ${SPLIT}: ${dvs(o[k])}`}]}); }));
    if (rowsP4.length) forest(c2, rowsP4, {zero: true, signed: true, aria: `${rec}: method pairs after RL`});
  });
  const sb = (D.seriesBest || {})[SPLIT];
  document.getElementById('tb-seriesbest').innerHTML = sb ? '' : '<p class="small">Only if the series\' best is not A1: its seeds 1 and 2, then this comparison.</p>';
  if (sb) table(document.getElementById('tb-seriesbest'), ['Series\' best minus A1, three seeds each', 'Seeds (best; A1)', 'Difference'],
    ['single', 'best_of_6'].filter(k => sb[k]).map(k => [`${esc(sb.best.toUpperCase())} − A1 · ${k === 'single' ? 'single path' : 'best-of-6'}`,
      `${sb[k].best_seeds.join(' · ')}; ${sb[k].a1_seeds.join(' · ')}`, dvs({x: sb[k].difference, lo: sb[k].ci[0], hi: sb[k].ci[1]})]));
  document.querySelectorAll('.splitname').forEach(e => e.textContent = SPLIT);
}
function drawRL() {
  const box = document.getElementById('ch-rl'); box.innerHTML = '';
  const groups = {}; D.rl.forEach(r => (groups[r.recipe || '?'] = groups[r.recipe || '?'] || []).push(r));
  Object.keys(groups).sort().forEach(g => {
    const wrap = document.createElement('div'); box.appendChild(wrap);
    const s0 = groups[g][0].settings || {};
    const floor = Math.max(...groups[g].map(r => (r.settings || {}).lp_sd_floor || 0));
    const opts = [s0.adv === 'dr' ? 'advantage: reward − group mean (Dr. GRPO)' : 'advantage: (reward − mean) / SD',
      s0.bc_balance ? 'BC term at the RL term\'s gradient norm' : '', floor ? `SD floor ${floor} in the log-probability (D)` : ''].filter(Boolean).join('; ');
    wrap.innerHTML = `<h3>${g === 'trial' ? 'Trial (200 updates, 2,662 scenes)' : g}</h3><p class="small">lr ${s0.lr != null ? Number(s0.lr).toExponential().replace('e-', 'e−') : '?'}, KL β ${s0.kl_beta ?? 0}, ${s0.g} candidates × ${s0.scenes_per_update} scenes; ${opts}</p>` +
      (g === 'RL7' ? '<p class="small">For A, RL7 is RL0: A has no Gaussian steps, so RL0\'s run stands in (R13).</p>' : '');
    const c = document.createElement('div'); c.className = 'chart'; wrap.appendChild(c);
    const rows = [];
    groups[g].forEach(r => r.checkpoints.forEach(cp => {
      const s = cp.single && cp.single.vsStart, b = cp.b6 && cp.b6.vsStart;
      if (cp.u === 0 || (!s && !b)) return;
      rows.push({label: `${r.method}${r.seed ? ' seed ' + r.seed : ''} · ${cp.u} updates`, color: C(r.method), marks: [
        {...(s || {}), kind: 'solid', tip: s ? `<b>${esc(r.run)}</b>, update ${cp.u}<br>single path ${f1(cp.single.x)}, change from start ${dvs(s)}` : ''},
        {...(b || {}), kind: 'hollow', tip: b ? `<b>${esc(r.run)}</b>, update ${cp.u}<br>best-of-6 ${f1(cp.b6.x)}, change from start ${dvs(b)}` : ''}]});
    }));
    if (!rows.length) { c.innerHTML = '<p class="small">Running or not scored on navdev yet: ' + groups[g].map(r => `${esc(r.run)} (${r.done || 0} updates)`).join(', ') + '.</p>'; return; }
    forest(c, rows, {zero: true, signed: true, aria: 'RL change from the start on navdev'});
  });
  const body = [];
  D.rl.forEach(r => r.checkpoints.forEach(cp => {
    const sub = cp.single && cp.single.subs || {};
    body.push([esc(r.run), r.recipe || '–', cp.u, cp.single ? f1(cp.single.x) : '–', cp.single ? dvs(cp.single.vsStart) : '–', cp.b6 ? f1(cp.b6.x) : '–', cp.b6 ? dvs(cp.b6.vsStart) : '–',
      sub.no_at_fault_collisions != null ? sub.no_at_fault_collisions.toFixed(3) : '–', sub.drivable_area_compliance != null ? sub.drivable_area_compliance.toFixed(3) : '–',
      sub.time_to_collision_within_bound != null ? sub.time_to_collision_within_bound.toFixed(3) : '–', sub.ego_progress != null ? sub.ego_progress.toFixed(3) : '–', cp.b6 && cp.b6.div ? cp.b6.div.endpoint_m : '–']);
  }));
  table(document.getElementById('tb-rl'), ['RL run', 'Recipe', 'Update', 'Single path', 'Change', 'Best-of-6', 'Change', 'No collision', 'Drivable area', 'TTC', 'Progress', 'Diversity (m)'], body);
  const cb = [];
  D.rl.forEach(r => (r.curve || []).forEach(w => cb.push([esc(r.run), `${w.updates[0]}–${w.updates[1]}`, w.reward?.toFixed(3) ?? '–', w.best_of_g?.toFixed(3) ?? '–', w.equal_reward_groups != null ? (100 * w.equal_reward_groups).toFixed(0) + '%' : '–', w.kl?.toFixed(4) ?? '–', w.grad_norm?.toFixed(1) ?? '–', w.bc?.toFixed(4) ?? '–', w.paths_per_s?.toFixed(0) ?? '–'])));
  table(document.getElementById('tb-curve'), ['RL run', 'Updates', 'Reward', 'Best of the group', 'Equal-reward groups', 'KL', 'Gradient norm', 'BC loss', 'Paths/s'], cb);
}
function tiles() {
  const t = document.getElementById('tiles'), out = [];
  const best = (k) => D.runs.filter(r => r['navtest_' + k]).sort((a, b) => b['navtest_' + k].x - a['navtest_' + k].x)[0];
  const h1 = D.refs.h1_text_navtest;
  const bs = best('single'), bb = best('best_of_6');
  if (h1) out.push(['H1, text output', f1(h1.x), 'navtest PDMS, the reference']);
  if (bs) out.push(['Best single path', f1(bs.navtest_single.x), `${esc(bs.run)}, navtest; vs H1 ${dvs(bs.navtest_single.vsH1)}`]);
  if (bb) out.push(['Best best-of-6', f1(bb.navtest_best_of_6.x), `${esc(bb.run)}, navtest (oracle pick)`]);
  const rr = D.methods.rl_recipe;
  const nRec = D.rl.filter(r => r.recipe && r.recipe.startsWith('RL')).length;
  if (rr && rr.every_recipe_lowers_both_methods)
    out.push(['RL recipe (R12, R13)', 'none helps', `best: RL${rr.recipe}, mean navdev change ${sgn(100 * rr.mean_gain_navdev_single[rr.recipe])}; phase 4 waits for the researcher`]);
  else
    out.push(['RL recipe (R12, R13)', rr ? 'RL' + rr.recipe : 'pending', rr ? `mean navdev gain ${sgn(100 * rr.mean_gain_navdev_single[rr.recipe])}`
      + Object.entries(rr.gain_by_method[rr.recipe] || {}).map(([m, g]) => `; ${m} ${sgn(100 * g)}`).join('') : `${nRec} of 15 recipe runs started`]);
  t.innerHTML = out.map(([l, v, n]) => `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="note">${n}</div></div>`).join('');
}
function notes() {
  for (const k of ['done', 'meaning', 'problems', 'next', 'decisions', 'queue']) {
    const e = document.getElementById('n-' + k); if (e) e.innerHTML = (NOTES[k] || []).map(x => `<li>${x}</li>`).join('');
  }
  document.getElementById('headline').innerHTML = NOTES.headline || '';
  document.getElementById('lede').innerHTML = NOTES.lede || '';
  document.getElementById('asof').textContent = NOTES.asof || D.generated || '';
}
document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
  SPLIT = b.dataset.split; document.querySelectorAll('.seg button').forEach(x => x.setAttribute('aria-pressed', x.dataset.split === SPLIT)); drawPhase1();
}));
function drawSel() {
  const box = document.getElementById('sel'); box.innerHTML = '';
  const chains = D.selectors || [];
  if (!chains.length) { box.innerHTML = '<p class="small">No selector is trained yet.</p>'; return; }
  const f2 = x => x == null ? '–' : x.toFixed(2);
  chains.forEach(ch => {
    const h = document.createElement('h3');
    h.textContent = ch.chain === 'rehearsal' ? 'Rehearsal: A1 without RL (no contender; used for no choice)' : `Contender ${ch.chain.toUpperCase()}`;
    box.appendChild(h);
    if (ch.adoption) { const p = document.createElement('p'); p.className = 'small';
      p.innerHTML = `Adoption on navdev (R14, the cross-fitted seed-0 selector): ${ch.adoption.adopted ? '<b>adopted</b>' : 'not adopted'}; a gain is ${ch.adoption.gain_claimed ? 'claimed (its drive interval excludes 0)' : 'not claimed'}.`;
      box.appendChild(p); }
    const c = document.createElement('div'); c.className = 'chart'; box.appendChild(c);
    const rows = ch.variants.filter(v => v.navdev).map(v => ({label: v.variant, color: 'var(--ink-2)', marks: [{...v.navdev.gain, kind: 'solid',
      tip: `<b>${esc(v.name)}</b><br>selected minus own pick, navdev: ${dvs(v.navdev.gain)}`}]}));
    if (rows.length) forest(c, rows, {zero: true, signed: true, aria: 'Selected path minus own pick, navdev'});
    const tb = document.createElement('div'); box.appendChild(tb);
    table(tb, ['Variant', 'Selected · own pick · best', 'Gap closed', 'Switches (wins / losses)', 'Fail NC or DAC (selected / own)', 'Held-out epoch, margin · training'],
      ch.variants.map(v => { const n = v.navdev || {};
        return [esc(v.variant), `${f2(n.selected)} · ${f2(n.own)} · ${f2(n.best)}`, n.share == null ? '–' : (100 * n.share).toFixed(0) + '%',
          n.switches ? `${(100 * n.switch).toFixed(1)}% (${n.switches.wins} / ${n.switches.losses})` : '–',
          n.fail ? `${n.fail.selected} / ${n.fail.own_pick}` : '–',
          `${v.heldout.epoch ?? '–'}, ${v.heldout.delta ?? '–'} · ${v.train && v.train.elapsed_s ? Math.round(v.train.elapsed_s / 60) + ' min' : '–'}`]; }));
  });
}
function drawAll() { drawPhase1(); drawRL(); drawSel(); }
notes(); tiles(); drawAll();
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAll, 120); });
"""

BODY = """
<main>
<header>
  <div class="kicker">Path-generator study · progress report · <span id="asof"></span></div>
  <h1 id="headline"></h1>
  <p class="lede" id="lede"></p>
  <div class="tiles" id="tiles"></div>
</header>
<section><h2>Since the plan was approved</h2><ul class="points" id="n-done"></ul></section>
<section>
  <h2>Phase 1: every training run at its navdev choices</h2>
  <p class="sub">Each row is one training run. Its epoch, generation steps and sampling setting were chosen on navdev (R10).
  navtest is for reporting only, and no choice uses it. Whiskers are 95% intervals that resample drives.</p>
  <div class="controls">Split <span class="seg"><button data-split="navtest" aria-pressed="true">navtest (report)</button><button data-split="navdev" aria-pressed="false">navdev (choices)</button></span></div>
  <div class="legend"><span><i class="key solid"></i>Single path (the method's own best guess)</span><span><i class="key hollow"></i>Best-of-6 (the best of 6 samples, an oracle pick)</span><span><i class="key line"></i>H1's text output</span></div>
  <div class="chart" id="ch-phase1"></div>
  <details><summary>Table: <span class="splitname"></span> scores, differences to H1 and diversity</summary><div id="tb-phase1"></div></details>
  <h3>Layer test: layers 12/18/24 minus layer 24, paired on the same scenes</h3>
  <div class="chart" id="ch-layer"></div>
  <details><summary>Table: layer test on <span class="splitname"></span></summary><div id="tb-layer"></div></details>
  <h3>Method pairs, each at its layer choice, paired on the same scenes</h3>
  <div class="chart" id="ch-pairs"></div>
  <details><summary>Table: method pairs on <span class="splitname"></span></summary><div id="tb-pairs"></div></details>
  <h3>Phase 3: three seeds of each method's chosen configuration, at seed 0's choices</h3>
  <p class="sub">Seeds 1 and 2 reuse seed 0's epoch, steps and sampling setting (R10). The spread between seeds is training noise.</p>
  <div id="tb-seeds"></div>
  <h3>Method pairs with every seed: seed-averaged means, resampling drives and seeds</h3>
  <p class="sub">Each mark is the difference of two methods' seed-averaged means. Its 95% interval redraws drives (with all their scenes)
  and each method's seeds, as R5's decision does. A method with one seed gets the larger between-seed SD of the three-seed methods as an
  added draw. These intervals include training noise, which the paired intervals above leave out.</p>
  <div class="chart" id="ch-seedpairs"></div>
  <details><summary>Table: method pairs with every seed on <span class="splitname"></span></summary><div id="tb-seedpairs"></div></details>
</section>
<section>
  <h2>Phase 2: how continuous, A's continuity series against A1</h2>
  <p class="sub">Each configuration adds one feature that makes A's path tokens behave more like continuous numbers. Each mark is a
  training run minus A1, both at their own navdev choices, paired on the same scenes. R17 calls a difference a measured effect only if
  it exceeds 1.4 points (about two seed SDs of a one-seed difference) and its drive interval excludes 0, on navtest's single path.
  Smaller differences are within seed noise.</p>
  <div class="chart" id="ch-series"></div>
  <details><summary>Table: continuity series minus A1 on <span class="splitname"></span></summary><div id="tb-series"></div></details>
  <h3>The series' best against A1, three seeds each (R17)</h3>
  <div id="tb-seriesbest"></div>
</section>
<section>
  <h2>Phase 4: every method before and after a matched RL stage</h2>
  <p class="sub">RL4 and RL6 run 1,500 updates on every method with the same seeds (A and D: three; B and C: one). Each RL seed's
  checkpoint is chosen on navdev by R11's rule, and update 0 (no RL) is allowed. "Before" is phase 3, at R10's choices. Each RL seed is
  paired with the supervised seed it started from, so shared seed noise cancels. With one RL seed, B's and C's intervals hold drive noise
  only. R5's ranking after RL is reported, not deciding (R16).</p>
  <div id="ph4"></div>
</section>
<section>
  <h2>RL: change from the starting point on navdev</h2>
  <p class="sub">Each RL run starts from its method's R11 starting epoch. The marks show the paired change in navdev PDMS from that start,
  after the given number of updates. R12 and R13 pick the recipe with the best mean single-path gain over A and D; if every
  recipe lowers both methods, phase 4 waits for the researcher.</p>
  <div class="legend"><span><i class="key solid"></i>Single path</span><span><i class="key hollow"></i>Best-of-6</span><span><i class="key" style="background:var(--mA)"></i>A</span><span><i class="key" style="background:var(--mD)"></i>D</span></div>
  <div class="mult" id="ch-rl"></div>
  <details><summary>Table: every RL checkpoint on navdev, with sub-scores</summary><div id="tb-rl"></div></details>
  <details><summary>Table: RL training curves on the training scenes (blocks of 50 updates)</summary><div id="tb-curve"></div></details>
</section>
<section>
  <h2>Phase 6: the selector, per chain (navdev)</h2>
  <p class="sub">A selector scores each of a path generator's candidates (its own pick and 6 samples) and keeps the own pick unless
  another is predicted better by a margin set on held-out training drives (R14). Each mark is the selected path minus the own pick on
  navdev, with its drive interval. The cross-fitted seed-0 selector decides adoption; the other variants are its second seed, the
  in-fold control and the ablations. navtest comes once per contender, after the final system is chosen (R16).</p>
  <div id="sel"></div>
</section>
<section><h2>What this means</h2><ul class="points" id="n-meaning"></ul></section>
<section><h2>Problems and fixes</h2><ul class="points" id="n-problems"></ul></section>
<section><h2>Next</h2><ul class="points" id="n-next"></ul></section>
<section><h2>For the researcher</h2><ul class="points" id="n-decisions"></ul></section>
<section><h2>Queue</h2><ul class="points" id="n-queue"></ul>
  <p class="small">Numbers come from the aggregate file generated <span id="gen"></span> on the 4090 desktop (pilot_sft/heads/collect_results.py).</p></section>
</main>
<div id="tip" role="tooltip"></div>
"""


def build(agg: dict, notes: dict, ms: Optional[dict] = None) -> str:
    data = json.dumps(page_data(agg, ms)).replace("</", "<\\/")
    nj = json.dumps(notes).replace("</", "<\\/")
    title = html.escape(notes.get("title", "Path-generator progress"))
    body = BODY.replace('<span id="gen"></span>', html.escape(str(agg.get("generated", ""))))
    return (f"<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{title}</title>\n"
            f"<style>{CSS}</style>\n</head>\n<body>{body}\n<script type=\"application/json\" id=\"data\">{data}</script>\n"
            f"<script type=\"application/json\" id=\"notes\">{nj}</script>\n<script>{JS}</script>\n</body>\n</html>\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--aggregate", required=True)
    p.add_argument("--notes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method-seeds", default=None, help="method_seeds.py's output: method pairs with every seed")
    a = p.parse_args()
    agg = json.load(open(a.aggregate, encoding="utf-8"))
    notes = json.load(open(a.notes, encoding="utf-8"))
    ms = json.load(open(a.method_seeds, encoding="utf-8")) if a.method_seeds else None
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(agg, notes, ms))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
