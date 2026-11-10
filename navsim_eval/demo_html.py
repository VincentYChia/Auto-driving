"""The interactive page of render_demo.py (--html): one navtest scene at a time, pure parts only.

render_demo.py reads the scenes with NAVSIM and hands this module plain arrays; nothing here imports NAVSIM, so
the tests run anywhere. The page (demo_template.html, filled with one JSON payload) shows the front camera and a
bird's-eye view (BEV): NAVSIM's own map drawing as a background image, with the other agents, the ego box and
every plan drawn on top as SVG, and a 0-4 s time slider.

Frames and times follow NAVSIM: x forward, y left, heading counter-clockwise, all in the ego rear-axle frame at
the planning moment (t = 0). Each plan is drawn as NAVSIM receives it (8 poses, 0.5 s apart), but its ego box moves
along the motion NAVSIM scores: the PDM simulator's car tracking the plan, 41 poses 0.1 s apart (render_demo.py
runs NAVSIM's own simulator). Other agents replay their logged boxes (NAVSIM v1 is non-reactive) over NAVSIM's
observation frames, t = 0, 0.5, ..., 5 s; each frame's boxes come in that frame's own ego frame and are moved into
the t = 0 frame with the ego poses (:func:`relative_tracks`). The page interpolates every pose list with
:func:`pose_at`'s rule, which is the rule of the observation NAVSIM scores against.

BEV pixels: NAVSIM's configure_bev_ax shows y from +half_y (left edge) to -half_y (right edge) and x from
-half_x (bottom) to +half_x (top), so a point lands at column (half_y - y) / (2 half_y) * width and row
(half_x - x) / (2 half_x) * height (:func:`bev_pixel`); :func:`check_bev_axes` asserts that a rendered axes obeys it.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from navsim_eval.predictions import poses_from_waypoints
from navsim_tools import geometry

TEMPLATE = Path(__file__).with_name("demo_template.html")
PLACEHOLDER = "__DEMO_DATA_JSON__"
STEP = 0.5  # s between NAVSIM poses (NAVSIM_INTERVAL_LENGTH)
NUM_POSES = 8  # the 4 s plan
OBS_FRAMES = 11  # frames of NAVSIM's observation, t = 0, 0.5, ..., 5 s (metric_cache_processor: 5 s horizon)
MOTION_STEP = 0.1  # s between simulated ego poses (default_scoring_parameters.yaml proposal_sampling)
MARKERS = ["circle", "square", "diamond", "triangle", "triangle-down"]  # one waypoint shape per model
MAP_NAMES = {"us-nv-las-vegas-strip": "Las Vegas", "us-ma-boston": "Boston",
             "us-pa-pittsburgh-hazelwood": "Pittsburgh", "sg-one-north": "Singapore"}
TYPE_LABELS = {"vehicle": "vehicle", "pedestrian": "pedestrian", "bicycle": "bicycle", "traffic_cone": "traffic cone",
               "barrier": "barrier", "czone_sign": "construction-zone sign", "generic_object": "other object"}
SUBSCORES = (("pdms", "score"), ("nc", "no_at_fault_collisions"), ("dac", "drivable_area_compliance"),
             ("ttc", "time_to_collision_within_bound"), ("ep", "ego_progress"), ("c", "comfort"))


# ----------------------------------------------------------------------------------------------- geometry

def relative_tracks(ego_poses: Sequence, boxes: Sequence, names: Sequence, tokens: Sequence) -> List[Dict]:
    """Other agents' boxes over K frames, in the ego frame of the first frame (t = 0).

    ``ego_poses``: K global ``(x, y, heading)`` ego rear-axle poses, ``ego_poses[0]`` at t = 0. ``boxes``: K arrays
    of NAVSIM boxes ``(x, y, z, length, width, height, heading)``, each in its own frame's ego frame (as NAVSIM's
    gt_boxes_oriented_box reads them). ``names`` and ``tokens``: K sequences of type names and track tokens.
    Returns one dict per track in first-seen order: token, type, length and width (where first seen), and
    ``poses``, a (K, 3) array with NaN rows where the track is absent.
    """
    k_frames = len(ego_poses)
    origin = np.asarray(ego_poses[0], dtype=np.float64)
    tracks: Dict[str, Dict] = {}
    for k in range(k_frames):
        if len(tokens[k]) == 0:
            continue
        b = np.asarray(boxes[k], dtype=np.float64).reshape(len(tokens[k]), -1)
        rel = geometry.change_frame(b[:, [0, 1, 6]], np.asarray(ego_poses[k], dtype=np.float64), origin)
        for i, token in enumerate(tokens[k]):
            track = tracks.get(token)
            if track is None:
                track = tracks[token] = {"token": str(token), "type": str(names[k][i]), "length": float(b[i, 3]),
                                         "width": float(b[i, 4]), "poses": np.full((k_frames, 3), np.nan)}
            track["poses"][k] = rel[i]
    return list(tracks.values())


def crop_tracks(tracks: List[Dict], half_x: float, half_y: float, pad: float = 8.0) -> List[Dict]:
    """Tracks whose centre comes within ``pad`` metres of the view (|x| <= half_x, |y| <= half_y) at some frame."""
    keep = []
    for track in tracks:
        p = track["poses"]
        p = p[~np.isnan(p).any(axis=1)]
        if len(p) and np.any((np.abs(p[:, 0]) <= half_x + pad) & (np.abs(p[:, 1]) <= half_y + pad)):
            keep.append(track)
    return keep


def pose_at(keys, t: float, step: float = STEP) -> Optional[np.ndarray]:
    """The pose at time ``t`` from key poses at 0, step, 2 step, ... (NaN rows: absent); None where absent.

    This is the rule of the observation NAVSIM scores against (metric_cache_processor._interpolate_gt_observation
    with StateInterpolator):

    - a track present in one key only is held at that pose at every time (NAVSIM keeps it for the whole horizon);
    - otherwise it exists from its first present key to its last, both included, and is absent outside them;
    - in between, the position is linear and the heading turns along the shorter arc between the neighbouring
      present keys, across absent keys (NAVSIM interpolates over a missing frame).

    Plans and simulated motion have every key present, so for them this is plain linear interpolation. The page's
    poseAt (demo_template.html) implements the same rule; tests/eval/test_demo_html.py runs both on the same cases.
    """
    keys = np.asarray(keys, dtype=np.float64).reshape(-1, 3)
    present = [k for k in range(len(keys)) if not np.isnan(keys[k]).any()]
    if not present:
        return None
    if len(present) == 1:
        return keys[present[0]].copy()
    u = t / step
    if u < present[0] - 1e-9 or u > present[-1] + 1e-9:
        return None
    j = 0
    while j + 1 < len(present) and present[j + 1] <= u + 1e-9:
        j += 1
    ka = present[j]
    a = keys[ka]
    if abs(u - ka) < 1e-9 or j == len(present) - 1:
        return a.copy()
    kb = present[j + 1]
    b = keys[kb]
    f = (u - ka) / (kb - ka)
    dh = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
    h = a[2] + f * dh
    return np.array([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), math.atan2(math.sin(h), math.cos(h))])


def with_origin(poses) -> np.ndarray:
    """``(N, 3)`` future poses with the t = 0 pose (the origin of the ego frame) put in front: ``(N + 1, 3)``."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    return np.vstack([np.zeros((1, 3)), poses])


def model_plan(waypoints, num_poses: int = NUM_POSES) -> Optional[np.ndarray]:
    """A model's 4 s plan as NAVSIM scored it: the first 8 waypoints with travel-direction headings
    (PrecomputedTrajectoryAgent's poses_from_waypoints), after the origin. None when there are too few waypoints."""
    try:
        return with_origin(poses_from_waypoints(np.asarray(waypoints, dtype=np.float64), num_poses))
    except ValueError:
        return None


def bev_pixel(x, y, half_x: float, half_y: float, width: float, height: float):
    """Image column and row (continuous, from the top-left corner) of ego-frame point ``(x, y)`` in a BEV image
    covering |x| <= half_x (up) and |y| <= half_y (left), as NAVSIM's configure_bev_ax lays it out."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return (half_y - y) / (2 * half_y) * width, (half_x - x) / (2 * half_x) * height


def check_bev_axes(ax, half_x: float, half_y: float, width: int, height: int, tol: float = 0.5) -> float:
    """Assert that the matplotlib axes ``ax`` (filling a ``width`` x ``height`` px figure) maps data to pixels as
    :func:`bev_pixel` does; return the largest error in pixels. NAVSIM plots (y, x) as (horizontal, vertical)."""
    ax.figure.canvas.draw()
    xs = np.array([half_x, -half_x, 0.0, 0.37 * half_x])
    ys = np.array([half_y, -half_y, 0.0, -0.61 * half_y])
    disp = ax.transData.transform(np.stack([ys, xs], axis=1))  # display pixels, origin bottom-left
    col, row = bev_pixel(xs, ys, half_x, half_y, width, height)
    err = float(max(np.abs(disp[:, 0] - col).max(), np.abs((height - disp[:, 1]) - row).max()))
    if err > tol:
        raise AssertionError(f"BEV axes off by {err:.2f} px from the page's pixel mapping")
    return err


# ----------------------------------------------------------------------------------------------- images

def jpeg_data_uri(image: np.ndarray, width: int = 1024, quality: int = 80) -> str:
    """An RGB uint8 image as a JPEG data URI, scaled down to ``width`` pixels when wider."""
    from PIL import Image

    im = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def png_data_uri(png: bytes, colors: int = 0) -> str:
    """PNG bytes as a data URI; with ``colors`` > 0, first reduced to that many palette colours (flat map fills)."""
    if colors:
        from PIL import Image

        im = Image.open(io.BytesIO(png)).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                                                 dither=Image.Dither.NONE)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        png = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ----------------------------------------------------------------------------------------------- payload

def score_entry(row, zeroed_by: str = "") -> Dict:
    """PDMS and the sub-scores of one NAVSIM score-CSV row (a pandas Series or a dict), plus why it scored 0."""
    out = {key: round(float(row[col]), 3) for key, col in SUBSCORES if col in row}
    out["zeroed_by"] = zeroed_by or None
    return out


def why_text(why: str, reference: str, comparison: str) -> str:
    """render_demo.choose()'s label for a scene, spelled out with the model names."""
    texts = {
        "random sample": "Seeded random sample of the scenes that every model was scored on.",
        "reference model far ahead": f"{reference} scores 0.8 or more here and {comparison} scores 0.",
        "comparison model ahead": f"{comparison} scores more than 0.3 above {reference} here.",
        "reference model fails": f"{reference} scores 0 here.",
    }
    if why in texts:
        return texts[why]
    if why.endswith("-turn command"):
        return f"Picked for its route command: {why.split('-')[0]}."
    return why


def map_label(map_name: str) -> str:
    return MAP_NAMES.get(map_name, map_name)


def series_specs(model_names: Sequence[str], colors: Sequence[str], ink: str, grey: str,
                 with_cv: bool = True) -> List[Dict]:
    """Legend and drawing order for the page: human, the models (the first is the reference), constant velocity.

    Same colours and line styles as the PNG figures; each model also gets its own waypoint marker shape so
    identity never rests on colour alone. Colours follow the model, never its rank in a scene.
    """
    if len(model_names) > len(colors):
        raise ValueError(f"{len(model_names)} models but only {len(colors)} series colours")
    specs = [{"key": "human", "name": "Human driver (recorded)", "color": ink, "line": "dash", "width": 1.6,
              "marker": None}]
    for i, name in enumerate(model_names):
        specs.append({"key": f"m{i}", "name": name, "color": colors[i], "line": "solid",
                      "width": 3.0 if i == 0 else 2.2, "marker": MARKERS[i % len(MARKERS)]})
    if with_cv:
        specs.append({"key": "cv", "name": "Constant velocity", "color": grey, "line": "dot", "width": 1.8,
                      "marker": None})
    return specs


def pose_list(poses, decimals: int = 2) -> List[Optional[List[float]]]:
    """``(N, 3)`` poses as JSON-ready rows, ``None`` for absent (NaN) rows; heading kept to 3 decimals."""
    out: List[Optional[List[float]]] = []
    for p in np.asarray(poses, dtype=np.float64).reshape(-1, 3):
        out.append(None if np.isnan(p).any() else [round(float(p[0]), decimals), round(float(p[1]), decimals),
                                                    round(float(p[2]), 3)])
    return out


def scene_payload(token: str, why: str, why_long: str, command: str, map_name: str, speed: float, camera: str,
                  views: Dict[str, str], plans: Dict[str, Optional[np.ndarray]], scores: Dict[str, Optional[Dict]],
                  tracks: List[Dict], motion: Optional[Dict[str, Optional[np.ndarray]]] = None) -> Dict:
    """One scene for the page. ``plans``: key -> (9, 3) poses at t = 0..4 s or None; ``motion``: key -> the
    (41, 3) simulated ego poses at t = 0, 0.1, ..., 4 s that NAVSIM scored for that plan, or None (the page then
    moves the box along the plan itself); ``views``: view name -> image data URI; ``tracks``: relative_tracks()
    output over NAVSIM's observation frames (already cropped)."""
    motion = motion or {}
    return {
        "token": token, "why": why, "why_text": why_long, "command": command, "map": map_name,
        "map_label": map_label(map_name), "speed": round(float(speed), 2), "camera": camera, "views": views,
        "plans": {key: (pose_list(p) if p is not None else None) for key, p in plans.items()},
        "motion": {key: (pose_list(m) if m is not None else None) for key, m in motion.items()},
        "scores": scores,
        "agents": [{"type": t["type"], "l": round(t["length"], 2), "w": round(t["width"], 2),
                    "p": pose_list(t["poses"])} for t in tracks],
    }


def view_spec(half_x: float, half_y: float, label: str) -> Dict:
    """One BEV view size for the page, with a round scale-bar length (about a sixth of the width)."""
    return {"half_x": float(half_x), "half_y": float(half_y), "label": label,
            "scale_m": 10 if 2 * half_y <= 80 else 20}


def page_payload(scenes: List[Dict], series: List[Dict], views: Dict[str, Dict], ego: Dict,
                 agent_styles: Dict[str, Dict], agent_keys: List[Dict], map_keys: List[Dict], seed: int,
                 generated: str, default_view: Optional[str] = None) -> Dict:
    """Everything the page reads. ``ego``: length, width, rear_axle_to_center (metres), fill and line colours;
    ``agent_styles``: NAVSIM type name -> fill and line colour; ``views``: view name -> view_spec()."""
    return {"step": STEP, "motion_step": MOTION_STEP, "horizon": STEP * NUM_POSES, "series": series, "views": views,
            "default_view": default_view or next(iter(views)), "ego": ego, "agent_styles": agent_styles,
            "agent_keys": agent_keys, "map_keys": map_keys, "type_labels": TYPE_LABELS, "seed": seed,
            "generated": generated, "scenes": scenes}


def build_html(payload: Dict, template: Optional[str] = None) -> str:
    """The template with the payload embedded as JSON (no network requests: everything is inline).

    ``<``, ``>`` and ``&`` are written as JSON unicode escapes, so no string in the data (a model name, say) can
    close the script element it sits in.
    """
    if template is None:
        template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"template must contain {PLACEHOLDER} exactly once")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return template.replace(PLACEHOLDER, data)


def write_html(path: str, payload: Dict) -> int:
    """Write the page; returns its size in bytes."""
    html = build_html(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
    return len(html.encode("utf-8"))
