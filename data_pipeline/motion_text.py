"""Object-motion context rendered as compact text (v0.1, format [proposed]).

This is the first, text-only form of claim C1's motion tokens: it lets the pilot and reasoner
consume tracks without new embedding layers, and it makes token budgets measurable. Learned
geometry-aware motion embeddings can replace it later without changing the canonical record.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

DYNAMIC_CATEGORIES = ("vehicle", "pedestrian", "bicycle")
STATIC_CATEGORIES = ("traffic_cone", "barrier", "czone_sign", "generic_object")


def summarize_tracks(
    agents: Dict[str, Any],
    max_objects: int = 12,
    categories: Sequence[str] = DYNAMIC_CATEGORIES,
    max_range_m: float = 60.0,
) -> List[Dict[str, Any]]:
    """Pick the nearest objects visible at the current frame and summarise their recent motion."""
    offsets = agents["frame_offsets_s"]
    rows = []
    for track in agents["tracks"]:
        if track["category"] not in categories:
            continue
        current = track["states"][-1]
        if current is None:
            continue
        x, y, _, yaw, vx, vy = current
        dist = math.hypot(x, y)
        if dist > max_range_m:
            continue
        oldest_slot = next(i for i, s in enumerate(track["states"]) if s is not None)
        ox, oy = track["states"][oldest_slot][:2]
        speed = None if vx is None or vy is None else math.hypot(vx, vy)
        rows.append(
            {
                "track_token": track["track_token"],
                "category": track["category"],
                "x": x,
                "y": y,
                "yaw_deg": math.degrees(yaw),
                "speed": speed,
                "moved_x": x - ox,
                "moved_y": y - oy,
                "seen_for_s": round(offsets[-1] - offsets[oldest_slot], 2),
                "distance": dist,
            }
        )
    rows.sort(key=lambda r: r["distance"])
    return rows[:max_objects]


def render(rows: List[Dict[str, Any]], precision: int = 1) -> str:
    if not rows:
        return "none"
    lines = []
    for i, r in enumerate(rows, 1):
        speed = "unknown" if r["speed"] is None else f"{r['speed']:.{precision}f} m/s"
        moved = (
            f"moved ({r['moved_x']:+.{precision}f}, {r['moved_y']:+.{precision}f}) m over {r['seen_for_s']:.1f} s"
            if r["seen_for_s"] > 0
            else "first seen now"
        )
        lines.append(
            f"{i}. {r['category']} at ({r['x']:.{precision}f}, {r['y']:.{precision}f}) m, "
            f"heading {r['yaw_deg']:+.0f} deg, speed {speed}, {moved}"
        )
    return "\n".join(lines)


def motion_context(agents: Dict[str, Any], max_objects: int = 12, max_range_m: Optional[float] = 60.0) -> str:
    return render(summarize_tracks(agents, max_objects=max_objects, max_range_m=max_range_m or math.inf))
