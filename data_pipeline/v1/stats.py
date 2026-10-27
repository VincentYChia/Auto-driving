"""Small, JSON-friendly statistics for manifests and the audit."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

INF = math.inf


def bin_counts(values, edges: Sequence[float]) -> List[int]:
    """Counts per bin ``[edges[k], edges[k+1])``. NaN and values outside the edges are not counted."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[~np.isnan(values)]
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right") - 1
    idx = idx[(idx >= 0) & (idx < len(edges) - 1)]
    return np.bincount(idx, minlength=len(edges) - 1).tolist()


def edges_json(edges: Sequence[float]) -> List[Any]:
    """JSON has no infinity: write it as the strings "-inf" / "inf"."""
    return [e if math.isfinite(e) else ("inf" if e > 0 else "-inf") for e in edges]


def histogram(values, edges: Sequence[float]) -> Dict[str, Any]:
    return {"edges": edges_json(edges), "counts": bin_counts(values, edges)}


def percentiles(values, qs: Sequence[float] = (1, 5, 50, 95, 99)) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{q:g}": None for q in qs}
    return {f"p{q:g}": round(float(np.percentile(values, q)), 4) for q in qs}


def value_counts(values: Iterable[Any]) -> Dict[str, int]:
    """Counts per value, most common first (ties by value)."""
    counts = Counter(str(v) for v in values)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
