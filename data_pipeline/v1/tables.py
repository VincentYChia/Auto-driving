"""Parquet in and out as dicts of numpy columns (``pyarrow``, already in NAVSIM's requirements).

Columns with extra dimensions, e.g. a (N, 21, 3) path, are stored as fixed-size lists, with their
shape in the field metadata so ``read`` restores them. Strings are object arrays of ``str``.
Writes go to a temporary file first, so an interrupted run never leaves a half-written table.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

Columns = Dict[str, np.ndarray]


def _to_arrow(values: np.ndarray):
    values = np.asarray(values)
    if values.dtype == object:
        return pa.array(values.tolist()), None
    if values.ndim == 1:
        return pa.array(values), None
    inner = values.shape[1:]
    flat = pa.array(np.ascontiguousarray(values).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, int(np.prod(inner))), list(inner)


def write(path: str, columns: Columns, metadata: Optional[Dict[str, Any]] = None) -> None:
    lengths = {len(v) for v in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"columns differ in length: { {k: len(v) for k, v in columns.items()} }")
    fields, arrays = [], []
    for name, values in columns.items():
        array, shape = _to_arrow(values)
        field_meta = {b"shape": json.dumps(shape).encode()} if shape else None
        fields.append(pa.field(name, array.type, metadata=field_meta))
        arrays.append(array)
    schema_meta = {b"meta": json.dumps(metadata).encode()} if metadata else None
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=schema_meta))
    tmp = f"{path}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def read(path: str, columns: Optional[Sequence[str]] = None) -> Columns:
    table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    out: Columns = {}
    for field in table.schema:
        column = table.column(field.name).combine_chunks()
        shape = (field.metadata or {}).get(b"shape")
        if shape:
            values = column.flatten().to_numpy(zero_copy_only=False)
            out[field.name] = values.reshape((len(column), *json.loads(shape)))
        else:
            out[field.name] = column.to_numpy(zero_copy_only=False)
    return out


def read_metadata(path: str) -> Dict[str, Any]:
    meta = pq.read_schema(path).metadata or {}
    return json.loads(meta[b"meta"]) if b"meta" in meta else {}
