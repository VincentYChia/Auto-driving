"""Load OpenScene / NAVSIM log pickles without running arbitrary code.

``pickle.load`` can call any importable function, so a tampered download could run code on load.
The logs need only plain containers and numpy arrays and scalars: OpenScene writes them with
``pickle.HIGHEST_PROTOCOL`` from lists, dicts, strings and numpy values
(``DriveEngine/process_data/create_openscene_metadata.py``). Any other global is refused with
``pickle.UnpicklingError`` naming it, so an unexpected type fails loudly instead of loading.
"""

from __future__ import annotations

import _compat_pickle
import importlib
import io
import pickle
from typing import Any, BinaryIO

# numpy 2 renamed numpy.core to numpy._core. Pickles from either version must load under either.
_NUMPY_CORE = {
    ("multiarray", "_reconstruct"),  # arrays, protocol < 5 or non-contiguous
    ("multiarray", "scalar"),  # numpy scalars, e.g. np.float64
    ("numeric", "_frombuffer"),  # contiguous arrays, protocol 5
}
_ALLOWED = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),  # bytes in protocol-2 pickles
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    ("builtins", "slice"),
    ("builtins", "range"),
}


def _numpy_core(submodule: str, name: str):
    for package in ("numpy._core", "numpy.core"):
        try:
            module = importlib.import_module(f"{package}.{submodule}")
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise pickle.UnpicklingError(f"this numpy has no {submodule}.{name}")


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Protocol 0-2 pickles use Python 2 names (e.g. __builtin__.set). pickle.Unpickler maps them
        # in the find_class this method replaces, so map them here too.
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        else:
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        for package in ("numpy._core.", "numpy.core."):
            if module.startswith(package) and (module[len(package):], name) in _NUMPY_CORE:
                return _numpy_core(module[len(package):], name)
        if (module, name) in _ALLOWED:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}: not on the allow-list in {__name__}")


def load(fileobj: BinaryIO) -> Any:
    return RestrictedUnpickler(fileobj).load()


def loads(data: bytes) -> Any:
    return load(io.BytesIO(data))
