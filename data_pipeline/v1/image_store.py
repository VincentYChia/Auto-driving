"""Where the original CAM_F0 JPEGs of a dataset live: loose files, or one pack per camera archive.

    from data_pipeline.v1.image_store import read_image
    data = read_image("data/pretrain/v1.0-xl", "images/CAM_F0/<log>/<token>.jpg", source_archive=70)

Step 3 (``openscene_images fetch``) keeps each selected JPEG byte for byte, either as the loose file
``<dataset>/<image_path>`` (``--storage loose``, the default and how v1-pilot and v1-large were built)
or as the member ``<image_path>`` of ``<dataset>/images/packs/<source_archive>.zip``
(``--storage pack`` [proposed format, v1-xl spec §4]; packing itself is agreed 2026-11-08 because a
loose v1-xl would need ~415K more files than Anvil's quota allows, D). The canonical record's
``image_path`` and ``source_archive`` (``canonical.py``) name the file either way, so the schema and
the meaning of ``image_path`` (the native JPEG's bytes) do not change.

Pack format: uncompressed (ZIP_STORED) zip; members sorted by name, each with the 1980-01-01 timestamp,
``create_system`` 3 and mode 0644, so the same frames give the same bytes on every OS; Zip64 only
where the zip format needs it (Python adds it on its own).
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from typing import Dict, Iterable, Optional, Sequence, Tuple

PACK_DIR = ("images", "packs")
DATE_TIME = (1980, 1, 1, 0, 0, 0)  # the earliest time a zip can store
CREATE_SYSTEM = 3  # Unix; Python's default is 0 on Windows and 3 elsewhere, which would change the bytes
EXTERNAL_ATTR = 0o100644 << 16  # regular file, rw-r--r--

_open: Dict[str, zipfile.ZipFile] = {}
_open_pid: Optional[int] = None


def pack_relpath(source_archive: int) -> str:
    """The pack of one camera archive, relative to the dataset directory, with "/" on every OS."""
    return "/".join(PACK_DIR) + f"/{int(source_archive)}.zip"


def pack_path(dataset_root: str, source_archive: int) -> str:
    return os.path.join(dataset_root, *PACK_DIR, f"{int(source_archive)}.zip")


def _pack(path: str) -> zipfile.ZipFile:
    """One open ZipFile per pack and process. A forked worker re-opens its own: a file descriptor
    inherited from the parent shares its read position, so two processes seeking it would race."""
    global _open_pid
    pid = os.getpid()
    if _open_pid != pid:
        _open.clear()  # the parent's handles: dropped here, never closed for the parent
        _open_pid = pid
    z = _open.get(path)
    if z is None:
        z = _open[path] = zipfile.ZipFile(path)
    return z


def read_image(dataset_root: str, image_path: str, source_archive: Optional[int]) -> bytes:
    """The bytes of one original: the loose file if it exists, else its member in the archive's pack.

    Raises FileNotFoundError (no loose file, no pack, or not in the pack; the message names the pack
    and the archive) or ``zipfile.BadZipFile`` (the member's CRC-32 does not match, i.e. damaged bytes).
    """
    loose = os.path.join(dataset_root, *str(image_path).split("/"))
    if os.path.exists(loose):
        with open(loose, "rb") as f:
            return f.read()
    if source_archive is None:
        raise FileNotFoundError(loose)
    path = os.path.abspath(pack_path(dataset_root, source_archive))
    if not os.path.exists(path):
        raise FileNotFoundError(f"{loose} (no loose file) and {path} (no pack)")
    try:
        return _pack(path).read(str(image_path))
    except KeyError:
        raise FileNotFoundError(f"{image_path} is not in {path} (archive {source_archive})") from None


def close_all() -> None:
    """Close this process's open packs (tests, or before a pack is rewritten)."""
    global _open_pid
    if _open_pid == os.getpid():
        for z in _open.values():
            z.close()
    _open.clear()
    _open_pid = None


def write_pack(path: str, members: Iterable[Tuple[str, bytes]]) -> Dict[str, object]:
    """Write ``members`` (name, bytes) to ``path`` as a pack: ``path.part``, fsync, read back and
    compared with ``members`` (``_check_part``), then ``os.replace``.

    Returns {"members", "bytes", "sha256"} of the finished file. The bytes depend only on the
    members, not on their order, the clock or the OS.
    """
    items = sorted(members, key=lambda item: item[0])
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: a member name appears twice")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    with open(part, "wb") as f:
        with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for name, data in items:
                info = zipfile.ZipInfo(name, date_time=DATE_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = CREATE_SYSTEM
                info.external_attr = EXTERNAL_ATTR
                z.writestr(info, data)
        f.flush()
        os.fsync(f.fileno())
    _check_part(part, items)
    digest = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    size = os.path.getsize(part)
    _forget(path)
    os.replace(part, path)
    _fsync_dir(os.path.dirname(path) or ".")
    return {"members": len(items), "bytes": size, "sha256": digest.hexdigest()}


def _check_part(part: str, items: Sequence[Tuple[str, bytes]]) -> None:
    """``part`` reads back as exactly ``items``, every CRC-32 included, before it may become the pack.

    A file damaged while it was written (e.g. a second job truncating the same ``.part``, or a write
    error) raises here, before the rename and the record, instead of passing the size and hash checks.
    """
    try:
        with zipfile.ZipFile(part) as z:
            if z.namelist() != [name for name, _ in items]:
                raise RuntimeError(f"{part}: the members read back differ from those written")
            for name, data in items:
                if z.read(name) != data:
                    raise RuntimeError(f"{part}: member {name} reads back different bytes")
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"{part} does not read back as written ({error}); was another job writing it?") from None


def _forget(path: str) -> None:
    """Drop this process's handle on a pack about to be replaced, so later reads see the new file."""
    z = _open.pop(os.path.abspath(path), None)
    if z is not None and _open_pid == os.getpid():
        z.close()


def _fsync_dir(directory: str) -> None:
    """Make the rename durable on POSIX; Windows cannot open a directory for fsync."""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
