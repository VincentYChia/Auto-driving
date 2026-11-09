"""The images of a ``pilot_v1`` view, from ``<view>/images.zip`` when the view is packed, else the loose files.

``data_pipeline/v1/pilot_view.py build --storage zip`` writes one uncompressed archive per view whose member
names are the jsonl ``"image"`` field (``images/<log>/<token>.jpg``), with the JPEG bytes the loose layout
would hold; ``--storage loose`` (the default) writes those files under the view directory. Readers use
``ViewImages(view).read(item["image"])`` and get the same bytes either way:

    images = ViewImages("data/pretrain/v1.0-xl/views/pilot_v1/v1-xl")
    picture = images.open(item["image"]).convert("RGB")

Stdlib and Pillow only (Pillow only for ``open``), so it imports nothing from ``data_pipeline`` or
``navsim_eval`` and runs anywhere the view is copied. The archive is opened on the first read and again in
each new process (keyed by PID), so a forked data-loader worker never shares the parent's file position.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Optional

ZIP_NAME = "images.zip"  # pilot_view.ZIP_NAME


class ViewImages:
    def __init__(self, view_dir: str):
        self.view_dir = view_dir
        self.zip_path = os.path.join(view_dir, ZIP_NAME)
        self.packed = os.path.isfile(self.zip_path)  # decided once: a view is either packed or loose
        self._zip: Optional[zipfile.ZipFile] = None
        self._pid: Optional[int] = None

    def _archive(self) -> zipfile.ZipFile:
        if self._zip is None or self._pid != os.getpid():
            self._zip, self._pid = zipfile.ZipFile(self.zip_path), os.getpid()  # a fork drops the parent's handle
        return self._zip

    def read(self, rel: str) -> bytes:
        """The bytes of image ``rel`` (the jsonl ``"image"`` field, ``/``-separated, relative to the view)."""
        if self.packed:
            try:
                return self._archive().read(rel)
            except KeyError:
                raise FileNotFoundError(f"{rel} is not in {self.zip_path}") from None
        with open(os.path.join(self.view_dir, *rel.split("/")), "rb") as f:
            return f.read()

    def open(self, rel: str):
        """Image ``rel`` as a ``PIL.Image`` (decoded from memory, so no file stays open)."""
        from PIL import Image

        return Image.open(io.BytesIO(self.read(rel)))

    def close(self) -> None:
        if self._zip is not None and self._pid == os.getpid():
            self._zip.close()
        self._zip = self._pid = None

    def __getstate__(self):  # picklable for spawned workers: they re-open the archive themselves
        return {**self.__dict__, "_zip": None, "_pid": None}
