"""Provenance helpers for manifests: git commits, checksums, timestamps, and writing a manifest whole."""

from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def git_commit(path: str) -> Optional[str]:
    out = _git(path, "rev-parse", "HEAD")
    return out.strip() if out else None


def git_dirty(path: str) -> Optional[bool]:
    out = _git(path, "status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out.strip())


def _git(path: str, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_text(path: str, text: str, newline: Optional[str] = None) -> None:
    """Write ``text`` to ``path`` whole or not at all: ``path.part``, flush, fsync, then ``os.replace``.

    For the files a later step takes as "done" (a fetch record, ``canonical.json``, ``view.json``,
    ``qa/summary.json``): a job stopped mid-write leaves only the ``.part``, never a truncated marker.
    ``newline`` is ``open``'s, so the bytes are those a plain ``open(path, "w")`` write would give.
    """
    part = path + ".part"
    with open(part, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
