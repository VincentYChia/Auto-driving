"""Provenance helpers for manifests: git commits, checksums, timestamps."""

from __future__ import annotations

import datetime
import hashlib
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
