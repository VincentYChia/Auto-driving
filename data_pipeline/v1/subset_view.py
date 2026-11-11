"""A derived set's pilot view cut from its parent's built view [proposed 2026-11-11].

A derived set (``derived_sets.DERIVED``) lies inside its parent, so its view is the parent view's rows whose
token is in the set's token files, kept byte-for-byte and in the parent's order, and the parent's ``images.zip``
(linked, not copied: it holds every image the subset needs, and the readers look images up by name). Rows
identical to the parent's mean a run on the subset and a run on the parent differ only in which samples they see.
Standard library only, so it runs wherever the views are (Anvil, the lab desktop).

    python -m data_pipeline.v1.subset_view --parent .../pilot_v1/v1-full-thin-p12 \\
        --tokens data/pretrain/v1.0-full/splits/v1-hard --out .../pilot_v1/v1-hard-p12

``--tokens`` is the prefix of ``<prefix>_tokens_{train,dev}.txt``. Every listed token must be a row of the parent
view's split, or nothing is written. Writes ``train.jsonl``, ``dev.jsonl``, ``view.json`` (the parent's, with the
set, counts and the token files' SHA-256) and the ``images.zip`` link.
"""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

ZIP_NAME = "images.zip"  # pilot_view.ZIP_NAME


def _tokens(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_token(line: str) -> str:
    """A view row's token: the last part of its ``id`` (``openscene/<log>/<token>``)."""
    return json.loads(line)["id"].rsplit("/", 1)[-1]


def cut_split(parent_jsonl: str, tokens: Sequence[str], out_jsonl: str) -> int:
    """Write the parent's rows whose token is in ``tokens``, in the parent's order; returns the row count."""
    wanted = set(tokens)
    if len(wanted) != len(tokens):
        raise ValueError(f"{out_jsonl}: the token list repeats {len(tokens) - len(wanted)} tokens")
    found = set()
    tmp = out_jsonl + ".part"
    with open(parent_jsonl, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8", newline="") as dst:
        for line in src:
            if not line.strip():
                continue
            tok = row_token(line)
            if tok in wanted:
                if tok in found:
                    raise ValueError(f"{parent_jsonl} holds token {tok} twice")
                found.add(tok)
                dst.write(line if line.endswith("\n") else line + "\n")
    missing = wanted - found
    if missing:
        os.remove(tmp)
        raise ValueError(f"{len(missing)} tokens are not rows of {parent_jsonl}, e.g. {sorted(missing)[0]}")
    os.replace(tmp, out_jsonl)
    return len(found)


def build(parent: str, tokens_prefix: str, out: str, set_name: Optional[str] = None) -> Dict[str, Any]:
    """Cut the view (module docstring); returns its ``view.json``."""
    if os.path.abspath(parent) == os.path.abspath(out):
        raise ValueError("the subset view must be a new folder, not the parent's")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(parent, "view.json"), encoding="utf-8") as f:
        view = json.load(f)
    counts, sha = {}, {}
    for split in ("train", "dev"):
        path = f"{tokens_prefix}_tokens_{split}.txt"
        counts[split] = cut_split(os.path.join(parent, f"{split}.jsonl"), _tokens(path),
                                  os.path.join(out, f"{split}.jsonl"))
        sha[split] = _sha256(path)
    link = os.path.join(out, ZIP_NAME)
    target = os.path.relpath(os.path.join(parent, ZIP_NAME), out)
    if os.path.lexists(link):
        if os.readlink(link) != target:
            raise FileExistsError(f"{link} exists and points elsewhere")
    else:
        os.symlink(target, link)
    name = set_name or os.path.basename(tokens_prefix)
    view.update({"set": name, "samples": counts,
                 "subset_of": {"view": os.path.abspath(parent), "set": view.get("set"), "samples": view.get("samples"),
                               "images_zip": "linked from the parent view (it holds every image of the subset)"},
                 "token_files_sha256": sha,
                 "source": f"rows of the parent view whose token is in {name}'s token files (subset_view.py)"})
    with open(os.path.join(out, "view.json"), "w", encoding="utf-8") as f:
        json.dump(view, f, indent=1)
        f.write("\n")
    return view


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="A derived set's pilot view, cut from its parent's built view.")
    p.add_argument("--parent", required=True, help="the parent's view folder (train.jsonl, dev.jsonl, images.zip)")
    p.add_argument("--tokens", required=True, help="prefix of the set's <prefix>_tokens_{train,dev}.txt")
    p.add_argument("--out", required=True, help="the new view folder")
    p.add_argument("--set", help="the set's name (default: the token prefix's file name)")
    args = p.parse_args(argv)
    view = build(args.parent, args.tokens, args.out, args.set)
    print(json.dumps({"set": view["set"], "samples": view["samples"]}))


if __name__ == "__main__":
    main()
