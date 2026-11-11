"""A derived set's view cut from its parent view (``data_pipeline/v1/subset_view.py``)."""

import json
import os

import pytest

from data_pipeline.v1 import subset_view


def _parent(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    rows = {s: [{"id": f"openscene/log/{s}{i}", "prompt": f"p{i}"} for i in range(4)] for s in ("train", "dev")}
    for s, rs in rows.items():
        (parent / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (parent / "images.zip").write_bytes(b"zip")
    (parent / "view.json").write_text(json.dumps({"set": "v1-full-thin", "samples": {"train": 4, "dev": 4}}))
    return parent


def _tokens(tmp_path, train, dev):
    for split, toks in (("train", train), ("dev", dev)):
        (tmp_path / f"v1-hard_tokens_{split}.txt").write_text("".join(f"{t}\n" for t in toks))
    return str(tmp_path / "v1-hard")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_the_subset_keeps_the_parent_rows_in_order_and_links_the_images(tmp_path):
    parent = _parent(tmp_path)
    prefix = _tokens(tmp_path, ["train3", "train1"], ["dev0", "dev1", "dev2", "dev3"])
    out = tmp_path / "v1-hard-p12"
    view = subset_view.build(str(parent), prefix, str(out))
    lines = (out / "train.jsonl").read_text().splitlines()
    assert lines == (parent / "train.jsonl").read_text().splitlines()[1::2]  # parent order, bytes unchanged
    assert (out / "dev.jsonl").read_text() == (parent / "dev.jsonl").read_text()
    assert os.readlink(out / "images.zip") == os.path.join("..", "parent", "images.zip")
    assert view["set"] == "v1-hard" and view["samples"] == {"train": 2, "dev": 4}
    assert view["subset_of"]["set"] == "v1-full-thin"


def test_a_token_missing_from_the_parent_writes_nothing(tmp_path):
    parent = _parent(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="not rows"):
        subset_view.cut_split(str(parent / "train.jsonl"), ["train0", "train9"], str(out / "train.jsonl"))
    assert not (out / "train.jsonl").exists() and not (out / "train.jsonl.part").exists()
    with pytest.raises(ValueError, match="repeats"):
        subset_view.cut_split(str(parent / "train.jsonl"), ["train0", "train0"], str(out / "train.jsonl"))
