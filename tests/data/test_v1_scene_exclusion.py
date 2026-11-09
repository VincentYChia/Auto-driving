"""The warm-up exclusion [proposed 2026-11-09]: ``scene_exclusion.window_mask``, ``select(train_excluded=)`` and
``openscene_images.plan(exclude_scenes=)`` (PRETRAIN_V1_PLAN.md §10)."""

import hashlib
import json
import os
import pickle
import shutil

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import openscene_images, scene_exclusion, selection  # noqa: E402
from data_pipeline.v1.selection import SelectionConfig  # noqa: E402
from tests.data.test_v1_select_audit import fake  # noqa: E402
from tests.data.test_v1_xl_plan import copy_index, sha, small, tokens, write_map_and_tree  # noqa: E402,F401

S = 1_000_000


def rows_at(spec):
    """Candidate columns from (log, token, t_s, split)."""
    return {"log_name": np.array([s[0] for s in spec], dtype=object), "token": np.array([s[1] for s in spec], dtype=object),
            "t_us": np.array([int(round(s[2] * S)) for s in spec], dtype=np.int64),
            "our_split": np.array([s[3] for s in spec], dtype=object)}


# --- the mask ------------------------------------------------------------------------------------


def test_window_is_inclusive_at_window_s_and_stays_in_its_log():
    cands = rows_at([("a", "w", 100.0, "train"),  # the listed scene
                     ("a", "in_lo", 93.0, "train"), ("a", "in_hi", 107.0, "train"),  # exactly 7.0 s: excluded
                     ("a", "out_lo", 92.999999, "train"), ("a", "out_hi", 107.000001, "train"),  # 1 µs beyond
                     ("a", "near", 101.5, "dev"),
                     ("b", "other_log", 100.0, "train")])  # the same time in another log
    mask, report = scene_exclusion.window_mask(cands, ["w", "absent"], 7.0)
    assert dict(zip(cands["token"], mask.tolist())) == {"w": True, "in_lo": True, "in_hi": True, "out_lo": False,
                                                       "out_hi": False, "near": True, "other_log": False}
    assert report["tokens"] == 2 and report["tokens_in_index"] == 1 and report["not_located"] == ["absent"]
    assert report["rows_in_window_by_split"] == {"train": 3, "dev": 1}
    assert scene_exclusion.window_mask(cands, ["w"], 0.0)[0].tolist() == [True] + [False] * 6
    assert not scene_exclusion.window_mask(cands, [], 7.0)[0].any()
    with pytest.raises(ValueError):
        scene_exclusion.window_mask(cands, ["w"], -1.0)


def test_listed_frames_that_are_not_rows_are_placed_from_the_metadata(tmp_path):
    cands = rows_at([("a", "r0", 0.0, "train"), ("a", "r20", 20.0, "train"), ("a", "r40", 40.0, "train"),
                     ("b", "s0", 0.0, "train")])
    # "gap" is a frame of log a at 21 s that is not a candidate row; "w" is a listed row, so log a is scanned
    frames = {"a": [{"token": "gap", "log_name": "a", "timestamp": 21 * S}, {"token": "r40", "log_name": "a",
                                                                            "timestamp": 40 * S}],
              "b": [{"token": "hidden", "log_name": "b", "timestamp": 0}]}
    for log, fr in frames.items():
        with open(tmp_path / f"{log}.pkl", "wb") as f:
            pickle.dump(fr, f)
    located = scene_exclusion.locate_in_metadata(str(tmp_path), {"gap", "hidden", "nowhere"}, ["a"])
    assert located == {"gap": ("a", 21 * S)}  # log b is not scanned
    mask, report = scene_exclusion.window_mask(cands, ["gap", "r40", "nowhere"], 7.0, located)
    assert mask.tolist() == [False, True, True, False]
    assert report["located_in_metadata"] == {"gap": "a"} and report["not_located"] == ["nowhere"]
    # a located time is ignored for a token that is a row
    assert scene_exclusion.window_mask(cands, ["r0"], 1.0, {"r0": ("a", 40 * S)})[0].tolist() == [True, False, False,
                                                                                                    False]


def test_token_list_from_a_scene_filter(tmp_path):
    (tmp_path / "f.yaml").write_text('log_names:\n  - "L"\ntokens:\n  - "b2"\n  - "a1"\n  - "b2"\n')
    assert scene_exclusion.tokens_from_scene_filter(str(tmp_path / "f.yaml")) == ["a1", "b2"]
    scene_exclusion.main(["from-filter", str(tmp_path / "f.yaml"), "--out", str(tmp_path / "t.txt")])
    assert (tmp_path / "t.txt").read_bytes() == b"a1\nb2\n"


def test_committed_warmup_list_is_the_known_source():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pretrain", "v1.0-xl", "splits",
                        scene_exclusion.WARMUP_FILE)
    data = open(path, "rb").read()
    assert hashlib.sha256(data).hexdigest() == scene_exclusion.WARMUP_SHA256 and b"\r" not in data
    lines = data.decode().splitlines()
    assert len(lines) == scene_exclusion.WARMUP_SOURCE["tokens"] == 563 and lines == sorted(set(lines))
    assert scene_exclusion.source_of(path)["commit"] == "3e8291bfa89ff247231e0227778840cd0a036896"


# --- select(train_excluded=) ---------------------------------------------------------------------


def test_masked_rows_are_never_train_and_dev_is_untouched():
    cands = fake(200, ["train", "train", "dev"])
    cands["decision_lat"][::3] = "curve_left"
    cands["city"] = np.array(["c1", "c2"] * 300, dtype=object)
    config = SelectionConfig("t", n_train=40, n_dev=20, spacing_s=2.0, balance_alpha=0.5)
    plain, plain_report = selection.select(cands, config)
    none, _ = selection.select(cands, config, train_excluded=None)
    nothing, _ = selection.select(cands, config, train_excluded=np.zeros(600, dtype=bool))
    assert none.tolist() == plain.tolist() == nothing.tolist()  # the defaults change nothing
    mask = np.zeros(600, dtype=bool)
    mask[:100] = True  # the first 50 s of log0
    mask[400:600] = True  # all of dev
    rows, report = selection.select(cands, config, train_excluded=mask)
    split = cands["our_split"][rows]
    assert not (mask[rows] & (split == "train")).any() and (split == "train").sum() == 40
    assert rows[split == "dev"].tolist() == plain[cands["our_split"][plain] == "dev"].tolist()  # dev ignores it
    assert report["splits"]["train"]["train_excluded"] == {"candidates": 100, "eligible": 100, "after_thinning": 100,
                                                           "pool_left": 300}
    assert "train_excluded" not in report["splits"]["dev"] and "train_excluded" not in plain_report["splits"]["train"]
    # the same draw with masked rows skipped: unmasked rows of the plain draw in other logs mostly stay
    kept = set(plain[(cands["log_name"][plain] == "log1")].tolist())
    assert len(kept & set(rows.tolist())) >= len(kept) // 2


def test_a_masked_fixed_train_row_raises_and_a_masked_fixed_dev_row_does_not():
    cands = fake(100, ["train", "dev"])
    config = SelectionConfig("t", n_train=10, n_dev=5, spacing_s=2.0)
    mask = np.zeros(200, dtype=bool)
    mask[[0, 100]] = True
    with pytest.raises(RuntimeError, match="fixed train rows are in train_excluded"):
        selection.select(cands, config, fixed={"train": [0, 10], "dev": [100]}, train_excluded=mask)
    rows, _ = selection.select(cands, config, fixed={"train": [10], "dev": [100]}, train_excluded=mask)
    assert 100 in rows and 10 in rows and 0 not in rows
    with pytest.raises(ValueError, match="boolean mask"):
        selection.select(cands, config, train_excluded=np.zeros(5, dtype=bool))


# --- plan(exclude_scenes=), on the synthetic index -----------------------------------------------


def test_xl_plan_drops_base_rows_in_the_window_and_reports_them(small, tmp_path):
    large, xl, maps = small
    base_train = tokens(large / "splits" / "v1-large_tokens_train.txt")
    base_dev = tokens(large / "splits" / "v1-large_tokens_dev.txt")
    listed = tmp_path / "list" / "warm_tokens.txt"
    listed.parent.mkdir()
    listed.write_bytes("".join(f"{t}\n" for t in sorted([base_train[0], base_dev[0], "not_in_index"])).encode())
    manifest = openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(large / "splits" / "v1-large"),
                                     exclude_scenes=str(listed), exclude_window_s=7.0, allow_unlocated=True)
    s = xl / "splits"
    e = manifest["scene_exclusion"]
    xl_train, nd_train = set(tokens(s / "v1-xl_tokens_train.txt")), set(tokens(s / "v1-xl-nd_tokens_train.txt"))
    dropped = set(base_train) - xl_train
    assert dropped and base_train[0] in dropped and dropped == set(base_train) - nd_train
    assert e["base_train_dropped"] == len(dropped) and e["tokens_in_index"] == 2 and e["not_located"] == ["not_in_index"]
    assert e["dev_listed_tokens"] == [base_dev[0]] and e["sha256"] == sha(listed) and e["source"] is None
    assert e["metadata"] is None and e["unlocated_allowed"] is True
    assert e["train_pool_excluded"] >= len(dropped)
    assert (set(base_train) - dropped) <= nd_train <= xl_train and len(xl_train) == 30
    for split in ("train", "dev"):  # the base's own files are copied unchanged; dev is the base's in every set
        assert (s / f"v1-large_tokens_{split}.txt").read_bytes() == (large / "splits" / f"v1-large_tokens_{split}.txt").read_bytes()
    for name in manifest["selections"]:
        assert tokens(s / f"{name}_tokens_dev.txt") == base_dev
    assert manifest["frames_to_keep"] == 30 + 4 + len(dropped)
    assert manifest["selections"]["v1-large"]["train_not_in"] == {"v1-xl": len(dropped), "v1-xl-nd": len(dropped)}
    assert (s / "warm_tokens.txt").read_bytes() == listed.read_bytes()  # copied next to the token files
    # no train row of v1-xl is within the window
    import data_pipeline.v1.tables as tables
    cands = tables.read(str(xl / "index" / "candidates.parquet"), ["token", "log_name", "t_us", "our_split"])
    mask, _ = scene_exclusion.window_mask(cands, tokens(listed), 7.0)
    in_window = set(cands["token"][mask].tolist())
    assert not (xl_train & in_window) and dropped <= in_window


def test_a_list_with_tokens_not_in_the_index_needs_the_metadata(small, tmp_path):
    """Fail closed: a token that is not a row may be a frame next to train rows (review, 2026-11-09)."""
    large, xl, maps = small
    base_train = tokens(large / "splits" / "v1-large_tokens_train.txt")
    prefix = str(large / "splits" / "v1-large")
    listed = tmp_path / "list" / "warm_tokens.txt"
    listed.parent.mkdir()
    listed.write_bytes("".join(f"{t}\n" for t in sorted([base_train[0], "not_in_index"])).encode())
    with pytest.raises(ValueError, match="not index rows"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=prefix, exclude_scenes=str(listed))
    assert not (xl / "splits" / "v1-xl_tokens_train.txt").exists()  # nothing written
    with pytest.raises(ValueError, match="allow_unlocated needs exclude_scenes"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=prefix, allow_unlocated=True)
    # with the metadata (a folder of log pickles here), the plan runs and records the scan
    meta = tmp_path / "meta"
    meta.mkdir()
    import data_pipeline.v1.tables as tables
    cands = tables.read(str(xl / "index" / "candidates.parquet"), ["token", "log_name"])
    log = cands["log_name"][cands["token"].tolist().index(base_train[0])]
    with open(meta / f"{log}.pkl", "wb") as f:
        pickle.dump([{"token": base_train[0], "log_name": log, "timestamp": 0}], f)
    manifest = openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=prefix, exclude_scenes=str(listed),
                                     exclude_metadata=str(meta))
    e = manifest["scene_exclusion"]
    assert e["metadata"] == {"file": "meta", "logs_scanned": 1} and e["not_located"] == ["not_in_index"]
    assert "unlocated_allowed" not in e
    # a list whose tokens are all index rows needs no metadata
    listed.write_bytes(f"{base_train[0]}\n".encode())
    (xl / "splits" / "warm_tokens.txt").unlink()
    assert openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=prefix,
                                 exclude_scenes=str(listed))["scene_exclusion"]["tokens_not_in_index"] == 0


def test_exclusion_is_for_v1_xl_only_and_refuses_a_bad_list(small, tmp_path):
    large, xl, maps = small
    listed = tmp_path / "w.txt"
    listed.write_bytes(b"b\na\n")
    prefix = str(large / "splits" / "v1-large")
    with pytest.raises(ValueError, match="sorted"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=prefix, exclude_scenes=str(listed))
    listed.write_bytes(b"a\r\nb\r\n")
    with pytest.raises(ValueError, match="CRLF"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=prefix, exclude_scenes=str(listed))
    with pytest.raises(ValueError, match="v1-xl only"):
        openscene_images.plan(str(xl), *maps, set_name="v1-large", exclude_scenes=str(listed))
    with pytest.raises(SystemExit):
        openscene_images.main(["plan", "--out", str(xl), "--set", "v1-large", "--exclude-scenes", str(listed)])


# --- the real index (local only) -----------------------------------------------------------------

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
REAL = os.path.join(REPO, "data", "pretrain")
METADATA = os.path.join(REAL, "downloads", "openscene_metadata_trainval.tgz")
XL_TRAIN_SHA = "7f6eb97653c690c54a6a0e0a6101e8123efcea9a62f13989bec4c71fedec7643"  # V, 2026-11-09
ND_TRAIN_SHA = "5cb194088089ae1b997ef499cb58377e87fe778558ad5cd0e1cd9d9b455a0923"  # V, 2026-11-09
DEV_SHA = "79090629e278705a066890efbfc8271acd9204e2085c3f3ddd8d668316ab7205"


def test_real_xl_plan_with_the_warmup_exclusion(tmp_path):
    index = os.path.join(REAL, "v1.0-large", "index", "candidates.parquet")
    if not (os.path.exists(index) and os.path.exists(METADATA)):
        pytest.skip("the real candidate index or OpenScene's metadata tgz is not here")
    out = tmp_path / "xl"
    (out / "index").mkdir(parents=True)
    try:
        os.link(index, out / "index" / "candidates.parquet")
    except OSError:
        shutil.copyfile(index, out / "index" / "candidates.parquet")
    with open(os.path.join(REAL, "v1.0-large", "images", "archives.json"), encoding="utf-8") as f:
        archives = {int(n): a for n, a in json.load(f).items()}
    committed = os.path.join(REAL, "v1.0-xl", "splits")
    manifest = openscene_images.plan(str(out), *write_map_and_tree(tmp_path / "hf", archives), set_name="v1-xl",
                                     extends=os.path.join(REAL, "v1.0-large", "splits", "v1-large"),
                                     exclude_scenes=os.path.join(committed, scene_exclusion.WARMUP_FILE),
                                     exclude_metadata=METADATA)
    s = out / "splits"
    assert sha(s / "v1-xl_tokens_train.txt") == XL_TRAIN_SHA and len(tokens(s / "v1-xl_tokens_train.txt")) == 200_000
    assert sha(s / "v1-xl-nd_tokens_train.txt") == ND_TRAIN_SHA and len(tokens(s / "v1-xl-nd_tokens_train.txt")) == 176_352
    for name in ("v1-xl", "v1-xl-nd", "v1-large"):
        assert sha(s / f"{name}_tokens_dev.txt") == DEV_SHA
    for name in os.listdir(s):  # the committed plan is this plan (manifest aside: it holds times and paths)
        if name != "manifest.json":
            assert (s / name).read_bytes() == open(os.path.join(committed, name), "rb").read(), name
    e = manifest["scene_exclusion"]
    assert (e["tokens_in_index"], len(e["located_in_metadata"]), e["tokens_not_located"]) == (448, 6, 109)
    assert (e["base_train_dropped"], e["base_train_listed_tokens"], e["base_dev_in_window"]) == (1727, 57, 49)
    assert len(e["dev_listed_tokens"]) == 3 and e["train_pool_excluded"] == 8580
    assert manifest["frames_to_keep"] == 203_727
    assert manifest["selections"]["v1-xl"]["splits"]["train"]["stages"] == {"others_1": 55_421, "cruise": 45_114,
                                                                           "others_2": 1_192, "fill": 0}
