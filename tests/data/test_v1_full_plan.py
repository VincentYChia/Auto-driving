"""The v1.0-full plan [proposed 2026-11-10]: ``token_sets`` (dev-v2's held-out logs and draw, ``every_row``) and
``openscene_images.plan(set_name="v1-full")`` (PRETRAIN_V1_PLAN.md §12)."""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import navsim_splits, openscene_images, scene_exclusion, tables, token_sets  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402
from tests.data.test_v1_pool_plan import in_city  # noqa: E402
from tests.data.test_v1_xl_plan import sha, tokens, write_map_and_tree  # noqa: E402

S = 1_000_000
LV, BOS = "us-nv-las-vegas-strip", "us-ma-boston"


# --- the rules on hand-made rows -----------------------------------------------------------------


def rows(spec):
    """Candidate columns from (log, t_s, split, city, navtrain, command); one anchor per entry."""
    n = len(spec)
    return {"token": np.array([f"{s[0]}@{s[1]:05.1f}" for s in spec], dtype=object),
            "log_name": np.array([s[0] for s in spec], dtype=object),
            "t_us": np.array([int(round(s[1] * S)) for s in spec], dtype=np.int64),
            "our_split": np.array([s[2] for s in spec], dtype=object),
            "city": np.array([s[3] for s in spec], dtype=object),
            "in_navtrain": np.array([s[4] for s in spec], dtype=bool),
            "command": np.array([s[5] for s in spec], dtype=object),
            "has_cam_f0": np.ones(n, dtype=bool), "pose_ok": np.ones(n, dtype=bool),
            "standstill": np.zeros(n, dtype=bool)}


def log(drive, k):
    return f"{drive}_{k:05d}_{k + 99:05d}"


A, B, C, D, T = ("2021.06.07.12.00.00_veh-35", "2021.06.07.13.00.00_veh-36", "2021.06.08.12.00.00_veh-40",
                 "2021.06.08.13.00.00_veh-41", "2021.06.01.10.00.00_veh-01")


def test_quotas_and_drives():
    assert token_sets.quotas(2_000, token_sets.NAVTEST_CITY_SHARE) == {LV: 669, BOS: 613,
                                                                      "us-pa-pittsburgh-hazelwood": 402, "sg-one-north": 316}
    assert token_sets.quotas(40, token_sets.NAVTEST_CITY_SHARE) == {LV: 14, BOS: 12, "us-pa-pittsburgh-hazelwood": 8,
                                                                    "sg-one-north": 6}
    assert token_sets.quotas(3, (("a", 1.0), ("b", 1.0))) == {"a": 2, "b": 1}  # a tie goes to the earlier name
    assert token_sets.drive_of("2021.06.07.12.00.00_veh-35_00061_00120") == "2021.06.07.12.00.00_veh-35"
    with pytest.raises(ValueError, match="not a nuPlan log name"):
        token_sets.drive_of("lv_keep")


def small_rule(**kw):
    args = dict(name="dev-v2", city_share=((LV, 2.0), (BOS, 1.0)), n_logs=3, n_anchors=9, spacing_s=1.0,
                standstill_period_s=5.0, seed=(20261110, 2))
    args.update(kw)
    return token_sets.DevRule(**args)


def val_rows():
    """Four val drives (A: 2 LV logs, no navtrain; B: 1 LV log, navtrain; C, D: 1 Boston log each) and a train log."""
    spec = []
    for lg, city, navtrain in ((log(A, 1), LV, ()), (log(A, 101), LV, ()), (log(B, 1), LV, range(0, 40)),
                               (log(C, 1), BOS, (0, 1)), (log(D, 1), BOS, (0, 1, 2, 3, 4)), (log(T, 1), LV, ())):
        split = "train" if lg.startswith(T) else "dev"
        for k in range(40):
            spec.append((lg, k * 0.5, split, city, k in navtrain, "straight"))
    return rows(spec)


def test_dev_logs_are_whole_val_drives_with_the_fewest_navtrain_rows_per_city():
    cands = val_rows()
    window = np.zeros(len(cands["t_us"]), dtype=bool)
    logs, report = token_sets.choose_dev_logs(cands, window, small_rule())
    assert logs == sorted([log(A, 1), log(A, 101), log(C, 1)])  # LV: A's 2 logs (0 navtrain), not B; Boston: C
    assert report["per_city"][LV]["drives"] == [A] and report["per_city"][BOS]["navtrain_rows_held_out"] == 2
    assert report["drives_with_logs_in_two_splits"] == []
    # a drive with a train log is never held out; the nearest reachable count is taken (here 1 LV log: B)
    cands["our_split"][cands["log_name"] == log(A, 101)] = "train"
    logs, report = token_sets.choose_dev_logs(cands, window, small_rule())
    assert log(A, 1) not in logs and log(B, 1) in logs and report["drives_with_logs_in_two_splits"] == [A]
    assert report["per_city"][LV]["logs"] == 1 and report["per_city"][LV]["logs_wanted"] == 2


def test_dev_draw_takes_navtrain_first_then_spaced_rows_with_a_command():
    cands = val_rows()
    cands["command"][(cands["log_name"] == log(C, 1)) & (cands["t_us"] >= 10 * S)] = token_sets.UNKNOWN_COMMAND
    window = np.zeros(len(cands["t_us"]), dtype=bool)
    window[(cands["log_name"] == log(A, 1)) & (cands["t_us"] < 5 * S)] = True  # the first 5 s of A's first log
    logs = sorted([log(A, 1), log(A, 101), log(C, 1)])
    rule = small_rule()
    got, report = token_sets.draw_dev(cands, window, logs, rule)
    assert len(got) == 9 and report["per_city"][LV]["anchors"] == 6 and report["per_city"][BOS]["anchors"] == 3
    assert set(cands["log_name"][got]) <= set(logs) and not window[got].any()
    assert report["per_city"][BOS]["navtrain_tokens"] == 1  # C's two navtrain rows are 0.5 s apart: one is taken
    boston = got[cands["city"][got] == BOS]
    assert (cands["command"][boston] != token_sets.UNKNOWN_COMMAND).all()  # the fill skips unknown commands
    for lg in logs:
        t = np.sort(cands["t_us"][got][cands["log_name"][got] == lg])
        assert (np.diff(t) >= 0.95 * S).all()
    again, _ = token_sets.draw_dev(cands, window, logs[::-1], rule)
    assert again.tolist() == got.tolist()  # deterministic, whatever order the logs come in
    loose, _ = token_sets.draw_dev(cands, window, logs, small_rule(fill_needs_command=False))
    assert len(loose) == 9
    with pytest.raises(ValueError, match="too few rows"):
        token_sets.draw_dev(cands, window, logs, small_rule(n_anchors=90))


def test_every_row_keeps_everything_that_passes_and_releases_val_logs():
    cands = val_rows()
    cands["standstill"][:] = True  # nothing is thinned
    cands["pose_ok"][5] = False
    window = np.zeros(len(cands["t_us"]), dtype=bool)
    window[(cands["log_name"] == log(B, 1)) & (cands["t_us"] < 2 * S)] = True  # 4 rows of a released val log
    held = [log(A, 1), log(A, 101), log(C, 1)]
    got, report = token_sets.every_row(cands, window, held)
    groups = report["by_group"]
    assert groups["train_logs"]["left_after"] == {"index_row": 40, "eligible": 40, "outside_window": 40}
    assert groups["released_val_logs"]["left_after"] == {"index_row": 80, "eligible": 80, "outside_window": 76}
    assert groups["held_out_val_logs"]["left_after"] == {"index_row": 120, "eligible": 119, "outside_window": 119}
    assert groups["held_out_val_logs"]["excluded_by_first_failing_rule"] == {"eligible": 1, "outside_window": 0}
    assert len(got["train"]) == 116 and len(got["held_out"]) == 119 and groups["train_logs"]["standstill_rows_kept"] == 40
    assert set(cands["log_name"][got["held_out"]]) == set(held)
    with pytest.raises(ValueError, match="not a val"):
        token_sets.every_row(cands, window, [log(T, 1)])


def test_check_full_raises_on_a_broken_post_condition():
    cands = val_rows()
    window = np.zeros(len(cands["t_us"]), dtype=bool)
    held = [log(A, 1), log(A, 101), log(C, 1)]
    got, _ = token_sets.every_row(cands, window, held)
    dev, _ = token_sets.draw_dev(cands, window, held, small_rule())
    token_sets.check_full(cands, got["train"], got["held_out"], dev, window, held)
    with pytest.raises(RuntimeError, match="not held out"):  # half of drive A
        token_sets.check_full(cands, got["train"], got["held_out"], dev, window, held[1:])
    with pytest.raises(RuntimeError, match="held-out log or drive"):
        token_sets.check_full(cands, np.union1d(got["train"], got["held_out"][:1]), got["held_out"], dev, window, held)
    near = np.flatnonzero(cands["log_name"] == log(A, 1))[:2]  # 0.5 s apart
    with pytest.raises(RuntimeError, match="closer than"):
        token_sets.check_full(cands, got["train"], got["held_out"], near, window, held)
    with pytest.raises(RuntimeError, match="base train rows"):
        token_sets.check_full(cands, got["train"], got["held_out"], dev, window, held, got["held_out"][:1])


# --- plan(set v1-full), on a synthetic index ------------------------------------------------------

TRAIN_LOGS = (log(T, 1), log("2021.06.01.11.00.00_veh-02", 1))
VAL_LOGS = (log(A, 1), log(A, 101), log(B, 1), log(C, 1), log(D, 1))


@pytest.fixture
def full_inputs(tmp_path, monkeypatch):
    """A synthetic index of 7 nuPlan-named 60-frame logs, the base v1-all files, a warm-up list, the metadata."""
    cities = {log(C, 1): BOS, log(D, 1): BOS, TRAIN_LOGS[1]: BOS}
    logs = {name: in_city(v1.make_log(name, 60), cities.get(name, LV)) for name in TRAIN_LOGS + VAL_LOGS}
    for frame in logs[log(C, 1)][30:]:
        frame["driving_command"] = [0, 0, 0, 1]  # unknown: the fill skips these rows
    meta = v1.write_logs_dir(tmp_path / "logs", list(logs.values()))
    navtrain = ([f"{log(B, 1)}_tok{k:02d}" for k in range(3, 50)] + [f"{log(C, 1)}_tok{k:02d}" for k in (10, 20)]
                + [f"{log(D, 1)}_tok{k:02d}" for k in range(10, 20)] + [f"{TRAIN_LOGS[0]}_tok{k:02d}" for k in (30, 31)])
    root = v1.write_navsim_root(tmp_path / "navsim", train=list(TRAIN_LOGS), val=list(VAL_LOGS),
                                navtrain_logs=list(logs), navtrain_tokens=navtrain)
    out = tmp_path / "full"
    oi.build_index(str(meta), navsim_splits.load(str(root)), str(out))
    lists = tmp_path / "lists"
    lists.mkdir()
    # warm-up scenes: train row tok20 of the first train log, tok40 of a released val log (B), and tok01 of the
    # first train log, a frame but not a row (no history): the plan places it from the metadata (it scans the logs
    # that hold a listed row)
    warm = [f"{TRAIN_LOGS[0]}_tok20", f"{log(B, 1)}_tok40", f"{TRAIN_LOGS[0]}_tok01"]
    scene_exclusion.write_tokens(lists / scene_exclusion.WARMUP_FILE, warm)
    base = tmp_path / "pool" / "splits"
    base.mkdir(parents=True)
    scene_exclusion.write_tokens(base / "v1-all_tokens_train.txt",
                                 [f"{TRAIN_LOGS[0]}_tok{k:02d}" for k in (40, 50)] + [f"{TRAIN_LOGS[1]}_tok05"])
    scene_exclusion.write_tokens(base / "v1-all_tokens_dev.txt", [f"{log(B, 1)}_tok10", f"{log(C, 1)}_tok05",
                                                                 f"{log(B, 1)}_tok40"])  # the last inside the window
    maps = write_map_and_tree(tmp_path / "hf", {0: {"logs": list(TRAIN_LOGS + VAL_LOGS[:2])},
                                                1: {"logs": list(VAL_LOGS[2:])}})
    monkeypatch.setattr(token_sets, "DEV_V2", small_rule())
    return {"out": out, "maps": maps, "extends": str(base / "v1-all"), "meta": str(meta), "root": str(root),
            "warm": str(lists / scene_exclusion.WARMUP_FILE), "warm_tokens": warm}


def plan_full(p, **kw):
    args = dict(set_name=token_sets.FULL, extends=p["extends"], exclude_scenes=p["warm"], exclude_metadata=p["meta"])
    args.update(kw)
    return openscene_images.plan(str(p["out"]), *p["maps"], **args)


def test_full_plan_writes_the_resplit_sets(full_inputs):
    p = full_inputs
    manifest = plan_full(p)
    s = p["out"] / "splits"
    assert list(manifest["selections"]) == list(token_sets.FULL_SETS) and manifest["base"]["set"] == "v1-all"
    held = (s / "dev-v2_logs.txt").read_text().split()
    assert held == sorted([log(A, 1), log(A, 101), log(C, 1)])  # LV: A (no navtrain), Boston: C (2, not D's 10)
    cands = tables.read(str(p["out"] / "index" / "candidates.parquet"),
                        ["token", "log_name", "t_us", "our_split", "command", "in_navtrain"])
    window, _ = scene_exclusion.window_mask(cands, p["warm_tokens"], 7.0, {p["warm_tokens"][2]: (TRAIN_LOGS[0], 1_500_000)})
    passing = ~window
    in_held = np.isin(cands["log_name"], held)
    want_train = set(cands["token"][passing & ~in_held].tolist())  # TRAIN logs and the released val logs B and D
    train, dev = set(tokens(s / "v1-full_tokens_train.txt")), set(tokens(s / "v1-full_tokens_dev.txt"))
    assert train == want_train and any(t.startswith(log(D, 1)) for t in train) and any(t.startswith(log(B, 1)) for t in train)
    assert f"{log(B, 1)}_tok45" not in train and f"{TRAIN_LOGS[0]}_tok05" not in train  # inside the window
    assert manifest["scene_exclusion"]["located_in_metadata"] == {p["warm_tokens"][2]: TRAIN_LOGS[0]}
    assert (s / "v1-full-devlogs_tokens_train.txt").read_bytes() == b""
    devlogs = set(tokens(s / "v1-full-devlogs_tokens_dev.txt"))
    assert devlogs == set(cands["token"][passing & in_held].tolist()) and dev <= devlogs and len(dev) == 9
    row_of = {t: i for i, t in enumerate(cands["token"].tolist())}
    dev_rows = np.array([row_of[t] for t in dev])
    assert (cands["in_navtrain"][dev_rows] | (cands["command"][dev_rows] != "unknown")).all()
    assert f"{log(C, 1)}_tok10" in dev or f"{log(C, 1)}_tok20" in dev  # navtrain tokens come first
    assert manifest["selections"]["v1-full"]["sha256"]["dev"] == manifest["dev_v2"]["sha256"] == sha(s / "v1-full_tokens_dev.txt")
    assert (s / "dev-v1_tokens.txt").read_bytes() == open(p["extends"] + "_tokens_dev.txt", "rb").read()
    for split in ("train", "dev"):
        assert (s / f"v1-all_tokens_{split}.txt").read_bytes() == open(p["extends"] + f"_tokens_{split}.txt", "rb").read()
    assert manifest["dev_v1"]["in_v1-full_train"] == 1 and manifest["dev_v1"]["in_held_out_logs"] == 1  # B tok10, C tok05
    assert manifest["dev_v1"]["inside_window"] == 1
    assert manifest["frames_to_keep"] == len(train | devlogs) and manifest["covered_by_base"]["base_frames_not_kept"] == 1
    assert manifest["dev_logs"]["in_training_views"] == 0 and manifest["dev_logs"]["held_out_logs"] == 3
    groups = manifest["every_row"]["by_group"]
    assert manifest["with_the_old_dev"]["train"] == groups["train_logs"]["left_after"]["outside_window"]
    assert len(train) == groups["train_logs"]["left_after"]["outside_window"] + \
        groups["released_val_logs"]["left_after"]["outside_window"]
    assert manifest["selections"]["v1-full"]["splits"]["dev"]["selected"] == 9
    conflicts = (s / "v1-full_label_conflicts.tsv").read_text().splitlines()
    assert conflicts[0] == "token\tset\treasons" and all(line.split("\t")[1].endswith(("/train", "/dev")) for line in conflicts[1:])
    assert not [f for f in os.listdir(s) if f.endswith(".part")]


def test_full_plan_is_deterministic_and_refuses_bad_inputs(full_inputs):
    p = full_inputs
    plan_full(p)
    s = p["out"] / "splits"
    first = {f: (s / f).read_bytes() for f in os.listdir(s) if f != "manifest.json"}
    plan_full(p, exclude_scenes=str(s / scene_exclusion.WARMUP_FILE))  # again, from the copy in splits/
    assert {f: (s / f).read_bytes() for f in os.listdir(s) if f != "manifest.json"} == first
    for missing in ("extends", "exclude_scenes", "exclude_metadata"):
        with pytest.raises(ValueError, match=f"needs {missing}"):
            plan_full(p, **{missing: None})
    with pytest.raises(ValueError, match="allow_unlocated is not for v1-full"):
        plan_full(p, allow_unlocated=True)
    with pytest.raises(ValueError, match="navtrain is for v1-all only"):
        plan_full(p, navtrain=p["warm"])
    with pytest.raises(ValueError, match="base set's own folder"):
        plan_full(p, extends=str(s / "v1-all"))
    assert {f: (s / f).read_bytes() for f in os.listdir(s) if f != "manifest.json"} == first  # nothing rewritten
    for argv in (["--extends", p["extends"]], ["--extends", p["extends"], "--exclude-scenes", p["warm"],
                                               "--exclude-metadata", p["meta"], "--navtrain", p["warm"]]):
        with pytest.raises(SystemExit):
            openscene_images.main(["plan", "--out", str(p["out"]), "--set", "v1-full", *argv])


def test_view_split_comes_from_the_set_s_token_files(tmp_path):
    from data_pipeline.v1 import pilot_view

    (tmp_path / "splits").mkdir()
    scene_exclusion.write_tokens(tmp_path / "splits" / "s_tokens_train.txt", ["a", "b"])
    scene_exclusion.write_tokens(tmp_path / "splits" / "s_tokens_dev.txt", ["c"])
    toks = np.array(["a", "b", "c", "x"], dtype=object)
    index = np.array(["train", "dev", "dev", "dev"], dtype=object)  # b: a val row the re-split released
    got = pilot_view.set_split(str(tmp_path), "s", toks, np.array([0, 1, 2]), index)
    assert got.tolist() == ["train", "train", "dev", "dev"] and index.tolist()[1] == "dev"  # the input is not changed
    with pytest.raises(ValueError, match="neither of its token files"):
        pilot_view.set_split(str(tmp_path), "s", toks, np.array([3]), index)


# --- plan(set v1-full) on the real index (local only) ---------------------------------------------

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
REAL = os.path.join(REPO, "data", "pretrain")
METADATA = os.path.join(REAL, "downloads", "openscene_metadata_trainval.tgz")
COMMITTED = os.path.join(REAL, "v1.0-full", "splits")
REAL_FILES = {  # V, 2026-11-10: (lines, SHA-256) of the committed plan
    "v1-full_tokens_train.txt": (675_701, "759fb537d3df10c4dfafa06445de0c74f5c5131147e36fa002e5e7d76d724115"),
    "v1-full_tokens_dev.txt": (2_000, "0661d4815bd0cae50f1d3495ed1b024d2e40711965197603b9b1d593d36eb951"),
    "v1-full-devlogs_tokens_train.txt": (0, hashlib.sha256(b"").hexdigest()),
    "v1-full-devlogs_tokens_dev.txt": (14_828, "2e9203dd3ea86034aa4ea84e42164a17de4373b32de16ce0d5ebd9fce9be44e0"),
    "dev-v2_logs.txt": (40, "eff9eb2e884858bcc602b2461eb3017738fa04b2267293690da2743232815ed4"),
    "dev-v1_tokens.txt": (2_000, "79090629e278705a066890efbfc8271acd9204e2085c3f3ddd8d668316ab7205"),
}


def test_real_full_plan_counts_and_hashes(tmp_path):
    index = os.path.join(REAL, "v1.0-large", "index", "candidates.parquet")
    if not (os.path.exists(index) and os.path.exists(METADATA)):
        pytest.skip("the real candidate index or OpenScene's metadata tgz is not here")
    out = tmp_path / "full"
    (out / "index").mkdir(parents=True)
    try:
        os.link(index, out / "index" / "candidates.parquet")
    except OSError:
        shutil.copyfile(index, out / "index" / "candidates.parquet")
    with open(os.path.join(REAL, "v1.0-large", "images", "archives.json"), encoding="utf-8") as f:
        archives = {int(n): a for n, a in json.load(f).items()}
    manifest = openscene_images.plan(str(out), *write_map_and_tree(tmp_path / "hf", archives), set_name=token_sets.FULL,
                                     extends=os.path.join(REAL, "v1.0-pool", "splits", "v1-all"),
                                     exclude_scenes=os.path.join(REAL, "v1.0-xl", "splits", scene_exclusion.WARMUP_FILE),
                                     exclude_metadata=METADATA)
    s = out / "splits"
    for name, (count, digest) in REAL_FILES.items():
        assert len(tokens(s / name)) == count and sha(s / name) == digest, name
    for name in os.listdir(s):  # the committed plan is this plan (manifest aside: it holds times and the commit)
        if name != "manifest.json":
            assert (s / name).read_bytes() == open(os.path.join(COMMITTED, name), "rb").read(), name
    assert manifest["frames_to_keep"] == 690_529 and manifest["largest_archive"] == {
        "archive": 124, "frames": 11_650, "bytes": 19_295_805_663}
    groups = manifest["every_row"]["by_group"]
    assert {g: r["left_after"]["outside_window"] for g, r in groups.items()} == {
        "train_logs": 594_978, "released_val_logs": 80_723, "held_out_val_logs": 14_828}
    assert manifest["with_the_old_dev"]["train"] == 594_978 and manifest["covered_by_base"]["frames"] == 428_645
    logs = manifest["dev_v2"]["logs"]
    assert {c: (r["logs"], len(r["drives"])) for c, r in logs["per_city"].items()} == {
        LV: (14, 8), BOS: (12, 2), "us-pa-pittsburgh-hazelwood": (8, 2), "sg-one-north": (6, 1)}
    assert logs["navtrain_rows_held_out"] == 1_521 and manifest["navtrain_rows"]["in_v1-full_train"] == 96_353
    assert manifest["dev_v2"]["draw"]["navtrain_tokens"] == 743
    committed = json.load(open(os.path.join(COMMITTED, "manifest.json"), encoding="utf-8"))
    for key in ("frames_to_keep", "archive_order", "frames_to_keep_per_archive", "every_row", "dev_v1", "dev_logs",
                "navtrain_rows", "covered_by_base", "base"):
        assert json.loads(json.dumps(manifest[key])) == committed[key], key
    for key in ("logs", "file", "sha256"):
        assert json.loads(json.dumps(manifest["dev_v2"][key])) == committed["dev_v2"][key], key
