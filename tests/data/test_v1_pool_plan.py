"""The v1.0-pool plan [proposed 2026-11-09]: ``token_sets`` (v1-nt, v1-xl-mix, v1-nt-mix, v1-all) and
``openscene_images.plan(set_name="v1-all")`` (PRETRAIN_V1_PLAN.md §11)."""

import hashlib
import json
import math
import os
import pickle
import shutil

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import navsim_splits, openscene_images, scene_exclusion, tables, token_sets  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from navsim_tools import maneuvers  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402
from tests.data.test_v1_xl_plan import sha, tokens, write_map_and_tree  # noqa: E402

S = 1_000_000
LV, BOS = "us-nv-las-vegas-strip", "us-ma-boston"


# --- token_sets on hand-made rows ----------------------------------------------------------------


def rows(spec):
    """Candidate columns from (log, t_s, split, city, label-ish kind); one 0.5 s anchor per entry."""
    n = len(spec)
    cands = {"token": np.array([f"{s[0]}_{s[1]:05.1f}" for s in spec], dtype=object),
             "log_name": np.array([s[0] for s in spec], dtype=object),
             "t_us": np.array([int(round(s[1] * S)) for s in spec], dtype=np.int64),
             "our_split": np.array([s[2] for s in spec], dtype=object),
             "city": np.array([s[3] for s in spec], dtype=object),
             "has_cam_f0": np.ones(n, dtype=bool), "pose_ok": np.ones(n, dtype=bool),
             "standstill": np.zeros(n, dtype=bool)}
    return cands


def test_listed_rows_take_the_first_failing_rule_as_the_one_reason():
    cands = rows([("a", t, "train", LV) for t in (0.0, 0.5, 1.0, 1.5)] + [("d", t, "dev", LV) for t in (0.0, 0.5)])
    cands["has_cam_f0"][2] = False  # a_001.0: ineligible
    window = np.zeros(6, dtype=bool)
    window[[1, 2, 5]] = True  # a_000.5 and a_001.0 (also ineligible), and d_000.5 (also a dev row)
    listed = ["a_000.0", "a_000.5", "a_001.0", "a_001.5", "d_000.0", "d_000.5", "not_a_row"]
    got = token_sets.listed_train_rows(cands, listed, window)
    assert cands["token"][got.selected].tolist() == ["a_000.0", "a_001.5"]
    assert got.unindexed == ["not_a_row"]
    r = got.report
    assert r["excluded_by_first_failing_rule"] == {"index_row": 1, "train_log": 2, "eligible": 1, "outside_window": 1}
    assert r["also_fail_a_later_rule"] == {"train_log": {"eligible": 0, "outside_window": 1},
                                           "eligible": {"outside_window": 1}}
    assert r["left_after"] == {"index_row": 6, "train_log": 4, "eligible": 3, "outside_window": 2}
    assert r["train_log_failures_by_split"] == {"dev": 2}
    assert sum(r["excluded_by_first_failing_rule"].values()) + r["selected"] == r["tokens"] == 7
    assert [name for name, _ in token_sets.LIST_RULES] == ["index_row", "train_log", "eligible", "outside_window"]
    with pytest.raises(ValueError, match="twice"):
        token_sets.listed_train_rows(cands, ["a_000.0", "a_000.0"], window)
    with pytest.raises(ValueError, match="boolean mask"):
        token_sets.listed_train_rows(cands, listed, window[:3])


def test_the_pool_is_thinned_before_the_window_and_holds_only_eligible_train_rows():
    spec = [("s", 0.5 * k, "train", LV) for k in range(30)] + [("d", 0.0, "dev", LV)]
    cands = rows(spec)
    cands["standstill"][:30] = True  # one standstill per 5 s: rows 0, 10, 20
    cands["pose_ok"][20] = False  # ineligible, so thinning keeps 21 instead
    window = np.zeros(31, dtype=bool)
    window[10] = True
    pool, report = token_sets.train_pool(cands, window)
    assert np.flatnonzero(pool).tolist() == [0, 21]  # 10 is thinned in, then cut by the window: 20 does not replace it
    assert report["eligible"] == 29 and report["after_standstill_thinning"] == 3 and report["after_window"] == 2


def test_mix_rows_follow_the_city_and_maneuver_rule_and_thin_stay_stopped_by_the_pinned_draw():
    labels_spec = [(BOS, maneuvers.KEEP), (BOS, maneuvers.STAY), (BOS, maneuvers.TURN_LEFT),
                   (LV, maneuvers.KEEP), (LV, maneuvers.START), (LV, maneuvers.STAY),
                   (LV, maneuvers.TURN_RIGHT), (LV, maneuvers.SIDEWAYS), (LV, maneuvers.BRAKE)]
    spec, labels = [], []
    for k in range(40):  # 40 copies of each kind, in their own logs
        for j, (city, label) in enumerate(labels_spec):
            spec.append((f"log{j}_{k}", 0.0, "train", city))
            labels.append(label)
    cands = rows(spec)
    labels = np.array(labels, dtype=object)
    n = len(labels)
    pool = np.ones(n, dtype=bool)
    base = np.flatnonzero(np.isin(labels, [maneuvers.STAY]) | (np.arange(n) < 9))  # every stay row, and one of each
    got, report = token_sets.mix_rows(cands, base, pool, labels)
    inside = np.zeros(n, dtype=bool)
    inside[got] = True
    city = cands["city"]
    assert inside[(city == BOS) & (labels != maneuvers.STAY)].all()  # (a) outside Las Vegas, all but stay-stopped
    assert inside[(city == LV) & np.isin(labels, list(token_sets.XL_MIX.gated_add))].all()  # (b) Las Vegas keep-list
    assert not inside[(city == LV) & np.isin(labels, [maneuvers.KEEP, maneuvers.START]) & (np.arange(n) >= 9)].any()
    assert inside[np.arange(n) < 9].sum() >= 7  # base rows stay, but for stay-stopped ones the draw left out
    stay_kept = np.flatnonzero(inside & (labels == maneuvers.STAY))
    n_other = int((inside & (labels != maneuvers.STAY)).sum())
    assert len(stay_kept) == round(n_other / 99) == report["stay_kept"] and set(stay_kept) <= set(base)
    # the draw, pinned: default_rng([20261027, 31]).choice over the union's stay rows in index-row order
    stay_rows = np.flatnonzero(labels == maneuvers.STAY)
    want = np.sort(np.random.default_rng([20261027, 31]).choice(stay_rows, size=round(n_other / 99), replace=False))
    assert stay_kept.tolist() == want.tolist()
    again, _ = token_sets.mix_rows(cands, base[::-1], pool, labels)
    assert again.tolist() == got.tolist()  # deterministic, whatever order the base comes in
    assert report["base_left_out_are_all_stay"] and report["stay_in_union_from_base"] == report["stay_in_union"]
    pool[base[0]] = False
    with pytest.raises(ValueError, match="base rows are not pool rows"):
        token_sets.mix_rows(cands, base, pool, labels)


def test_the_stay_count_rounds_one_percent_of_the_others():
    rule = token_sets.XL_MIX
    assert rule.seed == (20261027, 31) and rule.stay_share == 0.01 and rule.stay == maneuvers.STAY
    for n in range(0, 300_000, 997):
        assert rule.keep(n) == round(n / 99) == (2 * n + 99) // 198
    assert rule.keep(235_654) == 2_380  # v1-xl-mix: 235,654 others, 2,380 stay-stopped (V)
    # pool_mix.py's rule has 235,399 others: round keeps 2,378 (237,777 in all), floor 2,377 (recount, 2026-11-09)
    assert rule.keep(235_399) == 2_378 and math.floor(235_399 * 0.01 / 0.99) == 2_377


def test_explain_unindexed_finds_the_timing_gap_side(tmp_path):
    t = [k * 0.5 for k in range(30)]
    t_gap = t[:10] + [x + 0.5 for x in t[10:]]  # frame 10 onward 0.5 s late: a 1.0 s link between frames 9 and 10
    frames = [{"token": f"g{k:02d}", "log_name": "g", "timestamp": int(x * S)} for k, x in enumerate(t_gap)]
    with open(tmp_path / "g.pkl", "wb") as f:
        pickle.dump(frames[::-1], f)  # out of time order on purpose
    # g05: the gap (link 9) is in its future; g11: in its history; g01: no room; g15: no gap (unexplained)
    out = token_sets.explain_unindexed(str(tmp_path), ["g05", "g11", "g01", "g15", "elsewhere"])
    assert out["why"] == {"not a frame of the metadata": 1, "no gap found (unexplained)": 1,
                          "timing gap in the future only": 1, "timing gap in the history only": 1,
                          "too close to the log's start or end": 1}
    assert out["largest_gap"] == {"1.0 s": 2} and out["frames_found"] == 4 and out["in_logs"] == 1
    assert token_sets.explain_unindexed(str(tmp_path), [])["tokens"] == 0


def test_check_sets_raises_on_a_broken_post_condition():
    cands = rows([("a", 0.0, "train", LV), ("a", 0.5, "train", LV), ("d", 0.0, "dev", LV)])
    window = np.zeros(3, dtype=bool)
    dev = np.array([2])
    token_sets.check_sets(cands, {"x": np.array([0, 1])}, dev, window)
    window[1] = True
    with pytest.raises(RuntimeError, match="warm-up window"):
        token_sets.check_sets(cands, {"x": np.array([0, 1])}, dev, window)
    with pytest.raises(RuntimeError, match="TRAIN log"):
        token_sets.check_sets(cands, {"x": np.array([0, 2])}, dev, np.zeros(3, dtype=bool))
    sets = {token_sets.POOL: np.array([0]), token_sets.NT_MIX: np.array([0]), token_sets.MIX: np.array([0]),
            token_sets.NT: np.array([1])}
    with pytest.raises(RuntimeError, match="v1-nt is not inside v1-nt-mix"):
        token_sets.check_sets(cands, sets, dev, np.zeros(3, dtype=bool))


# --- plan(set v1-all), on a synthetic index ------------------------------------------------------


def straight(frames):
    """Drive straight at 10 m/s along yaw 0.4 instead of the synthetic arc: 'keep going'."""
    for i, frame in enumerate(frames):
        t = i * 0.5
        frame["ego2global_translation"] = [1000.0 + 10 * t * math.cos(0.4), 500.0 + 10 * t * math.sin(0.4), 0.0]
        frame["ego2global_rotation"] = [math.cos(0.2), 0.0, 0.0, math.sin(0.2)]
        frame["ego_dynamic_state"] = [10.0, 0.0, 0.0, 0.0]
    return frames


def in_city(frames, city):
    for frame in frames:
        frame["map_location"] = city
    return frames


LOGS = ("lv_keep", "lv_side", "lv_stop", "bos_side", "bos_stop", "gap_log", "dev_lv")


@pytest.fixture
def pool_inputs(tmp_path, monkeypatch):
    """A synthetic index of seven 60-frame logs, a base set, a navtrain list (known), a warm-up list, the metadata."""
    logs = {
        "lv_keep": straight(v1.make_log("lv_keep", 60)),  # keep going: not added in Las Vegas
        "lv_side": v1.make_log("lv_side", 60),  # the synthetic arc: 6.4 m sideways in 4 s, a sideways shift
        "lv_stop": v1.make_standstill(v1.make_log("lv_stop", 60)),  # stay stopped: never added
        "bos_side": in_city(v1.make_log("bos_side", 60), BOS),
        "bos_stop": in_city(v1.make_standstill(v1.make_log("bos_stop", 60)), BOS),
        "gap_log": v1.make_log("gap_log", 60),
        "dev_lv": v1.make_log("dev_lv", 60),
    }
    del logs["gap_log"][30]  # a dropped frame: a 1.0 s link between tok29 and tok31
    logs["bos_side"][40]["cams"]["CAM_F0"]["data_path"] = ""  # no CAM_F0: ineligible (rule 2)
    meta = v1.write_logs_dir(tmp_path / "logs", list(logs.values()))
    navtrain = ([f"lv_side_tok{k:02d}" for k in range(10, 21)] + [f"dev_lv_tok{k:02d}" for k in (5, 6, 7, 8, 20)]
                + ["bos_side_tok40"] + [f"bos_side_tok{k:02d}" for k in range(18, 23)]
                + [f"gap_log_tok{k:02d}" for k in (25, 29, 31)])
    root = v1.write_navsim_root(tmp_path / "navsim", train=[l for l in LOGS if l != "dev_lv"], val=["dev_lv"],
                                navtrain_logs=list(LOGS), navtrain_tokens=navtrain)
    pool = tmp_path / "pool"
    oi.build_index(str(meta), navsim_splits.load(str(root)), str(pool))
    lists = tmp_path / "lists"
    lists.mkdir()
    scene_exclusion.write_tokens(lists / scene_exclusion.NAVTRAIN_FILE, navtrain)
    # warm-up scenes: rows bos_side tok20, dev_lv tok30 and lv_keep tok50, and lv_keep tok01, a frame but not a row
    # (no history): the plan places it from the metadata, scanning the logs that hold a listed row
    scene_exclusion.write_tokens(lists / scene_exclusion.WARMUP_FILE,
                                 ["bos_side_tok20", "dev_lv_tok30", "lv_keep_tok01", "lv_keep_tok50"])
    monkeypatch.setitem(scene_exclusion.KNOWN_LISTS, sha(lists / scene_exclusion.NAVTRAIN_FILE),
                        {"filter": "navtrain", "tokens": len(navtrain)})
    base = tmp_path / "xl" / "splits"
    base.mkdir(parents=True)
    scene_exclusion.write_tokens(base / "v1-xl-nd_tokens_train.txt",
                                 ["lv_keep_tok20", "lv_keep_tok22", "lv_keep_tok24", "lv_side_tok30"]
                                 + [f"lv_stop_tok{k:02d}" for k in (3, 13, 23, 33, 43)])
    scene_exclusion.write_tokens(base / "v1-xl-nd_tokens_dev.txt", ["dev_lv_tok10", "dev_lv_tok20", "dev_lv_tok30"])
    maps = write_map_and_tree(tmp_path / "hf", {0: {"logs": list(LOGS[:4])}, 1: {"logs": list(LOGS[4:])}})
    return {"out": pool, "maps": maps, "extends": str(base / "v1-xl-nd"), "meta": str(meta),
            "navtrain": str(lists / scene_exclusion.NAVTRAIN_FILE), "warm": str(lists / scene_exclusion.WARMUP_FILE)}


def plan_pool(p, **kw):
    args = dict(set_name=token_sets.POOL, extends=p["extends"], navtrain=p["navtrain"], exclude_scenes=p["warm"],
                exclude_metadata=p["meta"])
    args.update(kw)
    return openscene_images.plan(str(p["out"]), *p["maps"], **args)


def test_pool_plan_writes_four_nested_sets_with_the_base_dev(pool_inputs):
    p = pool_inputs
    manifest = plan_pool(p)
    s = p["out"] / "splits"
    assert list(manifest["selections"]) == list(token_sets.SETS) == ["v1-all", "v1-nt-mix", "v1-xl-mix", "v1-nt"]
    assert manifest["base"]["set"] == "v1-xl-nd" and "v1-xl-nd" not in manifest["selections"]  # fetch skips the base
    base_dev = open(p["extends"] + "_tokens_dev.txt", "rb").read()
    train = {}
    for name in token_sets.SETS:
        assert (s / f"{name}_tokens_dev.txt").read_bytes() == base_dev  # every dev is the base's, byte for byte
        train[name] = set(tokens(s / f"{name}_tokens_train.txt"))
        assert manifest["selections"][name]["sha256"] == {k: sha(s / f"{name}_tokens_{k}.txt") for k in ("train", "dev")}
        assert manifest["selections"][name]["train"] == len(train[name])
    assert train["v1-nt-mix"] == train["v1-nt"] | train["v1-xl-mix"] and train["v1-nt-mix"] <= train["v1-all"]
    assert manifest["frames_to_keep"] == len(train["v1-all"]) + 3
    for copy in ("v1-xl-nd_tokens_train.txt", "v1-xl-nd_tokens_dev.txt"):
        assert (s / copy).read_bytes() == open(os.path.join(os.path.dirname(p["extends"]), copy), "rb").read()
    assert (s / scene_exclusion.NAVTRAIN_FILE).read_bytes() == open(p["navtrain"], "rb").read()
    assert (s / scene_exclusion.WARMUP_FILE).read_bytes() == open(p["warm"], "rb").read()
    assert (s / "v1-all_label_conflicts.tsv").read_text().splitlines()[0] == "token\tset\treasons"
    assert not [f for f in os.listdir(s) if f.endswith(".part")]

    # v1-nt: each excluded navtrain token has one reason, the first rule it fails
    nt = manifest["navtrain"]
    assert train["v1-nt"] == {f"lv_side_tok{k:02d}" for k in range(10, 21)}
    assert nt["excluded_by_first_failing_rule"] == {"index_row": 3, "train_log": 5, "eligible": 1, "outside_window": 5}
    assert nt["also_fail_a_later_rule"]["train_log"] == {"eligible": 0, "outside_window": 1}  # dev_lv tok20
    assert nt["not_index_rows"]["why"] == {"timing gap in the future only": 2, "timing gap in the history only": 1}
    assert nt["window_failures"] == {"warmup_tokens": 1, "only_through_frames_placed_from_the_metadata": 0}
    assert nt["train_log_failures"]["in_pinned_dev_logs"]["rows"] == 5 and nt["index_in_navtrain_matches"]

    # nothing near a warm-up scene, in a dev log or ineligible is in any train set
    cands = tables.read(str(p["out"] / "index" / "candidates.parquet"), ["token", "log_name", "t_us", "our_split"])
    window, _ = scene_exclusion.window_mask(cands, ["bos_side_tok20", "dev_lv_tok30", "lv_keep_tok01", "lv_keep_tok50"], 7.0,
                                            {"lv_keep_tok01": ("lv_keep", 1_500_000)})
    near = set(cands["token"][window].tolist())
    assert {"lv_keep_tok05", "lv_keep_tok15", "lv_keep_tok36", "bos_side_tok06", "bos_side_tok34"} <= near
    assert manifest["scene_exclusion"]["located_in_metadata"] == {"lv_keep_tok01": "lv_keep"}
    for name, ts in train.items():
        assert not ts & near and not any(t.startswith("dev_lv") for t in ts) and "bos_side_tok40" not in ts, name

    # v1-xl-mix by city and maneuver
    mix = manifest["mix"]
    assert train["v1-xl-mix"] >= {f"bos_side_tok{k:02d}" for k in (3, 4, 5, 35, 51)}  # Boston: all but stay-stopped
    assert not any(t.startswith("bos_stop") for t in train["v1-xl-mix"])  # stay-stopped is never added
    assert {f"lv_side_tok{k:02d}" for k in range(3, 52)} <= train["v1-xl-mix"]  # Las Vegas sideways shifts
    assert train["v1-xl-mix"] & {f"lv_keep_tok{k:02d}" for k in range(3, 52)} == {"lv_keep_tok20", "lv_keep_tok22",
                                                                                   "lv_keep_tok24"}  # base only
    assert mix["added_gated_by_maneuver"] == {"sideways shift (>2 m, lane change or curve)": mix["added_gated"]}
    assert mix["left_out_gated"] == {"keep going": 17}  # lv_keep tok16-35, base aside
    assert mix["left_out_outside_gated"] == {"stay stopped": 5}  # bos_stop's thinned standstills
    stay = {t for t in train["v1-xl-mix"] if t.startswith("lv_stop")}
    assert len(stay) == mix["stay_kept"] == round(mix["other_in_union"] / 99) == 1
    assert manifest["selections"]["v1-xl-mix"]["train_detail"]["maneuver"]["stay stopped"] == 1

    # the pool: thinned standstills (one per 5 s), and v1-all = pool ∪ v1-nt
    assert {t for t in train["v1-all"] if t.startswith("bos_stop")} == {f"bos_stop_tok{k:02d}" for k in (3, 13, 23, 33, 43)}
    assert manifest["pool"]["after_window"] == len(train["v1-all"])


def test_pool_plan_is_deterministic_and_refuses_bad_inputs(pool_inputs, tmp_path, monkeypatch):
    p = pool_inputs
    plan_pool(p)
    s = p["out"] / "splits"
    first = {f: (s / f).read_bytes() for f in os.listdir(s) if f != "manifest.json"}
    plan_pool(p, navtrain=str(s / scene_exclusion.NAVTRAIN_FILE))  # again, from the copy in splits/
    assert {f: (s / f).read_bytes() for f in os.listdir(s) if f != "manifest.json"} == first
    for missing in ("extends", "navtrain", "exclude_scenes", "exclude_metadata"):
        with pytest.raises(ValueError, match=f"needs {missing}"):
            plan_pool(p, **{missing: None})
    with pytest.raises(ValueError, match="allow_unlocated is not for"):
        plan_pool(p, allow_unlocated=True)
    with pytest.raises(ValueError, match="navtrain is for v1-all only"):
        openscene_images.plan(str(p["out"]), *p["maps"], set_name="v1-xl", extends=p["extends"], navtrain=p["navtrain"])
    bad = tmp_path / "bad"
    bad.mkdir()
    data = open(p["navtrain"], "rb").read()
    (bad / "crlf.txt").write_bytes(data.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="CRLF"):
        plan_pool(p, navtrain=str(bad / "crlf.txt"))
    (bad / "unknown.txt").write_bytes(data + b"zzzz_not_navtrain\n")  # another list: its SHA-256 is not known
    with pytest.raises(ValueError, match="not a known navtrain token list"):
        plan_pool(p, navtrain=str(bad / "unknown.txt"))
    (bad / "warm.txt").write_bytes(b"lv_keep_tok01\nbos_side_tok20\n")  # not sorted
    with pytest.raises(ValueError, match="sorted"):
        plan_pool(p, exclude_scenes=str(bad / "warm.txt"))
    with pytest.raises(ValueError, match="base set's own folder"):
        plan_pool(p, extends=str(s / "v1-xl-nd"))
    assert {f: (s / f).read_bytes() for f in os.listdir(s) if f != "manifest.json"} == first  # nothing rewritten
    with pytest.raises(SystemExit):
        openscene_images.main(["plan", "--out", str(p["out"]), "--set", "v1-all", "--extends", p["extends"]])
    with pytest.raises(SystemExit):
        openscene_images.main(["plan", "--out", str(p["out"]), "--set", "v1-xl", "--navtrain", p["navtrain"]])


def test_a_failure_after_the_token_files_leaves_no_file(pool_inputs, monkeypatch):
    p = pool_inputs

    def broken(*args, **kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(openscene_images, "_write_label_conflicts", broken)
    with pytest.raises(RuntimeError, match="stop"):
        plan_pool(p)
    assert not (p["out"] / "splits").exists() or list((p["out"] / "splits").iterdir()) == []


def test_pool_label_conflicts_name_every_set_that_holds_the_row(tmp_path, monkeypatch):
    """The pool's sets are not nested (v1-nt and v1-xl-mix overlap), so the ``set`` column lists every holder."""
    cands = rows([("a", t, "train", LV) for t in (0.0, 0.5, 1.0, 1.5)] + [("d", 0.0, "dev", LV)])
    monkeypatch.setattr(openscene_images.novelty, "label_conflicts",
                        lambda c, r: [(int(x), ["accel_mismatch"]) for x in r])  # flag every row
    sets = [(token_sets.POOL, np.array([0, 1, 2, 3, 4]), None), (token_sets.NT_MIX, np.array([0, 1, 2, 4]), None),
            (token_sets.MIX, np.array([0, 1, 4]), None), (token_sets.NT, np.array([0, 2, 4]), None)]
    got = openscene_images._write_label_conflicts(cands, sets, str(tmp_path), token_sets.POOL, nested=False)
    lines = (tmp_path / "v1-all_label_conflicts.tsv").read_text(encoding="utf-8").splitlines()
    assert lines == ["token\tset\treasons",
                     "a_000.0\tv1-all,v1-nt-mix,v1-xl-mix,v1-nt/train\taccel_mismatch",  # in v1-nt and v1-xl-mix
                     "a_000.5\tv1-all,v1-nt-mix,v1-xl-mix/train\taccel_mismatch",
                     "a_001.0\tv1-all,v1-nt-mix,v1-nt/train\taccel_mismatch",  # in v1-nt, not v1-xl-mix
                     "a_001.5\tv1-all/train\taccel_mismatch",
                     "d_000.0\tv1-all,v1-nt-mix,v1-xl-mix,v1-nt/dev\taccel_mismatch"]
    assert got["rows"] == {token_sets.POOL: {"train": 4, "dev": 1}, token_sets.NT_MIX: {"train": 3, "dev": 1},
                           token_sets.MIX: {"train": 2, "dev": 1}, token_sets.NT: {"train": 2, "dev": 1}}
    openscene_images._write_label_conflicts(cands, sets, str(tmp_path), "nested")  # v1-xl's form: the smallest
    assert (tmp_path / "nested_label_conflicts.tsv").read_text(encoding="utf-8").splitlines()[1:3] == [
        "a_000.0\tv1-nt/train\taccel_mismatch", "a_000.5\tv1-xl-mix/train\taccel_mismatch"]


# --- plan(set v1-all) on the real index (local only) ----------------------------------------------

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
REAL = os.path.join(REPO, "data", "pretrain")
METADATA = os.path.join(REAL, "downloads", "openscene_metadata_trainval.tgz")
COMMITTED = os.path.join(REAL, "v1.0-pool", "splits")
REAL_SETS = {  # V, 2026-11-09: the plan and its independent recount agree on every count and hash
    "v1-nt": (80_626, "ecd1b53885ffda4503ffd58edc6f8b5c08f4080eda43d9682c3767457044cc73"),
    "v1-xl-mix": (238_034, "05a88a8c176da079dcec93ef7fcaef7a593142e073d7e68ae6ac425fe4e2a5d8"),
    "v1-nt-mix": (259_929, "00ed5a88bc61229f7e94e2e3f59d5b41c266a93a1baaf99e5157b8c2cf047bb4"),
    "v1-all": (426_694, "92081c07454b747fb7a120dfec15c813f8b8b4a2969c28fd10fb5bf7826da683"),
}
DEV_SHA = "79090629e278705a066890efbfc8271acd9204e2085c3f3ddd8d668316ab7205"


def test_committed_navtrain_list_is_the_known_source():
    path = os.path.join(COMMITTED, scene_exclusion.NAVTRAIN_FILE)
    data = open(path, "rb").read()
    assert hashlib.sha256(data).hexdigest() == scene_exclusion.NAVTRAIN_SHA256 and b"\r" not in data
    lines = data.decode().splitlines()
    assert len(lines) == scene_exclusion.NAVTRAIN_SOURCE["tokens"] == 103_288 and lines == sorted(set(lines))
    assert scene_exclusion.source_of(path)["filter"] == "navtrain"


def test_real_pool_plan_counts_and_hashes(tmp_path):
    index = os.path.join(REAL, "v1.0-large", "index", "candidates.parquet")
    if not (os.path.exists(index) and os.path.exists(METADATA)):
        pytest.skip("the real candidate index or OpenScene's metadata tgz is not here")
    out = tmp_path / "pool"
    (out / "index").mkdir(parents=True)
    try:
        os.link(index, out / "index" / "candidates.parquet")
    except OSError:
        shutil.copyfile(index, out / "index" / "candidates.parquet")
    with open(os.path.join(REAL, "v1.0-large", "images", "archives.json"), encoding="utf-8") as f:
        archives = {int(n): a for n, a in json.load(f).items()}
    xl = os.path.join(REAL, "v1.0-xl", "splits")
    manifest = openscene_images.plan(str(out), *write_map_and_tree(tmp_path / "hf", archives), set_name=token_sets.POOL,
                                     extends=os.path.join(xl, "v1-xl-nd"),
                                     navtrain=os.path.join(COMMITTED, scene_exclusion.NAVTRAIN_FILE),
                                     exclude_scenes=os.path.join(xl, scene_exclusion.WARMUP_FILE), exclude_metadata=METADATA)
    s = out / "splits"
    for name, (count, digest) in REAL_SETS.items():
        assert len(tokens(s / f"{name}_tokens_train.txt")) == count and sha(s / f"{name}_tokens_train.txt") == digest, name
        assert sha(s / f"{name}_tokens_dev.txt") == DEV_SHA
    for name in os.listdir(s):  # the committed plan is this plan (manifest aside: it holds times and the commit)
        if name != "manifest.json":
            assert (s / name).read_bytes() == open(os.path.join(COMMITTED, name), "rb").read(), name
    assert manifest["frames_to_keep"] == 428_694 and max(manifest["frames_to_keep_per_archive"].values()) == 7_447
    nt = manifest["navtrain"]
    assert nt["left_after"] == {"index_row": 102_373, "train_log": 84_325, "eligible": 84_325, "outside_window": 80_626}
    assert nt["excluded_by_first_failing_rule"] == {"index_row": 915, "train_log": 18_048, "eligible": 0,
                                                    "outside_window": 3_699}
    assert nt["also_fail_a_later_rule"]["train_log"]["outside_window"] == 800
    assert nt["not_index_rows"]["why"] == {"timing gap in the future only": 671, "timing gap in the history only": 235,
                                           "timing gaps in the history and the future": 9}
    assert nt["index_in_navtrain_matches"]
    assert (manifest["pool"]["after_standstill_thinning"], manifest["pool"]["after_window"]) == (435_274, 426_694)
    mix = manifest["mix"]
    assert (mix["added_outside_gated"], mix["added_gated"], mix["union"], mix["stay_in_union"], mix["stay_kept"]) == \
        (54_231, 28_957, 259_540, 23_886, 2_380)
    assert manifest["selections"]["v1-xl-mix"]["splits"]["train"]["logs"] == 1_084
    committed = json.load(open(os.path.join(COMMITTED, "manifest.json"), encoding="utf-8"))
    for key in ("frames_to_keep", "archive_order", "frames_to_keep_per_archive", "navtrain", "mix", "pool", "base"):
        assert json.loads(json.dumps(manifest[key])) == committed[key], key  # as written: tuples become lists
