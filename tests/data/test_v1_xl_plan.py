import hashlib
import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import navsim_splits, novelty, openscene_images, selection  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from data_pipeline.v1.selection import SelectionConfig  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402

OFFSETS_S = oi.PATH_OFFSETS_S
LV, BOS = "us-nv-las-vegas-strip", "us-ma-boston"


# --- near-duplicates and the nested filter, on hand-made rows --------------------------------------


def drive(log, t_s, city=LV, speed=10.0, accel=0.0, lon="keep_speed", lat="straight", frame=None,
          yaw_at=None, shift_y=0.0):
    """One anchor: straight-line motion in its own frame, x = v·t + a·t²/2 over the stored offsets."""
    path = np.zeros((len(OFFSETS_S), 3))
    path[:, 0] = speed * OFFSETS_S + 0.5 * accel * OFFSETS_S ** 2
    path[oi.ANCHOR + 1:, 1] = shift_y
    if yaw_at is not None:
        path[oi.ANCHOR + yaw_at[0], 2] = np.radians(yaw_at[1])
    return {"log_name": log, "t_us": int(round(t_s * 1e6)), "frame_index": int(round(t_s / 0.5)) if frame is None else frame,
            "city": city, "decision_lon": lon, "decision_lat": lat, "speed": speed, "path_xyyaw": path}


def table(rows):
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    out = {k: np.array(v, dtype=object) for k, v in cols.items() if isinstance(v[0], str)}
    out["t_us"] = np.array(cols["t_us"], dtype=np.int64)
    out["frame_index"] = np.array(cols["frame_index"], dtype=np.int32)
    out["speed"] = np.array(cols["speed"], dtype=np.float32)
    out["path_xyyaw"] = np.stack(cols["path_xyyaw"]).astype(np.float32)
    return out


def test_near_duplicate_needs_every_condition():
    def pair(**b):
        return table([drive("a", 0.0), drive("a", 1.0, **b)])

    assert novelty.near_duplicate(pair(), 0, 1) and novelty.near_duplicate(pair(), 1, 0)
    # |dv| >= 0.5 m/s (the faster target also drifts 0.9-1.4 m ahead, so the ADE limit is lifted here)
    assert not novelty.near_duplicate(pair(speed=10.6), 0, 1, target_ade_m=5.0)
    assert novelty.near_duplicate(pair(speed=10.4), 0, 1, target_ade_m=5.0)
    assert not novelty.near_duplicate(pair(speed=10.4), 0, 1)  # 0.9 m ADE
    assert not novelty.near_duplicate(pair(lat="curve_left"), 0, 1)
    assert not novelty.near_duplicate(pair(lon="accelerate"), 0, 1)
    assert not novelty.near_duplicate(pair(frame=3), 0, 1)  # frame_index must step by exactly k = 2
    assert not novelty.near_duplicate(pair(shift_y=0.7), 0, 1)  # target ADE >= 0.6 m
    assert novelty.near_duplicate(pair(shift_y=0.5), 0, 1)
    rows = table([drive("a", 0.0, yaw_at=(2, 2.5)), drive("a", 1.0)])  # heading 2.5 deg at the later anchor
    assert not novelty.near_duplicate(rows, 0, 1)
    assert novelty.near_duplicate(rows, 0, 1, dheading_deg=3.0)
    other_log = table([drive("a", 0.0), drive("b", 1.0)])
    assert not novelty.near_duplicate(other_log, 0, 1)


def lv_and_boston():
    """Rows (index: role) of one Las Vegas and one Boston log; base rows are marked B."""
    spec = [
        ("lv", 0, {}, "B"), ("lv", 1, {}, "+"), ("lv", 2, {}, "B"),  # 1 repeats both neighbours
        ("lv", 10, dict(speed=0.2, lon="start"), "B"),  # stopped, zero target ...
        ("lv", 11, dict(speed=0.4, accel=1.5, lon="start"), "+"),  # ... next to a start-up: ADE >= 0.6
        ("lv", 20, {}, "+"), ("lv", 21, {}, "+"),  # no base row near: the first stays, the second repeats it
        ("lv", 30, {}, "+"), ("lv", 31, {}, "B"),  # a base row ahead counts
        ("bos", 0, dict(city=BOS), "B"), ("bos", 1, dict(city=BOS), "+"),  # not gated
    ]
    cands = table([drive(log, t, **kw) for log, t, kw, _ in spec])
    base = np.array([i for i, s in enumerate(spec) if s[3] == "B"])
    return cands, np.arange(len(spec)), base


def test_nested_filter_drops_only_added_gated_repeats():
    cands, rows, base = lv_and_boston()
    kept = novelty.nested_filter(cands, rows, base)
    assert sorted(set(rows) - set(kept.tolist())) == [1, 6, 7]  # lv at 1, 21 and 30 s
    assert np.isin(base, kept).all() and np.isin(kept, rows).all()
    assert 4 in kept  # the start-up next to a standstill stays
    assert {9, 10} <= set(kept.tolist())  # Boston is not gated
    assert 10 not in novelty.nested_filter(cands, rows, base, cities=None)  # unless every city is gated


def test_nested_filter_is_deterministic_and_order_free():
    cands, rows, base = lv_and_boston()
    first = novelty.nested_filter(cands, rows, base)
    shuffled = np.random.default_rng(0).permutation(rows)
    assert novelty.nested_filter(cands, shuffled, base[::-1]).tolist() == first.tolist()
    assert novelty.nested_filter(cands, rows, base).tolist() == first.tolist()
    every_base = novelty.nested_filter(cands, rows, rows)
    assert every_base.tolist() == rows.tolist()  # base rows are never dropped, even when they repeat
    with pytest.raises(ValueError, match="base row"):
        novelty.nested_filter(cands, rows[:3], base)


def conflict_rows():
    t = OFFSETS_S

    def row(x, y=None, yaw=None, command="straight", lat="straight", accel_lon=0.0, consistent=True):
        path = np.zeros((len(t), 3))
        path[:, 0] = x
        path[:, 1] = 0.0 if y is None else y
        path[:, 2] = 0.0 if yaw is None else yaw
        return {"path_xyyaw": path, "path_t_s": t.copy(), "command": command, "decision_lat": lat,
                "accel_lon": accel_lon, "speed_consistent": consistent}

    yaw_jump = np.where(t >= 1.0, 0.2, 0.0)
    rows = [
        row(10 * t),  # clean
        row(10 * t, command="left", lat="turn_right"),
        row(10 * t, consistent=False),
        row(10 * t, accel_lon=4.0),  # the path does not accelerate
        row(20 * t + 5 * t ** 2, accel_lon=10.0),  # 10 m/s² throughout
        row(10 * t * np.cos(np.radians(30)), y=10 * t * np.sin(np.radians(30))),  # moves 30° off its heading
        row(np.zeros_like(t), yaw=yaw_jump),  # turns 0.2 rad on the spot
    ]
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    return {"path_xyyaw": np.stack(cols["path_xyyaw"]).astype(np.float32), "path_t_s": np.stack(cols["path_t_s"]),
            "command": np.array(cols["command"], dtype=object), "decision_lat": np.array(cols["decision_lat"], dtype=object),
            "accel_lon": np.array(cols["accel_lon"], dtype=np.float32), "speed_consistent": np.array(cols["speed_consistent"])}


def test_label_conflicts_name_each_hard_bad_test():
    cands = conflict_rows()
    found = dict(novelty.label_conflicts(cands, np.arange(7)))
    assert found == {1: ["command_contradicts_turn"], 2: ["speed_inconsistent"], 3: ["accel_mismatch"],
                     4: ["accel_gt_8"], 5: ["slip_gt_20deg"], 6: ["yaw_in_place"]}
    assert novelty.label_conflicts(cands, [0]) == [] and novelty.label_conflicts(cands, []) == []


# --- plan(set v1-xl), on a synthetic index -------------------------------------------------------


def write_map_and_tree(where, archives):
    """OpenScene's archive → log map and Hugging Face's tree listing, in their formats."""
    where.mkdir(parents=True, exist_ok=True)
    (where / "map.json").write_text(json.dumps({f"openscene_sensor_trainval_{n}": a["logs"] for n, a in archives.items()}))
    (where / "tree.json").write_text(json.dumps([
        {"type": "file", "path": f"openscene-v1.1/openscene_sensor_trainval_camera/{openscene_images.ARCHIVE_NAME.format(n=n)}",
         "lfs": {"oid": a.get("sha256", "0" * 64), "size": a.get("bytes", 1000)}} for n, a in archives.items()]))
    return str(where / "map.json"), str(where / "tree.json")


def copy_index(src, dst):
    (dst / "index").mkdir(parents=True)
    shutil.copyfile(src / "index" / "candidates.parquet", dst / "index" / "candidates.parquet")
    return dst


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(path):
    return path.read_text().split()


@pytest.fixture
def small(tmp_path, monkeypatch):
    """A synthetic index, v1-large planned in it, and small LARGE / XL configs."""
    logs = [v1.make_log("log_a", 60), v1.make_log("log_b", 60), v1.make_log("log_c", 60)]
    root = v1.write_navsim_root(tmp_path / "navsim", train=["log_a", "log_b"], val=["log_c"])
    large = tmp_path / "large"
    oi.build_index(str(v1.write_logs_dir(tmp_path / "logs", logs)), navsim_splits.load(str(root)), str(large))
    maps = write_map_and_tree(tmp_path / "hf", {0: {"logs": ["log_a"]}, 1: {"logs": ["log_b", "log_c"]}})
    monkeypatch.setattr(selection, "LARGE", SelectionConfig("v1-large", n_train=10, n_dev=4, balance_alpha=0.5))
    monkeypatch.setattr(selection, "XL", SelectionConfig("v1-xl", n_train=30, n_dev=4, spacing_s=1.0, balance_alpha=0.5))
    openscene_images.plan(str(large), *maps, set_name="v1-large")
    return large, copy_index(large, tmp_path / "xl"), maps


def test_xl_plan_nests_the_three_sets_with_one_dev_file(small):
    large, xl, maps = small
    manifest = openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(large / "splits" / "v1-large"))
    assert list(manifest["selections"]) == ["v1-xl", "v1-xl-nd", "v1-large"]
    s, L = xl / "splits", large / "splits"
    sets = {n: set(tokens(s / f"{n}_tokens_train.txt")) for n in manifest["selections"]}
    assert len(sets["v1-xl"]) == 30 and sets["v1-large"] <= sets["v1-xl-nd"] <= sets["v1-xl"]
    for split in ("train", "dev"):
        assert (s / f"v1-large_tokens_{split}.txt").read_bytes() == (L / f"v1-large_tokens_{split}.txt").read_bytes()
    for name in manifest["selections"]:
        assert (s / f"{name}_tokens_dev.txt").read_bytes() == (L / "v1-large_tokens_dev.txt").read_bytes()
        assert manifest["selections"][name]["sha256"] == {k: sha(s / f"{name}_tokens_{k}.txt") for k in ("train", "dev")}
        assert manifest["selections"][name]["train"] == len(sets[name])
    assert manifest["base"]["sha256"] == {k: sha(L / f"v1-large_tokens_{k}.txt") for k in ("train", "dev")}
    assert manifest["base"]["set"] == "v1-large" and manifest["novelty"]["set"] == "v1-xl-nd"
    assert manifest["frames_to_keep"] == 30 + 4 and manifest["smoke_archives"] == []
    train = manifest["selections"]["v1-xl"]["splits"]["train"]
    assert train["fixed"] == 10 and train["added"] == 20 and manifest["selections"]["v1-xl"]["splits"]["dev"]["added"] == 0
    assert (s / "v1-xl_label_conflicts.tsv").read_text().splitlines()[0] == "token\tset\treasons"


def test_xl_plan_refuses_a_bad_base(small, tmp_path):
    large, xl, maps = small
    prefix = str(large / "splits" / "v1-large")
    with pytest.raises(ValueError, match="extends"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl")
    with pytest.raises(ValueError, match="extends is for"):
        openscene_images.plan(str(xl), *maps, set_name="v1-large", extends=prefix)
    with pytest.raises(SystemExit):
        openscene_images.main(["plan", "--out", str(xl), "--set", "v1-xl"])
    bad = tmp_path / "bad"
    bad.mkdir()
    train = tokens(large / "splits" / "v1-large_tokens_train.txt")
    dev = tokens(large / "splits" / "v1-large_tokens_dev.txt")
    (bad / "b_tokens_train.txt").write_text("\n".join(train + ["no_such_token"]) + "\n", newline="\n")
    (bad / "b_tokens_dev.txt").write_text("\n".join(dev) + "\n", newline="\n")
    with pytest.raises(ValueError, match="not in the index"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(bad / "b"))
    (bad / "b_tokens_train.txt").write_text("\n".join(train) + "\n", newline="\n")
    (bad / "b_tokens_dev.txt").write_text("\n".join(dev + train[:1]) + "\n", newline="\n")  # a train token listed as dev
    with pytest.raises(ValueError, match="another split"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(bad / "b"))
    (bad / "b_tokens_dev.txt").write_text("\n".join(dev[:-1]) + "\n", newline="\n")  # 3 base dev rows, n_dev 4: a dev row is drawn
    with pytest.raises(RuntimeError, match="dev is not"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(bad / "b"))
    assert not (xl / "splits").exists()  # refused before anything was written
    (bad / "b_tokens_dev.txt").write_bytes(("\r\n".join(dev) + "\r\n").encode())  # a CRLF checkout
    with pytest.raises(ValueError, match="CRLF"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(bad / "b"))
    assert not (xl / "splits").exists()


def test_xl_plan_refuses_the_base_folder_and_leaves_no_files_when_it_fails(small, monkeypatch):
    large, xl, maps = small
    before = {p.name: p.read_bytes() for p in (large / "splits").iterdir()}
    archives = (large / "images" / "archives.json").read_bytes()
    with pytest.raises(ValueError, match="base set's own folder"):
        openscene_images.plan(str(large), *maps, set_name="v1-xl", extends=str(large / "splits" / "v1-large"))
    assert {p.name: p.read_bytes() for p in (large / "splits").iterdir()} == before
    assert (large / "images" / "archives.json").read_bytes() == archives

    def broken(*args, **kwargs):  # a failure after the token files are written
        raise RuntimeError("stop")

    original = openscene_images._write_label_conflicts
    monkeypatch.setattr(openscene_images, "_write_label_conflicts", broken)
    with pytest.raises(RuntimeError, match="stop"):
        openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(large / "splits" / "v1-large"))
    assert list((xl / "splits").iterdir()) == []  # no token file, no .part, no manifest
    monkeypatch.setattr(openscene_images, "_write_label_conflicts", original)
    openscene_images.plan(str(xl), *maps, set_name="v1-xl", extends=str(large / "splits" / "v1-large"))
    assert not [p for p in (xl / "splits").iterdir() if p.name.endswith(".part")]


# --- plan(set v1-xl) on the real index (local only) ----------------------------------------------

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
REAL = os.path.join(REPO, "data", "pretrain")
XL_TRAIN_SHA = "d29d242611447c67c7a03e1305f4fa9458f45404efa68e5d54818a1ad7e2cdf4"  # V, 2026-11-08: 3 scripts agree
ND_TRAIN_SHA = "f74bdbcded2d3c98f56e53663326f77886601e70eda021211274e6a1d6bdf94e"  # V, 2026-11-08: judge_nested.py
DEV_SHA = "79090629e278705a066890efbfc8271acd9204e2085c3f3ddd8d668316ab7205"  # v1-pilot's = v1-large's dev file


def test_real_xl_plan_counts_and_hashes(tmp_path):
    """Without --exclude-scenes: the first v1-xl plan (2026-11-08), byte for byte, so the new options change nothing
    by default. The committed plan has the warm-up exclusion: test_v1_scene_exclusion.py."""
    index = os.path.join(REAL, "v1.0-large", "index", "candidates.parquet")
    if not os.path.exists(index):
        pytest.skip("the real candidate index is not here")
    out = tmp_path / "xl"
    (out / "index").mkdir(parents=True)
    try:
        os.link(index, out / "index" / "candidates.parquet")
    except OSError:
        shutil.copyfile(index, out / "index" / "candidates.parquet")
    with open(os.path.join(REAL, "v1.0-large", "images", "archives.json"), encoding="utf-8") as f:
        archives = {int(n): a for n, a in json.load(f).items()}
    base = os.path.join(REAL, "v1.0-large", "splits", "v1-large")
    manifest = openscene_images.plan(str(out), *write_map_and_tree(tmp_path / "hf", archives), set_name="v1-xl",
                                     extends=base)
    s = out / "splits"
    assert sha(s / "v1-xl_tokens_train.txt") == XL_TRAIN_SHA and len(tokens(s / "v1-xl_tokens_train.txt")) == 200_000
    assert sha(s / "v1-xl-nd_tokens_train.txt") == ND_TRAIN_SHA and len(tokens(s / "v1-xl-nd_tokens_train.txt")) == 176_232
    pilot_dev = open(os.path.join(REAL, "v1.0-pilot", "splits", "v1-pilot_tokens_dev.txt"), "rb").read()
    for name in ("v1-xl", "v1-xl-nd", "v1-large"):
        assert (s / f"{name}_tokens_dev.txt").read_bytes() == pilot_dev and sha(s / f"{name}_tokens_dev.txt") == DEV_SHA
    large = set(tokens(s / "v1-large_tokens_train.txt"))
    assert large <= set(tokens(s / "v1-xl-nd_tokens_train.txt")) and large <= set(tokens(s / "v1-xl_tokens_train.txt"))
    assert manifest["frames_to_keep"] == 202_000 and manifest["base"]["prefix"] == "data/pretrain/v1.0-large/splits/v1-large"
    assert manifest["selections"]["v1-xl"]["splits"]["train"]["stages"] == {"others_1": 54_120, "cruise": 45_794,
                                                                           "others_2": 86, "fill": 0}
    assert manifest["label_conflicts"]["rows"]["v1-xl"] == {"train": 54, "dev": 1}
