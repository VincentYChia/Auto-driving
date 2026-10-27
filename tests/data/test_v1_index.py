import io
import json
import math
import os
import pickle

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import navsim_splits, safe_pickle, tables  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from navsim_tools import geometry, raw_frames  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402
from tests.shared import synthetic_navsim as syn  # noqa: E402

N = 30  # frames per synthetic log: windows need frames i-3 .. i+8, so 19 anchors (3 .. 21)
COL = {round(float(s), 1): k for k, s in enumerate(oi.PATH_OFFSETS_S)}


def no_splits():
    return navsim_splits.NavsimSplits(frozenset(), frozenset(), frozenset(), frozenset(), frozenset(), {}, {})


def index(frames, splits=None):
    return oi.index_log(frames, splits or no_splits(), f"{frames[0]['log_name']}.pkl")


def build(tmp_path, logs, **roots):
    root = v1.write_navsim_root(tmp_path / "navsim", **roots)
    out = tmp_path / "out"
    manifest = oi.build_index(str(v1.write_logs_dir(tmp_path / "logs", logs)), navsim_splits.load(str(root)), str(out))
    return manifest, tables.read(str(out / "index" / "candidates.parquet")), tables.read(str(out / "index" / "logs.parquet"))


def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    else:
        assert type(a) is type(b) and (a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)))


class Evil:
    def __reduce__(self):
        return (os.system, ("true",))


# --- safe_pickle -----------------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_safe_pickle_loads_what_openscene_writes(protocol):
    frames = v1.make_log("log_a", 3)
    frames[0]["anns"]["gt_names"] = np.array(frames[0]["anns"]["gt_names"])  # OpenScene stores a str array
    frames[0]["extra"] = (np.float64(1.5), np.int64(7), np.arange(6).reshape(2, 3)[:, ::2], {1, 2}, None, b"\x00")
    assert_same(safe_pickle.loads(pickle.dumps(frames, protocol=protocol)), frames)


def test_safe_pickle_refuses_other_globals():
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        safe_pickle.loads(pickle.dumps([Evil()]))
    unpickler = safe_pickle.RestrictedUnpickler(io.BytesIO(b""))
    for module, name in [("builtins", "eval"), ("numpy", "load"), ("os", "system"), ("numpy.core.multiarray", "frombuffer")]:
        with pytest.raises(pickle.UnpicklingError):
            unpickler.find_class(module, name)
    # numpy 1 and numpy 2 module names resolve to the same function under either numpy
    assert unpickler.find_class("numpy.core.multiarray", "_reconstruct") is unpickler.find_class(
        "numpy._core.multiarray", "_reconstruct"
    )


# --- one log ---------------------------------------------------------------------------------------


def test_index_matches_the_analytic_arc():
    frames = v1.make_log("log_a", N)
    log_row, rows = index(frames)
    assert log_row["n_candidates"] == 19 and log_row["n_no_room"] == 11 and log_row["n_gap_windows"] == 0
    assert list(rows["frame_index"]) == list(range(3, 22))
    assert rows["frame_id"][0] == "openscene/log_a/log_a_tok03"
    np.testing.assert_allclose(rows["speed"], syn.V, atol=1e-5)
    np.testing.assert_allclose(rows["yaw_rate"], syn.W, atol=1e-6)
    chord_speed = 2 * syn.V / syn.W * math.sin(syn.W * syn.DT) / (2 * syn.DT)
    np.testing.assert_allclose(rows["speed_from_path"], chord_speed, atol=1e-4)
    assert set(rows["decision_lon"]) == {"keep_speed"} and set(rows["decision_lat"]) == {"curve_left"}
    assert set(rows["command"]) == {"straight"} and rows["has_route"].all() and rows["command_onehot_ok"].all()
    assert rows["pose_ok"].all() and rows["speed_consistent"].all() and not rows["standstill"].any()
    assert not rows["crosses_scene_boundary"].any() and rows["has_cam_f0"].all()
    assert rows["cam_f0_path"][0] == "log_a/CAM_F0/0003.jpg"
    assert set(rows["city"]) == {"us-nv-las-vegas-strip"} and set(rows["traffic_side"]) == {"right"}

    radius, t = syn.V / syn.W, 4.0  # the arc in the anchor frame
    expected = [radius * math.sin(syn.W * t), radius * (1 - math.cos(syn.W * t)), syn.W * t]
    np.testing.assert_allclose(rows["path_xyyaw"][:, COL[4.0]], np.tile(expected, (19, 1)), atol=1e-4)
    np.testing.assert_allclose(rows["path_xyyaw"][:, COL[0.0]], 0.0, atol=1e-5)
    # anchor 3: history only back to -1.5 s; anchor 21: future only to +4.0 s
    assert list(rows["path_valid"][0]) == [False] * 5 + [True] * 16
    assert list(rows["path_valid"][-1]) == [True] * 17 + [False] * 4
    assert np.isnan(rows["path_xyyaw"][0, COL[-2.0]]).all() and np.isnan(rows["path_t_s"][0, COL[-2.0]])
    np.testing.assert_allclose(rows["path_t_s"][5][rows["path_valid"][5]], oi.PATH_OFFSETS_S[rows["path_valid"][5]], atol=1e-6)
    assert float(rows["max_time_err_s"].max()) < 1e-6


def test_batched_transform_equals_geometry():
    frames = v1.make_log("log_a", N)
    _, rows = index(frames)
    k = 10  # an anchor with the whole -4 .. +6 s window inside the log
    poses = np.stack([raw_frames.global_ego_pose(f) for f in frames])
    expected = geometry.absolute_to_relative(poses[k], poses[k - 8 : k + 13])
    np.testing.assert_allclose(rows["path_xyyaw"][rows["frame_index"] == k][0], expected, atol=1e-4)


def test_log_row_has_timing_and_calibration():
    log_row, _ = index(v1.make_log("log_a", N))
    assert log_row["n_frames"] == N and log_row["was_sorted"] and log_row["n_gaps"] == 0
    assert log_row["dt_hist"].sum() == N - 1 and log_row["dt_hist"][3] == N - 1  # all in [0.499, 0.501)
    np.testing.assert_array_equal(log_row["cam_f0_intrinsic"], np.eye(3).reshape(-1))
    np.testing.assert_array_equal(log_row["cam_f0_distortion"], np.zeros(5))
    assert not log_row["calib_varies"] and log_row["n_no_cam_f0"] == 0
    assert np.isnan(log_row["lidar2ego_rotation"]).all()  # the synthetic log has no lidar2ego fields
    frames = v1.make_log("log_a", N)
    frames[5]["cams"]["CAM_F0"]["cam_intrinsic"] = 2 * np.eye(3)
    assert index(frames)[0]["calib_varies"]


def test_timing_gap_breaks_windows_but_jitter_does_not():
    frames = [f for i, f in enumerate(v1.make_log("log_a", N)) if i != 15]
    log_row, rows = index(frames)
    assert log_row["n_gaps"] == 1 and log_row["n_candidates"] == 7  # anchors 3..6 and 19..21
    assert log_row["n_gap_windows"] == 18 - 7
    assert sorted(int(t[-2:]) for t in rows["token"]) == [3, 4, 5, 6, 19, 20, 21]  # tokens keep the original numbers

    jittered = v1.make_log("log_a", N)
    for i, frame in enumerate(jittered):
        frame["timestamp"] += 20_000 if i % 2 else -20_000  # gaps of 0.46 / 0.54 s
    log_row, rows = index(jittered)
    assert log_row["n_candidates"] == 19
    assert 0.039 < float(rows["max_time_err_s"].max()) <= 0.0401

    late = v1.make_log("log_a", N)
    for frame in late[15:]:
        frame["timestamp"] += 60_000  # one gap of 0.56 s, between frames 14 and 15
    assert index(late)[0]["n_candidates"] == 8  # anchors 3..6 and 18..21


def test_pose_jump_drops_windows_and_truncates_paths():
    frames = v1.make_log("log_a", N)
    for frame in frames[20:]:
        frame["ego2global_translation"] = [frame["ego2global_translation"][0] + 30.0] + list(frame["ego2global_translation"][1:])
    log_row, rows = index(frames)
    assert log_row["n_pose_jumps"] == 1
    bad = rows["frame_index"][~rows["pose_ok"]]
    assert list(bad) == list(range(12, 22))  # every window holding the 19 -> 20 link
    row = rows["frame_index"] == 11  # window ends at frame 19: kept, path stops before the jump
    assert rows["path_valid"][row][0, COL[4.0]] and not rows["path_valid"][row][0, COL[4.5]:].any()
    assert float(rows["max_step_m"][rows["frame_index"] == 12][0]) > 25


def test_flags_command_route_camera_scene_and_order():
    frames = v1.make_log("log_a", N)
    frames[10]["cams"]["CAM_F0"] = None
    frames[12]["driving_command"] = [0, 0, 0, 1]
    frames[13]["driving_command"] = [1, 1, 0, 0]
    frames[14]["roadblock_ids"] = []
    for frame in frames[15:]:
        frame["scene_token"] = "scene1"
    shuffled = [frames[k] for k in np.random.default_rng(0).permutation(N)]
    log_row, rows = index(shuffled)
    assert not log_row["was_sorted"] and log_row["n_candidates"] == 19
    at = {int(k): i for i, k in enumerate(rows["frame_index"])}
    assert not rows["has_cam_f0"][at[10]] and rows["cam_f0_path"][at[10]] == "" and rows["has_cam_f0"].sum() == 18
    assert rows["command"][at[12]] == "unknown" and rows["command_onehot_ok"][at[12]]
    assert rows["command"][at[13]] == "unknown" and not rows["command_onehot_ok"][at[13]]
    assert not rows["has_route"][at[14]] and rows["has_route"].sum() == 18
    crossing = sorted(int(k) for k in rows["frame_index"][rows["crosses_scene_boundary"]])
    assert crossing == list(range(7, 18))  # windows i-3 .. i+8 that hold frames 14 and 15
    assert log_row["n_no_cam_f0"] == 1 and log_row["n_no_route"] == 1 and log_row["n_bad_command"] == 1


def test_standstill():
    _, rows = index(v1.make_standstill(v1.make_log("log_s", N)))
    assert rows["standstill"].all()
    assert set(rows["decision_lon"]) == {"stopped"} and set(rows["decision_lat"]) == {"stationary"}
    np.testing.assert_allclose(rows["speed_from_path"], 0.0)


# --- splits, guards and the whole build ------------------------------------------------------------


def test_navsim_splits_guard_every_evaluation_filter(tmp_path):
    root = v1.write_navsim_root(
        tmp_path,
        train=["log_a", "log_m"],
        val=["log_b"],
        test=["log_t"],
        navtrain_logs=["log_a"],
        navtrain_tokens=["log_a_tok05"],
        filters={
            "navtest": {"log_names": ["log_t"], "tokens": ["log_t_tok05"]},
            "warmup_two_stage": {"log_names": ["log_w"]},
            "competition_private_tokens": {"tokens": ["log_a_tok07"]},
            "navmini": {"log_names": ["log_m"]},
        },
    )
    splits = navsim_splits.load(str(root))
    assert splits.guarded_logs == {"log_t": "nuplan_test+navtest", "log_w": "warmup_two_stage"}
    assert splits.guarded_tokens == {"log_t_tok05": "navtest", "log_a_tok07": "competition_private_tokens"}
    assert splits.navtrain_tokens == {"log_a_tok05"} and splits.nuplan_split("log_x") == "none"
    assert splits.devkit_commit is None and navsim_splits.LOG_SPLIT_YAML in splits.files


def test_build_index_splits_and_guards(tmp_path):
    names = ["log_a", "log_b", "log_t", "log_w", "log_x"]
    manifest, cands, logs = build(
        tmp_path,
        [v1.make_log(name, N) for name in names],
        train=["log_a"],
        val=["log_b"],
        test=["log_t"],
        navtrain_logs=["log_a"],
        navtrain_tokens=["log_a_tok05", "log_a_tok00"],
        filters={"warmup_two_stage": {"log_names": ["log_w"]}, "competition_private_tokens": {"tokens": ["log_a_tok07"]}},
    )
    assert manifest["counts"] == {"logs": 5, "frames": 5 * N, "candidates": 5 * 19}
    split_of = dict(zip(logs["log_name"], logs["our_split"]))
    assert split_of == {"log_a": "train", "log_b": "dev", "log_t": "excluded", "log_w": "excluded", "log_x": "train"}
    assert dict(zip(logs["log_name"], logs["nuplan_split"]))["log_x"] == "none"
    row = {t: i for i, t in enumerate(cands["token"])}
    assert cands["our_split"][row["log_a_tok07"]] == "excluded"
    assert cands["guard_reason"][row["log_a_tok07"]] == "competition_private_tokens"
    assert (cands["our_split"][cands["log_name"] == "log_a"] == "train").sum() == 18
    assert cands["in_navtrain"].sum() == 1 and cands["in_navtrain"][row["log_a_tok05"]]
    navtrain = dict(zip(logs["log_name"], logs["n_navtrain_frames"]))
    assert navtrain["log_a"] == 2 and dict(zip(logs["log_name"], logs["n_navtrain_candidates"]))["log_a"] == 1
    assert manifest["navsim"]["navtrain_tokens"] == 2 and manifest["input"]["kind"] == "directory"
    with open(tmp_path / "out" / "index" / "manifest.json") as f:
        assert json.load(f)["outputs"]["candidates.parquet"]["rows"] == 5 * 19
    assert tables.read_metadata(str(tmp_path / "out" / "index" / "candidates.parquet"))["parameters"]["path_offsets_s"][0] == -4.0


def test_tgz_stream_equals_directory(tmp_path):
    logs = [v1.make_log("log_a", N), v1.make_log("log_b", 25)]
    root = navsim_splits.load(str(v1.write_navsim_root(tmp_path / "navsim", train=["log_a"], val=["log_b"])))
    oi.build_index(str(v1.write_logs_dir(tmp_path / "dir", logs)), root, str(tmp_path / "from_dir"))
    manifest = oi.build_index(str(v1.write_logs_tgz(tmp_path / "meta.tgz", logs)), root, str(tmp_path / "from_tgz"))
    assert manifest["input"]["kind"] == "tgz"
    for name in ("candidates.parquet", "logs.parquet"):
        a = tables.read(str(tmp_path / "from_dir" / "index" / name))
        b = tables.read(str(tmp_path / "from_tgz" / "index" / name))
        assert a.keys() == b.keys()
        for key in a:
            if key == "source_member":
                continue
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
            if a[key].dtype.kind == "f":
                np.testing.assert_array_equal(a[key], b[key])
            else:
                assert a[key].tolist() == b[key].tolist()
    one = oi.build_index(str(tmp_path / "meta.tgz"), root, str(tmp_path / "one"), max_logs=1)
    assert one["input"]["logs_read"] == 1 and one["input"]["max_logs"] == 1


def test_build_index_refuses_a_tampered_pickle(tmp_path):
    logs = v1.write_logs_dir(tmp_path / "logs", [v1.make_log("log_a", N)])
    with open(logs / "log_evil.pkl", "wb") as f:
        pickle.dump([Evil()], f)
    root = navsim_splits.load(str(v1.write_navsim_root(tmp_path / "navsim")))
    with pytest.raises(pickle.UnpicklingError, match="log_evil.pkl"):
        oi.build_index(str(logs), root, str(tmp_path / "out"))


def test_tables_round_trip_shapes_strings_and_nan(tmp_path):
    columns = {
        "s": np.array(["a", "", "ccc"], dtype=object),
        "f": np.array([1.5, np.nan, -2.0], dtype=np.float32),
        "path": np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3),
        "mask": np.array([[True, False], [False, True], [True, True]]),
        "i": np.array([1, 2, 3], dtype=np.int64),
    }
    tables.write(str(tmp_path / "t.parquet"), columns, metadata={"note": "x"})
    back = tables.read(str(tmp_path / "t.parquet"))
    for key, values in columns.items():
        assert back[key].dtype == values.dtype and back[key].shape == values.shape
        np.testing.assert_array_equal(back[key], values)
    assert tables.read_metadata(str(tmp_path / "t.parquet")) == {"note": "x"}
    assert tables.read(str(tmp_path / "t.parquet"), columns=["path"])["path"].shape == (3, 4, 3)
