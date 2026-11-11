"""Derived sets (``data_pipeline/v1/derived_sets.py``) and their views (``pilot_view.member_rows``)."""

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")
pytest.importorskip("PIL.Image")

from data_pipeline.v1 import canonical, derived_sets, pilot_view  # noqa: E402
from tests.data.test_v1_canonical import fetched  # noqa: E402,F401  (fixture)


def _cands():
    """Log a (train): 30 rows 0.5 s apart, the first 20 standstill, row 4 a navtrain scene; log b (a released val
    log): 10 standstill rows; log c: guarded (excluded), so no row of it is eligible."""
    logs, t, still, nav, split = [], [], [], [], []
    for log, n, n_still, sp in (("a", 30, 20, "train"), ("b", 10, 10, "dev"), ("c", 4, 4, "excluded")):
        logs += [log] * n
        t += [i * 500_000 for i in range(n)]
        still += [i < n_still for i in range(n)]
        nav += [log == "a" and i == 4 for i in range(n)]
        split += [sp] * n
    n = len(logs)
    return {"token": np.array([f"{l}{i:02d}" for l, i in zip(logs, [x // 500_000 for x in t])], dtype=object),
            "log_name": np.array(logs, dtype=object), "t_us": np.array(t, dtype=np.int64),
            "our_split": np.array(split, dtype=object), "standstill": np.array(still),
            "in_navtrain": np.array(nav), "has_cam_f0": np.ones(n, dtype=bool), "pose_ok": np.ones(n, dtype=bool)}


def test_thin_keeps_one_standstill_anchor_per_5_s_and_every_navtrain_row():
    cands = _cands()
    window = 25  # log a, +12.5 s: inside a warm-up window, so not in v1-full's train
    full_train = np.array([k for k in range(40) if k != window])
    thin = derived_sets.thin_rows(cands, full_train)
    a_still = [k for k in thin if k < 20]
    assert a_still == [0, 4, 10]  # 0 s and 5 s by rule 6, 2 s because it is a navtrain scene
    assert [k for k in thin if 20 <= k < 30] == [20, 21, 22, 23, 24, 26, 27, 28, 29]  # moving rows all stay
    assert [k for k in thin if k >= 30] == [30]  # the released val log, thinned the same way
    derived_sets.check_thin(cands, thin, full_train, base_train=np.array([0, 10, 21]))
    with pytest.raises(RuntimeError, match="v1-all"):
        derived_sets.check_thin(cands, thin, full_train, base_train=np.array([1]))  # a row the thinning dropped
    with pytest.raises(RuntimeError, match="navtrain"):
        derived_sets.check_thin(cands, thin[thin != 4], full_train, None)
    with pytest.raises(RuntimeError, match="closer than 5"):
        derived_sets.check_thin(cands, np.sort(np.append(thin, 2)), full_train, None)


def test_thinning_before_the_window_matches_v1_all_rule():
    """The window removes a kept standstill anchor; the next one 0.5 s later is NOT promoted (v1-all's order)."""
    cands = _cands()
    full_train = np.array([k for k in range(40) if k != 10])  # row 10 (5 s, a kept anchor) is in a window
    thin = derived_sets.thin_rows(cands, full_train)
    assert [k for k in thin if k < 20] == [0, 4]


def _splits(tmp_path, name, train, dev):
    (tmp_path / "splits").mkdir(exist_ok=True)
    for split, tokens in (("train", train), ("dev", dev)):
        (tmp_path / "splits" / f"{name}_tokens_{split}.txt").write_text("".join(f"{t}\n" for t in tokens))


def test_member_rows_come_from_the_token_files(tmp_path):
    _splits(tmp_path, "dev-v2", [], ["t3", "t1"])
    tokens = np.array(["t0", "t1", "t2", "t3"], dtype=object)
    assert list(pilot_view.member_rows(str(tmp_path), "dev-v2", tokens)) == [1, 3]
    _splits(tmp_path, "dev-v2", [], ["t1", "t9"])
    with pytest.raises(ValueError, match="no frame"):
        pilot_view.member_rows(str(tmp_path), "dev-v2", tokens)


def test_a_derived_view_builds_without_a_membership_column(fetched):  # noqa: F811
    out, metadata = fetched
    canonical.build(str(out), ("smoke",), str(metadata))
    smoke = canonical.set_tokens(str(out), "smoke")
    train = sorted(t for t, s in smoke.items() if s == "train")
    dev = sorted(t for t, s in smoke.items() if s == "dev")
    _splits(out, "dev-v2", [], dev[:2])
    spec = pilot_view.build(str(out), "dev-v2", width=16, workers=1, storage="zip", prompt_version="pilot_v1.2")
    view = out / "views" / "pilot_v1.2" / "dev-v2"
    assert spec["samples"] == {"train": 0, "dev": 2} and spec["images_zip"]["members"] == 2
    assert (view / "train.jsonl").read_text() == ""
    ids = [json.loads(line)["id"] for line in (view / "dev.jsonl").read_text().splitlines()]
    assert sorted(i.rsplit("/", 1)[1] for i in ids) == dev[:2]  # ids are openscene/<log>/<token>
    # a derived train set: its split is its token files', so a dev-of-the-index token can be train
    _splits(out, "v1-full-thin", train[:1] + dev[2:3], dev[:1])
    spec = pilot_view.build(str(out), "v1-full-thin", width=16, workers=1)
    assert spec["samples"] == {"train": 2, "dev": 1}


def _hard_cands():
    """One log, 12 rows 1 s apart, ego frame paths: rows 0-3 navtrain (endpoint errors 2, 4, 6, 8 m), rows 4-7
    moving non-navtrain (1, 3, 5, 7 m), rows 8-11 standstill (0 m)."""
    from data_pipeline.v1.openscene_index import PATH_OFFSETS_S
    n, k = 12, len(PATH_OFFSETS_S)
    err = [2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0]
    speed = np.array([5.0] * 8 + [0.0] * 4)
    path = np.zeros((n, k, 3), dtype=np.float32)
    for r in range(n):
        for c, t in enumerate(PATH_OFFSETS_S):
            if t > 0:  # straight ahead at the speed, then shifted sideways by the row's error at +4 s
                path[r, c] = (speed[r] * t, err[r] * min(t, 4.0) / 4.0, 0.0)
    return {"token": np.array([f"h{i:02d}" for i in range(n)], dtype=object),
            "log_name": np.array(["a"] * n, dtype=object), "t_us": np.arange(n, dtype=np.int64) * 1_000_000,
            "standstill": np.array([False] * 8 + [True] * 4), "in_navtrain": np.array([True] * 4 + [False] * 8),
            "speed": speed, "path_xyyaw": path}


def test_v1_hard_keeps_navtrain_the_rows_as_hard_as_navtrain_and_thinned_standstills():
    cands = _hard_cands()
    err = derived_sets.cv_endpoint_error(cands, np.arange(12))
    assert np.allclose(err, [2, 4, 6, 8, 1, 3, 5, 7, 0, 0, 0, 0], atol=1e-4)
    thin = np.arange(12)
    hard, cut, parts = derived_sets.hard_rows(cands, thin, quantile=0.25, period_s=2.5)
    assert cut == pytest.approx(3.5, abs=1e-4)  # navtrain errors 2, 4, 6, 8: the 0.25 quantile
    assert list(hard) == [0, 1, 2, 3, 6, 7, 8, 11]  # 5 m and 7 m pass, 1 m and 3 m do not; stills 0 s and 3 s
    assert parts == {"navtrain": 4, "hard_moving": 2, "standstill": 2, "dropped_moving_below_cut": 2,
                     "dropped_standstill": 2}
    derived_sets.check_hard(cands, hard, thin, cut, period_s=2.5)
    with pytest.raises(RuntimeError, match="below the cut"):
        derived_sets.check_hard(cands, np.append(hard, 5), thin, cut, period_s=2.5)
    with pytest.raises(RuntimeError, match="at or above the cut"):
        derived_sets.check_hard(cands, hard[hard != 7], thin, cut, period_s=2.5)
    with pytest.raises(RuntimeError, match="navtrain"):
        derived_sets.check_hard(cands, hard[hard != 0], thin, cut, period_s=2.5)
    with pytest.raises(RuntimeError, match="never removed"):
        derived_sets.check_hard(cands, hard[hard < 8], thin, cut, period_s=2.5)
    with pytest.raises(RuntimeError, match="closer than"):
        derived_sets.check_hard(cands, np.sort(np.append(hard, 9)), thin, cut, period_s=2.5)
