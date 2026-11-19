"""``pilot_sft/heads/features.py``: the stored-feature layout, a resumed write, and the ego vector read back
from a ``pilot_v1.2`` prompt. Numpy only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import features  # noqa: E402

PROMPT = (
    "Front camera image (CAM_F0).\n"
    "Ego state: speed 2.4 m/s, acceleration -0.7 m/s², yaw rate +0.00 rad/s.\n"
    "Past poses (x forward and y left in metres, heading in radians) at -1.5, -1.0, -0.5 and 0 s: "
    "(-4.23, 0.01, 0.00) (-2.65, -0.02, 0.00) (-1.25, -0.01, -0.03) (0.00, 0.00, 0.00).\n"
    "Route command: turn right.\n"
    "Predict the path for the next 4 seconds: 8 positions (x, y) in metres, one every 0.5 s."
)


def test_parse_ego_reads_the_prompt_back():
    ego = features.parse_ego(PROMPT)
    assert len(ego) == features.EGO_DIM == 19
    assert ego[:3] == [2.4, -0.7, 0.0]
    assert ego[3:15] == [-4.23, 0.01, 0.0, -2.65, -0.02, 0.0, -1.25, -0.01, -0.03, 0.0, 0.0, 0.0]
    assert ego[15:] == [0.0, 0.0, 1.0, 0.0]  # turn right


def test_parse_ego_refuses_a_masked_or_old_prompt():
    with pytest.raises(ValueError):
        features.parse_ego(PROMPT.replace("Ego state: speed 2.4 m/s, acceleration -0.7 m/s², yaw rate +0.00 rad/s.",
                                          "Ego state: not available."))
    with pytest.raises(ValueError):
        features.parse_ego(PROMPT.replace("Past poses", "Past positions"))


def _rows(n, t, d, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**16, size=(n, t, d), dtype=np.uint16)


def _metas(start, n):
    return [{"id": f"tok{start + i}", "ego": [float(start + i)] * features.EGO_DIM,
             "target_xy": [[float(start + i), 0.0]] * 8, "target_yaw": [0.0] * 8} for i in range(n)]


def test_write_then_read_back(tmp_path):
    w = features.Writer(str(tmp_path), [12, 24], rows=5, tokens=6, width=4, settings={"model": "m"})
    a12, a24 = _rows(3, 6, 4, 0), _rows(3, 6, 4, 1)
    w.write({12: a12, 24: a24}, [6, 4, 5], _metas(0, 3))
    b12, b24 = _rows(2, 3, 4, 2), _rows(2, 3, 4, 3)  # a shorter batch: padded with zeros to 6 tokens
    w.write({12: b12, 24: b24}, [3, 2], _metas(3, 2))
    w.finish({"elapsed_s": 1.0})

    s = features.Store(str(tmp_path))
    assert s.rows == 5 and s.layers == [12, 24] and s.ids == [f"tok{i}" for i in range(5)]
    assert s.lengths.tolist() == [6, 4, 5, 3, 2]
    got = s.batch([4, 0, 3])  # out of file order on purpose
    assert got["feats"].shape == (3, 2, 6, 4)  # cut to the longest row in the batch (6)
    np.testing.assert_array_equal(got["feats"][1, 0], a12[0])
    np.testing.assert_array_equal(got["feats"][1, 1], a24[0])
    np.testing.assert_array_equal(got["feats"][0, 0, :3], b12[1])
    assert not got["feats"][0, :, 3:].any()  # zero padding past the batch's own width
    assert got["mask"].tolist()[0] == [True, True, False, False, False, False]
    assert got["ego"][:, 0].tolist() == [4.0, 0.0, 3.0]
    assert got["target_xy"][:, 0, 0].tolist() == [4.0, 0.0, 3.0]
    assert json.loads((tmp_path / "cache.json").read_text())["complete"] is True


def test_resume_keeps_written_rows_and_drops_a_torn_meta_line(tmp_path):
    w = features.Writer(str(tmp_path), [24], rows=4, tokens=2, width=3, settings={})
    first = _rows(2, 2, 3, 5)
    w.write({24: first}, [2, 2], _metas(0, 2))
    w.meta.write('{"id": "torn"')  # a crash mid-line, after the last full batch
    w.meta.close()

    w2 = features.Writer(str(tmp_path), [24], rows=4, tokens=2, width=3, settings={})
    assert w2.done == 2
    w2.write({24: _rows(2, 2, 3, 6)}, [2, 1], _metas(2, 2))
    w2.finish({})
    s = features.Store(str(tmp_path))
    assert s.ids == ["tok0", "tok1", "tok2", "tok3"]
    np.testing.assert_array_equal(s.batch([0, 1])["feats"][:, 0], first)


def test_a_different_cache_in_the_same_folder_is_refused(tmp_path):
    features.Writer(str(tmp_path), [24], rows=4, tokens=2, width=3, settings={})
    with pytest.raises(SystemExit):
        features.Writer(str(tmp_path), [12], rows=4, tokens=2, width=3, settings={})


def test_an_unfinished_cache_is_not_read_by_default(tmp_path):
    w = features.Writer(str(tmp_path), [24], rows=4, tokens=2, width=3, settings={})
    w.write({24: _rows(1, 2, 3, 7)}, [2], _metas(0, 1))
    with pytest.raises(SystemExit):
        features.Store(str(tmp_path))
    assert features.Store(str(tmp_path), allow_incomplete=True).rows == 1


@pytest.mark.skipif(not hasattr(__import__("os"), "preadv"), reason="positional reads need Linux")
def test_positional_reads_match_the_memmap(tmp_path):
    w = features.Writer(str(tmp_path), [12, 24], rows=6, tokens=5, width=4, settings={})
    w.write({12: _rows(6, 5, 4, 8), 24: _rows(6, 5, 4, 9)}, [5, 3, 4, 5, 2, 1], _metas(0, 6))
    w.finish({})
    s = features.Store(str(tmp_path))
    fast = s.batch([5, 1, 3])
    s.force_memmap = True
    slow = s.batch([5, 1, 3])
    np.testing.assert_array_equal(fast["feats"], slow["feats"])
