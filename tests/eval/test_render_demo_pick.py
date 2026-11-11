"""render_demo --pick hard: scenes by what the human did next, busiest first (no NAVSIM)."""

from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
from navsim_eval import render_demo as rd  # noqa: E402


def _path(turn=0.0, lateral=0.0, travel=20.0):
    t = np.linspace(0.125, 1.0, 8)
    return np.column_stack([travel * t, lateral * t, turn * t])


def test_hard_candidates_pool_by_maneuver_and_add_reference_failures(tmp_path):
    tokens = [f"t{i}" for i in range(6)]
    human = np.stack([_path(turn=0.6), _path(turn=-0.6), _path(lateral=3.0), _path(), _path(turn=0.6), _path(turn=0.6)])
    np.savez(tmp_path / "s.npz", tokens=np.array(tokens), human=human, speed=np.full(6, 5.0))
    ref = pd.DataFrame({"score": [1.0, 0.9, 0.8, 1.0, 0.0, 0.7]}, index=tokens)
    got = rd.hard_candidates(str(tmp_path / "s.npz"), [ref], seed=0)
    by = {}
    for t, label, short in got:
        by.setdefault(short, set()).add(t)
    assert by["left turn"] == {"t0", "t4", "t5"} and by["right turn"] == {"t1"}
    assert by["lane change or curve"] == {"t2"}
    assert "t3" not in {t for t, _, _ in got} or by.get("fails") is None  # keep going is never a candidate
    assert by.get("fails", set()) == set()  # t4 already a left-turn candidate, so not repeated as a failure


def test_pick_hard_takes_the_busiest_of_each_maneuver_and_labels_them():
    cands = [("a", "turn left (>20 deg)", "left turn"), ("b", "turn left (>20 deg)", "left turn"),
             ("c", "turn left (>20 deg)", "left turn"), ("d", "turn left (>20 deg)", "left turn"),
             ("e", "brake to a stop", "brake to a stop"), ("f", "turn right (>20 deg)", "fails")]
    busy = {"a": 3, "b": 9, "c": 9, "d": 1, "e": 4, "f": 7}
    picked = rd.pick_hard(cands, busy)
    assert [t for t, _ in picked] == ["b", "c", "a", "e", "f"]  # 3 left turns busiest first (tie keeps order)
    assert picked[0][1] == "left turn · 9 road users within 30 m"
    assert picked[-1][1].startswith("reference model fails: right turn")
    assert [t for t, _ in rd.pick_hard(cands, {"a": 1})] == ["a"]  # unloaded candidates are skipped


def test_road_users_near_counts_vehicles_pedestrians_and_cyclists_inside_the_radius():
    boxes = np.array([[5, 0, 0, 4, 2, 1, 0], [0, 29, 0, 1, 1, 1, 0], [31, 0, 0, 4, 2, 1, 0], [3, 3, 0, 1, 1, 1, 0]])
    ann = SimpleNamespace(boxes=boxes, names=["vehicle", "pedestrian", "vehicle", "traffic_cone"])
    frames = [SimpleNamespace(annotations=None)] * 3 + [SimpleNamespace(annotations=ann)]
    scene = SimpleNamespace(frames=frames, scene_metadata=SimpleNamespace(num_history_frames=4))
    assert rd.road_users_near(scene) == 2
