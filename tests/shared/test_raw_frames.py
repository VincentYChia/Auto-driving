import math

import numpy as np
import pytest

from navsim_tools import raw_frames
from tests.shared import synthetic_navsim as syn

CURRENT = 3  # NAVSIM: 4 history frames -> current index 3


def test_ego_history_and_future_match_analytic():
    frames = syn.make_frames()
    hist = raw_frames.ego_history_local(frames, CURRENT)
    origin = syn.ego_global(CURRENT * syn.DT)
    for i in range(CURRENT + 1):
        np.testing.assert_allclose(hist[i], syn.to_local(syn.ego_global(i * syn.DT), origin), atol=1e-9)
    np.testing.assert_allclose(hist[-1], 0.0, atol=1e-12)
    fut = raw_frames.future_poses_local(frames, CURRENT, 10)
    assert fut.shape == (10, 3)
    for k in range(10):
        expected = syn.to_local(syn.ego_global((CURRENT + 1 + k) * syn.DT), origin)
        np.testing.assert_allclose(fut[k], expected, atol=1e-9)
    # driving forward on a left arc: x increases, y and yaw turn positive
    assert np.all(np.diff(fut[:, 0]) > 0) and fut[-1, 1] > 0 and fut[-1, 2] > 0


def test_agent_tracks_are_ego_motion_compensated():
    frames = syn.make_frames()
    tracks = raw_frames.agent_tracks(frames, CURRENT, list(range(CURRENT + 1)), max_range=100.0)
    assert set(tracks) == {"track_car", "track_ped"}  # far object filtered by range
    origin = syn.ego_global(CURRENT * syn.DT)
    car = tracks["track_car"]
    assert car["category"] == "vehicle"
    assert car["size_lwh"] == pytest.approx([4.5, 1.9, 1.6])
    for k, state in enumerate(car["states"]):
        expected = syn.to_local(syn.car_global(k * syn.DT), origin)
        assert state[0] == pytest.approx(expected[0], abs=1e-3)
        assert state[1] == pytest.approx(expected[1], abs=1e-3)
        assert state[3] == pytest.approx(expected[2], abs=1e-4)
        # velocity expressed in current-frame axes, not frame k's
        expected_v = syn.rotate(syn.CAR_V, -origin[2])
        assert state[4] == pytest.approx(expected_v[0], abs=1e-3)
        assert state[5] == pytest.approx(expected_v[1], abs=1e-3)


def test_agent_tracks_missing_frames_and_nan_velocity():
    frames = syn.make_frames()
    tracks = raw_frames.agent_tracks(frames, CURRENT, list(range(CURRENT + 1)))
    ped = tracks["track_ped"]["states"]
    assert ped[0] is None and ped[1] is None and ped[2] is not None
    assert ped[3][4] is None and ped[3][5] is None  # NaN velocity -> None
    # the static pedestrian stays at the same current-frame position in every frame it is seen
    assert ped[2][0] == pytest.approx(ped[3][0], abs=1e-3) and ped[2][1] == pytest.approx(ped[3][1], abs=1e-3)
    assert "track_far" in tracks  # no range filter


def test_sensor_fields():
    frames = syn.make_frames()
    assert raw_frames.camera_paths(frames[CURRENT])["CAM_F0"] == "synthetic_log/CAM_F0/0003.jpg"
    calib = raw_frames.camera_calibration(frames[CURRENT])["CAM_F0"]
    assert calib["cam_intrinsic"] == np.eye(3).tolist()
    assert raw_frames.driving_command(frames[0]) == [0, 1, 0, 0]
    assert raw_frames.ego_velocity(frames[0]).tolist() == [syn.V, 0.0]
    assert raw_frames.global_ego_pose(frames[0])[2] == pytest.approx(syn.YAW0)
    assert math.isclose(raw_frames.ego_acceleration(frames[0])[1], syn.V * syn.W)
