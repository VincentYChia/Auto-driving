import math

import numpy as np
import pytest

from navsim_tools import geometry


def navsim_convert_absolute_to_relative(origin, poses):
    """Verbatim logic of NAVSIM's convert_absolute_to_relative_se2_array (pdm_geometry_utils.py)."""
    theta = -origin[2]
    origin_array = np.array([[origin[0], origin[1], origin[2]]], dtype=np.float64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points_rel = poses - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = np.arctan2(np.sin(points_rel[:, 2]), np.cos(points_rel[:, 2]))
    return points_rel


def test_absolute_to_relative_matches_navsim():
    rng = np.random.default_rng(0)
    origin = np.array([12.0, -4.0, 2.7])
    poses = np.column_stack([rng.normal(size=20) * 50, rng.normal(size=20) * 50, rng.uniform(-6, 6, 20)])
    ours = geometry.absolute_to_relative(origin, poses)
    np.testing.assert_allclose(ours, navsim_convert_absolute_to_relative(origin, poses.copy()), atol=1e-12)


def test_relative_absolute_round_trip_and_change_frame():
    rng = np.random.default_rng(1)
    a, b = np.array([3.0, 4.0, 0.5]), np.array([-7.0, 2.0, -2.0])
    local_a = np.column_stack([rng.normal(size=5), rng.normal(size=5), rng.uniform(-3, 3, 5)])
    world = geometry.relative_to_absolute(a, local_a)
    np.testing.assert_allclose(geometry.absolute_to_relative(a, world), local_a, atol=1e-12)
    np.testing.assert_allclose(geometry.change_frame(local_a, a, b), geometry.absolute_to_relative(b, world), atol=1e-12)


def test_yaw_from_quaternion():
    for yaw in (-3.0, -1.0, 0.0, 0.7, 3.1):
        q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        assert geometry.yaw_from_quaternion(q) == pytest.approx(yaw, abs=1e-12)


def test_waypoints_to_poses_headings():
    poses = geometry.waypoints_to_poses([[1, 0], [2, 1], [2, 1.01], [2, 3]])
    assert poses[0, 2] == pytest.approx(0.0)
    assert poses[1, 2] == pytest.approx(math.atan2(1, 1))
    assert poses[2, 2] == pytest.approx(poses[1, 2])  # tiny step keeps heading
    assert poses[3, 2] == pytest.approx(math.pi / 2)


def test_waypoints_to_poses_stationary():
    np.testing.assert_allclose(geometry.waypoints_to_poses(np.zeros((8, 2)))[:, 2], 0.0)


def test_backfill_history():
    hist = np.array([[-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    out = geometry.backfill_history(hist, 10)
    assert out.shape == (10, 2)
    np.testing.assert_allclose(out[:, 0], np.arange(-9.0, 1.0))
    np.testing.assert_allclose(out[-4:], hist)
    np.testing.assert_allclose(geometry.backfill_history(hist[-1:], 3), np.zeros((3, 2)))
    np.testing.assert_allclose(geometry.backfill_history(hist, 2), hist[-2:])
