import math

import numpy as np
import pytest

from data_pipeline import labels


def straight(speeds, dt=0.5):
    """Future poses for a straight drive with per-step speeds."""
    x = np.cumsum(np.asarray(speeds) * dt)
    return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])


@pytest.mark.parametrize(
    "v0, speeds, expected",
    [
        (10.0, [10] * 10, "keep_speed"),
        (10.0, [11, 12, 13, 14, 15, 15, 15, 15, 15, 15], "accelerate"),
        (10.0, [9, 8, 7, 6, 5, 5, 5, 5, 5, 5], "decelerate"),
        (5.0, [3, 1, 0, 0, 0, 0, 0, 0, 0, 0], "stop"),
        (0.0, [0] * 10, "stopped"),
        (0.0, [0.5, 1, 1.5, 2, 2.5, 3, 3, 3, 3, 3], "start"),
    ],
)
def test_longitudinal(v0, speeds, expected):
    assert labels.longitudinal_action(v0, straight(speeds)[:, :2]) == expected


def arc(yaw_rate, speed=10.0, n=10, dt=0.5):
    poses = []
    for i in range(1, n + 1):
        t = i * dt
        yaw = yaw_rate * t
        if abs(yaw_rate) < 1e-9:
            x, y = speed * t, 0.0
        else:
            x = speed / yaw_rate * math.sin(yaw)
            y = speed / yaw_rate * (1 - math.cos(yaw))
        poses.append([x, y, yaw])
    return np.array(poses)


def test_lateral_classes():
    assert labels.lateral_action(arc(0.0)) == "straight"
    assert labels.lateral_action(arc(0.25)) == "turn_left"     # ~57 deg at 4 s
    assert labels.lateral_action(arc(-0.25)) == "turn_right"
    assert labels.lateral_action(arc(0.05)) == "curve_left"     # ~11 deg at 4 s
    lane_change = arc(0.0)
    lane_change[:, 1] = np.linspace(0.35, 3.5, 10)
    assert labels.lateral_action(lane_change) == "lane_change_left"
    lane_change[:, 1] *= -1
    assert labels.lateral_action(lane_change) == "lane_change_right"
    assert labels.lateral_action(np.zeros((10, 3))) == "stationary"


def test_describe_command():
    assert labels.describe_command([1, 0, 0, 0]) == "left"
    assert labels.describe_command([0, 0, 1, 0]) == "right"
    assert labels.describe_command([0, 0, 0, 0]) == "unknown"
