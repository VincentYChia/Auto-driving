"""navsim_eval.failure_analysis: maneuver labels and the sign of the along/across-track error."""

import numpy as np

from navsim_eval.failure_analysis import T, errors, maneuver, speed_band


def path(xs, ys, headings):
    return np.stack([xs, ys, headings], axis=1)[None]


def straight(speed):
    return path(speed * T, np.zeros(8), np.zeros(8))


def test_maneuver_labels():
    turn_left = path(5 * T, 0.3 * T ** 2, np.linspace(0.05, 0.6, 8))
    turn_right = path(5 * T, -0.3 * T ** 2, np.linspace(-0.05, -0.6, 8))
    stopped = path(np.zeros(8), np.zeros(8), np.zeros(8))
    start = path(0.5 * T ** 2, np.zeros(8), np.zeros(8))
    brake = path(np.array([2.4, 4.6, 6.5, 8.0, 9.2, 9.9, 10.2, 10.25]), np.zeros(8), np.zeros(8))
    shift = path(8 * T, np.linspace(0.2, 3.5, 8), np.zeros(8))
    humans = np.concatenate([turn_left, turn_right, stopped, start, brake, shift, straight(8)])
    speed = np.array([5, 5, 0, 0, 5, 8, 8], dtype=float)
    assert list(maneuver(humans, speed)) == [
        "turn left (>20 deg)", "turn right (>20 deg)", "stay stopped", "start from a stop", "brake to a stop",
        "sideways shift (>2 m, lane change or curve)", "keep going"]


def test_errors_along_and_across():
    human = straight(5)
    ahead = straight(6)  # ends 4 m further along
    left = human.copy()
    left[0, :, 1] += 1.5
    e = errors(np.concatenate([ahead, left]), np.concatenate([human, human]))
    assert np.allclose(e["along"], [4.0, 0.0]) and np.allclose(e["across"], [0.0, 1.5])
    assert np.allclose(e["fde"], [4.0, 1.5])
    # a human path that ends heading left (90 degrees): ending further along it means further left in ego frame
    turned = path(np.full(8, 5.0), np.linspace(1, 5, 8), np.full(8, np.pi / 2))
    further = turned.copy()
    further[0, -1, 1] += 2.0
    assert np.isclose(errors(further, turned)["along"][0], 2.0)


def test_speed_band_edges():
    assert list(speed_band(np.array([0.0, 0.5, 4.0, 11.9, 20.0]))) == [
        "stopped (<0.5 m/s)", "slow (0.5-4)", "medium (4-8)", "fast (8-12)", "very fast (>12)"]
