"""navsim_tools.maneuvers: the shared maneuver labels (navsim_eval's failure analysis and the v1 training sets)."""

import numpy as np
import pytest

from navsim_tools import maneuvers as m

T = np.array(m.HORIZON_S)


def path(xs, ys, headings):
    return np.stack([np.broadcast_to(xs, 8), np.broadcast_to(ys, 8), np.broadcast_to(headings, 8)], axis=1)[None]


def test_each_label_and_the_priority_order():
    cases = [  # (path, speed now, label)
        (path(8 * T, 0.0, 0.0), 8.0, m.KEEP),
        (path(8 * T, np.linspace(0.2, 2.5, 8), 0.0), 8.0, m.SIDEWAYS),
        (path(np.array([2.4, 4.6, 6.5, 8.0, 9.2, 9.9, 10.2, 10.25]), 0.0, 0.0), 5.0, m.BRAKE),
        (path(0.5 * T ** 2, 0.0, 0.0), 0.0, m.START),
        (path(0.0, 0.0, 0.0), 0.0, m.STAY),
        (path(5 * T, 0.3 * T ** 2, np.linspace(0.05, 0.6, 8)), 5.0, m.TURN_LEFT),
        (path(5 * T, -0.3 * T ** 2, np.linspace(-0.05, -0.6, 8)), 5.0, m.TURN_RIGHT),
        # priority: a later rule overrides an earlier one
        (path(np.array([2.4, 4.6, 6.5, 8.0, 9.2, 9.9, 10.2, 10.25]), 3.0, 0.0), 5.0, m.BRAKE),  # brake > sideways
        (path(0.0, 0.0, 0.0), 0.4, m.STAY),  # stopped now: stay, not brake
        (path(0.0, 0.0, 0.0), 2.0, m.BRAKE),  # 2 m/s now and stopped at 4 s: brake
        (path(5 * T, 3.0, np.linspace(0.05, 0.4, 8)), 5.0, m.TURN_LEFT),  # turn > sideways
        (path(0.2 * T, 0.0, np.linspace(0.0, -0.5, 8)), 0.0, m.TURN_RIGHT),  # turn > stay stopped
        (path(8 * T, 0.0, 0.34), 8.0, m.KEEP),  # just under TURN_RAD is not a turn
        (path(8 * T, 0.0, 0.36 + 2 * np.pi), 8.0, m.TURN_LEFT),  # headings are wrapped
    ]
    humans = np.concatenate([c[0] for c in cases])
    speed = np.array([c[1] for c in cases])
    assert list(m.maneuver(humans, speed)) == [c[2] for c in cases]
    assert set(m.LABELS) == {m.KEEP, m.SIDEWAYS, m.BRAKE, m.START, m.STAY, m.TURN_LEFT, m.TURN_RIGHT}


def test_speed_band_edges():
    assert list(m.speed_band(np.array([0.0, 0.49, 0.5, 4.0, 11.9, 12.0]))) == [
        "stopped (<0.5 m/s)", "stopped (<0.5 m/s)", "slow (0.5-4)", "medium (4-8)", "fast (8-12)", "very fast (>12)"]


def test_future_poses_reads_the_eight_target_poses_as_float64():
    offsets = np.arange(-8, 13) * 0.5  # the v1 index's PATH_OFFSETS_S
    path_xyyaw = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
    out = m.future_poses(path_xyyaw, offsets)
    assert out.dtype == np.float64 and out.shape == (2, 8, 3)
    np.testing.assert_array_equal(out, path_xyyaw[:, 9:17, :].astype(np.float64))
    with pytest.raises(ValueError, match="no single pose"):
        m.future_poses(path_xyyaw[:, :12], offsets[:12])


def test_failure_analysis_uses_the_shared_labels():
    from navsim_eval import failure_analysis as fa

    assert fa.maneuver is m.maneuver and fa.speed_band is m.speed_band
    assert (fa.TURN_RAD, fa.SPEED_EDGES, fa.SPEED_NAMES) == (m.TURN_RAD, m.SPEED_EDGES, m.SPEED_NAMES)
