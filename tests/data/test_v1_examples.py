import math
import re

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline import labels  # noqa: E402
from data_pipeline.v1 import audit, examples, navsim_splits  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402


def random_future(rng):
    """8 poses at 0.5 s: a random speed profile on an arc, plus a random sideways drift."""
    speeds = np.clip(rng.uniform(0, 20) + np.cumsum(rng.normal(0, 1.5, 8)), 0, None) * rng.choice([1.0, 0.0], p=[0.9, 0.1])
    yaw_rate = rng.choice([0.0, rng.normal(0, 0.15), rng.normal(0, 0.6)])
    s = np.cumsum(speeds * 0.5)
    yaw = yaw_rate * np.arange(1, 9) * 0.5
    x, y = np.cumsum(speeds * 0.5 * np.cos(yaw)), np.cumsum(speeds * 0.5 * np.sin(yaw))
    y = y + rng.choice([0.0, rng.normal(0, 2.0)]) * np.linspace(0, 1, 8)
    return float(max(speeds[0] + rng.normal(0, 1.0), 0.0)), np.column_stack([x, y, yaw]), s


def test_the_reasons_agree_with_the_label_rules():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        speed, future, _ = random_future(rng)
        lon = labels.longitudinal_action(speed, future[:, :2])
        lat = labels.lateral_action(future)
        seen |= {lon, lat}
        ev = examples.longitudinal_evidence(speed, future[:, :2])
        now, ahead, need, stop = ev["speed_now"], ev["speed_ahead"], ev["change_needed"], labels.STOP_SPEED
        expected = {
            "stopped": now < stop and ahead < stop,
            "stop": now >= stop and ahead < stop,
            "start": now < stop <= ahead,
            "accelerate": ahead - now > need,
            "decelerate": now - ahead > need,
            "keep_speed": abs(ahead - now) <= need,
        }
        assert expected[lon], (lon, ev)
        ev = examples.lateral_evidence(future)
        deg, shift = abs(ev["heading_deg"]), ev["shift_m"]
        expected = {
            "stationary": ev["ahead_m"] < labels.MIN_FORWARD_M,
            "turn_left": deg > labels.TURN_DEG, "turn_right": deg > labels.TURN_DEG,
            "lane_change_left": shift > labels.LANE_CHANGE_LATERAL_M and deg < labels.LANE_CHANGE_MAX_DEG,
            "lane_change_right": shift < -labels.LANE_CHANGE_LATERAL_M and deg < labels.LANE_CHANGE_MAX_DEG,
            "curve_left": labels.CURVE_DEG < deg <= labels.TURN_DEG, "curve_right": labels.CURVE_DEG < deg <= labels.TURN_DEG,
            "straight": deg <= labels.CURVE_DEG,
        }
        assert expected[lat], (lat, ev)
        assert examples.why_longitudinal(examples.longitudinal_evidence(speed, future[:, :2]), lon)
        assert examples.why_lateral(examples.lateral_evidence(future), lat)
    assert {"stopped", "stop", "accelerate", "decelerate", "keep_speed", "turn_left", "curve_right", "straight"} <= seen


def test_reasons_on_the_analytic_arc():
    t = np.arange(1, 9) * 0.5
    radius, w = 125.0, 0.08
    future = np.column_stack([radius * np.sin(w * t), radius * (1 - np.cos(w * t)), w * t])
    ev = examples.lateral_evidence(future)
    assert ev["heading_deg"] == pytest.approx(math.degrees(0.32)) and abs(ev["shift_m"]) < 1e-9
    assert "curve" in examples.why_lateral(ev, "curve_left")
    ev = examples.longitudinal_evidence(10.0, future[:, :2])
    assert ev["speed_ahead"] == pytest.approx(10.0, abs=0.01) and ev["change_needed"] == pytest.approx(1.5)
    assert "within ±1.5" in examples.why_longitudinal(ev, "keep_speed")
    assert "within ±2 m" in examples.why_command("straight", True)
    assert "unknown" in examples.why_command("unknown", False)


def test_audit_writes_the_examples_page(tmp_path):
    logs = [v1.make_log("log_a", 40), v1.make_standstill(v1.make_log("log_s", 40)), v1.make_log("log_b", 40)]
    root = v1.write_navsim_root(tmp_path / "navsim", train=["log_a", "log_s"], val=["log_b"])
    out = tmp_path / "out"
    oi.build_index(str(v1.write_logs_dir(tmp_path / "logs", logs)), navsim_splits.load(str(root)), str(out))
    audit.main(["--out", str(out)])
    page = (out / "audit" / "examples.html").read_text()
    assert page.startswith("<!doctype html>") and page.count('<section class="example">') == 3
    assert "keep_speed + curve_left" in page and "stopped + stationary" in page
    assert re.search(r"openscene/log_[a-z]/log_[a-z]_tok\d\d", page)  # per-frame data: this page stays local
    assert "nan" not in page.lower().replace("nanoseconds", "")
    fragment = examples.render([], "Title", "<p>intro</p>", standalone=False)
    assert fragment.startswith("<title>Title</title>") and "<body>" not in fragment
