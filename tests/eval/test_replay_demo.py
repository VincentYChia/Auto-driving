"""navsim_eval.replay_demo: windows, frames, re-plan statistics, road users and page assembly (no NAVSIM)."""

import json
import math
import re
import shutil
import subprocess

import numpy as np
import pytest

from navsim_eval import replay_demo as rd
from navsim_tools import geometry


def test_window_cuts_to_the_history_and_future_a_replan_needs():
    assert rd.window(200, 100, 8.0, 12.0) == (84, 124)  # 16 frames before, 24 after
    assert rd.window(200, 5, 8.0, 12.0) == (3, 29)  # first frame with 1.5 s of history
    assert rd.window(110, 100, 8.0, 12.0) == (84, 101)  # last frame with 4 s of recorded future
    with pytest.raises(ValueError):
        rd.window(200, 2, 8.0, 12.0)  # the centre itself lacks history
    with pytest.raises(ValueError):
        rd.window(200, 195, 8.0, 12.0)  # ... or future


def test_steady_frames_and_command_text():
    assert rd.steady_frames([0, 500_000, 1_000_000, 1_500_000])
    assert rd.steady_frames([0, 510_000, 1_000_000])
    assert not rd.steady_frames([0, 500_000, 2_000_000])  # a gap in the log
    assert rd.command_text([1, 0, 0, 0]) == "left"
    assert rd.command_text([0, 1, 0, 0]) == "straight"
    assert rd.command_text([0, 0, 1, 0]) == "right"
    assert rd.command_text([0, 0, 0, 0]) == "unknown"


def test_plan_in_drive_turns_the_plan_with_the_car():
    plan = [[5.0 * (j + 1), 0.0] for j in range(8)]  # straight ahead in the car's frame
    out = rd.plan_in_drive([10.0, 2.0, math.pi / 2], plan)  # the car faces +y (left) in the drive frame
    np.testing.assert_allclose(out[:, 0], 10.0, atol=1e-9)
    np.testing.assert_allclose(out[:, 1], 2.0 + 5.0 * np.arange(1, 9), atol=1e-9)
    assert rd.plan_in_drive([0, 0, 0], None) is None


def straight_drive(n=20, speed=8.0):
    """Drive-frame ego poses of a car going straight along x at ``speed``, 0.5 s apart."""
    return np.array([[speed * rd.STEP * i, 0.0, 0.0] for i in range(n)])


def test_human_future_is_the_next_eight_poses():
    ego = straight_drive()
    np.testing.assert_allclose(rd.human_future(ego, 3), ego[4:12, :2])
    with pytest.raises(ValueError):
        rd.human_future(ego, 15)


def test_replan_stats_error_and_change():
    ego = straight_drive()
    humans = [rd.human_future(ego, k) for k in range(5)]
    exact = [h.copy() for h in humans]  # plans that are exactly what the human did
    stats = rd.replan_stats(exact, humans)
    assert stats[0]["change"] is None
    for s in stats:
        assert s["ade"] == pytest.approx(0.0) and s["fde"] == pytest.approx(0.0)
    for s in stats[1:]:
        assert s["change"] == pytest.approx(0.0)  # consistent plans over the overlapping 3.5 s
    shifted = [h + [0.0, 1.5] for h in humans]  # every plan 1.5 m to the left, all agreeing with each other
    stats = rd.replan_stats(shifted, humans)
    assert stats[2]["ade"] == pytest.approx(1.5) and stats[2]["fde"] == pytest.approx(1.5)
    assert stats[2]["change"] == pytest.approx(0.0)
    jump = list(exact)
    jump[3] = humans[3] + [0.0, 2.0]  # the model changes its mind at frame 3 and back at frame 4
    stats = rd.replan_stats(jump, humans)
    assert stats[3]["change"] == pytest.approx(2.0) and stats[4]["change"] == pytest.approx(2.0)
    assert stats[2]["change"] == pytest.approx(0.0)
    gap = list(exact)
    gap[2] = None
    stats = rd.replan_stats(gap, humans)
    assert stats[2]["ade"] is None and stats[3]["change"] is None


def test_reaches_measure_from_the_car():
    ego = straight_drive(speed=10.0)
    humans = [rd.human_future(ego, k) for k in range(3)]
    plans = [h * [0.5, 1.0] + [ego[k, 0] * 0.5, 0.0] for k, h in enumerate(humans)]  # half the distance ahead
    plan_r, human_r = rd.reaches(ego[:3], plans, humans)
    assert human_r == pytest.approx([40.0, 40.0, 40.0])
    assert plan_r == pytest.approx([20.0, 20.0, 20.0])
    assert rd.reaches(ego[:1], [None], humans[:1])[0] == [None]


def test_map_frame_covers_the_path_with_whole_pixel_sides():
    pts = np.array([[-50.0, -10.0], [150.0, 30.0]])
    m = rd.map_frame(pts, margin=20.0, px_per_m=5.0, max_px=2400)
    assert (m["cx"], m["cy"]) == pytest.approx((50.0, 10.0))
    # 240 x 80 m at 4 px/m is 960 x 320 px, rounded up to 1000 x 400: the half sizes grow to 125 x 50 m
    assert m["px_per_m"] == 4.0 and (m["height_px"], m["width_px"]) == (1000, 400)
    assert (m["half_x"], m["half_y"]) == (125.0, 50.0)
    # bev_map_png's figure: exactly width_px by height_px pixels at 100 dpi (matplotlib truncates)
    w_in, h_in = m["width_px"] / 100, m["width_px"] * m["half_x"] / m["half_y"] / 100
    assert int(w_in * 100) == m["width_px"] and int(h_in * 100) == m["height_px"]
    big = rd.map_frame(np.array([[0.0, 0.0], [1000.0, 0.0]]), margin=0.0, px_per_m=5.0, max_px=2400)
    assert big["px_per_m"] == 2.0 and big["height_px"] == 2000  # 1,000 m tall within 2,400 px


@pytest.mark.parametrize("length", [37.3, 91.9, 133.7, 250.1, 401.4, 777.7])
def test_map_frame_pixels_are_exact_for_any_drive(length):
    for angle in (0.0, 0.7, 1.4):
        pts = np.array([[0.0, 0.0], [length * math.cos(angle), length * math.sin(angle)]])
        m = rd.map_frame(pts)
        h_in = m["width_px"] * m["half_x"] / m["half_y"] / 100
        assert int(h_in * 100) == m["height_px"] and int(m["width_px"] / 100 * 100) == m["width_px"]
        assert m["half_x"] >= abs(pts[:, 0] - m["cx"]).max() and m["half_y"] >= abs(pts[:, 1] - m["cy"]).max()


def box_in(ego, world, length=4.5, width=1.9):
    """A NAVSIM box (x, y, z, l, w, h, heading) of a global pose, as annotated in the frame of ``ego``."""
    rel = geometry.absolute_to_relative(ego, np.asarray(world, dtype=float)[None])[0]
    return np.array([rel[0], rel[1], 0.4, length, width, 1.6, rel[2]])


def test_drive_tracks_put_every_frame_in_the_drive_frame_and_crop():
    origin = np.array([1000.0, 500.0, 0.4])  # the ego at the scene's planning moment
    ego = [origin + [3.0 * k * math.cos(0.4), 3.0 * k * math.sin(0.4), 0.05 * k] for k in range(-2, 3)]
    parked = np.array([1010.0, 512.0, 1.0])  # fixed in the world
    far = np.array([5000.0, 500.0, 0.0])
    boxes = [np.stack([box_in(e, parked), box_in(e, far)]) for e in ego]
    names = [["vehicle", "vehicle"]] * 5
    tokens = [["p", "f"]] * 5
    frame = rd.map_frame(geometry.absolute_to_relative(origin, np.stack(ego))[:, :2])
    tracks = rd.drive_tracks(ego, boxes, names, tokens, origin, frame)
    assert len(tracks) == 1  # the far car is cropped
    want = geometry.absolute_to_relative(origin, parked[None])[0]
    for p in tracks[0]["p"]:
        np.testing.assert_allclose(p, want, atol=0.01)
    assert tracks[0]["l"] == 4.5 and tracks[0]["w"] == 1.9


def payload_drive(n_replans=5):
    ego = straight_drive(n=n_replans + 11)
    replans = [{"token": f"tok{k}", "t": (k - 2) * rd.STEP, "speed": 8.0, "command": "straight"}
               for k in range(n_replans)]
    return {"label": "test drive", "token": "tok2", "log": "log", "map": "us-ma-boston", "ego": ego.tolist(),
            "replan_offset": 3, "replans": replans, "tracks": [], "map_frame": rd.map_frame(ego[:, :2]),
            "map_src": "data:image/png;base64,"}


def test_drive_payload_turns_local_plans_into_drive_frame_stats():
    d = payload_drive()
    preds = {f"tok{k}": np.array([[4.0 * (j + 1), 0.0] for j in range(8)]) for k in range(5) if k != 1}
    out = rd.drive_payload(d, preds, {"tok0": "data:image/jpeg;base64,"})
    assert out["missing_plans"] == 1 and out["t_first"] == -1.0 and out["t_last"] == 1.0
    f = out["frames"]
    assert f[0]["ade"] == pytest.approx(0.0) and f[1]["plan"] is None and f[0]["cam"] and f[1]["cam"] is None
    assert f[3]["plan"][0] == pytest.approx([d["ego"][6][0] + 4.0, 0.0])  # frame 3 sits at ego[3 + 3]
    assert f[4]["change"] == pytest.approx(0.0) and f[2]["change"] is None
    assert f[0]["reach"] == pytest.approx(32.0) and f[0]["human_reach"] == pytest.approx(32.0)


def test_build_html_embeds_the_payload_once_and_safely():
    d = rd.drive_payload(payload_drive(), {}, {})
    d["label"] = "a </script> label"
    payload = {"name": "m", "drives": [d], "styles": {}, "step": 0.5, "generated": "now", "navtest_pdms": None}
    html = rd.build_html(payload)
    assert "</script> label" not in html
    blob = re.search(r'<script type="application/json" id="replay-data">(.*?)</script>', html, re.S).group(1)
    assert json.loads(blob)["drives"][0]["label"] == "a </script> label"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_template_script_parses(tmp_path):
    text = rd.TEMPLATE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", text, re.S)
    assert len(scripts) == 1
    js = tmp_path / "page.js"
    js.write_text(scripts[0], encoding="utf-8")
    subprocess.run(["node", "--check", str(js)], check=True)
