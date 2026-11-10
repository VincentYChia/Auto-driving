"""navsim_eval.demo_html: frames of other road users, interpolation, BEV pixels and page assembly (no NAVSIM)."""

import json
import math
import re
import shutil
import subprocess

import numpy as np
import pytest

from navsim_eval import demo_html as dh
from navsim_tools import geometry

SERIES = ["#1baf7a", "#eb6834", "#6250d6", "#e87ba4", "#008300"]


def navsim_box_to_global(box, ego):
    """navsim_scenario_utils.gt_boxes_oriented_box, verbatim logic: rotate by the ego heading, then translate."""
    sin, cos = np.sin(ego[2]), np.cos(ego[2])
    x, y, h = box[0], box[1], box[6]
    return np.array([x * cos - y * sin + ego[0], x * sin + y * cos + ego[1], np.arctan2(np.sin(h + ego[2]),
                                                                                         np.cos(h + ego[2]))])


def box_in(ego, world, length=4.5, width=1.9):
    """A NAVSIM box (x, y, z, l, w, h, heading) of a global pose, as annotated in the frame of ``ego``."""
    rel = geometry.absolute_to_relative(ego, np.asarray(world, dtype=float)[None])[0]
    return np.array([rel[0], rel[1], 0.4, length, width, 1.6, rel[2]])


EGO = [np.array([100.0, 50.0, 0.3]), np.array([104.0, 51.5, 0.45]), np.array([108.0, 53.5, 0.7])]


def test_relative_tracks_puts_every_frame_in_the_first_ego_frame():
    parked = np.array([120.0, 60.0, -0.2])  # fixed in the world: must stay put in the t = 0 frame
    moving = [np.array([90.0 + 3 * k, 45.0, 0.1 * k]) for k in range(3)]
    boxes = [np.stack([box_in(e, parked), box_in(e, moving[k], 0.7, 0.7)]) for k, e in enumerate(EGO)]
    for k, e in enumerate(EGO):  # the boxes really are in NAVSIM's convention
        np.testing.assert_allclose(navsim_box_to_global(boxes[k][0], e), parked, atol=1e-9)
    tracks = dh.relative_tracks(EGO, boxes, [["vehicle", "pedestrian"]] * 3, [["car", "walker"]] * 3)
    assert [t["token"] for t in tracks] == ["car", "walker"]
    car, walker = tracks
    assert car["type"] == "vehicle" and car["length"] == 4.5 and car["width"] == 1.9
    want = geometry.absolute_to_relative(EGO[0], parked[None])[0]
    np.testing.assert_allclose(car["poses"], np.tile(want, (3, 1)), atol=1e-9)
    np.testing.assert_allclose(walker["poses"], geometry.absolute_to_relative(EGO[0], np.stack(moving)), atol=1e-9)


def test_relative_tracks_marks_absent_frames_and_crop_keeps_nearby_tracks():
    near, far = np.array([105.0, 52.0, 0.0]), np.array([300.0, 50.0, 0.0])
    boxes = [np.stack([box_in(EGO[0], near), box_in(EGO[0], far)]), np.zeros((0, 7)), box_in(EGO[2], near)[None]]
    tracks = dh.relative_tracks(EGO, boxes, [["vehicle", "barrier"], [], ["vehicle"]], [["a", "b"], [], ["a"]])
    a, b = tracks
    assert np.isnan(a["poses"][1]).all() and not np.isnan(a["poses"][[0, 2]]).any()
    assert not np.isnan(b["poses"][0]).any() and np.isnan(b["poses"][1:]).all()
    assert [t["token"] for t in dh.crop_tracks(tracks, 32, 32, pad=8)] == ["a"]


def test_pose_at_interpolates_and_handles_tracks_that_come_and_go():
    keys = np.array([[0, 0, 0], [5, 1, 0.4], [10, 4, 1.0]], dtype=float)
    np.testing.assert_allclose(dh.pose_at(keys, 0.5), keys[1])
    np.testing.assert_allclose(dh.pose_at(keys, 0.25), [2.5, 0.5, 0.2])
    np.testing.assert_allclose(dh.pose_at(keys, 0.75), [7.5, 2.5, 0.7])
    np.testing.assert_allclose(dh.pose_at(keys, 1.0), keys[2])
    assert dh.pose_at(keys, 1.05) is None and dh.pose_at(keys, -0.05) is None  # outside its frames: gone
    np.testing.assert_allclose(dh.pose_at(keys, 3 * 0.1 * 5 / 3), keys[1])  # float noise at a key time
    wrap = np.array([[0, 0, 3.0], [1, 0, -3.0]])  # crossing +-pi: turn the short way (through pi), not through 0
    h = dh.pose_at(wrap, 0.25)[2]
    assert abs(abs(h) - math.pi) < 0.01
    gap = np.array([[0, 0, 0], [np.nan] * 3, [2, 0, 0]])  # a missed frame inside a track: bridged
    np.testing.assert_allclose(dh.pose_at(gap, 0.2), [0.4, 0, 0])
    np.testing.assert_allclose(dh.pose_at(gap, 0.5), [1, 0, 0])
    np.testing.assert_allclose(dh.pose_at(gap, 0.8), [1.6, 0, 0])
    late = np.array([[np.nan] * 3, [np.nan] * 3, [1, 0, 0], [2, 0, 0], [np.nan] * 3])  # from 1.0 to 1.5 s only
    assert dh.pose_at(late, 0.95) is None and dh.pose_at(late, 1.55) is None
    np.testing.assert_allclose(dh.pose_at(late, 1.0), [1, 0, 0])
    np.testing.assert_allclose(dh.pose_at(late, 1.25), [1.5, 0, 0])
    np.testing.assert_allclose(dh.pose_at(late, 1.5), [2, 0, 0])
    once = np.full((11, 3), np.nan)
    once[9] = [27.2, -7.0, 0.3]  # logged at 4.5 s only: held there at every time, from t = 0
    for t in (0.0, 0.3, 2.0, 4.0, 5.0):
        np.testing.assert_allclose(dh.pose_at(once, t), once[9])
    assert dh.pose_at(np.full((11, 3), np.nan), 1.0) is None


def navsim_observation_pose(keys, t, step=0.5):
    """navsim metric_cache_processor._interpolate_gt_observation with StateInterpolator, verbatim logic: a track's
    logged (time, x, y, heading) states, heading np.unwrap'ed, interpolated linearly for start <= t <= end and
    normalised; a track logged once is that logged object at every time."""
    keys = np.asarray(keys, dtype=float)
    present = ~np.isnan(keys).any(axis=1)
    times, states = np.flatnonzero(present) * step, keys[present].copy()
    if len(times) == 0:
        return None
    if len(times) == 1:  # interpolator.start_time == interpolator.end_time
        return states[0]
    states[:, 2] = np.unwrap(states[:, 2], axis=0)
    if not times[0] <= t <= times[-1]:
        return None
    out = np.array([np.interp(t, times, states[:, i]) for i in range(3)])
    out[2] = np.arctan2(np.sin(out[2]), np.cos(out[2]))
    return out


def test_pose_at_is_the_observation_navsim_scores_against():
    rng = np.random.default_rng(3)
    grid = np.arange(0, int(5.0 / 0.1) + 1, 1, dtype=float) * 0.1  # NAVSIM's 10 Hz times over its 5 s
    checked = 0
    for _ in range(300):
        keys = np.stack([rng.uniform(-40, 40, 11), rng.uniform(-40, 40, 11), rng.uniform(-math.pi, math.pi, 11)], 1)
        keys[rng.random(11) < rng.choice([0.0, 0.3, 0.8, 0.95])] = np.nan  # gaps, late starts, early ends, singles
        for t in grid:
            want, got = navsim_observation_pose(keys, t), dh.pose_at(keys, t)
            assert (want is None) == (got is None), (keys, t)
            if want is not None:
                np.testing.assert_allclose(got[:2], want[:2], atol=1e-9)
                assert abs(geometry.normalize_angle(got[2] - want[2])) < 1e-9
                checked += 1
    assert checked > 3000


def test_pose_at_hits_every_key_on_a_fine_time_grid():
    keys = dh.with_origin(np.stack([np.arange(1, 9) * 2.0, np.zeros(8), np.linspace(0.1, 0.8, 8)], axis=1))
    grid = np.array([dh.pose_at(keys, 0.1 * j) for j in range(41)])  # 0.1 s steps, as NAVSIM simulates
    np.testing.assert_allclose(grid[::5], keys, atol=1e-12)
    np.testing.assert_allclose(grid[1], [0.4, 0, 0.02], atol=1e-12)


def test_plans_match_what_navsim_scores():
    from navsim_eval.predictions import poses_from_waypoints

    wp = np.stack([np.arange(1, 11) * 2.0, 0.05 * np.arange(1, 11) ** 2], axis=1)  # 10 waypoints, curving left
    plan = dh.model_plan(wp)
    assert plan.shape == (9, 3) and not plan[0].any()
    np.testing.assert_array_equal(plan[1:], poses_from_waypoints(wp, 8))  # exactly the float32 poses NAVSIM got
    assert dh.model_plan(wp[:5]) is None


def test_bev_pixel_matches_navsim_axes_layout():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for half_x, half_y, w, h in [(32, 32, 400, 400), (64, 64, 800, 800), (32, 48, 600, 400)]:
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        # navsim/visualization/plots.py configure_bev_ax, verbatim logic (figure_margin = 2 * half)
        ax.set_aspect("equal")
        ax.set_xlim(-half_y, half_y)
        ax.set_ylim(-half_x, half_x)
        ax.invert_xaxis()
        assert dh.check_bev_axes(ax, half_x, half_y, *fig.canvas.get_width_height()) < 1e-6
        plt.close(fig)
    col, row = dh.bev_pixel(10.0, 5.0, 32, 32, 400, 400)  # 10 m ahead, 5 m left: up and to the left of centre
    assert (col, row) == (pytest.approx(168.75), pytest.approx(137.5))
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-30, 30)  # wrong layout: x not inverted
    ax.set_ylim(-32, 32)
    with pytest.raises(AssertionError):
        dh.check_bev_axes(ax, 32, 32, 400, 400)
    plt.close(fig)


def test_scores_and_texts():
    row = {"score": 0.0, "no_at_fault_collisions": 1.0, "drivable_area_compliance": 0.0, "ego_progress": 0.0,
           "time_to_collision_within_bound": 1.0, "comfort": 1.0, "driving_direction_compliance": 1.0}
    assert dh.score_entry(row, "left the road") == {"pdms": 0.0, "nc": 1.0, "dac": 0.0, "ttc": 1.0, "ep": 0.0,
                                                    "c": 1.0, "zeroed_by": "left the road"}
    assert dh.score_entry(dict(row, score=0.8766), "")["zeroed_by"] is None
    assert dh.score_entry(dict(row, score=0.8766))["pdms"] == 0.877
    assert dh.why_text("reference model far ahead", "A", "B") == "A scores 0.8 or more here and B scores 0."
    assert dh.why_text("left-turn command", "A", "B") == "Picked for its route command: left."
    assert dh.map_label("us-nv-las-vegas-strip") == "Las Vegas" and dh.map_label("elsewhere") == "elsewhere"
    specs = dh.series_specs(["A", "B"], SERIES, "#0b0b0b", "#52514e")
    assert [s["key"] for s in specs] == ["human", "m0", "m1", "cv"]
    assert [s["color"] for s in specs] == ["#0b0b0b", SERIES[0], SERIES[1], "#52514e"]
    assert specs[1]["marker"] != specs[2]["marker"]
    with pytest.raises(ValueError):
        dh.series_specs(list("ABCDEF"), SERIES, "#0b0b0b", "#52514e")


def dummy_payload(name="Model </script><!-- & co"):
    plan = dh.with_origin(np.stack([np.arange(1, 9) * 1.5, np.zeros(8), np.zeros(8)], axis=1))
    sim = np.stack([np.arange(41) * 0.3, 0.01 * np.arange(41), np.zeros(41)], axis=1)  # 0.1 s steps
    track = {"token": "t", "type": "vehicle", "length": 4.5, "width": 1.9,
             "poses": np.array([[5.0, 3.0, 0.0]] * 4 + [[np.nan] * 3] * 7)}
    scene = dh.scene_payload("abcdef0123456789", "random sample", "why", "left", "us-ma-boston", 3.0,
                             "data:image/jpeg;base64,AAAA", {"near": "data:image/png;base64,AAAA"},
                             {"human": plan, "m0": plan, "m1": None}, {"human": None, "m0": None, "m1": None},
                             [track], {"human": sim, "m0": sim})
    return dh.page_payload([scene], dh.series_specs([name, "B"], SERIES, "#0b0b0b", "#52514e", with_cv=False),
                           {"near": dh.view_spec(32, 32, "64 m")},
                           {"length": 5.176, "width": 2.297, "rear_axle_to_center": 1.461, "fill": "#DE7061",
                            "line": "black"}, {"vehicle": {"fill": "#699CDB", "line": "black"}}, [], [], 0, "now")


def embedded_json(page):
    return json.loads(re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', page, re.S).group(1))


def test_build_html_embeds_the_data_safely_and_offline():
    payload = dummy_payload()
    page = dh.build_html(payload)
    assert dh.PLACEHOLDER not in page
    assert page.count("</script>") == 2  # the data cannot close its script element early
    assert embedded_json(page) == json.loads(json.dumps(payload))
    scene = embedded_json(page)["scenes"][0]
    assert scene["agents"][0]["p"][3] == [5.0, 3.0, 0.0] and scene["agents"][0]["p"][4] is None
    assert len(scene["agents"][0]["p"]) == dh.OBS_FRAMES
    assert scene["plans"]["m1"] is None and len(scene["plans"]["m0"]) == 9
    assert len(scene["motion"]["m0"]) == 41 and scene["motion"]["m0"][40] == [12.0, 0.4, 0.0]
    assert "m1" not in scene["motion"] and embedded_json(page)["motion_step"] == dh.MOTION_STEP == 0.1
    lowered = page.lower()
    for remote in ('src="http', "src='http", 'href="http', "href='http", "url(http", "@import", "<link", "fetch("):
        assert remote not in lowered, remote
    with pytest.raises(ValueError):
        dh.build_html(payload, template="<html></html>")
    bad = dummy_payload()
    bad["scenes"][0]["speed"] = float("nan")
    with pytest.raises(ValueError):
        dh.build_html(bad)


def page_js(block):
    """One marked block of the page's script (``/* <block>:begin ... */ ... /* <block>:end */``), for node."""
    source = dh.TEMPLATE.read_text(encoding="utf-8")
    return re.search(r"/\* %s:begin.*?\*/(.*?)/\* %s:end \*/" % (block, block), source, re.S).group(1)


def run_node(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    return json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60,
                                     check=True).stdout)


def test_page_interpolation_matches_python():
    motion = np.stack([np.arange(41) * 0.37, 0.002 * np.arange(41) ** 2, np.linspace(0, 0.6, 41)], axis=1).tolist()
    cases = [  # (keys, step): a track with gaps and a +-pi crossing, one that comes and goes, one logged once,
        # one never present, and a simulated ego motion at 0.1 s
        ([[0, 0, 0], [3, 0.5, 0.4], None, [9, 2, 3.1], [12, 3, -3.1], [15, 4, -2.0], None, None, [20, 5, 0.3], None,
          None], 0.5),
        ([None, None, [1, 2, 0.3], [2, 3, 0.5], None, [4, 3, 1.2], None, None, None, None, None], 0.5),
        ([None] * 9 + [[27.2, -7.0, 0.3], None], 0.5),
        ([None] * 11, 0.5),
        (motion, 0.1),
    ]
    times = [round(0.05 * i, 2) for i in range(0, 101)] + [0.1 * 5 / 1, 1.5 + 1e-12, 4.0 - 1e-12, -0.2, 5.2]
    js = run_node(page_js("poseAt") + "\nconst cases = %s, times = %s;\nconsole.log(JSON.stringify(cases.map("
                  "(c) => times.map((t) => poseAt(c[0], t, c[1])))));" % (json.dumps(cases), json.dumps(times)))
    for (keys, step), got_case in zip(cases, js):
        np_keys = np.array([k if k is not None else [np.nan] * 3 for k in keys], dtype=float)
        for t, got in zip(times, got_case):
            want = dh.pose_at(np_keys, t, step)
            if want is None:
                assert got is None, (keys, t)
            else:
                assert got is not None, (keys, t)
                np.testing.assert_allclose(got, want, atol=1e-9, err_msg=str((keys, t)))


def test_page_step_keys_never_skip_the_nearest_pose():
    times = [0.0, 0.8, 1.0, 1.2, 0.25, 3.9, 1.5 + 2e-16, 1.5 - 2e-16, 2 * 0.1 * 7.5]
    js = run_node(page_js("steps") + "\nconst times = %s;\nconsole.log(JSON.stringify(times.map((t) => "
                  "[nextKeyTime(t, 0.5), prevKeyTime(t, 0.5)])));" % json.dumps(times))
    want = [[0.5, -0.5], [1.0, 0.5], [1.5, 0.5], [1.5, 1.0], [0.5, 0.0], [4.0, 3.5], [2.0, 1.0], [2.0, 1.0],
            [2.0, 1.0]]  # setTime clamps to 0..4 s
    np.testing.assert_allclose(js, want, atol=1e-12)


def test_image_data_uris():
    pytest.importorskip("PIL")
    import base64
    import io

    from PIL import Image

    image = (np.random.default_rng(0).random((108, 192, 3)) * 255).astype(np.uint8)
    uri = dh.jpeg_data_uri(image, width=96)
    assert uri.startswith("data:image/jpeg;base64,")
    assert Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).size == (96, 54)
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    small = dh.png_data_uri(buf.getvalue(), colors=16)
    decoded = Image.open(io.BytesIO(base64.b64decode(small.split(",", 1)[1])))
    assert decoded.mode == "P" and decoded.size == (192, 108)
