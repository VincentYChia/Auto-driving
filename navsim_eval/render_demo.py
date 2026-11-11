"""Demonstration figures: several models' planned paths on the same navtest scenes, for showing, not scoring.

Each figure has two panels: the front camera at the planning moment, and NAVSIM's bird's-eye view (map,
other agents, ego) with the human driver's recorded 4 s path and each model's planned 4 s path, labelled
with that model's PDMS on the scene (from its NAVSIM score CSV). NAVSIM scores one 4-second plan per
scene (non-reactive), so a figure shows one plan, not continuous driving.

Scenes are chosen reproducibly from the scenes every model has a prediction and a score for: a seeded
random sample (representative), plus labelled extras (the reference model far ahead of the second, the
second ahead of the reference, the reference failing, a left-turn and a right-turn command).

    python navsim_eval/render_demo.py --out <dir> \\
        --model "0.8B pilot=<predictions dir>=<score csv>" --model "MVLAD-AD=<...>=<...>" ...

With --html <file> (instead of or beside --out) it writes one self-contained page that shows one scene at a time:
the front camera beside the BEV, whose background is NAVSIM's own map drawing and whose road users, ego box and
plans are drawn on top; a 0-4 s slider moves each plan's ego box as NAVSIM's simulated car drove that plan when
scoring it (NAVSIM's own PDM simulator, run here from --metric-cache, the cache the scores came from) while the
other road users replay their logged motion by the rule NAVSIM scores against; each plan has a toggle and its
PDMS with sub-scores (demo_html.py, demo_template.html). The run checks that NAVSIM's scorer gives the simulated
motion the PDMS in the score CSV. --human adds the score CSV of NAVSIM's human agent, so the recorded path shows
its own PDMS.

Run in the NAVSIM env on Anvil (needs OPENSCENE_DATA_ROOT with navsim_logs/test and sensor_blobs/test,
and the nuPlan maps). The images and the page show nuPlan data: write them under $NAVSIM_EXP_ROOT and keep them
out of the repo.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim_eval import demo_html  # noqa: E402
from navsim_eval.predictions import load_predictions  # noqa: E402
from navsim_tools import geometry  # noqa: E402

# model colours from the light chart palette, skipping blue, yellow and red: NAVSIM's BEV already uses them for
# vehicles, static objects and the ego car. Human path in ink, drawn on top; constant velocity in grey dots.
SERIES = ["#1baf7a", "#eb6834", "#6250d6", "#e87ba4", "#008300"]
GREY = "#52514e"
INK = "#0b0b0b"
COMMANDS = ["left", "straight", "right", "unknown"]


def load_scores(path: str):
    import pandas as pd

    d = pd.read_csv(path)
    return d[d.token != "average"].set_index("token")


def failure(row) -> str:
    """The sub-score that zeroed this scene, in words, or '' when the scene scored above 0.

    Only the two multipliers of NAVSIM v1's PDMS can zero a scene: PDMS = NC x DAC x (5 TTC + 2 C + 5 EP) / 12
    (V: exact on all 12,146 constant-velocity rows, 2026-11-09). Driving-direction compliance is reported in the
    CSV but has no weight in v1, so it is not a reason."""
    if float(row["score"]) > 0:
        return ""
    reasons = [("no_at_fault_collisions", "collision"), ("drivable_area_compliance", "left the road")]
    hit = [text for col, text in reasons if col in row and float(row[col]) == 0]
    return ", ".join(hit) or "score 0"


def choose(scores: List, n_random: int, seed: int) -> List[Tuple[str, str]]:
    """(token, why) pairs; scores[0] is the reference model, scores[1] (if given) the comparison."""
    common = sorted(set.intersection(*(set(s.index) for s in scores)))
    rng = np.random.default_rng(seed)
    picked: List[Tuple[str, str]] = [(t, "random sample") for t in rng.choice(common, n_random, replace=False)]
    used = {t for t, _ in picked}
    a = scores[0].loc[common, "score"].astype(float)

    def extra(pool, why, k):
        pool = [t for t in pool if t not in used]
        for t in rng.choice(pool, min(k, len(pool)), replace=False) if pool else []:
            picked.append((t, why))
            used.add(t)

    if len(scores) > 1:
        b = scores[1].loc[common, "score"].astype(float)
        extra(list(a[(a >= 0.8) & (b == 0)].index), "reference model far ahead", 3)
        extra(list((b - a)[(b - a) > 0.3].index), "comparison model ahead", 2)
    extra(list(a[a == 0].index), "reference model fails", 2)
    return picked


# --pick hard [proposed 2026-11-11]: the researcher found the first pages "too many models at once and the drive is
# too simple", so this mode picks scenes by what the human driver did next (navsim_tools.maneuvers, the labels the
# training data and failure_analysis.py use), busiest first: (label, short name, scenes)
HARD_PLAN = [("turn left (>20 deg)", "left turn", 3), ("turn right (>20 deg)", "right turn", 3),
             ("sideways shift (>2 m, lane change or curve)", "lane change or curve", 2),
             ("brake to a stop", "brake to a stop", 2), ("start from a stop", "start from a stop", 1)]
HARD_POOL = 40  # seeded candidates per maneuver; the busiest of them are shown
BUSY_M = 30.0  # road users (vehicles, pedestrians, cyclists) within this distance of the ego car at t = 0 count
ROAD_USERS = ("vehicle", "pedestrian", "bicycle")


def hard_candidates(scenes_npz: str, scores: List, seed: int, fails: int = 1) -> List[Tuple[str, str, str]]:
    """(token, maneuver label, short name) candidates for --pick hard: per HARD_PLAN maneuver a seeded pool of the
    scenes every model was scored on, plus a pool of hard scenes the reference model scored 0 (short name
    "fails"). ``scenes_npz`` is failure_analysis.py's --cache (each navtest scene's recorded path and speed)."""
    from navsim_tools.maneuvers import maneuver

    d = np.load(scenes_npz, allow_pickle=False)
    labels = dict(zip(d["tokens"].tolist(), maneuver(d["human"], d["speed"]).tolist()))
    common = sorted(set(labels) & set.intersection(*(set(s.index) for s in scores)))
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, str, str]] = []
    for label, short, _ in HARD_PLAN:
        pool = [t for t in common if labels[t] == label]
        out += [(t, label, short) for t in rng.choice(pool, min(HARD_POOL, len(pool)), replace=False)] if pool else []
    hard = {label for label, _, _ in HARD_PLAN}
    ref = scores[0]["score"].astype(float)
    pool = [t for t in common if labels[t] in hard and float(ref.loc[t]) == 0 and t not in {c[0] for c in out}]
    if fails and pool:
        out += [(t, labels[t], "fails") for t in rng.choice(pool, min(HARD_POOL, len(pool)), replace=False)]
    return out


def road_users_near(scene, radius: float = BUSY_M) -> int:
    """Vehicles, pedestrians and cyclists within ``radius`` of the ego car at the planning moment (t = 0)."""
    ann = scene.frames[scene.scene_metadata.num_history_frames - 1].annotations
    boxes = np.asarray(ann.boxes, dtype=np.float64).reshape(-1, 7)
    names = np.asarray(ann.names)
    near = np.hypot(boxes[:, 0], boxes[:, 1]) < radius
    return int((near & np.isin(names, ROAD_USERS)).sum())


def pick_hard(candidates: List[Tuple[str, str, str]], busy: Dict[str, int], fails: int = 1) -> List[Tuple[str, str]]:
    """The busiest loaded candidates of each maneuver (HARD_PLAN counts), then ``fails`` failures; ``busy`` maps a
    loaded token to its road-user count (unloaded tokens are skipped). Ties keep the seeded candidate order."""
    picked: List[Tuple[str, str]] = []
    used = set()
    for short, k in [(s, n) for _, s, n in HARD_PLAN] + [("fails", fails)]:
        pool = [(t, label) for t, label, s in candidates if s == short and t in busy and t not in used]
        pool.sort(key=lambda c: -busy[c[0]])  # stable: equal counts keep the seeded order
        for t, label in pool[:k]:
            name = next(s for lab, s, _ in HARD_PLAN if lab == label)
            why = (f"reference model fails: {name}" if short == "fails" else name) + \
                  f" · {busy[t]} road users within {BUSY_M:g} m"
            picked.append((t, why))
            used.add(t)
    return picked


def navsim_styles() -> Dict:
    """NAVSIM's BEV colours for road users and map layers, and the ego footprint it draws (the Pacifica)."""
    from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
    from navsim.visualization.config import AGENT_CONFIG, MAP_LAYER_CONFIG
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.abstract_map import SemanticMapLayer as Layer

    agents = {name: {"fill": AGENT_CONFIG[kind]["fill_color"], "line": AGENT_CONFIG[kind]["line_color"]}
              for name, kind in tracked_object_types.items() if kind in AGENT_CONFIG and name != "ego"}
    static = ["traffic_cone", "barrier", "czone_sign", "generic_object"]
    keys = [dict(agents[n], label=n) for n in ("vehicle", "pedestrian", "bicycle")]
    if len({agents[n]["fill"] for n in static}) == 1:
        keys.append(dict(agents["traffic_cone"], label="static object (cone, barrier, sign, other)"))
    else:
        keys += [dict(agents[n], label=demo_html.TYPE_LABELS[n]) for n in static]
    ego, vehicle = AGENT_CONFIG[TrackedObjectType.EGO], get_pacifica_parameters()
    cfg = MAP_LAYER_CONFIG
    return {
        "agents": agents, "agent_keys": keys,
        "ego": {"length": vehicle.length, "width": vehicle.width, "rear_axle_to_center": vehicle.rear_axle_to_center,
                "fill": ego["fill_color"], "line": ego["line_color"]},
        "map_keys": [
            {"label": "road: lanes and intersections (drivable)", "fill": cfg[Layer.LANE]["fill_color"]},
            {"label": "car park (drivable)", "fill": cfg[Layer.CARPARK_AREA]["fill_color"]},
            {"label": "walkway", "fill": cfg[Layer.WALKWAYS]["fill_color"]},
            {"label": "crosswalk", "fill": cfg[Layer.CROSSWALK]["fill_color"],
             "opacity": cfg[Layer.CROSSWALK]["fill_color_alpha"]},
            {"label": "lane centre line", "line": cfg[Layer.BASELINE_PATHS]["line_color"], "dash": True}],
    }


def bev_map_png(map_api, ego_pose, half_x: float, half_y: float, size_px: int) -> bytes:
    """NAVSIM's own map drawing (map layer only: no road users, no paths) with the axes filling the image.

    NAVSIM reads the view size, and the radius of its map query, from BEV_PLOT_CONFIG["figure_margin"]; it is set
    for the duration of the drawing (at NAVSIM's own 64 m for the default view) and restored afterwards.
    """
    import matplotlib.pyplot as plt
    from navsim.visualization import config as vis
    from navsim.visualization.bev import add_map_to_bev_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax
    from nuplan.common.actor_state.state_representation import StateSE2

    saved = vis.BEV_PLOT_CONFIG["figure_margin"]
    vis.BEV_PLOT_CONFIG["figure_margin"] = (2 * half_x, 2 * half_y)
    try:
        fig = plt.figure(figsize=(size_px / 100, size_px * half_x / half_y / 100), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        add_map_to_bev_ax(ax, map_api, StateSE2(*ego_pose))
        configure_bev_ax(ax)
        configure_ax(ax)
        ax.set_axis_off()
        demo_html.check_bev_axes(ax, half_x, half_y, *fig.canvas.get_width_height())
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=vis.BEV_PLOT_CONFIG["background_color"])
        plt.close(fig)
    finally:
        vis.BEV_PLOT_CONFIG["figure_margin"] = saved
    return buf.getvalue()


def navsim_scoring():
    """NAVSIM's PDM simulator and scorer, built from the scoring config that run_pdm_score uses by default."""
    import navsim
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(navsim.__file__).parent / "planning/script/config/pdm_scoring/"
                         "default_scoring_parameters.yaml")
    return instantiate(cfg.simulator), instantiate(cfg.scorer)


def simulate(scoring, metric_cache, plan: np.ndarray) -> Tuple[np.ndarray, float]:
    """The ego motion NAVSIM scores for ``plan`` ((9, 3) poses at t = 0..4 s in the ego frame, origin first), step
    by step as navsim.evaluate.pdm_score does it: the plan moved to the global frame from the metric cache's
    initial ego state and sampled at 0.1 s, then tracked by the PDM simulator (LQR controller and kinematic bicycle
    model), beside PDM-Closed's trajectory as proposal 0, and scored. Returns the 41 simulated rear-axle poses at
    t = 0, 0.1, ..., 4 s in the ego frame at t = 0, and the PDMS NAVSIM's scorer gives them."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    simulator, scorer = scoring
    init = metric_cache.ego_state
    sampling = simulator.proposal_sampling
    # float64 keeps each agent's values: the models' and constant velocity's float32 poses, the human's float64
    trajectory = Trajectory(np.asarray(plan[1:], dtype=np.float64),
                            TrajectorySampling(num_poses=demo_html.NUM_POSES, interval_length=demo_html.STEP))
    states = np.stack([get_trajectory_as_array(metric_cache.trajectory, sampling, init.time_point),
                       get_trajectory_as_array(transform_trajectory(trajectory, init), sampling, init.time_point)])
    simulated = simulator.simulate_proposals(states, init)
    scores = scorer.score_proposals(simulated, metric_cache.observation, metric_cache.centerline,
                                    metric_cache.route_lane_ids, metric_cache.drivable_area_map)
    ra = init.rear_axle
    return geometry.absolute_to_relative([ra.x, ra.y, ra.heading], simulated[1, :, StateIndex.STATE_SE2]), \
        float(scores[1])


def page_scene(scene, token: str, why: str, command: str, models: List[Dict], cv, human, views: Dict,
               bev_px: int, metric_cache, scoring) -> Tuple[Dict, List[str]]:
    """One scene of the page: camera, map backgrounds, road users over NAVSIM's observation frames (t = 0..5 s),
    every plan, the motion NAVSIM simulated for it, and its scores. Also returns one line per scored plan whose
    simulated motion does not get the CSV's PDMS from NAVSIM's scorer (empty when all agree)."""
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

    n_hist = scene.scene_metadata.num_history_frames
    frames = scene.frames[n_hist - 1: n_hist - 1 + demo_html.OBS_FRAMES]  # t = 0, 0.5, ..., 5 s
    if len(frames) != demo_html.OBS_FRAMES:
        raise ValueError(f"{token}: {len(frames)} frames from t = 0, NAVSIM's observation needs "
                         f"{demo_html.OBS_FRAMES} (load scenes with num_future_frames >= 10)")
    now = frames[0]
    ra = metric_cache.ego_state.rear_axle
    off = np.asarray(now.ego_status.ego_pose, dtype=np.float64) - [ra.x, ra.y, ra.heading]
    if max(abs(off[0]), abs(off[1]), abs(geometry.normalize_angle(off[2]))) > 1e-3:
        raise ValueError(f"{token}: the metric cache's ego state is not the scene's ego pose at t = 0 ({off})")
    image = now.cameras.cam_f0.image
    images = {name: demo_html.png_data_uri(bev_map_png(scene.map_api, now.ego_status.ego_pose, v["half_x"],
                                                       v["half_y"], bev_px), colors=128)
              for name, v in views.items()}
    tracks = demo_html.relative_tracks([f.ego_status.ego_pose for f in frames], [f.annotations.boxes for f in frames],
                                       [f.annotations.names for f in frames],
                                       [f.annotations.track_tokens for f in frames])
    widest = max(views.values(), key=lambda v: v["half_y"])
    tracks = demo_html.crop_tracks(tracks, widest["half_x"], widest["half_y"])
    agent_input = scene.get_agent_input()
    speed = float(np.linalg.norm(np.asarray(agent_input.ego_statuses[-1].ego_velocity, dtype=float)[:2]))

    rows = {}

    def scored(key, table):
        if table is None or token not in table.index:
            return None
        rows[key] = row = table.loc[token]
        return demo_html.score_entry(row, failure(row))

    # each plan exactly as the agent that was scored produces it: HumanAgent, PrecomputedTrajectoryAgent, and
    # NAVSIM's ConstantVelocityAgent
    plans = {"human": demo_html.with_origin(scene.get_future_trajectory(demo_html.NUM_POSES).poses[:, :3])}
    scores = {"human": scored("human", human)}
    for i, m in enumerate(models):
        wp = m["pred"].get(token)
        plans[f"m{i}"] = demo_html.model_plan(wp) if wp is not None else None
        scores[f"m{i}"] = scored(f"m{i}", m["score"])
    if cv is not None:
        plans["cv"] = demo_html.with_origin(ConstantVelocityAgent().compute_trajectory(agent_input).poses)
        scores["cv"] = scored("cv", cv)
    motion, disagree = {}, []
    for key, plan in plans.items():
        if plan is None:
            continue
        motion[key], pdms = simulate(scoring, metric_cache, plan)
        if key in rows and abs(pdms - float(rows[key]["score"])) > 1e-6:
            disagree.append(f"{key}: NAVSIM's scorer gives the simulated motion PDMS {pdms:.4f}, the CSV "
                            f"{float(rows[key]['score']):.4f}")
    page = demo_html.scene_payload(
        token, why, demo_html.why_text(why, models[0]["name"], models[1]["name"] if len(models) > 1 else ""), command,
        scene.scene_metadata.map_name, speed, demo_html.jpeg_data_uri(image) if image is not None else None,
        images, plans, scores, tracks, motion)
    return page, disagree


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=None, help="directory for the PNG figures and index.json")
    p.add_argument("--html", default=None, help="write the interactive page (one self-contained file) here")
    p.add_argument("--model", action="append", required=True,
                   help='"name=predictions dir or file=score csv"; the first is the reference, the second the comparison')
    p.add_argument("--cv", default=None, help="score csv of NAVSIM's constant-velocity agent: also draw its path "
                   "(the current speed held straight ahead, as that agent plans)")
    p.add_argument("--human", default=None, help="page only: score csv of NAVSIM's human agent (the recorded path)")
    p.add_argument("--metric-cache", default=None,
                   help="page only, required with --html: NAVSIM's metric cache that the score csvs were computed "
                   "from; the page moves each plan's ego box as NAVSIM's simulator drives it from that cache")
    p.add_argument("--random", type=int, default=6)
    p.add_argument("--turns", type=int, default=1, help="extra scenes per turn command (left, right)")
    p.add_argument("--pick", choices=("mixed", "hard"), default="mixed",
                   help="mixed: a seeded random sample plus labelled extras (--random, --turns); hard: scenes by what "
                   "the human did next (left and right turns, lane changes and curves, braking to a stop, starting "
                   "off), the busiest of a seeded pool each, plus one the reference model fails (needs --scenes)")
    p.add_argument("--scenes", default=None,
                   help="--pick hard: failure_analysis.py's --cache npz (each navtest scene's recorded path and speed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--bev-px", type=int, default=800, help="page only: pixel width of each map background")
    args = p.parse_args()
    if not (args.out or args.html):
        p.error("give --out (PNG figures), --html (interactive page) or both")
    if args.html and not args.metric_cache:
        p.error("--html needs --metric-cache: the page shows the ego motion NAVSIM simulated and scored")
    if args.pick == "hard" and not args.scenes:
        p.error("--pick hard needs --scenes (failure_analysis.py's --cache npz)")
    if args.html:
        series = demo_html.series_specs([s.split("=", 1)[0] for s in args.model], SERIES, INK, GREY,
                                        with_cv=args.cv is not None)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax

    models = []
    for spec in args.model:
        name, pred, csv = spec.split("=", 2)
        models.append({"name": name, "pred": load_predictions(pred), "score": load_scores(csv)})
    hard = args.pick == "hard"
    candidates = hard_candidates(args.scenes, [m["score"] for m in models], args.seed) if hard else []
    picked = [] if hard else choose([m["score"] for m in models], args.random, args.seed)
    cv = load_scores(args.cv) if args.cv else None

    data = Path(os.environ["OPENSCENE_DATA_ROOT"])
    no_cam = dict(cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False,
                  cam_b0=False, lidar_pc=False)
    # turn commands need the agent input, so pick them after loading a pool of candidate scenes
    pool_tokens = [t for t, _ in picked] + [t for t, _, _ in candidates]
    if args.turns and not hard:
        common = sorted(set.intersection(*(set(m["score"].index) for m in models)))
        rng = np.random.default_rng(args.seed + 1)
        pool_tokens += [t for t in rng.choice(common, 400, replace=False) if t not in pool_tokens]
    loader = SceneLoader(data / "navsim_logs" / "test", data / "sensor_blobs" / "test",
                         SceneFilter(tokens=pool_tokens, frame_interval=1), SensorConfig(cam_f0=[3], **no_cam))
    available = set(loader.tokens)  # frame_interval=1: navtest tokens sit on every frame, not a 14-frame grid
    print(f"[render] {len(available)} of {len(pool_tokens)} candidate scenes loaded", flush=True)
    picked = [(t, why) for t, why in picked if t in available]

    def command_of(scene) -> str:
        c = np.asarray(scene.get_agent_input().ego_statuses[-1].driving_command).reshape(-1)
        return COMMANDS[int(np.argmax(c))] if c.size == 4 and c.max() > 0 else "unknown"

    if hard:
        busy = {t: road_users_near(loader.get_scene_from_token(t)) for t in dict.fromkeys(pool_tokens) if t in available}
        picked = pick_hard(candidates, busy)
        print(f"[render] --pick hard: {len(picked)} scenes from {len(busy)} loaded candidates", flush=True)
    elif args.turns:
        want = {"left": args.turns, "right": args.turns}
        for t in pool_tokens[len(picked):]:
            if t not in available or t in {x for x, _ in picked} or not any(want.values()):
                continue
            cmd = command_of(loader.get_scene_from_token(t))
            if want.get(cmd, 0) > 0:
                picked.append((t, f"{cmd}-turn command"))
                want[cmd] -= 1

    if args.html:
        from navsim.common.dataloader import MetricCacheLoader
        from navsim.visualization.config import BEV_PLOT_CONFIG

        human_scores = load_scores(args.human) if args.human else None
        styles = navsim_styles()
        margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]  # NAVSIM's view: 64 m square
        views = {"near": demo_html.view_spec(margin_x / 2, margin_y / 2, f"{margin_y:g} m (NAVSIM's view)"),
                 "wide": demo_html.view_spec(margin_x, margin_y, f"{2 * margin_y:g} m")}
        caches, scoring = MetricCacheLoader(Path(args.metric_cache)), navsim_scoring()
        missing = [t for t, _ in picked if t not in caches.metric_cache_paths]
        if missing:
            raise SystemExit(f"{len(missing)} chosen scenes have no metric cache in {args.metric_cache}: {missing[:3]}")
        pages, disagree, n_scored = [], [], 0
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    index = []
    for k, (token, why) in enumerate(picked):
        scene = loader.get_scene_from_token(token)
        if args.html:
            page, off = page_scene(scene, token, why, command_of(scene), models, cv, human_scores, views,
                                   args.bev_px, caches.get_from_token(token), scoring)
            pages.append(page)
            scored = sum(v is not None and page["motion"].get(key) is not None for key, v in page["scores"].items())
            n_scored += scored
            disagree += [f"scene {k + 1:02d} {token[:8]} {line}" for line in off]
            print(f"[render] page scene {k + 1:02d} {token[:8]}: {why}, {len(page['agents'])} road users; NAVSIM's "
                  f"simulated motion gets the CSV's PDMS for {scored - len(off)} of {scored} scored plans", flush=True)
        if not args.out:
            continue
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.55, 1]})
        fig.patch.set_facecolor("white")
        image = frame.cameras.cam_f0.image
        if image is not None:
            ax_cam.imshow(image)
        configure_ax(ax_cam)
        ax_cam.set_title("Front camera at the planning moment", fontsize=10, color=INK, loc="left")

        add_configured_bev_on_ax(ax_bev, scene.map_api, frame)
        human = scene.get_future_trajectory(8).poses[:, :2]  # the 4 s the models plan
        line = np.vstack([[0, 0], human])
        ax_bev.plot(line[:, 1], line[:, 0], color=INK, linestyle="--", linewidth=1.6, zorder=20,
                    label="Human driver (recorded)")
        entry = {"token": token, "why": why, "command": command_of(scene), "map": scene.scene_metadata.map_name,
                 "models": {}}
        for i, m in enumerate(models):
            wp = m["pred"].get(token)
            row = m["score"].loc[token]
            pdms, why_zero = float(row["score"]), failure(row)
            entry["models"][m["name"]] = {"pdms": round(pdms, 3), "zeroed_by": why_zero or None}
            if wp is None:
                continue
            path = np.vstack([[0, 0], np.asarray(wp, dtype=float)[:8, :2]])
            label = f"{m['name']}: PDMS {pdms:.2f}" + (f" ({why_zero})" if why_zero else "")
            ax_bev.plot(path[:, 1], path[:, 0], color=SERIES[i % len(SERIES)], linewidth=3.0 if i == 0 else 2.2,
                        marker="o", markersize=4 if i == 0 else 3, zorder=15 - i, label=label)  # first model on top
        if cv is not None and token in cv.index:
            v = np.asarray(scene.get_agent_input().ego_statuses[-1].ego_velocity, dtype=float)[:2]
            t = 0.5 * np.arange(1, 9)
            path = np.vstack([[0, 0], np.stack([np.linalg.norm(v) * t, 0 * t], axis=1)])
            row = cv.loc[token]
            pdms, why_zero = float(row["score"]), failure(row)
            entry["models"]["Constant velocity"] = {"pdms": round(pdms, 3), "zeroed_by": why_zero or None}
            label = f"Constant velocity: PDMS {pdms:.2f}" + (f" ({why_zero})" if why_zero else "")
            ax_bev.plot(path[:, 1], path[:, 0], color=GREY, linewidth=1.8, linestyle=":", zorder=4, label=label)
        configure_bev_ax(ax_bev)
        configure_ax(ax_bev)
        ax_bev.set_title("Bird's-eye view, 4 s plans (ego at centre, facing up)", fontsize=10, color=INK, loc="left")
        ax_bev.legend(loc="lower left", fontsize=8, framealpha=0.92)
        fig.suptitle(f"navtest scene {token[:8]}  ·  {scene.scene_metadata.map_name}  ·  route command: "
                     f"{entry['command']}  ·  {why}", fontsize=11, color=INK, x=0.01, ha="left")
        fig.text(0.01, 0.01, "PDMS: NAVSIM's 0-1 driving score for this 4 s plan (collisions, staying on the road, "
                 "safe distance, progress, comfort); 1 is perfect.", fontsize=8, color="#52514e")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        name = f"{k + 1:02d}_{token[:8]}.png"
        fig.savefig(os.path.join(args.out, name), dpi=args.dpi, facecolor="white")
        plt.close(fig)
        entry["file"] = name
        index.append(entry)
        print(f"[render] {name}: {why}", flush=True)
    if args.out:
        with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"models": [m["name"] for m in models], "scenes": index}, f, indent=1)
    if args.html:
        payload = demo_html.page_payload(
            pages, series, views, styles["ego"], styles["agents"], styles["agent_keys"], styles["map_keys"],
            seed=args.seed, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), default_view="near")
        size = demo_html.write_html(args.html, payload)
        print(f"[render] wrote {args.html}: {len(pages)} scenes, {size / 1e6:.1f} MB", flush=True)
        print(f"[render] check: NAVSIM's scorer reproduces the CSV PDMS from the simulated motion for "
              f"{n_scored - len(disagree)} of {n_scored} scored plans", flush=True)
        for line in disagree:
            print(f"[render] WARNING {line}: the page's boxes may not show what was scored", flush=True)


if __name__ == "__main__":
    main()
