"""R14's boundary paths (an ablation): perturbations of the logged path, the 4 kept on both sides of the NC / DAC
limits, and training the selector with them as extra training-only candidates."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HEADS = HERE.parents[1] / "pilot_sft" / "heads"
sys.path.insert(0, str(HEADS))
sys.path.insert(0, str(HERE))

import boundary_paths as bp  # noqa: E402


def straight(step=2.0):
    return np.array([[step * (i + 1), 0.0] for i in range(8)])


def test_lateral_offset_grows_to_d_at_4_s_along_the_left_normal():
    p = bp.lateral(straight(), 2.0)
    assert np.allclose(p[:, 0], straight()[:, 0]) and np.allclose(p[:, 1], 2.0 * np.arange(1, 9) / 8)
    north = np.array([[0.0, 1.5 * (i + 1)] for i in range(8)])  # driving along +y: left is -x
    q = bp.lateral(north, 1.0)
    assert np.allclose(q[:, 0], -np.arange(1, 9) / 8) and np.allclose(q[:, 1], north[:, 1])
    assert np.allclose(bp.lateral(np.zeros((8, 2)), -1.0)[:, 1], -np.arange(1, 9) / 8)  # stopped: +x is forward


def test_longitudinal_moves_each_waypoint_along_the_path():
    assert np.allclose(bp.longitudinal(straight(), 0.5), straight(1.0))
    assert np.allclose(bp.longitudinal(straight(), 1.4), straight(2.8))  # beyond the end: straight on
    bend = np.array([[1.0, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]])
    b = bp.longitudinal(bend, 0.5)  # arc lengths 1..8 halved: 0.5..4, all on the first leg
    assert np.allclose(b, [[0.5 * (i + 1), 0] for i in range(8)])
    assert np.allclose(bp.longitudinal(np.zeros((8, 2)), 1.2), 0.0)
    assert len(bp.NAMES) == 16 and len(set(bp.NAMES)) == 16


def res(fail=()):
    ok = {"no_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0, "score": 0.9}
    bad = {"no_at_fault_collisions": 1.0, "drivable_area_compliance": 0.0, "score": 0.0}
    return {n: (bad if n in fail else ok) for n in bp.NAMES}


def test_pick_keeps_the_pair_across_the_nearest_limit():
    # left fails from 1.5 m, right from 3 m; faster fails at once (1.2), slower never
    r = res({"lat_p1.5", "lat_p2", "lat_p3", "lat_p4", "lat_m3", "lat_m4", "lon_1.2", "lon_1.4"})
    assert bp.pick(r) == ["lat_p1", "lat_p1.5", "lon_1.2", "lon_1.4"]
    assert bp.pick(res()) == ["lat_p4", "lat_m4", "lon_0.6", "lon_1.4"]  # no limit found: the outermost
    r = res({"lat_m2", "lat_m3", "lat_m4", "lat_p2", "lat_p3", "lat_p4", "lon_0.6"})  # a tie: the left side
    assert bp.pick(r)[:2] == ["lat_p1.5", "lat_p2"] and bp.pick(r)[2:] == ["lon_0.8", "lon_0.6"]
    r = res()
    r["lat_m0.5"] = {"score": 0.0, "error": "x"}  # an unscorable path counts as failing
    assert bp.pick(r)[:2] == ["lat_m0.5", "lat_m1"]


def test_make_pick_and_train_with_extra_candidates(tmp_path):
    torch = pytest.importorskip("torch")
    import selector as sl
    from test_head_selector import fake_subscores, make_candidates, make_store, run

    tokens = make_store(tmp_path / "cache" / "train")
    (tmp_path / "rl.txt").write_text("\n".join(tokens[:-2]) + "\n")  # two scenes without RL scenes' boundary paths
    run("boundary_paths.py", "make", "--cache", str(tmp_path / "cache" / "train"), "--only-tokens",
        str(tmp_path / "rl.txt"), "--out", str(tmp_path / "pert"))
    rows = [json.loads(l) for l in (tmp_path / "pert" / "lat_p1" / "shard_000_of_001.jsonl").read_text().splitlines()]
    assert [r["token"] for r in rows] == tokens[:-2]
    lines = []
    for t in tokens[:-2]:
        res_ = []
        for n in bp.NAMES:
            w = [json.loads(l) for l in (tmp_path / "pert" / n / "shard_000_of_001.jsonl").read_text().splitlines()
                 if json.loads(l)["token"] == t][0]["waypoints"]
            s = fake_subscores(np.asarray(w))
            res_.append({**s, "score": float(sl.compose(np.array([s[k] for k in sl.SUBS])))})
        lines.append(json.dumps({"token": t, "names": bp.NAMES, "results": res_}) + "\n")
    (tmp_path / "pert_scores.jsonl").write_text("".join(lines))
    out = run("boundary_paths.py", "pick", "--paths", str(tmp_path / "pert"), "--scores", str(tmp_path / "pert_scores.jsonl"),
              "--out", str(tmp_path / "boundary"))
    assert json.loads(out.splitlines()[-1])["scenes"] == len(tokens) - 2
    picked = [json.loads(l) for l in (tmp_path / "boundary" / "scores.jsonl").read_text().splitlines()]
    assert all(p["names"] == [f"boundary_{j}" for j in range(4)] and len(p["perturbations"]) == 4 for p in picked)

    gen = tmp_path / "gen.pt"
    torch.save({"config": {"layers": [12, 24], "max_tokens": 6}, "stats": {"mean": torch.zeros(16), "std": torch.ones(16)}}, gen)
    steps = tmp_path / "gen" / "navtrain_rl_10" / "steps2"
    _, _, cl = make_candidates(steps, tokens)
    (tmp_path / "scores.jsonl").write_text("".join(cl))
    sel = tmp_path / "sel"
    run("train_selector.py", "--cache", str(tmp_path / "cache" / "train"), "--generator", str(gen), "--candidates",
        str(steps), str(tmp_path / "scores.jsonl"), "--extra-candidates", str(tmp_path / "boundary"),
        str(tmp_path / "boundary" / "scores.jsonl"), "--scale", "2", "--out", str(sel), "--width", "32", "--depth", "2",
        "--heads", "4", "--batch", "4", "--warmup", "2", "--workers", "1", "--heldout-drives", "1", "--epochs", "1",
        "--imitation", "0.1", "--listwise", "0.25")
    runj = json.loads((sel / "run.json").read_text())
    assert runj["extra_candidates_per_scene"] == 4 and runj["candidates_per_scene"] == 7
    assert runj["extra_sources"][0]["scenes"] == len(tokens) - 2
    assert json.loads((sel / "choice.json").read_text())["epochs_done"] == 1
