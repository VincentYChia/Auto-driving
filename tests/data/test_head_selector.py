"""The selector (docs/HEADS_PLAN.md R14): NAVSIM v1.1's composition, per-step kinematics, candidates scored alone,
the masked EP term, the own-pick margin, the measures, and training plus selection end to end on CPU."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HEADS = Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"
sys.path.insert(0, str(HEADS))

import drive_map  # noqa: E402
import features  # noqa: E402
import selector as sl  # noqa: E402

LOGS = ["2021.05.12.19.36.12_veh-35_00005_00204", "2021.06.07.12.54.00_veh-38_01100_01500",
        "2021.07.24.20.37.45_veh-17_00015_00375"]


def test_compose_is_navsim_v11():
    p = np.array([[1, 1, 1, 1, 1], [0.5, 1, 1, 0.8, 1], [1, 0, 1, 1, 1], [1, 1, 0, 0.5, 0]], dtype=np.float64)
    want = p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12
    assert np.allclose(sl.compose(p), want)
    assert np.allclose(sl.compose(torch.from_numpy(p)).numpy(), want)
    assert sl.compose(p)[1] == pytest.approx(0.5 * 11 / 12)


def test_kinematics_of_a_straight_path_and_a_stop():
    xy = torch.tensor([[5.0 * (i + 1), 0.0] for i in range(8)])  # 10 m/s straight ahead
    k = sl.kinematics(xy[None], torch.tensor([8.0]))[0]
    assert torch.allclose(k[:, 0], torch.zeros(8)) and torch.allclose(k[:, 1], torch.full((8,), 10.0))
    assert k[0, 2] == pytest.approx(4.0) and torch.allclose(k[1:, 2], torch.zeros(7))
    left = sl.kinematics(torch.tensor([[[0.0, 1.0 * (i + 1)] for i in range(8)]]))[0]
    assert torch.allclose(left[:, 0], torch.full((8,), np.pi / 2))
    stop = sl.kinematics(torch.zeros(1, 8, 2))[0]
    assert torch.count_nonzero(stop) == 0  # a stopped path: no heading, no speed


def tiny(**kw):
    cfg = sl.SelectorConfig(layers=[12, 24], d_in=16, width=32, depth=2, heads=4, max_tokens=8, **kw)
    torch.manual_seed(0)
    m = sl.Selector(cfg, {"mean": torch.zeros(16), "std": torch.ones(16)})
    for prm in m.parameters():  # the zero-initialised AdaLN gates would hide any dependence on the scene
        torch.nn.init.normal_(prm, std=0.2)
    return m.eval()


def scene_inputs(b=2, k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, 2, 6, 16, generator=g).to(torch.bfloat16)
    mask = torch.ones(b, 6, dtype=torch.bool)
    mask[1, 4:] = False
    ego = torch.randn(b, features.EGO_DIM, generator=g)
    paths = torch.cumsum(torch.rand(b, k, 8, 2, generator=g) * torch.tensor([2.0, 0.3]), dim=2)
    return feats, mask, ego, paths


def test_each_candidate_is_scored_alone():
    m = tiny()
    feats, mask, ego, paths = scene_inputs()
    with torch.no_grad():
        out = m(feats, mask, ego, paths)
        perm = torch.tensor([3, 0, 4, 1, 2])
        assert torch.allclose(m(feats, mask, ego, paths[:, perm]), out[:, perm], atol=1e-5)
        assert torch.allclose(m(feats, mask, ego, paths[:, :2]), out[:, :2], atol=1e-5)  # others absent
        assert not torch.allclose(m(feats, mask, ego.flip(0), paths), out, atol=1e-3)  # the ego state matters
        f2 = feats.clone()
        f2[0, :, :3] = 0
        assert not torch.allclose(m(f2, mask, ego, paths)[0], out[0], atol=1e-4)  # the scene matters
    assert out.shape == (2, 5, 5)


def test_loss_leaves_out_ep_where_nc_or_dac_fail_and_unscored_candidates():
    cfg = sl.SelectorConfig()
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    sub = torch.tensor([[[0.0, 1, 1, 0.0, 1], [1, 1, 1, 0.7, 1], [0.5, 1, 0, 0.3, 1], [1, 1, 1, 0.9, 1]]])
    pdms = sl.compose(sub)
    valid = torch.tensor([[True, True, True, False]])
    out = sl.loss(logits, sub, pdms, valid, cfg)
    out["loss"].backward()
    g = logits.grad[0]
    assert g[0, 3] == 0 and g[1, 3] != 0 and g[2, 3] != 0  # EP: out where NC x DAC = 0, in at NC = 1/2
    assert torch.count_nonzero(g[3]) == 0  # an unscored candidate adds nothing
    assert g[2, 0] == 0  # NC's soft target 1/2 at logit 0 (p = 1/2): zero gradient
    assert set(out) >= {"loss", "bce_NC", "bce_DAC", "bce_TTC", "bce_EP", "bce_C"}


def test_ablation_terms():
    feats, mask, ego, paths = scene_inputs()
    sub = torch.rand(2, 5, 5)
    valid = torch.ones(2, 5, dtype=torch.bool)
    ade = torch.rand(2, 5) * 3
    for kw in ({"listwise": 0.25}, {"imitation": 0.1}, {"target": "pdms"}):
        m = tiny(**kw).train()
        logits = m(feats, mask, ego, paths)
        assert logits.shape[-1] == m.cfg.n_out
        out = sl.loss(logits, sub, sl.compose(sub), valid, m.cfg, ade)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        pr = sl.predicted(logits.detach(), m.cfg)
        assert pr["score"].shape == (2, 5) and bool(((pr["score"] >= 0) & (pr["score"] <= 1)).all())
    assert "listwise" in sl.loss(tiny(listwise=0.25)(feats, mask, ego, paths), sub, sl.compose(sub), valid,
                                 sl.SelectorConfig(listwise=0.25))


def test_select_keeps_the_own_pick_unless_another_is_more_than_delta_higher():
    s = np.array([[0.80, 0.82, 0.79], [0.5, 0.9, 0.9], [0.7, 0.7, 0.6]])
    assert sl.select(s, 0.03).tolist() == [0, 1, 0]
    assert sl.select(s, 0.0).tolist() == [1, 1, 0]  # a tie with the own pick keeps it
    true = np.array([[0.9, 0.5, 0.5], [0.2, 0.8, 0.9], [0.6, 0.6, 0.6]])
    ch = sl.choose_delta(s, true, deltas=(0.0, 0.03, 0.5))
    assert ch["delta"] == 0.03 and ch["selected_pdms"] == pytest.approx((0.9 + 0.8 + 0.6) / 3)
    flat = sl.choose_delta(np.zeros((3, 3)), true, deltas=(0.0, 0.1))
    assert flat["delta"] == 0.1  # a tie goes to the larger margin


def test_report_measures():
    true = np.array([[0.5, 1.0, 0.0], [0.9, 0.9, 0.2], [0.7, 0.0, 0.8], [0.0, 0.6, 0.6]])
    sub = np.ones((4, 3, 5))
    sub[0, 2, 1] = 0  # candidate 2 of scene 0 leaves the drivable area
    sub[2, 1, 0] = 0
    sub[3, 0, 0] = 0
    score = np.array([[0.4, 0.9, 0.1], [0.8, 0.8, 0.1], [0.6, 0.9, 0.5], [0.1, 0.5, 0.7]])
    chosen = sl.select(score, 0.0)
    assert chosen.tolist() == [1, 0, 1, 2]
    r = sl.report(true, sub, score, chosen, ["a", "a", "b", "b"], paths=np.random.default_rng(0).normal(size=(4, 3, 8, 2)))
    assert r["selected_pdms"] == pytest.approx((1.0 + 0.9 + 0.0 + 0.6) / 4)
    assert r["own_pick_pdms"] == pytest.approx((0.5 + 0.9 + 0.7 + 0.0) / 4)
    assert r["best_of_k_pdms"] == pytest.approx((1.0 + 0.9 + 0.8 + 0.6) / 4)
    assert r["switch_rate"] == 0.75 and r["switches"] == {"wins": 2, "losses": 1, "ties": 0}
    assert r["picks_failing_nc_or_dac"] == {"selected": 1, "own_pick": 1}
    assert r["top1_exact"] == 0.5 and r["top1_tie_aware"] == 0.75
    assert r["share_of_best_of_k_gap_closed"] == pytest.approx(0.1 / 0.3)  # mean gain 0.1 of a mean gap of 0.3
    assert sum(b["scenes"] for b in r["gain_by_spread_quartile"]) == 4
    lo, hi = r["selected_minus_own_ci"]
    assert lo <= r["selected_minus_own"] <= hi


def test_heldout_drives_are_fixed_by_the_seed():
    ids = [f"openscene/{l}/t{i}" for l in LOGS for i in range(3)]
    h = drive_map.heldout_drives(ids, 1)
    assert len(h) == 1 and h == drive_map.heldout_drives(list(reversed(ids)), 1)
    assert drive_map.heldout_drives(ids, 5) == {drive_map.drive_of_log(l) for l in LOGS}


# ---- end to end ------------------------------------------------------------------------------------------------
def run(script, *args):
    r = subprocess.run([sys.executable, str(HEADS / script), *args], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


def make_store(folder, per_drive=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"openscene/{l}/tok{d}x{i}" for d, l in enumerate(LOGS) for i in range(per_drive)]
    w = features.Writer(str(folder), [12, 24], rows=len(ids), tokens=6, width=16, settings={"model": "synthetic"})
    for s in range(0, len(ids), 4):
        b = ids[s : s + 4]
        feats = {l: rng.integers(0, 2**15, size=(len(b), 6, 16), dtype=np.uint16) & 0x3FFF for l in (12, 24)}
        metas = [{"id": i, "ego": rng.normal(size=features.EGO_DIM).tolist(),
                  "target_xy": np.cumsum(rng.uniform(0, 2, size=(8, 2)) * [1, 0.1], axis=0).tolist(),
                  "target_yaw": [0.0] * 8} for i in b]
        w.write(feats, [6] * len(b), metas)
    w.finish({})
    return [i.rsplit("/", 1)[-1] for i in ids]


def fake_subscores(path):
    """A deterministic stand-in for the PDM scorer: leaving |y| < 2 m fails DAC; progress grows with x."""
    dac = float(np.abs(path[:, 1]).max() < 2.0)
    return {"no_at_fault_collisions": 1.0, "drivable_area_compliance": dac, "time_to_collision_within_bound": 1.0,
            "ego_progress": float(min(1.0, path[-1, 0] / 16.0)) * dac, "comfort": 1.0, "driving_direction_compliance": 1.0}


def make_candidates(root, tokens, seed=0, errors=True):
    rng = np.random.default_rng(seed)
    names = ["single"] + [f"scale2_sample_{j}" for j in range(6)] + ["sample_0"]  # sample_0: another setting
    paths = {t: [np.cumsum(np.stack([rng.uniform(1, 2.5, 8), rng.normal(0, 0.5, 8)], 1), axis=0) for _ in names]
             for t in tokens}
    for j, n in enumerate(names):
        (root / n).mkdir(parents=True, exist_ok=True)
        (root / n / "shard_000_of_001.jsonl").write_text(
            "".join(json.dumps({"token": t, "waypoints": paths[t][j].round(3).tolist()}) + "\n" for t in tokens))
    lines = []
    for i, t in enumerate(tokens):
        res = []
        for j in range(len(names)):
            s = fake_subscores(np.asarray(paths[t][j]).round(3))
            res.append({**s, "score": float(sl.compose(np.array([s[k] for k in sl.SUBS])))})
        if errors and i == 0:
            res[3] = {"score": 0.0, "error": "metric cache: missing"}
        lines.append(json.dumps({"token": t, "names": names, "results": res}) + "\n")
    return names, paths, lines


def test_train_and_apply_end_to_end(tmp_path):
    tokens = make_store(tmp_path / "cache" / "train")
    gen = tmp_path / "gen.pt"
    torch.save({"config": {"layers": [12, 24], "max_tokens": 6}, "stats": {"mean": torch.zeros(16), "std": torch.ones(16)}}, gen)
    steps = tmp_path / "gen" / "navtrain_rl_10" / "steps2"
    _, _, lines = make_candidates(steps, tokens)
    (tmp_path / "scores.jsonl").write_text("".join(lines))
    out = tmp_path / "sel"
    common = ["--cache", str(tmp_path / "cache" / "train"), "--generator", str(gen), "--candidates", str(steps),
              str(tmp_path / "scores.jsonl"), "--scale", "2", "--out", str(out), "--width", "32", "--depth", "2",
              "--heads", "4", "--batch", "4", "--warmup", "2", "--workers", "1", "--heldout-drives", "1"]
    run("train_selector.py", *common, "--epochs", "1")
    run("train_selector.py", *common, "--epochs", "2", "--resume")
    for f in ("epoch_1.pt", "epoch_2.pt", "checkpoint.pt", "choice.json", "run.json", "log.jsonl"):
        assert (out / f).exists(), f
    runj = json.loads((out / "run.json").read_text())
    assert runj["scenes"] == {"train": 12, "heldout": 6} and runj["candidates_per_scene"] == 7
    assert runj["unscorable_candidates"] == 1 and len(runj["drives"]["heldout"]) == 1
    choice = json.loads((out / "choice.json").read_text())
    assert choice["epochs_done"] == 2 and choice["delta"] in sl.DELTAS and choice["weights"] in ("epoch_1", "epoch_2")
    held = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines() if '"heldout"' in l]
    assert [h["epoch"] for h in held] == [1, 2]

    # apply it to another split's candidates, with true scores from a score_candidates file and from NAVSIM files
    dev_tokens = make_store(tmp_path / "cache" / "dev", per_drive=3, seed=5)
    exp = tmp_path / "exp"
    dsteps = exp / "heads" / "gen" / "navdev_epoch_1" / "steps2"
    names, paths, dlines = make_candidates(dsteps, dev_tokens, seed=6, errors=False)
    (tmp_path / "dev_scores.jsonl").write_text("".join(dlines))
    drives = {t: drive_map.drive_of_log(LOGS[int(t[3])]) for t in dev_tokens}
    (tmp_path / "drives.json").write_text(json.dumps(drives))
    app = ["--selector", str(out), "--cache", str(tmp_path / "cache" / "dev"), "--paths", str(dsteps), "--scale", "2",
           "--drives", str(tmp_path / "drives.json")]
    run("apply_selector.py", *app, "--out", str(tmp_path / "applied"), "--scores", str(tmp_path / "dev_scores.jsonl"))
    rows = [json.loads(l) for l in (tmp_path / "applied" / "selected" / "shard_000_of_001.jsonl").read_text().splitlines()]
    assert [r["token"] for r in rows] == dev_tokens
    sel = [json.loads(l) for l in (tmp_path / "applied" / "selection.jsonl").read_text().splitlines()]
    for r, s in zip(rows, sel):
        assert s["chosen_name"] in names[:7] and len(s["score"]) == 7
        res = json.loads(dlines[dev_tokens.index(s["token"])])["results"]
        assert s["true_chosen"] == pytest.approx(res[s["chosen"]]["score"]) and s["true_own"] == pytest.approx(res[0]["score"])
        assert np.allclose(r["waypoints"], np.asarray(paths[r["token"]][names.index(s["chosen_name"])]).round(3))
    rep = json.loads((tmp_path / "applied" / "report.json").read_text())
    assert rep["report"]["scenes"] == 9 and rep["report"]["candidates"] == 7 and rep["intervals_by"] == "drive"
    assert rep["names"] == names[:7]  # the own pick and the chosen setting's samples only

    # the same truth from NAVSIM score files named as the driver names them
    for j, n in enumerate(names[:7]):
        d = exp / f"heads-gen-navdev_epoch_1-steps2-{n}_navdev_v1" / "2026.11.20.12.00.00"
        d.mkdir(parents=True)
        recs = [{"token": t, "valid": True, **json.loads(dlines[i])["results"][j]} for i, t in enumerate(dev_tokens)]
        cols = ["token", "valid", *sl.SUBS, "driving_direction_compliance", "score"]
        (d / "scores.csv").write_text(",".join(cols) + "\n" + "".join(",".join(str(r[c]) for c in cols) + "\n" for r in recs)
                                      + "average,True," + ",".join(["0.5"] * (len(cols) - 2)) + "\n")
    run("apply_selector.py", *app, "--out", str(tmp_path / "applied2"), "--exp", str(exp), "--split", "navdev")
    rep2 = json.loads((tmp_path / "applied2" / "report.json").read_text())
    assert rep2["report"]["selected_pdms"] == pytest.approx(rep["report"]["selected_pdms"], abs=1e-6)
