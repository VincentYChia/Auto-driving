"""R15's tuning settings for B and C (docs/HEADS_PLAN.md): B's prediction target (clean path, noise, v) and noise
schedule (cosine, DDPM's linear), C's training-time sampling (uniform, pi0's Beta, SD3's logit-normal); and D's
truncation flag. Each trains, predicts and runs RL from the command line."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pilot_sft" / "heads"))
sys.path.insert(0, str(HERE))

import model as hm  # noqa: E402
from test_head_train import make_split, run  # noqa: E402


def head(**kw):
    cfg = hm.HeadConfig(layers=[12, 24], d_in=16, width=32, depth=1, heads=4, max_tokens=8, **kw)
    torch.manual_seed(0)
    return hm.Head(cfg, {"mean": torch.zeros(16), "std": torch.ones(16)})


def test_linear_schedule_ends_where_ddpm_does():
    h = head(kind="ddim", schedule="linear")
    t = torch.linspace(0, 1, 11)
    a = h.alpha_bar(t)
    assert a[0] == pytest.approx(1.0) and a[-1] == pytest.approx(4.32e-5, rel=0.01)
    assert bool((a[1:] < a[:-1]).all())
    c = head(kind="ddim").alpha_bar(t)  # the cosine schedule of phase 1 is unchanged
    assert c[0] == pytest.approx(1.0, abs=1e-3) and c[-1] < 1e-6


@pytest.mark.parametrize("target", ["x0", "eps", "v"])
def test_each_target_gives_back_the_clean_path(target):
    h = head(kind="ddim", target=target)
    g = torch.Generator().manual_seed(1)
    x1 = torch.randn(3, 1, 8, 2, generator=g)
    noise = torch.randn(3, 1, 8, 2, generator=g)
    t = torch.tensor([0.1, 0.4, 0.8])
    a = h.alpha_bar(t)[:, None, None, None]
    z = a.sqrt() * x1 + (1 - a).sqrt() * noise
    out = {"x0": x1, "eps": noise, "v": a.sqrt() * noise - (1 - a).sqrt() * x1}[target]  # a perfect network
    assert torch.allclose(h.ddim_x0(out, z, a), x1, atol=1e-4)


def test_c_time_sampling():
    torch.manual_seed(0)
    u = head(kind="flow").flow_times(20000, "cpu")
    b = head(kind="flow", time_sampling="beta").flow_times(20000, "cpu")
    ln = head(kind="flow", time_sampling="logit-normal").flow_times(20000, "cpu")
    assert u.mean() == pytest.approx(0.5, abs=0.01) and ln.mean() == pytest.approx(0.5, abs=0.01)
    assert b.mean() == pytest.approx(0.999 * 0.4, abs=0.01) and float(b.max()) <= 0.999  # toward the noisy end
    assert float(ln.std()) < float(u.std())  # logit-normal concentrates in the middle
    with pytest.raises(ValueError):
        head(kind="flow", time_sampling="x").flow_times(4, "cpu")


@pytest.mark.parametrize("kind,flags", [
    ("ddim", ["--target", "eps", "--schedule", "linear"]),
    ("ddim", ["--target", "v"]),
    ("flow", ["--flow-sigma", "1.0", "--time-sampling", "beta"]),
    ("flow", ["--time-sampling", "logit-normal"]),
    ("anchor", ["--anchors", "4", "--anchor-trunc", "0.1"]),
])
def test_train_predict_and_rl_with_each_r15_setting(tmp_path, kind, flags):
    cache = tmp_path / "cache"
    make_split(cache / "train", 16, seed=0)
    make_split(cache / "dev", 4, seed=1)
    out = tmp_path / "run"
    run("train_head.py", "--cache", str(cache), "--out", str(out), "--kind", kind, "--width", "32", "--depth", "2",
        "--heads", "4", "--batch", "8", "--warmup", "2", "--workers", "1", "--epochs", "1", *flags)
    cfg = json.loads((out / "run.json").read_text())["config"]
    pairs = dict(zip(flags[::2], flags[1::2]))
    want = {"--target": "target", "--schedule": "schedule", "--time-sampling": "time_sampling",
            "--flow-sigma": "flow_sigma", "--anchor-trunc": "anchor_trunc", "--anchors": "anchors"}
    for flag, key in want.items():
        if flag in pairs:
            assert str(cfg[key]) == pairs[flag] or float(cfg[key]) == float(pairs[flag])
    run("predict_head.py", "--run", str(out), "--cache", str(cache / "dev"), "--split", "navdev", "--k", "2")
    folder = next((out / "navdev_final").glob("steps*"))
    for name in ("single", "sample_0", "sample_1"):
        rows = [json.loads(line) for line in (folder / name / "shard_000_of_001.jsonl").read_text().splitlines()]
        assert len(rows) == 4 and all(abs(v) < 1e3 for r in rows for p in r["waypoints"] for v in p)
    # RL's sampling and log-probabilities under the same settings (a policy step, no scorer needed)
    from predict_head import load_head
    h = load_head(str(out / "final.pt"), "cpu")
    b = {"feats": torch.randint(0, 2 ** 14, (2, 2, 6, 16), dtype=torch.int16).view(torch.bfloat16),
         "mask": torch.ones(2, 6, dtype=torch.bool), "ego": torch.randn(2, 19)}
    paths, trace = h.rl_sample(b["feats"], b["mask"], b["ego"], g=3, steps=2, scale=1.0)
    lp = h.rl_logprob(b["feats"], b["mask"], b["ego"], trace, lp_floor=0.1)  # no reference: the log-probability only
    assert paths.shape == (2, 3, 8, 2) and lp.shape == (2, 3) and bool(torch.isfinite(lp).all())
    lp.sum().backward()
