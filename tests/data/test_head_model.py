"""``pilot_sft/heads/model.py``: every head option trains and generates with the right shapes, the masked
head's tokens round-trip, and K paths sharing one row see only their own tokens. CPU, tiny network."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import model as hm  # noqa: E402


def tiny(kind, **kw):
    cfg = hm.HeadConfig(kind=kind, layers=[12, 24], d_in=32, width=32, depth=2, heads=4, mlp=2, max_tokens=16,
                        anchors=5, **kw)
    stats = {"mean": torch.zeros(16), "std": torch.ones(16) * 5,
             "anchors": torch.randn(5, 16)}
    return hm.Head(cfg, stats)


def batch(b=3, t=7):
    torch.manual_seed(0)
    feats = torch.randn(b, 2, t, 32).to(torch.bfloat16)
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, 5:] = False
    ego = torch.randn(b, hm.EGO_DIM)
    xy = torch.cumsum(torch.rand(b, 8, 2) * torch.tensor([2.0, 0.2]), dim=1)
    return feats, mask, ego, xy


@pytest.mark.parametrize("kind,kw", [
    ("masked", {}), ("masked", {"soft_sigma": 1.0, "l1_weight": 1.0, "offset": True}),
    ("masked", {"metric_embed": False}), ("ddim", {}), ("flow", {}), ("anchor", {}), ("regression", {}),
])
def test_each_head_trains_and_generates(kind, kw):
    head = tiny(kind, **kw)
    feats, mask, ego, xy = batch()
    out = head.loss(feats, mask, ego, xy)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    for k, steps in ((1, 1), (3, 2), (2, 4)):
        paths, score = head.generate(feats, mask, ego, k=k, steps=steps, temperature=1.0)
        assert paths.shape == (3, k, 8, 2) and score.shape == (3, k)
        assert torch.isfinite(paths).all()


def test_masked_tokens_round_trip_within_half_a_bin():
    head = tiny("masked", bin_m=0.1)
    _, _, _, xy = batch()
    back = head.centers(head.bins(xy)).reshape(xy.shape)
    assert (back - xy).abs().max() <= 0.05 + 1e-5


def test_masked_generation_commits_every_token():
    head = tiny("masked")
    feats, mask, ego, _ = batch()
    for steps in (1, 2, 4, 8):
        paths, _ = head.generate(feats, mask, ego, k=2, steps=steps, temperature=0.0)
        lo = torch.tensor([head.cfg.x_range[0], head.cfg.y_range[0]])
        assert ((paths - lo) / head.cfg.bin_m).sub(((paths - lo) / head.cfg.bin_m).round()).abs().max() < 1e-3


def test_paths_in_one_row_do_not_see_each_other():
    torch.manual_seed(1)
    head = tiny("ddim").eval()
    feats, mask, ego, _ = batch()
    z = torch.randn(3, 2, 8, 2)
    both = head.cont_forward(z, *head.trunk.memory(feats, mask), ego, torch.full((3,), 0.5), k=2)
    z2 = z.clone()
    z2[:, 1] += 3.0  # change only the second path's input
    again = head.cont_forward(z2, *head.trunk.memory(feats, mask), ego, torch.full((3,), 0.5), k=2)
    assert torch.allclose(both[:, 0], again[:, 0], atol=1e-5)
    assert not torch.allclose(both[:, 1], again[:, 1])


def test_padding_tokens_are_ignored():
    head = tiny("flow").eval()
    feats, mask, ego, _ = batch()
    z = torch.randn(3, 1, 8, 2)
    a = head.cont_forward(z, *head.trunk.memory(feats, mask), ego, torch.full((3,), 0.3))
    feats2 = feats.clone()
    feats2[0, :, 5:] = 100.0  # row 0's padding
    b = head.cont_forward(z, *head.trunk.memory(feats2, mask), ego, torch.full((3,), 0.3))
    assert torch.allclose(a, b, atol=1e-5)


def test_kmeans_finds_separated_clusters():
    g = torch.Generator().manual_seed(0)
    pts = torch.cat([torch.randn(50, 16, generator=g) * 0.1 + c for c in (0.0, 5.0, -5.0)])
    c = hm.kmeans_anchors(pts, 3)
    assert sorted(round(v) for v in c.mean(1).tolist()) == [-5, 0, 5]


def test_rows_longer_than_the_position_table_still_run():
    head = tiny("masked").eval()  # max_tokens 16
    feats = torch.randn(2, 2, 18, 32).to(torch.bfloat16)
    mask = torch.ones(2, 18, dtype=torch.bool)
    paths, _ = head.generate(feats, mask, torch.randn(2, hm.EGO_DIM), k=1, steps=2, temperature=0.0)
    assert paths.shape == (2, 1, 8, 2) and torch.isfinite(paths).all()


@pytest.mark.parametrize("kind", ["masked", "ddim", "flow", "anchor", "regression"])
def test_temperature_zero_is_the_deterministic_best_guess(kind):
    head = tiny(kind).eval()
    feats, mask, ego, _ = batch()
    a, _ = head.generate(feats, mask, ego, k=1, steps=2, temperature=0.0, generator=torch.Generator().manual_seed(1))
    b, _ = head.generate(feats, mask, ego, k=1, steps=2, temperature=0.0, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)  # no randomness left at temperature 0, whatever the seed


@pytest.mark.parametrize("kind", ["ddim", "flow"])
def test_sampling_scale_sets_the_spread_of_the_k_paths(kind):
    head = tiny(kind).eval()
    feats, mask, ego, _ = batch()
    spread = {}
    for x in (0.0, 1.0):
        paths, _ = head.generate(feats, mask, ego, k=4, steps=2, temperature=x, generator=torch.Generator().manual_seed(0))
        spread[x] = float((paths - paths.mean(1, keepdim=True)).abs().mean())
    assert spread[0.0] == 0.0 and spread[1.0] > 0.0  # scale 0: K identical paths; scale 1: K different ones
