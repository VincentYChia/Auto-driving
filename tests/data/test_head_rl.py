"""RL policies of the path generators (docs/HEADS_PLAN.md phase 4): A's and D's sampled candidates, their exact
log-probabilities with gradients, and a policy-gradient sanity check with a toy reward. CPU, tiny network."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model as hm  # noqa: E402
from test_head_model import batch, tiny  # noqa: E402


def test_masked_logprob_matches_the_sampling_probabilities():
    head = tiny("masked").eval()
    feats, mask, ego, _ = batch()
    kvs, mem_mask = head.trunk.memory(feats, mask)
    trace = {}
    _, mean_lp = head.masked_generate(kvs, mem_mask, ego, k=3, steps=4, temperature=1.0,
                                      generator=torch.Generator().manual_seed(0), trace=trace)
    lp = head.rl_logprob(feats, mask, ego, trace)
    assert lp.shape == (3, 3)
    assert torch.allclose(lp, mean_lp * 16, atol=1e-3)  # every token's log-probability, at the step it was committed


def test_masked_logprob_has_gradients_and_every_token_is_counted():
    head = tiny("masked")
    feats, mask, ego, _ = batch()
    xy, trace = head.rl_sample(feats, mask, ego, g=4, steps=2, scale=1.5, generator=torch.Generator().manual_seed(1))
    assert xy.shape == (3, 4, 8, 2) and (trace["step_of"] >= 0).all()
    lp = head.rl_logprob(feats, mask, ego, trace)
    lp.sum().backward()
    assert torch.isfinite(lp).all() and head.out_x.weight.grad.abs().sum() > 0


def test_anchor_logprob_is_finite_has_gradients_and_depends_on_the_pick():
    head = tiny("anchor")
    feats, mask, ego, _ = batch()
    xy, trace = head.rl_sample(feats, mask, ego, g=3, steps=2, scale=1.0, generator=torch.Generator().manual_seed(2))
    assert xy.shape == (3, 3, 8, 2)
    lp = head.rl_logprob(feats, mask, ego, trace)
    assert lp.shape == (3, 3) and torch.isfinite(lp).all()
    lp.sum().backward()
    assert head.score.weight.grad.abs().sum() > 0 and head.out.weight.grad.abs().sum() > 0
    other = dict(trace, pick=(trace["pick"] + 1) % head.cfg.anchors)
    with torch.no_grad():
        assert not torch.allclose(head.rl_logprob(feats, mask, ego, other), lp.detach())


def test_regression_has_no_rl_policy():
    head = tiny("regression")
    feats, mask, ego, _ = batch()
    with pytest.raises(NotImplementedError):
        head.rl_sample(feats, mask, ego, g=2, steps=2, scale=1.0)


@pytest.mark.parametrize("kind", ["masked", "anchor", "ddim", "flow"])
def test_policy_gradient_raises_a_toy_reward(kind):
    """Reward = the end point's x coordinate. Group-relative advantages and -A * log p should push it up."""
    torch.manual_seed(0)
    head = tiny(kind, bin_m=5.0) if kind == "masked" else tiny(kind)  # 20 x-bins: a signal 25 updates can follow
    feats, mask, ego, _ = batch(b=4)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(3)

    def mean_reward():
        with torch.no_grad():
            xy, _ = head.rl_sample(feats, mask, ego, g=8, steps=2, scale=1.0, generator=torch.Generator().manual_seed(9))
        return float(xy[..., -1, 0].mean())

    before = mean_reward()
    for _ in range(25):
        xy, trace = head.rl_sample(feats, mask, ego, g=8, steps=2, scale=1.0, generator=gen)
        r = xy[..., -1, 0]
        adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
        loss = -(adv * head.rl_logprob(feats, mask, ego, trace)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert mean_reward() > before


@pytest.mark.parametrize("kind", ["masked", "anchor", "ddim", "flow"])
def test_one_step_moves_probability_toward_the_better_candidate(kind):
    """Two candidates, rewards 1 and 0 (advantages +1 and -1): one small gradient step on -A * log p widens the
    gap log p(better) - log p(worse). That gap is the quantity the update raises to first order. Each value alone
    can move either way when the candidates share terms, e.g. D's two candidates drawing the same anchor."""
    torch.manual_seed(0)
    head = tiny(kind)
    feats, mask, ego, _ = batch(b=1)
    _, trace = head.rl_sample(feats, mask, ego, g=2, steps=2, scale=1.0, generator=torch.Generator().manual_seed(5))
    before = head.rl_logprob(feats, mask, ego, trace).detach()
    adv = torch.tensor([[1.0, -1.0]])
    opt = torch.optim.SGD(head.parameters(), lr=1e-3)
    (-(adv * head.rl_logprob(feats, mask, ego, trace)).mean()).backward()
    opt.step()
    after = head.rl_logprob(feats, mask, ego, trace).detach()
    assert (after[0, 0] - after[0, 1]) > (before[0, 0] - before[0, 1])


@pytest.mark.parametrize("kind", ["ddim", "flow"])
def test_continuous_logprob_is_finite_and_has_gradients(kind):
    head = tiny(kind)
    feats, mask, ego, _ = batch()
    xy, trace = head.rl_sample(feats, mask, ego, g=3, steps=2, scale=1.0, generator=torch.Generator().manual_seed(4))
    assert xy.shape == (3, 3, 8, 2) and trace["z"].shape == (3, 3, 3, 8, 2)
    lp = head.rl_logprob(feats, mask, ego, trace)
    assert lp.shape == (3, 3) and torch.isfinite(lp).all()
    lp.sum().backward()
    assert head.out.weight.grad.abs().sum() > 0


def test_flow_sde_reduces_to_the_euler_ode_without_noise(monkeypatch):
    """With Flow-GRPO's noise level a = 0 the SDE step is exactly the deterministic Euler step used in evaluation."""
    head = tiny("flow")
    monkeypatch.setattr(head, "FLOW_GRPO_A", 0.0)
    z, v = torch.randn(2, 1, 8, 2), torch.randn(2, 1, 8, 2)
    mu, sd = head._flow_sde_step(z, v, 0.5, 0.25)
    assert torch.allclose(mu, z + 0.25 * v) and float(sd) == 0.0


@pytest.mark.parametrize("kind", ["masked", "anchor", "ddim", "flow"])
def test_kl_to_the_reference_is_zero_for_itself_and_positive_after_a_change(kind):
    import copy
    head = tiny(kind)
    feats, mask, ego, _ = batch()
    _, trace = head.rl_sample(feats, mask, ego, g=3, steps=2, scale=1.0, generator=torch.Generator().manual_seed(6))
    ref = copy.deepcopy(head).eval()
    lp, kl = head.rl_logprob(feats, mask, ego, trace, ref)
    assert kl.shape == (3, 3) and kl.abs().max() < 1e-5  # the same weights: no divergence
    with torch.no_grad():
        for q in head.parameters():
            q.add_(torch.randn_like(q) * 0.05)
    lp, kl = head.rl_logprob(feats, mask, ego, trace, ref)
    assert (kl > 0).all()
    kl.sum().backward()
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in head.parameters())


@pytest.mark.parametrize("kind", ["masked", "anchor", "ddim", "flow"])
def test_the_sd_floor_only_shrinks_small_gaussian_steps(kind):
    """R13: with --lp-sd-floor, every Gaussian step's SD is at least the floor inside the log-probability. A has no
    Gaussian steps, so nothing changes; for D, B and C the small-SD steps (D's last step sits at the 0.01 floor)
    count less, so the gradient is smaller, and the sampled candidates are the same."""
    head = tiny(kind)
    feats, mask, ego, _ = batch()
    _, trace = head.rl_sample(feats, mask, ego, g=3, steps=2, scale=1.0, generator=torch.Generator().manual_seed(8))
    norms, lps = [], []
    for floor in (0.0, 0.1):
        head.zero_grad(set_to_none=True)
        lp = head.rl_logprob(feats, mask, ego, trace, lp_floor=floor)
        assert torch.isfinite(lp).all()
        lp.sum().backward()
        norms.append(float(torch.norm(torch.stack([q.grad.norm() for q in head.parameters() if q.grad is not None]))))
        lps.append(lp.detach())
    if kind == "masked":
        assert torch.equal(lps[0], lps[1]) and norms[0] == norms[1]
    else:
        assert not torch.allclose(lps[0], lps[1]) and norms[1] < norms[0]
