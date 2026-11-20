"""The selector (docs/HEADS_PLAN.md R14): scores each candidate path of a scene and picks one.

It predicts the five PDM sub-scores of every candidate and combines them with NAVSIM v1.1's formula,
NC x DAC x (5 TTC + 5 EP + 2 C) / 12 (``compose``; driving-direction compliance has weight 0 in v1.1). The network
is the path generator's body (``model.Trunk``): the stored layers the chosen path generator reads, each projected
to the selector's width with a layer embedding and read by cross-attention, and the ego state through AdaLN. Each
candidate is one token, made from its decoded waypoints only (never the generator's hidden states): the 8 (x, y)
points normalised with the generator's statistics, plus heading, speed and acceleration derived per step, each
through Fourier features, then a 2-layer MLP. The self-attention mask is the identity, so there is no attention
across candidates and a candidate's score does not depend on which others are present. The own pick carries no
flag; it is kept unless another candidate is predicted to be more than ``delta`` higher (``select``).

Ablations (R14, one run each): ``target="pdms"`` predicts the PDMS directly; ``listwise`` adds a listwise KL term;
``imitation`` adds an imitation output, softmax(-ADE / 1 m) to the logged path over the scene's candidates, used
at selection with weight ``imitation`` in the log domain (``selection_score``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as hm

SUBS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress",
        "comfort")  # the order of the outputs
SHORT = ("NC", "DAC", "TTC", "EP", "C")
DT = 0.5  # s between waypoints (2 Hz)
# per-step channels: normalised x, y; heading (rad); speed (m/s); acceleration (m/s^2), each divided by its unit
# before the Fourier features, so wavelengths run from 100 units down to 1 unit
UNITS = (0.1, 0.1, 0.01, 0.1, 0.1)
N_FOURIER = 8
TOKEN_IN = hm.N_WP * len(UNITS) * (2 * N_FOURIER + 1)
IMITATION_T = 1.0  # m: the temperature of the imitation target softmax(-ADE / T)
LISTWISE_T = 0.5


@dataclass
class SelectorConfig:
    layers: List[int] = field(default_factory=lambda: [12, 18, 24])
    d_in: int = 1024
    width: int = 256
    depth: int = 4
    heads: int = 4
    mlp: int = 4
    max_tokens: int = 512
    dropout: float = 0.1
    target: str = "subscores"  # subscores (the design) | pdms (ablation)
    listwise: float = 0.0  # weight of the listwise term (ablation: 0.25)
    imitation: float = 0.0  # weight of the imitation output at selection (ablation: 0.1); 0 = no imitation output

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def n_out(self) -> int:
        return (len(SUBS) if self.target == "subscores" else 1) + (1 if self.imitation > 0 else 0)


def compose(p):
    """NAVSIM v1.1 PDMS from sub-scores in ``SUBS`` order, ``[..., 5]`` (torch or numpy)."""
    nc, dac, ttc, ep, c = (p[..., i] for i in range(5))
    return nc * dac * (5 * ttc + 5 * ep + 2 * c) / 12


def kinematics(xy: torch.Tensor, v0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Heading, speed and acceleration per step of paths ``[..., 8, 2]`` that start at the origin: ``[..., 8, 3]``.

    Heading is that of the step's displacement (0 when the step is under 0.1 m, where it is undefined); speed is the
    step length over 0.5 s; acceleration is the change of speed, from ``v0`` (the ego speed, ``[...]``) at the first
    step, or 0 there without it."""
    prev = torch.cat([torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], dim=-2)
    d = xy - prev
    dist = d.norm(dim=-1)
    heading = torch.where(dist > 0.1, torch.atan2(d[..., 1], d[..., 0]), torch.zeros_like(dist))
    speed = dist / DT
    first = speed[..., :1] if v0 is None else v0[..., None].to(speed.dtype)
    accel = (speed - torch.cat([first, speed[..., :-1]], dim=-1)) / DT
    return torch.stack([heading, speed, accel], dim=-1)


class Selector(nn.Module):
    def __init__(self, cfg: SelectorConfig, stats: Dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        hc = hm.HeadConfig(kind="regression", layers=list(cfg.layers), d_in=cfg.d_in, width=cfg.width,
                           depth=cfg.depth, heads=cfg.heads, mlp=cfg.mlp, max_tokens=cfg.max_tokens)
        self.trunk = hm.Trunk(hc)
        w = cfg.width
        self.register_buffer("mean", stats["mean"].float().reshape(hm.N_WP, 2))
        self.register_buffer("std", stats["std"].float().reshape(hm.N_WP, 2).clamp_min(0.05))
        self.register_buffer("units", torch.tensor(UNITS), persistent=False)
        self.token = nn.Sequential(nn.Linear(TOKEN_IN, w), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(w, w))
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(w, cfg.n_out)

    def encode(self, paths: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Candidate tokens from their waypoints ``[B, K, 8, 2]`` (metres): ``[B, K, width]``."""
        paths = paths.float()
        z = (paths - self.mean) / self.std
        kin = kinematics(paths, ego[:, None, 0].float().expand(paths.shape[:2]))
        ch = torch.cat([z, kin], dim=-1) / self.units  # [B, K, 8, 5]
        f = hm.fourier(ch, N_FOURIER, 100.0)  # [B, K, 8, 5, 17]
        return self.token(f.flatten(2))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor, ego: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """Logits ``[B, K, n_out]`` for ``K`` candidates per scene. The scene's memory is read once for all K."""
        b, k = paths.shape[:2]
        kvs, mem_mask = self.trunk.memory(feats, mask)
        cond = self.trunk.condition(ego.float(), torch.zeros(b, device=ego.device))
        x = self.encode(paths, ego)
        self_mask = hm.block_mask(k, 1, paths.device)  # the identity: no attention across candidates
        for blk, kv in zip(self.trunk.blocks, kvs):
            x = x + self.drop(blk(x, kv, mem_mask, cond, self_mask) - x)
        s, c = self.trunk.out_ada(cond)[:, None].chunk(2, dim=-1)
        return self.out(self.trunk.out_norm(x) * (1 + c) + s)


def predicted(logits: torch.Tensor, cfg: SelectorConfig) -> Dict[str, torch.Tensor]:
    """``sub`` (probabilities ``[B, K, 5]``, subscores target only), ``pdms`` (the composed or direct prediction,
    ``[B, K]``) and ``score`` (what selection ranks: ``pdms``, times the imitation probability to the power
    ``cfg.imitation`` when that output exists)."""
    out = {}
    if cfg.target == "subscores":
        out["sub"] = torch.sigmoid(logits[..., : len(SUBS)].float())
        out["pdms"] = compose(out["sub"])
    else:
        out["pdms"] = torch.sigmoid(logits[..., 0].float())
    out["score"] = out["pdms"]
    if cfg.imitation > 0:
        log_q = torch.log_softmax(logits[..., -1].float(), dim=-1)
        out["score"] = torch.exp(torch.log(out["pdms"].clamp_min(1e-6)) + cfg.imitation * log_q)
    return out


def minmax(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    big = torch.finfo(x.dtype).max
    lo = torch.where(valid, x, torch.full_like(x, big)).min(-1, keepdim=True).values
    hi = torch.where(valid, x, torch.full_like(x, -big)).max(-1, keepdim=True).values
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def loss(logits: torch.Tensor, sub: torch.Tensor, pdms: torch.Tensor, valid: torch.Tensor, cfg: SelectorConfig,
         ade: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """BCE with soft targets per sub-score (NC's 1/2 stays 1/2), each averaged over its valid entries, then summed.
    The EP term is left out where the target NC x DAC is 0 (the scorer sets EP to 0 there). ``valid`` [B, K] drops
    candidates NAVSIM could not score. Ablations: ``pdms`` target; listwise KL; imitation (needs ``ade`` [B, K], the
    candidates' ADE to the logged path)."""
    valid = valid.bool()
    out = {}
    if cfg.target == "subscores":
        lg = logits[..., : len(SUBS)].float()
        terms = []
        for i, name in enumerate(SHORT):
            m = valid if name != "EP" else valid & (sub[..., 0] * sub[..., 1] > 0)
            if m.any():
                t = F.binary_cross_entropy_with_logits(lg[..., i][m], sub[..., i][m].float())
            else:
                t = lg.sum() * 0.0
            out[f"bce_{name}"] = t
            terms.append(t)
        total = sum(terms)
    else:
        total = F.binary_cross_entropy_with_logits(logits[..., 0].float()[valid], pdms[valid].float())
        out["bce_pdms"] = total
    if cfg.listwise > 0:
        pred = predicted(logits, cfg)["pdms"]
        spread = torch.where(valid, pdms, torch.zeros_like(pdms)).amax(-1) - \
            torch.where(valid, pdms, torch.ones_like(pdms)).amin(-1)
        rows = spread > 1e-6  # scenes whose candidates differ in true PDMS
        if rows.any():
            neg = torch.finfo(torch.float32).min
            p = torch.softmax(torch.where(valid, minmax(pdms.float(), valid) / LISTWISE_T, torch.full_like(pred, neg)), -1)
            log_q = torch.log_softmax(torch.where(valid, minmax(pred, valid) / LISTWISE_T, torch.full_like(pred, neg)), -1)
            kl = (p * (torch.log(p.clamp_min(1e-12)) - log_q)).sum(-1)
            out["listwise"] = kl[rows].mean()
            total = total + cfg.listwise * out["listwise"]
    if cfg.imitation > 0:
        if ade is None:
            raise ValueError("the imitation output needs each candidate's ADE to the logged path")
        neg = torch.finfo(torch.float32).min
        tgt = torch.softmax(torch.where(valid, -ade.float() / IMITATION_T, torch.full_like(ade.float(), neg)), -1)
        log_q = torch.log_softmax(torch.where(valid, logits[..., -1].float(), torch.full_like(ade.float(), neg)), -1)
        out["imitation"] = -(tgt * log_q.clamp_min(-1e4)).sum(-1).mean()
        total = total + out["imitation"]
    out["loss"] = total
    return out


# ---- selection and its measures (numpy) ------------------------------------------------------------------------
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1)  # the own-pick margins tried on the held-out training drives


def select(score: np.ndarray, delta: float, own: int = 0) -> np.ndarray:
    """Per scene, the candidate with the highest predicted score, unless it is no more than ``delta`` above the own
    pick (index ``own``), which is then kept. A tie for the highest goes to the earliest candidate."""
    best = score.argmax(1)
    gain = score[np.arange(len(score)), best] - score[:, own]
    return np.where(gain > delta, best, own)


def choose_delta(score: np.ndarray, true: np.ndarray, deltas: Sequence[float] = DELTAS) -> Dict:
    """The margin with the highest mean true PDMS of the selected paths; a tie goes to the larger margin (closer to
    the own pick)."""
    res = {f"{d:g}": float(true[np.arange(len(true)), select(score, d)].mean()) for d in deltas}
    best = max(deltas, key=lambda d: (round(res[f"{d:g}"], 10), d))
    return {"delta": best, "selected_pdms": res[f"{best:g}"], "by_delta": res}


def drive_ci(x: np.ndarray, groups: Sequence[str], n: int = 10000, seed: int = 0) -> List[float]:
    """95% interval of the mean of ``x`` by the drive-level bootstrap (docs/HEADS_PLAN.md R3)."""
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    s, c = np.bincount(inv, weights=x), np.bincount(inv)
    i = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    m = s[i].sum(1) / c[i].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of probabilities ``p`` against targets ``y`` in [0, 1], equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / max(len(p), 1))


def report(true: np.ndarray, true_sub: np.ndarray, score: np.ndarray, chosen: np.ndarray, drives: Sequence[str],
           paths: Optional[np.ndarray] = None, pred_sub: Optional[np.ndarray] = None, seed: int = 0) -> Dict:
    """R14's measures for one candidate set: ``true`` PDMS ``[N, K]``, ``true_sub`` ``[N, K, 5]`` in ``SUBS`` order,
    the predicted ``score`` ``[N, K]``, the ``chosen`` candidate per scene, each scene's drive, and optionally the
    candidates' waypoints ``[N, K, 8, 2]`` (gain by candidate spread) and the predicted sub-scores (calibration).
    Index 0 is the own pick."""
    n, k = true.shape
    r = np.arange(n)
    sel, own = true[r, chosen], true[:, 0]
    best = true.max(1)
    diff = sel - own
    out = {"scenes": int(n), "candidates": int(k), "drives": int(len(set(drives))),
           "selected_pdms": float(sel.mean()), "own_pick_pdms": float(own.mean()),
           "selected_minus_own": float(diff.mean()), "selected_minus_own_ci": drive_ci(diff, drives),
           "best_of_k_pdms": float(best.mean()),
           "random_pick_pdms": float(true[r, np.random.default_rng(seed).integers(0, k, n)].mean()),
           "mean_candidate_pdms": float(true.mean(1).mean())}
    gap = float((best - own).mean())
    out["share_of_best_of_k_gap_closed"] = float(diff.mean() / gap) if gap > 0 else None
    switched = chosen != 0
    out["switch_rate"] = float(switched.mean())
    out["switches"] = {"wins": int((switched & (diff > 1e-9)).sum()), "losses": int((switched & (diff < -1e-9)).sum()),
                       "ties": int((switched & (np.abs(diff) <= 1e-9)).sum())}
    fail = lambda s: (s[:, 0] == 0) | (s[:, 1] == 0)  # noqa: E731  (NC or DAC failed)
    out["picks_failing_nc_or_dac"] = {"selected": int(fail(true_sub[r, chosen]).sum()), "own_pick": int(fail(true_sub[:, 0]).sum())}
    out["sub_scores"] = {SHORT[i]: {"selected": float(true_sub[r, chosen, i].mean()), "own_pick": float(true_sub[:, 0, i].mean())}
                         for i in range(len(SUBS))}
    top = score.argmax(1)
    out["top1_exact"] = float((top == true.argmax(1)).mean())
    out["top1_tie_aware"] = float((true[r, top] >= best - 1e-9).mean())
    if pred_sub is not None:
        out["calibration"] = {SHORT[i]: {"mean_predicted": float(pred_sub[..., i].mean()), "mean_true": float(true_sub[..., i].mean()),
                                         "ece": ece(pred_sub[..., i].ravel(), true_sub[..., i].ravel())} for i in range(len(SUBS))}
    if paths is not None and k > 1:
        end = paths[:, :, -1]  # [N, K, 2]
        dist = np.linalg.norm(end[:, :, None] - end[:, None], axis=-1)
        spread = dist.sum((1, 2)) / (k * (k - 1))  # mean pairwise end-point distance
        q = np.quantile(spread, [0.25, 0.5, 0.75])
        bins = np.digitize(spread, q)
        out["gain_by_spread_quartile"] = [{"spread_m": [float(spread[bins == b].min()), float(spread[bins == b].max())],
                                           "scenes": int((bins == b).sum()), "selected_minus_own": float(diff[bins == b].mean())}
                                          for b in range(4) if (bins == b).any()]
    return out
