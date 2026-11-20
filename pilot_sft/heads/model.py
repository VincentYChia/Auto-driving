"""Path generators ("heads") on stored pilot features: one shared network body, four ways to make paths.

The body (``Trunk``) is the same for every option, so the comparison isolates the kind of generation. It reads
the stored layers through cross-attention (one LayerNorm and projection per stored layer, a layer embedding and
a token-position embedding), takes the ego state and the generation time through AdaLN-Zero, and runs
``depth`` blocks of self-attention, cross-attention and an MLP over a few query tokens. Only the query tokens'
input and output layers change between options:

- ``masked`` (A): 16 tokens, x and y of the 8 waypoints on a metric grid (``bin_m``). Training hides a random
  share and predicts it; generation starts fully hidden and commits the most confident tokens over ``steps``
  rounds, the goal (last waypoint) first. The "how continuous" ladder: distance-aware bin embeddings, soft
  labels (``soft_sigma``), an L1 loss on a local expectation readout (``l1_weight``, ``expect_window``), and a
  deterministic sub-bin offset (``offset``).
- ``ddim`` (B): 8 waypoint tokens, Gaussian diffusion on normalised (x, y) predicting the clean path, DDIM.
- ``flow`` (C): the same tokens, rectified flow from a narrow Gaussian start (``flow_sigma``), Euler steps.
- ``anchor`` (D): one token per anchor path (``anchors`` k-means paths), truncated diffusion from the noised
  anchors (DiffusionDrive), a score per anchor; generation ranks the refined anchors.
- ``regression`` (R): the control without generation: 8 query tokens, an L1 loss on the normalised
  waypoints, one path.

``generate(..., temperature=0)`` is every method's deterministic best guess; above 0 it is each method's one
sampling setting (docs/HEADS_PLAN.md R7).

K paths per scene share one copy of the scene's features: their tokens sit side by side in one row with a
block-diagonal self-attention mask, so the memory is never repeated. Settings are [proposed] starting points.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_WP = 8
EGO_DIM = 19  # features.EGO_DIM
EGO_SCALE = [10.0, 3.0, 0.5] + [10.0, 2.0, 0.5] * 4 + [1.0] * 4  # speed, accel, yaw rate; poses; command


@dataclass
class HeadConfig:
    kind: str = "masked"  # masked (A) | ddim (B) | flow (C) | anchor (D) | regression (R)
    layers: List[int] = field(default_factory=lambda: [12, 18, 24])
    d_in: int = 1024
    width: int = 384
    depth: int = 12
    heads: int = 6
    mlp: int = 4
    max_tokens: int = 512
    # masked (A)
    bin_m: float = 0.1
    x_range: Tuple[float, float] = (-5.0, 90.0)
    y_range: Tuple[float, float] = (-45.0, 45.0)
    metric_embed: bool = True
    soft_sigma: float = 0.0  # in bins; 0 = hard labels
    l1_weight: float = 0.0
    expect_window: int = 3  # bins on each side of the committed bin for the expectation readout
    offset: bool = False
    # continuous (B, C, D)
    flow_sigma: float = 0.1
    anchors: int = 20
    anchor_trunc: float = 0.05  # DiffusionDrive: 50 of 1,000 noise steps
    steps: int = 4

    def to_dict(self) -> Dict:
        return asdict(self)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1] (scaled to 0-1000), [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def fourier(x: torch.Tensor, n: int = 16, scale: float = 100.0) -> torch.Tensor:
    """Fourier features of metric values (metres), [...] -> [..., 2n + 1]: smooth in distance."""
    freqs = torch.exp(torch.linspace(0.0, math.log(scale), n, device=x.device)) * (2 * math.pi / scale)
    a = x[..., None].float() * freqs
    return torch.cat([x[..., None].float() / scale, torch.cos(a), torch.sin(a)], dim=-1)


class Block(nn.Module):
    """AdaLN-Zero self-attention and MLP (DiT), plus cross-attention to the stored features."""

    def __init__(self, width: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.o1 = nn.Linear(width, width)
        self.nx = nn.LayerNorm(width, eps=1e-6)
        self.q2 = nn.Linear(width, width)
        self.kv2 = nn.Linear(width, 2 * width)
        self.o2 = nn.Linear(width, width)
        self.n3 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, mlp * width), nn.GELU(approximate="tanh"), nn.Linear(mlp * width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        for m in (self.ada[1], self.o2):  # zero-init: each block starts as the identity
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def memory_kv(self, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = self.kv2(mem).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def forward(self, x, kv, mem_mask, cond, self_mask):
        b, n, d = x.shape
        s1, c1, g1, s3, c3, g3 = self.ada(cond)[:, None].chunk(6, dim=-1)
        h = self.n1(x) * (1 + c1) + s1
        q, k, v = (self.split(t) for t in self.qkv(h).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self_mask)
        x = x + g1 * self.o1(a.transpose(1, 2).reshape(b, n, d))
        q = self.split(self.q2(self.nx(x)))
        a = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=mem_mask)
        x = x + self.o2(a.transpose(1, 2).reshape(b, n, d))
        h = self.n3(x) * (1 + c3) + s3
        return x + g3 * self.mlp(h)


class Trunk(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        w, L = cfg.width, len(cfg.layers)
        self.mem_norm = nn.ModuleList(nn.LayerNorm(cfg.d_in) for _ in range(L))
        self.mem_proj = nn.ModuleList(nn.Linear(cfg.d_in, w) for _ in range(L))
        self.mem_layer = nn.Parameter(torch.zeros(L, w))
        self.mem_pos = nn.Parameter(torch.randn(cfg.max_tokens, w) * 0.02)
        self.cond = nn.Sequential(nn.Linear(EGO_DIM + w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList(Block(w, cfg.heads, cfg.mlp) for _ in range(cfg.depth))
        self.out_norm = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
        nn.init.zeros_(self.out_ada[1].weight)
        nn.init.zeros_(self.out_ada[1].bias)
        self.register_buffer("ego_scale", torch.tensor(EGO_SCALE), persistent=False)
        self.width = w

    def memory(self, feats: torch.Tensor, mask: torch.Tensor):
        """feats [B, L, t, d] (bf16 or float), mask [B, t] -> per-block (k, v) and the attention mask."""
        b, L, t, _ = feats.shape
        # a row longer than any training row (1 navtest scene of 12,146 has 420 tokens, training at most 419)
        # takes the last position embedding for the extra tokens
        pos = self.mem_pos[torch.arange(t, device=feats.device).clamp(max=self.mem_pos.shape[0] - 1)]
        parts = [self.mem_proj[i](self.mem_norm[i](feats[:, i].float())) + self.mem_layer[i] + pos
                 for i in range(L)]
        mem = torch.cat(parts, dim=1)  # [B, L*t, w]
        mem_mask = mask.repeat(1, L)[:, None, None, :]  # [B, 1, 1, L*t], True = attend
        return [blk.memory_kv(mem) for blk in self.blocks], mem_mask

    def condition(self, ego: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.cond(torch.cat([ego / self.ego_scale, timestep_embedding(t, self.width)], dim=-1))

    def forward(self, x, kvs, mem_mask, cond, self_mask=None):
        for blk, kv in zip(self.blocks, kvs):
            x = blk(x, kv, mem_mask, cond, self_mask)
        s, c = self.out_ada(cond)[:, None].chunk(2, dim=-1)
        return self.out_norm(x) * (1 + c) + s


def block_mask(k: int, n: int, device) -> Optional[torch.Tensor]:
    """Self-attention mask for k paths of n tokens in one row: each path sees only its own tokens."""
    if k == 1:
        return None
    ids = torch.arange(k * n, device=device) // n
    return (ids[:, None] == ids[None, :])[None, None]


class Head(nn.Module):
    """One path generator: the trunk plus the option's token input and output layers."""

    def __init__(self, cfg: HeadConfig, stats: Dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        self.trunk = Trunk(cfg)
        w = cfg.width
        # normalisation of (x, y) per waypoint, from the training targets
        self.register_buffer("mean", stats["mean"].float().reshape(N_WP, 2))
        self.register_buffer("std", stats["std"].float().reshape(N_WP, 2).clamp_min(0.05))
        if cfg.kind == "masked":
            self.vx = int(round((cfg.x_range[1] - cfg.x_range[0]) / cfg.bin_m)) + 1
            self.vy = int(round((cfg.y_range[1] - cfg.y_range[0]) / cfg.bin_m)) + 1
            self.slot = nn.Parameter(torch.randn(2 * N_WP, w) * 0.02)
            self.mask_tok = nn.Parameter(torch.randn(w) * 0.02)
            if cfg.metric_embed:
                self.emb = nn.Sequential(nn.Linear(33, w), nn.SiLU(), nn.Linear(w, w))
            else:
                self.emb_x, self.emb_y = nn.Embedding(self.vx, w), nn.Embedding(self.vy, w)
            self.out_x, self.out_y = nn.Linear(w, self.vx), nn.Linear(w, self.vy)
            if cfg.offset:
                self.out_off = nn.Linear(w, 1)
        elif cfg.kind in ("ddim", "flow"):
            self.slot = nn.Parameter(torch.randn(N_WP, w) * 0.02)
            self.inp = nn.Linear(2, w)
            self.out = nn.Linear(w, 2)
        elif cfg.kind == "anchor":
            self.register_buffer("anchor_paths", stats["anchors"].float().reshape(cfg.anchors, 2 * N_WP))
            self.anchor_id = nn.Parameter(torch.randn(cfg.anchors, w) * 0.02)
            self.inp = nn.Sequential(nn.Linear(2 * N_WP, w), nn.SiLU(), nn.Linear(w, w))
            self.out = nn.Linear(w, 2 * N_WP)
            self.score = nn.Linear(w, 1)
        elif cfg.kind == "regression":  # R: the control without generation, one path from 8 query tokens
            self.slot = nn.Parameter(torch.randn(N_WP, w) * 0.02)
            self.out = nn.Linear(w, 2)
        else:
            raise ValueError(f"unknown head kind {cfg.kind}")

    # ---- shared helpers -------------------------------------------------------------------------------------
    def norm(self, xy):
        return (xy - self.mean) / self.std

    def denorm(self, z):
        return z * self.std + self.mean

    @staticmethod
    def alpha_bar(t: torch.Tensor) -> torch.Tensor:  # cosine schedule, t in [0, 1]
        return torch.cos((t.clamp(0, 1) + 0.008) / 1.008 * math.pi / 2) ** 2

    # ---- masked (A) -------------------------------------------------------------------------------------------
    def bins(self, xy: torch.Tensor) -> torch.Tensor:
        """[B, 8, 2] metres -> [B, 16] token ids, x then y per waypoint, clipped to the grid."""
        c = self.cfg
        bx = ((xy[..., 0] - c.x_range[0]) / c.bin_m).round().clamp(0, self.vx - 1)
        by = ((xy[..., 1] - c.y_range[0]) / c.bin_m).round().clamp(0, self.vy - 1)
        return torch.stack([bx, by], dim=-1).reshape(xy.shape[0], 2 * N_WP).long()

    def centers(self, tok: torch.Tensor) -> torch.Tensor:
        """[..., 16] ids -> [..., 16] metres at the bin centres."""
        c = self.cfg
        lo = torch.tensor([c.x_range[0], c.y_range[0]] * N_WP, device=tok.device)
        return lo + tok.float() * c.bin_m

    def embed_tokens(self, tok, known):
        """tok [B, M] (M = k * 16), known [B, M] bool -> [B, M, w]."""
        m = tok.shape[1]
        slot = self.slot.repeat(m // (2 * N_WP), 1)
        if self.cfg.metric_embed:
            e = self.emb(fourier(self.centers(tok.reshape(-1, 2 * N_WP)).reshape(tok.shape)))
        else:
            is_x = (torch.arange(m, device=tok.device) % 2 == 0)[None].expand_as(tok)
            e = torch.where(is_x[..., None], self.emb_x(tok.clamp(max=self.vx - 1)), self.emb_y(tok.clamp(max=self.vy - 1)))
        e = torch.where(known[..., None], e, self.mask_tok.expand_as(e))
        return e + slot

    def logits(self, h):
        """h [B, M, w] -> x logits for even slots [B, M/2, vx], y logits for odd slots [B, M/2, vy]."""
        return self.out_x(h[:, 0::2]), self.out_y(h[:, 1::2])

    def soft_targets(self, tgt: torch.Tensor, v: int) -> torch.Tensor:
        idx = torch.arange(v, device=tgt.device, dtype=torch.float32)
        d = idx[None, None] - tgt[..., None].float()
        p = torch.exp(-0.5 * (d / self.cfg.soft_sigma) ** 2) * (d.abs() <= 3 * self.cfg.soft_sigma + 0.5)
        return p / p.sum(-1, keepdim=True)

    def local_expectation(self, logits: torch.Tensor, center: torch.Tensor, lo: float) -> torch.Tensor:
        """Expected value (metres) of the bins within ``expect_window`` of ``center`` [B, M] under softmax."""
        w = self.cfg.expect_window
        offs = torch.arange(-w, w + 1, device=logits.device)
        idx = (center[..., None] + offs).clamp(0, logits.shape[-1] - 1)
        p = torch.softmax(logits.float(), -1).gather(-1, idx)
        vals = lo + idx.float() * self.cfg.bin_m
        return (p * vals).sum(-1) / p.sum(-1).clamp_min(1e-9)

    def masked_loss(self, kvs, mem_mask, ego, xy):
        c, b = self.cfg, xy.shape[0]
        tok = self.bins(xy)
        r = torch.rand(b, device=xy.device).clamp_min(1.0 / (2 * N_WP))
        hidden = torch.rand(b, 2 * N_WP, device=xy.device) < r[:, None]
        hidden[torch.arange(b), torch.randint(0, 2 * N_WP, (b,), device=xy.device)] = True  # at least one
        h = self.trunk(self.embed_tokens(tok, ~hidden), kvs, mem_mask, self.trunk.condition(ego, r))
        lx, ly = self.logits(h)
        hx, hy = hidden[:, 0::2], hidden[:, 1::2]
        tx, ty = tok[:, 0::2], tok[:, 1::2]
        if c.soft_sigma > 0:
            ce_x = -(self.soft_targets(tx, self.vx) * F.log_softmax(lx.float(), -1)).sum(-1)
            ce_y = -(self.soft_targets(ty, self.vy) * F.log_softmax(ly.float(), -1)).sum(-1)
        else:
            ce_x = F.cross_entropy(lx.float().transpose(1, 2), tx, reduction="none")
            ce_y = F.cross_entropy(ly.float().transpose(1, 2), ty, reduction="none")
        n = hidden.sum().clamp_min(1)
        loss = ((ce_x * hx).sum() + (ce_y * hy).sum()) / n
        out = {"ce": loss.detach()}
        if c.l1_weight > 0:
            ex = self.local_expectation(lx, lx.argmax(-1), c.x_range[0])
            ey = self.local_expectation(ly, ly.argmax(-1), c.y_range[0])
            l1 = (((ex - xy[..., 0]).abs() * hx).sum() + ((ey - xy[..., 1]).abs() * hy).sum()) / n
            loss = loss + c.l1_weight * l1
            out["l1"] = l1.detach()
        if c.offset:  # sub-bin residual of the tokens the model can see, in bins (deterministic, outside RL)
            resid = (xy.reshape(b, 2 * N_WP) - self.centers(tok)) / c.bin_m
            pred = 0.5 * torch.tanh(self.out_off(h).squeeze(-1).float())
            known = ~hidden
            off = ((pred - resid).abs() * known).sum() / known.sum().clamp_min(1)
            loss = loss + off
            out["offset"] = off.detach()
        out["loss"] = loss
        return out

    @torch.no_grad()
    def masked_generate(self, kvs, mem_mask, ego, k: int, steps: int, temperature: float, generator=None):
        """K paths per scene: the goal (last waypoint's x and y) sampled first (greedy when temperature is 0),
        then the most confident hidden tokens, ``steps`` rounds in all. Returns metres [B, k, 8, 2] and the
        mean log-probability of the committed tokens [B, k]."""
        c, b, dev = self.cfg, ego.shape[0], ego.device
        m = 2 * N_WP
        tok = torch.zeros(b, k * m, dtype=torch.long, device=dev)
        known = torch.zeros(b, k * m, dtype=torch.bool, device=dev)
        value = torch.zeros(b, k * m, device=dev)
        logp = torch.zeros(b, k * m, device=dev)
        smask = block_mask(k, m, dev)
        slot = torch.arange(k * m, device=dev) % m
        goal = (slot >= m - 2)[None].expand(b, -1)
        schedule = [int(math.floor(m * math.cos(math.pi / 2 * (s + 1) / steps))) for s in range(steps)]
        schedule[-1] = 0
        for s in range(steps):
            t = torch.full((b,), (m - known[:, :m].sum(1)[0].item()) / m, device=dev)
            h = self.trunk(self.embed_tokens(tok, known), kvs, mem_mask, self.trunk.condition(ego, t), smask)
            lx, ly = self.logits(h)
            scores, picks, vals, lps = [], [], [], []
            for lg, lo in ((lx, c.x_range[0]), (ly, c.y_range[0])):
                lsm = F.log_softmax(lg.double(), -1)  # fp64 Gumbel sampling
                if temperature > 0:
                    g = -torch.log(-torch.log(torch.rand(lsm.shape, dtype=torch.float64, device=dev,
                                                         generator=generator).clamp(1e-20, 1 - 1e-12)))
                    pick = (lsm / temperature + g).argmax(-1)
                else:
                    pick = lsm.argmax(-1)
                lp = lsm.gather(-1, pick[..., None]).squeeze(-1)
                v = (self.local_expectation(lg, pick, lo) if c.l1_weight > 0 else lo + pick.float() * c.bin_m)
                picks.append(pick), lps.append(lp.float()), vals.append(v)
                scores.append(lsm.max(-1).values.float())
            def merge(a, bb):
                out = torch.empty(b, k * m, dtype=a.dtype, device=dev)
                out[:, 0::2], out[:, 1::2] = a, bb
                return out
            pick, lp, val, conf = merge(*picks), merge(*lps), merge(*vals), merge(*scores)
            if s == 0:  # the goal first, sampled; then confidence order
                conf = torch.where(goal, torch.full_like(conf, 1e9), conf)
            conf = torch.where(known, torch.full_like(conf, -1e9), conf)
            n_commit = (m - schedule[s]) - int(known[0, :m].sum().item())
            per_path = conf.view(b, k, m)
            top = per_path.topk(max(n_commit, 0), dim=-1).indices if n_commit > 0 else None
            if top is not None:
                new = torch.zeros(b, k, m, dtype=torch.bool, device=dev).scatter_(-1, top, True).view(b, k * m)
                new &= ~known
                tok = torch.where(new, pick, tok)
                value = torch.where(new, val, value)
                logp = torch.where(new, lp, logp)
                known |= new
        if c.offset:  # one pass with every token known: the deterministic sub-bin offsets
            t = torch.zeros(b, device=dev)
            h = self.trunk(self.embed_tokens(tok, known), kvs, mem_mask, self.trunk.condition(ego, t), smask)
            value = value + c.bin_m * 0.5 * torch.tanh(self.out_off(h).squeeze(-1).float())
        xy = value.view(b, k, N_WP, 2)
        return xy, logp.view(b, k, m).mean(-1)

    # ---- continuous (B, C) ------------------------------------------------------------------------------------
    def cont_forward(self, z, kvs, mem_mask, ego, t, k=1):
        b = z.shape[0]
        x = self.inp(z.reshape(b, k * N_WP, 2)) + self.slot.repeat(k, 1)
        h = self.trunk(x, kvs, mem_mask, self.trunk.condition(ego, t), block_mask(k, N_WP, z.device))
        return self.out(h).reshape(b, k, N_WP, 2)

    def cont_loss(self, kvs, mem_mask, ego, xy):
        b, x1 = xy.shape[0], self.norm(xy)[:, None]
        t = torch.rand(b, device=xy.device)
        if self.cfg.kind == "ddim":
            a = self.alpha_bar(t)[:, None, None, None]
            zt = a.sqrt() * x1 + (1 - a).sqrt() * torch.randn_like(x1)
            loss = F.mse_loss(self.cont_forward(zt, kvs, mem_mask, ego, t).float(), x1)
        else:
            z0 = torch.randn_like(x1) * self.cfg.flow_sigma
            tt = t[:, None, None, None]
            zt = (1 - tt) * z0 + tt * x1
            loss = F.mse_loss(self.cont_forward(zt, kvs, mem_mask, ego, t).float(), x1 - z0)
        return {"loss": loss, "mse": loss.detach()}

    @torch.no_grad()
    def cont_generate(self, kvs, mem_mask, ego, k: int, steps: int, scale: float = 1.0, generator=None):
        """B and C from a start of scale x their training noise: 0 is the zero-noise start, the deterministic
        best guess; the sampler itself is deterministic (DDIM with eta 0, Euler steps)."""
        b, dev = ego.shape[0], ego.device
        shape = (b, k, N_WP, 2)
        if self.cfg.kind == "ddim":  # deterministic DDIM (eta 0), predicting the clean path
            z = torch.randn(shape, device=dev, generator=generator) * scale
            ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
            for i in range(steps):
                t, tn = ts[i], ts[i + 1]
                x0 = self.cont_forward(z, kvs, mem_mask, ego, t.expand(b), k).float()
                a, an = self.alpha_bar(t), self.alpha_bar(tn)
                eps = (z - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-6)
                z = an.sqrt() * x0 + (1 - an).sqrt() * eps
            out = z
        else:  # Euler steps along the learned velocity
            z = torch.randn(shape, device=dev, generator=generator) * self.cfg.flow_sigma * scale
            for i in range(steps):
                t = torch.full((b,), i / steps, device=dev)
                z = z + self.cont_forward(z, kvs, mem_mask, ego, t, k).float() / steps
            out = z
        return self.denorm(out), torch.zeros(b, k, device=dev)

    # ---- anchored (D) -----------------------------------------------------------------------------------------
    def anchor_forward(self, z, kvs, mem_mask, ego, t):
        """z [B, A, 16] noised anchors (normalised) -> clean paths [B, A, 16] and scores [B, A]."""
        x = self.inp(z) + self.anchor_id
        h = self.trunk(x, kvs, mem_mask, self.trunk.condition(ego, t))
        return self.out(h), self.score(h).squeeze(-1)

    def anchor_loss(self, kvs, mem_mask, ego, xy):
        b = xy.shape[0]
        x1 = self.norm(xy).reshape(b, 2 * N_WP)
        anchors = self.anchor_paths[None].expand(b, -1, -1)
        t = torch.rand(b, device=xy.device) * self.cfg.anchor_trunc
        a = self.alpha_bar(t)[:, None, None]
        zt = a.sqrt() * anchors + (1 - a).sqrt() * torch.randn_like(anchors)
        pred, score = self.anchor_forward(zt, kvs, mem_mask, ego, t)
        best = ((anchors - x1[:, None]) ** 2).sum(-1).argmin(-1)
        cls = F.cross_entropy(score.float(), best)
        reg = F.l1_loss(pred.float()[torch.arange(b), best], x1)
        return {"loss": cls + reg, "cls": cls.detach(), "reg": reg.detach()}

    @torch.no_grad()
    def anchor_generate(self, kvs, mem_mask, ego, k: int, steps: int, scale: float = 1.0, generator=None):
        """The top-k refined anchors by score. The anchors start with scale x their training noise: 0 starts
        from the clean anchors, the deterministic best guess. More paths than anchors come from further noise
        draws."""
        a = self.cfg.anchors
        if k > a:
            parts = [self.anchor_generate(kvs, mem_mask, ego, min(a, k - i), steps, scale, generator)
                     for i in range(0, k, a)]
            return torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
        b, dev = ego.shape[0], ego.device
        anchors = self.anchor_paths[None].expand(b, -1, -1)
        ts = torch.linspace(self.cfg.anchor_trunc, 0.0, steps + 1, device=dev)
        a0 = self.alpha_bar(ts[0])
        noise = torch.randn(anchors.shape, device=dev, generator=generator) * scale
        z = a0.sqrt() * anchors + (1 - a0).sqrt() * noise
        for i in range(steps):
            t, tn = ts[i], ts[i + 1]
            x0, score = self.anchor_forward(z, kvs, mem_mask, ego, t.expand(b))
            x0 = x0.float()
            a, an = self.alpha_bar(t), self.alpha_bar(tn)
            eps = (z - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-6)
            z = an.sqrt() * x0 + (1 - an).sqrt() * eps
        order = score.float().argsort(-1, descending=True)[:, :k]
        paths = z.gather(1, order[..., None].expand(-1, -1, 2 * N_WP)).reshape(b, k, N_WP, 2)
        return self.denorm(paths), torch.log_softmax(score.float(), -1).gather(1, order)

    # ---- regression (R) ---------------------------------------------------------------------------------------
    def regression_forward(self, kvs, mem_mask, ego):
        b = ego.shape[0]
        x = self.slot[None].expand(b, -1, -1)
        h = self.trunk(x, kvs, mem_mask, self.trunk.condition(ego, torch.zeros(b, device=ego.device)))
        return self.out(h).float()  # normalised (x, y) per waypoint, [B, 8, 2]

    def regression_loss(self, kvs, mem_mask, ego, xy):
        loss = F.l1_loss(self.regression_forward(kvs, mem_mask, ego), self.norm(xy))
        return {"loss": loss, "l1": loss.detach()}

    # ---- entry points -----------------------------------------------------------------------------------------
    def loss(self, feats, mask, ego, xy):
        kvs, mem_mask = self.trunk.memory(feats, mask)
        if self.cfg.kind == "masked":
            return self.masked_loss(kvs, mem_mask, ego, xy)
        if self.cfg.kind == "anchor":
            return self.anchor_loss(kvs, mem_mask, ego, xy)
        if self.cfg.kind == "regression":
            return self.regression_loss(kvs, mem_mask, ego, xy)
        return self.cont_loss(kvs, mem_mask, ego, xy)

    @torch.no_grad()
    def generate(self, feats, mask, ego, k: int = 1, steps: Optional[int] = None, temperature: float = 1.0,
                 generator=None):
        """[B, k, 8, 2] metres and a per-path score (higher = the generator prefers it).

        ``temperature`` is each method's one sampling setting (docs/HEADS_PLAN.md R7): A's sampling temperature,
        B and C's start-noise multiplier, D's starting-noise multiplier on the anchors. **0 gives every method's
        deterministic best guess**: A's most confident tokens, B and C from the zero-noise start, D from the
        clean anchors. R always gives its one path (repeated k times)."""
        steps = steps or self.cfg.steps
        kvs, mem_mask = self.trunk.memory(feats, mask)
        if self.cfg.kind == "masked":
            return self.masked_generate(kvs, mem_mask, ego, k, steps, temperature, generator)
        if self.cfg.kind == "anchor":
            return self.anchor_generate(kvs, mem_mask, ego, k, steps, temperature, generator)
        if self.cfg.kind == "regression":
            one = self.denorm(self.regression_forward(kvs, mem_mask, ego))
            return one[:, None].expand(-1, k, -1, -1), torch.zeros(ego.shape[0], k, device=ego.device)
        return self.cont_generate(kvs, mem_mask, ego, k, steps, temperature, generator)


def kmeans_anchors(paths: torch.Tensor, n: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """n k-means centres of [N, 16] normalised paths (k-means++ start)."""
    g = torch.Generator().manual_seed(seed)
    x = paths.float()
    centres = [x[torch.randint(0, len(x), (1,), generator=g)].squeeze(0)]
    for _ in range(1, n):
        d = torch.cdist(x, torch.stack(centres)).min(1).values ** 2
        centres.append(x[torch.multinomial(d / d.sum(), 1, generator=g)].squeeze(0))
    c = torch.stack(centres)
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)
        for j in range(n):
            sel = x[assign == j]
            if len(sel):
                c[j] = sel.mean(0)
    return c
