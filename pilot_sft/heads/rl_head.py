"""The matched short RL stage of docs/HEADS_PLAN.md (phase 4) for one path generator: one GPU, with rewards from
``pdm_server.py``.

    python pilot_sft/heads/rl_head.py --run <exp>/heads/<sft run> --weights epoch_<k> --out <exp>/heads/<run>-rl-s0 \\
        --rl-scenes <view of the RL scenes> --steps <S> --scale <x> --seed 0

**Each update (the settings are the plan's, the same for every method):**
- Take 32 scenes: the RL scenes, shuffled once with seed 0, in the same order for every method and seed. They are
  rows of the stored training features, and their tokens key the navtrain metric cache.
- Sample 8 candidate paths per scene from the method's stochastic policy (``Head.rl_sample``) at its R7 setting.
- Reward each candidate with NAVSIM v1.1 PDMS, from the server.
- Compute group-relative advantages: (reward - group mean) / (group SD + 1e-4). A scene whose 8 rewards are equal
  gives no gradient, and is counted.
- One gradient step on -mean(advantage x log p) + kl_beta x mean(KL), with log p the exact log-probability of
  what was sampled (``Head.rl_logprob``) and KL its exact KL to the frozen starting policy along that path:
  AdamW, betas (0.9, 0.95), no weight decay, 50 warm-up steps then constant, gradient clip 1.0, bf16.
- The learning rate, kl_beta, and candidates x scenes per update are the R12 recipe's (RL0 = 1e-5, no KL,
  8 x 32).
- R13's options (all off by default, which is the registered recipe):
  - ``--adv dr``: the advantage is reward - group mean, without the SD division (Dr. GRPO).
  - ``--bc-balance 1``: adds lambda x the method's own supervised loss on the logged paths of the update's scenes
    (behaviour cloning). lambda is fixed so that, at the starting weights, this term's gradient norm equals the RL
    term's: each the mean over the first 4 batches, from a separate noise seed. It is logged in run.json.
  - ``--lp-sd-floor 0.1``: every Gaussian step's SD is floored at 0.1 inside the log-probability and KL (D, B, C);
    sampling is unchanged.

**Outputs:**
- Every ``--save-every`` updates, ``rl_<u>.pt`` in the format of the supervised epochs, so ``predict_head.py
  --weights rl_<u>`` loads it.
- ``checkpoint.pt`` for ``--resume``.
- ``log.jsonl``: the reward and its sub-scores, the best-of-8 reward, the share of equal-reward groups, scorer
  errors, the loss, the gradient norm and the time split.
- ``run.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
from pdm_client import PDMClient  # noqa: E402
from predict_head import load_head  # noqa: E402
from train_head import to_torch  # noqa: E402

SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress",
        "comfort", "driving_direction_compliance"]


def rl_rows(store: features.Store, rl_view: str, limit_tokens=None):
    """Rows of the stored training features that are RL scenes (the view's train ids), and their bare tokens."""
    with open(os.path.join(rl_view, "train.jsonl"), encoding="utf-8") as f:
        ids = {json.loads(l)["id"] for l in f if l.strip()}
    rows = [i for i, x in enumerate(store.ids) if x in ids]
    if limit_tokens is not None:
        rows = [i for i in rows if store.ids[i].rsplit("/", 1)[-1] in limit_tokens]
    return rows


def save(path: str, obj) -> None:
    torch.save(obj, path + ".part")
    os.replace(path + ".part", path)


def advantages(r: torch.Tensor, kind: str) -> torch.Tensor:
    """Group-relative advantages [B, G] from rewards [B, G]. 'grpo' (as registered): (r - mean) / (SD + 1e-4);
    'dr' (R13, Dr. GRPO): r - mean. A group whose rewards are all equal gives zeros either way."""
    std = r.std(1, keepdim=True)
    centred = r - r.mean(1, keepdim=True)
    if kind == "dr":
        return torch.where(std > 0, centred, torch.zeros_like(r))
    return torch.where(std > 0, centred / (std + 1e-4), torch.zeros_like(r))


def grad_norm(params) -> float:
    g = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.norm(torch.stack(g))) if g else 0.0


def bc_loss(head, b, device: str, seed: int) -> torch.Tensor:
    """R13's imitation term: the method's own supervised loss on the logged paths of the batch, as in training (bf16).
    Its random draws (masks, noise levels) use their own seed, so the sampling generator is untouched."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            return head.loss(b["feats"], b["mask"], b["ego"], b["xy"])["loss"].float()


def measure_bc_weight(head, store, order, args, device: str, client, batches: int = 4):
    """R13: the BC weight that makes the imitation term's gradient norm equal ``args.bc_balance`` x the RL term's at
    the starting weights, each the mean over the first ``batches`` batches of the scene order. The candidates come
    from a separate generator, so the first update's candidates match the recipes without BC. No weight changes."""
    gen = torch.Generator(device=device).manual_seed(5000 + args.seed)
    params = [p for p in head.parameters() if p.requires_grad]
    rl_n, bc_n = [], []
    for u in range(batches):
        idx = order[u * args.scenes_per_update : (u + 1) * args.scenes_per_update]
        b = to_torch(store.batch(idx), device)
        tokens = [store.ids[i].rsplit("/", 1)[-1] for i in idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            xy, trace = head.rl_sample(b["feats"], b["mask"], b["ego"], args.g, args.steps, args.scale, gen)
        paths = xy.float().cpu().numpy().round(3)
        res = client.score([(t, paths[i].tolist()) for i, t in enumerate(tokens)])
        r = torch.tensor([[c["score"] for c in per] for per in res], dtype=torch.float32, device=device)
        head.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            lp = head.rl_logprob(b["feats"], b["mask"], b["ego"], trace, lp_floor=args.lp_sd_floor)
        (-(advantages(r, args.adv) * lp.float()).mean()).backward()
        rl_n.append(grad_norm(params))
        head.zero_grad(set_to_none=True)
        bc_loss(head, b, device, 9000 + u).backward()
        bc_n.append(grad_norm(params))
        head.zero_grad(set_to_none=True)
    w = args.bc_balance * float(np.mean(rl_n)) / max(float(np.mean(bc_n)), 1e-12)
    return w, [round(x, 5) for x in rl_n], [round(x, 5) for x in bc_n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="the supervised run to start from")
    p.add_argument("--weights", required=True, help="epoch_<k>: its navdev best-of-6 epoch (R10)")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=os.path.expanduser("~/navsim_workspace/head_cache/h1_l12-18-24/train"))
    p.add_argument("--rl-scenes", required=True, help="the view whose train.jsonl ids are the RL scenes (v1-nt)")
    p.add_argument("--only-tokens", default=None, help="a file of tokens to restrict to (the throughput trial)")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--scale", type=float, required=True, help="the method's R7 sampling setting")
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--scenes-per-update", type=int, default=32)
    p.add_argument("--g", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.0,
                   help="weight of the exact KL to the frozen starting policy along each sampled path (R12); 0 = none")
    p.add_argument("--adv", choices=("grpo", "dr"), default="grpo",
                   help="grpo: (r - mean) / (SD + 1e-4), as registered; dr: r - mean (R13, Dr. GRPO)")
    p.add_argument("--bc-balance", type=float, default=0.0,
                   help="R13: add the supervised loss on the logged paths, weighted so its gradient norm is this "
                        "multiple of the RL term's at the start; 0 = none")
    p.add_argument("--lp-sd-floor", type=float, default=0.0,
                   help="R13: floor on every Gaussian step's SD inside the log-probability and KL; 0 = exact")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0, help="the RL seed: sampling noise (the scene order is always seed 0)")
    p.add_argument("--port", type=int, default=47411)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    src = torch.load(os.path.join(args.run, f"{args.weights}.pt"), map_location="cpu", weights_only=False)
    head = load_head(os.path.join(args.run, f"{args.weights}.pt"), device)
    head.train()
    ref = None
    if args.kl_beta > 0:  # the frozen starting policy, for the KL term
        ref = load_head(os.path.join(args.run, f"{args.weights}.pt"), device)
        for q in ref.parameters():
            q.requires_grad_(False)
    store = features.Store(args.cache, head.cfg.layers)
    only = None
    if args.only_tokens:
        with open(args.only_tokens, encoding="utf-8") as f:
            only = {l.strip() for l in f if l.strip()}
    rows = rl_rows(store, args.rl_scenes, only)
    order = random.Random(0).sample(rows, len(rows))  # the same scene order for every method and seed
    need = max(args.updates, 4 if args.bc_balance > 0 else 0) * args.scenes_per_update  # BC's weight reads 4 batches
    while len(order) < need:  # more updates than scenes: further passes, each its own fixed permutation
        order += random.Random(len(order)).sample(rows, len(rows))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    gen = torch.Generator(device=device).manual_seed(1000 + args.seed)
    u0, bc_w = 0, 0.0
    ck = os.path.join(args.out, "checkpoint.pt")
    if args.resume and os.path.exists(ck):
        c = torch.load(ck, map_location="cpu", weights_only=False)
        head.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        sched.load_state_dict(c["scheduler"])
        gen.set_state(c["generator"])
        u0, bc_w = c["update"], c.get("bc_weight", 0.0)
        print(f"resumed at update {u0}", flush=True)
    run = {"args": vars(args), "start": {"run": args.run, "weights": args.weights, "dev": src.get("dev")},
           "config": head.cfg.to_dict(), "rl_scenes": len(rows), "scenes_used": need,
           "passes": round(need / max(1, len(rows)), 3), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    print(f"RL of {head.cfg.kind} from {args.run}/{args.weights}: {len(rows)} RL scenes, {args.updates} updates x "
          f"{args.scenes_per_update} scenes x {args.g} candidates ({run['passes']} passes)", flush=True)

    client = PDMClient(args.port)
    if args.bc_balance > 0:
        if bc_w <= 0:  # measured once, at the starting weights; a resumed run keeps its weight
            bc_w, rl_n, bc_n = measure_bc_weight(head, store, order, args, device, client)
            run["bc"] = {"weight": bc_w, "balance": args.bc_balance, "rl_grad_norms": rl_n, "bc_grad_norms": bc_n,
                         "rule": "docs/HEADS_PLAN.md R13"}
            print(f"BC weight {bc_w:.6g} (RL gradient norms {rl_n}, BC {bc_n})", flush=True)
        run.setdefault("bc", {"weight": bc_w, "balance": args.bc_balance, "rule": "docs/HEADS_PLAN.md R13"})
        with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1)
    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")
    weights = {k: src[k] for k in ("config", "stats")}
    for u in range(u0, args.updates):
        t0 = time.time()
        idx = order[u * args.scenes_per_update : (u + 1) * args.scenes_per_update]
        b = to_torch(store.batch(idx), device)
        tokens = [store.ids[i].rsplit("/", 1)[-1] for i in idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            xy, trace = head.rl_sample(b["feats"], b["mask"], b["ego"], args.g, args.steps, args.scale, gen)
        t1 = time.time()
        paths = xy.float().cpu().numpy().round(3)
        res = client.score([(t, paths[i].tolist()) for i, t in enumerate(tokens)])
        t2 = time.time()
        r = torch.tensor([[c["score"] for c in per] for per in res], dtype=torch.float32, device=device)
        errors = sum(1 for per in res for c in per if "error" in c)
        std = r.std(1, keepdim=True)
        adv = advantages(r, args.adv)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            if ref is not None:
                lp, kl = head.rl_logprob(b["feats"], b["mask"], b["ego"], trace, ref, lp_floor=args.lp_sd_floor)
            else:
                lp = head.rl_logprob(b["feats"], b["mask"], b["ego"], trace, lp_floor=args.lp_sd_floor)
                kl = torch.zeros(1, device=device)
        rl_loss = -(adv * lp.float()).mean() + args.kl_beta * kl.float().mean()
        bc = bc_loss(head, b, device, 100003 * (args.seed + 1) + u) if bc_w > 0 else None
        loss = rl_loss + bc_w * bc if bc is not None else rl_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        t3 = time.time()
        rec = {"update": u + 1, "reward": round(float(r.mean()), 4), "best_of_g": round(float(r.max(1).values.mean()), 4),
               "equal_reward_groups": round(float((std.squeeze(1) == 0).float().mean()), 3), "scorer_errors": errors,
               "loss": round(float(loss), 5), "kl": round(float(kl.float().mean()), 5), "grad_norm": round(float(gn), 3),
               **({"bc": round(float(bc), 5)} if bc is not None else {}),
               "lr": sched.get_last_lr()[0],
               **{k: round(float(np.mean([c.get(k, 0.0) for per in res for c in per])), 4) for k in SUBS},
               "s_sample": round(t1 - t0, 2), "s_score": round(t2 - t1, 2), "s_update": round(t3 - t2, 2),
               "paths_per_s": round(len(tokens) * args.g / max(t2 - t1, 1e-6), 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if (u + 1) % 10 == 0 or u == u0:
            print(json.dumps(rec), flush=True)
        if (u + 1) % args.save_every == 0 or u + 1 == args.updates:
            save(os.path.join(args.out, f"rl_{u + 1}.pt"), {**weights, "model": head.state_dict(), "epoch": f"rl_{u + 1}"})
            save(ck, {"model": head.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                      "generator": gen.get_state(), "update": u + 1, "bc_weight": bc_w})
    client.close()
    run["finished_updates"] = args.updates
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)


if __name__ == "__main__":
    main()
