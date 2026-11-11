# Pilot experiment design: which data and training give the best pilot [proposed 2026-11-11]

The researcher's direction (2026-11-11): more epochs, the 4-past-poses prompt (pilot_v1.2) and more data all look like they help, so keep going and design the runs so we can say which data and training give the best results. This file is the plan and its status. Results land in `README.md` (tables) and `results/`.

## Base recipe and what varies
- **Fixed for every new run:** Qwen3.5-0.8B LoRA rank 32, lr 2e-4 cosine with 3% warm-up, ego dropout 0.5 (training batches only: dev and NAVSIM always see the ego state and the past poses), vision encoder frozen, batch 4 × accumulation 4 on the lab RTX 4090, per-epoch snapshots, prompt pilot_v1.2, seed 0.
- **Factors:**
  - Distinct samples: 100k (v1-large), 176k (v1-xl-nd), 490k (v1-full-thin).
  - Epochs: 3, 4 or 6.
  - Prompt: v1.1 or v1.2.
  - Composition: uniform draw, NAVSIM's navtrain scenes, or a navtrain-rich mix.
  - Vision encoder: frozen or trained.
  - Model: 0.8B or 2B.
- **Confound to keep in mind:** the sets differ in mix as well as size. Las Vegas is 72% of v1-xl-nd and 70% of v1-full-thin. Stay-stopped samples are 13.5% and 7.5%. Size comparisons are of the sets as built.

## Measurements (the same for every run)
1. **NAVSIM PDMS** on all 12,146 navtest scenes (v1.1 devkit), paired against the comparison run on the same scenes, with a 95% bootstrap CI over scenes. This is the decision metric.
2. **dev-v2 ADE** on the same seeded 1,000 of dev-v2's 2,000 samples for every run, old and new: the prediction job runs it (`eval_dev_v2.json`). The old dev (dev-v1) stays in `eval.json` for runs whose view carries it.
3. **Train ADE** on 1,000 of the run's own training samples, overall and per epoch: the fit and memorization check behind the instinctual-layer target.

**Decision rule:** a level wins when its paired CI excludes 0 and the gain is larger than the seed-to-seed difference (run N1), with dev-v2 agreeing in direction. One seed per run otherwise; the paired CI covers scene sampling, not training noise, which is why N1 exists.

## Runs

| ID | Run | Distinct samples | Epochs | Prompt | Answers | Against | Status (2026-11-11) |
|---|---|---|---|---|---|---|---|
| B0 | `v1pilot-egodrop50` | 45k | 2 | v1.1 | baseline | | 0.752 |
| B1 | `large-ep3` | 100k | 3 | v1.1 | | | 0.779 |
| B2 | `xlnd-ep4s` | 176k | 4 | v1.1 | | B1: +0.028 (+0.024 to +0.033) | **0.807** |
| P0 | `v1large-p12-unmasked-ep3` | 100k | 3 | v1.2, poses never blinded | variant | B1: +0.005 (+0.001 to +0.010) | 0.784 |
| P1 | `v1large-p12-ep3` | 100k | 3 | v1.2 | prompt at 100k | B1 | 4090 queue 1, training |
| P2 | `xlnd-p12-ep4` | 176k | 4 | v1.2 | prompt at 176k | B2 | queue 2, after V1 |
| S1 | `v1fullthin-p12-ep4` | 490k | 4 | v1.2 | size 176k → 490k at 4 epochs | P2 | queue 1, after P1 (~65 h) |
| E1 | `xlnd-p12-ep6` | 176k | 6 | v1.2 | epochs 4 → 6 at 176k | P2 | queue 2, after P2 |
| E2 | `v1large-p12-ep6` | 100k | 6 | v1.2 | epochs 3 → 6 at 100k; size at 6 epochs | P1; E1 | queue 1, last |
| N1 | `v1large-p12-ep3-s1` | 100k | 3 | v1.2, seed 1 | run-to-run noise | P1 | queue 1, after S1 |
| C1 | `v1nt-p12-ep3` | 81k navtrain | 3 | v1.2 | composition: NAVSIM's scenes vs a uniform draw | P1 | trained; NAVSIM score pending |
| C2 | `v1ntmix-p12-ep4` | 260k navtrain-rich | 4 | v1.2 | composition at a larger size | P2, S1 | queue 1, after N1 |
| V1 | `v1large-p12-visfull-ep3` | 100k | 3 | v1.2, vision trained | vision encoder at 100k | P1 | queue 2, training |
| V2 | `xlnd-visfull-ep4s` (H100) | 176k | 4 | v1.1, vision trained | vision encoder at 176k | B2 | Anvil `ai`, half done |
| M1 | `2b-v1large-p12-ep3` | 100k | 3 | v1.2, 2B | model size | P1 | queue 2, after E1 |
| M0 | `2b-large-ep3` (H100) | 100k | 3 | v1.1, 2B | model size | B1 | Anvil `ai`, part done |

## How the pairs combine
- **Prompt (P1 vs B1, P2 vs B2):** the same question at two sizes. Adopt v1.2 for good if both agree in sign.
- **Epochs (E2 vs P1, E1 vs P2) and size (S1 vs P2 at 4 epochs, E1 vs E2 at 6 epochs):** sizes {100k, 176k, 490k} × epochs {3–4, 6}. That separates "more distinct data" from "more passes" without running the whole grid. Per-epoch dev and train curves come free from the snapshots. An epoch-k snapshot of a longer run is not the same as a k-epoch run, because its learning rate has not decayed. So snapshots show trends, and only final adapters are scored on NAVSIM.
- **Distinct data vs repetition at matched samples seen:** S1's epoch-2 snapshot (980k samples seen) against E1 (1.06M). Score that snapshot on NAVSIM if the dev curves leave it open.
- **Composition (C1 vs P1; C2 vs P2 and S1):** whether NAVSIM's own scenes beat more of the uniform draw.
- **2B** stays at key points. Once data and epochs are settled, the 2B key-point run should use the winning set instead of v1-large.

## Order and compute
- Each 4090 queue trains about 8.4 samples/s (0.8B, both queues sharing the GPU), so roughly 0.73M samples a day per queue.
- Anvil (and with it the only route to the 4090) is down from Nov 16 04:00 to Nov 18 06:00. The queues run on their own through it. Scoring and copies wait until it is back.
