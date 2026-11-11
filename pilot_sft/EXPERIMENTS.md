# Pilot experiment design: which data and training give the best pilot [proposed 2026-11-11]

The researcher's direction (2026-11-11):
- More epochs, the 4-past-poses prompt (pilot_v1.2) and more data all look like they help, so keep going, and design the runs so we can say which data and training give the best results.
- Then, the same day: "176k should always work better than 100k so we should not run 100k when we can run 176k. The larger the better." The goal is the best capability, as fast as possible, while still knowing what made it best.

So every new run uses at least the 176k set, and the factors are tested there: vision encoder, epochs, prompt and seed. The 490k set carries the best-capability runs. Runs on 100k are only the ones already finished or nearly finished. This file is the plan and its status. Results land in `README.md` (tables) and `results/`.

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
| P1 | `v1large-p12-ep3` | 100k | 3 | v1.2 | prompt at 100k | B1 | queue 1, training: was 77% done when 100k was dropped, so it finishes; scored tonight |
| C1 | `v1nt-p12-ep3` | 81k navtrain | 3 | v1.2 | composition: NAVSIM's own scenes | P1 | trained; NAVSIM score pending |
| **P2** | `xlnd-p12-ep4` | 176k | 4 | v1.2 | prompt at 176k; the frozen 176k reference | B2 | queue 2, training since 09:13; ~Thu 11:00 |
| **S1** | `v1fullthin-p12-ep4` | 490k | 4 | v1.2 | size 176k → 490k; best-capability candidate | P2 | queue 1, after P1; ~Sat 10:00 |
| **VX** | `xlnd-p12-visfull-ep4` | 176k | 4 | v1.2, vision trained | vision encoder | P2 | queue 2, after P2; ~Fri 20:00 |
| **E1** | `xlnd-p12-ep6` | 176k | 6 | v1.2 | epochs 4 → 6 | P2 | queue 2, after VX; ~Sun 10:00 |
| **N1** | `xlnd-p12-ep4-s1` | 176k | 4 | v1.2, seed 1 | run-to-run noise (the decision threshold) | P2 | queue 2, after E1; ~Mon (scored after the outage) |
| S2 | `v1fullthin-p12-visfull-ep4` | 490k | 4 | v1.2, vision trained | best capability, if VX wins | S1 | queue 1, after S1, ~81 h; TENTATIVE: swapped for the epochs winner if VX loses |
| M2 | `2b-xlnd-p12-ep4` | 176k | 4 | v1.2, 2B | model size with more data | P2 | queue 2, last; TENTATIVE: GPU memory beside queue 1 is checked first |
| V2 | `xlnd-visfull-ep4s` (H100) | 176k | 4 | v1.1, vision trained | vision encoder at v1.1 | B2 | Anvil `ai`, over half done; continuation queued |
| — | dropped 2026-11-11 (100k) | | | | | | `v1large-p12-visfull-ep3` stopped at 14%; the 100k seed replicate, v1-large 6 epochs, v1-nt-mix and 2B on 100k (4090 and H100) never started |

## How the pairs combine
- **Every factor against P2.** P2 (176k, 4 epochs, v1.2, frozen, seed 0) is the reference. VX changes only the vision encoder, E1 only the epochs, N1 only the seed, S1 only the set (490k), M2 only the model. P2 against B2 is the prompt at 176k, and P1 against B1 the prompt at 100k.
- **Noise:** N1 against P2 is the seed-to-seed difference. A factor's gain counts only above it.
- **Best capability:** S1 is the expected best. Its successor S2 combines the winners: vision trained if VX beats P2, otherwise the epochs E1 points to.
- **Per-epoch curves:** dev and train curves come free from the snapshots. An epoch-k snapshot of a longer run is not the same as a k-epoch run, because its learning rate has not decayed. So snapshots show trends, and only final adapters are scored on NAVSIM.
- **Distinct data vs repetition at matched samples seen:** S1's epoch-2 snapshot (980k samples seen) against E1 (1.06M). Score that snapshot on NAVSIM if the dev curves leave it open.
- **2B** stays at key points: M2 on the 176k set, not 100k.

## Can the data grow past 490k?
**Within nuPlan, no.** v1-full-thin (490k) is every usable front-camera frame of OpenScene's trainval logs at 0.5 s, with long standstills thinned. v1-full (676k) adds back 186k standstill frames, which repeat the same image and path. Beyond that, the options are (sizes and terms from `docs/training_data/DATASETS.md`; [proposed], each needs the researcher's sign-off before any download or build):

| Source | What it adds | Size | Cost and risk |
|---|---|---|---|
| Mirror augmentation (flip the image, negate y, heading and yaw rate, swap left and right) | A mirrored copy of every sample, no download | ×2 of any set | Code in training only. Mirroring flips the driving side: right-hand traffic becomes left-hand, which may cost direction compliance. Test it at 176k first |
| nuReasoning (Motional, Sep 2026) | NAVSIM-like inputs (8 cameras, ego, route commands, map), the nuPlan maker's data | 20k × 20 s = 105 h at 10 Hz (V); at 0.5 s, with 1.5 s of history and 4 s of future inside each clip, about 29 anchors per clip, so roughly 580k samples (D) | Overlap with nuPlan logs, and so with navtest, is unknown and must be checked before any use. Rig differs (D). Non-commercial terms (V) |
| nuPlan sensor blobs at 10 Hz | Frames between our 0.5 s anchors, same drives | 5× the frames; 19 TB (V) | Mostly near-duplicates of frames we have. Worth it only if S1 shows denser sampling helps |
| L2D (Yaak) | 6 cameras, GPS, IMU, CAN, route waypoints | 100k episodes, 500+ h (V) | Germany, a different rig, some learner drivers (V); licence Apache-2.0 (R) |
| CoVLA, ZOD, Waymo, WOD-E2E, Argoverse 2, nuScenes | Front camera + ego path | 30k to ~600k frames each (R/D) | New rigs and cities, left-hand traffic (CoVLA), a converter per dataset |

## Order and compute
- Each 4090 queue trains about 8.4 samples/s (0.8B, both queues sharing the GPU), so roughly 0.73M samples a day per queue.
- Anvil (and with it the only route to the 4090) is down from Nov 16 04:00 to Nov 18 06:00. The queues run on their own through it. Scoring and copies wait until it is back.
