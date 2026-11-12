# Pilot experiment design: which data and training give the best pilot [proposed 2026-11-11]

The researcher's direction (2026-11-11):
- More epochs, the 4-past-poses prompt (pilot_v1.2) and more data all look like they help, so keep going, and design the runs so we can say which data and training give the best results.
- Then, the same day: "176k should always work better than 100k so we should not run 100k when we can run 176k. The larger the better." The goal is the best capability, as fast as possible, while still knowing what made it best.

- And, later that day: "We are assuming more data in the dataset is better and 4 past poses is better (although blinding can be tested)." Blinding (ego dropout) and epochs are the factors to push, finding where each stops helping. The 490k result is wanted before the Nov 16 outage if possible, without skipping or weakening any comparison.

So every new run uses at least the 176k set and the v1.2 prompt, and the factors are tested there: blinding, epochs, vision encoder and seed. Blinding so far: 0.8 lost to 0.5 on NAVSIM at 45k (−0.010, CI −0.014 to −0.006). 0 against 0.5 was only compared on dev, on the 11k smoke set, where 0.5 cost 0.05 m on full input. So the levels bracket 0.5: 0.25 and 0.7. The 490k set carries the best-capability runs. Runs on 100k are only the ones already finished or nearly finished. This file is the plan and its status. Results land in `README.md` (tables) and `results/`.

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

**Dev is not NAVSIM (V, 2026-11-11):** v1-nt has the worst dev ADE and the best PDMS. Our dev draws from all frames, while navtest keeps only NAVSIM's non-trivial scenes. So compare dev only between runs on the same training mix. The 743 navtrain tokens inside dev-v2's 2,000 could make a NAVSIM-matched dev subset [proposed].

**Decision rule:** a level wins when its paired CI excludes 0 and the gain is larger than the seed-to-seed difference (run N1), with dev-v2 agreeing in direction. One seed per run otherwise; the paired CI covers scene sampling, not training noise, which is why N1 exists.

## Schedule from 2026-11-11 evening [agreed]
The researcher: "Ok keep going I want to see how far we get with just this. Then we will introduce things like action heads and RL. As well as narrators I think." Two datasets, both with prompt v1.2: the 490k set (v1-full-thin) and **v1-hard** (236,133: every navtrain frame 96,353, the moving frames at least as hard as navtrain's easiest quarter 132,344, standstills thinned to one per 20 s 7,436; `data_pipeline/v1/derived_sets.py`, view cut from the 490k view by `subset_view.py`). The 176k set is no longer one of them, so P2 is parked (checkpoint kept) and its factor runs are replaced by minis.

**Revised 2026-11-11 ~19:50 [agreed]:**
- **Researcher's words:** "The prompt is fine. I like its minimal. Go for 7 minis... Right after that we should train the 490k versions. We should also test of Navsim the results of the minis as we get them."
- **Prompt:** v1.2 stays.
- **Minis (7 of the 18):** 4 references, on both sets at seeds 0 and 1, give the dataset answer and the noise floor. 3 settings run on v1-hard only: ego dropout 0.25, ego dropout 0, and the vision encoder trained.
- **Why the rest were cut:**
  - At 2,500 steps the learning-rate minis favour the higher rate whatever is true for a full run.
  - Rank 64 is the setting least likely to matter at this size.
  - Running every setting on both sets doubled the cost.
- **Win rule:** a setting wins when its paired NAVSIM gain over `mini-hard-base-s0` is larger than the seed-to-seed gap. That gap is hard-s1 vs hard-s0, with thin-s1 vs thin-s0 as a second estimate.
- **Full runs right after the minis, one per queue:** the 490k set (S3) and v1-hard (H1), both 4 epochs.
  - Each starts at the standard recipe as soon as its queue frees, so the GPU doesn't idle while the mini scores come in. On queue 1 that is S1 resuming at its checkpoint.
  - If a setting wins, both restart with it. If none does, S1 *is* S3.
- **After them:** F1 (the 490k adapter + 1 epoch of v1-nt), then H2 (v1-hard, 8 epochs).
- **Every mini is scored on NAVSIM as it lands.**

| Queue | Order (2026-11-11 19:50) |
|---|---|
| 1 | mini-hard-base-s0 (running), mini-hard-base-s1, mini-hard-visfull, then the full 490k run (`v1fullthin-p12-ep4`, provisional standard recipe) + predictions, then F1 + predictions |
| 2 | mini-thin-base-s0 (running), mini-thin-base-s1, mini-hard-drop25, mini-hard-drop0, older adapters' dev-v2 evals, then H1 (`v1hard-p12-ep4`, provisional standard recipe) + predictions |

### Mini results (2026-11-12, V)
NAVSIM v1.1 PDMS on all 12,146 navtest scenes (0 failed, 0 fallbacks, 12,146 unique predictions each). Each mini: 0.8B, prompt v1.2, 2,500 steps (40,000 samples). Paired differences use a 95% bootstrap CI over scenes; aggregates are in `../docs/navsim_eval/results/qwen35-0.8b-mini-*`. Dev and train ADE come from 200 samples (dev = dev-v2, drawn from all frames).

| Mini | PDMS (95% CI) | Paired vs `hard-base-s0` | Scenes scoring 0 | Dev ADE | Train ADE |
|---|---|---|---|---|---|
| hard, vision encoder trained | **0.769 (0.762–0.775)** | **+0.036 (+0.031 to +0.041)**; vs hard-s1 +0.044 | 15.5% | 0.792 | 0.736 |
| hard, standard, seed 0 | 0.733 (0.726–0.739) | — | 18.8% | 0.821 | 0.742 |
| hard, standard, seed 1 | 0.725 (0.718–0.732) | −0.007 (−0.012 to −0.003): the seed gap | 19.4% | 0.850 | 0.804 |
| hard, ego dropout 0.25 | 0.725 (0.718–0.731) | −0.008 (−0.013 to −0.003); vs hard-s1 −0.000 | 19.6% | 0.852 | 0.809 |
| hard, ego dropout 0 (never blinded) | 0.716 (0.710–0.723) | −0.016 (−0.022 to −0.011); vs hard-s1 −0.009 | 20.3% | 0.848 | 0.785 |
| 490k, standard, seed 0 | 0.697 (0.690–0.704) | −0.036 (−0.041 to −0.031) | 21.6% | 0.740 | 0.700 |
| 490k, standard, seed 1 | 0.688 (0.681–0.695) | vs hard-s1: −0.037 (−0.042 to −0.032); vs thin-s0: −0.009 (−0.014 to −0.004) | 22.5% | 0.784 | 0.715 |

- **Dataset (D):** at equal compute (40,000 samples), v1-hard beats the 490k set by 0.036 at seed 0 and 0.037 at seed 1. The seed-to-seed gap is 0.007 (hard) and 0.009 (490k), so the dataset effect is about 4–5× the noise.
  - Dev ADE ranks the sets the other way (490k 0.74–0.78, v1-hard 0.82–0.85), because dev is drawn from all frames, like the 490k set. As before, dev ADE only compares runs within one training mix.
  - Caveat: in 40k samples the v1-hard draw holds ~41% navtrain against ~20% for the 490k draw, so this measures composition at a fixed budget. The full runs (4 epochs each, and H2 at equal compute) test it at length.
- **Settings:**
  - **Vision encoder trained wins:** +0.036 vs seed 0 and +0.044 vs seed 1, 4–6× the seed gap, with every sub-score up (NC 0.966 vs 0.956, DAC 0.872 vs 0.845, EP 0.716 vs 0.682). It trains 122M parameters instead of 22M and is only 5–8% slower.
  - **Ego dropout 0.25 does not win:** within the seed gap.
  - **Never blinding (ego dropout 0) loses:** −0.016 vs seed 0, about 2× the seed gap. Some blinding helps; 0.25 vs 0.5 makes no difference, so 0.5 stays.
- **Decision (agreed rule):** both full runs restart with the vision encoder trained (2026-11-12 09:05).
  - S1 (the standard recipe) is parked at step 7,285 with its checkpoint kept.
  - S3 = `v1fullthin-p12-visfull-ep4` (queue 1) and H1 = `v1hard-p12-visfull-ep4` (queue 2, after the last mini and the dev-v2 evals).
  - Each trains (`TRAIN_ONLY=1`), then predicts navtest, then runs the remaining evals, so the NAVSIM score comes first.
  - F1 now starts from S3's adapter.
  - If ego dropout 0 also wins, both restart with it too. (It lost; 0.5 stays.)
- **Researcher, 2026-11-12 ~10:00:** "We run the full complete training on v1 hard I think. Then we run full training on full 490k. Make sense? Once we get one trained we should test it." Also: "You can train one at a time that is fine. It should train a bit faster if it fits too."
  - One GPU job at a time on queue 1 (E-cores 16-31). Queue 2 stays empty.
  - **Order:** H1 train (310, started 10:23) → H1 navtest predictions + dev-v2 (311) → S3 train (320, resumes at its step-2,300 checkpoint) → S3 predictions (321) → the long evals of both (330/331) → older adapters' dev-v2 evals (340).
  - F1 and H2 are parked until these two results are in.

The plan before this revision:

| ID | Run | Queue | What it answers | Est. scored |
|---|---|---|---|---|
| S1 | `v1fullthin-p12-ep4` (490k, standard recipe, 4 epochs, snapshots) | 1 | the 490k set; its epoch snapshots give an epochs curve | Sat Nov 14 |
| m* | 18 minis (below) | 2 | which settings win, on both sets | Wed–Fri, as each lands |
| F1 | S1's adapter + 1 epoch of v1-nt (`INIT_ADAPTER`) | 1, after S1 | big set first, then navtrain (the WAM-Diff / WAM-Flow / DriveVLA-W0 recipe) vs either alone | Sat Nov 14 |
| H1 | v1-hard, winning settings, 4 epochs | 2, after the minis | v1-hard at its best settings | Sun Nov 15 |
| S3 | 490k, winning settings, 4 epochs | 1, after F1 | the 490k set at its best settings: against H1 at equal epochs | Wed Nov 18 (after the outage) |
| H2 | v1-hard, winning settings, 8 epochs | 2, after H1 | equal compute to S3 (1.89M vs 1.96M samples seen): which data per GPU-hour; and v1-hard's epochs | Wed–Thu Nov 18–19 |

**Minis:** 2,500 optimizer steps (40,000 samples, a random draw of the set; the cosine schedule ends at the mini's last step, `MAX_STEPS`), small post-training evals (`EVAL_SAMPLES=200`), then all 12,146 navtest scenes and the same 1,000 dev-v2 samples. About 2.4 h each on one queue. Every setting runs on both sets, in this order: the standard recipe with seeds 0 and 1 (the noise floor, and a first dataset hint); ego dropout 0.25, 0.7 and 0 (standard 0.5); vision encoder trained; learning rate 1e-4 and 4e-4 (standard 2e-4); LoRA rank 64 (alpha 128, the same alpha/rank ratio). A setting wins when its paired NAVSIM gain over the seed-0 mini is larger than the seed-to-seed gap, on both sets. If the gap is too large to separate settings, the minis get longer. Minis cannot test epochs: S1's snapshots and H1 vs H2 do. A small effect at mini length may not hold at full length; the full runs check the winners.

**Applied on the desktop 2026-11-11 18:12** (after it hung ~14:31–14:54 and was rebooted at 17:44): queue 1 runs P1's remaining predictions, then S1, then F1 (`qwen35-0.8b-v1fullthin-p12-ep4-then-nt-ep1`, standard recipe from S1's adapter). Queue 2 runs the 18 minis as `qwen35-0.8b-mini-{thin,hard}-{base-s0, base-s1, drop25, drop70, drop0, visfull, lr1e-4, lr4e-4, rank64}`, each setting on the 490k set, then on v1-hard, then the older adapters' dev-v2 evals. The v1-hard view was cut on the desktop from its 490k view; its train and dev files match Anvil's sha256. P2 and its prediction job are parked with the step-11,000 checkpoint. The 176k factor runs and S2 are set aside, not deleted. First mini: ~0.55 steps/s while sharing the GPU with a prediction job.

## Earlier runs

| ID | Run | Distinct samples | Epochs | Prompt | Answers | Against | Status (2026-11-11) |
|---|---|---|---|---|---|---|---|
| B0 | `v1pilot-egodrop50` | 45k | 2 | v1.1 | baseline | | 0.752 |
| B1 | `large-ep3` | 100k | 3 | v1.1 | | | 0.779 |
| B2 | `xlnd-ep4s` | 176k | 4 | v1.1 | | B1: +0.028 (+0.024 to +0.033) | **0.807** |
| P0 | `v1large-p12-unmasked-ep3` | 100k | 3 | v1.2, poses never blinded | variant | B1: +0.005 (+0.001 to +0.010) | 0.784 |
| P1 | `v1large-p12-ep3` | 100k | 3 | v1.2 | prompt at 100k | B1 | **0.786**; B1: +0.007 (+0.003 to +0.012); P0: +0.002 (−0.003 to +0.006), so blinding the poses line makes no difference. Predictions resumed after the desktop hang |
| C1 | `v1nt-p12-ep3` | 81k navtrain | 3 | v1.2 | composition: NAVSIM's own scenes | P1 | **0.829, best**; +0.021 vs B2 (+0.017 to +0.026) with half the data. Dev ADE ranked it worst |
| **P2** | `xlnd-p12-ep4` | 176k | 4 | v1.2 | prompt at 176k; the frozen 176k reference | B2 | parked 2026-11-11 evening (176k is no longer one of the two sets); checkpoint kept, so it resumes exactly if wanted |
| **S1** | `v1fullthin-p12-ep4` | 490k | 4 | v1.2 | size 176k → 490k; best-capability candidate | P2 | queue 1, after P1's predictions (~19:00 2026-11-11); ~122k steps at ~0.5/s, predictions ~Sat afternoon, scored before the outage |
| VX, E1, D25, N1, D70 | `xlnd-p12-*` (vision, 6 epochs, dropout 0.25 / 0.7, seed 1) | 176k | | | | P2 | dropped 2026-11-11 evening: the same questions now run as minis on the 490k set and v1-hard |
| S2, M2 | 490k with vision trained; 2B on 176k | | | | | | dropped: S3 takes the winning settings; 2B waits for a dataset winner |
| V2 | `xlnd-visfull-ep4s` (H100) | 176k | 4 | v1.1, vision trained | vision encoder at v1.1 | B2 | Anvil `ai`, over half done; continuation queued |
| — | dropped 2026-11-11 (100k) | | | | | | `v1large-p12-visfull-ep3` stopped at 14%; the 100k seed replicate, v1-large 6 epochs, v1-nt-mix and 2B on 100k (4090 and H100) never started |

## How the pairs combine
- **Every factor against P2.** P2 (176k, 4 epochs, v1.2, ego dropout 0.5, frozen, seed 0) is the reference. VX changes only the vision encoder, E1 only the epochs, D25 and D70 only the blinding rate, N1 only the seed, S1 only the set (490k), M2 only the model.
- **Blinding:** D25, P2 and D70 give three points (0.25, 0.5, 0.7). If 0.25 wins, 0 comes next. If both neighbours lose, 0.5 is near the best. P0 against P1 (tonight) also shows what blinding the poses line adds at 100k: P0 blinds only the ego state, P1 blinds both. P2 against B2 is the prompt at 176k, and P1 against B1 the prompt at 100k.
- **Noise:** N1 against P2 is the seed-to-seed difference. A factor's gain counts only above it.
- **Best capability:** S1 is the expected best. Its successor S2 combines the winners: vision trained if VX beats P2, otherwise the epochs E1 points to.
- **Per-epoch curves:** dev and train curves come free from the snapshots. An epoch-k snapshot of a longer run is not the same as a k-epoch run, because its learning rate has not decayed. So snapshots show trends, and only final adapters are scored on NAVSIM.
- **Distinct data vs repetition at matched samples seen:** S1's epoch-2 snapshot (980k samples seen) against E1 (1.06M). Score that snapshot on NAVSIM if the dev curves leave it open.
- **2B** stays at key points: M2 on the 176k set, not 100k.

## Can the data grow past 490k?
**Within nuPlan, no.** v1-full-thin (490k) is every usable front-camera frame of OpenScene's trainval logs at 0.5 s, with long standstills thinned. v1-full (676k) adds back 186k standstill frames, which repeat the same image and path. Beyond that, the options are (sizes and terms from `docs/training_data/DATASETS.md`; [proposed], each needs the researcher's sign-off before any download or build):

| Source | What it adds | Size | Cost and risk |
|---|---|---|---|
| Mirror augmentation | — | — | **Ruled out by the researcher (2026-11-11):** "Avoid mirroring I don't really want to deal with anything that adds in" |
| nuReasoning (Motional, Sep 2026) | NAVSIM-like inputs (8 cameras, ego, route commands, map), the nuPlan maker's data | 20k × 20 s = 105 h at 10 Hz (V); at 0.5 s, with 1.5 s of history and 4 s of future inside each clip, about 29 anchors per clip, so roughly 580k samples (D) | Overlap with nuPlan logs, and so with navtest, is unknown and must be checked before any use. Rig differs (D). Non-commercial terms (V) |
| nuPlan sensor blobs at 10 Hz | Frames between our 0.5 s anchors, same drives | 5× the frames; 19 TB (V) | Mostly near-duplicates of frames we have. Worth it only if S1 shows denser sampling helps |
| L2D (Yaak) | 6 cameras, GPS, IMU, CAN, route waypoints | 100k episodes, 500+ h (V) | Germany, a different rig, some learner drivers (V); licence Apache-2.0 (R) |
| CoVLA, ZOD, Waymo, WOD-E2E, Argoverse 2, nuScenes | Front camera + ego path | 30k to ~600k frames each (R/D) | New rigs and cities, left-hand traffic (CoVLA), a converter per dataset |

### What the 490k set is made of (2026-11-11)
One source: nuPlan through OpenScene's 2 Hz release, 1,270 training logs in 4 cities. Every one of the 489,771 rows carries four real past poses (−1.5 … 0 s; the build requires the whole −1.5 … +4.0 s window with no timing gap), and so does every dev-v2 row and every navtest scene. The rows by difficulty, where "hard" means a constant-speed, constant-heading path ends at least 3.9 m from the human's at 4 s (navtrain's 25th percentile; a proxy for NAVSIM's filter, which uses the score of that path; counts V, categories D):

| | Rows | Share | Las Vegas |
|---|---|---|---|
| navtrain (NAVSIM's non-trivial scenes; 96,353 of its 103,288) | 96,353 | 19.7% | 66% |
| Hard, not in navtrain (≥ 3.9 m) | 132,112 | 27.0% | 65% |
| Medium (1–3.9 m) | 168,240 | 34.4% | 72% |
| Easy, moving (< 1 m) | 60,125 | 12.3% | 74% |
| Standing still (path under 0.5 m; thinned to one per 5 s per log) | 32,941 | 6.7% | 79% |

v1-full (676k) holds the same moving rows plus 186k more standstill frames. Standstill frames stay in every set, thinned, never removed (researcher, 2026-11-11).

## Order and compute
- Each 4090 queue trains about 8.4 samples/s (0.8B, both queues sharing the GPU), so roughly 0.73M samples a day per queue.
- Anvil (and with it the only route to the 4090) is down from Nov 16 04:00 to Nov 18 06:00. The queues run on their own through it. Scoring and copies wait until it is back.
