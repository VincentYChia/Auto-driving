# Gaps in the published work that our results fill (running list)

Started 2026-11-19. Updated whenever a result lands. Researcher, 2026-11-19: "a running list of gaps we can definitively say we filled."

**What counts as filled.** An item is listed under "Filled" only when all three hold:
1. **We measured it ourselves.** NAVSIM v1.1 PDMS on the full official navtest (12,146 scenes) with NAVSIM's own scorer, with paired bootstrap 95% CIs over scenes.
2. **The comparison is controlled.** One thing changes; everything else is held fixed.
3. **A literature search found no published version of the same controlled comparison.** "Not found" means not found in the searches recorded in the notes below, not proof that no such work exists.

Items that are new only in part are listed with exactly what is new and what is already published. Items that contradict published work are listed as conflicts, not as gaps.

Literature checks:
- `research_notes/SFT findings literature check/sft_claims.md`, 2026-11-19, about 42 searches and page reads. This is in the outer project folder, not in this repository.
- The head research, `research_notes/Driving action head diffusion type/` and its report in `reports/`, 2026-11-19.

## Filled

| # | Gap | Our result | Search outcome |
|---|---|---|---|
| F1 | **At equal compute, is it better to repeat hard data or to add more data?** No driving paper compares more epochs on a hard subset against fewer epochs on a larger set at matched samples seen. | v1-hard (236,133 samples) × 8 epochs (H2, 1.89M samples seen) beats the 490k set × 4 epochs (S3, 1.96M seen): **+1.0 PDMS (+0.6 to +1.3)**. Doubling the hard set's epochs (H1 → H2) adds nothing: +0.0 (−0.4 to +0.4). | Not found. The nearest work is MOSAIC (CVPR 2026, Hydra-MDP, EPDMS), which shortens epochs as its budget grows but never compares at matched samples seen. The analogue for language models is Muennighoff et al. 2023. |

Caveats for F1: one seed per run, and the 490k set contains the hard set. The model is a 0.8B VLA writing waypoints as text, trained on one front camera image with supervised training only.

## New in part

| # | What is new | What is already published | Our result |
|---|---|---|---|
| P1 | A difficulty-filtered subset compared against its own 2× superset at matched epochs and at matched compute, for a VLA. | Curated subsets can match larger pools on NAVSIM: MOSAIC, AutoScale and TTCov, none of which uses a difficulty filter or a superset at matched epochs or compute. NAVSIM's navtrain is itself a difficulty filter, but it was never compared against training on all of trainval. | **+0.9 (+0.6 to +1.3)** at equal epochs (H1 − S3), **+1.0 (+0.6 to +1.3)** at equal compute (H2 − S3), **+3.6** at 40k samples (minis, two seeds). |
| P2 | A seed-to-seed PDMS spread for VLA supervised training on the full navtest. | Multi-seed navtest PDMS is published for non-VLA planners: NAVSIM's TransFuser 83.3 / 84.0 / 84.4 (±0.56), LTF ±0.6, Ego-MLP ±0.9, plus ManeuverFlow, CLOVER and MD-Drive. | 0.7–0.9 PDMS between two seeds, on both training sets. Only two seeds, and only at mini length (40,000 samples). |
| P3 | PDMS by supervised-training epoch for a driving VLA, and dev ADE ranking checkpoints within one run differently from PDMS. | Open-loop metrics are known to correlate poorly with driving quality: Codevilla et al. 2018, PDM 2023, the NAVSIM paper and the ego-status paper. Codevilla 2018 also pooled checkpoints from within its runs. | S3 by epoch: **77.3 / 81.9 / 84.5 / 84.7**. Dev ADE is best after epoch 2 (0.600 m), where PDMS is 2.8 below the final model. Across mixes, S3's dev ADE beats H1's by 0.135 m, yet H1 scores higher. Caveat: the cosine schedule spans all 4 epochs, so a snapshot is not a shorter run. |
| P4 | NAVSIM's main branch deleted the warm-up filter files (`warmup_test_e2e.yaml`, `private_test_e2e.yaml`) in commit 9ca2c88 (v2.2, 2025-04-27). A train/test guard built from main therefore misses the warm-up scenes. | navsim issue #40 (Dec 2024) reported that warm-up logs overlap the train logs, and the maintainers confirmed it. No issue or PR reports the deletion. | Found 2026-11-08. Our sets held warm-up tokens (v1-pilot 36, v1-large 57) before the ±7 s scene guard. It could be filed as a GitHub issue, which is outward-facing, so only with the researcher's go. |

## Conflicts with published work

| # | Our result | Published | Differences that may explain it |
|---|---|---|---|
| X1 | Training the vision encoder during VLA SFT: **+2.5 PDMS (+2.0 to +2.9)** on the full navtest at full length (176k set, 4 epochs). The minis gave +3.6. | CARE-Drive (Qwen3-VL-2B, NAVSIM v1): unfreezing the vision tower **−5.1** (−7.6 to −2.5), on an 885-scene subset after 700 updates. | Full navtest against a subset; a full-length run against 700 updates; 0.8B against 2B; our vision learning rate is 2e-5, a tenth of the LoRA rate. Report ours as a conflicting data point, not a first. |

## Open, to be filled by the head plan

These come from the head research. "No paper found" is as recorded in its notes. None is filled yet. Each moves to "Filled" only when its result is measured as described at the top.

| # | Gap ("no paper found") | Step that fills it | Status |
|---|---|---|---|
| H1 | Masked and continuous heads compared on one backbone, with the same data, on NAVSIM | Steps 3–5: A against D, B and C on the same stored features and body | code done; layer probe queued |
| H2 | A separate masked-diffusion head on a standard left-to-right VLA (every published one runs inside the language model) | Step 3: option A | queued (layer probe) |
| H3 | Soft labels, and a token plus continuous offset, isolated for trajectories | A's ladder: rung 2 (soft labels), rung 4 (offset) | planned |
| H4 | "Discrete coarse + continuous residual" against pure discrete and pure continuous heads on the same backbone | A rung 4 against A rung 1, D and B | planned |
| H5 | A quantitative diversity measure for masked plans, and masked against continuous diversity | Best-of-6 and a diversity measure for every option | planned |
| H6 | DDIM against rectified flow on one architecture on NAVSIM | B against C | planned |
| H7 | 1 / 2 / 4 / 8-step sweeps for these heads on a driving benchmark | Every option, from the same weights | planned |
| H8 | Generation time against K paths for a driving head (on our 4090) | `predict_head.py` `timing.json`, every option | planned |
| H9 | The same RL recipe on a masked and a continuous head, same backbone and reward | Step 6: matched short RL for A and D | planned (navtrain metric cache job queued) |
| H10 | RL GPU-hours and PDM-scorer throughput | Step 6, reported as measured | planned |
| H11 | Multi-seed PDMS spread for a masked-diffusion planner on the full navtest | A second seed of the chosen A rung | not in the plan yet |
| H12 | AdaLN ego conditioning against ego as tokens | One extra run of the chosen head | not in the plan yet |

## Log

- 2026-11-19: list started. F1 filled. P1–P4 and X1 classified after the SFT literature check. H1–H12 copied from the head research.
