# SFT findings: prior-art check (2026-11-19)

Evidence tags: **V** = read in the paper's text, tables or code (arXiv HTML, ar5iv, or GitHub API); **R** = abstract or search snippet only.
Scope: public papers and code only. About 42 searches and page reads in total.

## Summary

| Claim | Verdict | Closest prior work | Tag |
|---|---|---|---|
| C1. Hardness-filtered 236k subset beats 2x larger 490k superset (equal epochs and equal compute), VLA SFT, full navtest | **PARTLY PUBLISHED.** The exact comparison was not found. Published work shows that a curated subset can match a larger pool on NAVSIM, but none filters by difficulty and compares against a larger superset at matched epochs and compute. | MOSAIC, [arXiv 2604.08366](https://arxiv.org/html/2604.08366) (CVPR 2026); AutoScale, [arXiv 2605.21372](https://arxiv.org/html/2605.21372); TTCov, [arXiv 2607.22697](https://arxiv.org/html/2607.22697); NAVSIM navtrain filter, [arXiv 2406.15349](https://arxiv.org/html/2406.15349) | V |
| C2. Training vs freezing the vision encoder in driving-VLA SFT, NAVSIM (+2.5) | **PARTLY PUBLISHED.** CARE-Drive has already run this ablation for a VLA on NAVSIM v1 and found the opposite sign. Our full-navtest result goes the other way. | CARE-Drive, [arXiv 2611.15360](https://arxiv.org/html/2611.15360): frozen 74.2 vs unfrozen 69.1 PDMS, on an 885-scene subset; ReDrive (not a VLA), [arXiv 2609.33854](https://arxiv.org/html/2609.33854v1): fine-tuned 89.2 vs frozen 85.7 | V |
| C3. Seed-to-seed PDMS spread for VLA SFT (0.7 to 0.9) | **PARTLY PUBLISHED.** Multi-seed PDMS on navtest exists for non-VLA planners. None was found for a VLA. | NAVSIM paper: TransFuser seeds 83.3 / 84.0 / 84.4 (±0.56), LTF ±0.6, Ego-MLP ±0.9; also ManeuverFlow, CLOVER, MOSAIC, TTCov | V |
| C4. PDMS by SFT epoch for a VLA, and within-run ADE misranking checkpoints | **PARTLY PUBLISHED.** The general point that open-loop metrics correlate poorly with driving quality is well established, and CARLA work pooled within-run checkpoints into that analysis. PDMS by SFT epoch for a driving VLA was not found, and neither was within-run ADE-vs-PDMS misranking on NAVSIM. | Codevilla et al. ECCV 2018, [arXiv 1809.04843](https://ar5iv.labs.arxiv.org/html/1809.04843); PDM / "Parting with Misconceptions", [arXiv 2306.07962](https://arxiv.org/html/2306.07962); NAVSIM | V |
| C5. Equal-compute trade: repeat hard data (8 epochs x 236k) vs add data (4 epochs x 490k) | **NOT FOUND (gap holds)** | Nearest: MOSAIC shortens epochs as the budget grows, which is not a controlled comparison. The LLM analogue is Muennighoff et al. 2023, "Scaling Data-Constrained Language Models". | V (MOSAIC), R (LLM) |
| C6. `warmup_test_e2e` filter missing from NAVSIM main branch, so main-branch guards miss warm-up scenes | **PARTLY PUBLISHED.** Overlap between warm-up logs and train logs was reported in Issue #40. No issue or PR reports that the filter was removed from main. | [navsim#40](https://github.com/autonomousvision/navsim/issues/40) (Dec 2024); removal commit [9ca2c88](https://github.com/autonomousvision/navsim/commit/9ca2c88f38ac8a70148b883d16b1cafe657b7200) ("version2.2", 2025-04-27) | V |

One line per claim:
- **C1:** On NAVSIM, MOSAIC (Hydra-MDP, EPDMS) matches full-pool performance with 42% fewer clips. AutoScale's 50k selected synthetic samples roughly match a 237k set. TTCov's selected subsets beat navtrain. None of them uses a difficulty filter against a larger superset, none matches epochs and compute, and none uses a VLA.
- **C2:** CARE-Drive (Qwen3-VL-2B, NAVSIM v1, paired log-bootstrap CI) found that unfreezing the visual tower costs 5.1 PDMS (CI −7.6 to −2.5), measured on an 885-scene subset. So the ablation exists but with the opposite sign. Our +2.5 on full navtest is a conflicting data point, not a first.
- **C3:** Multi-seed navtest PDMS is reported for TransFuser, LTF, Ego-MLP, ManeuverFlow, CLOVER and MD-Drive, and as EPDMS for Hydra-MDP and LTF. The spread is 0.02 to 0.6 std, and 1.1 range for TransFuser. No VLA SFT seed study was found.
- **C4:** Open-loop vs closed-loop misalignment is textbook (Codevilla 2018, PDM 2023, NAVSIM 2024, Ego-status 2024, Bench2Drive 2024), and Codevilla 2018 already pooled checkpoints from 2K to 500K iterations of each run. No paper reports PDMS by SFT epoch for a driving VLA. NoRD even picks its SFT checkpoint by minimum validation loss.
- **C5:** No driving paper compares repeated epochs on a hard subset against fewer epochs on a larger set at matched samples seen.
- **C6:** Issue #40 showed that warm-up test logs also appear in the mini train-log split, and the maintainers confirmed the overlap. The specific failure, that `warmup_test_e2e.yaml` and `private_test_e2e.yaml` were deleted from main in v2.2 so main-branch guards miss them, is unreported. The OpenScene repo has no related issues.

---

## C1. Data selection at a fixed budget (hardness-filtered subset vs 2x superset)

**Queries run**
1. `data scaling laws end-to-end autonomous driving NAVSIM training data selection`
2. `hard example mining data curation end-to-end planner NAVSIM PDMS fewer training samples outperform full dataset`
3. `nuPlan planner training data filtering difficult scenarios subset outperforms full training set closed-loop ablation`
4. `driving VLA NAVSIM data quality versus quantity training data ablation fewer samples higher PDMS`
5. `Preliminary investigation into data scaling laws imitation learning end-to-end autonomous driving nuPlan ...`
6. Pages read: MOSAIC (2 passes), TTCov (2 passes), NAVSIM paper (2 passes), NoRD, AutoScale, CARE-Drive (difficulty weighting section)

**Findings, closest first**
- **MOSAIC: "Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems"** ([arXiv 2604.08366](https://arxiv.org/html/2604.08366), CVPR 2026). **V.**
  - Method: selects clips from a pool (Navtrain: 4,141-clip pool, budgets 100 to 2,400 clips; OpenScene: 31.5k pool, budgets 250 to 8,000) by fitting per-domain scaling laws. The planner is Hydra-MDP (VoVNet), not a VLA. The metric is EPDMS on navtest.
  - Baselines: Random, Uncertainty, Coreset, Chameleon. There is **no difficulty-based baseline.** Cheap proxies (imitation loss, gradient norm) are only compared as ranking signals, and "none of the inexpensive alternatives yield a ranking that correlates strongly with EPDMS."
  - Headline claim: "reaches the full performance of training with all data samples using only 2,400 clips, i.e., 42% fewer samples". That is a match, not a beat.
  - Training length is **not matched**: 60 epochs up to 800 clips, 50 at 1,600, 45 at 2,400 "to reduce compute cost".
  - How close it comes: same question family (budgeted selection vs full pool, on NAVSIM). It differs on selection signal, model class, the absence of a superset-beating result, and the absence of epoch or compute matching.
- **AutoScale: "Closed Loop Dynamic Driving Data Mixture for Real-Synthetic Co-Training"** ([arXiv 2605.21372](https://arxiv.org/html/2605.21372), May 2026). **V.**
  - Method: selects 50k or 100k synthetic (SimScale 3DGS) samples to add to navtrain, using closed-loop-feedback cluster reweighting. It compares against the full 237k SimScale planner set. Planners are LTF and DiffusionDrive, not VLAs. The metric is EPDMS.
  - Results: on navtest, 50k is slightly below 237k (LTF 84.1 vs 84.4; DiffusionDrive 85.6 vs 85.9). On navhard, LTF 50k beats 237k (31.5 vs 30.2) but DiffusionDrive does not (32.4 vs 32.6).
  - Caveats: not compute-matched, no seeds, and the selection is learned rather than difficulty-based.
  - How close it comes: "smaller selected set ≥ larger set" on NAVSIM, but for synthetic augmentation with no compute control.
- **TTCov: "Test-Time Coverage: Test-Conditioned Data Curation for Deployment-Aware Learning"** ([arXiv 2607.22697](https://arxiv.org/html/2607.22697)). **V.**
  - Method: selects from OpenScene trainval by matching LLM-extracted proposition statistics to the test distribution. The planner is LTF. The metric is EPDMS on navhard_two_stage, 100 epochs, 3 seeds.
  - Results: at 1x navtrain size, a random OpenScene subset scores 20.15 against navtrain's 24.70. This is indirect evidence that navtrain's difficulty filter beats a random equal-size subset. TTCov at 1.25x and 1.5x scores 25.49 and 26.40, beating navtrain.
  - Gaps: no difficulty baseline, and no comparison against the full OpenScene superset.
- **NAVSIM paper** ([arXiv 2406.15349](https://arxiv.org/html/2406.15349)). **V.**
  - The navtrain and navtest filter removes scenes where the constant-velocity agent scores PDMS > 0.8 (near-trivial), and scenes where the human trajectory scores < 0.8 (noisy labels). "Our filtering primarily removes scenes with static or fast longitudinal movement."
  - The paper trains 114 small planners on 25/50/100% of navtrain and on an equally sized uniform OpenScene subset. This was only used to study PDMS vs closed-loop correlation; per-subset scores are not reported.
  - It never trains on full trainval and compares against navtrain.
  - So the rationale for difficulty filtering is published, but the data-selection ablation is not.
- **CARE-Drive** ([arXiv 2611.15360](https://arxiv.org/html/2611.15360), Nov 2026). **V.**
  - "Difficulty-weighted imitation learning": per-epoch PDMS-gap reweighting on 92,772 navtrain scenes raises full-navtest PDMS from 85.863 to 86.462 (+0.6) for a Qwen3-VL-2B VLA backbone.
  - This reweights the loss and keeps the same data, with no superset. It is the closest *VLA* difficulty result.
- **FactorDrive** ([arXiv 2608.09591](https://arxiv.org/pdf/2608.09591)). **R.** Mines 4,910 hard samples (PDMS < 0.80, 4.8% of navtrain) for post-training. This is not a subset-vs-superset SFT comparison.
- **NoRD** ([arXiv 2602.21172](https://arxiv.org/html/2602.21172v2), CVPR 2026). **V.** Uses 80k SFT samples (an 80/20 split of navtrain, selection unspecified) and says only that "naively reducing the amount of SFT data significantly degrades performance", without numbers.
- **Data-scaling studies.**
  - [arXiv 2412.02689](https://arxiv.org/html/2412.02689v2) (ONE-Drive, about 4M demos): power law in open-loop but not in closed-loop; a small amount of added long-tail data helps the matching scenarios. **R.**
  - [arXiv 2504.04338](https://arxiv.org/html/2504.04338v1) (NVIDIA, CVPRW 2025, internal 16 to 8,192 h): quantity scaling, not selection. **R.**
  - Neither uses NAVSIM. Neither compares a selected subset with a larger set.
- **nuPlan.** Test14-hard (PDM-Closed worst-20-per-type) is a hard *evaluation* split, not training selection. PDM and PlanTF cap samples per scenario type for balance, not difficulty. **R.**

**Verdict: PARTLY PUBLISHED, and the exact comparison is not found.** What remains open:
- a difficulty filter (constant-velocity endpoint error, standstill thinning) compared against a 2x superset that contains it;
- at both equal epochs and equal compute;
- for a VLA, on full navtest PDMS, with paired CIs.

What is not new: "curated NAVSIM subsets can match a larger pool" (MOSAIC, AutoScale) and "navtrain beats a random equal-size subset" (TTCov). Word the claim accordingly.

## C2. Training vs freezing the vision encoder (NAVSIM)

**Queries run**
1. `freeze vision encoder ablation vision-language-action driving NAVSIM PDMS fine-tune ViT`
2. `"frozen" "vision encoder" VLM trajectory planning NAVSIM ablation unfreezing visual encoder improves PDMS` (extended)
3. `Qwen2.5-VL NAVSIM supervised fine-tuning "vision tower" trainable vs frozen ablation planning`
4. `"fine-tuning the visual encoder" OR "freezing the visual encoder" driving VLM NAVSIM PDMS table`
5. Pages read: CARE-Drive (2 passes), EvoDriveVLA, ReDrive, OpenVLA

**Findings**
- **CARE-Drive** ([arXiv 2611.15360](https://arxiv.org/html/2611.15360), v2 13 Nov 2026). **V. This is the closest match.**
  - Setup: Qwen3-VL-2B with 3 cameras x 4 frames, plus a future-feature auxiliary loss (frozen V-JEPA 2.1 teacher). The comparison uses a 700-update, ten-epoch budget, the same seed, and NAVSIM v1.
  - Section 4.3.2, Table 4: frozen tower 74.198 vs unfrozen tower (plus merger) 69.097 test PDMS. "Unfreezing reduces test PDMS by 5.101 points", with a paired log-bootstrap 95% interval of [−7.646, −2.548].
  - The ablation was run on an **885-scene sampled test set**, not the full navtest. The tower learning rate is not stated.
  - **Opposite sign to ours.**
- **ReDrive** ([arXiv 2609.33854](https://arxiv.org/html/2609.33854v1)). **V.**
  - Setup: V-JEPA2 ViT-L encoder with an action-DiT planner. It is **not a VLA** and has no language model.
  - Table 5: frozen encoder 85.7 PDMS, versus 89.2 when the encoder is fine-tuned with the planning objective (+3.5). Single run.
  - Same sign as ours, but a different model class.
- **EvoDriveVLA** ([arXiv 2603.09465](https://arxiv.org/html/2603.09465)). **V.**
  - Setup: Qwen2.5-VL-3B. It claims VLAs "suffer from degraded perception after unfreezing the visual encoder".
  - Its frozen vs unfrozen ablation (Table 3) is on **nuScenes open-loop** (L2 and collision), not NAVSIM. There, unfrozen without KD (0.53 / 0.14) beats frozen (0.56 / 0.18), reading the checkmark columns from the HTML.
- **FROST-Drive** ([arXiv 2601.03460](https://arxiv.org/html/2601.03460), WACVW 2026). **R.** On Waymo E2E (RFS metric), a fine-tuned 14B VLM encoder does worse than a frozen one. Not NAVSIM.
- **Other NAVSIM VLAs** keep the VLM frozen by design, with no ablation: DiffAdapterVLA 88.3 and VFlash 87.1. **R.** NoRD trains all components, including the vision encoder, with no ablation (**V**). WAM-Flow has no freeze ablation (**V**).
- **Robotics context.**
  - **OpenVLA** ([arXiv 2406.09246](https://arxiv.org/html/2406.09246)). **V.** "we found fine-tuning the vision encoder during VLA training to be crucial", even though "Prior work on VLMs found that freezing vision encoders ... typically leads to higher performance". In the Table 1 fine-tuning study: full fine-tuning 69.7% ± 7.2, frozen vision 47.0% ± 6.9, LoRA r32 68.2%.
  - **VLM4VLA** ([arXiv 2601.03309](https://arxiv.org/pdf/2601.03309)). **R.** Freezing the Qwen2.5-VL-3B vision encoder drops SimplerBridge from 48.0 to 23.95.

**Verdict: PARTLY PUBLISHED.** A frozen-vs-trained vision tower ablation for a driving VLA on NAVSIM v1 already exists (CARE-Drive), and it reports the **opposite** effect, on an 885-scene subset with a fixed 700-update budget. Our result is still not published:
- training the tower helps by +2.5 (CI +2.0 to +2.9);
- on full navtest with paired scene-bootstrap CIs;
- for a LoRA, text-waypoint, single-camera SFT VLA;
- replicated at small scale (+3.6).

Present it as "contrary to CARE-Drive's finding", not as the first such ablation.

## C3. Seed-to-seed PDMS spread

**Queries run**
1. `VLA NAVSIM PDMS "three seeds" OR "random seeds" mean standard deviation ...`
2. `NAVSIM navtest PDMS seed variance reproducibility end-to-end planner "± " multiple training seeds Hydra-MDP DiffusionDrive`
3. `VLM driving SFT multiple seeds NAVSIM standard deviation Qwen "seeds" PDMS variance reasoning-planning`
4. Pages read: NAVSIM paper, CLOVER, ManeuverFlow, MOSAIC, TTCov, CARE-Drive, NoRD, Codevilla 2019

**Findings, all for non-VLA planners unless stated**
- **NAVSIM paper** ([arXiv 2406.15349](https://arxiv.org/html/2406.15349)). **V.**
  - TransFuser default config, seeds A1 to A3: 83.3 / 84.0 / 84.4 PDMS on navtest. That is a 1.1-point range, and the paper notes "a standard deviation of ±0.56 in PDMS".
  - Table 3, 3 seeds: TransFuser 83.9 ± 0.4, LTF 83.5 ± 0.6, Ego-Status MLP 66.4 ± 0.9.
  - **This is the closest published analogue to our 0.7 to 0.9.**
- **ManeuverFlow** ([arXiv 2610.13409](https://arxiv.org/html/2610.13409)). **V.** Three independent training runs, each evaluated 5 times. NAVSIM v1: TF 90.58 ± 0.05, LTF 89.98 ± 0.06. These include a scorer and refiner stage.
- **CLOVER** ([arXiv 2605.15120](https://arxiv.org/html/2605.15120)). **V.** Generator plus scorer, DINOv2-S. "all runs differ by less than 0.02 PDMS" over 3 seeds.
- **MOSAIC** ([arXiv 2604.08366](https://arxiv.org/html/2604.08366)). **V.** Hydra-MDP, EPDMS on navtest, 3 seeds. Std is mostly 0.2 to 0.6 at Navtrain budgets of 100 to 1,600 clips, and up to 1.1 to 1.7 at OpenScene 250.
- **TTCov** ([arXiv 2607.22697](https://arxiv.org/html/2607.22697)). **V.** LTF, EPDMS on navhard, 3 seeds, for example 24.42 ± 0.66.
- **MD-Drive.** ±0.3 over 8 seeds (already known to us; not re-read).
- **VLAs.**
  - CARE-Drive uses fixed seeds only (**V**).
  - NoRD's "seeds" are best-of-N samples, not training seeds (**V**).
  - iDriveVLA ([arXiv 2609.30818](https://arxiv.org/html/2609.30818)) and WAM-Flow report single runs (**V**).
  - No VLA multi-seed navtest result was found.
- **CARLA context.** Codevilla et al. ICCV 2019, "Exploring the Limitations of Behavior Cloning" ([arXiv 1904.08980](https://ar5iv.labs.arxiv.org/html/1904.08980)). **V.** Over 12 seeds of CILRS on NoCrash, "the success rate can change by up to 42%".

**Verdict: PARTLY PUBLISHED.** Multi-seed PDMS on full navtest is established for classical end-to-end planners: TransFuser ±0.56 with a 1.1 range, which is similar in size to ours. For VLA supervised fine-tuning, no seed study was found. Our 0.7 to 0.9 is new as a VLA number but in line with TransFuser's.

## C4. PDMS by SFT epoch, and open-loop ADE misranking checkpoints

**Queries run**
1. `driving vision-language model supervised fine-tuning NAVSIM PDMS across epochs checkpoint "epoch" ablation ...`
2. `open-loop L2 error checkpoint selection misleading closed-loop driving score same training run epochs end-to-end driving`
3. Pages read: Codevilla 2018 (ar5iv), PDM / "Parting with Misconceptions", NAVSIM, WAM-Flow, CARE-Drive, NoRD, iDriveVLA

**Findings**
- **Open-loop vs closed-loop misalignment is well established.**
  - **Codevilla et al. ECCV 2018, "On Offline Evaluation of Vision-based Driving Models"** ([arXiv 1809.04843](https://ar5iv.labs.arxiv.org/html/1809.04843)). **V.**
    - 45 CARLA models (data amount, cameras, noise, depth, loss, regularization). "for each model we perform both online and offline evaluation after the following numbers of training mini-batches": 2K, 4K, ... 500K.
    - So within-run checkpoints were pooled into the analysis. MSE correlates only |r| = 0.39 with success rate in the basic setup.
    - **This is the nearest published version of the within-run checkpoint point**, but in CARLA, not NAVSIM, and not for a VLA. I did not verify a per-run misranking figure.
  - **PDM, "Parting with Misconceptions about Learning-based Vehicle Motion Planning"** (CoRL 2023, [arXiv 2306.07962](https://arxiv.org/html/2306.07962)). **V.** "uncovering a negative correlation between both evaluation schemes": PDM-Open has OLS 86 and CLS-R 54, while PDM-Closed has CLS-R 92 and OLS 42. It does not look at checkpoints.
  - **NAVSIM** ([arXiv 2406.15349](https://arxiv.org/html/2406.15349)). **V.** Across 114 planners, PDMS correlates better with closed-loop score than open-loop score does (numbers are in the figures). It does not look at checkpoints or epochs.
  - **"Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?"** (CVPR 2024, [arXiv 2312.03031](https://arxiv.org/abs/2312.03031)). **R.** Ego-status-only models reach SOTA nuScenes L2, so open-loop L2 is misleading.
  - **Bench2Drive** ([arXiv 2406.03877](https://arxiv.org/html/2406.03877v3)). **R.** "open-loop metrics like L2 error could not reflect the actual driving performance".
- **PDMS by epoch or checkpoint for driving VLAs: not found.**
  - **WAM-Flow** ([arXiv 2512.06112](https://arxiv.org/html/2512.06112), Janus-1.5B). **V.** PDMS vs **pre-training (VQA) epochs**: it peaks at 3 epochs, +3.3 over 0 epochs. SFT is fixed at 2 epochs, so this is not an SFT-epoch curve.
  - **CARE-Drive.** **V.** Validation PDMS per *training stage* (Val497 subset; Table 9), not per SFT epoch.
  - **NoRD.** **V.** Selects its SFT checkpoint by "minimum evaluation loss", which is the practice our ADE-misranking finding cautions against. It reports no PDMS by checkpoint.
  - **MOSAIC.** **V.** Loss and gradient proxies fail to rank *data domains* the way EPDMS does (Kendall-tau). This is the same flavour of finding, but for data rather than checkpoints.

**Verdict: PARTLY PUBLISHED.**
- Confirmed as established, so cite it rather than claim it: "open-loop metrics correlate poorly with closed-loop or PDMS" (Codevilla 2018; PDM 2023; NAVSIM 2024; Li et al. 2024; Bench2Drive 2024). Codevilla 2018 even pooled within-run checkpoints (CARLA).
- Not found:
  - a PDMS-by-SFT-epoch curve for a driving VLA (77.3 / 81.9 / 84.5 / 84.7, with 8 epochs equal to 4);
  - a demonstration on NAVSIM that held-out ADE misranks checkpoints and training mixes relative to PDMS within one VLA run.

## C5. Equal-compute trade: repeat hard data vs add more data

**Queries run**
1. `end-to-end driving imitation learning repeated epochs versus more unique data equal compute data-constrained scaling planner nuPlan NAVSIM`
2. Plus checks inside MOSAIC, AutoScale, TTCov and the scaling-law papers from C1

**Findings**
- **MOSAIC** (**V**) changes epochs with budget (60 / 50 / 45) purely to save compute. It reports no repeated-vs-fresh comparison and no "data repetition" discussion.
- **AutoScale** (**V**) uses "identical training protocols" across mixtures of different sizes, which is not compute-matched. It does not discuss repetition.
- **TTCov** (**V**) uses a fixed 100 epochs across budgets, which is not compute-matched.
- Neither data-scaling-law paper (2412.02689, 2504.04338; **R**) studies epochs vs unique data.
- The LLM analogue: Muennighoff et al., "Scaling Data-Constrained Language Models" (NeurIPS 2023), which finds that up to about 4 epochs of repeated data is nearly as good as unique data. **R**, from prior knowledge, not re-read here.

**Verdict: NOT FOUND (gap holds).** No driving paper compares (repeat a hard subset for more epochs) against (fewer epochs on a larger superset) at matched samples seen, on NAVSIM or nuPlan. The closest contrast is the LLM data-constrained-scaling literature, which is worth citing as context.

## C6. Tooling: `warmup_test_e2e` filter missing from NAVSIM main

**Searches.** GitHub REST search API, issues and PRs.
- `repo:autonomousvision/navsim` with: warmup, warm-up, warmup_test_e2e, private_test_e2e, leak, leakage, overlap, contamination, "navtrain navtest", split, scene_filter, duplicate, "missing yaml", "test logs train".
- `repo:OpenDriveLab/OpenScene` with: warmup, overlap, leak, split.
- Read in full: issues #40, #11, #107, #103, #27, #92, #177, #88, #161.
- Repo trees and commits checked through the contents and commits API.

**Findings (all V)**
- **Current state of main.** The scene_filter folder on `main` contains `all_scenes, competition_private_tokens, navhard_two_stage, navmini, navsafe_two_stage, navtest, navtest_two_stage, navtrain, private_test_hard_two_stage, warmup_navsafe_two_stage_extended, warmup_two_stage`. It has **no `warmup_test_e2e.yaml` and no `private_test_e2e.yaml`**. Branch `v1.1` has both.
- **When they were removed.** Commit [9ca2c88](https://github.com/autonomousvision/navsim/commit/9ca2c88f38ac8a70148b883d16b1cafe657b7200) ("version2.2", 2025-04-27) **removed** both `warmup_test_e2e.yaml` and `private_test_e2e.yaml`, and added `private_test_hard_two_stage.yaml`. The file existed at `7bf3a35a68` (version2.0.1).
- **[Issue #40](https://github.com/autonomousvision/navsim/issues/40), "Sensor logs exist in both warmup test split and training split"** (closed 2024-12-19). **This is the closest near-miss.**
  - The user found 43 logs listed in `warmup_test_e2e.yaml` that are also `train_logs` in `default_train_val_test_log_split.yaml` (mini).
  - The maintainer replied: "the warmup-test split is indeed based on the mini split - not only the test logs within mini. Thus, there can be overlaps between the mini train logs and the warmup test ones", and recommended navtest as the held-out split.
  - The issue also led to a correction of the earlier answer in #11, which had said warmup_test_e2e filters only mini's test logs.
- **[Issue #92](https://github.com/autonomousvision/navsim/issues/92).** The maintainer states: "using the navhard/navtest/test/warmup/private_test splits for training is not allowed".
- **[Issue #11](https://github.com/autonomousvision/navsim/issues/11).** warmup_test_e2e is built with the same challenging-scene filter as navtrain; the original answer was corrected after #40. Navtrain "includes overlapping scenes, while trainval consists only of non-overlapping scenes".
- **Other split issues** (#107 navtrain is a subset of trainval; #103 tokens do not align with nuPlan; #161 navtrain is the same across v1.1 and v2.2; #177 using all 103k navtrain for training) do not involve test-split leakage.
- **OpenScene repo.** Zero hits for warmup, overlap or leak. The single "split" hit (#6) is a tar EOF error.

**Verdict: PARTLY PUBLISHED.**
- Published: warm-up test logs overlap the mini train-log split (#40, confirmed by the maintainers).
- Not reported in any issue or PR: that the v1.1 challenge filters (`warmup_test_e2e`, `private_test_e2e`) were deleted from `main` in the v2.2 commit, so a train/test exclusion guard built from `main`'s scene filters silently fails to exclude the 2024 warm-up scenes.
- This would make a clean, new GitHub issue. Cite #40 as related, and give commit 9ca2c88 as the removal point.
