# Learned selectors over candidate paths on NAVSIM: how published planners build them (state as of 2026-11-20)

**Evidence tags.** **V** = read in the paper's full text, a table, or the code. arXiv HTML pages and ACL/arXiv PDFs were converted to text with a local script, and table cells were read one by one. The NAVSIM v1.1 code came from the `v1.1` tag on GitHub. **R** = self-reported in an abstract or search snippet, or a baseline number that a paper copies from another paper's table. **D** = derived here, with the arithmetic shown.

**Splits.** Every number names its split:
- **v1 navtest PDMS**: 12,146 scenes.
- **navval PDMS**: NAVSIM's validation split. DrivoR runs its ablations on it.
- **v2 navtest EPDMS**.
- **navhard two-stage EPDMS**.

These are different metrics or splits, and they are never mixed. EPDMS\* means NAVSIM v2's code before its human-filtering fix; EPDMS means the fixed code.

**Terms.** These come from `pilot_sft/EXPERIMENTS.md` "Definitions": path generator, stored features, selector, single-path PDMS, best-of-6 (oracle), diversity, configuration A–D. The terms below are new and are defined before use:
- **Own pick**: the path generator's deterministic best guess (temperature 0). It is one of the K candidates.
- **Best-of-K (oracle)**: best-of-6 generalised to K candidates.
- **In-fold candidates**: candidates generated for scenes whose drives the generator was trained on. **Out-of-fold (cross-fitted) candidates**: candidates from a generator that never saw that scene's drive.
- **Sub-score composition**: predict NC, DAC, TTC, EP and C separately, then combine them with the PDMS formula. The alternative is **direct PDMS**: predict the PDMS as one number.
- **Pointwise, pairwise and listwise losses**:
  - Pointwise: each candidate's own target, e.g. BCE or MSE.
  - Pairwise: a loss on pairs of candidates from the same scene, e.g. a margin-rank or Bradley–Terry loss.
  - Listwise: a loss on a scene's whole candidate set, e.g. a softmax over candidates fitted to a softmax of the true scores.
- **Gap closed**: (selected − own pick) / (best-of-K − own pick). It is the share of the oracle headroom that a selector recovers. Versatile reports this.
- **Own-pick margin δ**: the selector switches away from the own pick only when another candidate's predicted composed PDMS exceeds the own pick's by more than δ.

**The NAVSIM v1.1 formula the selector must reproduce** (V, code at tag v1.1):
- `PDMS = NC × DAC × (5·TTC + 5·EP + 2·C) / 12`.
- DDC is computed but has weight 0 (`driving_direction_weight: 0.0`).
- NC ∈ {0, ½, 1}; ½ is a collision with a static object.
- The official `pdm_score()` scores each agent path together with that scene's PDM-Closed path, as a stack of two proposals.
- EP is the path's raw progress divided by the larger of the two progress values, when that maximum is above 5 m. Otherwise EP = 1 for every path that passes the gates.
- EP is forced to 0 whenever NC × DAC = 0.
- Source: `navsim/evaluate/pdm_score.py` and `pdm_scorer.py::_aggregate_scores` in the [v1.1 devkit](https://github.com/autonomousvision/navsim/tree/v1.1).

---

## Summary

1. **Every published NAVSIM selector found trains a pointwise per-sub-score BCE on PDM-scorer labels.** Almost all of them combine the predicted sub-scores with the PDMS formula or a log-weighted version of it. Many add an imitation term: a soft cross-entropy toward the logged path. The exception is iPad, which regresses the PDMS directly with BCE (V, all sources below).
2. **Sub-scores beat direct PDMS in the two ablations found**:
   - Hydra-MDP: 83.0 vs 80.2 (−2.8) on an early NAVSIM v1 version.
   - DrivoR: 90.0 vs 88.2 (−1.8) navval PDMS.

   All V.
3. **No matched ablation of listwise against pointwise losses exists on NAVSIM.** The closest results:
   - A pairwise margin-rank loss added to BCE gives +0.2 PDMS (DiffusionDriveV2).
   - A listwise expected-reward objective beats a one-hot "best trajectory" target by +7.5 EPDMS (ZTRS).

   All V.
4. **Learned selector over the generator's own pick, v1 navtest**:
   - +1.0 (DiffusionDriveV2's selector on DiffusionDrive).
   - +0.7 to +1.4 (Versatile, frozen planners, 3 seeds).
   - +2.8 (FlowR2A, 60 proposals vs its fixed single proposal).
   - Only +0.2 when the candidates sit in a narrow mode (ResAD).

   All V or D.
5. **No NAVSIM paper cross-fits the selector's training candidates.** Every generated-candidate selector is trained on in-fold candidates. Several papers do report a train/test candidate shift:
   - WoTE, GTRS and FlowR2A: vocabulary at training against generated proposals at test.
   - TOAD: scorers re-used on other planners' candidates.
   - CLOVER: with the same generator's 64-candidate pools, a scorer's top-1 scores 95.0 on navtrain against 92.4 on navtest.

   Cross-fitting is standard in parsing rerankers. Collins & Koo 2005 parsed the training set in 18 held-out chunks, which "prevented the initial model from being unrealistically 'good' on the training sentences". Charniak & Johnson 2005 used 20 folds (V).
6. **Gains grow with K and saturate around 32–64 generated candidates**:
   - DrivoR: 8 → 64: 87.6 → 90.0 navval.
   - ManeuverFlow: 8 → 64: 90.06 → 90.58.
   - CLOVER: 32/64/96: 93.9/94.5/94.4.

   Typical selectors are 2–4 transformer layers of width 256, adding about 6–8M parameters (V).

---

## Q1. Which NAVSIM methods use a learned selector, and how each is built

### 1a. Design table

| Method (venue / arXiv) | Scene input to the selector | Candidate encoding | Targets | Loss | Candidates: source, K | Combination at inference | Tag |
|---|---|---|---|---|---|---|---|
| **Hydra-MDP** (NAVSIM 2024 winner, [2406.06978](https://arxiv.org/html/2406.06978)) | Transfuser camera + LiDAR environment tokens | Fixed vocabulary from k-means on 700K nuPlan paths, 40 × (x, y, heading). MLP to one query each, then a transformer encoder over the vocabulary queries, plus ego status, then a decoder cross-attending to the scene | Imitation: softmax of −L2 to the logged path over the vocabulary. Sub-scores NC, DAC, TTC, C, and EP in the "-EP" variant, from offline PDM simulation of the vocabulary | Soft cross-entropy (listwise over the vocabulary) for imitation, plus BCE per sub-score | Fixed, K = 4,096 or 8,192 | −(w1 log S_im + w2 log S_NC + w3 log S_DAC + w4 log(5 S_TTC + 2 S_C + 5 S_EP)). Weights by grid search: 0.01 ≤ w1 ≤ 0.1, 0.1 ≤ w2, w3 ≤ 1, 1 ≤ w4 ≤ 10. The grid-search split is not stated | V |
| **Hydra-MDP++** ([2503.12820](https://arxiv.org/html/2503.12820)) | Image only (ResNet-34 / V2-99), temporal SE fusion | Same as Hydra-MDP | Imitation, plus NAVSIM sub-scores, plus TL, DDC, LK (EC is not a distillation target) | L_im + Σ BCE | Fixed, 8,192 | Same log-weighted form, grid-searched | V |
| **GTRS-Dense / GTRS-Aug** ([2506.06664](https://arxiv.org/html/2506.06664), CVPR 2025 challenge report, v1 only) | Image backbone (V2-99, EVA-ViT-L, ViT-L) | Trajectory tokenizer, then a transformer decoder | "Builds upon Hydra-MDP"; loss not restated (R). GTRS-Aug adds EMA-teacher soft targets clipped to ±δ, and a training-only refinement decoder over the top-k | As Hydra-MDP (R) | Trained on a 16,384 vocabulary with 50% random vocabulary dropout per batch. Inference on 100 DDPM proposals (100 steps) ∪ 8,192 vocabulary | Hydra-style | V |
| **DriveSuprim** ([2506.06659](https://arxiv.org/html/2506.06659)) | Image (ResNet-34 / V2-99 / ViT-L), 3 cameras | 2-layer MLP per trajectory. Coarse decoder and refinement decoder, each 3 layers, 256-d | Normalised L2 to the human path, plus every NAVSIM sub-score except EC | BCE per metric (coarse + refine), plus imitation. EMA-teacher soft labels clipped at δ = 0.15; the human path shifted ≤ 1 m toward the teacher | Fixed 8,192, filtered to the top 256 for refinement | Σ λ log s (imitation, gates) + λ_avg log Σ λ s. v1 coefficients: imitation 0.05, NC 0.5, DAC 0.5, EP 5, TTC 5, C 2, λ_avg 8 | V |
| **iPad** ([2505.15111](https://arxiv.org/html/2505.15111)) | BEV proposal queries from its ProFormer (images only) | Max-pool over time of each proposal's final queries, then an MLP | **Direct PDMS**: NC·DAC·(5EP + 5TTC + 2C)/12 of its own final proposals, simulated online in training (Ray) | BCE on the scalar | Generated, iteratively refined, N = 64, 4 iterations | Argmax of the predicted score | V |
| **WoTE** (ICCV 2025, [2504.01941](https://arxiv.org/html/2504.01941)) | Transfuser BEV, plus a BEV world model (2 transformer decoder layers) predicting future BEV states per candidate (0 → 2 → 4 s) | Shared MLP trajectory encoder. The reward model applies 2D convolutions over current and future BEV, pools, and concatenates action embeddings | Imitation: softmax(−L2) over anchors, with cross-entropy. 5 PDM sub-scores with BCE | CE + BCE, plus focal loss on BEV semantics | **Trained on 256 k-means anchors with precomputed scores; tested on the refined anchors** | −(0.1 log r_im + 0.5 log NC + 0.5 log DAC + 1.0 log(5TTC + 2C + 5EP)) | V |
| **ZTRS** ([2510.24108](https://arxiv.org/html/2510.24108)) | Image tokens (ResNet-34 / V2-99 / ViT-L), backbone fully fine-tuned | 40 steps × 7 features (x, y, heading + motion differences) = 280-D. MLP 280→1024→256, plus 1 transformer encoder layer, plus 3 decoder layers (256-d, 8 heads) | EPDMS sub-scores (scoring heads). The policy head uses EPDMS − λ·[EC violation] as advantage. **No human demonstrations** | BCE per metric; policy by EPO, Σ_a Ψ(s,a) ∇π(a; s) (listwise expected reward over the whole vocabulary) | Fixed, 16,384 (or 8,192) | Weighted average of the policy likelihood and the predicted scores, Hydra-style | V |
| **GoalFlow** (CVPR 2025, [2503.05689](https://arxiv.org/html/2503.05689)) | BEV (camera + LiDAR) | **Learned part scores goal points only**: 4,096 or 8,192 clustered endpoints, queries + ego, decoder over BEV | Distance score (softmax of −L2 to the GT endpoint) and DAC score (shadow vehicle inside the drivable area) | Soft CE (distance) + BCE (DAC) | Goal vocabulary; then 128/256 rectified-flow trajectories | Goal: w1 log δ_dis + w2 log δ_dac. **Trajectory "scorer" is a rule**: −λ1·Φ(distance to goal) + λ2·Φ(progress), plus a shadow-trajectory check | V |
| **DiffusionDriveV2** ([2512.07745](https://arxiv.org/html/2512.07745)) | BEV via deformable cross-attention, plus agent and map queries | Trajectory coordinates as queries, then an MLP score head | NAVSIM sub-scores | BCE, plus a **margin-rank loss** on the continuous metric | Generated, K = 20 anchors. Trained 20 epochs after RL on its own outputs, with 2 multiplicative-noise copies (std 0.1–0.2) and 1% of the GTRS vocabulary | Coarse-to-fine (DriveSuprim-style) | V |
| **TrajHF** ([2503.10434v2](https://arxiv.org/html/2503.10434v2)) | — | — | **No learned NAVSIM selector.** "EM" is an unsupervised EM-style clustering over 15 samples (25 iterations). "PDMS selector" is the oracle. Its Bradley–Terry reward model is for driving style on internal data | — | 15 diffusion samples | EM centroid | V |
| **Versatile Samples** (Purdue/Bosch, [2609.01799v2](https://arxiv.org/html/2609.01799), corresponding author Jiaru Zhang) | **Frozen** planner's scene-context tokens (width 128 MeanFuser / 256 DiffusionDrive); the scorer is trained alone | Flattened path, then a 2-layer MLP to one query. 3-layer transformer decoder over the context. 2-layer MLP head | 9 EPDMS components, **9 independent networks**, no weight sharing | Focal BCE (γ = 2) on 8 binary components, MSE on EP. Masked where a label is undefined. λ = 1 | Frozen planner's proposals (DiffusionDrive N = 20, MeanFuser N = 8), plus designed lateral/longitudinal perturbations of the human path straddling the DAC/NC boundary (437k rows) | EPDMS formula, argmax. Checkpoint chosen on a held-out ~5% of navtrain logs | V |
| **DrivoR** (CVPR 2026, [2601.05083v2](https://arxiv.org/html/2601.05083v2)) | 64 register tokens (16 per camera × 4) from DINOv2 ViT-S with LoRA | Decoded trajectory, then an MLP to a **new** query. **Stop-gradient** into the trajectory decoder. 4-layer decoder, 256-d | 6 PDMS sub-scores (one MLP each) | BCE, λ_c = 1 | Generated jointly, 64 learned queries (WTA L1) | Weighted sub-scores. Weights are adjustable at inference ("behaviour control"). About 40M parameters in total | V |
| **SparseDriveV2** ([2603.29163](https://arxiv.org/html/2603.29163v1)) | ResNet-34 scene features | Factorised vocabulary: 1,024 paths × 256 velocity profiles. Coarse factorised scoring, then a progressive filter to 20 × 20 = 400 composed paths | Soft CE to the logged path (path, velocity, trajectory levels), plus BCE per sub-score (rule-based teacher) | Soft CE + α·BCE | Fixed but combinatorial | "Same weighting scheme as the benchmark" | V |
| **FlowR2A** ([2606.24231](https://arxiv.org/html/2606.24231)) | BEV via grid-sample cross-attention | Sinusoidal embedding of the path. 2-layer transformer; the within-path self-attention of its decoder block is removed, and each proposal is scored on its own | NC, DAC, **continuous TTC** (min(ttc, 2 s)/2 s), EP, **v2 HC instead of v1 comfort** (v1 comfort is "almost all 1"). Auxiliary per-timestep TTC-time (MSE) and ego-area (BCE) | BCE (+ MSE aux) | Stage 1: vocabulary samples. **Stage 2 (2 epochs, generator frozen)**: its own proposals with the inference sampling settings + 32 random vocabulary paths per scene, labelled online. K = 60 | (Π NC, DAC)·(Σ w·{EP, TTC, HC}), w = 1. Official 5/5/2 weights give the same 92.8 | V |
| **ResAD** ([2510.08562v2](https://arxiv.org/html/2510.08562v2)) | Perception features + ego | Positional embedding of candidates, then a transformer cross-attending to the scene | Imitation (softmax of −L2 over the candidates) + BCE per sub-score | Soft CE + BCE | Generated, K_train = 20, K_infer = 200 | "Highest weighted score". The ranker adds 6M parameters (62M → 68M) and 3.7 ms | V |
| **CLOVER** ([2605.15120v2](https://arxiv.org/html/2605.15120)) | DrivoR-style (DINOv2-S LoRA, 4 cameras) | Candidate waypoints, one token per trajectory | Sub-scores composed by the official rule (v1 weights NC/DAC/DDC/TTC/EP/C = 1/1/0/5/5/2) | Component-specific classification/regression | Generated, K = 64. The scorer is refit on current proposals each cycle (online labels) | Official PDMS composition. For v2, an optional soft re-rank toward a single-mode anchor | V |
| **ManeuverFlow** ([2610.13409](https://arxiv.org/html/2610.13409)) | Current BEV + predicted future BEV, sampled along each path | Score tokens with **set-level self-attention across candidates** | 5 PDM factors | Weighted BCE with factor weights, label smoothing and logit regularisation | Generated online each step, N = 64 | Fixed log-PDM composition | V |
| **World4Scorer** ([2609.36438](https://arxiv.org/html/2609.36438)) | DINOv2 LoRA scene tokens; a shared predictor (4 blocks, self-attention across candidates) | Action encoder on stop-gradient coordinates | NC, DAC, DDC, TTC, EP (soft), C | BCE (+ 0.1 L1 on EP for bank paths) | 64 generated + 16 per scene from a bank of perturbed paths with precomputed labels | Σ α log p (NC, DAC, DDC) + log Σ β p (TTC, EP, C) | V |
| **Adaptive-WAM** ([2608.06008v2](https://arxiv.org/html/2608.06008)) | Fine-tuned DINOv2-S on the image. **No ego state or command** | Flattened 8 poses, then an MLP, concatenated with image features. 6 independent 2-layer MLP heads | NC, DAC, DDC, TTC, EP, C as soft [0, 1] targets | Soft BCE-with-logits, stop-gradient | Up to 6 exits, or 64 proposals | PDMS composition | V |
| **DiffusionDrive** (own-pick baseline in other papers, [2411.15139](https://arxiv.org/html/2411.15139)) | — | Per-anchor confidence head | y = 1 for the anchor closest to the logged path, else 0 (**imitation only**) | BCE | 20 anchors | Highest confidence | V |

Other learned or test-time selectors found, not used as design templates:
- **Centaur** ([2503.11650](https://arxiv.org/html/2503.11650)) runs test-time training on Hydra-MDP. It takes one gradient step per frame on the "cluster entropy" of the top-100 candidates, grouped into 5 direction clusters (V).
- **World4Drive** ([2507.00603](https://arxiv.org/html/2507.00603)) has a ScoreNet over K intention modes. It is trained with focal loss toward the mode whose predicted future latent is closest to the real one, with no PDM labels (V).
- **TOAD** ([2606.07170](https://arxiv.org/html/2606.07170v1)) uses DrivoR's scorer as a reward for cross-entropy-method search, with 64 samples per iteration, warm-started at the planner's pick (V).
- **GuideFlow** ([2511.18729v3](https://arxiv.org/html/2511.18729v3)) applies a pre-trained GTRS-Dense scorer off the shelf to 256 generated paths (V).
- **TransDiffuser** ([2505.09315](https://arxiv.org/html/2505.09315)) uses "rejection sampling following TrajHF" and the "top-1 scoring" path, without describing a learned scorer. This resembles TrajHF's oracle PDMS selector, so it is **not counted** as a learned selector (V for the wording; the mechanism is unclear).

### 1b. Gain of the selector over the own pick or no-selector baseline: v1 navtest PDMS

| Method | Baseline → with selector | Δ | Tag | Note |
|---|---|---|---|---|
| DiffusionDriveV2's selector on DiffusionDrive | 88.1 (own confidence head) → 89.1 | +1.0 | V (Tab. 9), D | Same 20 candidates; only the picker changes |
| DiffusionDriveV2, selector design ablation | base 89.7 → coarse-to-fine 89.9 → + margin-rank loss 90.1 | +0.2, +0.2 | V | Candidates fixed |
| Versatile, plain scorer (own proposals only) | DiffusionDrive 88.1 → 89.3; MeanFuser 89.0 → 89.7 | +1.2; +0.7 | V for scorer rows (3-seed means); R for the baselines (quoted from MeanFuser's paper); D | Planner frozen |
| Versatile, + designed boundary samples | 88.1 → 89.5; 89.0 → 89.9 | +1.4; +0.9 | V, D | +0.2 over the plain scorer on v1 |
| FlowR2A | 90.0 (1 proposal, fixed r_high 0.95, t_init 0.5) → 92.8 (60 proposals + selector) | +2.8 | V, D | Proposals are reward-conditioned |
| FlowR2A, proposal-count table | 1 / 4 / 64 proposals: 89.4 / 91.8 / 92.9 | +3.5 (1 → 64) | V, D | Gain mostly in EP (84 → 90) |
| ResAD, at its base model | 84.9 → 85.1 (+ ranker) | +0.2 | V, D | "Choices concentrate in a narrow mode"; mean candidate PDMS 86.1 vs selected 88.3 for the full model |
| WoTE (20-epoch ablation) | 81.0 (Transfuser, one regressed path) → 83.2 (evaluator, no future) → 85.6 (with world model) | +2.2, +4.6 | V, D | The baseline is a different model; the text says 87.1 while Tab. 1 says 88.3 (internal conflict) |
| GoalFlow | 85.6 (average of samples) → 88.5 (+ learned goal distance score) → 89.4 (+ goal DAC score) → 90.3 (+ rule-based trajectory scorer) | +0.9 for the trajectory rule | V, D | The learned part scores goals |
| TrajHF | 86.4 (one sample) → 87.6 (EM aggregation, not learned); oracle 94.3 | +1.2 | V, D | No learned selector |
| Hydra-MDP (early NAVSIM version; Transfuser = 78.0 in this table) | 80.9 (vocabulary scored by imitation only) → 83.0 (+ sub-score BCE) → 85.7 (+ tuned weights) → 86.5 (+ EP target) | +2.1, +2.7, +0.8 | V, D | Absolute values are not comparable with v1.1 tables |
| Hydra-MDP++ | 85.0 (equal weights) → 86.5 (tuned weights) | +1.5 | V, D | Same model; only the combination changes |
| DrivoR (navval, **not navtest**) | 80.1 (1 query, regression) → 90.0 (64 queries + scorer) | +9.9 | V, D | Includes the generator change |
| Centaur (NAVSIM v1.0, local) | Hydra-MDP 90.3 → test-time training 92.6. Leaderboard v1.1: 92.10 ± 0.33 over 3 runs | +2.3 | V, D | Scorer adapted at test time |
| TOAD (search with DrivoR's scorer, public checkpoints) | ZTRS 86.9 → 89.0; GTRS 90.4 → 90.9; Hydra-MDP 90.9 → 91.4; iPad 91.7 → 93.4; RAP-DINO 93.8 → 93.9; DrivoR 94.6 → 94.7 | +0.1 to +2.1 | V, D | Search finds new paths, not only a re-rank |
| Adaptive-WAM | fixed exit B15 (one path) 90.62 → scorer-gated early exit 90.79 | +0.17 | V, D | Up to 6 candidates |
| iPad, GTRS, ZTRS, DriveSuprim, CLOVER, ManeuverFlow, World4Scorer | No own-pick or no-selector row on v1 navtest | — | not found | — |

**Oracle headroom reported alongside** (V):
- **DiffusionDrive / DiffusionDriveV2**: best-of-20 PDMS 93.5 / 94.9, against deployed 88.1 / 91.2.
- **DriveSuprim**: the best path among its own top-1 / 4 / 16 / 256 predicted candidates scores 91.9 / 94.5 / 96.1 / 98.7.
- **CLOVER's DrivoR-style baseline**: selected 93.69 against best-of-64 99.33 (gap 5.64); after its Stage 2, 94.48 against 99.39 (gap 4.95).

### 1c. NAVSIM v2 navtest EPDMS (kept separate)
- Versatile (V for the scorer rows; R for the baselines):
  - DiffusionDrive 88.3 → plain scorer 89.7 → designed samples 90.1.
  - MeanFuser 89.5 → 90.1 → 90.4.
  - The designed samples close 5.5% / 6.9% of the gap to the best-of-N ceiling (95.87 at N = 20; 94.33 at N = 8).
  - A random-sample control with the same number of extra rows (437k) gives 89.72 / 90.22, against 90.05 / 90.41 for the designed samples (3 seeds, std 0.01–0.10).
- DriveSuprim ablation (V): Hydra-MDP++ 81.4 → coarse-to-fine 82.4 → + rotation augmentation 82.7 → + self-distillation 83.1. Its soft labels (83.1) beat label smoothing (82.5) and temperature-scaled cross-entropy (82.7).
- ZTRS (V): ResNet-34 83.2, V2-99 85.3, ViT-L 86.2. ZTRS's table also lists GTRS-Dense at V2-99 84.0 and ViT-L 84.7 (R).
- CLOVER (V): 89.3 without, 90.4 with anchor-assisted re-ranking. EC rises from 35.0 to 75.5.
- World4Scorer (latest devkits, V): 91.4 framewise; 93.0 with inertial re-ranking toward the previous plan (EC 77.0 → 92.8).
- SparseDriveV2 (V): 86.7 EPDMS\*, 90.1 EPDMS.
- FlowR2A (V): 88.9.
- Hydra-MDP++ (V): ResNet-34 80.6, V2-99 84.1. It uses its own EPDMS definition (5/2/5/5/5 weights over 22).

### 1d. navhard two-stage EPDMS (kept separate)
- GTRS (V):
  - Random pick among diffusion proposals: 25.6.
  - GTRS-Dense on the diffusion proposals only, never trained on them: 36.7 (+11.1).
  - On the vocabulary: 39.7.
  - On the union of proposals and vocabulary: 40.8 (16k vocabulary) or 42.0 (8k).
  - With vocabulary dropout: 43.4.
  - Six-model ensemble: 49.4.
- GuideFlow (V): 27.1 → 43.0 with the off-the-shelf GTRS-Dense scorer; 46.7 with an adjusted scoring strategy.
- TOAD (V):
  - Search with DrivoR's scorer: iPad 34.7 → 49.8; Hydra-MDP 40.9 → 49.7; GTRS 45.4 → 51.7; ZTRS 48.1 → 49.2; DrivoR 54.6 → 56.3.
  - **Re-scoring other planners' candidates with another scorer**: iPad + DrivoR scorer 45.6, + GTRS scorer 34.5, + SparseDriveV2 scorer 30.3. Hydra-MDP + DrivoR scorer 37.3, + GTRS scorer 40.3, + SparseDriveV2 scorer **9.5**.
- ZTRS (V): V2-99 45.5 against GTRS-Dense 41.7 and DriveSuprim 42.1.

---

## Q2. In-fold against out-of-fold training candidates

### NAVSIM papers: none cross-fits
Every selector that scores generated candidates trains on candidates for its own training scenes, made by a generator trained on those scenes (V):
- Trained online or jointly: iPad, DrivoR, CLOVER, ManeuverFlow, World4Scorer, Adaptive-WAM, ResAD.
- Trained after the generator: DiffusionDriveV2, FlowR2A stage 2, Versatile.

No paper compares in-fold with out-of-fold candidates for the selector. No paper splits navtrain to make out-of-fold candidates (searched; not found).

What is reported about train/test candidate shift:
- **Vocabulary at training, generated paths at test.**
  - WoTE trains on 256 fixed anchors with precomputed scores and tests on refined anchors. It scores 84.0 (anchors), 85.3 (1,024 anchors it never saw) and 85.6 (refined anchors), all v1 navtest in the 20-epoch setting (V).
  - GTRS's scorer, trained only on a vocabulary, transfers zero-shot to diffusion proposals: +11.1 EPDMS on navhard over a random pick (V).
  - GTRS names the problem: methods with "a small number of dynamic proposals [DiffusionDrive, GoalFlow] often fail to generalize to unseen trajectories, since the scorer is only exposed to a narrow subset during training" (V, a claim with no measurement).
  - FlowR2A: "The stage-1 selector sees only vocabulary trajectories, which differ from decoder proposals at inference." It fixes this with a stage 2 on its own proposals plus 32 random vocabulary paths "to keep the selector calibrated across the full quality range". No number is given for stage 2 against stage 1 (V; gain not found).
- **Scorers moved to other planners' candidates** (TOAD, navhard, V): re-scoring is inconsistent, and a fixed-vocabulary scorer can collapse (Hydra-MDP's candidates + SparseDriveV2's scorer: 40.9 → 9.5). TOAD's explanation: vocabulary-trained scorers "receive no supervision away from that set", while DrivoR's scorer is "trained jointly with a trajectory generator … forcing it to score freely decoded trajectories". This is a hypothesis.
- **Train/test gap at a fixed candidate pool** (CLOVER App. F, V): the trained CLOVER generator makes 64 candidates for every navtrain and navtest scene, and scorers are trained on the navtrain pools.

  | Scorer backbone | Top-1 PDMS, navtrain | Top-1 PDMS, navtest |
  |---|---|---|
  | DINOv2-S | 95.0 | 92.4 |
  | Wan2.2-5B features | 97.7 | 92.8 |

  The navtrain pools are in-fold for both the generator and the scorer, so this gap mixes the two effects.
- **In-fold candidates are too safe** (Versatile, V):
  - "Many strong planners concentrate their proposals around safe mode, providing limited supervision near decision boundaries."
  - On navtrain, DiffusionDrive's proposals fail NC 16.6% and DAC 15.8% of the time; MeanFuser's fail 8.4% and 8.9%.
  - No matching navtest failure rate is given, so the train/test difference cannot be computed (not found).
- **The selector as the weak link** (DiffusionDriveV2, V): selectors "typically have fewer parameters, leading to weaker generalization … prone to failure in out-of-distribution scenarios". This is a claim with no measurement.

### Outside driving: the precedents for cross-fitting
- **Collins & Koo 2005** ([Computational Linguistics 31(1)](https://aclanthology.org/J05-1003.pdf), V): "In order to gain a representative set of training data, the 36,000 training sentences were parsed in 2,000 sentence chunks, each chunk being parsed with a model trained on the remaining 34,000 sentences (this prevented the initial model from being unrealistically 'good' on the training sentences)."
  - That is 18 folds, so each fold model sees 94% of the data (D: 34,000 / 36,000).
  - The dev set was parsed with the 36,000-sentence model, and the test set with the 40,000-sentence model.
  - No in-fold control is reported.
- **Charniak & Johnson 2005** ([ACL P05-1022](https://aclanthology.org/P05-1022.pdf), V): "We used the 20-fold cross-validation technique described in Collins (2000) to compute the n-best parses", "to avoid training the n-best parser on the sentence it was being used to parse."
- **Cobbe et al. 2021**, GSM8K verifiers ([arXiv 2110.14168](https://arxiv.org/abs/2110.14168), V):
  - The verifier trains on 100 samples per **training** problem (in-fold), from a generator fine-tuned for only 2 epochs, "since the diversity of generated solutions begins to collapse after this point".
  - "test@100 performance degrades much more sharply than test@1 as we increase the number of epochs … the model becomes increasingly uncalibrated and overconfident."
  - They limit the generator's training rather than cross-fit.
- **Lee, Auli & Ranzato 2021**, DrNMT ([ACL 2021](https://aclanthology.org/2021.acl-long.563.pdf), V):
  - The MT models that produce the 50-best lists are trained on bitext only.
  - A reranker trained only on bitext n-best lists, which are in-fold for those models, loses 2.0 BLEU on validation (27.6 → 25.6) and "starts overfitting after 15 passes".
  - Adding n-best lists on back-translated monolingual sentences, which the MT models never saw, fixes it.
  - The authors attribute the gain to more data, so the fold effect is confounded with data size (D: our reading).

### Not found
- An in-fold vs out-of-fold ablation for a driving selector.
- Any NAVSIM paper that splits training drives for the generator and the selector.
- Train vs test candidate statistics (failure rates, diversity, oracle) from the same generator on NAVSIM, apart from CLOVER's top-1 gap.

---

## Q3. Sub-scores against direct PDMS; listwise against pointwise

### Sub-score composition against direct PDMS (two ablations found)
- **Hydra-MDP, Tab. 1** (V; early NAVSIM version):
  - Hydra-MDP-𝒱8192, with sub-score targets and unweighted combination: 83.0.
  - Hydra-MDP-𝒱8192-PDM, where "the learning target is the overall PDM score": 80.2.
  - Δ −2.8 (D). The authors attribute it to "the irregular distribution of the PDM score".
- **DrivoR, Tab. 6** (V; navval PDMS): separate, disentangled scoring decoder predicting 6 sub-scores 90.0, against the same decoder predicting 1 score 88.2. Δ −1.8 (D). The authors' explanation: "learning the final output of the scoring formula … is more difficult than predicting its components."
- **Related, for a path generator rather than a selector** (FlowR2A, Tab. 4, V): reward conditioning on the score only gives 84.5, + sub-rewards 86.7, + per-timestep rewards 89.4 (single proposal, v1 navtest).
- **iPad uses direct PDMS** (BCE on the scalar) and reaches 91.7, but it runs no ablation (V).
- **Composition weights**:
  - Tuned log weights add +2.7 (Hydra-MDP) and +1.5 (Hydra-MDP++) (V, D). The tuning split is not stated, which risks test-split tuning (not found).
  - FlowR2A's selector gets the same 92.8 with unit weights or official 5/5/2. Other weight settings stay within 0.3 PDMS (V).

### Listwise or pairwise against pointwise: no matched NAVSIM ablation found
Closest evidence:
- **DiffusionDriveV2, Tab. 8** (V): pointwise BCE + a pairwise margin-rank loss on the continuous metric gives +0.2 (89.9 → 90.1, v1 navtest).
- **ZTRS, Tab. 4** (v2 navtest EPDMS, V). The IL/RL check marks did not render; the mapping below follows the paper's text.
  - Imitation of the human path: 86.2.
  - Imitation of the single best-EPDMS vocabulary path (one-hot): 76.7.
  - EPO, a listwise expected-reward objective over all 16,384 paths with the EPDMS reward: 84.2.
  - EPO with an EC penalty: 85.3.

  In the authors' words, a one-hot target "fails to capture the underlying pattern" because "many trajectories in 𝒜 can achieve high EPDMS".
- **Imitation and simulation targets are complementary** (WoTE, Tab. 4, V; v1 navtest, 20-epoch setting): imitation reward only 83.5, simulation sub-scores only 83.6, both 85.6.
- **Soft labels help pointwise BCE** (DriveSuprim, Tabs. 11 and 13, V; v2 EPDMS): EMA-teacher soft labels give 83.1, against 82.6 for hard labels, 82.5 for label smoothing and 82.7 for temperature-scaled cross-entropy.
- **Ties at the top are pervasive** (Adaptive-WAM, V): "more than 95% of scenes contain trajectory groups that are all perfect, all zero, or tied at the top". The authors therefore use soft per-component BCE "to avoid imposing an artificial ordering". Their scorer picks the exact top-score candidate in 91.2% of scenes, and one within 5% of the top in 94.4%. It picks a path at least 50% worse than an available near-perfect one in 0.42% of scenes.
- **World4Scorer** (V): its NAVSIM scorer is pointwise BCE. Its listwise cross-entropy term (weight 0.25, temperature 0.5) appears only in its manipulation transfer experiment.
- **Outside driving, DrNMT** (V):
  - The reranker is **listwise**: KL between a softmax over the n hypothesis scores and a softmax of min-max-normalised sentence BLEU / T.
  - Min-max normalisation is worth +0.4 BLEU.
  - A set-level variant with self-attention across hypotheses equals the per-hypothesis scorer (27.6 = 27.6).
  - No pointwise comparison is reported.

---

## Q4. Selector size, K, and diversity

**Size** (V unless marked):
- **DriveSuprim**: 3 + 3 decoder layers, 256-d. Total 61M against Hydra-MDP++'s 53M (ResNet-34), so the refinement adds about 8M (D: 61 − 53; the paper says "approximately 7M").
- **ResAD**: the ranker adds 6M parameters (62M → 68M) and 3.7 ms.
- **DrivoR**: 4-layer, 256-d decoders; about 40M in total with ViT-S.
- **ZTRS**: 3 decoder layers at 256-d, 83.4M with V2-99 (GTRS-Dense 83.0M).
- **FlowR2A**: 2 layers.
- **Versatile**: 9 small networks, each a 2-layer MLP + 3-layer decoder + 2-layer head. One scorer trains for 100 epochs in about 50 minutes on one RTX 5090.
- **WoTE**: 2-layer world model; 18.7 ms for 256 candidates on an L20.

**K: fixed vocabularies** (V):
- Hydra-MDP: 4,096 → 8,192 gives 82.6 → 83.0 (v1).
- Hydra-MDP scaling study in SparseDriveV2 (v2 EPDMS): 1,024 → 2,048 → 4,096 → 8,192 → 16,384 anchors give 85.02 → 85.80 → 86.33 → 86.78 → 87.35, with no saturation before out-of-memory.
- SparseDriveV2 factorised vocabulary (v2 EPDMS): 512×128 → 1,024×256 gives 88.7 → 90.1.
- DriveSuprim, number of filtered candidates (v2): 64 / 256 / 512 / 1,024 gives 81.8 / 83.1 / 82.9 / 83.0.
- ZTRS: training on 16,384 and inferring on 8,192 lowers navtest (85.3 → 82.0) but raises navhard (43.4 → 45.5).

**K: generated candidates.** Gains saturate around 32–64 (V):

| Method | Split | Selected PDMS by K |
|---|---|---|
| DrivoR | navval | 1: 80.1, 8: 87.6, 16: 88.1, 32: 89.5, 64: 90.0, 128: 90.0 |
| WoTE | v1 navtest | 64: 84.5, 128: 85.4, 256: 85.6 |
| ManeuverFlow (k = 3) | v1 navtest | 8: 90.06, 16: 90.28, 32: 90.41, 64: 90.58, 128: 90.56 |
| CLOVER | v1 navtest | 32: 93.9, 64: 94.5, 96: 94.4 |
| FlowR2A | v1 navtest | 1: 89.4, 4: 91.8, 64: 92.9 |
| ResAD | v1 navtest | K_infer 20: 88.3, 200: 88.8 |

- In TransDiffuser, 10 / 15 candidates give 89.6 / 91.7, but its selection may use the oracle (V numbers, mechanism unclear).
- Outside driving, DrNMT's reranker trained on 5 hypotheses still gains +3.2 BLEU when reranking 200. It "suffers little from the mismatch" between training and test K (V).

**Low diversity: gains shrink** (V, D):
- ResAD's ranker adds +0.2 on a base model whose choices "concentrate in a narrow mode". Adding diverse candidates (its IRP module) then gives +1.6.
- CLOVER, Tab. 5: the DrivoR-style baseline's 64 candidates are 1.80 m apart (pairwise ADE). Stage 1 spreads them to 5.97 m:
  - Best-of-64 rises from 99.33 to 99.76.
  - The selected path rises only from 93.69 to 94.13.
  - The mean candidate falls from 79.72 to 73.74.
  - The gap stays about 5.6 points. Diversity helps only when the scorer can still rank the extra candidates.
- FlowR2A: on a feasible, tight pool (mean proposal 89.4 ± 7.4), the selector's gain is "concentrated on progress (EP, 84 to 90), while safety metrics remain saturated".
- Versatile: MeanFuser (N = 8, fewer failures) gains less than DiffusionDrive (N = 20): +0.7 against +1.2 (D).
- DiffusionDriveV2: RL lowers diversity from 42.3 to 30.3, while the PDMS of its top-10 ranked raw paths (PDMS@10) rises from 75.3 to 84.4. No selector-gain-vs-diversity test is run.

---

## Recommendation for our selector

### R1. Inputs
- **Scene: the stored features of the same layers the chosen path generator reads.**
  - Project each layer from 1,024 to 256 and add a learned layer embedding.
  - Read them by cross-attention, optionally after compressing to 32–64 learned latent tokens.
  - Evidence: Versatile's selector reads only a frozen planner's scene tokens (width 128/256) and still gains +0.7 to +1.4 PDMS. DrivoR scores well from 64 compressed register tokens (V). The layer test showed D benefits from layers 12/18/24 (project result).
- **Add ego state** (speed, acceleration, yaw rate, route command, 4 past poses), as the path generator does with AdaLN.
  - Evidence: Hydra-MDP, ResAD and World4Scorer add ego status to the candidate or state tokens (V). EP and comfort depend on the current speed.
  - Adaptive-WAM omits ego state and still works (V), so this is a design choice; test removing it once.
- **Do not feed the path generator's hidden states, only its decoded paths.**
  - Evidence: DrivoR's shared branch gives 84.7; separate decoders 86.8; separate + new trajectory embedding + stop-gradient 90.0 navval (V). Versatile: "the scorer receives the scene context directly rather than through the planner's decision, so it is free to disagree with the planner's own ranking" (V).
- **Keep the selector small**: 2–4 transformer layers, width 256, about 5M parameters (D: about 1M per layer of self-attention + cross-attention + feed-forward, plus input projections).
  - Evidence: published selectors add 6–8M parameters (V).
  - CLOVER's 5B-feature scorer overfits (top-1 97.7 on navtrain against 92.8 on navtest, V). Our frozen 0.8B features with 80,626 scenes × K paths make overfitting the main risk.
  - Use dropout, and choose the epoch on held-out navtrain drives (Versatile uses about 5% of logs).

### R2. Targets
- **Predict the sub-scores and compose them with the v1.1 formula** NC × DAC × (5·TTC + 5·EP + 2·C)/12. Use this as the default, with the official weights.
  - Evidence: the only two ablations favour composition, by 2.8 (Hydra-MDP) and 1.8 navval (DrivoR) (V).
  - Composition is what Versatile, CLOVER, SparseDriveV2, FlowR2A, ManeuverFlow and Adaptive-WAM deploy (V).
  - FlowR2A finds the weights barely matter (≤ 0.3) (V).
  - Run tuned log-domain weights (Hydra-style) only as an ablation, tuned on held-out navtrain drives, never on navdev or navtest. Hydra's +1.5 to +2.7 came from an unstated split.
- **Labels must reproduce the official per-path scoring**: each candidate scored together with its scene's PDM-Closed path only, never jointly with the other candidates.
  - The reason: EP is normalised by the larger progress of the two (V, code). Scoring all K candidates in one call would make every EP label relative to the other candidates.
  - **Mask the EP loss where NC × DAC = 0**, because the scorer forces EP to 0 there (V, code). Otherwise the EP head learns the gates twice. This follows Versatile's masking of undefined labels (V).
- **NC has three values {0, ½, 1}**: use a soft BCE target of 0.5 for the static-object case.
- **C is nearly always 1 in v1** (FlowR2A, V): keep its head but expect little signal.
- **DDC has weight 0 in v1.1** (V): an auxiliary head at most.
- Two optional extras, each as one ablation:
  - a **continuous TTC label**, min(ttc, 2 s)/2 s (FlowR2A, V);
  - an **imitation target**, softmax(−L2 to the logged path) over the K candidates. It enters inference with a small weight, as Hydra (0.01–0.1), DriveSuprim (0.05) and WoTE (0.1) do. WoTE's imitation + simulation 85.6 beats either alone, 83.5 / 83.6 (V).

### R3. Loss
- **Pointwise BCE per sub-score as the primary loss**, with soft targets. Counter the rare failures with focal BCE (γ = 2), as in Versatile, or with per-component weights, as in ManeuverFlow. Use BCE or MSE for EP.
  - Evidence: every NAVSIM selector found uses per-sub-score BCE (V).
  - Gate failures are a minority of candidates: Versatile counts 8–17% NC failures on navtrain for DiffusionDrive/MeanFuser (V). We expect in-fold pools to fail even less often, but that is our inference, not a measurement.
- **No listwise or pairwise term by default.** Run one pre-registered ablation:
  - BCE + listwise KL, softmax over the scene's K composed predictions against softmax(min-max-normalised true PDMS / T), with weight 0.25.
  - BCE + DiffusionDriveV2's margin-rank loss on EP.
  - Evidence: no matched NAVSIM ablation exists. The nearest gains are small (+0.2, DiffusionDriveV2). Ties dominate the top of the candidate list (Adaptive-WAM >95% of scenes), so ranking losses get little signal. DrNMT shows the listwise form works and that min-max normalisation helps (+0.4 BLEU) (V).
- **Optional soft-label self-distillation**: an EMA teacher with targets clipped to ±0.15 of the label. DriveSuprim gains +0.5 EPDMS over hard labels and beats label smoothing (V). It adds complexity, so only after the base selector is adopted.

### R4. Candidate encoding
- **One token per candidate from the decoded waypoints.**
  - Inputs: the 8 (x, y) points plus derived heading, speed and acceleration per step, normalised as in the path generator, with Fourier/sinusoidal features. Then a 2-layer MLP to 256, then decoder layers cross-attending to the scene.
  - Candidate coordinates are inputs only (stop-gradient), which our separate network gives for free.
  - Evidence: DrivoR, Versatile, ZTRS (x, y, heading + motion differences, 7 per step), CLOVER ("one token per trajectory") and FlowR2A (sinusoidal) (V).
  - No paper ablates the encoding itself (not found), so the derived kinematics are our choice, not evidence.
- **Start without self-attention across candidates.** Each candidate's score then does not depend on which other candidates are in the set. This also keeps the in-fold vs out-of-fold comparison clean. Test cross-candidate attention once.
  - Evidence: DrNMT's set-level reranker equals the per-hypothesis one (V). FlowR2A scores each proposal on its own (V). ManeuverFlow and World4Scorer use cross-candidate attention without an isolated ablation (V).
- Path-aligned feature sampling (ManeuverFlow, FlowR2A, DiffusionDriveV2) needs BEV features, which our stored features are not. One untested option: project each waypoint into the front image and bias attention toward nearby image tokens. This is our idea, not evidence.

### R5. The own pick: include it as a candidate; decide in a separate rule
- **Yes, always include the own pick among the K candidates.** It makes "selected ≥ own pick" reachable in every scene, and it is the baseline the adoption test compares against.
  - Evidence: TOAD always returns whichever of {searched path, base pick} scores higher, and "comparing against τ_base guarantees the returned trajectory never has a lower cost than τ_base" (V).
  - CLOVER's soft re-rank toward a single-mode planner's path adds +1.1 v2 EPDMS (89.3 → 90.4) (V).
  - World4Scorer's re-ranking toward the previous plan adds +1.6 (V).
- **Do not give the network a flag marking the own pick.** Keep it a pure sub-score estimator, as Versatile and DrivoR do (V). Put the preference in an **own-pick margin δ** outside the network:
  - Pick argmax over candidates of the composed score, but stay on the own pick unless the best candidate is predicted to be more than δ better.
  - Choose δ (δ = 0 allowed) on held-out navtrain drives.
  - Reason: selectors occasionally promote a failing proposal. FlowR2A lists this as its "more actionable bottleneck"; Adaptive-WAM's large-gap errors hit 0.42% of scenes (V).
  - Reason against the flag: in-fold, the own pick is "unrealistically good" (Collins & Koo's phrase, V). A flag would teach a selector trained in-fold to always keep it.

### R6. Training data and cross-fitting
- **Keep the planned cross-fitting.** It is the parsing-reranker standard (Collins & Koo, Charniak & Johnson, V), and no NAVSIM paper tests it.
  - Generate candidates with **exactly the deployment sampler**: same K, steps and temperature, own pick included. FlowR2A trains its stage-2 selector "using the same sampling configuration as inference" (V).
  - Split by **drive across the path generator's whole training set** (all 236,133 v1-hard scenes, not only the 80,626 navtrain scenes that have a metric cache), so no adjacent frame of a held-out drive leaks into a half generator.
- **Measure the cost of using half-data generators.**
  - Collins & Koo's fold models saw 94% of the data. Ours see 50%, so out-of-fold candidates come from a weaker generator than the deployed one.
  - Compare the half generators with the full one on navdev: single-path PDMS, best-of-K, gate-failure rates and diversity.
  - If the gap is large and compute allows, 4 folds bring each generator to 75%, at about 3 full training runs per configuration instead of about 1 (D: folds × (folds − 1)/folds).
  - **The same applies to RL**: if the final generator is RL-trained, the half generators need the same RL. RL changes diversity a lot (DiffusionDriveV2 42.3 → 30.3, V; in our runs A's 6 candidates went from 6.4 m to 0.8 m apart under RL4).
- **Add generator-independent boundary paths per scene**: 2–4 lateral and longitudinal perturbations of the logged path that straddle the DAC and NC boundaries, built as in Versatile.
  - Versatile's designed samples add +0.3–0.4 EPDMS and +0.2 PDMS over the plain scorer, while a matched random set adds 0.0–0.1 (V).
  - FlowR2A (32 random vocabulary paths per scene), DiffusionDriveV2 (noisy copies + 1% of the GTRS vocabulary) and World4Scorer (a 16-path bank) do the same for calibration (V).
  - These paths do not depend on the generator, so they have no fold problem.
- **Label budget** (D):
  - 80,626 scenes × 6 candidates = 483,756 paths, about 1.9 h at our scoring service's 72 paths/s (4 workers).
  - The in-fold control adds the same again.
  - 4 boundary paths per scene add about 1.2 h.
- **One residual in-fold source that cross-fitting cannot remove**: H1 itself was LoRA-trained on these 80,626 navtrain scenes. Its stored features, and the answer-start tokens in particular, may encode the logged path more cleanly on training scenes than on navdev/navtest. No paper addresses this (not found). navdev stays the clean test. A cheap check: the selector's navtrain-held-out vs navdev gap, and one run on image tokens only.

### R7. What to measure
1. **Adoption (pre-registered)**: the paired difference in single-path PDMS between the selected path and the own pick on navdev, with the drive-level bootstrap interval already in our pipeline. All selector settings (δ, epoch, weights) are fixed on held-out navtrain drives first. After adoption, report the full official navtest (12,146 scenes, NAVSIM's own scorer).
2. **Headroom and capture**: best-of-K, a random pick, the mean candidate, and **gap closed** (Versatile's metric, V). This shows how much of the best-of-6 headroom is captured.
   - Current headroom (D, from our navtest phase-1 runs): D's best-of-6 93.7 − own pick 82.9 = 10.8; A1's 91.8 − 84.7 = 7.1.
3. **Where the gain comes from**:
   - Sub-score deltas (NC, DAC, TTC, EP, C). Versatile finds the gain in the multiplicative gates; FlowR2A finds it in EP (V).
   - The rate of picks that fail a gate. Versatile: 3.70% → 3.35% (V).
   - The switch rate away from the own pick, with wins, losses and ties among the switches.
4. **Ranking quality**:
   - Top-1 exact and tie-aware accuracy (Adaptive-WAM: 91.2% / 94.4%, V).
   - Pairwise ordering accuracy on non-tied pairs (World4Scorer 64.2%; CLOVER 94.72% for ≥ 0.95 vs ≤ 0.50 groups, V).
   - The large-gap error rate (≥ 20% and ≥ 50% worse than an available path; Adaptive-WAM 0.57% / 0.42%, V).
   - Calibration per sub-score: Brier score and reliability for NC, DAC and TTC; mean absolute error for EP.
5. **The in-fold bias** (the planned in-fold control):
   - Navdev PDMS of the in-fold-trained selector against the cross-fitted one.
   - The candidate-distribution shift itself, compared across three sets: in-fold candidates from the full generator on navtrain; out-of-fold candidates from the half generators; full-generator candidates on navdev.
   - For each set: own-pick PDMS, mean candidate PDMS, best-of-K, NC/DAC/TTC failure rates and diversity (m).
   - Also each selector's navtrain-held-out vs navdev top-1 gap (CLOVER App. F style).
   - Optional third arm: train on in-fold + out-of-fold candidates together (DrNMT's mixed data, V).
6. **K and diversity**:
   - Selector gain at K = 6 against K = 12 or 24 if sampling is cheap. The literature saturates at 32–64 (V). DrNMT shows train K ≠ test K is tolerable (V).
   - Selector gain by scene bins of candidate spread (our diversity measure). ResAD's narrow-mode result predicts little gain on collapsed pools such as A after RL4.
7. **Pre-registered ablations**, each with ≥ 2 seeds (Versatile reports 3 seeds with std 0.01–0.10 EPDMS, V):
   - composed vs direct PDMS;
   - BCE vs BCE + listwise;
   - δ = 0 vs tuned δ;
   - with vs without boundary paths;
   - with vs without the imitation term;
   - cross-fitted vs in-fold training candidates.

---

## Not found (searched, absent)
- Any NAVSIM selector trained on out-of-fold (cross-fitted) candidates, or an in-fold vs out-of-fold ablation for a driving selector.
- A matched listwise vs pointwise ablation for a NAVSIM selector (same model, same candidates).
- A NAVSIM selector built on stored features of a frozen vision-language model. The nearest are Versatile (frozen planner features) and Adaptive-WAM (Wan features tested as a scorer backbone, no gain over DINOv2-S).
- FlowR2A's stage-2 vs stage-1 selector numbers; the values in iPad's and ResAD's PDMS-vs-K figures (figures only).
- GTRS's v1 navtest PDMS in its own paper (only TOAD's re-evaluation, 90.4 for V2-99).

## Sources
- Hydra-MDP: https://arxiv.org/html/2406.06978 · Hydra-MDP++: https://arxiv.org/html/2503.12820 · GTRS: https://arxiv.org/html/2506.06664 · DriveSuprim: https://arxiv.org/html/2506.06659
- iPad: https://arxiv.org/html/2505.15111 · WoTE: https://arxiv.org/html/2504.01941 · ZTRS: https://arxiv.org/html/2510.24108 · GoalFlow: https://arxiv.org/html/2503.05689
- DiffusionDriveV2: https://arxiv.org/html/2512.07745 · DiffusionDrive: https://arxiv.org/html/2411.15139 · TrajHF: https://arxiv.org/html/2503.10434v2 · TransDiffuser: https://arxiv.org/html/2505.09315
- Versatile Samples for Learned Trajectory Scoring: https://arxiv.org/html/2609.01799 (code https://github.com/PurdueDigitalTwin/Versatile-Scoring)
- TOAD: https://arxiv.org/html/2606.07170v1 · DrivoR: https://arxiv.org/html/2601.05083v2 · SparseDriveV2: https://arxiv.org/html/2603.29163v1
- FlowR2A: https://arxiv.org/html/2606.24231 · ResAD: https://arxiv.org/html/2510.08562v2 · CLOVER: https://arxiv.org/html/2605.15120 · ManeuverFlow: https://arxiv.org/html/2610.13409
- World4Scorer (Outcome-Grounded World Modeling): https://arxiv.org/html/2609.36438 · Adaptive-WAM: https://arxiv.org/html/2608.06008 · Centaur: https://arxiv.org/html/2503.11650 · World4Drive: https://arxiv.org/html/2507.00603 · GuideFlow: https://arxiv.org/html/2511.18729v3
- NAVSIM v1.1 devkit: https://github.com/autonomousvision/navsim/tree/v1.1 (docs/metrics.md, navsim/evaluate/pdm_score.py, pdm_scorer.py, default_scoring_parameters.yaml)
- Collins & Koo 2005: https://aclanthology.org/J05-1003.pdf · Charniak & Johnson 2005: https://aclanthology.org/P05-1022.pdf · Cobbe et al. 2021: https://arxiv.org/abs/2110.14168 · Lee, Auli & Ranzato 2021: https://aclanthology.org/2021.acl-long.563.pdf
