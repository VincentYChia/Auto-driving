# RL for diffusion-type action heads, focused on RL against the NAVSIM PDM score (state as of Nov 2026)

Evidence tags: **V** = read in the paper's own full text, tables or official repo docs during this session (arXiv HTML full text, fetched and table-extracted; spot-check the table before quoting it in a decision document). **R** = self-reported claim seen only in an abstract-level summary, search snippet or third-party overview, not checked against a table. **D** = derived by me, with the arithmetic shown. Unless a line says otherwise, every PDMS is NAVSIM v1 navtest. v2 EPDMS and navhard numbers are labelled separately. Already-known numbers (ReCogDrive 84.1 → 86.5 → 90.8; WAM-Diff2 88.3 → 91.1 with selection) are not re-derived here.

## Q1. Continuous diffusion and flow RL (DiffGRPO, DPPO, DDPO, ReinFlow, Flow-GRPO, DiffusionDriveV2, DriveDPO, TrajHF, others): how they get likelihoods, NAVSIM gains, stability

### Takeaway
Every continuous-head RL method on NAVSIM uses the same trick. It makes the denoising chain stochastic (DDPM with η=1, or an ODE turned into an SDE). It then treats each denoising step as a Gaussian policy with a known mean and variance, so the log-likelihood is exact: a sum of per-step Gaussian log-probabilities. In the published methods this is plain REINFORCE or GRPO with group-standardized PDMS advantages, a behaviour-cloning (BC) loss in place of KL, std floors and a per-step discount. Reported single-sample gains are +1.0 to +4.3 PDMS, and the gain gets smaller as the supervised starting point gets stronger. Most of the stability work is variance and entropy control: std floors, noise shape, and how advantages are grouped.

### Cited Findings
**Generic (non-driving) continuous methods**
- **DPPO** (robotics): policy-gradient fine-tuning of diffusion policies. The authors credit its results to "structured and on-manifold exploration, stable training, and strong policy robustness", and claim the best overall performance and efficiency against other RL methods for diffusion policies (R, abstract only; the two-layer-MDP and per-step Gaussian details were not read in full text) — [DPPO abs](https://arxiv.org/abs/2409.00588)
- **ReinFlow** (flow matching): it "injects learnable noise into a flow policy's deterministic path", which turns the flow into a discrete-time Markov process with exact likelihood. It fine-tunes Rectified Flow and Shortcut policies at 4 steps and at a single step. Reported results: +135.36% average episode reward on locomotion and 82.63% less wall time than DPPO; +40.34% average success on manipulation at 4 or 1 steps, with 23.20% less compute time than fine-tuned DDIM policies (R, abstract) — [ReinFlow abs](https://arxiv.org/abs/2505.22094)
- **Flow-GRPO** (text-to-image): ODE-to-SDE conversion that keeps the same marginal at every timestep, plus "denoising reduction" (fewer steps during training, the full count at inference). Reported results: GenEval 63% → 95% and visual text rendering 59% → 92% on SD3.5-M, with "very little reward hacking" (R, abstract) — [Flow-GRPO abs](https://arxiv.org/abs/2505.05470)

**ReCogDrive DiffGRPO (VLM + DDPM diffusion planner)**
- Policy form: each denoising step is a Gaussian policy N(μ_θ(x_t,t), σ_t² I). Chain log-prob = log p(x_T) + Σ_t log π_θ(x_{t-1}|x_t). The whole trajectory is one composite action, and the reward is PDMS from the NAVSIM simulator (V) — [ReCogDrive](https://arxiv.org/html/2506.08052)
- Loss: L = L_RL − λ·L_BC. L_RL weights each step's log-prob by the group-standardized advantage Â_i = (r_i − mean)/std and a discount γ^(t−1). L_BC is a log-likelihood on samples from a frozen reference policy and is used as "a stable surrogate for KL". There is no PPO clip, and the update count is 1, so π = π_old (V) — [ReCogDrive](https://arxiv.org/html/2506.08052)
- Hyperparameters: RL for 10 epochs, batch 128, lr 1e-4 cosine to 0, γ = 0.6. Minimum denoising std 0.02, minimum log-variance std 0.10, denoised clip ±1.0, noise clip ±3.0 in Table 12 but ±5.0 in the text (conflict). The planner trains with 100 steps and runs inference with 5. The paper does not state the group size G (V) — [ReCogDrive](https://arxiv.org/html/2506.08052)
- Sub-metric movement from RL, Table 3 (planner without RL → with DiffGRPO): NC 98.1→97.9, DAC 94.7→97.3, TTC 94.2→94.9, EP 80.9→87.3 (V). Gains: EP +6.4, DAC +2.6 (D: 87.3−80.9; 97.3−94.7). So most of the RL gain comes from progress and drivable area, and NC drops slightly — [ReCogDrive](https://arxiv.org/html/2506.08052)
- Ablations (PDMS): BC weight 0.001 → 90.8, 0.01 → 90.3, 0.1 → 89.3. Minimum sampling std 0.01 → 89.5, 0.02 → 90.8, 0.04 → 90.5. Discount 0.6 → 90.8, 0.8 → 90.5, 1.0 → 90.3 (V). Conflict: Appendix Table 12 lists BC weight 1e-2, while the best ablation row is 0.001 — [ReCogDrive](https://arxiv.org/html/2506.08052)
- The repo README lists Base-RL 90.8 and Large-RL 90.4 PDMS, so the larger variant did not gain more from RL (V, README) — [ReCogDrive repo](https://github.com/xiaomi-research/recogdrive)

**DiffusionDriveV2 (truncated, anchored diffusion; ResNet-34; no VLM)**
- Policy form: π_θ(τ_{t−1}|τ_t) = N(μ_θ, η(1−α_t)I). Training uses η = 1 ("equivalent to DDPM") and evaluation uses η = 0. Inference uses 2 truncated denoising steps. The estimator is REINFORCE summed over the truncated steps with discount γ = 0.8. The paper describes no importance ratio or clipping (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- Exploration: scale-adaptive multiplicative noise τ′ = (1+ε)τ, with separate longitudinal and lateral scalars, replaces additive per-point noise, which gave "jagged ... broken line" paths. Std floors: 0.04 for sampling and ≥0.1 when evaluating log π, "to avoid large-magnitude gradients" (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- Intra-Anchor GRPO normalizes advantages only within one anchor's group, so turning and going straight are never compared. Inter-Anchor Truncated GRPO gives collisions advantage −1 and clamps all other advantages to max(0, A). L = L_RL + λ·L_IL, with "BC loss weight 0.1" in Table 7. Training: 8× L20, RL 10 epochs, batch 512, lr 2e-4, initialized from DiffusionDrive (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- NAVSIM v1: DiffusionDrive 88.1 → DiffusionDrive + selector (pre-RL) 89.1 → DiffusionDriveV2 (RL + selector) 91.2. EP 82.2 → 87.5, DAC 96.2 → 97.9 (V). Like-for-like RL gain with the selector held fixed: +2.1 (D: 91.2−89.1). The paper's headline +3.1 (D: 91.2−88.1) includes the selector — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- Ablations, short schedule (PDMS): additive noise 89.7 vs multiplicative 90.1; intra-anchor GRPO off 89.2 vs on 90.1; inter-anchor truncation off 89.5 vs on 90.1 (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- NAVSIM v2 navtest EPDMS: 85.5. The paper reports no pre-RL EPDMS (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- Failure modes the paper names: vanilla diffusion "converg[es] to a single high-probability mode"; naive cross-anchor advantages would make the policy "collapse to the single, more common 'go straight' mode"; entropy collapse, handled by the std floor; and lost global comparability when groups are isolated, handled by the truncation (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)

**DriveWLA (flow-matching DiT action expert, VLM frozen)**
- Flow-GRPO on the action expert: 10 stochastic flow steps, step discount γ = 0.8, group size 8. Groups are drawn only from scenes where the imitation policy scores PDMS < 90. The reward is a vector: NC, DAC, EP, TTC and C are standardized within the group and weighted 2/2/5/5/2, with variance shrinkage τ = 0.1. λ_BC = 0.1. Only the action-branch LoRA (rank 16) is trained; the VLM is frozen in both the imitation and RL stages (V) — [DriveWLA](https://arxiv.org/html/2610.17903v1)
- NAVSIM v1: imitation 90.8 → Flow-GRPO 92.1 → factorized-advantage Flow-GRPO† 92.3 (V); +1.3 / +1.5 (D: 92.1−90.8; 92.3−90.8). v2 EPDMS 90.1, with no imitation-only v2 number (V) — [DriveWLA](https://arxiv.org/html/2610.17903v1)

**EponaV2 (rectified-flow planner inside a world model; Qwen3-VL-4B language part, 6.7B total)**
- Flow-GRPO-style SDE with σ_t = a·√(t/(1−t)). Each step is a Gaussian policy N(μ_θ, σ_t²|Δt| I). L = L_rl + λ_il·L_il "to stabilize". Only the planner is trained in RL, on navtrain (V) — [EponaV2](https://arxiv.org/html/2605.14696v1)
- The reward is **not** PDMS: it is the discrepancy between the simulated rollout and the ground-truth trajectory, because "calculating PDMS requires manual perception labels" (V) — [EponaV2](https://arxiv.org/html/2605.14696v1)
- NAVSIM v1: 89.4 → 90.4 with GRPO (V); +1.0 (D). Whole-model training took about 10 days on 64 H20 (V) — [EponaV2](https://arxiv.org/html/2605.14696v1)

**TrajHF (DDPM diffusion policy, RLHF)**
- 10-step SDE denoising for both training and inference, cast as a T-step MDP with the reward only at the terminal step. L = L_RL + α·L_BC, with α = 0.01 on NavSim and no KL term. NavSim GRPO fine-tuning: 45 epochs on 16 GPUs, per-GPU batch 16 (V) — [TrajHF v2](https://arxiv.org/html/2503.10434v2)
- PDMS: single sample 86.4; EM over 15 candidates 87.6; PDMS-oracle selector 94.3 (the paper's grey "upper bound"). There is no pre-RL vs post-RL row, so the RL gain cannot be computed. The v1 abstract claimed 93.95, and the v2 abstract retreats to "comparable to the state-of-the-art" (V for v2 tables; R for the v1 claim) — [TrajHF v2](https://arxiv.org/html/2503.10434v2), [search summary of v1](https://arxiv.org/abs/2503.10434?context=cs)
- Stability: DPO was "unstable" and overfit one preference dimension. RL on the "Normal" split raised meanADE from 0.7734 to 2.5843 for the defensive style, which contradicts the text's "without significant performance degradation" (V) — [TrajHF v2](https://arxiv.org/html/2503.10434v2)

**DriveDPO (not diffusion: a softmax over 8,192 k-means anchors)**
- DPO log-probs are the anchor softmax probabilities. K = 1024 candidates per iteration, β = 0.1. The rejected trajectory is the anchor closest to the human trajectory with PDMS < 0.3. NAVSIM v1: 88.8 without DPO → 90.0 with Safety DPO (V); +1.2 (D). 6× L20 (V) — [DriveDPO](https://arxiv.org/html/2509.17940v1)

**PaIR-Drive (GRPO on a separate tree-structured RL sampler that runs beside an IL head; no VLM)**
- G = 15, clip 0.2, dynamic KL, 4× L40 for 50 epochs. Transfuser 84.0 → 89.7; sequential GRPO 87.9. DiffusionDrive 88.1 → 91.2. Best-of-6: Transfuser 93.3, DiffusionDrive 94.0. v2 EPDMS: Transfuser 79.7 → 86.6, DiffusionDrive 84.3 → 87.9 (V) — [PaIR-Drive](https://arxiv.org/html/2603.13842v1)
- Side effect: EC (extended comfort, v2) fell from 87.7 to 74.2 for Transfuser after RL (V). This looks like reward-shaped comfort erosion — [PaIR-Drive](https://arxiv.org/html/2603.13842v1)

### Inferences
- The continuous-head RL recipe on NAVSIM is settled and cheap to apply to a small head:
  - Use stochastic DDPM or SDE sampling during training and deterministic sampling at evaluation.
  - Sum the exact per-step Gaussian log-probs.
  - Use a group-standardized PDMS advantage and one on-policy update, with no clip.
  - Add a BC loss (weight 0.001–0.1) instead of KL.
  - Floor the std at about 0.02–0.04 for sampling and about 0.1 inside the log-prob.
  - Discount the denoising steps (γ 0.6–0.8).
  
  Every hyperparameter above is attested in at least two NAVSIM papers (ReCogDrive, DiffusionDriveV2, DriveWLA).
- Truncated or anchored diffusion (2 steps) and flow matching (10 SDE steps) both admit this recipe. DiffusionDriveV2 shows that anchors need anchor-aware advantage grouping, or GRPO collapses onto the dominant "go straight" mode. That is extra design work specific to anchored heads.
- Gains shrink as the IL start gets stronger: about +4.3 from 86.5 (ReCogDrive), +2.1 from 89.1 (DiffusionDriveV2 with selector), +1.0 from 89.4 (EponaV2), +1.3–1.5 from 90.8 (DriveWLA).
- RL raises EP and DAC and leaves NC flat or slightly lower. It can erode comfort-type v2 terms (PaIR-Drive EC −13.5, D: 87.7−74.2), so v2 EPDMS should be monitored when optimizing v1 PDMS.

### Gaps
- DDPO (Black et al.) was not fetched in this session. Its score-function and importance-sampling variants are background knowledge only, not verified here.
- DPPO's full text was not read. Which denoising steps it fine-tunes and its exact std clipping are unverified.
- ReCogDrive's group size G and its RL GPU-hours are not in the paper or README, and the RL launch script URL I tried returned 404. DiffusionDriveV2 also omits G and the number of anchors.
- None of these papers reports RL wall-clock or GPU-hours separately from the whole training run.
- I found no NAVSIM paper that applies ReinFlow specifically (learnable noise, 1–4 steps).

## Q2. Masked and discrete diffusion RL (d1/diffu-GRPO, wd1, SPG, UniGRPO, coupled-GRPO, TraceRL; WAM-Diff, DriveFine, ReflectDrive, AutoVLA): likelihood estimation, results, difficulty relative to continuous

### Takeaway
Masked-diffusion LLMs have no tractable exact sequence likelihood. The generic methods approximate it in different ways:
- **d1** and **WAM-Diff**: a one-step mean-field estimate.
- **coupled-GRPO**: complementary masks to cut variance.
- **SPG**: an upper and a lower bound.
- **wd1**: a ratio-free weighted likelihood.
- **TraceRL** and **DriveFine**: log-probs along the actual unmasking trajectory.

In driving, masked and discrete-flow VLAs report the largest single-sample RL gains from comparable starting points: +2.9 to +4.4 PDMS from about 86.6. All of them run RL on the whole 1.5–8B model, not on a separate head. ReflectDrive has no RL at all in its arXiv version; its gain comes from training-free, scorer-guided inpainting. AutoVLA's GRFT is autoregressive over discrete tokens, not diffusion.

### Cited Findings
**Generic dLLM methods**
- **d1 / diffu-GRPO**:
  - Likelihood: per-token log-prob from a single forward pass with the completion fully masked and the prompt randomly masked (p_mask = 0.15). The sequence log-prob is a mean-field sum of the per-token terms.
  - Masking rate: p_mask 0.1 and 0.3 were most stable; 0.5 was less stable; 0.7 degraded after about 3000 steps.
  - Settings: μ up to 12–24 inner updates; LLaDA-8B-Instruct with LoRA r = 128; 8× A100-80G for 3,800–7,700 RL steps (V).
  - Results (best length, base → diffu-GRPO): GSM8K 78.2 → 81.9, MATH500 36.2 → 39.2, Countdown 20.7 → 37.1, Sudoku 11.7 → 18.4 (V).
  - Source: [d1](https://arxiv.org/html/2504.12216v2)
- **wd1**: a "ratio-free" objective that rewrites RL as a weighted log-likelihood, so only the current policy's likelihood has to be approximated (d1 approximates the current, old and reference likelihoods). Reports up to +59% accuracy over d1 on LLaDA-8B, and wd1++ reaches 44.2% MATH500 / 84.5% GSM8K after 20 RL steps (R, abstract) — [wd1](https://arxiv.org/abs/2507.08838)
- **SPG**: argues that the ELBO is a one-sided approximation that biases the policy gradient, and uses both an upper and a lower bound on the log-likelihood. Reported over state-of-the-art dLLM RL: +3.6% GSM8K, +2.6% MATH500, +18.4% Countdown, +27.0% Sudoku. It also beats ELBO and one-step estimators (R, abstract) — [SPG](https://arxiv.org/abs/2510.09541)
- **coupled-GRPO (DiffuCoder-7B)**: complementary mask noise to reduce the variance of token log-likelihood estimates; +4.4% EvalPlus (R, abstract) — [DiffuCoder](https://arxiv.org/abs/2506.20639)
- **TraceRL (TraDo)**: builds the "preferred inference trajectory" into post-training and adds a diffusion-based value model for stability; supports block diffusion. TraDo-8B shows +6.1% relative math accuracy over Qwen2.5-7B-Instruct (R, abstract) — [TraceRL](https://arxiv.org/abs/2509.06949)

**Driving: WAM-Diff (LLaDA-V, 8.4B, masked diffusion, GSPO)**
- Method: per-token log-probs by one-step unmasking "following" d1. The GSPO sequence ratio is length-normalized, (π_θ/π_old)^(1/|x|), with PPO clipping, group-standardized advantages and no KL mentioned. Group size G = 3. Waypoints are quantized over [−100, 100] at 0.01, giving 20,001 tokens and a 32-token output (V) — [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- NAVSIM v1 (Table 5): 86.6 (+MoE, before RL) → 91.0 (+GSPO). Sub-metrics: NC 98.0→99.1, DAC 95.5→98.3, TTC 94.2→96.5, EP 80.7→84.4 (V). Gain +4.4 (D: 91.0−86.6); the text claims +5.3, which conflicts with the table (V) — [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- Group-size ablation: no RL 86.6, G = 2 88.9, G = 3 91.0. Reward ablation: TTC-only 90.7, Comfort-only 90.6, EP-only 90.3, full 91.0. Decoding order: random 90.0, causal 88.9, reverse-causal 91.0 (V) — [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- GSPO vs GRPO: Fig. 10 shows a "steady higher reward than GRPO", with no numbers. The authors argue that token-level GRPO credit is unstable when MoE routing changes (V, qualitative) — [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- NAVSIM v2 EPDMS 89.7, with no pre-RL v2 number (V) — [WAM-Diff](https://arxiv.org/html/2512.11872v1)

**Driving: WAM-Flow (Janus-1.5B, discrete flow matching / CTMC, GRPO)**
- Method: per-token ratios over action tokens. The paper does not explain how the CTMC path likelihood is computed. Reward: (NC × DAC) × weighted mean of EP:TTC:C at 5:5:2, with NC = 0.5 for collisions with static objects. G = 3 (G = 2 → 89.2, G = 4 → 89.6, "excessive variance"). RL for 0.5 epoch on 103K NAVSIM samples, batch 32, lr 1e-6, on 32 Ascend 910B NPUs (V) — [WAM-Flow](https://arxiv.org/html/2512.06112v2)
- NAVSIM v1: 86.7 → 90.3 after GRPO; DAC 95.1→98.3, TTC 94.4→97.0, EP 81.8→82.3 (V); +3.6 (D). Steps after RL: 1 step 89.1 (0.09 s), 5 steps 90.3 (0.48 s), 10 steps 90.2. v2 EPDMS 84.7 (V) — [WAM-Flow](https://arxiv.org/html/2512.06112v2)

**Driving: DriveFine (LaViDa/LLaDA-8B, masked diffusion with a block-MoE refinement expert)**
- Generation-expert GRPO:
  - Samples follow the real denoising path over s steps, aggregating every τ neighbouring steps, and keep ⌊s/τ⌋ sampled paths. The ratio is computed only on the tokens unmasked at that step. This is a trajectory-aware estimate, not mean-field.
  - The advantage is r − mean with no std division, citing d1. The objective is clipped GRPO with KL to the reference.
  - Group size: 10 in the RFT stage. The text says 8 is best, which conflicts.
  - RFT: 1 epoch, batch 16, lr 1e-6 (V).
  - Source: [DriveFine](https://arxiv.org/html/2602.14577)
- Refiner RL: the offline advantage is the pairwise reward-difference matrix r_i − r_j over the generator's samples. The online advantage compares refined samples against their source trajectory (V) — [DriveFine](https://arxiv.org/html/2602.14577)
- NAVSIM v1 (Table 4): SFT 86.7 → +GRPO 89.6 → +offline refine 90.3 → +online refine 90.8 (V). Totals: +2.9 for GRPO alone and +4.1 overall (D: 90.8−86.7). Table 1 headline 90.7; score-based-RFT variant DriveFine* 91.8 (the text says 91.9); best-of-6 DriveFine† 94.2 (V) — [DriveFine](https://arxiv.org/html/2602.14577)
- Separate benchmarks: NAVSIM v2 EPDMS 87.1 (86.7 in the text), and 89.7 on the bug-fixed NAVSIM. Navhard EPDMS: stage 1 74.4, stage 2 41.0, vs ReCogDrive 68.9 / 37.8 (V) — [DriveFine](https://arxiv.org/html/2602.14577)

**Driving: ReflectDrive (LLaDA-V masked diffusion)**
- The fetched arXiv version has **no RL stage**. Training is SFT only, and RL is listed as future work for goal-point selection. Its gain is training-free "reflective inference": goal-conditioned inpainting with K = 3 goals, then safety-guided regeneration. A rule-based PDM-style scorer finds the first unsafe waypoint, a local token search inside a Manhattan neighbourhood δ ≤ 10 fixes an anchor token, and the model re-inpaints the rest. The loop is capped at 10 iterations, and most cases resolve in 1–3 (V) — [ReflectDrive](https://arxiv.org/html/2509.20109)
- PDMS: base 84.8 → goal-only 87.4 → safety-only 90.3 → full 91.1 → ground-truth-agent oracle 94.7 (V). Full gain +6.3 (D: 91.1−84.8) without any RL — [ReflectDrive](https://arxiv.org/html/2509.20109)

**Driving: AutoVLA (Qwen2.5-VL-3B, autoregressive discrete action tokens; not diffusion)**
- 2,048-token K-disk action codebook, 10 tokens of 0.5 s each. GRFT is GRPO with KL β = 0.04, lr 3e-5, 6,000 steps, one update per step with no clip or old-policy tracking. LoRA r = 8 with the vision encoder frozen. Reward = PDMS − λ·(CoT length penalty), λ_r = 0.3. G is not stated (V) — [AutoVLA](https://arxiv.org/html/2506.13757)
- PDMS: SFT 80.54 → post-RFT 89.11 → best-of-6 oracle 92.12 (V). RFT gain +8.57 (D). RFT cut average runtime by 66.8% (V) — [AutoVLA](https://arxiv.org/html/2506.13757)

### Inferences
- How hard each family is:
  - **Continuous heads.** The exact per-step Gaussian log-prob makes RL implementation mechanically simple. The difficulty is in tuning exploration (std floors, noise shape, discounting).
  - **Masked diffusion.** You have to choose a likelihood estimator, and the field has not converged on one. The options are biased one-step mean-field (d1, WAM-Diff), trajectory-aware (DriveFine, TraceRL), bounds (SPG) and ratio-free (wd1).
- A small masked-diffusion head emitting only 16 waypoint tokens (8 × 2) in a few unmasking steps can afford the exact trajectory likelihood. That is the product of the categorical probabilities of the tokens committed at each step under the realized unmasking order, which is the DriveFine and TraceRL idea. This removes most of the estimator question that matters for 256-token reasoning dLLMs.
- Discrete heads get a free inference-time capability: inpainting with fixed anchor tokens. ReflectDrive gets +6.3 from scorer-guided re-inpainting with no RL. A continuous head would need guidance or anchored re-noising to do the same.
- Driving discrete-diffusion RL works with tiny groups (G = 3 in WAM-Diff and WAM-Flow; G = 4 was worse in WAM-Flow). Generic dLLM RL uses larger groups. Small G keeps PDM-scorer calls cheap.

### Gaps
- UniGRPO (MMaDA, arXiv 2505.15809) was not fetched. Its masking scheme is unverified here.
- The full texts of wd1, SPG, coupled-GRPO and TraceRL were not read, so their numbers are R (abstract only).
- None of the driving masked-diffusion papers applies wd1, SPG or coupled-GRPO, and none runs a head-to-head of likelihood estimators on NAVSIM.
- The "ReflectDrive RL" in the brief could not be found. The fetched arXiv HTML has no RL stage; a later version or a follow-up might add one.
- I found no NAVSIM paper that runs RL on a *separate small* masked-diffusion head over frozen or cached VLM features. Every discrete-diffusion driving RL paper trains the whole 1.5–8B model.

## Q3. RL on the VLM itself vs on the head: which papers train which part, and what each buys

### Takeaway
Papers fall into two groups.
- **Head-only RL** with the VLM or backbone frozen: ReCogDrive's diffusion planner, DriveWLA's flow expert (LoRA), EponaV2's planner, and DiffusionDriveV2, which has no VLM. Reported gains are +1.0 to +4.3.
- **Whole-model RL**, where the VLM *is* the action generator: WAM-Diff, WAM-Flow, DriveFine, AutoVLA, Curious-VLA (text waypoints) and NoRD. Reported gains are +2.7 to +8.6, with larger gains from weaker SFT starts.

No paper runs a controlled comparison of RL on the VLM against RL on the head for the same model. The only cross-claim is DriveFine's figure-level assertion that loosely coupled diffusion planners reward-hack under PDMS RL.

### Cited Findings
- **ReCogDrive**: stage 2 freezes the VLM and trains the diffusion planner (DDPM, 200 epochs). Stage 3 RL "optimizes the diffusion model", and the text does not explicitly restate the VLM's status (V; VLM frozen in RL = R/inferred) — [ReCogDrive](https://arxiv.org/html/2506.08052)
- **DriveWLA**: the VLM, VLM adapter, world expert and base action weights are frozen, and RL trains only the action-branch LoRA. 90.8 → 92.3 (V) — [DriveWLA](https://arxiv.org/html/2610.17903v1)
- **EponaV2**: RL "fine-tun[es] only the planner". 89.4 → 90.4 (V) — [EponaV2](https://arxiv.org/html/2605.14696v1)
- **Curious-VLA** (Qwen2.5-VL-3B, pure text-waypoint VLM; the closest analogue to a text-waypoint Qwen VLA):
  - RL: GRPO with group size 8, batch 256, 130 steps across 3 outer loops, on 8× H100. The reward is a focal-rescaled PDMS.
  - NAVSIM v1: SFT 87.6 → RL 90.3 → best-of-6 94.8 (V); +2.7 (D). NAVSIM v2 EPDMS 85.3 (V).
  - Exploration collapse after IL: k = 8 sample diversity pADE/pFDE was only 0.090/0.200, rising to 0.641/1.415 after its diversity-oriented SFT plus RL (V).
  - Source: [Curious-VLA](https://arxiv.org/html/2603.06049)
- **NoRD** (Qwen2.5-VL-3B, 2,048-entry k-disc token codebook, no reasoning, 80K SFT samples): navtest PDMS (v1/v2 not stated) SFT 76.66 → GRPO 77.18 → Dr. GRPO 85.62 (V). Gains: +0.52 and +8.96 (D). Settings: G = 8, no KL, 160 RL steps on 30 A100 (V). The failure mechanism is std normalization: groups with very low or very high reward variance get badly scaled advantages, so hard turning scenes produce almost no gradient. Dr. GRPO, which only subtracts the group mean, fixes it (V) — [NoRD](https://arxiv.org/html/2602.21172v2)
- **AutoVLA**: LoRA RL on the whole VLM, which emits action tokens; 80.54 → 89.11 (V) — [AutoVLA](https://arxiv.org/html/2506.13757)
- **WAM-Diff / WAM-Flow / DriveFine**: RL runs on the whole masked or discrete-flow VLA (8.4B / 1.5B / 8–9B). Gains: +4.4 / +3.6 / +2.9 for GRPO alone (V/D, see Q2) — [WAM-Diff](https://arxiv.org/html/2512.11872v1), [WAM-Flow](https://arxiv.org/html/2512.06112v2), [DriveFine](https://arxiv.org/html/2602.14577)
- **DriveFine's claim**: diffusion planners such as ReCogDrive and DiffusionDrive "suffer a significant drop in EPDMS" under PDMS-oriented RFT (Fig. 2), which the authors attribute to "weak coupling" with the VLM causing reward hacking. The text gives no numbers. In contrast, DriveFine's own PDMS-RFT raised both scores: PDMS 86.7 → 90.5, EPDMS 86.8 → 89.1 (claim: R; DriveFine's own numbers: V) — [DriveFine](https://arxiv.org/html/2602.14577)
- **Other GRPO-on-VLM NAVSIM work seen only in snippets**: MTDrive (multi-turn mtGRPO, 91.1 PDMS; R) — [MTDrive](https://arxiv.org/html/2601.22930v1). ELF-VLA targets groups where "all explored actions yield a zero-value driving score" (R) — [ELF-VLA](https://arxiv.org/html/2603.01063)

### Inferences
- For a head that reads **cached** VLM hidden states, head-only RL is the cheap path. Rollouts never re-run the VLM, so the cost per PDMS sample is a small head forward pass plus a PDM-scorer call. That only holds while the VLM is frozen. RL on the VLM itself invalidates the cache and needs a full VLM forward pass per rollout.
- What each part buys:
  - Head RL on a frozen VLM: +1 to +4 PDMS, mostly EP and DAC.
  - VLM RL with tokens as actions: similar or larger gains from equal starts, about 86.6 → 90–91 (WAM-Diff, WAM-Flow, DriveFine), and also improves VLM-level behaviour (AutoVLA's 66.8% runtime cut, Curious-VLA's exploration).
  - The agreed recipe (DiffGRPO on the head, GRPO on the VLM) is consistent with this: the two act on different parameters. No paper reports both stages stacked on NAVSIM with separate deltas.
- With a text-waypoint Qwen VLM, use Dr.-GRPO-style advantages (no std division) and check post-SFT sample diversity before GRPO on the VLM side. NoRD and Curious-VLA both show plain GRPO stalling on narrow SFT policies.

### Gaps
- No paper on NAVSIM runs a controlled ablation of RL on the VLM vs RL on the head vs both for one architecture.
- DriveFine's "EPDMS drop" claim for diffusion planners has no reported numbers.
- ReCogDrive's VLM status during RL is not stated explicitly in the text.

## Q4. What a NAVSIM RL setup needs (metric cache, PDM scorer in the loop, group size, samples per scene, compute)

### Takeaway
Every paper scores rollouts with NAVSIM's PDM simulator, using either PDMS or its sub-terms. That requires a precomputed **navtrain metric cache**; ReCogDrive ships a dedicated script and warns about NumPy-related cache inconsistencies. Group sizes range from 3 (WAM-Diff, WAM-Flow) to 8 (DriveWLA, NoRD, Curious-VLA), 10 (DriveFine) and 15 (PaIR-Drive). RL runs are short: 0.5–2 epochs, or 10 epochs for the small heads. No paper reports RL GPU-hours.

### Cited Findings
- **ReCogDrive repo**:
  - Build the navtrain metric cache with `sh cache_dataset/run_metric_caching_train.sh` (navtest: `run_metric_caching.sh`). The cached metrics "will be used for evaluation during RL training". Use NumPy ≥ 1.26.4 "to avoid potential errors during metric caching".
  - RL launch: `training/run_recogdrive_train_multi_node_rl_2b.sh`. A 2025-09-30 news item reports a fix to "inconsistencies in the training metric cache".
  - (V, repo docs) — [Train_Eval.md](https://github.com/xiaomi-research/recogdrive/blob/main/docs/Train_Eval.md), [repo](https://github.com/xiaomi-research/recogdrive)
- **Reward formulas**:
  - DiffusionDriveV2 and AutoVLA use the standard PDMS = NC × DAC × (5·EP + 5·TTC + 2·C)/12 on the final clean trajectory (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745), [AutoVLA](https://arxiv.org/html/2506.13757)
  - WAM-Flow decomposes it with partial NC credit (V) — [WAM-Flow](https://arxiv.org/html/2512.06112v2)
  - DriveWLA uses group-standardized sub-metric vectors (V) — [DriveWLA](https://arxiv.org/html/2610.17903v1)
  - EponaV2 avoids PDMS altogether (V) — [EponaV2](https://arxiv.org/html/2605.14696v1)
- **Data**: navtrain is about 103K samples (WAM-Diff, WAM-Flow) from "1,192 training scenes"; navtest is "136 evaluation scenes". These "scenes" counts are logs, not samples (V) — [WAM-Flow](https://arxiv.org/html/2512.06112v2), [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- **Group size, samples per scene and compute by paper**:

| Paper (head) | G / samples per scene | RL schedule | Hardware (as reported) | Tag |
|---|---|---|---|---|
| ReCogDrive (diffusion) | not stated | 10 ep, batch 128, lr 1e-4 | 32× H20 for whole pipeline | V |
| DiffusionDriveV2 (trunc. diffusion) | G per anchor, not stated | 10 ep, batch 512, lr 2e-4 | 8× L20 | V |
| DriveWLA (flow) | 8, only scenes with IL PDMS < 90 | n/a | 16× A100 for IL; RL n/a | V |
| TrajHF (diffusion) | K = 8 (internal data) | 45 ep, 16 GPUs × batch 16 | n/a | V |
| WAM-Diff (masked) | 3 (2 → 88.9, 3 → 91.0) | 2 ep GSPO | 32× Ascend 910B | V |
| WAM-Flow (discrete flow) | 3 (2 → 89.2, 4 → 89.6) | 0.5 ep, batch 32, lr 1e-6 | 32× Ascend 910B | V |
| DriveFine (masked) | 10 (text: 8) | 1 ep, batch 16, lr 1e-6 | n/a | V |
| AutoVLA (AR tokens) | not stated | 6,000 steps, LoRA r = 8 | 8× L40S for SFT | V |
| Curious-VLA (text) | 8 | 130 steps, batch 256 | 8× H100 | V |
| NoRD (AR tokens) | 8 | 160 steps, batch 128 | 30× A100 | V |
| PaIR-Drive (RL sampler) | 15 (5 → 89.1, 9 → 89.3, 12/15 → 93.3, best-of-N) | 50 ep | 4× L40 | V |
| DriveDPO (anchor softmax) | 1,024 candidates per iteration | 10 ep DPO | 6× L20 | V |

Sources for the table: [ReCogDrive](https://arxiv.org/html/2506.08052), [DiffusionDriveV2](https://arxiv.org/html/2512.07745), [DriveWLA](https://arxiv.org/html/2610.17903v1), [TrajHF](https://arxiv.org/html/2503.10434v2), [WAM-Diff](https://arxiv.org/html/2512.11872v1), [WAM-Flow](https://arxiv.org/html/2512.06112v2), [DriveFine](https://arxiv.org/html/2602.14577), [AutoVLA](https://arxiv.org/html/2506.13757), [Curious-VLA](https://arxiv.org/html/2603.06049), [NoRD](https://arxiv.org/html/2602.21172v2), [PaIR-Drive](https://arxiv.org/html/2603.13842v1), [DriveDPO](https://arxiv.org/html/2509.17940v1)

- **Rough PDM-scorer workload** (D): one navtrain epoch at G = 3 is about 103K × 3 ≈ 309K scorer calls; at G = 8 it is about 824K. DiffusionDriveV2-style 10 epochs at G = 8 would be about 8.2M calls. The same arithmetic covers any G.

### Inferences
- The fixed cost is the navtrain metric cache, built once on CPU, which suits the Anvil CPU allocation. After that, each RL step costs head sampling plus PDM scoring per sample.
- With a small head on cached features, PDM scoring rather than GPU time is probably the bottleneck for G = 8 over 10 epochs. Scoring should be parallelized on CPU workers.
- Filtering to hard scenes (DriveWLA: only IL PDMS < 90) cuts scorer calls and concentrates the gradient where advantages are non-zero. It also addresses NoRD's and ELF-VLA's problem of groups whose rewards are all equal.

### Gaps
- No paper reports RL-stage GPU-hours or PDM-scorer throughput.
- The navtrain metric-cache size and build time were not found.
- ReCogDrive's RL group size could not be recovered: the RL script fetch returned 404.

## Q5. Evidence that RL gains depend on head type, and whether selector or best-of-K gains stack with RL gains

### Takeaway
No paper runs the same RL recipe on two head *families* (continuous vs masked diffusion) under one backbone. Cross-paper, both families start from about 86.5–86.7 and land at 89.6–91.0 after RL, so there is no evidence that the head family limits the RL gain. The starting PDMS predicts the gain better.

Selection does stack with RL:
- DiffusionDriveV2's learned selector adds about +1.0 before RL, and RL adds +2.1 on top.
- RL concentrates sample quality: PDMS@10 rises from 75.3 to 84.4 even as diversity falls from 42.3 to 30.3.
- Oracle best-of-6 still adds +2.8 to +4.5 after RL in every paper that reports it. That headroom is what a learned selector can target.

### Cited Findings
- **Matched-start cross-paper comparison** (single sample, NAVSIM v1):

| Paper | Family | Before RL | After RL | Gain | Tag |
|---|---|---|---|---|---|
| ReCogDrive | continuous DDPM | 86.5 | 90.8 | known +4.3 | given / V |
| WAM-Diff | masked diffusion | 86.6 | 91.0 | +4.4 | V/D |
| WAM-Flow | discrete flow | 86.7 | 90.3 | +3.6 | V/D |
| DriveFine | masked diffusion, GRPO only | 86.7 | 89.6 | +2.9 | V/D |

Sources: [ReCogDrive](https://arxiv.org/html/2506.08052), [WAM-Diff](https://arxiv.org/html/2512.11872v1), [WAM-Flow](https://arxiv.org/html/2512.06112v2), [DriveFine](https://arxiv.org/html/2602.14577)

- **PaIR-Drive** runs the same RL framework on two IL heads: regression Transfuser +5.7 (84.0 → 89.7) and truncated-diffusion DiffusionDrive +3.1 (88.1 → 91.2). The authors note that the "enhancement tends to diminish with stronger IL policies". The difference is confounded with start strength, and the RL acts on a separate sampler, not on the head itself (V) — [PaIR-Drive](https://arxiv.org/html/2603.13842v1)
- **DiffusionDriveV2, selector and RL stacking**: DiffusionDrive 88.1 → +selector 89.1 → +RL (with selector) 91.2 (V). Selector +1.0, RL +2.1 (D). The selector ablation (base 89.7 → coarse-to-fine 89.9 → +rank loss 90.1) adds +0.4 (D) on the short schedule (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- **DiffusionDriveV2, quality of the 20 raw samples** (pre-RL → post-RL): Div 42.3 → 30.3; PDMS@1 93.5 → 94.9; PDMS@5 84.3 → 91.1; PDMS@10 75.3 → 84.4 (V). Diversity falls by 28% relative (D: 12.0/42.3) while the top-10 average rises +9.1 (D). The paper also flags selector dependence as an out-of-distribution risk (V) — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- **Oracle best-of-6 remaining after RL** (not deployable; uses the PDMS oracle):
  - AutoVLA 89.11 → 92.12 (+3.0 D) — [AutoVLA](https://arxiv.org/html/2506.13757)
  - DriveFine 90.7 → 94.2 (+3.5 D) — [DriveFine](https://arxiv.org/html/2602.14577)
  - Curious-VLA 90.3 → 94.8 (+4.5 D) — [Curious-VLA](https://arxiv.org/html/2603.06049)
  - PaIR-Drive with DiffusionDrive 91.2 → 94.0 (+2.8 D); with Transfuser 89.7 → 93.3 (+3.6 D) — [PaIR-Drive](https://arxiv.org/html/2603.13842v1)
  - TrajHF single 86.4 → EM over 15 candidates 87.6 (+1.2 D, deployable) → PDMS oracle 94.3 — [TrajHF](https://arxiv.org/html/2503.10434v2)
- **Deployable post-RL selection or refinement**: DriveFine's refiner adds +1.2 on top of GRPO (D: 90.8 − 89.6), and its score-based RFT variant reaches 91.8 (V) — [DriveFine](https://arxiv.org/html/2602.14577). ReflectDrive's scorer-guided reflection adds +6.3 with no RL (V/D) — [ReflectDrive](https://arxiv.org/html/2509.20109). WAM-Diff2 88.3 → 91.1 with selection (given).
- **DriveDPO**: the choice of preference-pair selection matters at the +0.4–0.7 scale (vanilla 89.3, distance-based 89.7, imitation-based 90.0) (V) — [DriveDPO](https://arxiv.org/html/2509.17940v1)

### Inferences
- The head-family decision should rest on implementation cost and on the selector and inference-time story more than on expected RL gain. From about 86.6, continuous and masked/discrete heads both reach about 90–91 single-sample after RL.
- What separates the families:
  - **Continuous.** RL is easy to implement (exact Gaussian log-probs; ReCogDrive's recipe is directly reusable).
  - **Anchored or truncated.** Needs anchor-aware advantages, but pairs naturally with a selector: DiffusionDriveV2 reports the strongest evidence that RL and selection stack.
  - **Masked/discrete.** Needs a likelihood-estimator choice. In return it gets token-level inpainting and refinement for free (ReflectDrive +6.3, DriveFine refiner +1.2), which overlaps with what a learned selector or refiner provides.
- RL narrows sample diversity (DiffusionDriveV2 −28%; Curious-VLA describes IL exploration collapse). The agreed order (RL, then K plans plus a selector) needs a diversity check after RL. Otherwise K samples may collapse and leave the selector little to choose from. DiffusionDriveV2's per-anchor GRPO is one published way to keep K modes alive.
- Post-RL oracle best-of-6 headroom of about 3–4.5 PDMS is consistent across head types, so a learned selector has a similar target whichever head is chosen.

### Gaps
- No paper runs a controlled same-backbone comparison of RL on a continuous vs a masked-diffusion head. The cross-paper comparison is confounded by backbone (1.5B–8B), sensors and scorer versions.
- No paper reports a *learned* (deployable) selector applied after RL on a masked-diffusion head with separate deltas, apart from DriveFine's refiner and score-based RFT variant.
- DiffusionDriveV2's exact definition of PDMS@k (top-k sorted average vs random k) was not confirmed in this pass.
- v2 EPDMS before/after RL is missing in most papers (DiffusionDriveV2, WAM-Diff, WAM-Flow, DriveWLA). PaIR-Drive is the exception (Transfuser 79.7 → 86.6, DiffusionDrive 84.3 → 87.9), along with DriveFine's robustness table (86.8 → 89.1), so effects on v1 and v2 cannot be compared in general.
