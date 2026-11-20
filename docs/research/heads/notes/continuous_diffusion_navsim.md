# Continuous diffusion and flow-matching trajectory planners on NAVSIM (state as of 2026-11-19)

Evidence tags: **V** = read in the paper's full text, tables or code (arXiv HTML or GitHub source, extracted with an automated page reader; table cells were not hand-checked against the PDF). **R** = self-reported in an abstract, search snippet or project page, or copied from another paper's comparison table. **D** = derived here, with the arithmetic shown. Splits are given on every number: **v1 navtest PDMS**, **v2 navtest EPDMS** and **navhard EPDMS** (two-stage) are different metrics and are never mixed. "Sel" = what picks the final plan out of K candidates: learned score head, PDM-style scorer, or oracle best-of-N.

## Q1. NAVSIM scores per planner: before and after RL, with and without test-time selection

### Takeaway
Continuous heads cluster at **86–89 v1 navtest PDMS after supervised training only**, and reach **90–93 after RL and/or learned selection**. The VLA rows closest to our setup are: ReCogDrive's DiT head, text 84.1 → diffusion 86.5 → RL 90.8 on the 2B checkpoint per the README; DiffAdapterVLA, 88.3 SFT → 90.3 RL; DriveVLA-W0's flow expert, 87.2, below its own query regressor at 88.4. Scores above about 92 come from oracle best-of-N, rejection sampling, LiDAR plus reward-conditioned scorers, or training on the test split. They are not like-for-like with an 85.6 single-camera SFT model.

### Cited Findings

#### Master table: NAVSIM v1 navtest PDMS (all rows are v1 navtest unless marked)

| Planner (venue) | Inputs / backbone | Generative type | SFT-only PDMS | After RL | With selection / best-of-N | Tag | Source |
|---|---|---|---|---|---|---|---|
| **ReCogDrive** (ICLR 2026) | 1 front cam, InternVL3 VLM + DiT planner (35M) | DDPM/DDIM, ε-pred | text 84.1 → diffusion 86.5 | 90.8 (DiffGRPO) | none (single sample reported) | V | [paper](https://arxiv.org/html/2506.08052) |
| ReCogDrive README checkpoints | Base = 2B VLM + 35M; Large = 8B + 35M | same | 2B: VLM 84.1 → IL 86.5; 8B: VLM 86.4 → IL 86.5 | 2B: 90.8; 8B: 90.4 | – | V (README) | [GitHub](https://github.com/xiaomi-research/recogdrive) |
| ReCogDrive head ablation (same VLM) | – | text / MLP / query / vanilla diffusion / refined DiT | 84.1 / 75.5 / 85.0 / 85.4 / 86.5 | – | – | V | [paper Table 4](https://arxiv.org/html/2506.08052) |
| **DiffusionDrive** (CVPR 2025) | Cam + LiDAR, ResNet-34 (Transfuser) | truncated DDIM from 20 k-means anchors, x0-pred | 88.1 | – | top-1 by learned confidence over 20 anchors | V | [paper](https://arxiv.org/html/2411.15139) |
| TransfuserDP (DiffusionDrive's vanilla baseline) | same | vanilla DDIM, 20 steps, UNet | 84.6 | – | – | V | [paper Table 2](https://arxiv.org/html/2411.15139) |
| TransfuserTD (truncated, UNet) | same | truncated, 2 steps | 85.7 | – | – | V | same |
| **DiffusionDriveV2** (arXiv 2512.07745) | Cam + LiDAR, ResNet-34 | truncated diffusion + GRPO | 88.1 (= DiffusionDrive start point) | 91.2 (with mode selector) | selector alone on DiffusionDrive: 88.1 → 89.1 | V | [paper](https://arxiv.org/html/2512.07745) |
| **GoalFlow** (CVPR 2025) | Cam + LiDAR, V2-99 (main row) | rectified flow, Gaussian σ=0.1 init, goal-point guided | 85.6 (M0: no goal, average of samples) | – | 90.3 (goal scorer + trajectory scorer over 128–256 samples); 92.1 with GT endpoint as goal | V | [paper](https://arxiv.org/html/2503.05689) |
| GoalFlow, ResNet-34 (as re-reported by DiffusionDriveV2) | ResNet-34 | same | – | – | 85.7 | R | [DiffusionDriveV2 Table 1](https://arxiv.org/html/2512.07745) |
| **TransDiffuser** (arXiv 2505.09315) | Cam + LiDAR, frozen Transfuser encoder, 251M total | DDPM ε-pred, 10 steps, no anchors | – | – | 94.85 (94.9 in table), 30 candidates + **rejection sampling following TrajHF** + top-1 score | V | [paper](https://arxiv.org/html/2505.09315) |
| **DiffE2E** (arXiv 2505.19516) | Cam + LiDAR, V2-99 | hybrid diffusion + supervised queries, x0-pred, 2 steps | 92.7 | – | none described | V | [paper](https://arxiv.org/html/2505.19516) |
| DiffE2E as re-reported by DIVER | – | – | 89.8 | – | – | R (conflicts with 92.7) | [DIVER Table II](https://arxiv.org/html/2507.04049) |
| **DIVER** (arXiv 2507.04049) | Cam + LiDAR, ResNet-34 | diffusion + RL for diversity, 6 anchors | – | 88.3 | highest-score sample | V | [paper](https://arxiv.org/html/2507.04049) |
| **ResAD** (CVPR 2026) | Cam + LiDAR, ResNet-34 / V2-99 | DDPM train (T=1000), DDIM 2 steps, x0-pred on normalized residuals | 88.8 (R34), 90.6 (V2-99) | – | Trajectory Ranker over K_infer=200 (20 in Table 5) | V | [paper](https://arxiv.org/html/2510.08562v2) |
| **FlowR2A** (NeurIPS 2026, arXiv 2606.24231) | Cam + LiDAR, ResNet-34 (Transfuser) | reward-conditioned flow matching, 20 Euler steps, CFG w=5 | 90.0 (single proposal) | – (reward-conditioned, not RL) | 92.8 (60 proposals + 2-layer transformer selector) | V | [paper](https://arxiv.org/html/2606.24231) |
| **DriveVLA-W0** (arXiv 2510.12796) | 1 front cam, VLA + 500M action expert | query regression / flow matching (10 steps) / AR FAST tokens | query 88.4; **flow 87.2**; AR 85.3 (same pretrained VLA, Table 4) | – | 90.2 (query + Hydra-style multi-anchor); 93.0 (AR expert, best-of-6 following AutoVLA) | V | [paper](https://arxiv.org/html/2510.12796) |
| **DiffAdapterVLA** (arXiv 2609.15322) | 1 cam, frozen Cosmos-Reason2-8B (Alpamayo-1.5), adapters on last 12 layers | DDPM ε-pred, 100 train / 5 inference steps (x0 projection) | 88.3 | 90.3 (DiffGRPO, 8 samples per condition) | none, single output | V | [paper](https://arxiv.org/html/2609.15322) |
| Same-paper baselines in DiffAdapterVLA | Cosmos-Reason2 VLM + external DiT reading the VLM | – | DiT+LoRA 81.9; DiT no LoRA 76.4; Alpamayo 1.5 stage-2 action expert 71.1 | – | – | V | [paper Table 2](https://arxiv.org/html/2609.15322) |
| **DP-VLA / DIPOLE** (arXiv 2601.00898) | 3 front cams, Florence-2 encoder, ~1B total | "diffusion action head" (type not stated) | 88.3 | 89.7 (DIPOLE, trained on navtrain); 89.0 (DPPO, navtest-trained) | 94.8 only when RL is **trained on the navtest split** "without GT"; not comparable | V | [paper Table 4](https://arxiv.org/html/2601.00898) |
| **Epona** (ICCV 2025) | camera, 2 s history, autoregressive video world model | rectified-flow TrajDiT (2 layers, ~50M), 100 steps | 86.2 (78.1 without video prediction) | – | – | V | [paper](https://arxiv.org/html/2506.24113) |
| **VITAL** (arXiv 2610.20604) | front image + ego + command; video world-action model (SimWAM-based) | flow matching, velocity target | 90.8 with **1 step**, within 0.13 of 10 steps | bounded residual RL actor (scores not visible) | none ("avoids candidate search") | R (abstract/intro text; tables did not render) | [paper](https://arxiv.org/html/2610.20604v1) |
| GuideFlow (CVPR 2026) | multi-view cams, ResNet-34 | rectified flow + constraints + EBM | no v1 PDMS reported | – | – | V | [paper](https://arxiv.org/html/2511.18729v3) |
| DiffVLA (Bosch/Tsinghua) | Senna-VLM (Vicuna-7B) + VoV-99 BEV, truncated diffusion, 32 anchors | DiffusionDrive-style | no navtest number; see navhard table | – | – | V | [paper](https://arxiv.org/html/2505.19381) |

**Context rows (not continuous diffusion/flow, or numbers only from other papers' tables):**
- AdaThinkDrive (InternVL3-8B, **text** waypoints x, y, heading): Non-Think SFT 83.3, Think SFT 86.2, Non-Think RL 88.3, Think RL 88.9, AdaThinkDrive 90.3; **best-of-4 with the NAVSIM reference evaluator 93.0** (oracle selection) — V, [paper](https://arxiv.org/html/2509.13769).
- AutoVLA (discrete action tokens) 89.1; DriveDPO (Cam + LiDAR) 90.0; WoTE 88.3; BrainWAM 89.5; CoWorld-VLA 89.8; OneVL 88.8 — R, from [DiffAdapterVLA Table 1](https://arxiv.org/html/2609.15322).
- DriveSuprim (selection over 8192 vocabulary, ResNet-34) 89.9; Hydra-MDP (V2-99) 90.3; ARTEMIS 87.0 — R, [DiffusionDriveV2 Table 1](https://arxiv.org/html/2512.07745). iPad (64 proposals, ResNet-34) 91.7 — R, [FlowR2A Table 1](https://arxiv.org/html/2606.24231).
- WAM-Flow (CVPR 2026) is **discrete** flow matching (other researcher's scope). Per search snippets: 1 step 89.1, 5 steps 90.3 v1 PDMS, v2 EPDMS 84.7 — R, [arXiv 2512.06112](https://arxiv.org/html/2512.06112v2).
- DriveWorld-VLA 91.3 v1 PDMS per search snippet — R, [arXiv 2602.06521](https://arxiv.org/html/2602.06521).
- DR-VLA (flow-matching action expert, quadruped/web-video pretraining) 92.1 v1 PDMS, 89.5 without its world expert — R (search snippet), [arXiv 2610.18373](https://arxiv.org/html/2610.18373v1).

#### NAVSIM v2 navtest EPDMS (kept separate)
- DiffusionDriveV2 85.5; DriveSuprim 83.1; ARTEMIS 83.1; Hydra-MDP++ 81.4; Transfuser 76.7 — V, [DiffusionDriveV2 Table 2](https://arxiv.org/html/2512.07745).
- FlowR2A (ResNet-34) 88.9; DriveSuprim V2-99 86.0; Hydra-MDP++ V2-99 85.1 — V for FlowR2A, R for the others, [FlowR2A Table 2](https://arxiv.org/html/2606.24231).
- DiffAdapterVLA 86.3 (EC 61.4); DriveVLA-W0 86.1; ReCogDrive 83.6; Curious-VLA 85.3 — V for DiffAdapterVLA, R for the others, [DiffAdapterVLA Table 7](https://arxiv.org/html/2609.15322).
- DriveVLA-W0 86.1 (EC 58.9); the paper does not say which expert was used — V, [paper Table 2](https://arxiv.org/html/2510.12796).
- ReCogDrive 83.6 EPDMS (EC 86.5) in its "extended metrics" table — V, [paper Table 6](https://arxiv.org/html/2506.08052).
- ResAD 85.5; DiffusionDrive 84.5 — V for ResAD, R for DiffusionDrive, [ResAD Table 2](https://arxiv.org/html/2510.08562v2).
- VITAL 91.1 EPDMS on v2 navtest — R (abstract), [paper](https://arxiv.org/html/2610.20604v1).

#### navhard (two-stage) EPDMS (kept separate)
- GuideFlow ResNet-34 without scorer 27.1; GuideFlow + GTRS-Dense scorer 43.0; GuideFlow + scorer with changed scoring strategy† 46.7. Same table: LTF 23.1, DiffusionDrive* 24.2, GTRS-DP* 23.8, GTRS-Dense (V2-99, scorer) 41.7, DriveSuprim (V2-99) 42.1, DiffVLA 45.0 — V for GuideFlow, R for the others, [GuideFlow Table 1](https://arxiv.org/html/2511.18729v3). The paper's claim of "+1.3 over prior best" conflicts with its own DiffVLA row at 45.0.
- GuideFlow's GitHub model zoo lists "Private NavHard" scores of about 49.4–51.3, which conflict with the paper's 43.0 — R (search snippet), [GitHub](https://github.com/adept-thu/GuideFlow).
- DiffVLA 45.0 extended_pdm_score_combined on the private test set of the 2025 Autonomous Grand Challenge (stage-two LK 59.9, TTC 76.5) — V, [paper](https://arxiv.org/html/2505.19381).
- DiffVLA++ (not diffusion: VLM regresses waypoints from the last-layer hidden state, plus an 8192-vocabulary E2E branch, offline scorer ensemble) 49.12 on the ICCV 2025 challenge leaderboard; VLA branch 48.0, E2E branch 43.7 on navhard two-stage — V, [paper](https://arxiv.org/html/2510.17148v4).
- VITAL 38.6 on navhard — R (abstract), [paper](https://arxiv.org/html/2610.20604v1).

#### RL deltas on the same model (v1 navtest PDMS)
- ReCogDrive 2B: 86.5 → 90.8 = **+4.3**; 8B: 86.5 → 90.4 = **+3.9** — D from [README](https://github.com/xiaomi-research/recogdrive).
- DiffusionDriveV2: 88.1 → 91.2 = +3.1 in total, of which the mode selector alone adds 89.1 − 88.1 = +1.0, so GRPO adds about 91.2 − 89.1 = **+2.1** with the selector held fixed — D from [paper](https://arxiv.org/html/2512.07745).
- DiffAdapterVLA: 88.3 → 90.3 = **+2.0** — D from [paper](https://arxiv.org/html/2609.15322).
- DP-VLA: 88.3 → 89.7 = **+1.4** (DIPOLE trained on navtrain) — D from [paper](https://arxiv.org/html/2601.00898).
- DIVER vs DiffusionDrive: 88.3 vs 88.1 = +0.2; its RL targets diversity, not PDMS — D from [paper](https://arxiv.org/html/2507.04049).

### Inferences
- The like-for-like SFT-only band for **single-camera VLA + continuous head** is 86.5 (ReCogDrive) to 88.3 (DiffAdapterVLA, 8B frozen). DriveVLA-W0's flow expert sits at 87.2. Our 85.6 text model is about 1–3 points below that band before any head swap.
- The ReCogDrive README shows the diffusion head gains **+2.4 over text on the 2B VLM (84.1 → 86.5)** but only **+0.1 on the 8B VLM (86.4 → 86.5)** (D). The supervised head gain shrinks as the text baseline gets stronger. At 85.6 we sit between those two cases, so the supervised gain from a continuous head is likely small. The large gain (+3.9 to +4.3) arrives with RL.
- Rows at 92.8–94.9 (FlowR2A with 60 proposals, TransDiffuser with rejection sampling, DiffE2E, DP-VLA trained on the test split) all depend on LiDAR, heavy candidate filtering, or test-split leakage. None of them is a fair target for a 1-camera 0.8B VLA.

### Gaps
- ReCogDrive conflict: the paper text says InternVL3-8B and puts 90.8 in Table 1, while the README assigns 90.8 to the **2B** Base model and 90.4 to the 8B. I could not resolve which backbone the paper's headline row uses.
- DiffE2E: its own Table 2 says 92.7 while DIVER quotes 89.8. The cause is unresolved, possibly a different paper version.
- I found no 2026 VLA with a GR00T- or π0-style flow-matching expert that reports SFT-vs-RL on v1 navtest, other than DriveVLA-W0 (flow expert, no RL) and DiffAdapterVLA (DDPM).
- VITAL's tables (backbone, 10-step score, before/after-RL numbers) did not render, so only the abstract and intro numbers are available.
- A search snippet mentioned an "OSWAM" one-step model at 90.3 PDMS, but I found no primary URL. It is omitted from the table.

## Q2. How each planner is built (formulation, denoiser, conditioning, output, steps, latency)

### Takeaway
Two families dominate. **Anchored x0-prediction diffusion** (DiffusionDrive, DiffusionDriveV2, ResAD, DiffVLA) works in 1–2 steps. **Gaussian-start heads** (ε-prediction DDPM/DDIM or rectified flow) need 5–20 steps for ε-prediction and 1–5 for flow. The VLA heads closest to our design, ReCogDrive and DiffAdapterVLA, are small DiTs (384 hidden) that cross-attend to VLM hidden states, inject ego state via AdaLN or a history encoder, use ε-prediction with 100 training timesteps, and run 5 inference steps.

### Cited Findings
- **ReCogDrive DiT planner**:
  - DiT blocks with SwiGLU, RoPE, QK-Norm and RMSNorm. Each block alternates self-attention over waypoints with cross-attention to VLM hidden states F_h ∈ R^{L×D}, plus an average-pooled global embedding — V, [paper](https://arxiv.org/html/2506.08052).
  - History trajectory is concatenated with the noisy trajectory embeddings. Ego speed and acceleration enter through **AdaLN**. Output is (x, y, heading), x_t ∈ R^{N×3}, with ε-prediction loss — V, [paper](https://arxiv.org/html/2506.08052).
  - **16 layers, hidden 384, head size 8; 100 training timesteps; 5 inference steps; DDPM/DDIM** — V, [paper appendix](https://arxiv.org/html/2506.08052).
  - Planner is 35M parameters — V, [README](https://github.com/xiaomi-research/recogdrive).
  - The text says noise is clipped to ±5.0, but the hyperparameter table says ±3.0 (internal conflict) — V, [paper](https://arxiv.org/html/2506.08052).
  - Planner stage: 200 epochs, batch 512, lr 1e-4 → 1e-6 — V, [paper](https://arxiv.org/html/2506.08052).
- **DiffAdapterVLA**:
  - Trajectory tokens run **inside the last K=12 frozen VLM layers** under asymmetric joint attention. Each such layer has a DiffAdapter (384-dim bottleneck, 8 heads) whose cross-attention uses trajectory states as queries and condition tokens as keys and values — V, [paper](https://arxiv.org/html/2609.15322).
  - Four past ego states (7-dim descriptors) pass through a history encoder into one shared context vector — V, [paper](https://arxiv.org/html/2609.15322).
  - Output: 8 poses at 2 Hz, normalized by the training-data pose range and clipped during sampling — V, [paper](https://arxiv.org/html/2609.15322).
  - DDPM ε-prediction, 100 training steps, 5 deterministic x0-projection steps; only about 3.05% of parameters are trained — V, [paper](https://arxiv.org/html/2609.15322).
- **DriveVLA-W0 flow expert**:
  - A 500M action expert joined to the VLA through Joint Attention (Q/K/V concatenated). Trained with rectified-flow MSE and run with **10 ODE steps** — V, [paper](https://arxiv.org/html/2510.12796).
  - Latency about 145 ms, versus about 74 ms for the query regressor and 95 ms for AR on NAVSIM; the backbone alone takes 117.8 ms — V, [paper](https://arxiv.org/html/2510.12796).
- **DiffusionDrive**:
  - Cascade diffusion decoder with 2 layers, parameters shared across timesteps. Each layer: deformable spatial cross-attention to BEV features at the trajectory coordinates, cross-attention to agent/map queries, FFN, then a timestep-modulation layer — V, [paper](https://arxiv.org/html/2411.15139).
  - An MLP predicts a confidence score and an offset relative to the noisy trajectory. Loss = L1 reconstruction on the anchor nearest the GT, plus BCE on the confidences — V, [paper](https://arxiv.org/html/2411.15139).
  - 60M parameters; 3.8 ms per step and 7.6 ms per plan on an RTX 4090 (45 FPS) — V, [paper](https://arxiv.org/html/2411.15139).
  - Code: `DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample")`; anchors have shape (20, 8, 2); the regression head outputs (x, y, heading) with `heading.tanh()*π` — V, [code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py).
- **GoalFlow**:
  - Rectified flow, x_t = (1−t)x0 + tτ_norm, target velocity v = τ_norm − x0. Initial noise is N(0, σ²I) with σ = 0.1 — V, [paper](https://arxiv.org/html/2503.05689).
  - Attention layers over concatenated environment (BEV + ego), goal (sinusoidal), trajectory and time features — V, [paper](https://arxiv.org/html/2503.05689).
  - Inference time: 1 step 10.4 ms, 5 steps 49.0 ms, 20 steps 177.8 ms (timing hardware not stated) — V, [paper](https://arxiv.org/html/2503.05689).
- **TransDiffuser**:
  - DDPM ε-MSE, 10 steps for training and inference. Cross-attention over {BEV, image, LiDAR, action embedding, ego embedding}. Outputs ego-frame **deltas** x̂_τ = s_τ − s_{τ−1}. No anchors — V, [paper](https://arxiv.org/html/2505.09315).
  - 251M parameters total, 62.8M trainable — V, [paper](https://arxiv.org/html/2505.09315).
- **DiffE2E**:
  - Square-cosine schedule, 100 forward steps, **2 reverse steps**, x0-prediction (L2). Noisy trajectory tokens are concatenated with learned supervision queries, then self-attention and cross-attention to the fused perception, goal and timestep context — V, [paper](https://arxiv.org/html/2505.19516).
  - 42.8 ms on an RTX 3090, versus DiffusionDrive 39.1 ms and GoalFlow 66.8 ms in the same table — V, [paper Table 8](https://arxiv.org/html/2505.19516).
- **FlowR2A**: 4 transformer blocks (self-attention, cross-attention to scene features, AdaLN injection of reward), 20 Euler steps, CFG w = 5. 91.3 ms in total on an H20 at batch 1 with 60 proposals, of which denoising is 70.2 ms — V, [paper](https://arxiv.org/html/2606.24231).
- **GuideFlow**:
  - Rectified flow from a Gaussian prior; anchors are conditions, not starting points. The flow state x_t ∈ R^{T×2} is embedded by MLP + sinusoidal t, cross-attends sequentially to agent then map tokens, and an MLP outputs the velocity — V, [paper](https://arxiv.org/html/2511.18729v3).
  - 100 Euler steps; 3.6 FPS on an RTX 4090 — V, [paper](https://arxiv.org/html/2511.18729v3).
- **Epona TrajDiT**: 2 layers, about 50M parameters, rectified flow, 100 sampling steps. Outputs relative (Δθ, Δx, Δy). About 0.03 s at 10 steps and 0.3 s at 100 steps on an RTX 4090 — V, [paper](https://arxiv.org/html/2506.24113).
- **ResAD**:
  - Predicts the **residual against an inertial (constant-state) reference** with point-wise residual normalization (PRNorm). DDPM T = 1000, DDIM 2 steps, x0-prediction — V, [paper](https://arxiv.org/html/2510.08562v2).
  - 11.4 ms and 68M parameters with the ranker; 7.7 ms without it (RTX 4090) — V, [paper](https://arxiv.org/html/2510.08562v2).
- **VITAL**: predicts 40 ego-frame pose samples over 4 s through a visual-motion scaffold plus a 4×76 residual interface. The flow-matching velocity target is u = z⋆ − ε. Latency was measured on an RTX 4090 at batch 1, but only the 3.29× ratio is given — V (method text), [paper](https://arxiv.org/html/2610.20604v1).
- **End-to-end VLA latencies** in one paper's shared protocol (A800 used for training; the inference GPU is not named): DiffAdapterVLA 80.4 ms; ReCogDrive 739.1 ms; DriveVLA-W0 (flow) 1186.8 ms; Alpamayo 1.5 action expert 1003.8 ms — V, [DiffAdapterVLA Table 2](https://arxiv.org/html/2609.15322).
- ReCogDrive's own timing: text decoding 0.5839 s vs diffusion planner 0.0750 s per sample — V, [paper Table 4](https://arxiv.org/html/2506.08052). That is 0.5839 / 0.0750 = **7.8× faster** (D).

### Inferences
- Our planned design (one prefill, cached hidden states, a small DiT with AdaLN ego, 8×(x, y)) is essentially ReCogDrive's planner minus the heading channel. The 35M / 16-layer / 384-hidden configuration is a sourced starting size.
- For small heads, the reported cost is 3.8–10 ms per denoising step on an RTX 4090. A 1–4 step head costs little next to the VLM forward pass. In DiffAdapterVLA, going from 2 to 5 steps adds only about 12 ms end to end (68.5 → 80.8 ms) (D, from DiffusionDrive's 3.8 ms/step, GoalFlow's 10.4 ms/step and DiffAdapterVLA Table 6).

### Gaps
- No paper gives ReCogDrive's trajectory normalization or its group size G. Its inference GPU is also unstated.
- DriveVLA-W0 gives no flow-expert size separate from the 500M action expert.
- DP-VLA's head type (DDPM vs flow) and its step count are not stated in the fetched text.

## Q3. Truncated / anchored diffusion (DiffusionDrive) and goal-guided flow (GoalFlow)

### Takeaway
Truncation is what makes 2-step diffusion work. DiffusionDrive noises k-means anchors only slightly (training t < 50 of 1000; inference starts at t = 8) and predicts x0. Against vanilla 20-step DDIM it gains +1.1 PDMS at the same architecture, cuts planning time by 9.4×, and lifts diversity from 11% to 70%. GoalFlow shows rectified flow loses only 1.4 PDMS going from 5 steps to 1, but its 90.3 relies on goal-point scoring plus a trajectory scorer over 128–256 samples.

### Cited Findings
- **Anchors**: K-Means on training trajectories, N_anchor = 20 on NAVSIM. Truncation "by 50/1000"; forward noising τ_k^i = √ᾱ^i·a_k + √(1−ᾱ^i)·ε. Inference uses 2 DDIM steps — V, [paper](https://arxiv.org/html/2411.15139).
- **Code details**:
  - Training uses `timesteps = torch.randint(0, 50, ...)`, then clamps to [−1, 1] — V, [code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py).
  - Inference noises anchors at `trunc_timesteps = 8` and then denoises with `roll_timesteps` = [10, 0] (`step_num = 2`, `step_ratio = 20/step_num`) — V, [code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py).
  - Normalization: x → 2(x+1.2)/56.9 − 1, y → 2(y+20)/46 − 1, heading → 2(h+2)/3.9 − 1 — V, [code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py).
- **Noise actually applied** (D). This assumes the diffusers defaults beta_start = 1e-4 and beta_end = 0.02 with scaled_linear betas, since the code does not override them.
  - ᾱ_8 = 0.9990, so the noise std is 0.032 in normalized units, about **0.90 m in x and 0.73 m in y** (0.032 × 56.9/2 and 0.032 × 46/2).
  - At the training maximum t = 49, ᾱ = 0.9911, std 0.094, about **2.7 m in x and 2.2 m in y**.
  - So the "diffusion" is a small Gaussian perturbation of the anchors, refined in 2 steps (computed from the code constants above).
- **Roadmap table** (v1 navtest, RTX 4090) — V, [paper Table 2](https://arxiv.org/html/2411.15139):

| Model | PDMS | Steps | Diversity D | Total planning time | FPS | Params |
|---|---|---|---|---|---|---|
| Transfuser (MLP) | 84.0 | 1 | 0% | 0.2 ms | 60 | 56M |
| TransfuserDP (vanilla DDIM, UNet) | 84.6 | 20 | 11% | 130.0 ms | 7 | 101M |
| TransfuserTD (truncated, UNet) | 85.7 | 2 | 70% | 13.8 ms | 27 | 102M |
| DiffusionDrive (cascade decoder) | 88.1 | 2 | 74% | 7.6 ms | 45 | 60M |

- **What truncation buys** (D):
  - Same UNet decoder: +1.1 PDMS (85.7 − 84.6), 10× fewer steps (20 → 2), 130.0/13.8 = **9.4× faster**, D from 11% to 70% (+59 points).
  - The full DiffusionDrive over vanilla adds +3.5 PDMS and is 130.0/7.6 = 17.1× faster.
  - The text claims a "64%" diversity gain, but the table difference is 63 points — V, [paper](https://arxiv.org/html/2411.15139).
- **Steps ablation**: 1 step 87.9, 2 steps 88.1, 3 steps 88.1 — V, [paper Table 4](https://arxiv.org/html/2411.15139).
- **Sampled noises N_infer**: 10 → 84.9, 20 → 88.1, 40 → 88.2 — V, [paper Table 6](https://arxiv.org/html/2411.15139). The paper does not ablate the number of anchors.
- **Cascade stages**: 1 → 87.4 (59M), 2 → 88.1 (60M), 4 → 88.2 (65M) — V, [paper Table 5](https://arxiv.org/html/2411.15139).
- **GoalFlow goal guidance**:
  - Endpoints (x, y, θ) are clustered into a vocabulary of 4096 or 8192 goals. Each goal is scored by a distance softmax plus a binary drivable-area (shadow-box) score; the best goal is encoded sinusoidally and fed to the flow — V, [paper](https://arxiv.org/html/2503.05689).
  - Component ablation (v1 navtest): base rectified flow, no goal, averaging samples 85.6 → +distance score 88.5 → +DAC score 89.4 → +trajectory scorer 90.3 — V, [paper Table 2](https://arxiv.org/html/2503.05689).
  - A "shadow trajectory" generated with the goal masked replaces the main output if the two differ a lot — V, [paper](https://arxiv.org/html/2503.05689).
- **GoalFlow steps**: 20 steps 89.9 (177.8 ms), 10 steps 90.1 (92.4 ms), **5 steps 90.3 (49.0 ms)**, **1 step 88.9 (10.4 ms)**. The intro's "dropped by only 1.6%" conflicts with the table's 1.4 points — V, [paper Table 3](https://arxiv.org/html/2503.05689).
- **GoalFlow noise scale**: σ = 0.05 → 90.1, 0.1 → 90.3, 0.2 → 49.0, 0.3 → 18.8. Flow from a wide Gaussian fails; the start distribution has to be narrow in normalized space — V, [paper Table 4](https://arxiv.org/html/2503.05689).
- **DiffusionDriveV2** keeps 2 denoising steps and the truncated schedule, then adds GRPO — V, [paper](https://arxiv.org/html/2512.07745).
- **DiffVLA** uses 32 noised k-means anchors from NavTrain in a DiffusionDrive-style planner — V, [paper](https://arxiv.org/html/2505.19381).

### Inferences
- Both 1–2-step successes (anchored DiffusionDrive and narrow-σ GoalFlow) start very close to the data manifold: 0.03–0.1 std in [−1, 1] space. For our head, "few steps" is mostly a matter of where sampling starts and of predicting x0 or velocity. The choice of DDIM versus flow matters less (D, from the σ and truncation numbers above).
- GoalFlow's +4.7 PDMS from goal and trajectory scoring (85.6 → 90.3) is selection and guidance, not the generator. The bare rectified flow sits at 85.6, below DiffusionDrive's 88.1 (D).

### Gaps
- DiffusionDrive never ablates anchor count (for example 20 vs 64 vs 256) on NAVSIM.
- GoalFlow's trajectory normalization operator ℋ and its scorer weights λ1 and λ2 are not given.

## Q4. Multimodality: diversity metrics, mode collapse, how K plans are produced and scored

### Takeaway
Vanilla Gaussian diffusion trained on single-GT demonstrations collapses (DiffusionDrive: 20 noises × 20 steps reach D = 11%; DiT-free TransfuserTD has Div 0.1 in V2's metric). Anchors restore diversity, but most diverse proposals are poor: DiffusionDrive's mean proposal PDMS is 60.3 and its PDMS@10 is 75.3. The final score therefore comes from a learned selector. RL (DiffusionDriveV2) and reward conditioning (FlowR2A) trade raw diversity for much better proposal quality.

### Cited Findings
- **DiffusionDrive diversity metric** D = 1 − mean IoU of each sampled trajectory's area against the union of the others. Vanilla diffusion scores D = 11%: "20 random noises denoised over 20 steps converge to similar trajectories" — V, [paper](https://arxiv.org/html/2411.15139).
- **DiffusionDriveV2** raw outputs (20 trajectories each, before the selector) — V, [paper Table 3](https://arxiv.org/html/2512.07745):

| Model | Div. | PDMS@1 | PDMS@5 | PDMS@10 |
|---|---|---|---|---|
| TransfuserTD | 0.1 | 85.7 | 85.7 | 85.7 |
| DiffusionDrive | 42.3 | 93.5 | 84.3 | 75.3 |
| DiffusionDriveV2 | 30.3 | 94.9 | 91.1 | 84.4 |

- Its RL design: intra-anchor GRPO (advantages normalized only within one anchor's group), inter-anchor truncated GRPO (negative advantages zeroed, collisions −1), multiplicative exploration noise (89.7 additive vs 90.1 multiplicative), BC weight 0.1, denoising discount γ = 0.8 — V, [paper](https://arxiv.org/html/2512.07745).
- **Proposal quality, mean ± std PDMS over all proposals**: DiffusionDrive 60.3 ± 33.1; iPad 77.9 ± 22.4; FlowR2A 89.4 ± 7.4 — V for FlowR2A, R for the others, [FlowR2A Table 3](https://arxiv.org/html/2606.24231). The same table's "1 / 4 / all proposals" columns read 69.7 / 85.3 / 88.1 for DiffusionDrive and 89.4 / 91.8 / 92.9 for FlowR2A. Their exact definition was not visible in the extraction.
- ResAD mean PDMS over its multimodal outputs is 86.1, versus 60.3 for DiffusionDrive — V, [ResAD Table 5](https://arxiv.org/html/2510.08562v2).
- **FlowR2A candidates**: conditions the flow decoder on per-timestep reward targets via AdaLN, with classifier-free reward guidance (w = 5) and anchored sampling. K is swept over (r_high ∈ [0.9, 1.0], t_init ∈ [0.5, 0.9]). 60 proposals are ranked by a 2-layer transformer selector. Ablations: reward granularity (score only 84.5 → sub-rewards 86.7 → per-timestep 89.4) and reward noise augmentation (σ = 0: 84.8 → 0.05: 89.4) — V, [paper](https://arxiv.org/html/2606.24231).
- **TransDiffuser** diversity rises with candidates and steps (diversity metric from its Table III/IV): 10 candidates PDMS 89.6 / D 56; 15 → 91.7 / 63; 30 → 94.9 / 70. Steps 5 → 92.4 / 65; 10 → 94.9 / 70; 20 → 94.3 / 88. Its representation-decorrelation loss: β = 0 → 94.3 / D 66, β = 0.02 → 94.9 / D 70. Selection is rejection sampling (following TrajHF) plus a top-1 score — V, [paper](https://arxiv.org/html/2505.09315).
- **DIVER** time-conditioned diversity (NAVSIM, average over 1–3 s): Transfuser_mmt 0.13 vs DIVER 0.21. It uses 6 anchors and selects the highest-scoring sample by comfort and scene compliance — V, [paper](https://arxiv.org/html/2507.04049).
- **GuideFlow**: one trajectory per anchor across 256 FPS-sampled anchors, scored by GTRS-Dense. No diversity metric; it argues only qualitatively that flow matching "inherently mitigates mode collapse" — V, [paper](https://arxiv.org/html/2511.18729v3).
- **GoalFlow**: 128–256 samples per scene; no diversity metric reported — V, [paper](https://arxiv.org/html/2503.05689).
- **Oracle best-of-N inflation**:
  - AdaThinkDrive 90.3 → 93.0 with best-of-4 using the NAVSIM reference evaluator — V, [paper](https://arxiv.org/html/2509.13769).
  - DriveVLA-W0 AR expert 85.3 → 93.0 with best-of-6 — V, [paper](https://arxiv.org/html/2510.12796).
  - That is +2.7 (D) and +7.7 (D) respectively.
- **Learned selector on DiffusionDrive proposals**: 88.1 → 89.1 — V, [DiffusionDriveV2](https://arxiv.org/html/2512.07745).

### Inferences
- For a selector-based design, sample count matters as much as diversity. DiffusionDrive drops to 84.9 at 10 samples and TransDiffuser to 89.6 at 10 candidates. Both use top-1 by their own scorer. So K around 20–30 seems to be where learned-selector methods saturate (D, from the tables above).
- The gap between oracle PDMS@1 over 20 candidates (93.5 DiffusionDrive, 94.9 V2) and the deployed top-1 (88.1, 91.2) is about 4–5 points. That is the headroom a perfect selector could unlock (D: 93.5 − 88.1 = 5.4; 94.9 − 91.2 = 3.7).

### Gaps
- I found no continuous-head **VLA** (ReCogDrive, DiffAdapterVLA, DriveVLA-W0) that reports diversity or multi-sample metrics. All three report single samples.
- Diversity metrics are not standardized: DiffusionDrive D (%), DiffusionDriveV2 Div., TransDiffuser D and DIVER Div. are on different scales.

## Q5. Training stability and tricks (normalization, loss target, guidance, schedules)

### Takeaway
x0 / "sample" prediction plus tight normalization of trajectories into [−1, 1] are the common thread in the few-step heads that work. ε-prediction heads (ReCogDrive, DiffAdapterVLA) need about 5 steps, and DiffAdapterVLA collapses completely at 1 step. Classifier-free guidance appears in reward- and goal-conditioned flows (FlowR2A w = 5; GuideFlow condition dropout p = 0.2) and in DIPOLE's positive/negative policy guidance.

### Cited Findings
- **Normalization**:
  - DiffusionDrive min-max to [−1, 1]: x ∈ [−1.2, 55.7] m, y ∈ [−20, 26] m, heading ∈ [−2, 1.9] rad, plus clamping — V, [code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py). The bounds are D from the code constants (−1.2 + 56.9 = 55.7; −20 + 46 = 26; −2 + 3.9 = 1.9).
  - DiffAdapterVLA normalizes poses by the training-data range and clips during sampling — V, [paper](https://arxiv.org/html/2609.15322).
  - ResAD's point-wise residual normalization over an inertial reference adds +0.9 (TRM 86.3 → +PRNorm 87.2), then IRP reaches 88.8. On other planners: Transfuser 84.0 → +TRM 85.2 → +PRNorm 85.6; TransfuserDP 84.5 → 85.5 → 85.8 — V, [paper Tables 3–4](https://arxiv.org/html/2510.08562v2).
- **Prediction target**:
  - DiffusionDrive `prediction_type="sample"` (x0) — V, [code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py).
  - DiffE2E x0-prediction. CARLA ablation: switching to noise-prediction output drops Longest6 DS from 82.9 to 20.1 — V, [paper Table 3](https://arxiv.org/html/2505.19516). The same table: full diffusion without the supervised queries 70.1; the hybrid 82.9.
  - ReCogDrive and DiffAdapterVLA use ε-prediction — V, [ReCogDrive](https://arxiv.org/html/2506.08052), [DiffAdapterVLA](https://arxiv.org/html/2609.15322).
  - GoalFlow, GuideFlow, FlowR2A, Epona and DriveVLA-W0 predict rectified-flow velocity — V, [GoalFlow](https://arxiv.org/html/2503.05689), [GuideFlow](https://arxiv.org/html/2511.18729v3), [Epona](https://arxiv.org/html/2506.24113), [DriveVLA-W0](https://arxiv.org/html/2510.12796).
- **Noise scale**: GoalFlow σ = 0.1 is best; σ ≥ 0.2 collapses (49.0, 18.8) — V, [paper Table 4](https://arxiv.org/html/2503.05689).
- **Schedules**:
  - DiffusionDrive scaled_linear, 1000 steps, truncated to 50 — V, [code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py).
  - DiffE2E square-cosine, 100 steps — V, [paper](https://arxiv.org/html/2505.19516).
  - ReCogDrive and DiffAdapterVLA 100 training steps — V, [ReCogDrive](https://arxiv.org/html/2506.08052), [DiffAdapterVLA](https://arxiv.org/html/2609.15322).
  - GoalFlow rescales timesteps nonlinearly at sampling — V, [paper](https://arxiv.org/html/2503.05689).
- **CFG and guidance**:
  - FlowR2A CFG w = 5 ("performance saturates") — V, [paper](https://arxiv.org/html/2606.24231).
  - GuideFlow drops conditions with p = 0.2 and uses v_guide = (1−γ)v_uncond + γv_cond (γ not reported) — V, [paper](https://arxiv.org/html/2511.18729v3).
  - DIPOLE trains positive and negative LoRA policies weighted σ(βG) and 1−σ(βG), and samples ε̃ = (1+w)ε⁺ − wε⁻ — V, [paper](https://arxiv.org/html/2601.00898).
- **RL stability knobs**:
  - ReCogDrive DiffGRPO: BC weight λ = 0.001 → 90.8; 0.01 → 90.3; 0.1 → 89.3. Discount over denoising steps γ = 0.6 → 90.8; 0.8 → 90.5; 1.0 → 90.3. σ_min = 0.02; no PPO clipping; 1 update iteration; 10 RL epochs — V, [paper](https://arxiv.org/html/2506.08052).
  - DiffusionDriveV2: BC 0.1, γ = 0.8, minimum std 0.04 — V, [paper](https://arxiv.org/html/2512.07745).
  - DiffAdapterVLA: DiffGRPO with official PDMS rewards and 8 samples per condition — V, [paper](https://arxiv.org/html/2609.15322).
- **VLM adaptation**: an external DiT on a frozen VLM scores 76.4 without LoRA and 81.9 with LoRA — V, [DiffAdapterVLA Table 2](https://arxiv.org/html/2609.15322).

### Inferences
- For a head that must run 1–4 steps and later be RL-trained, the evidence points to x0 or velocity prediction (not ε), [−1, 1] min-max or residual normalization, and a narrow starting distribution. ReCogDrive's RL settings (small BC weight about 1e-3, step discount 0.6, a σ floor) are the closest published recipe (D, synthesis of the cited findings).

### Gaps
- No NAVSIM paper directly compares ε vs x0 vs v prediction with the head held fixed. DiffE2E's ablation is on CARLA only.
- I found no published comparison of DDIM vs rectified flow under the same architecture on NAVSIM.

## Q6. Same-paper ablations: head type and steps-vs-PDMS curves

### Takeaway
Head type matters less than how the head reads the backbone. A generic MLP from pooled VLM states is bad (ReCogDrive 75.5). A query-based regressor can beat a flow head (DriveVLA-W0: 88.4 vs 87.2). Diffusion beats text by +0.1 to +2.4 before RL. Step curves split by formulation. ε-prediction DDPM needs about 5 steps (DiffAdapterVLA: 1.92 at 1 step, 82.6 at 2, 87.0 at 5). Anchored x0 and flow heads are near flat from 1 to 5 steps (DiffusionDrive 87.9 → 88.1; GoalFlow 88.9 → 90.3; VITAL 1 step within 0.13 of 10 steps).

### Cited Findings
- **ReCogDrive (same VLM, v1 navtest)** — V, [paper Table 4](https://arxiv.org/html/2506.08052):
  - Text 84.1 (0.584 s); MLP 75.5 (0.075 s); query decoder 85.0; vanilla diffusion 85.4.
  - Refinements: +SwiGLU 85.7, +RMSNorm 85.8, +RoPE & QK-Norm 86.5.
- **DriveVLA-W0 (same pretrained VLA, v1 navtest)** — V, [paper Table 4](https://arxiv.org/html/2510.12796):
  - Query regressor 88.4 (74 ms); flow matching 87.2 (10 steps, 145 ms); AR tokens 85.3.
  - On a larger in-house set, flow beats query in ADE (1.0362 vs 1.1248) and collision (0.0398 vs 0.0453). The authors attribute the query head's lag at scale to a representational bottleneck, and flow's to sample inefficiency.
- **DiffAdapterVLA steps (DDPM ε-pred, v1 navtest, 50-epoch setting)**: 1 step 1.92; 2 steps 82.63; 3 steps 84.32; 5 steps 87.00; 10 steps 87.07. End-to-end time per sample: 74.1 / 68.5 / 72.1 / 80.8 / 126.8 ms — V, [paper Table 6](https://arxiv.org/html/2609.15322).
- **DiffAdapterVLA, how the head reads the VLM** — V, [paper Tables 4–5](https://arxiv.org/html/2609.15322):
  - **Final cache only: 80.2**; late-layer conditions with the trajectory outside the VLM: 85.2; trajectory inside the VLM: 86.9.
  - Layer count K: 4 → 73.0; 8 → 84.1; 12 → 86.9; 16 → 82.1; 20 → 81.1.
- **DiffusionDrive steps**: 1 → 87.9; 2 → 88.1; 3 → 88.1 — V, [paper Table 4](https://arxiv.org/html/2411.15139).
- **GoalFlow steps**: 1 → 88.9; 5 → 90.3; 10 → 90.1; 20 → 89.9 — V, [paper Table 3](https://arxiv.org/html/2503.05689).
- **TransDiffuser steps (DDPM ε)**: 5 → 92.4; 10 → 94.9; 20 → 94.3 — V, [paper](https://arxiv.org/html/2505.09315).
- **FlowR2A steps (Euler)**: 10 → 92.2; 20 → 92.8; 30 → 92.7; 40 → 92.7 — V, [paper](https://arxiv.org/html/2606.24231).
- **GuideFlow steps (navhard EPDMS)**: 100 → 27.1; 50 → 25.5; 25 → 23.3; 10 → 21.9 — V, [paper Table 6](https://arxiv.org/html/2511.18729v3).
- **DiffE2E**: PDMS is lowest at 1 step, peaks at 2, then declines. Exact values exist only in its Fig. 5 — V (text), [paper](https://arxiv.org/html/2505.19516).
- **VITAL**: 1 step 90.8 v1 PDMS, within 0.13 of 10 steps, at 3.29× the end-to-end speed — R (abstract), [paper](https://arxiv.org/html/2610.20604v1).
- **Epona**: TrajDiT trained without the video-prediction objective drops from 86.2 to 78.1 — V, [paper Table 5](https://arxiv.org/html/2506.24113).
- **DiffusionDrive head comparison (no VLM)**: Transfuser MLP 84.0 vs vanilla diffusion UNet 84.6 vs truncated UNet 85.7 vs cascade decoder 88.1 — V, [paper Table 2](https://arxiv.org/html/2411.15139).

### Inferences
- For our "read cached hidden states from one prefill" design, DiffAdapterVLA's 80.2 (final cache only) vs 85.2 (multi-layer conditions, trajectory outside the VLM) is the most directly relevant number. Reading several late layers' hidden states, not just the last, was worth about +5 PDMS on a frozen 8B VLM (D: 85.2 − 80.2). Our setting differs: an unfrozen 0.8B backbone, where the last layer is trained for the task.
- Any 1–2 step target argues against plain ε-prediction from N(0, I) (DiffAdapterVLA collapses at 1 step). Anchored x0 or rectified-flow velocity heads show ≤ 1.4 PDMS loss at 1 step (D: DiffusionDrive 88.1 − 87.9 = 0.2; GoalFlow 90.3 − 88.9 = 1.4).
- A query-based regressor head is a strong single-plan control (DriveVLA-W0 88.4 > flow 87.2). But it gives no K candidates and no likelihood for GRPO, which matters for our RL plan.

### Gaps
- No same-backbone NAVSIM comparison exists of continuous flow vs discrete masked-diffusion heads on a VLM. WAM-Flow (discrete flow) is closest but has no continuous twin in the same paper.
- No paper reports steps-vs-PDMS for a VLA flow head below 10 steps, other than VITAL (1 vs 10, abstract only).
- No paper reports the cost of producing K candidate plans in a single batched pass from one VLM prefill. The only K-proposal timing found is FlowR2A's: 60 proposals × 20 steps, 70.2 ms of denoising on an H20.
