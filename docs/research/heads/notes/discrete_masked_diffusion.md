# Discrete / masked-diffusion action heads for driving and robot policies (state as of 2026-11-19)

Evidence tags: **V** = read in the paper's full text or tables (arXiv HTML read with a fetch tool, or PDF text I extracted and read myself for Discrete-WAM and WAM-Diff2). **R** = self-reported claim seen only in an abstract, search snippet, or third-party summary. **D** = derived by me, arithmetic shown. "NAVSIM v1" means navtest PDMS unless marked otherwise. Facts the brief already lists (WAM-Diff 20,001 tokens at 0.01 m; WAM-Flow 76.2→81.1→83.4; DriveFine 4,000 + 1,800 bins; E0 ~2,048-bin saturation; MVLAD-AD 0.54 m at 1 s) are not re-derived here.

## Q1. Which driving models use discrete masked diffusion for the plan, what do they score on NAVSIM v1 before and after RL, and how are their tokens built?

### Takeaway
At least nine public driving planners use masked or discrete diffusion. Without RL or search, most score 84.8–88.3 PDMS on NAVSIM v1. RL, inference-time search or self-editing adds +1.4 to +8.6, giving a final range of 88.9–92.2. Per-axis coordinate tokens (2 per waypoint, so 16 tokens for 8 waypoints) are the dominant design. Goal-first (reverse-causal) decoding beats causal decoding in every driving ablation found. 4–8 steps are enough.

### Cited Findings

**Summary table (NAVSIM v1 navtest PDMS unless stated)**

| Model (backbone, size) | Plan tokens | Steps / order | SFT-only PDMS | After RL / search | Oracle best-of-N |
|---|---|---|---|---|---|
| WAM-Diff (LLaDA-V, 8.4B) | numeric tokens, x,y per waypoint | 32 iters, confidence; reverse-causal best | 86.6 V | 91.0 GSPO V | – |
| WAM-Flow (Janus-1.5B, discrete flow matching) | numeric tokens, metric-aligned embeddings, 8 wp | 1–10 Euler steps | 86.7 V | 90.3 GRPO V | – |
| DriveFine (LaViDa/LLaDA-8B) | 0.05 m position + 0.1° heading bins | 12 steps + 1 refine, cosine, confidence | 86.7 V | 90.7–90.8 GRPO+refine; 91.8 score-based RFT V | 94.2 (best-of-6) V |
| ReflectDrive (LLaDA-V, size not stated) | per-axis 1D codebooks, range ±100 m | 5 steps, low-confidence remask | 84.8 (no reflection) V | 91.1 with reflection, no RL V | 94.7 with privileged GT agents V |
| ReflectDrive-2 (0.7B masked-diffusion LM + 0.1B ViT) | 16 tokens = 8 wp × (long, lat); goal = last pair | 1+3+3 or 1+1 steps | 84.8 → 87.7 (+DACF, +AutoEdit training) V | 91.0 (RL + AutoEdit) V | 94.8 (best-of-6) V |
| MD-Drive (DreamVL dLM) | (x,y) pairs + heading, Δg 0.25 m | 32 steps, distance-aware order | 80.3 V (68.5 with no discretization) | 88.9 GRPO V | pass@8 (Q5) |
| WAM-Diff2 (Qwen3-VL AR→block diffusion, 2B/8B) | text-tokenizer waypoints | saturates at 8–16 steps | 88.3 (no RL, no selection) V | 91.1 with score-based selection V | – |
| Discrete-WAM (~1B decoder-only, Xiaomi) | 60×60 joint acceleration grid with soft labels | confidence-replace rounds | v2 EPDMS 89.1 V | v1 92.2 PDMS; v2 EPDMS 90.4 V | – |
| AutoVLA (AR reference, not diffusion) | K-disk codebook 2048, 0.5 s per token | AR | 80.54 V | 89.11 GRPO RFT V | 92.12 (best-of-6) V |

**Per-model details**

- **WAM-Diff.**
  - LLaDA-V backbone, 8.4B total. Masked diffusion runs in the LLM itself: the text head is extended to 146,350 tokens (126,349 text + 20,001 numeric) V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
  - Each waypoint is an ⟨x,y⟩ pair. Heading is not encoded. Inference output length is 32 tokens; the waypoint count is not stated explicitly V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
  - Decoding: 32 masked-diffusion iterations with a decreasing mask schedule, confidence remasking by default, Fast-dLLM decoding V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
  - Order ablation (Table 7, NAVSIM v1): Random 90.0, Causal 88.9, Reverse-Causal 91.0 V. Decoding scheduler on/off (Table 5): 80.3 → 82.3 V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
  - SFT-only: NC 98.0, DAC 95.5, TTC 94.2, C 99.4, EP 80.7, PDMS 86.6. After GSPO: 99.1 / 98.3 / 96.5 / 99.9 / 84.4, PDMS 91.0 V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
  - NAVSIM v2 EPDMS 89.7 V. No latency is reported. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
  - MoE-LoRA ablation (64 experts, rank 32, +0.5B params, about 0.05B active): no MoE 84.7 → 64 experts 86.6 V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- **WAM-Flow** (discrete flow matching, not absorbing-mask diffusion).
  - Janus-1.5B converted to a non-causal model. Outputs 8 waypoints over 4 s V. [WAM-Flow](https://arxiv.org/html/2512.06112v1)
  - Steps vs PDMS and time (Table 6): 1 step 89.1 / 0.09 s; 2 steps 89.7 / 0.19 s; 3 steps 90.0 / 0.29 s; 5 steps 90.3 / 0.48 s; 10 steps 90.2 / 0.94 s V. The Janus AR baseline takes 0.27 s V. [WAM-Flow](https://arxiv.org/html/2512.06112v1)
  - Each extra step costs about (0.48 − 0.09)/4 ≈ 0.10 s on the full 1.5B model **D**.
  - SFT-only 86.7 → GRPO 90.3. NAVSIM v2 EPDMS 84.7 V. [WAM-Flow](https://arxiv.org/html/2512.06112v1)
- **DriveFine.**
  - LaViDa with LLaDA-8B. Diffusion is in the LLM V. [DriveFine](https://arxiv.org/html/2602.14577)
  - Decoding: 12 sampling steps plus 1 refining step, confidence-prioritized cosine schedule V. [DriveFine](https://arxiv.org/html/2602.14577)
  - 4 steps give 90.47 PDMS at about 207 ms average latency. Latency grows roughly in proportion to steps V. [DriveFine](https://arxiv.org/html/2602.14577)
  - Table 4: SFT 86.7 → GRPO 89.6 → offline RFT 90.3 → online RFT 90.8. Table 1 has 90.7, 91.8 with score-based RFT, and 94.2 for best-of-6 V. The paper's text and tables disagree (91.9 vs 91.8; 90.8 vs 90.7). [DriveFine](https://arxiv.org/html/2602.14577)
  - Other benchmarks: NAVSIM v2 EPDMS 87.1 (89.7 after the bug fix). Navhard EPDMS 74.4 in stage 1 and 41.0 in stage 2 V. [DriveFine](https://arxiv.org/html/2602.14577)
- **ReflectDrive** (ICLR 2026).
  - Initialized from LLaDA-V. Uses per-axis 1D codebooks (x and y quantized independently) over range ±100 m. Bin size and waypoint count are not stated V. [ReflectDrive](https://arxiv.org/html/2509.20109v1)
  - Decoding: 5 inference steps with low-confidence remasking. The step sweep peaks at 5 and then declines (figure only) V. [ReflectDrive](https://arxiv.org/html/2509.20109v1)
  - PDMS: 84.8 without reflective inference → 91.1 with it, no RL. Goal-conditioned only 87.4; safety-guided only 90.3 V. [ReflectDrive](https://arxiv.org/html/2509.20109v1)
  - Reflection uses gradient-free local search in a Manhattan token neighbourhood (example δ ≤ 10) plus inpainting, with at most 10 iterations; most plans converge in 1–3. No latency is measured V. [ReflectDrive](https://arxiv.org/html/2509.20109v1)
- **ReflectDrive-2** (arXiv 2605.04647; listed at NeurIPS 2026).
  - Backbone: a 0.7B masked-diffusion language model plus a 0.1B ViT, both fully fine-tuned V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
  - Plan: one longitudinal and one lateral token per waypoint, so L = 16 for 8 waypoints. The final (x8,y8) pair is the goal token. Heading is not tokenized. Precision is bounded by "the coordinate-bin size", which is not reported V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
  - Table 3 (without → with inference AutoEdit): DLM 84.8 → 85.0; +DACF 87.2 → 87.3; +AutoEdit training 87.7 → 88.0; +RL 89.1 → 91.0 V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
  - Step budget: full-step frames run 1+3+3 steps (decision, draft, reflect); lite-step frames run 1+1 (draft, reflect). The prefix KV is computed once and shared across all phases V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
- **MD-Drive** ("Towards Robust Vision-Language Reasoning…", arXiv 2611.15238).
  - DreamVL diffusion-LM VLM. Each waypoint is an (x,y) token pair; heading/steering is also quantized V. [MD-Drive](https://arxiv.org/html/2611.15238)
  - Ranges: x [−5, 70] m and y [−20, 25] m at Δg = 0.25 m; steering [−2, 2] at 0.04 V. [MD-Drive](https://arxiv.org/html/2611.15238)
  - Vocabulary check: 75/0.25 + 45/0.25 + 4/0.04 = 300 + 180 + 100 = 580, which matches the paper's "+580" **D**.
  - Inference: 32 steps. Results: SFT+DAD 80.3 → GRPO 88.9 (NC 97.1, DAC 94.0, TTC 92.2, C 99.2, EP 90.3) V. [MD-Drive](https://arxiv.org/html/2611.15238)
  - Speed: 243.9 tokens/s and 13.35 Hz, with "a 2.0-point drop". The hardware and the reference for the drop are not stated V. [MD-Drive](https://arxiv.org/html/2611.15238)
- **WAM-Diff2** (arXiv 2608.01035v4).
  - Converts an AR Qwen3-VL driving VLA into block diffusion by distillation. Waypoints go through the unified text tokenizer V. [WAM-Diff2](https://arxiv.org/html/2608.01035v4)
  - PDMS: AR B=1 88.1; B=32 88.3 (no RL, no candidate selection); 91.1 with score-based candidate selection. NAVSIM v2 EPDMS 88.6 / 90.7 V. [WAM-Diff2](https://arxiv.org/html/2608.01035v4)
  - Direct AR→diffusion adaptation drops to 84.1 PDMS; block-wise and model-wise distillation recover it to 88.3 V. [WAM-Diff2](https://arxiv.org/html/2608.01035v4)
  - Decoding latency falls from 22.7 to 1.5 ms/token, 15.1× overall, of which 2.8× comes from the switch to diffusion V. [WAM-Diff2](https://arxiv.org/pdf/2608.01035)
  - PDMS saturates at about 8–16 denoising steps (Fig. 3) V. [WAM-Diff2](https://arxiv.org/pdf/2608.01035)
- **Discrete-WAM** (Xiaomi EV, arXiv 2606.05645v2).
  - About 1B parameters: decoder-only, 18 layers, d = 2048, GQA with 8 KV heads; LoRA adapters about 30M V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
  - Action tokens: ego-frame accelerations after cubic-spline fitting. Each axis is clipped to [−4, 4] m/s² and split into 60 bins, giving a 60×60 = 3,600-entry joint grid V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
  - Targets are soft labels over the 4 neighbouring prototypes (bilinear interpolation). An auxiliary factorized x/y position classifier uses 0.02 m bins (3,350 × 2,500) V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
  - NAVSIM v1 PDMS 92.2 (NC 98.8, DAC 98.4, TTC 95.3, C 100, EP 88.7). NAVSIM v2: SFT 89.1 → SFT with decision 90.0 → RL 90.4 EPDMS V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
  - No SFT-only v1 number is given. Latency was measured on H20 GPUs against an AR decoder without KV cache; the results are only in a figure V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
- **No NAVSIM numbers:**
  - MVLAD-AD: LLaDA-initialized, about 7B; k-means joint (x,y) codebook N = 256; nuScenes average L2 1.28 m; 1.72 s per inference on an A100 V. [MVLAD-AD](https://arxiv.org/html/2602.20577)
  - dVLM-AD: LLaDA-V / LLaDA-8B with JSON-text waypoints; reports nuScenes L2 and WOD-E2E RFS 7.633 V. [dVLM-AD](https://arxiv.org/html/2512.04459)
- **Excluded:** DriveWAM (arXiv 2605.28544) is a continuous flow-matching video-action model, not discrete [DriveWAM](https://arxiv.org/abs/2605.28544).

### Inferences
- With SFT only, discrete planners land at 84.8–86.7 PDMS: WAM-Diff 86.6, WAM-Flow 86.7, DriveFine 86.7, ReflectDrive 84.8, ReflectDrive-2 84.8 (87.7 with its extra training losses). The 8B models are in the same band as our 0.8B text-waypoint model at 85.6, so the discrete formulation itself is not what buys the 90+ numbers.
- RL / RFT gains on NAVSIM v1 **D**:
  - WAM-Diff 91.0 − 86.6 = +4.4
  - WAM-Flow 90.3 − 86.7 = +3.6
  - DriveFine 90.7 − 86.7 = +4.0
  - ReflectDrive-2 89.1 − 87.7 = +1.4 without inference edit, 91.0 − 88.0 = +3.0 with it
  - MD-Drive 88.9 − 80.3 = +8.6
  - AutoVLA (AR) 89.11 − 80.54 = +8.57
- Search and self-correction on top of the model are worth about +3 to +6 (ReflectDrive 84.8 → 91.1 with no RL).
- Token design that has converged for 4 s / 8-waypoint plans: per-axis bins, x and y tokens per waypoint, 16 tokens total. Waypoint (absolute) targets were slightly better than displacements in MVLAD-AD (L2 1.28 vs 1.30). Heading tokens are optional; ReflectDrive-2 derives heading from the waypoints.
- Decoding order: generate the goal or far waypoint first. Evidence: WAM-Diff reverse-causal 91.0 vs causal 88.9; MD-Drive R-causal 86.5 vs causal 81.4; ReflectDrive and ReflectDrive-2 both decode a goal-token stage first.

### Gaps
- Not reported: ReflectDrive-2 bin size and vocabulary, ReflectDrive's bin resolution and waypoint count, and the token count per plan for WAM-Diff and DriveFine.
- I did not check the official NAVSIM leaderboard pages, so all numbers are paper self-reports on navtest. None of these papers states that it used the full official navtest with NAVSIM's own scorer in a way I verified.
- I found no other LLaDA, LaViDa or Dream driving VLA with NAVSIM v1 numbers beyond those listed.

## Q2. Is the masked diffusion inside the LLM, or in a separate small head reading cached VLM states?

### Takeaway
Every driving masked-diffusion planner found runs the denoising through the whole (diffusion) language model. For each step, the action tokens pass through all layers, with only the prefix KV cached. I found **no driving example of a separate small discrete-diffusion head on a standard autoregressive VLM**. Robotics has two clear examples:
- **TS-Mask VLA**: Qwen2.5-0.5B plus a separate discrete-diffusion action expert reading hidden states from all VLM layers.
- **DiscreteRTC**: Qwen2.5-VL-3B AR VLM plus a 36-block cross-attention DiT head, with discrete and flow-matching variants on the same head.

### Cited Findings
- **Driving (all in-LLM):**
  - WAM-Diff: the masked decoding "is the LLM itself" (LLaDA-V, 8.4B) V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
  - WAM-Flow converts Janus-1.5B to a non-causal model; its text head predicts the action tokens V. [WAM-Flow](https://arxiv.org/html/2512.06112v1)
  - DriveFine: the generation expert is the full LLaDA-8B. Its "refinement expert" is a replica of the last 4 of 32 blocks. Refine-block count vs PDMS: n=0 89.6, n=1 90.0 (+0.25B params), n=4 90.8 V. [DriveFine](https://arxiv.org/html/2602.14577)
  - DriveFine argues that bolting a separate diffusion planner onto a VLM gives "weak coupling", leading to reward hacking under PDMS RL. The size of the drop is shown only in a figure V. [DriveFine](https://arxiv.org/html/2602.14577)
- **ReflectDrive-2, the closest driving case to a head:**
  - Action tokens run through all blocks of the 0.7B dLM, but each block has a compact action FFN with hidden size cut from 4096 to 1024. The visual, instruction and ego prefix KV is cached and shared across the goal, draft and AutoEdit phases V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
  - Shrinking the action FFN trades accuracy on an in-house metric: minSADE goes 0.858 → 0.914 (worse) and meanSADE 1.710 → 1.539 (better) V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
- **WAM-Diff2** keeps one decoder-only backbone with a block-causal mask and "entirely remov[es] task-specific projection heads" V. [WAM-Diff2](https://arxiv.org/html/2608.01035v4)
- **Discrete-WAM** uses multi-task prediction heads on a 1B decoder-only backbone. Action tokens are edited by the whole backbone, with causal attention over context and bidirectional attention within the action block V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
- **Robotics, separate head (TS-Mask VLA, arXiv 2607.09818):**
  - The "Discrete Diffusion Action Expert" is N stacked Bridge-Attention + FFN blocks reading hidden states from all L layers of a LoRA-tuned Qwen2.5-0.5B. Block count and parameter count are not reported V. [TS-Mask VLA](https://arxiv.org/html/2607.09818)
  - Tokens: 256 uniform bins per dimension over [−1, 1]; confidence remasking on a cosine schedule V. [TS-Mask VLA](https://arxiv.org/html/2607.09818)
  - LIBERO average 95.7, vs VLA-Adapter-Pro (0.5B, reproduced) 92.2 and π0 94.2. CALVIN ABC→D average length 4.19 V. [TS-Mask VLA](https://arxiv.org/html/2607.09818)
  - No latency is reported and there is no in-LLM discrete-diffusion comparison V. [TS-Mask VLA](https://arxiv.org/html/2607.09818)
- **Robotics, separate head (DiscreteRTC, arXiv 2604.25050):**
  - Backbone: Qwen2.5-VL-3B, an AR VLM. Head: a layerwise cross-attention DiT with 36 blocks, inner dimension 2048, AdaLN, cross-attention to the matching VLM layer's hidden states V. [DiscreteRTC](https://arxiv.org/html/2604.25050)
  - The discrete and flow-matching variants share this DiT; only the input and output layers differ. Discrete variant: 256 bins per dimension, `nn.Embedding(257, 2048)` including [MASK], 8 MaskGIT steps on a cosine schedule V. [DiscreteRTC](https://arxiv.org/html/2604.25050)
  - On the real-robot Hockey Defend task, DiscreteRTC has 65% higher success than flow-matching RTC. It needs about 0.7× the compute of from-scratch generation, vs about 1.7× for flow-matching RTC V. [DiscreteRTC](https://arxiv.org/html/2604.25050)
- **E0** (PaliGemma plus a 300M action expert, width 1024, depth 18) encodes the observation once into a KV cache that is reused across denoising steps. Its noise is Gaussian on one-hot vectors, not masking V. [E0](https://arxiv.org/html/2511.21542v2)
- **Other in-backbone robotics models:** Discrete Diffusion VLA (OpenVLA 7B, in-backbone, shared 256-way head) V [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072); LLaDA-VLA (LLaDA) V [LLaDA-VLA](https://arxiv.org/html/2509.06932v2); dVLA (MMaDA backbone) R [dVLA](https://arxiv.org/html/2509.25681v1); UD-VLA (joint image + FAST tokens) R [UD-VLA](https://arxiv.org/html/2511.01718v1).
- **Searches run for a separate head:**
  - "discrete diffusion action head VLA 2026 arXiv masked"
  - "\"discrete diffusion\" action head lightweight separate VLM hidden states robot policy MaskGIT"
  - "masked discrete diffusion trajectory head autoregressive VLM driving planner separate decoder NAVSIM"
  - "diffusion language model driving VLA NAVSIM 2026 LLaDA trajectory tokens PDMS"
  - Every driving hit was in-LLM. A search summary of MD-Drive's related work notes that the separate-head pattern in driving is "AR VLM + continuous diffusion planner" R. [MD-Drive](https://arxiv.org/html/2611.15238)

### Inferences
- Our planned design (Qwen3.5-0.8B prefill once, then a small masked-diffusion head over 16 action tokens with AdaLN ego conditioning) has a robotics precedent: TS-Mask VLA's all-layer bridge attention and DiscreteRTC's layerwise cross-attention DiT with AdaLN. It has no driving/NAVSIM precedent, so a result would be a novel data point.
- DiscreteRTC is the only apples-to-apples discrete-vs-flow comparison on one head architecture. It is the model to copy for the planned "masked first, then continuous DiT" comparison.
- In ReflectDrive-2, each denoising step is a pass of 16 action tokens through every layer of a 0.7B model, with the prefix KV cached. A separate head with far fewer layers costs a fraction of that per step, so 4–8 steps and K plans can be batched in one pass over the cached VLM states. This is extrapolation, not measured.
- DriveFine's "weak coupling → reward hacking" argument is about continuous diffusion planners attached to VLMs. It is the main published risk for a separate head under PDMS RL. Keeping the head conditioned on many VLM layers (as TS-Mask and DiscreteRTC do) is the obvious mitigation, but this is untested in driving.

### Gaps
- TS-Mask VLA's expert size and latency are not reported. DiscreteRTC's head parameter count and absolute latency are only in figures.
- I found no driving paper that ablates "separate head" against "in-LLM masked diffusion".

## Q3. What does robotics evidence say about bin count, embeddings, soft labels, unmasking order and step count?

### Takeaway
Robotics discrete-diffusion VLAs mostly use 256 bins per dimension (E0 uses 2,048). They decode in about 8–12 steps with confidence-ranked unmasking. Confidence order beats parallel or random by about 1 point on LIBERO-Goal. Decaying-temperature Gumbel sampling beats argmax. Label smoothing helps modestly (E0: α = 0.1 gives about +1.4%). I found no VLA that ablates Gaussian or ordinal neighbour-bin labels. The closest soft-label design is driving-side: Discrete-WAM's bilinear 4-neighbour soft labels, which have no isolating ablation.

### Cited Findings
- **Discrete Diffusion VLA** (OpenVLA 7B).
  - Tokens: 256 quantile bins per dimension, 7 tokens per timestep, chunk H = 8 (56 tokens). Decoding: T = 12 rounds, cosine mask schedule, Gumbel resampling with decaying temperature, secondary re-masking threshold η = 0.5·(1 − t/T) V. [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)
  - Order on LIBERO-Goal (Table 7): parallel 95.6, random 95.8, confidence gap 96.6, max confidence 96.8 V. [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)
  - Temperature (Table 8): argmax 96.2, fixed T = 1 96.4, linear decay 96.8 V. [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)
  - Latency on an H800 (Table 6): 68.8 ms at 12 NFE vs OpenVLA AR 136.2 ms at 56 NFE and OpenVLA-OFT parallel 31.1 ms at 1 NFE. Re-masking adds under 1 ms V. [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)
  - LIBERO average 96.4 vs OpenVLA-OFT (L1, continuous) 97.1 and OpenVLA-OFT (discrete) 95.5 V. [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)
- **LLaDA-VLA.**
  - 32 action special tokens (V_a = 32, inferred from "32 additional special tokens"). "Localized special-token classification" restricts the softmax to action tokens: CALVIN average length 2.64 → 3.43 V. [LLaDA-VLA](https://arxiv.org/html/2509.06932v2)
  - Hierarchical action-level then token-level remasking: 3.43 → 4.01, using 10 steps V. [LLaDA-VLA](https://arxiv.org/html/2509.06932v2)
  - Chunk length (Table 6): K=3 3.90, K=5 4.01, K=8 3.53, K=10 3.36 V. [LLaDA-VLA](https://arxiv.org/html/2509.06932v2)
- **TS-Mask VLA.** 2D (temporal, then spatial) masking vs 1D: LIBERO-Long 85.0 → 91.6 V. Step-unroll loss: none 90.8, λ = 0.5 95.4, λ = 1.0 91.5 on LIBERO-Spatial V. [TS-Mask VLA](https://arxiv.org/html/2607.09818)
- **E0.**
  - Quantile bins, 2,048 bins in experiments, Gaussian noise on smoothed one-hot vectors (α = 0.1), N = 10 refinement steps, cross-entropy loss V. [E0](https://arxiv.org/html/2511.21542v2)
  - One-hot smoothing α = 0.1 gives about +1.4% success V. The paper argues masking causes "distributional mismatch" but gives no numeric mask-vs-Gaussian comparison V. [E0](https://arxiv.org/html/2511.21542v2)
  - Note: the paper's text about τ bias contradicts its Eq. 4; check the code before reimplementing V. [E0](https://arxiv.org/html/2511.21542v2)
- **DiscreteRTC.**
  - Real robot: 256 bins, 8 MaskGIT steps, decode temperature 0.0, choice temperature 0.1, CE plus auxiliary L1 (λ = 0.1). Kinetix: 512 bins, 5 steps V. [DiscreteRTC](https://arxiv.org/html/2604.25050)
  - Max-confidence decoding "usually uses all 8 steps" and does not give an AR-like order V. [DiscreteRTC](https://arxiv.org/html/2604.25050)
- **FAST / π0-FAST.**
  - Naive 256-bin tokenization gives 35 tokens per second at 5 Hz but 700 at 50 Hz. Naive-bin policies "are unable to make progress" at 20 and 50 Hz V. [FAST](https://arxiv.org/html/2501.09747)
  - π0-FAST takes about 750 ms per chunk vs about 100 ms for π0 on a 4090, and trains with 5× fewer GPU hours V. [FAST](https://arxiv.org/html/2501.09747)
  - On driving, AutoVLA's NAVSIM ablation: FAST (DCT) tokens 67.63 PDMS vs K-disk physical tokens 80.54 V. [AutoVLA](https://arxiv.org/html/2506.13757)
- **VQ-VLA.** Residual VQ-VAE with 4 codebook groups × 256 entries, embedding dimension 128, k-means init. The authors claim tokenizer quality scales roughly linearly with simulated action data R. I found no residual-vs-single-VQ ablation. [VQ-VLA](https://arxiv.org/html/2507.01016v1)
- **Fast-dVLA.**
  - Accelerates Dream-VLA, DD-VLA and UD-VLA with block diffusion and KV reuse: DD-VLA 152.1 → 402.7 tokens/s, LIBERO 0.963 → 0.966 V. [Fast-dVLA](https://arxiv.org/html/2603.25661)
  - Dream-VLA shows "a strong left-to-right decoding pattern at a global level". Lowering the confidence threshold causes "an approximately linear drop in performance" V. [Fast-dVLA](https://arxiv.org/html/2603.25661)
- **Embeddings (driving).**
  - MVLAD-AD: random embeddings raise L2 from 1.28 to 2.39 m vs geometry-aware ones V. [MVLAD-AD](https://arxiv.org/html/2602.20577)
  - WAM-Flow's metric-aligned embedding is a linear map R→R^d from the scalar value, L2-normalized, trained with a triplet-margin loss (α = 0.05) V. [WAM-Flow](https://arxiv.org/html/2512.06112v1)
- **Steps (driving).**
  - ReflectDrive peaks at 5 steps and then declines V [ReflectDrive](https://arxiv.org/html/2509.20109v1). WAM-Diff2 saturates at 8–16 V [WAM-Diff2](https://arxiv.org/pdf/2608.01035). WAM-Flow goes 89.1 at 1 step → 90.3 at 5 V [WAM-Flow](https://arxiv.org/html/2512.06112v1).
  - Discrete-WAM: under full_replace, more rounds raise L2 (R3 − R1 at 4 s: +0.0084 m), while confidence_replace lowers it (−0.0040 m) V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
- **Background found only in search snippets** (not VLA-specific): Gaussian-integrated soft bin targets beat one-hot CE in tabular forecasting [Tabular tokenization](https://arxiv.org/pdf/2603.07448), and a "Gaussian Cross Entropy" over numeric tokens for LLM number prediction [Smooth MMD alignment](https://arxiv.org/pdf/2606.27731), both R.

### Inferences
- For 16 tokens at 2 Hz, FAST/BPE compression buys nothing: the high-frequency failure it fixes does not arise. AutoVLA's NAVSIM result (FAST 67.63 vs physical tokens 80.54) argues against DCT tokens for driving.
- Defaults the evidence supports for our head:
  - restrict the softmax to the action vocabulary (LLaDA-VLA's LSC); trivially true for a separate head
  - use metric-aware embeddings: WAM-Flow +2.3 PDMS, and MVLAD-AD's random-embedding L2 nearly doubles
  - decode in 4–8 confidence-ranked steps with goal tokens first
  - use a decaying-temperature Gumbel choice
  - commit tokens selectively; do not fully replace every round
- Gaussian or ordinal soft labels over neighbouring bins are an open, cheap ablation. The closest evidence is E0's uniform smoothing (+1.4%) and Discrete-WAM's bilinear soft labels (no isolated number).

### Gaps
- No primary-source ablation of Gaussian vs one-hot labels for action tokens was found.
- E0's per-bin-count numbers are only in a figure.
- No number of steps was ablated in LLaDA-VLA or TS-Mask VLA.
- I did not read the Masked Generative Policy (ICLR 2026) paper; only its existence surfaced in a search [MGP note](https://en.papernotes.org/ICLR2026/robotics/masked_generative_policy_for_robotic_control/).

## Q4. How continuous should the discrete head be: residual offsets, quantization floors, and whether finer bins keep helping?

### Takeaway
The strongest new evidence is MD-Drive's NAVSIM granularity sweep: 0.5 m bins 78.2, 0.25 m 81.4, 0.1 m 80.8 PDMS, SFT-only. Finer bins stopped helping at about 0.25 m. The best-performing "continuous" designs avoid hard rounding altogether:
- Discrete-WAM: bilinear soft labels, expectation decoding with mode selection, plus a 0.02 m auxiliary position classifier and a trajectory regression loss.
- WAM-Flow: metric-induced corruption path plus metric-aligned embeddings.
- DiscreteRTC: CE plus auxiliary L1.

Classic offset heads (BeT, VQ-BeT) are called "quite important", but no numbers isolate their effect.

### Cited Findings
- **MD-Drive Table 8** (NAVSIM, SFT only, no RL):

  | Δg | Vocab added | Quant. error (m) | PDMS |
  |---|---|---|---|
  | no discretization | – | – | 68.5 |
  | 0.5 m | +340 | 0.25 | 78.2 |
  | 0.25 m | +580 | 0.125 | 81.4 |
  | 0.1 m | +1300 | 0.05 | 80.8 |

  The text calls 0.25 m "the best trade-off between quantization precision and vocabulary size" V. [MD-Drive](https://arxiv.org/html/2611.15238)
- **Quantization floor arithmetic D.** With uniform bins of width Δ and hard rounding, max error per axis is Δ/2 and RMS is Δ/√12 ≈ 0.289Δ. The 2D RMS Euclidean error is Δ/√6 ≈ 0.408Δ.
  - Δ = 0.25 m (MD-Drive best): 2D RMS ≈ 0.10 m.
  - Δ = 0.05 m (DriveFine): ≈ 0.020 m.
  - Δ = 0.01 m (WAM-Diff): ≈ 0.004 m.
- **AutoVLA K-disk codebook** (AR, 0.5 s motion tokens; Table 4):

  | K | ADE / FDE (m) | Usage |
  |---|---|---|
  | 256 | 0.0687 / 0.1034 | – |
  | 1024 | 0.0253 / 0.0282 | – |
  | 2048 | 0.0182 / 0.0203 | 100% |
  | 4096 | 0.0141 / 0.0155 | 91.46% |

  Physical action tokens vs text waypoints on NAVSIM: 80.54 vs 71.31 PDMS V. [AutoVLA](https://arxiv.org/html/2506.13757)
- **MVLAD-AD codebook size** (nuScenes L2): N = 128 1.73 m, 256 1.28 m, 384 2.76 m. The larger codebook was worse, with higher final training loss (0.53 vs 0.36) V. [MVLAD-AD](https://arxiv.org/html/2602.20577)
- **Discrete-WAM, a hybrid design.**
  - Training targets: soft labels over the 4 neighbouring acceleration prototypes by bilinear interpolation, which removes "deterministic hard-assignment quantization error"; remaining error comes only from distribution mismatch (derivation in the appendix) V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
  - Losses on decoded output: a decoded-acceleration loss, a trajectory regression loss after double integration, and a factorized x/y position classifier at 0.02 m V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
  - "Mode-aware decoding": fit a 1–3 component GMM to the categorical, pick one mode by top-p, then take the expectation within that mode. This avoids "decoding-induced mode averaging" V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
  - Result: v1 92.2 PDMS, but no ablation isolates the soft labels V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
- **Why Discrete-WAM needs those losses D.** Its bin width is 8/60 = 0.133 m/s², so the hard-rounding max error is 0.067 m/s². A constant acceleration error of that size integrates to 0.5 × 0.067 × 4² ≈ 0.53 m position error at 4 s. Acceleration-space tokens amplify quantization error, which is why the paper adds the soft labels and position losses.
- **WAM-Flow** corrupts along p_t(x|x1) = softmax(−β_t·d(x, x1)). Intermediate states therefore put mass on metrically near tokens, i.e. a built-in soft neighbour target V. Together with metric-aligned embeddings this gave +2.3 PDMS (81.1 → 83.4, already known) and 89.1 PDMS at a single step after RL V. [WAM-Flow](https://arxiv.org/html/2512.06112v1)
- **DiscreteRTC** trains the discrete head with CE plus auxiliary L1 (λ = 0.1) V. [DiscreteRTC](https://arxiv.org/html/2604.25050)
- **VQ-BeT** says "the offset prediction is quite important" but gives no numbers in the text (Fig. 5) V. VQ-BeT (residual VQ + offset) vs BeT (k-means bins + offset): PushT 0.78 vs 0.39, Kitchen 3.66 vs 3.07. Inference 15.1 ms vs about 100 ms for Diffusion Policy V. [VQ-BeT](https://arxiv.org/html/2403.03181)
- **E0**: precision improves up to about 2,048 bins, then saturates R (brief; not re-derived). [E0 project page](https://doo-mon.github.io/e0web/)
- **ReflectDrive-2** states that precision is bounded by the coordinate-bin size. It compensates with token-level AutoEdit plus perturbation augmentation, not a regression residual V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)

### Inferences
- In a separate head, vocabulary size is cheap. Unlike in an LLM, it does not grow the shared embedding or LM-head matrix. MD-Drive's saturation at 0.25 m may therefore partly reflect LLM vocabulary-extension costs. The safest reading is that hard bins finer than about 0.1–0.25 m are not where PDMS comes from.
- Suggested middle ground, which needs our own ablation:
  - per-axis bins of about 0.05–0.1 m over a data-fitted range
  - Gaussian or bilinear soft labels over neighbours
  - metric (sinusoidal or linear-in-value) token embeddings
  - optional decoding by expectation within the argmax neighbourhood, Discrete-WAM style, or a tiny per-token offset regression trained with L1
- An expectation readout gives sub-bin precision without a separate regression branch. It must be local (within the chosen mode) to avoid averaging two plan modes; Discrete-WAM names this failure explicitly.
- For RL on PDMS, the policy-gradient likelihood should stay on the categorical token. A continuous offset, if added, can stay deterministic so the RL likelihood stays clean. This is my design suggestion, not reported by any paper.

### Gaps
- No driving paper isolates "token + continuous residual" against "token only".
- VQ-BeT's offset ablation and E0's bin curve exist only as figures.
- I found no study of soft-label width (Gaussian σ) for trajectory tokens.

## Q5. How do masked-diffusion planners produce K diverse plans, and is low diversity or mode collapse reported?

### Takeaway
Driving masked-diffusion planners get diversity mainly by **sampling the goal or terminal token first and pruning with BEV NMS**:
- ReflectDrive: K = 3 goals, NMS 0.9 m.
- ReflectDrive-2: top-k goals, NMS about 1.2 m; best-of-6 = 3 goals × 2 drafts.

They then fill the rest by conditional inpainting. Oracle best-of-6 reaches 94.2–94.8 PDMS, so diversity is sufficient for selection. No paper reports outright mode collapse. Two diversity-shrinking factors are documented:
- longer SFT: MD-Drive's pass@8 falls while pass@1 rises
- low-precision Gumbel sampling, which lowers the effective temperature

### Cited Findings
- **ReflectDrive.** Samples goal candidates from the terminal-waypoint token distribution, keeps K = 3 after NMS at 0.9 m, inpaints each, and picks the best by a scorer V. [ReflectDrive](https://arxiv.org/html/2509.20109v1)
- **ReflectDrive-2.** Goals by top-k sampling plus BEV NMS; a threshold of about 1.2 m is reported as optimal. RL group G = N_g × I (3 goals × 2 drafts). Best-of-6 oracle 94.8 PDMS, matching the human reference of 94.8. No diversity metric and no temperature are reported V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
- **DriveFine.** Best-of-6: 94.2 PDMS. GRPO group size G vs PDMS: 0 86.7, 2 88.4, 4 89.1, 6 89.4, 8 89.6, 10 89.6 V. [DriveFine](https://arxiv.org/html/2602.14577)
- **WAM-Diff** GSPO G = 2 88.9, G = 3 91.0 V [WAM-Diff](https://arxiv.org/html/2512.11872v1). **WAM-Flow** GRPO G = 2 89.2, G = 3 90.3, G = 4 89.6 V [WAM-Flow](https://arxiv.org/html/2512.06112v1).
- **MD-Drive pass@1 / pass@8** across SFT checkpoints:

  | Checkpoint | pass@1 | pass@8 |
  |---|---|---|
  | 2659 | 61.18% | 94.75% |
  | 5318 | 69.42% | 96.72% |
  | 7977 | 71.13% | 96.42% |
  | 10636 | 71.56% | 95.99% |

  The pass tolerance ε is not given V. [MD-Drive](https://arxiv.org/html/2611.15238)
- **MD-Drive decoding order.** The order (near-to-far or far-to-near) is chosen per sample by comparing summed confidence over the first and second halves of the sequence. DAD 88.9 vs confidence-only 84.3 V. [MD-Drive](https://arxiv.org/html/2611.15238)
- **Discrete-WAM decision anchors.** Performance peaks at k = 16 anchors and degrades beyond. Larger anchor sets "introduce more low-quality anchors, increasing reward variance and weakening the GRPO advantage signal" V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
- **AutoVLA (AR) best-of-6** oracle: 92.12 vs 89.11 single V. [AutoVLA](https://arxiv.org/html/2506.13757)
- **Gumbel precision.** Truncating Gumbel noise from 64-bit to 32-bit lowers the effective sampling temperature in masked diffusion models. Sentence entropy falls from 5.66 to 5.17 (with "better" perplexity, 126.11 → 31.24), so diversity is silently reduced R. [Zheng et al., ICLR 2025](https://arxiv.org/html/2409.02908v6)
- **Robotics sampling knobs.** Discrete Diffusion VLA uses Gumbel resampling with a decaying temperature (best 96.8 vs argmax 96.2) V [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072). DiscreteRTC runs nearly deterministic: decode temperature 0.0, choice temperature 0.1 V [DiscreteRTC](https://arxiv.org/html/2604.25050).

### Inferences
- For a learned selector over K plans, the published recipe is: sample about 3–6 goal tokens (top-k or temperature on the x8/y8 tokens), apply BEV NMS at about 1 m, then decode the rest conditioned on each goal (1–2 drafts each).
- Oracle best-of-6 is about 94–95 PDMS vs about 91 for a single plan. That gap of about 3–4 points is the headroom a selector can capture.
- MD-Drive's pass@8 peaking at an intermediate checkpoint, together with its overfitted checkpoint gaining less from RL (Q6), suggests driving SFT toward near-perfect train fit erodes the diversity that RL and selection need. Track pass@K or oracle best-of-K during SFT, not only top-1 PDMS.
- Implementation note: sample Gumbel noise in fp64, or use explicit categorical sampling, when generating K plans; fp32 truncation lowers the effective temperature.

### Gaps
- No paper reports a quantitative diversity metric (pairwise endpoint distance, coverage) for masked-diffusion driving plans.
- No direct masked-vs-continuous diffusion diversity comparison was found.
- Mode collapse under RL for masked heads is not reported either way.

## Q6. What training stability and data needs are reported for these heads?

### Takeaway
Masked-diffusion driving heads are trained with plain masked-token CE and look stable in reports, but quantitative stability evidence is thin. The useful signals are:
- Pretraining on large driving and VQA data adds about 2–3 PDMS before RL.
- Overfitted SFT checkpoints gain far less from RL (MD-Drive: 84.3 → 85.8 vs 80.3 → 88.9).
- Converting an AR model straight to diffusion loses about 4 PDMS unless distilled.
- A small-head robotics recipe (E0) trains in about 24 GPU-hours.

### Cited Findings
- **WAM-Diff data schedule:**
  - MoE warm-up: 0.2 epochs on 668K nuPlan samples.
  - Multi-task SFT: 1 epoch on 668K nuPlan + 800K VQA.
  - NAVSIM: 3 epochs on 103K samples.
  - GSPO: 2 epochs, learning rate 1e-5, batch size 1, on 4×8 Ascend 910B. GSPO is reported as steadier than GRPO (figure only) V. [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- **WAM-Flow data.** VQA/driving pretraining adds +3.3 PDMS at 3 epochs. 0.65M pretraining samples give +1.9 with the numeric tokenizer (+5.2 for text); 6.5M adds a further +1.4 (+2.6 for text). GRPO without pretraining: 86.9 V. [WAM-Flow](https://arxiv.org/html/2512.06112v1)
- **DriveFine.** SFT 12 epochs, batch 64, learning rate 4e-5. RFT 1 epoch, batch 16, learning rate 1e-6. GRPO advantage uses r − mean(r) without std normalization. Stability is claimed but not quantified V. [DriveFine](https://arxiv.org/html/2602.14577)
- **MD-Drive.**
  - Data: 153K SFT and 58K RL samples; LoRA rank 8 on 4 H800 V. [MD-Drive](https://arxiv.org/html/2611.15238)
  - Overfitting: the non-overfitted checkpoint goes 80.3 → 88.9 after GRPO; the overfitted one goes 84.3 → 85.8 V. [MD-Drive](https://arxiv.org/html/2611.15238)
  - Seed spread (8 seeds): 71.64 ± 0.30 and 77.20 ± 0.23 at two checkpoints V. [MD-Drive](https://arxiv.org/html/2611.15238)
- **WAM-Diff2** (AR→diffusion conversion): direct adaptation 84.1, + block-wise distillation 87.7, + model-wise distillation 88.3 (AR baseline 88.1). An AR teacher scores 79.9 vs 88.3 for a diffusion teacher V. [WAM-Diff2](https://arxiv.org/html/2608.01035v4)
- **Discrete-WAM** training strategies (NAVSIM v2 EPDMS): from scratch 89.8, full fine-tune 89.7, LoRA-SFT 90.0. Pretraining was 200k steps on full nuPlan, then 10 SFT epochs and 2 RL epochs, on 32 H20 V. [Discrete-WAM](https://arxiv.org/pdf/2606.05645)
- **ReflectDrive-2** RL follows the d2 discrete-diffusion policy gradient with group-relative advantages, clipping and KL. The reward is PDMS on the final post-edit plan. No ε, λ_KL or stability analysis is reported V. [ReflectDrive-2](https://arxiv.org/html/2605.04647)
- **Robotics compute:**
  - E0: 30k steps, batch 32, about 24 h on one GPU; AdamW, gradient clip 1.0, EMA 0.999, peak LR 5e-5 V. [E0](https://arxiv.org/html/2511.21542v2)
  - Discrete Diffusion VLA: 100k–300k steps, batch 32, 4× A800. It reports "highest score across seeds" in one place and a "3-seed mean" in another V. [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)
  - DiscreteRTC: discrete head trained with cosine LR decay (flow head constant LR). VLM learning rate 1e-5, action head 1e-4. VLM in bf16, head in fp32 V. [DiscreteRTC](https://arxiv.org/html/2604.25050)
  - TS-Mask VLA's step-unroll loss: too strong (λ = 1.0) drops 95.4 to 91.5 V. [TS-Mask VLA](https://arxiv.org/html/2607.09818)

### Inferences
- MD-Drive's overfitted-checkpoint result bears directly on the "fit train near perfectly without memorizing" goal. Choose the SFT checkpoint for RL by pass@K or oracle best-of-K on a validation split, not by train fit or top-1 PDMS.
- A separate head trained from scratch on cached VLM states is closest to Discrete-WAM's "from scratch ≈ fine-tune" finding and E0's modest compute. This hints that data needs are driven by the planning data (NAVSIM navtrain of about 100K samples, plus nuPlan pretraining), not by head capacity. Untested for our configuration.
- Keep the head in fp32 as DiscreteRTC does, and use a cosine LR schedule. These are cheap stability defaults taken from the only same-head discrete-vs-flow comparison.

### Gaps
- No paper reports loss-curve instability, divergence rates, or multi-seed PDMS variance for masked-diffusion planners on full navtest.
- RL hyperparameter details (KL, clip) for masked-diffusion RL in driving are mostly unreported.
- I did not read the d2 / diffu-GRPO papers that define the likelihood approximations used for masked-diffusion RL.
