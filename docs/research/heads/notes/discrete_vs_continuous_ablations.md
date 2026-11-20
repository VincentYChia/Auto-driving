# Discrete vs continuous action heads: controlled comparisons and hybrids (robotics and driving, as of Nov 2026)

*Tags: **V** = verified from the paper's own text or tables (arXiv HTML full text read in this pass); **R** = self-reported, abstract, project page or search snippet only; **D** = derived by me (arithmetic shown). Control levels: **[C1]** same backbone, same data, same budget, only the action head/representation changes; **[C2]** same paper and data, but backbone, budget, recipe or reproduction differs; **[X]** cross-paper. Already-known items (WAM-Flow 76.2→81.1→83.4; E0 ~2,048-bin saturation; ReCogDrive 84.1→86.5→90.8) are not re-derived here.*

## Q1. Robotics: what do same-paper ablations show for discrete vs continuous actions?

### Takeaway
At a fixed backbone, the biggest gain comes from dropping autoregressive (AR) token-by-token decoding for parallel, chunked decoding (+13 to +14 points on LIBERO). Switching discrete to continuous adds a further 0.7 to 5 points, or nothing once discrete tokens are decoded with masked diffusion (95.1 discrete diffusion vs 94.9 continuous diffusion, [C1]). Continuous heads win clearly when the alternative is naive per-step binning, AR decoding, a weak backbone or long horizons (Octo, RoboVLMs). With the same backbone, AR token VLAs (pi0-FAST) train 3 to 7.5 times faster than flow-matching VLAs but run about 7.5 times slower.

### Cited Findings
**OpenVLA-OFT (OpenVLA 7B, LoRA per LIBERO suite, same data) [C1 for the head swap]**
- Table I, LIBERO average success (Spatial/Object/Goal/Long): AR discrete OpenVLA 76.5 (84.7/88.4/79.2/53.7); parallel decoding + action chunking (PD&AC), still discrete, 90.2 (91.3/92.7/90.5/86.5); PD&AC + continuous diffusion (50 steps) 95.4 (96.9/98.1/95.5/91.1); PD&AC + continuous L1 regression 95.3 (96.2/98.3/96.2/90.7). **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
  - D: AR→PD&AC = 90.2−76.5 = +13.7. Discrete PD&AC→L1 = 95.3−90.2 = +5.1. →diffusion = 95.4−90.2 = +5.2. L1 vs diffusion = 0.1.
- The authors attribute the roughly 5-point continuous gain to "higher precision in the action predictions". They note that more bins "improves precision but reduces the frequency of individual tokens". They call L1 and diffusion "comparable", because "the high-capacity OpenVLA model can effectively model the multi-task action distribution even with simple L1 regression." **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
- Table II (A100): PD&AC discrete 108.8 Hz / 0.0735 s; L1 109.7 Hz / 0.0729 s; diffusion with 50 test steps 4.2 Hz / 1.907 s. Diffusion trained with 50 steps and tested with fewer, LIBERO-Long success: 10 steps 91.0 (19.3 Hz), 5 steps 90.0 (35.1 Hz), 2 steps 85.7 (80.3 Hz), 1 step **0.0** (109.4 Hz). **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
- Training: 50–150K gradient steps for non-diffusion heads vs 100–250K for diffusion. Diffusion "converge[s] slower". Limitations: L1 "may struggle to accurately model truly multimodal action distributions", while diffusion "may better capture such multimodality but risk overfitting to suboptimal modes." **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)

**Discrete Diffusion VLA (DD-VLA)**
- **Cleanest head ablation found [C1]** (Appendix F, Table 13). Qwen2.5-VL with no robot pretraining, all four LIBERO suites trained jointly for 30k steps, 256 bins. LIBERO average: AR 81.2; parallel decoding (discrete, one pass) 94.2; FAST tokens 91.1; continuous diffusion (GR00T-style) 94.9; discrete (masked) diffusion 95.1. **V** — [DD-VLA](https://arxiv.org/html/2508.20072)
  - D: AR→parallel = +13.0. Parallel→continuous diffusion = +0.7. Parallel→discrete diffusion = +0.9. Discrete vs continuous diffusion = +0.2. FAST vs parallel = −3.1.
- Main results (OpenVLA backbone, 256 quantile bins, chunk H=8, T=12 refinement rounds, cosine mask schedule). LIBERO average: DD-VLA 96.4; OFT-Discrete (reproduced) 95.5; OFT-Diffusion 95.6; OFT-L1 97.1. **V**. [C2]: training budgets are not matched (DD-VLA trains 300k steps on Goal/Long). OFT-L1's 97.1 is the full OFT model with wrist camera and proprioception (OFT paper). The OFT-Diffusion Goal cell is 96.2 here but 95.5 in the OFT paper. — [DD-VLA](https://arxiv.org/html/2508.20072); [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
- Latency (Table 6, H800): AR 136.2 ms (56 NFE); OFT parallel 31.1 ms (1 NFE); OFT diffusion at 12 steps 67.1 ms; DD-VLA at 12 steps 68.8 ms. At equal step count, discrete and continuous iterative heads cost the same. **V** — [DD-VLA](https://arxiv.org/html/2508.20072)
- Out-of-distribution drops on LIBERO-Goal (language augmentation / vision augmentation): OFT-Discrete −8.0/−22.6; OFT-Diffusion −2.4/−29.0; OFT-L1 −3.2/−23.2; DD-VLA −0.8/−20.4. **V**. [C2] — [DD-VLA](https://arxiv.org/html/2508.20072)
- Decoding ablation (LIBERO-Goal): parallel 95.6; random order 95.8; confidence gap 96.6; max confidence 96.8. Temperature: hard (T=0) 96.2; fixed T=1 96.4; linear decay 96.8. **V** — [DD-VLA](https://arxiv.org/html/2508.20072)

**RoboVLMs (CALVIN ABCD→D, one-step structure, same backbone and data) [C1]**
- Average length, discrete vs continuous: LLaVA 1.85 vs 2.37; Flamingo 1.22 vs 1.30; KosMos 0.55 vs 4.09; PaliGemma 0.44 vs 4.04. **V**. Authors: "continuous action spaces consistently outperform discrete ones, particularly as task horizons increase", attributed to precision and to compounding error over long horizons. **V** — [RoboVLMs](https://arxiv.org/html/2412.14058)
  - D: gaps are +0.52 (LLaVA), +0.08 (Flamingo), +3.54 (KosMos), +3.60 (PaliGemma).
- A separate policy head (continuous) beats interleaving actions in the VLM: KosMos 4.49, PaliGemma 4.42. Discrete variants of these structures were not tested. **V** — [RoboVLMs](https://arxiv.org/html/2412.14058)

**Octo (Octo-Small, WidowX, 4 tasks × 10 trials) [C1]**
- Diffusion head 83%; MSE regression 35%; discretized cross-entropy 18%. **V**. The MSE head gave "hedging" policies that move slowly. The discrete head was "decisive" but "lacked precision and often missed the grasp." **V** — [Octo](https://arxiv.org/html/2405.12213)

**FAST / pi0-FAST vs pi0 (same PaliGemma-3B base, same data) [C2]**
- Tokens per 1-second chunk, naive binning vs FAST: BridgeV2 (5 Hz) 35 vs 20; DROID (15 Hz) 105 vs 29; Bussing (20 Hz) 140 vs 28; Shirt fold (50 Hz, 14-D) 700 vs 53. **V**. Naive binning "unable to make progress" on the 20 Hz and 50 Hz tasks (numbers only in a figure: **R**). — [FAST](https://arxiv.org/html/2501.09747)
- pi0-FAST vs pi0: "comparable" on small datasets (LIBERO, T-shirt folding). FAST reaches high performance with "3x fewer training steps" on Table Bussing and needs "5x fewer GPU hours" in generalist training. At matched compute FAST "clearly outperforms". The diffusion pi0 "often ignores the language instructions" on DROID. (Claims **V** as text; the underlying numbers are figure-only, **R**.) — [FAST](https://arxiv.org/html/2501.09747)
- Latency: pi0 predicts 1 s chunks "within 100ms" on a 4090; pi0-FAST "approximately 750ms". **V**. D: 750/100 = 7.5×. — [FAST](https://arxiv.org/html/2501.09747)
- Disagreement on LIBERO: Physical Intelligence's own Knowledge Insulation (KI) table lists pi0 at 96.8/98.8/95.8/85.2 and pi0-FAST at 96.4/96.8/88.6/60.2. **V** — [KI](https://arxiv.org/html/2505.23705). D: averages 376.6/4 = 94.15 vs 342.0/4 = 85.5. The 25-point LIBERO-Long gap does not fit FAST's "comparable on LIBERO" (which is figure-only).

**Knowledge Insulation (pi0 architecture: FAST tokens train the VLM, flow expert with stop-gradient) [C2]**
- DROID task progress: KI 0.55±0.09; pi0 0.49±0.09; pi0-FAST 0.45±0.09. pi0 needs "7.5 times as many training steps" to reach KI's performance, while KI trains about as fast as pi0-FAST at ~20% extra compute per step. Control rate: pi0 10 Hz vs AR VLAs 1.3 Hz. "the gradients from the action expert degrade the pre-trained VLM backbone". Freezing the backbone gave 0% on shirt folding. **V** — [KI](https://arxiv.org/html/2505.23705)

**HybridVLA (one 7B LLM, AR and diffusion outputs from the same model) [C1 within model]**
- RLBench, 10 tasks: diffusion-only 0.66; AR-only 0.62; hybrid ensemble 0.74. Without the collaborative training recipe: 0.60 and 0.57. Diffusion wins 6 of 10 tasks ("precise manipulation"); AR wins 4 ("scene semantic reasoning"). Speed: diffusion-only 9.4 Hz; hybrid 6.1 Hz. **V** — [HybridVLA](https://arxiv.org/html/2503.10631)

**NORA-1.5 (Qwen2.5-VL-3B; one jointly trained model with FAST tokens and a flow-matching expert) [C1 within model]**
- SimplerEnv Visual Matching / Variant Aggregation, FAST output vs flow output. Zero-shot: 74.0/54.3 vs 76.9/59.7. Fine-tuned: 72.1/67.4 vs 77.9/70.7. **V**. D: flow minus FAST = +2.9/+5.4 zero-shot and +5.8/+3.3 fine-tuned. — [NORA-1.5](https://arxiv.org/html/2511.14659v1)
- On a 50K-frame real-robot set (vs 4M frames for SimplerEnv), "flow-matching-based action generation performs worse than autoregressive decoding." The authors' explanation, not tested: the flow expert "may underperform in low-data settings". **V** — [NORA-1.5](https://arxiv.org/html/2511.14659v1)

**Discrete-only head comparisons (AR vs discrete diffusion vs discrete flow) [C1]**
- DFM-VLA (UniVLA/Emu3-based, FAST+BPE vocabulary of 1,024, 16 steps), CALVIN ABCD→D average length. AR 4.18 / discrete diffusion 4.32 / discrete flow matching 4.42. With 10% of the data: 1.71 / 2.84 / 3.21. With 50%: 3.01 / 3.88 / 4.03. **V** — [DFM-VLA](https://arxiv.org/html/2603.26320v2)
- UD-VLA (Emu3-based), CALVIN: AR 4.18 (50.2 tok/s); Jacobi 4.16; independent diffusion 4.35; joint discrete diffusion (JD3P) 4.64 (219.3 tok/s, 4.3×). **V** — [UD-VLA](https://arxiv.org/html/2511.01718). D: DFM-VLA's AR row (4.18, speed 50.2) is identical to UD-VLA's, so the AR baseline is probably shared.
- VQ-VLA (OpenVLA): LIBERO-90 bin-token baseline 74.76; AR with 5 actions 66.53; VQ chunking 86.61. 4.16 → 11.84 Hz. **V** — [VQ-VLA](https://arxiv.org/html/2507.01016)
- MiniVLA (Qwen2.5-0.5B, ~1B total), LIBERO-90: OpenVLA-7B bins 62%; MiniVLA bins 61.4%; residual VQ with 1 action 62.4%; residual VQ with 8-step chunks 77%; +history 82%. 12.5 Hz vs 5 Hz (L40s). **V** — [MiniVLA blog](https://ai.stanford.edu/blog/minivla/)

**Continuous vs continuous, and discrete-diffusion models with no controlled continuous baseline**
- CogACT (Prismatic-derived 7B, DiT head on one cognition-token feature) vs OpenVLA tokens. SIMPLER Google Visual Matching 74.8 vs 34.3; WidowX 51.3 vs 4.2; real Realman 71.2 vs 12.1. **V**, [C2]: the VLM is not stated to be identical. Head ablation: MLP-3 (3M) 50.6; MLP-7 (89M) 52.5; DiT-S (13M) 58.5; DiT-B (89M) 62.5; DiT-L (308M) 64.8. "With the same number of parameters, transformers outperform MLPs." **V** — [CogACT](https://arxiv.org/html/2411.19650)
- E0 (PaliGemma + 300M expert, 2,048 bins, 10 steps; Gaussian noise on one-hot actions). LIBERO: E0 97.2/99.4/95.0/92.2 vs pi0 96.8/98.8/95.8/85.2 vs pi0-FAST 96.4/96.8/88.6/60.2. 14-environment average: E0 60.8, pi0 51.4, pi0.5 50.1, pi0-FAST 52.2. Real Franka average: E0 45.6, pi0 43.1, pi0-FAST 10.0. **V**, [C2]: the paper says baselines were retrained in "the same experimental setup", but its LIBERO pi0 and pi0-FAST cells equal the published openpi numbers. — [E0 v2](https://arxiv.org/html/2511.21542v2)
- LLaDA-VLA (CALVIN 4.01; SimplerEnv WidowX 55.5) and dVLA (FAST tokens on MMaDA, LIBERO 96.4) report no same-backbone AR or continuous baseline. Only internal ablations: LLaDA-VLA baseline 2.64 → +localized classification 3.43 → +hierarchical decoding 4.01; dVLA without CoT 89.8 vs with CoT 96.4. **V** — [LLaDA-VLA](https://arxiv.org/html/2509.06932); [dVLA](https://arxiv.org/html/2509.25681)
- GR00T N1 runs no discrete-vs-continuous comparison. Its policy is flow matching only, and its latent-action labels use the continuous pre-quantized VQ embedding. **V** — [GR00T N1](https://arxiv.org/html/2503.14734)

### Inferences
- In the controlled tables, AR decoding is the main liability, not discreteness itself (+13 to +14 points from parallel decoding). Discrete tokens decoded in parallel or by masked diffusion sit within about 1 point of continuous diffusion (DD-VLA Table 13), but about 5 points below continuous in OFT. The OFT gap may reflect one-pass parallel discrete decoding without iterative refinement.
- Continuous heads win big ([C1] Octo, RoboVLMs) when the discrete baseline is naive per-step binning with weak decoding, or the backbone/policy is small or not robot-pretrained.

### Gaps
- No [C1] paper compares masked discrete diffusion against a pi0-style flow-matching expert (DD-VLA uses a GR00T-style diffusion head).
- The parameter count of the Qwen2.5-VL used in DD-VLA Table 13 was not stated in the text extracted.
- FAST and KI give most success numbers only in figures.
- Dita, VLA-0 (text actions), X-VLA, BEAST and the original Diffusion Policy paper were not checked in this pass.

## Q2. Driving: controlled comparisons of discrete tokens vs continuous diffusion vs regression

### Takeaway
Only two driving papers swap the decoder with everything else held fixed, and they disagree at scale. DriveVLA-W0 finds continuous decoders ahead at NAVSIM scale (query-regression 88.4 > flow 87.2 > AR tokens 85.3 PDMS) but AR ahead at 70M frames. Alpamayo-R1 finds flow matching clearly ahead of AR tokens (closed loop 1.27 vs 0.59, comfort 97% vs 44%, decoding about 25× faster). Text waypoints are the weakest output everywhere they are ablated. Several masked discrete diffusion VLAs reach 86.6–87.7 PDMS SFT-only and 90–91 after RL, the same range as continuous-head RL VLAs, but no paper runs that comparison controlled.

### Cited Findings
**DriveVLA-W0 (Emu3-8B or Qwen2.5-VL-7B backbone, ~500M action expert, all experts initialized from the same pretrained VLA) [C1]**
- NAVSIM (103k frames) PDMS: query-based (one pass) 88.4; flow matching (10 steps) 87.2; AR 85.3. EP: 82.2 / 80.9 / 79.3. DAC: 96.2 / 95.3 / 93.6. **V**. D: query−AR = 3.1; flow−AR = 1.9; query−flow = 1.2.
- In-house (70M frames), ADE / collision: query 1.1248 / 0.0453; flow 1.0362 / 0.0398; AR 1.0069 / 0.0295. **V**
- Authors: at small scale, continuous experts' "higher precision" gives "an edge over the discrete autoregressive approach" (they cite quantization error). At scale, AR's "sample-efficient, teacher-forced training" wins. Query-based hits a "representational bottleneck", and flow matching is "too sample-inefficient to converge". **V**
- Latency: query ~74 ms; flow (10 steps) ~145 ms; AR ~95 ms on NAVSIM (5.6 tokens on average) and ~170 ms in-house (17.8 tokens). Results are single runs. **V**
- Source for this group: [DriveVLA-W0](https://arxiv.org/html/2510.12796)

**Alpamayo-R1 (Cosmos-Reason; Table 12) [C1 for the decoder]**
- AR with a VQGAN-pretrained discrete trajectory tokenizer vs flow matching (5 steps). minADE6@6.4s 0.6811 vs 0.6440; AlpaSim score 0.59±0.17 vs 1.27±0.34; comfort 44.05% vs 97.38%. **V** — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
- Runtime (RTX 6000 Pro Blackwell): flow decoding 8.75 ms (5 steps) vs AR trajectory 222 ms (127 tokens). End-to-end 99 ms vs 312 ms. **V**. D: 222/8.75 ≈ 25×. — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
- The paper is inconsistent: the text says AR predicts "6 discrete trajectory tokens", Table 14 says 127 tokens, and Table 12 gives a relative decode speed of only 1.16×. Discrete tokens (128 per trajectory: acceleration and curvature bins) are still used during training. **V** — [Alpamayo-R1](https://arxiv.org/html/2511.00088)

**AutoVLA (Qwen2.5-VL-3B) [C1 for output format]**
- Text waypoints vs physical action tokens (K=2048 K-disk codebook, 10 tokens per 5 s): PDMS 71.31 vs 80.54; L2 0.89 vs 0.70 m; runtime 7.65 vs 3.95 s. **V**. D: +9.23 PDMS, about 1.9× faster.
- K-disk vs FAST tokens on NAVSIM: 80.54 vs 67.63 PDMS. Reconstruction ADE at K=2048: K-disk 0.0182 m; FAST 0.0281 m; RT-1-style bins 0.1014 m. **V**
- PDMS: SFT ("one-shot") 80.54 → RFT 89.11 (D: +8.57); best-of-6 oracle 92.12. **V**
- Source for this group: [AutoVLA](https://arxiv.org/html/2506.13757)

**ORION (Vicuna-1.5 + EVA-02-L, Bench2Drive closed loop) [C1 for the planner head]**
- Driving score / success rate: plain text output 42.23 / 13.14%; diffusion planner (K-means anchors, 20 modes) 71.97 / 46.54%; VAE planner 77.74 / 54.62%. **V**
- MLP regression head: the text gives only ORION's margin over it (+7.01 DS, +9.5% SR). D: MLP ≈ 77.74−7.01 = 70.73 DS and 54.62−9.5 = 45.12% SR.
- Authors: text fails from "inadequate numerical reasoning". The MLP "struggles with handling multi-modal trajectory". VAE training is "inherently more stable" than diffusion. **V**
- Source for this group: [ORION](https://arxiv.org/html/2503.19755)

**DiffusionDrive (Transfuser perception, NAVSIM navtest, RTX 4090) [C1 roadmap]**
- PDMS: Transfuser MLP regression 84.0 (1 step, mode diversity D=0%, 60 FPS); vanilla diffusion 84.6 (20 steps, D=11%, 7 FPS); truncated diffusion 85.7 (2 steps, D=70%); DiffusionDrive 88.1 (2 steps, D=74%, 45 FPS). **V** — [DiffusionDrive](https://arxiv.org/html/2411.15139)
  - D: vanilla diffusion over regression = +0.6. Anchors + truncation over vanilla diffusion = +3.5.
- Ablations: 1/2/3 steps give 87.9/88.1/88.1. Samples at inference (N_infer) 10/20/40 give 84.9/88.1/88.2. Vanilla diffusion: "different random noises converge to similar trajectories". **V** — [DiffusionDrive](https://arxiv.org/html/2411.15139)

**Vocabulary scoring (Hydra-MDP, same paper) [C2]**
- Navtest score: Transfuser 78.0; VADv2 imitation-only with a 4096/8192 vocabulary 79.7/80.9; Hydra-MDP 4096 / 8192 at 82.6 / 83.0; -W 85.7; -W-EP 86.5. Selection is an argmin over the vocabulary, with no regression offset added. **V** — [Hydra-MDP](https://arxiv.org/html/2406.06978)
- Hydra-MDP lists Transfuser at 78.0 while DiffusionDrive lists it at 84.0 (likely different NAVSIM or metric versions), so absolute values do not carry across these papers. **V** (numbers) — [Hydra-MDP](https://arxiv.org/html/2406.06978); [DiffusionDrive](https://arxiv.org/html/2411.15139)

**Masked discrete diffusion / discrete flow driving VLAs (none controlled against a continuous head)**
- WAM-Flow (Janus-1.5B, 20,001 tokens at 0.01 m, discrete flow matching).
  - PDMS vs steps: 1/2/3/5/10 give 89.1/89.7/90.0/90.3/90.2 at 0.09/0.19/0.29/0.48/0.94 s. Janus AR latency 0.27 s, but its PDMS is not reported. **V**
  - SFT 86.7 → GRPO 90.3. **V**. D: +3.6. 1→5 steps buys +1.2 PDMS for 5.3× the latency.
  - Source: [WAM-Flow](https://arxiv.org/html/2512.06112v2)
- WAM-Diff (LLaDA-V, 8.4B with MoE, 20,001 tokens at 0.01 m, 32 iterations).
  - SFT 86.6 → GSPO 91.0. **V**. D: +4.4.
  - Decoding order: random 90.0; causal 88.9; reverse-causal 91.0. **V**
  - Source: [WAM-Diff](https://arxiv.org/html/2512.11872)
- DriveFine (LaViDa/LLaDA-8B, 4,000 position bins at 0.05 m).
  - SFT 86.7 → GRPO 89.6 → offline refiner 90.3 → online refiner 90.8. 4 steps give 90.47 at ~207 ms. **V**
  - The authors claim that diffusion-head planners suffer an EPDMS drop and reward hacking under PDMS-RL. This is **V** as a statement, but no supporting table was seen.
  - Source: [DriveFine](https://arxiv.org/html/2602.14577)
- ReflectDrive (LLaDA-V, uniform x/y codebook over ±100 m, 5 steps). Without reflective inference 84.8 → +goal 87.4 → +safety-guided 90.3 → both 91.1. **V** — [ReflectDrive](https://arxiv.org/html/2509.20109v1)
- ReflectDrive-2 (0.7B masked-diffusion LM + 0.1B ViT, from proprietary weights, 16 discrete coordinate tokens).
  - SFT 87.7 (88.0 with AutoEdit) → RL 89.1 (91.0 with AutoEdit). **V**
  - Decoding steps: full-step frames use 1+3+3 decision–draft–reflect steps and lite-step frames 1+1 draft–reflect steps. The step sweep is said to plateau around "33–55 steps", which the paper does not reconcile with the 1+3+3 setting. **V**
  - Source: [ReflectDrive-2](https://arxiv.org/html/2605.04647)
- dVLM-AD (LLaDA-V, textual waypoints) vs its own AR VLM-AD (LLaVA-OV) on the same data. nuScenes behavior↔trajectory average 87.8 vs 85.5; WOD-E2E RFS 7.633 vs 7.215. **V**, [C2]: different backbones. "diffusion language-model fine-tuning is notably unstable in practice". **V** — [dVLM-AD](https://arxiv.org/html/2512.04459v1)
- ViLaD (LLaDA-V) vs AR VLMs of a different family: SFT L2 1.85 m vs 2.07–2.74 m; latency 2.78–4.28 s vs AR 28–60 s (3.18 s with caching). **V**, [X-ish] — [ViLaD](https://arxiv.org/html/2508.12603v1)

### Inferences
- The researcher's text-waypoint model (85.6) sits near the SFT-only level of the discrete masked-diffusion VLAs (86.6–87.7, [X]). Each paper that ablates text→tokens finds a large gain from a numeric token representation (AutoVLA +9.2; WAM-Flow, already known), and large gains from RL (+3.6 to +4.4).
- At NAVSIM scale (~100k frames), the only [C1] driving result favours continuous heads by 1.2–3.1 PDMS over AR tokens. It does not test masked or parallel discrete decoding, which closed most of the gap in robotics.

### Gaps
- No [C1] NAVSIM comparison of masked discrete diffusion vs a continuous flow/DiT head on the same backbone and data was found. DriveVLA-W0 compares AR (not masked) tokens; DriveFine, WAM-Diff, WAM-Flow and ReflectDrive compare only against other papers.
- DriveFine's claim that continuous heads reward-hack under PDMS-RL is not backed by a table I could see.
- The GTRS ablation (diffusion proposals + vocabulary scorer) was not verified beyond snippets.

## Q3. Hybrids: discrete coarse + continuous refinement, discrete flow matching, continuized discrete diffusion, vocabulary anchors + diffusion

### Takeaway
Hybrids that use discrete or vocabulary structure to place the mode and continuous or iterative refinement to fix the details give the best numbers in their own ablations. Examples: DiffusionDrive anchors (+3.5 over vanilla diffusion), GoalFlow goals (+2.9 for the goal score map, +4.7 with the full pipeline), HybridVLA (+0.08 over the best single head), KI/NORA-1.5 (tokens for training, flow for output), and VQ-BeT/CARP (coarse codes plus offset or finer scales). No paper found tests "discrete coarse token + continuous residual" on NAVSIM against a pure discrete or pure continuous head with the same backbone.

### Cited Findings
- **Anchored truncated diffusion (DiffusionDrive):** 20 K-means anchors + 2-step truncated diffusion give 88.1, vs 84.6 for vanilla 20-step diffusion and 84.0 for regression. Mode diversity is 74% vs 11%. DiffusionDrive also beats vocabulary-only VADv2-8192 (80.9) and Hydra-MDP-W-EP (86.5, the latter [X] within its Table 1). **V** — [DiffusionDrive](https://arxiv.org/html/2411.15139)
- **Goal-point vocabulary + flow (GoalFlow):**
  - Base rectified flow without a goal 85.6 → +distance score map 88.5 → +DAC score map 89.4 → +trajectory scorer 90.3. **V**. D: the goal adds +2.9 and the full pipeline +4.7.
  - Steps: 20/10/5/1 give 89.9/90.1/90.3/88.9 at 177.8/92.4/49.0/10.4 ms. Initial noise σ=0.2 collapses PDMS to 49.0. With too little noise, flow "degenerate[s] into a regression model". **V**
  - Source: [GoalFlow](https://arxiv.org/html/2503.05689)
- **Discrete tokens for training + continuous expert for output:**
  - KI (DROID 0.55 vs pi0 0.49 and pi0-FAST 0.45; trains like FAST, infers like flow). **V** — [KI](https://arxiv.org/html/2505.23705)
  - NORA-1.5: flow output beats FAST output by 2.9–5.8 points on SimplerEnv, but loses on 50K-frame real-robot data. **V** — [NORA-1.5](https://arxiv.org/html/2511.14659v1)
  - Alpamayo-R1 (discrete training tokens + flow inference). **V** — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
  - A 2026 survey describes "token targets train a backbone whose features condition a separate continuous head" as a general pattern. **R** — [Physical AI survey](https://arxiv.org/pdf/2610.21171)
- **AR + diffusion ensemble in one LLM (HybridVLA):** 0.74 vs 0.66 for diffusion-only and 0.62 for AR-only on RLBench. **V** — [HybridVLA](https://arxiv.org/html/2503.10631)
- **Discrete code + continuous offset:**
  - VQ-BeT (residual VQ + offset head): the offset is "quite important" and dropping the residual VQ layer has "a significant negative impact" (figure-only, **R** for magnitude). **V** (statement)
  - VQ-BeT vs BeT (k-means bins + offset): PushT 0.78 vs 0.39; Kitchen 3.66 vs 3.07. **V**
  - VQ-BeT vs Diffusion Policy-T: PushT 0.78 vs 0.74; BlockPush 1.79 vs 1.93; Kitchen inference 15.1 vs 98.6 ms. **V**. D: ≈6.5×.
  - Source for VQ-BeT: [VQ-BeT](https://arxiv.org/html/2403.03181)
- **Coarse-to-fine multi-scale tokens (CARP, residual VQ over 4 scales, V=512):**
  - State Square 0.98 vs Diffusion Policy-C 0.94; image Square 0.88 vs 0.92.
  - MimicGen multi-task 0.85 vs SDP 0.78.
  - Time for 400 actions: 3.07 vs 35.21 s (D: ≈11.5×).
  - Fewer than 4 scales gives "less precise predictions" (figure-only).
  - **V**, [C2] — [CARP](https://arxiv.org/html/2412.06782)
- **Discrete flow matching:**
  - DFM-VLA beats discrete diffusion and AR on the same architecture (4.42 vs 4.32 vs 4.18 on CALVIN), and beats continuous-diffusion RDT on real robot (70.8 vs 60.0, [X]). **V** — [DFM-VLA](https://arxiv.org/html/2603.26320v2)
  - WAM-Flow: 89.1 PDMS at 1 step. **V** — [WAM-Flow](https://arxiv.org/html/2512.06112v2)
- **Continuized discrete diffusion (E0):** Gaussian noise on one-hot discrete actions "avoid[s] the distributional mismatch inherent in mask-based diffusion". E0 vs pi0: +9.4 on the 14-environment average and LIBERO-Long 92.2 vs 85.2. **V** (D: 60.8−51.4 = 9.4). It runs no numeric comparison against mask-based discrete diffusion. **V** — [E0 v2](https://arxiv.org/html/2511.21542v2)
- **Discrete draft + discrete edit:**
  - DriveFine refinement expert: +0.7 (offline) and +0.5 (online) over GRPO alone; refinement blocks n=0→4 give 89.6→90.8. **V** — [DriveFine](https://arxiv.org/html/2602.14577)
  - ReflectDrive: safety-guided token search, 84.8→91.1. **V** — [ReflectDrive](https://arxiv.org/html/2509.20109v1)
  - ReflectDrive-2: AutoEdit adds only +0.3 after SFT but +1.9 after RL. **V** — [ReflectDrive-2](https://arxiv.org/html/2605.04647)
- **Other hybrid representations:**
  - MMaDA-VLA (continuous controls in one discrete token space, masked denoising) and MVLAD-AD (discrete action codebook + explanations, nuScenes) were seen only as snippets. **R** — [MMaDA-VLA](https://arxiv.org/pdf/2603.25406.pdf); [MVLAD-AD](https://arxiv.org/html/2602.20577)
  - GTRS: diffusion proposals + 16k "super-dense" vocabulary scorer; best single model 45.3 EPDMS on navhard. **R** — [GTRS](https://arxiv.org/html/2506.06664v1)

### Inferences
- "How continuous" is really two questions: how to place the mode and how to refine it. The evidence favours discrete or vocabulary structure (anchors, goals, tokens) for the first and continuous, or fine-grained discrete with metric-aware structure, for the second. Purely continuous diffusion from Gaussian noise without anchors mode-collapses (D=11%), and purely discrete coarse bins lack precision.
- A masked discrete diffusion head over fine bins (≤0.05 m) is already a "hybrid" in the WAM-Flow/E0 sense if its embedding is metric-aware.

### Gaps
- No public driving paper found with a coarse discrete token plus a continuous residual (VQ-BeT-style) on NAVSIM.
- No direct comparison of E0's Gaussian-on-one-hot vs mask-based discrete diffusion.
- No independent check of the MAR-style "continuous tokens with diffusion loss" idea for actions.

## Q4. Where each representation wins or loses: the pattern across papers

### Takeaway
- **Continuous wins on:** precision at small data and small backbones, smoothness/comfort, and latency with few steps (with anchors or distillation).
- **Discrete wins on:** training speed (FAST 3–5×, KI 7.5×), data scaling in some reports, language following, LLM compatibility, native categorical sampling for diversity and RL log-probs, and in-place editing.
- **The papers disagree on:** which one scales better and which one is more data-efficient.

### Cited Findings
- **Precision / quantization floor:**
  - OFT: about +5 for continuous over parallel discrete. **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
  - Octo: discrete "missed the grasp". **V** — [Octo](https://arxiv.org/html/2405.12213)
  - RoboVLMs: discrete loses most on long horizons. **V** — [RoboVLMs](https://arxiv.org/html/2412.14058)
  - DriveVLA-W0: blames quantization for AR's 3.1-PDMS NAVSIM gap. **V** — [DriveVLA-W0](https://arxiv.org/html/2510.12796)
  - AutoVLA reconstruction ADE: 0.018 m (K-disk) vs 0.101 m (bins). **V** — [AutoVLA](https://arxiv.org/html/2506.13757)
  - Fine bins used in driving: WAM-Diff and WAM-Flow 0.01 m (max error 0.005); DriveFine 0.05 m. **V** — [WAM-Diff](https://arxiv.org/html/2512.11872); [DriveFine](https://arxiv.org/html/2602.14577)
- **Multimodality:**
  - Regression hedges: Octo MSE "move very slowly". **V** — [Octo](https://arxiv.org/html/2405.12213)
  - ORION's MLP "struggles" with multimodality. **V** — [ORION](https://arxiv.org/html/2503.19755)
  - Vanilla diffusion collapses to D=11%; anchors restore D=74%. **V** — [DiffusionDrive](https://arxiv.org/html/2411.15139)
  - Low-noise flow degenerates to regression. **V** — [GoalFlow](https://arxiv.org/html/2503.05689)
  - Diffusion "may ... overfit to suboptimal modes". **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
- **Training speed/stability:**
  - FAST: 5× fewer GPU hours, and better at matched compute. **V** (claims) — [FAST](https://arxiv.org/html/2501.09747)
  - KI: pi0 needs 7.5× the steps. **V** — [KI](https://arxiv.org/html/2505.23705)
  - OFT: diffusion 100–250K vs 50–150K steps. **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
  - ORION: VAE more stable than diffusion. **V** — [ORION](https://arxiv.org/html/2503.19755)
  - dVLM-AD: masked-diffusion LM fine-tuning "notably unstable", and extra action vocabularies or heads "amplify this instability". **V** — [dVLM-AD](https://arxiv.org/html/2512.04459v1)
  - DriveFine (claim): diffusion heads need "hundreds of epochs". **V** (as statement) — [DriveFine](https://arxiv.org/html/2602.14577)
- **Data scale (disagreement):**
  - DriveVLA-W0: continuous better at 103k frames, AR better at 70M; flow "too sample-inefficient". **V** — [DriveVLA-W0](https://arxiv.org/html/2510.12796)
  - NORA-1.5: flow worse on small (50K-frame) data. **V** — [NORA-1.5](https://arxiv.org/html/2511.14659v1)
  - DFM-VLA: at 10% data, discrete flow 3.21 vs AR 1.71. **V** — [DFM-VLA](https://arxiv.org/html/2603.26320v2)
  - FAST: comparable on small data. **V** (claim) — [FAST](https://arxiv.org/html/2501.09747)
  - Alpamayo-R1: flow beats AR on large in-house data. **V** — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
- **Inference steps/latency:**
  - AR is slow: pi0-FAST 750 ms vs 100 ms. **V** — [FAST](https://arxiv.org/html/2501.09747)
  - Alpamayo-R1: AR 222 ms vs flow 8.75 ms. **V** — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
  - Iterative discrete and continuous heads cost the same per step: 68.8 vs 67.1 ms at 12 steps. **V** — [DD-VLA](https://arxiv.org/html/2508.20072)
  - Few steps work for anchored or goal flow (DiffusionDrive 1 step 87.9; GoalFlow 1 step 88.9) and for discrete flow (WAM-Flow 1 step 89.1). **V** — [DiffusionDrive](https://arxiv.org/html/2411.15139); [GoalFlow](https://arxiv.org/html/2503.05689); [WAM-Flow](https://arxiv.org/html/2512.06112v2)
  - Undistilled diffusion fails at 1 step (OFT LIBERO-Long 0.0). **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
- **LLM compatibility / backbone damage:**
  - Continuous-expert gradients "degrade the pre-trained VLM backbone", and stop-gradient improves language following. **V** — [KI](https://arxiv.org/html/2505.23705)
  - pi0 diffusion "often ignores the language instructions". **V** — [FAST](https://arxiv.org/html/2501.09747)
  - Text waypoints are worst: ORION 42.23 DS (**V**, [ORION](https://arxiv.org/html/2503.19755)); AutoVLA 71.31 PDMS (**V**, [AutoVLA](https://arxiv.org/html/2506.13757)).
- **RL-trainability on PDMS:**
  - Discrete heads take GRPO/GSPO directly on token log-probs. Gains: WAM-Flow +3.6, WAM-Diff +4.4, DriveFine +4.1 with refiner (D: 90.8−86.7), ReflectDrive-2 +3.3 (D: 91.0−87.7), AutoVLA +8.57. **V**/**D** — sources as in Q2.
  - Continuous-head example (already known): ReCogDrive 86.5→90.8.

### Inferences
- For an 8-waypoint, 2 Hz, 2-D driving output (16 numbers), FAST's argument about high-frequency token counts mostly does not apply. With 0.01–0.05 m bins the quantization error is far below PDMS sensitivity. The relevant trade-offs are therefore decoding parallelism, multimodality control (anchors/temperature), RL plumbing and backbone coupling, not precision.
- K plans in one pass: the evidence comes from batched sampling.
  - Anchored diffusion samples 20 noises and anchors in one batched pass with 2 steps (DiffusionDrive: N_infer 10→20 adds +3.2 PDMS, D: 88.1−84.9).
  - Discrete heads draw K samples through temperature (DD-VLA) or goal candidates (ReflectDrive uses K=3).
  - Vanilla diffusion gives K near-duplicates (D=11%).
  - Mode spread is therefore set by the anchor or temperature design more than by discrete vs continuous.
- Scaling evidence is contradictory (W0 vs Alpamayo vs NORA-1.5). It should not drive a choice at the user's data scale (~100k NAVSIM frames). At that scale the only [C1] driving result slightly favours continuous one-pass or flow heads over AR tokens.

### Gaps
- No paper isolates the RL-trainability of discrete vs continuous heads on the same backbone and reward.
- Comfort and smoothness of discrete masked heads in driving have not been measured against flow heads (only Alpamayo vs AR).

## Q5. Small backbones (≤2B) and heads that read cached VLM states

### Takeaway
At ≤2B, the evidence leans toward a generative head (flow or masked/discrete diffusion) over plain regression, and toward reading intermediate rather than final layers.
- SmolVLA at 0.5B: flow 80.25 vs regression 75.25.
- GR00T at 2B: the middle (12th) layer is "faster" and gives "higher success".
- ReflectDrive-2 shows a 0.8B discrete masked-diffusion driving model reaching 91.0 PDMS after RL with few decoding steps (1+1 to 1+3+3).
- KI warns that a head on a fully frozen backbone can fail (0% on shirt folding).

### Cited Findings
- **SmolVLA (SmolVLM2 ~0.5B, LIBERO):**
  - Flow matching 80.25 vs L1 regression 75.25 (Long 53 vs 38). **V**. D: +5.0 average, +15 on Long.
  - Layers read: first 8/16/24/32 give 75.0/78.5/79.5/80.3; every other layer 75.5; a 256M VLM 75.8. **V**
  - Expert attention: cross-attention + self-attention 85.5 vs cross-attention only 79.0 vs self-attention only 74.5. **V**
  - Source: [SmolVLA](https://arxiv.org/html/2506.01844)
- **OFT (7B):** L1 ≈ diffusion "because" the "high-capacity" model handles the distribution. **V** — [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
  - D: OFT shows a 0.1-point flow-vs-regression gap at 7B; SmolVLA shows 5.0 at 0.5B.
- **GR00T N1 (2.2B total, 1.34B VLM):** the DiT reads 12th-layer embeddings ("middle-layer instead of final-layer LLM embeddings resulted in both faster inference speed and higher downstream policy success rate"). 4 flow steps; 63.9 ms per 16-action chunk on an L40. **V** — [GR00T N1](https://arxiv.org/html/2503.14734)
- **CogACT:** the DiT head reads one cognition-token feature. At equal size (89M) a DiT head scores 62.5 vs 52.5 for an MLP. **V** — [CogACT](https://arxiv.org/html/2411.19650)
- **RoboVLMs:** a separate policy head beats interleaving actions in the VLM. The hypothesis is that it "preserves the VLM's original vision-language fusion capabilities." **V** — [RoboVLMs](https://arxiv.org/html/2412.14058)
- **KI:** freezing the backbone gave 0% on shirt folding. Stop-gradient, rather than full freezing, plus co-training is the recommended form of insulation. **V** — [KI](https://arxiv.org/html/2505.23705)
- **MiniVLA (0.5B Qwen):** binning 61.4% ≈ OpenVLA-7B 62% on LIBERO-90; VQ chunks 77%. **V** — [MiniVLA blog](https://ai.stanford.edu/blog/minivla/)
- **Small driving models:**
  - ReflectDrive-2 (0.7B masked-diffusion LM + 0.1B ViT): SFT 87.7, RL 91.0, 16 discrete coordinate tokens, 1+1 to 1+3+3 decoding steps. **V** — [ReflectDrive-2](https://arxiv.org/html/2605.04647)
  - WAM-Flow (1.5B): 89.1 at 1 step / 0.09 s. **V** — [WAM-Flow](https://arxiv.org/html/2512.06112v2)
  - Alpamayo-R1, 0.5B vs 10B: minADE6 0.913 vs 0.849; AlpaSim 0.35 vs 0.72. **V** — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
  - DriveVLA-W0: a ~500M expert reads the 7–8B VLA; the one-pass query head is fastest (74 ms) and best on NAVSIM. **V** — [DriveVLA-W0](https://arxiv.org/html/2510.12796)

### Inferences
- For a 0.8B backbone with cached hidden states, the closest evidence is:
  - SmolVLA: a generative head beats regression by about 5 points.
  - GR00T and SmolVLA: middle layers are as good as or better than the last layer.
  - CogACT: transformer heads beat MLP heads at equal size.
  - KI: frozen features can bottleneck. This risk is reduced here because the cached states come from a model already fine-tuned on driving text waypoints.
- A masked discrete head reading cached states cannot use the LLM's own vocabulary embeddings unless it is built that way. It is then architecturally a small token transformer, much like a DiT but with a categorical output. DD-VLA's [C1] result (95.1 vs 94.9) suggests that choice costs little in accuracy.

### Gaps
- No public paper was found that compares discrete vs continuous heads specifically on cached, frozen VLM hidden states at ≤1B.
- No controlled study of AdaLN ego-state conditioning vs token conditioning for action heads.
- ReflectDrive-2's backbone is proprietary, so its 0.8B result cannot be reproduced from public weights.
