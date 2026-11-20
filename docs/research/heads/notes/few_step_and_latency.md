# Few-step / one-step generation for action heads, and measured action-head latency on desktop/datacenter GPUs (as of 2026-11-19)

Evidence tags on every number: **[V]** = read from the paper text/tables, official docs or the repo (full text fetched this session); **[R]** = self-reported in a snippet, abstract listing, third-party summary or community post, not checked against a full table; **[D]** = derived by me, arithmetic shown. Estimates are never presented as measurements. Scope: on-device/embedded deployment is out of scope (coordinator, 2026-11-19), so latency below is from desktop/datacenter GPUs only (RTX 2080 Ti/3070 Ti/4090/5090, P5000, V100, L40, H100, H800, RTX 6000 Pro Blackwell).

## Q1. Few-step continuous methods for actions/trajectories: quality kept at 1, 2, 4 steps vs many steps

### Takeaway
Driving planners whose sampling starts close to the answer (**anchored/truncated diffusion, goal-conditioned flow**) lose almost nothing at 1–2 steps on NAVSIM: DiffusionDrive goes from 88.1 to 87.9 PDMS (2 → 1 step), GoalFlow from 90.3 to 88.9 (5 → 1). On robot benchmarks, **naive one-step sampling of a standard diffusion policy collapses to 0% success**. One-step methods built for it (OneDP distillation, consistency policy, shortcut, FlowPolicy, MeanFlow/MP1, streaming flow) match or nearly match 10–100-step teachers. Consistency-policy distillation is the weakest of these one-step options.

### Cited Findings
**Truncated / anchored diffusion (DiffusionDrive, NAVSIM navtest)**
- Inference uses 2 DDIM denoising steps [V], 20 K-means anchors [V], training schedule truncated to T_trunc = 50 of T = 1000 [V]; inference starts from noised anchors, not pure Gaussian noise [V] — [DiffusionDrive](https://arxiv.org/html/2411.15139)
- Step ablation (Table 4, 60M params): 1 step 87.9 PDMS, 2 steps 88.1, 3 steps 88.1 [V]. (The 3-step row lists EP = 92.2 vs 82.1–82.2 in the others; likely a typo in the paper [V as printed].) — [DiffusionDrive](https://arxiv.org/html/2411.15139)
- Versus vanilla diffusion (Table 2): TransfuserDP 20 steps, 6.5 ms/step, 130.0 ms plan time, 101M params, 7 FPS, 84.6 PDMS, diversity 11%; DiffusionDrive 2 steps, 3.8 ms/step, 7.6 ms plan time, 60M params, 45 FPS, 88.1 PDMS, diversity 74% (single RTX 4090) [V]. Transfuser (deterministic) 84.0 PDMS at 60 FPS [V]. — [DiffusionDrive](https://arxiv.org/html/2411.15139)

**Goal-conditioned flow matching (GoalFlow, NAVSIM)**
- Table 3 steps vs PDMS / inference time: 20 steps 89.9 / 177.8 ms; 10 steps 90.1 / 92.4 ms; 5 steps 90.3 / 49.0 ms; 1 step 88.9 / 10.4 ms [V]. GPU for timing not stated [V]. The paper's "1.6% drop" is relative (1.4/90.3) [V/D]. — [GoalFlow](https://arxiv.org/html/2503.05689)
- Generates 128/256 trajectories, a trajectory scorer picks one; scorer adds +0.9 PDMS (M2 89.4 → M3 90.3) [V]. — [GoalFlow](https://arxiv.org/html/2503.05689)

**Flow matching in a driving VLA (Alpamayo-R1)**
- Flow-matching trajectory decoder, 5 steps (δt = 0.2) in the latency table. Elsewhere the paper gives a default δt = 0.1 (~10 steps), so the step count is internally inconsistent [V]. — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
- Flow matching vs autoregressive trajectory tokens (Table 12): minADE6 0.6440 vs 0.6811, comfort 97.38% vs 44.05%, relative decode speed 1.16× vs 1.00× (AR baseline here = 6 VQGAN tokens) [V]. — [Alpamayo-R1](https://arxiv.org/html/2511.00088)

**One-step distillation (OneDP, Robomimic/Push-T, visual policies)**
- Naive DDIM at 1 step: **0.000 success on all six tasks** (vs DDPM-100 avg 0.829, DDIM-10 avg 0.836) [V]. — [OneDP](https://arxiv.org/html/2410.21257)
- OneDP-S (1 step) avg 0.843, OneDP-D 0.826 vs DDPM-100 teacher 0.829 [V]. EDM setting: teacher 35 NFE 0.829; OneDP-S 1 step 0.830 after 20 distillation epochs; Consistency Policy 1 step 0.672 and 3 steps 0.712 even after 450 epochs [V]. — [OneDP](https://arxiv.org/html/2410.21257)
- Distillation (reverse-KL / score-difference, SDS/VSD-inspired) costs 2–10% of pre-training [V]; 20 sim epochs vs 1000 pre-training epochs [V]. Real Franka avg 0.98 (OneDP-S, 1 step) vs 0.83 (DDIM-10) [V]. — [OneDP](https://arxiv.org/html/2410.21257)

**Consistency Policy (consistency-trajectory distillation from an EDM teacher)**
- Table I (Lift / Can / Square / ToolHang / Push-T): DDPM 1.00/.97/.93/.79/.87; DDIM 1.00/.82/.85/.14/.78; CP 1 step 1.00/.98/.92/.70/.82; CP 3 steps 1.00/.95/.96/.77/.84 [V]. Baseline NFEs (27, 9) are 100/15 steps divided by ParaDiGMS speedups, not literal step counts [V]. — [Consistency Policy](https://arxiv.org/html/2405.07503)
- Franka Kitchen p4: DDPM .98, DDIM .93, CP-1 .93, CP-3 .94 [V]. Distillation needs more training time than the teacher itself; CTM >40% slower than CD [V]. — [Consistency Policy](https://arxiv.org/html/2405.07503)

**Shortcut models (self-consistency, single training run)**
- FID (DiT-B): CelebA 128/4/1 steps: flow matching 7.3/(63.3)/(280.5) vs shortcut 6.9/13.8/20.5; ImageNet: FM 17.3/(108.2)/(324.8) vs shortcut 15.5/28.3/40.3 [V]. — [Shortcut models](https://arxiv.org/html/2410.12557)
- Robotics: Push-T / Transport success: Diffusion Policy 100 steps 0.95 / 1.00; Diffusion Policy 1 step 0.12 / 0.00; Shortcut Policy 1 step 0.87 / 0.80 [V]. Training ≈16% more compute than diffusion (3.5 vs 3 units/update) [V]. — [Shortcut models](https://arxiv.org/html/2410.12557)

**FlowPolicy (consistency flow matching, 3D point clouds, 37 Adroit + Meta-World tasks)**
- 1 step 70.0% avg vs DP3 (10 steps) 68.7%, Simple DP3 67.4%, Consistency Policy (1 step) 50.1% [V]. — [FlowPolicy](https://arxiv.org/html/2412.04987)

**MeanFlow / MP1**
- MeanFlow (image): FID 3.43 at 1 NFE on ImageNet 256 from scratch, no distillation [R, abstract snippet]. — [MeanFlow](https://arxiv.org/abs/2505.13447)
- MP1 (1 NFE) 78.9% avg vs DP3 (10 NFE) 68.7% and FlowPolicy (1 NFE) 71.6% on the same 37 tasks [V]. Note: FlowPolicy's own paper reports 70.0% — re-run discrepancy [V both]. — [MP1](https://arxiv.org/html/2507.10543); [FlowPolicy](https://arxiv.org/html/2412.04987)
- MP1 has **no NFE ablation** (only 1 NFE reported) [V]. MeanFlow-ratio ablation: ratio 0 (plain FM) 72.6%, 0.5 → 82.4%, 1.0 → 2.4% (collapse) [V]. — [MP1](https://arxiv.org/html/2507.10543)

**Streaming Flow Policy (flow in action space starting from the last executed action)**
- Push-T state: SFP 91.7% avg vs DP (100 DDPM) 90.7%; RoboMimic Can 98.4% vs 94.0%, Square 78.0% vs 77.2%, Lift 100% both; Push-T image 82.5% vs 83.8% [V, column assignment partly inferred by the fetch tool]. Step count is a Δt hyperparameter, not reported as a fixed number [V]. — [Streaming Flow Policy](https://arxiv.org/html/2505.21851)

### Inferences
- The decisive variable for 1–2-step quality is how close the start distribution is to the answer, not the ODE solver. Anchors (DiffusionDrive), a goal point (GoalFlow) or the last action (streaming flow) make 1 step nearly free. A standard diffusion policy started from pure noise fails at 1 step (OneDP DDIM-1 = 0.000; DP-1 0.12/0.00 in the shortcut paper).
- On NAVSIM, going to 1 step cost 0.2 PDMS for DiffusionDrive and 1.4 PDMS for GoalFlow against its 5-step best [D: 88.1−87.9; 90.3−88.9]. A 2-step anchored head is the safest documented setting for an 8-waypoint driving head. 1 step is supported by evidence but costs a little.
- Among one-step continuous options, self-consistency methods (shortcut, MeanFlow/MP1, FlowPolicy) train in one run with no separate teacher-distillation stage. OneDP-style distillation is cheap (2–10% extra training) and matches its teacher most closely in the robot tables. Consistency-policy distillation is the weakest at 1 step in OneDP's head-to-head comparison (0.672 vs 0.830).

### Gaps
- No paper found that sweeps 1/2/4 steps for shortcut, MeanFlow or consistency heads **on a driving benchmark** (NAVSIM/nuPlan); all their step evidence is robot or image benchmarks.
- MP1 and SmolVLA report no NFE sweep; Alpamayo-R1's step count is internally inconsistent (5 vs ~10).
- No reflow (2-rectified-flow) result on a driving or robot trajectory benchmark was found in this pass.

## Q2. Discrete masked diffusion: how many parallel decoding steps are needed for short (≈16–32 token) sequences?

### Takeaway
For driving plans of 16–32 tokens, the public evidence points to **about 3–5 parallel unmasking steps** being enough. ReflectDrive peaks at 5 steps and then declines. ReflectDrive-2 decodes 16 tokens with 1+3+3 or 1+1 steps and reports a plateau in the low single digits. DriveFine keeps 90.47 PDMS at 4 steps. Some systems use many more steps (WAM-Diff 32 iterations for 32 tokens, Discrete Diffusion VLA 12 rounds for 56 tokens), but they publish no step ablation that shows they need them.

### Cited Findings
- **ReflectDrive-2**: 8 waypoints × (1 longitudinal + 1 lateral) = **16 coordinate tokens** (L = 16, 4 s at 2 Hz) [V; the HTML renders these as "88"/"1616", read from Eq. 4]. Backbone: 0.7B masked-diffusion LM + 0.1B ViT, fully fine-tuned [V]. Full-step frames use S = 1+3+3 (decision–draft–reflect) steps; lite frames use 1+1 [V]. Text says PDMS "improves and then plateaus around 33–55 steps" [V as rendered; see Inferences]. 91.0 PDMS camera-only [R, NeurIPS listing]; best-of-6 oracle 94.8 [V]. — [ReflectDrive-2](https://arxiv.org/html/2605.04647); [NeurIPS listing](https://neurips.cc/virtual/2026/poster/152356)
- **ReflectDrive (v1)**: answer length 32, block length 32, **5 decoding steps** [V]; performance "improves during the initial steps, peaks at 5 steps, and subsequently declines" [V]. PDMS 84.8 without reflective inference → 91.1 with goal-conditioning + safety-guided regeneration [V]; reflection usually converges in 1–3 iterations (cap 10) [V]. No latency reported [V]. — [ReflectDrive](https://arxiv.org/html/2509.20109v1)
- **DriveFine** (LLaDA-8B base): default **12 sampling steps + 1 refining step** [V]; with 4 steps, 90.47 PDMS at 207 ms average latency (hardware not stated) [V]. Refinement blocks: 0/1/2/4/6/8 blocks → 89.6/90.0/90.4/90.8/90.8/90.7 PDMS at 8/8.25/8.5/9/9.5/10 B params [V]. Coordinate bins: 4,000 at 0.05 m; heading 1,800 at 0.1° [V]. Headline 90.7 → 91.8 PDMS after RFT [R, alphaXiv summary]. — [DriveFine](https://arxiv.org/html/2602.14577); [alphaXiv overview](https://www.alphaxiv.org/overview/2602.14577)
- **WAM-Diff** (LLaDA-V, 8.4B): output length fixed at 32 tokens; 20,001 numeric tokens over [−100, 100] at 0.01 resolution [V]; inference runs **32 masked-diffusion iterations** with early stop, no step ablation, no latency reported [V]. Remasking order matters: random 90.0, causal 88.9, reverse-causal 91.0 PDMS [V]. — [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- **WAM-Diff2**: AR → block-wise discrete diffusion distillation gives a "2.8x decoding speedup", "15.1x" with FlashInfer + CUDA Graphs [V, abstract]; ms, hardware and step counts not visible in the abstract [V]. — [WAM-Diff2](https://arxiv.org/abs/2608.01035)
- **Discrete Diffusion VLA** (robot): 56 tokens per chunk on LIBERO (8 steps × 7 dims) [V]; **T = 12** refinement rounds, cosine schedule [V]; steps-vs-success sweep only in a figure [V]. LIBERO avg 96.4% vs OpenVLA-OFT discrete 95.5% and OFT continuous L1 97.1% [V]. Secondary remasking costs <1 ms [V]. — [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)
- **LLaDA-VLA**: 35 tokens per chunk (5 steps × 7), **10 diffusion steps** (2 per action), hierarchical action-level then token-level remasking [V]; no step ablation or latency [V]. — [LLaDA-VLA](https://arxiv.org/html/2509.06932)

### Inferences
- The ReflectDrive-2 HTML doubles digits consistently ("88 waypoints" and "1616 tokens" where Eq. 4 shows 8 and 16). "Plateaus around 33–55 steps" therefore very likely means **3–5 steps**. This is an inference from the rendering pattern, not a verified reading. It also matches the 1+3+3 setting and ReflectDrive v1's 5-step peak.
- For a 16-token plan, 3–5 steps unmask 3–5 tokens per step [D: 16/5 ≈ 3.2 to 16/3 ≈ 5.3]. WAM-Diff's 32 iterations for 32 tokens is about 1 token per step, so it saves no steps over AR; its speed comes from fast-dLLM, not from fewer steps.
- Decode order matters (reverse-causal +1.0 over random, +2.1 over causal in WAM-Diff), so a MaskGIT-style confidence schedule plus a far-to-near order is a cheap knob for short trajectories.

### Gaps
- No public steps-vs-PDMS table (only figures) for ReflectDrive, ReflectDrive-2, DriveFine or Discrete Diffusion VLA; exact curves need the figures read by eye or the code.
- No masked-diffusion result at 1–2 steps for driving tokens; whether 2 steps hold quality for 16 tokens is unmeasured.

## Q3. Measured action-head latency on desktop/datacenter GPUs; scaling with steps, number of samples and head size

### Takeaway
On desktop GPUs, measured action heads take a few ms per denoising step. DiffusionDrive's 60M decoder takes 3.8 ms/step for 20 batched samples (RTX 4090). pi0's 300M action expert takes 27 ms for 10 steps (2.7 ms/step, RTX 4090). Alpamayo-R1's flow decoder takes 8.75 ms for 5 steps (RTX 6000 Pro Blackwell). GR00T N1.6's 4-step head takes 17 ms under torch.compile on an RTX 4090 [R]. Latency scales close to linearly with step count, with a small fixed overhead (GoalFlow ≈ 8.8 ms/step, OneDP 6.6 ms/step). Compilation choices (eager vs torch.compile vs TensorRT) change whole-policy latency by up to 4.4× on an H100, which is larger than the gain from going from 4 to 2 steps.

### Cited Findings
**Driving heads (desktop/datacenter GPUs)**
- DiffusionDrive: 3.8 ms/step, 7.6 ms for 2 steps, whole model 45 FPS on RTX 4090; TransfuserDP 6.5 ms/step × 20 = 130 ms [V]. — [DiffusionDrive](https://arxiv.org/html/2411.15139)
- GoalFlow: 10.4 ms (1 step), 49.0 ms (5), 92.4 ms (10), 177.8 ms (20); GPU unstated (training on RTX 4090/3090 nodes) [V]. — [GoalFlow](https://arxiv.org/html/2503.05689)
- Alpamayo-R1 on RTX 6000 Pro Blackwell (Table 14): vision 3.43 ms, prefill 16.54 ms, **flow trajectory decoding 8.75 ms (5 steps)**; trajectory-only total 29 ms; with 40 reasoning tokens 99 ms; AR trajectory (127 tokens) 222 ms → 312 ms total [V]. Paper mentions a "2B parameter diffusion-based trajectory decoder" for the 10B model [V]. — [Alpamayo-R1](https://arxiv.org/html/2511.00088)

**VLA action experts**
- pi0 on RTX 4090, 3 cameras: image encoders 14 ms, observation forward 32 ms, **10 flow steps 27 ms**, total 73 ms without network transfer (86 ms with 13 ms network latency) [V]. Action expert ~300M (width 1024, MLP 4096), H = 50; prefix K/V cached and only action tokens recomputed per step [V]. — [pi0](https://arxiv.org/html/2410.24164)
- GR00T N1: 2.2B params total, 1.34B in VLM; K = 4 flow steps; H = 16; 63.9 ms per 16-action chunk on L40 bf16; uses the 12th LLM layer [V]. — [GR00T N1](https://arxiv.org/html/2503.14734)
- GR00T N1.6 on RTX 4090, torch.compile, 4 steps, single view: data 2 ms, backbone 25 ms, **action head 17 ms**, E2E 44 ms (22.8 Hz); eager action head 55 ms, E2E 82 ms [R, search snippet of a README mirror]. — [Mintlify mirror of Isaac-GR00T docs](https://mintlify.wiki/NVIDIA/Isaac-GR00T/deployment/optimization)
- GR00T N1.7 repo benchmark (LIBERO `libero_10`, 32 DiT blocks, batch 1, action horizon 40, 4 steps, two 256×256 cameras, median complete-policy-call ms, eager / torch.compile / TensorRT full pipeline): H100 101.5 / 53.4 / 22.9; RTX Pro 6000 Blackwell 75.7 / 52.1 / 24.7; L40 123.7 / 66.8 / 35.0 [V]. torch.compile uses max-autotune on the action head only; no per-component split and no step sweep [V]. — [Isaac-GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)
- GR00T N1.7 README: horizon raised 16 → 40; "DiT action head changed from 32 to 16 diffusion layers" [V]. This conflicts with the 32-block LIBERO checkpoint used in the benchmark [V]. — [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- SmolVLA: 450M total, ~100M action expert, first 16 LLM layers, 10 flow steps, chunk 50 [V]; **no latency in ms reported** [V]; async inference cuts task time ~30% (13.75 → 9.70 s avg) [V]. — [SmolVLA](https://arxiv.org/html/2506.01844)
- Community custom CUDA kernels on RTX 5090: pi0.5 17.58 ms; pi0 18.43–24.48 ms; GR00T N1.6 13.08 ms (T=50) / 12.53 ms (T=16) [R, community forum; steps undefined]. — [NVIDIA developer forum](https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788)

**Robot-policy heads**
- Consistency Policy (P5000): DDPM 100 NFE 110 ms; DDIM 15 NFE 11 ms; CP 1 NFE 1 ms; CP 3 NFE 2 ms [V]. Laptop RTX 3070 Ti: DDIM 192/198 ms vs CP 21/22 ms (whole inference) [V]. — [Consistency Policy](https://arxiv.org/html/2405.07503)
- OneDP (V100): observation encoding 9 ms; DDPM-100 660 ms; DDIM-10 66 ms; OneDP 7 ms; headline 1.5 Hz → 62 Hz [V]. — [OneDP](https://arxiv.org/html/2410.21257)
- FlowPolicy (RTX 2080 Ti): 19.9 ms (1 step) vs DP3 145.7 ms (10 steps) [V]. MP1 (RTX 4090, 3 seeds sharing the GPU): 6.8 ms vs FlowPolicy 12.6 ms vs DP3 132.2 ms [V]. — [FlowPolicy](https://arxiv.org/html/2412.04987); [MP1](https://arxiv.org/html/2507.10543)
- Streaming Flow Policy per action: Push-T state 3.5 ms vs DP-100 40.2 ms, DDIM-10 4.4 ms, flow matching 5.8 ms; image 8.8 ms vs 127.2 ms (GPU not named) [V]. — [Streaming Flow Policy](https://arxiv.org/html/2505.21851)
- Discrete Diffusion VLA (H800, LIBERO-Goal): OpenVLA AR 136.2 ms (56 NFE); OFT parallel 31.1 ms (1 NFE); OFT diffusion 50 steps 199.9 ms, 12 steps 67.1 ms; Discrete Diffusion VLA 12 steps 68.8 ms [V]. — [Discrete Diffusion VLA](https://arxiv.org/html/2508.20072)

**Derived scaling numbers [D]**
- Steps → latency is close to linear. GoalFlow: slope (177.8 − 10.4)/(20 − 1) = 8.81 ms/step, intercept ≈ 10.4 − 8.81 ≈ 1.6 ms. OneDP: 660/100 = 66/10 = 6.6 ms/step. OFT diffusion: (199.9 − 67.1)/(50 − 12) = 3.49 ms/step, intercept ≈ 67.1 − 12 × 3.49 ≈ 25.2 ms (close to the 31.1 ms 1-NFE run). pi0: 27/10 = 2.7 ms/step (300M expert, 50 action tokens). Alpamayo-R1: 8.75/5 = 1.75 ms/step. DiffusionDrive: 3.8 ms/step (60M, 20 samples batched). GR00T N1.6: 17/4 ≈ 4.3 ms/step on RTX 4090 [D from R].
- Compilation effect, GR00T N1.7 whole policy: H100 eager/TensorRT = 101.5/22.9 = 4.4×; torch.compile/TensorRT = 53.4/22.9 = 2.3×; L40 123.7/35.0 = 3.5×.
- Head share of the pipeline: Alpamayo-R1 trajectory-only, 8.75/29 = 30%; pi0, 27/73 = 37%; GR00T N1.6 (RTX 4090, compile), 17/44 = 39% [R-based].

### Inferences
- With 2–5 denoising steps at 1.75–4.3 ms/step, a small head on a desktop GPU costs roughly 4–20 ms [D: 2 × 1.75 to 5 × 4.3 ≈ 3.5–21.5 ms]. The VLM prefill (16.5 ms in Alpamayo-R1, 25 ms backbone in GR00T N1.6, 32 ms in pi0) usually costs as much as or more than the head.
- Head size is a weaker predictor of per-step time than implementation: the 60M DiffusionDrive decoder (3.8 ms/step) is slower per step than pi0's 300M expert (2.7 ms/step), on different workloads. Prefix K/V caching (pi0) and compilation (GR00T eager → TensorRT 4.4×) dominate.

### Gaps
- No public per-step sweep of any VLA action head (GR00T fixes 4 steps, pi0 10, SmolVLA 10); step scaling is derived from planners and robot policies.
- No published GR00T DiT parameter count for N1.5/N1.6/N1.7; GR00T N1 only implies 2.2B − 1.34B = 0.86B for everything outside the VLM [D].
- No controlled head-size scaling measurement (e.g., 16 vs 32 DiT layers) was found; the README's 16-layer claim has no matching benchmark.
- GoalFlow's and Streaming Flow Policy's timing GPUs are not named.

## Q4. Cost of producing K plans: batching K samples through a small head vs through the LLM

### Takeaway
No public source measures latency against K directly. The best data are DiffusionDrive (20 samples in one batched pass at 3.8 ms/step for a 60M decoder on an RTX 4090) and GoalFlow (128–256 samples, 10.4 ms at 1 step). Quality saturates near K ≈ 20 for anchored diffusion: PDMS 84.9 / 88.1 / 88.2 at 10 / 20 / 40 samples. Producing plans by autoregressive decoding through the LLM is about 25× slower than a 5-step flow decoder for one plan in Alpamayo-R1 (222 ms vs 8.75 ms), so K plans should be batched through a small head, not decoded through the LLM.

### Cited Findings
- DiffusionDrive N_infer: 10 → 84.9, 20 → 88.1, 40 → 88.2 PDMS, same 60M model [V]; the 2-step plan time of 7.6 ms covers the batched anchors [V]. — [DiffusionDrive](https://arxiv.org/html/2411.15139)
- GoalFlow samples 128/256 trajectories and scores them; 1-step inference 10.4 ms (GPU unstated; the text leaves unclear whether this is per batch or per sample) [V]. — [GoalFlow](https://arxiv.org/html/2503.05689)
- DiffusionDriveV2, 20 raw trajectories: diversity 42.3 → 30.3 after RL, while PDMS@1/@5/@10 rise from 93.5/84.3/75.3 to 94.9/91.1/84.4 [V]; mode selector adds +1.0 (88.1 → 89.1) on DiffusionDrive [V]. — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- Alpamayo-R1 AR trajectory through the LLM: 127 tokens in 222 ms vs 5-step flow decoder 8.75 ms on RTX 6000 Pro Blackwell [V]. — [Alpamayo-R1](https://arxiv.org/html/2511.00088)
- ReflectDrive-2 best-of-6 oracle 94.8 PDMS; multi-sample inference cost not reported; the paper mentions "multi-batch drafting" and batch cache states [V]. — [ReflectDrive-2](https://arxiv.org/html/2605.04647)
- TensorRT engines in the GR00T pipeline are static in batch size (rebuild to change it) [V], so K must be fixed at export if TensorRT is used. — [Isaac-GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)

### Inferences
- AR cost per trajectory token in Alpamayo-R1 is 222/127 ≈ 1.75 ms [D], and the flow decoder is 222/8.75 ≈ 25× faster for one plan [D]. K AR plans would multiply decode work by K, while a head batches K samples against one shared prefix.
- A continuous head with K = 20 samples processes 20 × 8 = 160 waypoint tokens per step against a shared cached context [D]. At this size a desktop GPU is typically launch- or memory-bound, so K = 10–20 should cost much less than K × one sample. DiffusionDrive's 3.8 ms/step for 20 samples fits that picture, but this is an expectation, not a measured K sweep.
- In-LLM masked diffusion with K plans would push K × 16 tokens through every LLM layer at each step. That is much more compute per plan than a small head, and no source measures it.

### Gaps
- No paper reports latency vs number of samples K for any diffusion/flow driving head or VLA action expert (DiffusionDrive and GoalFlow report quality vs K only).
- No measurement of K batched masked-diffusion plans through an LLM.

## Q5. Which few-step methods are compatible with later RL fine-tuning?

### Takeaway
RL fine-tuning at small step counts has been shown for three families:
- **Flow policies**: π_RL with 1/2/4/8 steps (best at 2); ReinFlow at 1 and 4 steps on rectified-flow and shortcut policies.
- **Truncated anchored diffusion**: DiffusionDriveV2, GRPO with 2-step inference, 88.1 → 91.2 PDMS.
- **Masked discrete diffusion**: WAM-Diff with GSPO, 86.6 → 91.0; ReflectDrive-2; DriveFine.

For consistency and other one-step policies, RL evidence comes mostly from state-based actor–critic work (CPQL, Consistency-AC, one-step flow Q-learning), not from VLAs or driving.

### Cited Findings
- **π_RL** (pi0/pi0.5): Flow-Noise (learned per-step noise) or Flow-SDE (ODE → marginal-preserving SDE) makes log-likelihoods tractable [V]. Step ablation (Flow-SDE pi0, LIBERO-Spatial), post-RL success: K = 1 → 88.5, 2 → 97.0, 4 → 94.5, 8 → 86.7; at K = 1 "train success rate plummets", attributed to ODE-to-SDE discretization error [V]. Few-shot SFT → RL: pi0 LIBERO 57.6 → 97.6 (Flow-Noise), pi0.5 77.1 → 98.3; ManiSkill pi0.5 40.1 → 90.9 [V]. — [π_RL](https://arxiv.org/html/2510.25889)
- **ReinFlow**: fine-tunes at "four or even one denoising step": 1 step for Can/Square, 4 for Transport, vs DPPO's 5-step DDIM [V]. Base policies are rectified flow and shortcut models [V]. A learned noise net is added for the likelihood and discarded afterwards, leaving a deterministic flow [V]. Reported gains: +135.36% avg locomotion reward and +40.34% manipulation success [V]. Wall-time savings vs DPPO are quoted inconsistently across sections (82.63% / 62.82% / >50%) [V]. — [ReinFlow](https://arxiv.org/html/2505.22094)
- **Flow-GRPO**: ODE → SDE conversion preserving marginals; "Denoising Reduction" trains with fewer steps and samples with the full schedule; SD3.5-M GenEval 63% → 95% [R, snippets]. — [Flow-GRPO](https://arxiv.org/html/2505.05470v5)
- **DiffusionDriveV2** (truncated diffusion + GRPO): each step is a Gaussian policy N(μθ, η(1−α_t)I), with η = 1 (DDPM-like) for RL training and η = 0 (DDIM) at evaluation; 2-step inference is kept [V]. Ablations: Intra-Anchor GRPO +0.9 PDMS (89.2 → 90.1); Inter-Anchor Truncated GRPO +0.6 (89.5 → 90.1); multiplicative exploration noise beats additive (90.1 vs 89.7) [V]. Claims to be the first GRPO on a truncated diffusion model [R]. — [DiffusionDriveV2](https://arxiv.org/html/2512.07745)
- **WAM-Diff** (masked diffusion + GSPO): 86.6 PDMS without GSPO → 88.9 (G = 2) → 91.0 (G = 3); token log-probs from one-step unmasking; rewards = group-normalized PDMS [V]. — [WAM-Diff](https://arxiv.org/html/2512.11872v1)
- **ReflectDrive-2**: group-relative advantage over G = 6 trajectories (3 goals × 2 drafts) with a discrete-diffusion policy gradient following d2, clipped ratio + KL; terminal reward = PDMS [V]; AutoEdit gain +0.3 PDMS under SFT vs +1.9 after RL [R, abstract]. — [ReflectDrive-2](https://arxiv.org/html/2605.04647); [arXiv abstract](https://arxiv.org/abs/2605.04647)
- **DriveFine**: rule-based online GRPO for the generator, plus hybrid offline/online RL for the refiner [V]. The setup uses group size 10, but the ablation finds G = 8 best [V]. — [DriveFine](https://arxiv.org/html/2602.14577)
- **Consistency / one-step policies with RL** (state-based):
  - CPQL: one-step consistency policy, ~45× faster inference than Diffusion-QL, offline and online RL [R]. — [CPQL](https://arxiv.org/abs/2310.06343)
  - Consistency-AC (ICLR 2024): faster than a diffusion policy in online RL, with higher average performance [R]. — [Consistency-AC](https://arxiv.org/pdf/2309.16984.pdf)
  - One-Step Flow Q-Learning (ICLR 2026): one-step generation in both training and inference without distillation; notes that consistency models train unstably [R]. — [OFQL](https://arxiv.org/html/2508.13904v2)

### Inferences
- For a flow-matching or rectified-flow head, 2–4 steps is the documented RL sweet spot (π_RL best at K = 2; 1 step hurts SDE-based exploration). ReinFlow shows 1-step RL is possible when the exploration noise is learned rather than derived from the SDE. Using the same 2–4 steps in RL and at serving time avoids a train/serve mismatch.
- Anchored truncated diffusion has the most direct driving evidence for RL after few-step SFT (DiffusionDriveV2 on NAVSIM). However, RL reduced sample diversity (42.3 → 30.3), which matters for a learned selector that relies on K diverse plans.
- Masked discrete diffusion has the strongest RL record **on NAVSIM**: WAM-Diff gains +4.4 PDMS [D: 91.0 − 86.6], and RL is what makes ReflectDrive-2's self-editing pay off. Token log-probs are exact under one-step unmasking, which makes the policy gradient straightforward.
- Distilled one-step heads (OneDP, Consistency Policy) have no published VLA or driving RL follow-up found. RL after distillation would most likely need either a learned noise head (ReinFlow-style) or re-training a stochastic policy.

### Gaps
- No RL fine-tuning result found for MeanFlow/MP1, OneDP or shortcut heads inside a VLA, nor for any one-step head on NAVSIM.
- Flow-GRPO, CPQL, Consistency-AC and OFQL numbers here are snippet-level [R]; their full tables were not read in this pass.
