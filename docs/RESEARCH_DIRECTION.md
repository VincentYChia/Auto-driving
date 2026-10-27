# Research direction (living document)

This document is how we stay aligned between sessions. It records **what we are trying to show and why**.
- `docs/ROADMAP.md` covers the how: plans, compute and order of work.

When the direction changes, edit the relevant section and add a dated entry to the **Decision log** at the bottom.

Status markers:
- **[agreed]**: the researcher has confirmed it.
- **[proposed]**: a suggestion from research, not yet confirmed.
- **[open]**: undecided.

Evidence markers:
- **V**: checked against a primary source (code, README, vendor benchmark file).
- **R**: self-reported by a paper or repo, or seen only in an abstract or snippet. Not independently checked.
- **D**: derived by us from cited numbers.

---

## 1. The researcher's thought process (as stated, 2026-10-16)

### Target system [agreed]

```
Sensors ──► perception + tracker ──► object-motion tokens ─┐
front image + ego status + driving command ─────────────────┤
                                                             ▼
   FAST LOOP:  VLM quick pilot (Qwen3.5) ──► masked-diffusion action head ──► action
               (re-plans continuously: produces paths but does not execute all of
                them; keeps refreshing the path)
                                                             │ same inputs
                                                             ▼
   SLOW LOOP:  VLM reasoner (Gemma 4) ──► reasoning (fed into the next fast-loop context)
                                      └─► narration (interpretability / tracking model ability)
```

Constraint: **the whole stack runs on an in-vehicle edge computer.** It is meant as an edge piloting system, not a cheap server-based alternative.

> **Update 2026-10-20 [agreed]:** multi-frame and object-motion context are on the back burner. For now the pilot uses a single frame without previous motion context, so the perception + tracker → motion-token branch is deferred. See §2.2 (Claim 1) and `docs/OBJECT_MOTION.md`.

### Claimed contributions, as originally stated

Novel:
1. Feed the **past motion of surrounding objects** into the model's context **instead of past images**.
2. Use a **diffusion architecture with more discrete action tokens** than previous work (baseline MVLAD-AD uses 256).
3. Run **entirely on the edge**. *Clarified 2026-10-20 [agreed]:* the goal is to **lower compute in general**, not to meet a set compute limit.

Supporting, not novel:

4. Prompt engineering and quantization. Quantization was expected to matter only if we hit time problems.
5. Fine-tuning.
6. Model architecture choices.

### Next step [agreed]
**Collect the dataset and define its schema.** This step underpins claims 1 and 2, so it has to be done carefully.

---

## 2. Research assessment (2026-10-16) [proposed]

### 2.1 Bottom line
- **Each claim, as worded, has direct prior work.** None is defensible alone (§2.2).
- **Together they make one defensible question:**

  > Under a fixed on-vehicle latency budget, how should a masked-diffusion driving VLA spend its tokens?
  > - **Inputs:** tracked object motion instead of visual history.
  > - **Outputs:** more precision per action token at a fixed token count.
  > - Every result is measured on real edge hardware.

  The novelty moves from "we did X" to "we show the trade-off, under realistic conditions, that nobody has measured".
- **Quantization is mandatory for claim 3, not optional.**
  - Orin-class chips only run INT8/INT4.
  - On Thor, NVFP4 prefill is 3.5–5× faster than INT4 (V).
  - Without it the fast loop does not reach 10 Hz.
- **Reference sizes for one Thor-class vehicle computer** (reference points, not a target; see C3):
  - Fast pilot: about 2B (4B at most).
  - On-device reasoner: about 2–4B.
  - Plus a light action head and perception.
  - **Gemma-4-12B is not viable on the device today.** It runs FP16 only in NVIDIA's edge runtime, at about 10 tokens/s (V). Use it offline as a labeller or teacher.
- **NAVSIM v1 cannot validate claim 1.**
  - It is saturated: the best reported score is 95.6, above the human 94.8.
  - A model with no cameras reaches 91.1 (R).
  - Use **navhard two-stage EPDMS (NAVSIM v2) as the primary metric** and report the collision (NC) and time-to-collision (TTC) sub-scores.

### 2.2 Novelty check

#### Claim 1: object motion instead of images
**Closest prior work:**
- **Tracking-Meets-LMMs** (Mar 2025, BMVC'25). Feeds 3D track embeddings into a multimodal LLM, explicitly to avoid processing long video. Evaluated on DriveLM question answering, not planning. No latency numbers. (R)
- **DriveVLM-Dual** (Feb 2024). Feeds tracked objects' past trajectories to a VLM as text, alongside images. 410 ms on Orin-X. (R)
- **ORION** (Mar 2025) and **MindVLA-U1** (May 2026). Replace stacked frames with memory tokens. (R)
- **GPT-Driver / Agent-Driver** (2023). Text object states plus predicted futures. (V, code)
- **Driving on Memory** (Aug 2026): 91.1 PDMS with no cameras. **DriveZero** (Sep 2026): a teacher on ground-truth symbolic inputs scores 95.8, distilled into a 94.8 camera student. Another source gives the student 95.3 PDMS and 57.1 navhard EPDMS. (R)

**Counter-evidence to address:**
- Explicit tracking and motion modules (UniAD 83.4, PARA-Drive 84.0) did not beat single-frame TransFuser (84.0) on NAVSIM v1 (R).
- "Not All History Helps" (Aug 2026): stale history hurts planning (R).

**Verdict:** not novel as stated.

**What nobody has shown:** a controlled study on NAVSIM where track tokens replace multi-frame images at the **same token budget**:
- with a **deployable tracker**, not ground-truth tracks;
- with **edge latency measured end to end**;
- inside a **masked-diffusion VLA**.

**Defensible version (C1):**
> Tracker-derived, ego-motion-compensated motion tokens with geometry-aware embeddings match or beat multi-frame visual history at a fraction of the tokens and latency. The gain is measured on navhard EPDMS and NC/TTC, with ground-truth vs. predicted tracks, plus a robustness curve against tracker quality.

**Update 2026-10-20** (details in `docs/OBJECT_MOTION.md`):
- **Motion can be obtained on NAVSIM itself.** navtrain labels every frame, history included, with boxes, velocity and track IDs at 2 Hz (V). 2 Hz camera trackers are standard on nuScenes (R).
- **Anything scored on navhard must be camera-only.** The 2025 challenge's test frames came without LiDAR (V, maintainers).
- **No leading NAVSIM method feeds in other agents' motion.** History adds +0.1 to +2.7 PDMS on v1 (R). There is no navhard evidence either way.
- **Motion tokens are cheap:** about 0.4–4 ms on Thor, vs about 17–27 ms per extra image frame (D).
- **Proposed gate [proposed]:** an oracle-motion experiment, about 1 week to build, decides whether C1 is worth a tracker.
- **Status [agreed 2026-10-20]:** C1 is on the back burner. The next work is training data for the two small models.

#### Claim 2: more discrete action tokens
**Closest prior work** (all masked/discrete diffusion, all on NAVSIM; V via code or paper text unless noted):
- **WAM-Diff** (Dec 2025): 20,001 numeric tokens, 0.01 m resolution, separate x and y. 91.0 PDMS.
- **WAM-Flow** (Dec 2025, CVPR'26): 20,001 tokens with metric-aligned embeddings. Ablation: text tokens 76.2 → numeric 81.1 → metric-aligned 83.4.
- **DriveFine** (Feb 2026): 4,000 position bins at 0.05 m plus 1,800 heading bins.
- **ReflectDrive** (Sep 2025): separate x and y codebook.
- **E0** (robotics, Nov 2025): swept 128–8,192 bins and saw saturation around 2,048.

MVLAD-AD's 256 joint (x,y) tokens is the **smallest** vocabulary in this family.

**Verdict:** not novel as stated. Vocabulary size has already been scaled by 16–78×.

**What remains open** (none found in our search, which was budget-limited):
- hierarchical **joint-2D** coarse+fine tokens under masked diffusion;
- a **token plus continuous residual** inside a diffusion-LLM driving VLA;
- a controlled **vocabulary-structure scaling study**.

**Defensible version (C2):**
> More precision per action token at a fixed token count and step count. Hierarchical joint-2D tokens with geometry-aware embeddings at both levels, decoded coarse-first (± a continuous residual), beat separate-axis numeric tokens and the flat 256 codebook. Report quantization error → L2 / PDMS / EPDMS → latency.

This ties to claim 3 because **latency scales with tokens × steps, not vocabulary size**: MVLAD-AD uses 10 tokens × 10 steps; WAM-Diff uses 32 tokens × 32 steps.

**Rounding-error floor of joint codebooks** (D):

| Codebook | Mean error at 1 s | Notes |
|---|---|---|
| **256 centres (shipped, measured 2026-10-18, V)** | **0.54 m** (median 0.35, p95 1.39); 0.56 m at 2 s; 0.61 m at 3 s | The real `tokenizer_model.pth` from the official repo, checked against the shipped 1/2/3 s nuScenes waypoints. Centres span x −1.4…54.6 m and y −10.1…11.7 m, so farther waypoints are clipped. Only 52 of 256 centres lie within 10 m of the car |
| 1,024 centres (shipped) | ≈0.05 m | |
| 2,048 centres (shipped) | ≈0.04 m | |
| Separate-axis designs | ≤0.005–0.025 m per coordinate | |

**Drop option E (whole-trajectory vocabularies) as a standalone idea.** With one token per trajectory, masked diffusion reduces to classification. That is Hydra-MDP (8,192), GTRS (16,384) and SparseDriveV2 (up to 16,384) (V). Keep E only as a coarse anchor level inside C2.

#### Claim 3: entirely on the edge
**Closest prior work:**
- Li Auto VLA: ~4B mixture-of-experts at ~10 Hz on Thor-U, in production (R, media).
- Li Auto: 2.2B VLM at ~300 ms on Orin-X (R).
- **AUGUstus:** Qwen3.5-2B, 94.6 PDMS, 12 Hz on a real vehicle (V, README; chip not stated).
- **ReflectDrive-2:** masked discrete diffusion, ~30 ms on Thor (R).
- **GR00T N1.7:** VLM + diffusion action head, 40 ms (25 Hz) on Jetson AGX Thor (V).

**Verdict:** not novel by itself. On-vehicle VLAs exist in production, and a masked-diffusion driving planner has already been timed on Thor.

**What is still open:** the **full fast/slow stack** (perception + tracker + pilot + masked-diffusion head + asynchronous reasoner) running concurrently on **one** edge SoC, with end-to-end latency/accuracy Pareto curves. Ideally also an Orin-class data point, since Orin-X is what most current cars have.

**Defensible version (C3), updated 2026-10-20:**
- **[agreed]** Lower the compute of the whole stack as far as possible at equal driving quality. Compute here means latency, memory and energy per decision. There is no fixed budget.
- **[proposed]** Report accuracy-vs-compute curves. Measure on physical edge hardware (Jetson AGX Thor; Orin as a second point), used as reference platforms rather than targets.

#### The closest rivals: performance, architecture, training data (checked 2026-10-22)
A collaborator named these four as the strongest work in this space. All scores are R. Paper details came from third-party copies, because arXiv was blocked.

| Study | navtest PDMS / EPDMS (single sample) | Model | Action output (tokens × steps) | Extra at test time | Training data | Explanation output | Latency | Released |
|---|---|---|---|---|---|---|---|---|
| **WAM-Diff** (Dec 2025) | 91.0 / 89.7 | LLaDA-V 8B + SigLIP2; 64-expert LoRA-MoE | 20,001 numeric bins (0.01 m), x and y separate: 32 tokens × 32 steps | Classifier-free guidance; reverse-causal order | nuPlan 668k trajectories (split not stated) + 800k VQA + navtrain 103k; GSPO RL | None | Not reported | Inference + SFT code; weights (V) |
| **DriveFine** (Feb 2026; ECCV'26) | 90.7 / 89.7 (fixed evaluator). navhard stage 1 / 2: 74.4 / 41.0 | LaViDa (LLaDA-8B), ~9B with its refiner | 4,000 position bins + 1,800 heading bins; 12 steps + 1 refinement pass | Refinement expert | ReCogDrive QA + navtrain; GRPO + refiner RL | None | 207 ms at 4 steps (90.5 PDMS); hardware not stated | README only (V) |
| **ReflectDrive** (Sep 2025; ICLR'26) | 84.8 model alone; 91.1 with reflection | LLaDA-V ~8B | Per-axis codebook: 16 tokens | Goal candidates + rule-based safety scorer (constant-velocity agents) + inpainting, 1–3 rounds | Not found; no RL | None | Not reported | README + MIT licence (V) |
| **ReflectDrive-2** (May 2026) | 91.0 | Proprietary 0.7B masked-diffusion LM + 0.1B ViT | Goal token + 16 coordinate tokens + 1 AutoEdit pass | Same model edits its own draft | navtrain only: perturbation-recovery SFT + RL | None | ~31.8 ms on "Thor", averaged over full and lite frames | Nothing |

**Also: WAM-Diff2** (Aug 2026), the same team on Qwen3-VL-2B. 88.3 PDMS / 88.6 EPDMS single sample; 91.1 / 90.7 with score-based selection (R). Apache-2.0 code (V). It is the open ~2B rival.

**What this leaves open (D):**
- **None of the five outputs an explanation.**
- **Latency** is unreported, or comes from closed weights with unclear hardware.
- **Headline numbers stack extras:** guidance, scorers and oracle best-of-N. Report score per forward pass.
- **RL seems needed to pass ~90.**
- **Image tokens (~2.2k into an 8B model) dominate the cost,** not action tokens.

### 2.3 Edge compute: what "edge" means, and reference budgets

> **Update 2026-10-20 [agreed]:** there is no fixed compute target; the goal is to lower compute in general (C3). The tiers, budgets and size limits below are reference points for measurement, not constraints.

**Define edge by memory bandwidth and supported number formats, not TOPS.** Generating each token reads every weight once, so it is bandwidth-bound. Prompt processing (prefill) and image encoding are compute-bound.

| Tier | In cars | Dev proxy we would measure on | Key specs |
|---|---|---|---|
| Orin-class (current fleet) | DRIVE Orin-X: 254 INT8 TOPS, ~205 GB/s; Li Auto, NIO, … | Jetson AGX Orin 64 GB: 275 sparse INT8 TOPS, 204.8 GB/s, 15–60 W | **No FP8/FP4.** NVIDIA's edge runtime (TensorRT Edge-LLM): "Jetson Orin does not run FP8 or FP4 model engines" (V) |
| Thor-class (current generation) | DRIVE Thor: 1,000 INT8 TOPS / 2,000 FP4 TFLOPS sparse, 273 GB/s. Thor-U: 700 TOPS (Li Auto i8) | Jetson AGX Thor T5000: 2,070 FP4 TFLOPS sparse, 128 GB, 273 GB/s, 40–130 W | NVFP4 / FP8. Edge-LLM officially supports DRIVE Thor on DriveOS 7.2 (V) |

Other automotive chips (Horizon J6P, Qualcomm Ride Elite, XPeng Turing, NIO NX9031, Tesla AI5) lack open toolchains for this kind of research. They are context only.

**Measured on-device speed, one request at a time** (TensorRT Edge-LLM v0.10.0, JetPack 7.2) (V):
- Single-image column: image-encoder time + prefill for a ~296-token request (≈265 image tokens).
- Thor runs the language model in NVFP4; Orin in INT4-AWQ (4-bit weights, FP16 maths).

| Model | Thor: single image | Thor: 2,048-token prefill | Thor: decode | Orin: single image | Orin: 2,048-token prefill | Orin: decode |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | 10.9 + 18.2 ms | 43 ms | 122 tok/s (8.2 ms/step) | 34.3 + 106.7 ms | 721 ms | 82 tok/s (12 ms) |
| Qwen3.5-4B | 10.9 + 31.6 ms | 96 ms | 68 tok/s (15 ms) | 34.3 + 197.5 ms | 1,370 ms | 46 tok/s (22 ms) |
| Qwen3.5-9B | 14.5 + 44.0 ms | 135 ms | 40 tok/s (25 ms) | 49.7 + 275.3 ms | 2,017 ms | 28 tok/s (36 ms) |
| Qwen3-VL-2B | 11.4 + 12.7 ms | 28 ms | 136 tok/s | 36.2 + 45.7 ms | 298 ms | 89 tok/s |
| Gemma-4-E4B | 16.6 + 36.1 ms | – | 59 tok/s (96 with multi-token prediction) | not benchmarked | | |
| Gemma-4-12B | FP16 only: 1.1 + 146.8 ms | 623 ms | **10 tok/s** (25 with multi-token prediction) | not benchmarked | | |

**Measured VLM + diffusion action head on Jetson AGX Thor** (V): GR00T N1.7.
- Configuration: a 2B Qwen3-VL-type backbone (16 of its 28 layers kept), a 32-layer diffusion-transformer head, 4 denoising steps, one camera.
- Timing: backbone 13.6 ms + head 17.2 ms = **~40 ms (25 Hz)** in mixed NVFP4. The head's attention stays FP8. "No measurable loss in task success."
- How they got there:
  - the head's cross-attention keys and values are computed **once** and reused across denoising steps;
  - timestep constants are precomputed;
  - the whole head runs as one CUDA graph.

**Budgets** (D, from the rows above):

| Component | Thor-class (primary target) | Orin-class (stretch) |
|---|---|---|
| Fast pilot VLM | **Qwen3.5-2B NVFP4** with ≤ ~500 input tokens: ~30–45 ms. 4B: ~45–65 ms. 9B only with a separate head and few tokens, and it takes over the GPU | ≤2B INT4 with ≤ ~256 visual tokens: ~80–140 ms. The VLM cannot run every 100 ms cycle |
| Action head | Separate light masked-diffusion head reading cached VLM states, ~5–20 ms (GR00T's 4-step head: 17 ms). Running diffusion inside the LLM costs ~8 ms **per step** at 2B | ~12 ms per step for diffusion inside the LLM, so a separate head is mandatory |
| Reasoner | **Gemma-4-E2B/E4B NVFP4**: about 40–60 output tokens per 1 s cycle, *if it had the GPU to itself*. Sharing with the pilot cuts this further | Needs a second chip (DriveVLM-Dual used 2× Orin-X) |
| Perception + tracker | Camera-only for navhard, which has no LiDAR on its test frames (V). Camera 3D trackers on Orin: PETRv2 55.8 ms, UniAD-tiny 39 ms (V). With LiDAR (v1 or deployment): CenterPoint 35.7–40 ms, PointPillars 6.8 ms on Orin (V). Thor not measured | Same, slower |
| Memory | Not the constraint. Weights at 4-bit: 2B ≈ 1.1 GB, 4B ≈ 2.2 GB, 9B ≈ 5 GB (D) | Same |

- **Reference size on one Thor-class SoC:** pilot ~2B (4B at most) + reasoner ~2–4B + head + perception, roughly 6–7B parameters in 4-bit in total.
- **Industry cross-check:** Li Auto ships ~4B MoE at 10 Hz on Thor-U. NVIDIA positions 10B Alpamayo as a teacher and distils it for DRIVE Thor.
- **Visual tokens are the biggest lever.** Each extra 1,000 visual tokens adds ~60–125 ms on Thor and 0.3–1.2 s on Orin (D).
  - MVLAD-AD's tiled image mode (~3.6k tokens per frame, from reading the code) is not viable on the edge.
  - Use one ≤512-token view, or pruning: cutting 75–90% of visual tokens gives 3.7–6.4× faster prefill at ≤3–6% accuracy loss (R).

### 2.4 Architecture implications (supporting) [proposed]
1. **Separate light masked-diffusion head conditioned on the VLM's cached states.** This replaces MVLAD-AD's design, where every denoising step is a full LLM pass. It is the GR00T / Alpamayo action-expert pattern, but with discrete masked diffusion.
   - We found no discrete masked-diffusion action head sitting on a standard left-to-right VLM. Verify before claiming it.
   - It also makes base models swappable (roadmap plan 1).
2. **Multi-rate loop.**
   - The VLM refreshes its features at ~5–10 Hz; the head re-plans at 10–20 Hz from the newest ego state and tracker tokens. This is the user's "keep refreshing" design.
   - Prior work: "Think at 5 Hz, Act at 20 Hz" (Jul 2026: a frozen 7B VLM refreshes a cache at 5 Hz, a light expert acts every tick, trained on randomly aged caches; CARLA route completion 37 → 94% vs frame-skipping) (R). Also ASSCG (60% lower latency, 91.4 PDMS) and Slow Brain/Fast Planner (R).
3. **Reasoner output to the pilot** as a short decision token or latent slots, not free text.
   - Train with random staleness and random null-drop.
   - Trigger reasoning on uncertainty, not at a fixed rate: DualDriveVLA calls its VLM in 15% of scenes and reaches 91.0 PDMS at 3.2× throughput (R).
4. **No chain-of-thought in the pilot.** AUGUstus generates the action before any reasoning: Qwen3.5-2B, 94.0 PDMS with one decoding step, 73.6 ms (V). Each reasoning token costs ~8 ms (Thor) or ~12 ms (Orin) at 2B.
5. **Refreshing the plan by inpainting.**
   - Keep the tokens of the previous plan that execute during compute latency (Physical Intelligence's real-time chunking) and regenerate the rest.
   - ReflectDrive raised PDMS 84.8 → 91.1 with candidate selection plus safety-guided inpainting, both driven by external rule-based scorers (R; corrected 2026-10-22).
   - ReflectDrive-2 alternates a full and a lite edit pass (~41 / 19 ms on Thor) (R).
   - Measure temporal consistency with MomAD's TPC metric.

### 2.5 Supporting work packages [proposed]

**Prompt engineering**

Established:
- Ego-state text carries most of the nuScenes open-loop signal. AD-MLP reaches 0.29 m average L2 with no images.
- In DriveBench, removing images barely changes planning scores.
- So every prompt ablation must be run **with and without ego status** and evaluated on NAVSIM v2.

The output head matters more than the prompt: in ReCogDrive (2B), text waypoints 84.1 → diffusion planner 86.5 → plus RL 90.8 PDMS (V).

Open territory (no controlled driving studies found): coordinate format and precision, system prompts, few-shot examples, structured vs. free-text scene fields. A small, honest supporting study is possible here.

MVLAD-AD's prompt problems to fix:
- It says "sequence of images" and "rationale" but provides one image and no rationale.
- Training and inference use different instruction wording.
- It spends ~200 tokens on 2-decimal history coordinates.
- It omits velocity, acceleration and the driving command (WAM-Diff's prompt includes them).

**Quantization** (required)

| Target | Format |
|---|---|
| Thor | NVFP4 backbone, FP8 for sensitive layers, head attention ≥ FP8 |
| Orin | INT4-AWQ (W4A16) or INT8 |

Evidence:
- FP8 / W8A8 is near-lossless for VLMs, VLAs and diffusion LMs (R).
- W4A16 on the LLM: +3.5–4.7% L2 (OmniDrive deployment, V).
- Quantizing the diffusion-transformer attention dropped LIBERO success 97.6 → 84.0 (QuantVLA, R).
- Diffusion LMs accumulate quantization error across denoising steps. Calibrate across timesteps and mask ratios (DLLMQuant recovers >10 points at W4A4, R).

**Fine-tuning**

SFT → RL with the PDM score as reward (GRPO/GSPO) is standard and adds about 3–4 PDMS (R; corrected 2026-10-22):
- DriveFine: SFT 86.7 → GRPO 89.6 → refiner RL 90.8. Its 91.8 comes from an extra trained scorer.
- WAM-Diff: GSPO adds +4.4.
- ReCogDrive: 86.5 → 90.8.

Distillation from a privileged teacher fits C1 directly: train a teacher on ground-truth tracks, then a student on tracker tokens (DriveZero pattern).

### 2.6 Evaluation protocol implications [proposed]
- **Objective [agreed 2026-10-22]:** as a starting point, the highest NAVSIM score under constrained compute. Inputs and outputs are tailored to NAVSIM, including the output rate (Hz) and smoothing. How NAVSIM scores a plan, and what follows from it: `docs/NAVSIM_TAILORING.md`.
- **Primary: navhard two-stage EPDMS (v2).**
  - Best reported non-VLM: 58.1 (FPD-Drive-Pro). Best VLA: 47.0 (DriveVLA-M0) (R).
  - The ~11-point gap is where object-motion context could plausibly help.
- **Secondary:** navtest PDMS (v1), for comparability; nuScenes L2, for comparability with MVLAD-AD.
- **Always report:** NC and TTC sub-scores; token counts; on-device latency per component; Pareto curves at matched token budgets.
- **Temporal consistency and refresh** cannot be measured in NAVSIM: it runs at 2 Hz and its pseudo-simulation is non-sequential. Use TPC on consecutive frames offline, and a closed-loop simulator (Bench2Drive / nuPlan) if those become claims.
- **On-device latency** must be measured on a physical Jetson. Anvil's A100/H100 GPUs are not a proxy.

### 2.7 Ideas worth borrowing (ranked)
1. Action-first pilot, no chain-of-thought (AUGUstus).
2. Multi-rate cached-feature fast/slow loop, trained on stale features (Think@5Hz/Act@20Hz, ASSCG).
3. Masked-diffusion self-editing and inpainting for refresh and safety (ReflectDrive / -2, real-time chunking).
4. GR00T's head engineering: cross-attention K/V hoisting, precomputed timesteps, CUDA graphs, mixed NVFP4/FP8.
5. RL against the PDM score (GRPO), after SFT.
6. Teacher on symbolic ground-truth object tokens → student on tracker tokens (DriveZero).
7. Metric-aligned action-token embeddings (WAM-Flow) and decoding order (WAM-Diff: reverse-causal 91.0 vs causal 88.9) for C2.
8. Generate-and-score with sub-score heads plus test-time search (DrivoR / TOAD: +1.7 navhard EPDMS); synthetic data (SimScale: +5 to +8.6 navhard EPDMS) (R).

### 2.8 What the dataset and schema must support (next step)

> **Update 2026-10-18:** training data is now its own deliberate workstream. `docs/training_data/README.md` is authoritative for its requirements, process and decisions; this section is the original input to it.
**Source: NAVSIM navtrain** (OpenScene / nuPlan logs). Verified in `navsim/common/dataclasses.py` (V):

| Field | Content |
|---|---|
| Object boxes | x, y, z, length, width, height, heading. Auto-labels from nuPlan's offline perception, not human annotation (V, `nuplan_schema.md`) |
| Object metadata | class names, 3D velocity (a network estimate), instance and track tokens (consistent across frames) |
| Ego | pose, velocity, acceleration, driving command |
| Timing | 2 Hz; default scenes are 4 history + 10 future frames |
| Planning target | 8 poses over 4 s |
| Test-time history | 1.5 s span |

**Other sources:** tabulated in `docs/training_data/DATASETS.md` (2026-10-20). Proposed next source: the raw nuPlan DBs for the same logs (20 Hz). nuScenes plus Nu-X remain an option for MVLAD-AD comparability.

**Requirements:**
- Store **raw continuous** ego and object states. Keep tokenized views as versioned derivatives, so the C2 variants can be generated from the same data.
- Store motion inputs from **both** ground-truth tracks and the **deployable tracker** run on the same sensors. Record provenance and tracker-quality metrics, for C1 and its robustness curve.
- Keep explicit frame and coordinate conventions, ego-motion compensation and timestamps.
- Store references for the multi-frame image baseline at matched token budgets, plus token-count accounting.
- Released reasoning text for navtrain exists only as model-written, hindsight QA (ReCogDrive, 752k QA; V). Generate our own labels offline with a large model (e.g. Gemma-4-31B), version the model and prompt, and compare against ReCogDrive in the P5 pilot.
- **Never train on navtest / navhard / warmup** (NAVSIM rule).

### 2.9 Risks and blockers
- **MVLAD-AD weights:** resolved 2026-10-18. The fork's Git LFS store returns 404, but the official repo's media URLs serve every object, with sizes and sha256 matching the pointers (V). `scripts/download_mvlad_weights.sh` fetches and verifies them.
- **Claim 3 needs physical hardware:** a Jetson AGX Thor dev kit (DRIVE Thor proxy), ideally plus a Jetson AGX Orin.
- **Data scale gap:** AUGUstus's 94.6 PDMS used 512 H800 GPUs, 318K h of video and 513B tokens (V). Our results must be framed as controlled comparisons, not leaderboard chasing.
- Several NAVSIM and architecture numbers are self-reported (R), and the official leaderboards were unreachable from this environment.
- **No LiDAR on navhard / v2 test frames:** for the 2025 challenge, none was given for "any test frames (real or synthetic)" (V, NAVSIM issue #63). Any perception scored on navhard must be camera-only.

---

## 3. Open decisions for the researcher [open]
1. Accept the reframed claims (C1–C3 under the token-budget thesis)?
2. ~~Fix "the edge device" as one Jetson AGX Thor?~~ **Answered 2026-10-20:** no fixed compute limit; lower compute in general. Which hardware to measure on stays open (item 6).
3. On-device reasoner = **Gemma-4-E2B/E4B**, with 12B/31B used only offline as labeller or teacher?
4. Action head = **separate light masked-diffusion head** on cached VLM states, instead of MVLAD-style diffusion inside the LLM?
5. Primary metric = **navhard EPDMS**?
6. Hardware access: can we get a Jetson AGX Thor?
7. ~~Single-frame pilot without motion context?~~ **Answered 2026-10-20:** yes for now. Multi-frame and motion data are on the back burner; the oracle gate (`docs/OBJECT_MOTION.md` §5) is the way back in.

## Key sources (accessed 2026-10-16)
- NVIDIA TensorRT Edge-LLM benchmarks (v0.10.0) and support matrix: https://github.com/NVIDIA/TensorRT-Edge-LLM (`docs/source/user_guide/performance/performance-benchmarks.md`, `.../getting_started/support-matrix.md`)
- GR00T N1.7 on Jetson Thor: https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/vla/groot_n17_on_thor.md
- AUGUstus: https://github.com/haha-yuki-haha/AUGUstus · ReflectDrive: https://github.com/pixeli99/ReflectDrive · WAM-Diff: https://github.com/fudan-generative-vision/WAM-Diff · WAM-Flow: https://github.com/fudan-generative-vision/WAM-Flow · DriveFine: https://arxiv.org/abs/2602.14577 · ReflectDrive-2: https://arxiv.org/abs/2605.04647 · E0: https://arxiv.org/abs/2511.21542
- Tracking-Meets-LMMs: https://github.com/mbzuai-oryx/TrackingMeetsLMM · DriveVLM: https://arxiv.org/abs/2402.12289 · ORION: https://arxiv.org/abs/2503.19755 · Driving on Memory: https://arxiv.org/abs/2608.31029 · DriveZero: https://github.com/XiaomiAutoL3/DriveZero
- Think at 5 Hz, Act at 20 Hz: https://arxiv.org/abs/2607.15621 · ASSCG: https://arxiv.org/abs/2606.25509 · DualDriveVLA: https://arxiv.org/abs/2602.10719 · Real-time chunking: https://www.physicalintelligence.company/research/real_time_chunking · MomAD: https://arxiv.org/abs/2503.03125
- NAVSIM SOTA: FPD-Drive https://github.com/CountNemoChan/FPD-Drive · DriveVLA-M0 https://github.com/ZebinX/DriveVLA-M0 · NAVSIM dataclasses https://github.com/autonomousvision/navsim/blob/main/navsim/common/dataclasses.py
- Quantization: OmniDrive deploy https://github.com/NVlabs/OmniDrive/blob/main/deploy/README.md · QuantVLA https://arxiv.org/abs/2602.20309 · DLLMQuant https://arxiv.org/abs/2508.14090 · Quantization meets dLLMs https://arxiv.org/abs/2508.14896
- Token pruning: FastDriveVLA https://arxiv.org/abs/2507.23318 · Prune2Drive https://arxiv.org/abs/2508.13305 · ETA-VLA https://arxiv.org/abs/2603.25766
- Prompt evidence: AD-MLP https://github.com/E2E-AD/AD-MLP · Is Ego Status All You Need https://arxiv.org/abs/2312.03031 · DriveBench https://github.com/drive-bench/toolkit · ReCogDrive https://github.com/xiaomi-research/recogdrive
- Edge perception: CUDA-CenterPoint https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-CenterPoint · CUDA-PointPillars https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars

---

## Decision log

| Date | Decision / change | Status |
|---|---|---|
| 2026-10-16 | Initial direction recorded (system, claims 1–3, supporting 4–6, next step = dataset + schema). Qwen3.5 = fast pilot, Gemma 4 = slow reasoner. Prompt engineering and quantization added as work packages. | agreed |
| 2026-10-16 | Research assessment added (§2). Proposed: reframe claims as C1–C3 under the token-budget thesis; quantization mandatory; on-device size limits; drop whole-trajectory vocabularies as standalone; navhard EPDMS primary; separate light diffusion head; open decisions listed in §3. | proposed |
| 2026-10-18 | Researcher chose the first work items: (1) a quick MVLAD-AD baseline on NAVSIM; (2) start the training data for the Qwen pilot and Gemma reasoner. The §3 decisions are still unanswered. | agreed |
| 2026-10-18 | Built the NAVSIM bridge (`navsim_tools/`, `inference/predict_navsim.py`), the data pipeline (`data_pipeline/`, schema v0.1 in `docs/DATASET_SCHEMA.md`), Anvil job scripts (`scripts/anvil/`) and the runbook (`docs/RUNBOOK_NAVSIM.md`). The Python tests pass against the real NAVSIM v1.1 and v2 devkits on synthetic logs. GPU inference and the Slurm scripts are not yet run on Anvil. | done |
| 2026-10-18 | Measured MVLAD-AD's shipped 256-token codebook: 0.54 m mean rounding error at 1 s; centres stop at 54.6 m ahead (§2.2). | done |
| 2026-10-18 | Researcher: separate the two workstreams. (A) NAVSIM evaluation: deterministic, lower priority. (B) Training data: the foundation for months of research, so do it deliberately. The data code is now **prototype v0.1** (not for experiments); the design process, prototype limitations and open decisions D1–D11 are in `docs/training_data/README.md`. Code, tests, jobs and docs split by workstream (`navsim_eval/`, `data_pipeline/`, shared `navsim_tools/`). | agreed |
| 2026-10-20 | Surveyed other open datasets (now `docs/training_data/DATASETS.md`). Found: NAVSIM's object boxes are nuPlan offline-perception auto-labels (V); released navtrain reasoning text exists (ReCogDrive, model-written; V). Proposed: the raw nuPlan DBs for the same logs as the next source (D6); ReCogDrive as a comparison arm in the P5 pilot (D9). | proposed |
| 2026-10-20 | Researcher is leaning (not final) towards a single-frame pilot without previous motion context, if feasible, and accepts that this weakens the claims. Researched ways to obtain object motion (`docs/OBJECT_MOTION.md`) and tabulated every surveyed dataset (`docs/training_data/DATASETS.md`). Proposed: keep C1 open behind an oracle-motion gate; any tracker scored on navhard must be camera-only. | open |
| 2026-10-20 | Researcher: C3's idea stands, but the goal is to **lower compute in general**, not to meet a set compute limit. Edge tiers and budgets become reference points. | agreed |
| 2026-10-20 | Researcher: multi-frame and object-motion data go on the back burner (C1 deferred). Next: training data for the two small models (fast pilot, reasoner). P1 requirements drafted in `docs/training_data/P1_REQUIREMENTS.md`. | agreed |
| 2026-10-20 | Researcher: single-frame training data should come from several datasets, for quality and quantity, not navtrain alone. Proposed: navtrain stays the target for fine-tuning and evaluation; shortlist, mixing rules and training stages in `docs/training_data/P1_REQUIREMENTS.md` §3. | agreed (principle) |
| 2026-10-22 | Researcher: the starting objective is the highest NAVSIM score under constrained compute, with inputs and outputs tailored to NAVSIM, including output rate (Hz) and smoothing. A collaborator's suggestions (focus on efficiency and explainability; smoothing post-processing; frequency-domain trajectories) are recorded as context only. Analysis and proposed experiments: `docs/NAVSIM_TAILORING.md`. | agreed (objective) |
| 2026-10-22 | Researcher: planning comes first; build less aggressively. For now, gather frames with their recorded driving pattern, plus reasoning text where it exists, and write our own where it doesn't. The raw data of each candidate source and the path to training samples are laid out in `docs/training_data/DATA_ATLAS.md`. | agreed |
| 2026-10-27 | Researcher, for the first pretraining build: OpenScene trainval only, until access to other sources is confirmed; **native images, not rectified** (the aim is a general-purpose model that learns every camera, which the researcher sees as a stronger paper claim; rectify only if results are poor); front camera only; a small first test of 10k–100k samples. Execution plan with gates: `docs/training_data/PRETRAIN_V1_PLAN.md`. | agreed |
