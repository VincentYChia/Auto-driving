# Research roadmap: extending MVLAD-AD

> **Source of truth for what we claim and why:** `docs/RESEARCH_DIRECTION.md` covers the claims, the novelty check, the edge-compute budget and open decisions. This file is the work plan; where the two disagree, `RESEARCH_DIRECTION.md` wins.

What stays the same throughout:
- **Training data:** the nuScenes planning and Nu-X JSONs, kept as pre-training and auxiliary data.
- **The motion head:** a masked-diffusion head with restricted outputs, so it can only produce valid path tokens.
- **Reasoning text:** optional, and allowed to stay unrestricted.

Compute figures marked **(est.)** are back-of-envelope estimates, not measurements. Check them on the `mini` / `warmup` splits before committing to a budget.

---

## Background: what the repo does today

- **Base model:** LLaDA-V (8B masked-diffusion language model) with a SigLIP2-so400m image encoder.
- **Fine-tuning:** LoRA on all of the language model's linear layers, plus the image-to-language projector. The image encoder is frozen.
- **Input:** one front-camera image, plus 5 s of past ego waypoints written as text.
- **Path tokens:** 10 waypoints × 1 token each, drawn from a K-means codebook.
  - The shipped `unused_token_ids.npy` has **256** slots, and the tokenizer weight file's size matches 256 centres, not 2048.
  - Each waypoint embedding is trained offline (reconstruction + geometry + contrastive losses). It is written into rarely used vocabulary rows of both the input embeddings and the output layer, and stays **frozen** during fine-tuning.
- **Loss:** LLaDA masked-diffusion loss over the answer only. At path positions, any non-path token is blocked.
- **Training stages:**
  - Stage 1: path only, lr 1e-5.
  - Stage 2: path + Nu-X explanation, lr 5e-4, starting from the Stage-1 LoRA.
- **Inference:** decode the path tokens first, then fill in the explanation (`generate_fix_mask`).

Known problems to fix before starting:
- **Broken training entry point:** `train/scripts/stage*.sh` call `llava.train.train_mem`, but that module doesn't exist. Only `train_mem_{curr,mask,sc}` do.
- **Hardcoded 10-token horizon:** `inference/run_inference.py:55,68,79`, `train/llava/model/language_model/modeling_llada.py:1446,1763,2016`, `train/llava/train/train.py:854`.
- **Hardcoded LLaDA IDs:** mask ID `126336`; blocked special tokens `126080/126081/126346/126347`; the `<= 126080` assert in `tokenizer/find_unused_tokens.py`.
- **Rank mismatch:** the shipped adapter config says LoRA rank 64 / alpha 16, but the scripts say rank 256.
- **Dependencies:** `train/requirements.txt` pins `transformers==4.40.0.dev0`, which isn't installable from pip, and is missing `nuscenes-devkit`, `python-dotenv` and `pyquaternion`. The weights' configs record `transformers` 4.39.3.
- **Weights are not in this fork's LFS store** (404; the files are pointer stubs). Fixed by `scripts/download_mvlad_weights.sh`, which fetches them from the official repo `lan-qing/MVLAD-AD` and verifies each sha256 against its pointer.
- **Train/inference prompt mismatch (planning task):**
  - `train.py` rewrites the instruction to "...in the action token format:"; `run_inference.py` does not.
  - Fix: apply the same replacement at inference.
- **Image tokens:** the shipped config (`anyres_max_4`) gives roughly **3.6k image tokens** for a 1600×900 front frame (from reading `llava_arch.py` ~L456–490; not measured). This is not viable on edge hardware (see `RESEARCH_DIRECTION.md` §2.3).

---

## Plan

### Workstreams (separated 2026-10-18)

| | A. NAVSIM evaluation | B. Training data |
|---|---|---|
| Purpose | Score models on NAVSIM; first, a quick MVLAD-AD baseline | The foundation for fine-tuning the pilot (Qwen) and the reasoner (Gemma) |
| Pace | Deterministic; run when convenient | **Deliberate**: design and review before building |
| Code | `navsim_eval/` | `data_pipeline/` (prototype v0.1 only) |
| Jobs | `scripts/anvil/navsim_eval/` | `scripts/anvil/data_pipeline/` (guarded: smoke tests only) |
| Docs | `docs/navsim_eval/RUNBOOK.md` | `docs/training_data/README.md` (process, decisions) |
| Tests | `tests/eval/` | `tests/data/` |
| Status | Built; tests pass against the NAVSIM v1.1 and v2 devkits; ready to run on Anvil | **Design phase**: next is P1 requirements, then P2 real-data audit |

- **Shared by both:** read-only NAVSIM access (`navsim_tools/`, `tests/shared/`) and the Anvil config and downloads (`scripts/anvil/*.sh`).
- **No cross-imports:** neither workstream imports the other.

### Step 0: Groundwork (do first; everything else depends on it)
- Make the path length a config value (`num_action_tokens`) instead of hardcoding 10.
- Fix the training entry point and build a working environment (lockfile).
- Reproduce the planning L2 result on a small nuScenes val subset to confirm the pipeline works.
- Pull out the masked-diffusion loss (`modeling_llada.py` ~L2116–2222) and the restricted decoding into a self-contained module, so they don't depend on LLaDA internals. Steps 1–3 all need this.

### 1. Swapping in different base models

Three tiers, cheapest first:

| Tier | Example | What changes | Effort (est.) |
|---|---|---|---|
| a. Same family | Other LLaDA-V checkpoints | `LLADA_V_PATH`; retrain the waypoint embeddings if the hidden size differs | Days |
| b. Another masked-diffusion vision-language model | LaViDa, Dimple, MMaDA (check each one's code first) | Wrapper class, mask ID, special tokens, chat template, cache hooks, decoding loop | 2–4 weeks per model |
| c. Any vision-language model + a **separate masked-diffusion path head** | Qwen2.5-VL-3B/7B, InternVL3-2B/8B | The backbone becomes a frozen or LoRA'd encoder; a small transformer runs masked diffusion over path tokens using its hidden states | 4–6 weeks once; then swapping backbones is mostly config |

- **Recommendation: tier c.** Base models become interchangeable, and changing the output space doesn't touch the language model's vocabulary.
- **Cost:** it gives up the paper's "one unified diffusion over text and path" claim. Keep tier a as the baseline to compare against.

### 2. Expanding the output space

Shortlist, from the earlier analysis:
- **B. Coarse + fine tokens per waypoint:** e.g. 256 coarse × 256 fine, giving about 65k effective positions per point.
  - Decode coarse tokens first, then fine ones, the same "most important first" scheme already used for path-then-text.
  - That makes the path 2 × N_waypoints tokens.
- **D. Discrete token + continuous correction:** a small regression head (in the style of `EmbeddingDecoder`) predicts an (dx, dy) offset from the hidden state at each path position.
  - Trained with an L1 loss alongside the diffusion loss.
  - The discrete token keeps the output restricted; the head makes it effectively continuous.
- **E. Whole-path vocabulary:** about 4k–8k candidate paths, built by K-means over whole trajectories (as in VADv2 / Hydra-MDP); one token = one path.
  - Fits NAVSIM's scoring well and is smooth by construction, but has no per-waypoint fill-in.
  - Variant worth testing: E for the coarse path, then D for per-waypoint corrections.
  - **Update 2026-10-16: drop E as a standalone method.** One token per path turns masked diffusion into classification, which Hydra-MDP, GTRS and SparseDriveV2 already do at 8k–16k paths. Keep it only as a coarse anchor level.

**Prior work already scaled vocabulary size:**
- WAM-Diff: 20,001 numeric tokens.
- DriveFine: 4,000 position bins + 1,800 heading bins.
- WAM-Flow: 20,001 metric-aligned tokens.

So this plan is now framed as **precision per token at a fixed token count** (B, and D inside the denoiser), plus a controlled scaling study. See claim C2 in `RESEARCH_DIRECTION.md` §2.2.

What to compare: L2 at 1/2/3 s on nuScenes; PDMS / EPDMS on NAVSIM (see 3); codebook usage (dead tokens); decoding latency.

Also try per-timestep codebooks. `tokenizer/points_xy_{1,2,3}s.npy` already exist, but a single shared codebook is used today.

Also compare smoother parameterisations with fewer tokens: a spline through fewer poses, frequency-domain (DCT) coefficients, or control-space outputs. Score each one's ceiling offline first (`docs/NAVSIM_TAILORING.md` §4).

### 3. Getting it working on NAVSIM, then fine-tuning

**Update 2026-10-22:** the objective is the highest NAVSIM score under constrained compute [agreed]. How NAVSIM scores a plan, and what that means for horizon, output rate, smoothing and frame-to-frame consistency: `docs/NAVSIM_TAILORING.md`.

1. **Agent wrapper:** implement a NAVSIM agent that turns its inputs into a prompt (front camera, ego status, driving command) and outputs a `Trajectory` with 8 poses at 0.5 s (4 s).
   - Heading = direction between consecutive waypoints.
   - Run the model as a separate process or local server, because NAVSIM's environment and this repo's dependencies conflict.
2. **Zero-shot baseline** with the shipped nuScenes weights. Expect a low score; this mainly validates the plumbing. Blocked until the weights are obtained (see Known problems).
3. **navtrain → JSON converter,** reusing the `gen_data_planning.py` format.
   - Add the driving command and velocity/acceleration to the prompt.
   - Use NAVSIM's history: 4 frames at 2 Hz, not 10. NAVSIM calls this "2 s", but the frames span 1.5 s.
   - Primary metric: **navhard two-stage EPDMS (v2)**. NAVSIM v1 is saturated (see `RESEARCH_DIRECTION.md` §2.6).
4. **Rebuild the codebook(s)** from navtrain paths, for flat A/B, or E.
5. **Training plan:** Stage 1 on nuScenes (optional), then Stage 1 on navtrain with A, then with B. Optionally add Nu-X as an auxiliary task.
6. **Optional preference tuning:** sample K paths per scene, rank them with the PDM score, and train with DPO (`train/llava/train/train_dpo.py` exists).

Related work to compare against (reported numbers, not verified here):
- **DriveFine:** a masked-diffusion VLA on NAVSIM; reportedly 90.7 PDMS, 91.8 after RL.
- **ReCogDrive:** InternVL3-8B + diffusion planner; 89.6 PDMS.

### 4. Object-motion context instead of multiple images (now core claim C1)

**Idea:** keep a single high-resolution current frame, and summarise the past with a lightweight perception module that outputs compact tokens.

**Update 2026-10-16:**
- This is now the core claim C1, no longer a fallback. See `RESEARCH_DIRECTION.md` §2.2.
- **Primary tracker:** LiDAR 3D (CenterPoint / PointPillars) + a greedy tracker at 2 Hz. YOLO 2D is a fallback only.
- **Always produce two versions of the motion tokens:** from ground-truth tracks (upper bound / teacher) and from the deployable tracker (realistic).
- **Report** NC/TTC sub-scores and a robustness curve against tracker quality.

**Update 2026-10-20:**
- navhard / v2 challenge test frames come **without LiDAR** (V, NAVSIM maintainers). A tracker scored on navhard must therefore be camera-only; LiDAR remains an option for v1 navtest only.
- The options, the evidence and a proposed oracle-motion gate are in `docs/OBJECT_MOTION.md`.
- **Deferred [agreed 2026-10-20]:** multi-frame and object-motion context are on the back burner. The pilot uses a single frame for now.

- **Detector + tracker:** YOLO (v8/11, n/s/m) + ByteTrack / BoT-SORT over the 4 past frames. Per object: class, box, image-space velocity → a few dozen tokens in total, as text or a learned embedding.
- **Ego motion:** already provided by NAVSIM's ego status. It doesn't need vision.
- **YOLO's limitation:** it gives 2D boxes with no depth, so it can't tell an approaching object from a growing one.
- **Stronger alternatives:**
  - NAVSIM includes LiDAR at test time, so a LiDAR 3D detector (e.g. CenterPoint / PointPillars) gives tracks in metres.
  - Or add monocular depth to the YOLO boxes.
- **Fallback that still uses images:** multiple frames with spatial pooling (`mm_spatial_pool_stride`) or the Q-Former / perceiver resamplers in `train/llava/model/multimodal_resampler/`.
- **Comparison:** 1 frame vs 1 frame + tracks vs 4 pooled frames, at matched token budget.

### 5. Prompt engineering (supporting)
- **Fix MVLAD-AD's prompts first:**
  - The train/inference wording mismatch.
  - It claims an image sequence and a rationale that aren't provided.
  - It spends ~200 tokens on 2-decimal history coordinates.
  - It omits velocity, acceleration and the driving command (WAM-Diff includes them).
- **Ablations:**
  - Coordinate format and precision.
  - Structured fields vs prose.
  - System prompt.
  - Few-shot examples.
  - Prompt length vs latency.
- Run every ablation **with and without ego status** (the AD-MLP shortcut), on NAVSIM v2.
- **No chain-of-thought in the fast pilot:** ~8 ms per reasoning token on Thor at 2B.

### 6. Quantization (supporting, but required for the edge claim)
- **Thor:** NVFP4 backbone, FP8 for sensitive layers. Keep the action-head attention ≥ FP8.
- **Orin:** INT4-AWQ (W4A16) or INT8; Orin has no FP8/FP4.
- **Diffusion-specific:** calibrate across denoising timesteps and mask ratios, because diffusion LMs accumulate quantization error across steps.
- **Measure:** accuracy change (PDMS/EPDMS, L2) and on-device latency per component, on a physical Jetson.

---

## Compute breakdown, per component

> This section covers **training and evaluation on GPUs** (Anvil).
>
> **On-vehicle (edge) budgets** are in `RESEARCH_DIRECTION.md` §2.3, using NVIDIA's measured Jetson Thor/Orin numbers.
>
> **Correction 2026-10-16:** the shipped config produces ~3.6k image tokens per frame, not ~1.1k total. LLM-side figures below that assumed ~1.1k-token sequences are about **3× too low** for the current config. Switching to a single 384×384 view (729 tokens) restores them.

### Reference numbers
- **LLaDA-V language model:** ~8B parameters → **~16 GB** of weights in bf16.
- **SigLIP2-so400m:** ~0.4B parameters → **~0.8 GB**. One 384-pixel image becomes **729 visual tokens**. With `anyres_max_4` tiling, a 1600×900 front frame becomes **~3.6k** (729 overview + ≤~2.9k tiles + row-newline tokens; from code reading).
- **Typical prompt today:** ~3.6k image tokens + ~300 text tokens + 10 path tokens (+118 explanation tokens for Nu-X).
- **One forward pass costs** ≈ 2 × params × sequence length FLOPs. **Diffusion decoding repeats the full forward pass every step**, with no KV cache by default. dLLM-Cache reduces this.

### A. NAVSIM itself (data + scoring; no model)

| Item | Requirement |
|---|---|
| navtrain | 14 GB logs + **445 GB** sensors (~300 GB without history) |
| navtest | ~1 GB logs + **223 GB** sensors |
| navhard_two_stage | ~0.9 GB + 31 GB |
| mini / warmup | 151 GB / 1.2 GB. Start with warmup |
| Maps | nuPlan maps (`download_maps`); a few GB (est.) |
| **Total disk** | **~0.7–1 TB** NVMe, plus room for metric caches and your own feature caches (est.) |
| Metric caching + PDM scoring | **CPU-bound** (Ray multiprocessing). No GPU needed. Scales with cores. Benchmark on warmup first; the docs give no runtime |
| RAM | 32–64 GB comfortable (est.) |
| Environment | Separate conda env (`environment.yml`, nuplan-devkit). Keep it apart from the model env |

### B. Image encoder (SigLIP2-so400m)
- **Inference:** under 1 GB of memory, a few ms per image on a modern GPU. Negligible.
- **Training:** frozen today. Unfreezing it at the configured `mm_vision_tower_lr=2e-6` adds roughly 0.4B parameters' worth of gradients and optimizer state (~5–7 GB, est.).
- **The real cost is downstream:** every image adds 729+ tokens to the language model's sequence.

### C. Language model (LLaDA-8B, or swapped base model)

**Inference (est.)**

| Setup | Seq. length | Steps | FLOPs / sample | 1× RTX 4090 (24 GB) | 1× H100 (80 GB) |
|---|---|---|---|---|---|
| Planning, 1 image | ~1.1k | 10 | ~180 TFLOP | ~2–3 s | ~0.4–0.6 s |
| Path + explanation, 1 image | ~1.2k | 2 + 128 | ~2.5 PFLOP | ~30 s | ~5–7 s |
| 4 frames, unpooled | ~3.3k | 10 | ~530 TFLOP | ~6–8 s; memory tight | ~1.5 s |
| 8 cams × 4 frames, unpooled | ~23k | 10 | — | Not practical | Not practical. Needs pooling / resampler / plan 4 |

- **Whole navtest** (~12k samples), planning, 1 image: **~7–10 h on a 4090, ~1.5–2 h on an H100** (est.). dLLM-Cache and fewer steps cut this further.
- **Minimum GPU for inference:** 24 GB, bf16. 16 GB cards need 8-bit or 4-bit quantization.

**LoRA fine-tuning (est.)**

| Item | Value |
|---|---|
| Frozen weights | ~16 GB bf16 |
| LoRA (adapter is ~360 MB on disk) + projector + Adam state | ~3–5 GB |
| Activations (gradient checkpointing, batch 4, ~1.1k tokens) | ~10–20 GB |
| **Practical GPU** | **80 GB (A100/H100)**. A 24 GB card works only at batch 1 + 4-bit (QLoRA), and slowly |
| Paper's reference | 4×H100, global batch 32, 8 epochs ≈ **9 h per stage** on nuScenes (~15k–30k samples) |
| navtrain (~100k samples), 1 image, 8 epochs | ~**30–60 h on 4×H100** (~120–240 GPU-hours) |
| Same with 3 epochs | ~12–25 h on 4×H100 |
| navtrain with 4 frames | ~3× the above unless pooled |
| Rough cloud cost | ~120–240 H100-hours per full single-image run. Check current rental prices |

**Smaller backbones (tier c):** a 2–3B backbone cuts all of the above by ~3–4× and makes single-24 GB-GPU training realistic.

### D. Point tokenizer / codebooks
- **K-means** over 2D waypoints or whole trajectories: CPU, minutes.
- **Embedding training** (`tokenizer/tokenizer.py`): small MLPs on 2D points. Minutes to an hour on any GPU.
- **Option B:** two codebooks; same cost.
- **Option E:** K-means over 4k–8k whole trajectories; CPU, minutes.

### E. Output-head add-ons
- **D (correction head):** a few million parameters. Negligible compute and memory.
- **B:** doubles the path tokens (10 → 20). Adds ~1% to the sequence length; negligible.
- **E:** one token instead of 10 means fewer decoding steps, so **cheaper** inference.
- **Separate diffusion path head (tier c):** about 50–300M parameters. Small next to the backbone. Run several decoding steps over the head only, not the backbone, which makes decoding much cheaper than today.

### F. Motion-history processor (plan 4)

| Module | Size | Cost |
|---|---|---|
| YOLOv8/11-n … m | 3–25M params | ~2–10 ms per frame on a 4090; 4 frames per sample → under 50 ms |
| ByteTrack / BoT-SORT | — | CPU, ~ms |
| LiDAR 3D detector (CenterPoint / PointPillars) | 5–10M params | ~10–50 ms per sweep on GPU |
| Tokens added to the language model | ~20–100 | Negligible compared with 729 tokens per image |

- **Overall:** under 5% of total compute, and it removes most of the multi-image cost.
- **Pre-computing:** run it once over navtrain and store the tracks, so training never pays for it.

### G. Preference tuning with PDM scores (optional, plan 3.6)
- **GPU cost:** K samples per scene × navtrain scenes × planning-inference cost. For example, K=8 → ~8× a navtest-sized pass over ~100k scenes: **~20–40 H100-hours** (est.).
- **CPU cost:** PDM scoring of all K candidates, the same scorer as evaluation × K.
- **The DPO training run itself:** about the same as a LoRA run.

### Hardware tiers

| Tier | Hardware | Covers |
|---|---|---|
| Dev / eval | 1× 24 GB GPU, 16–32 core CPU, 64 GB RAM, 1–2 TB NVMe | Running NAVSIM, zero-shot eval, tokenizer work, YOLO pre-compute, small backbones |
| Training | 4–8× 80 GB GPUs (cloud rental is fine) | LoRA on 7–8B backbones over navtrain; multi-frame experiments |

---

## Sources
- NAVSIM install / splits / agent docs: <https://github.com/autonomousvision/navsim/tree/main/docs>
- OpenScene / NAVSIM dataset sizes: `docs/splits.md` in the repo above
- DriveFine: <https://arxiv.org/abs/2602.14577>
- ReCogDrive: <https://arxiv.org/abs/2506.08052>
- MVLAD-AD: <https://arxiv.org/abs/2602.20577>
