# Build a metric-aware masked head first

Build the masked discrete-diffusion head first, as planned, but make it continuous wherever geometry enters. Build the comparison "DiT" as an anchored, few-step continuous head on the same transformer trunk, not as a vanilla Gaussian DiT. Head family alone will not decide NAVSIM v1 navtest PDMS. From supervised starts of 86.5–86.7, a continuous DiT head (ReCogDrive 86.5 → 90.8 [V]), in-LLM masked diffusion (WAM-Diff 86.6 → 91.0 [V]) and discrete flow (WAM-Flow 86.7 → 90.3 [V]) all land near 90–91 after RL. The only fully controlled masked-vs-continuous swap found puts them 0.2 points apart (LIBERO 95.1 vs 94.9 [V]). What separates the families is plumbing. A masked head over 16 per-axis tokens gives K plans by sampling the goal token first, an exact categorical likelihood for PDMS RL (the plan is short), and free inpainting. A continuous head keeps its quality at 1–2 steps only when sampling starts near the answer (anchors, or a flow start of σ ≈ 0.1). Otherwise an ε-prediction DiT scores 1.92 PDMS at 1 step [V] and vanilla diffusion reaches only 11% diversity [V]. So "how continuous" has a two-part answer: discrete in what gets sampled and scored by RL, continuous in embeddings, targets, auxiliary losses and readout. Two cautions apply throughout. No driving paper yet puts a separate masked-diffusion head on an autoregressive VLM. And how the head reads the VLM moves PDMS more than head type does (final layer only 80.2 vs multi-layer 85.2 [V]). The deciding ablation is a same-trunk swap of input and output layers, judged on post-RL single-sample PDMS and oracle best-of-6 on the full official navtest. Supervised top-1 will not separate the arms.

## Every head family converges near 90–91 PDMS after RL

Every number in this section is NAVSIM v1 navtest PDMS as each paper self-reports it. The notes could not confirm that any of these papers used the full official navtest with NAVSIM's own scorer, so cross-paper gaps under about a point are noise.

With supervised training only, masked discrete-diffusion VLAs score **84.8–86.7**: WAM-Diff 86.6 [V] ([WAM-Diff](https://arxiv.org/html/2512.11872v1)), DriveFine 86.7 [V] ([DriveFine](https://arxiv.org/html/2602.14577)), and ReflectDrive 84.8 before reflective inference [V] ([ReflectDrive](https://arxiv.org/html/2509.20109v1)). ReflectDrive-2 starts at 84.8 and reaches 87.7 with its extra training losses [V] ([ReflectDrive-2](https://arxiv.org/html/2605.04647)). The discrete-flow model WAM-Flow sits at 86.7 [V] ([WAM-Flow](https://arxiv.org/html/2512.06112v1)).

Single-camera VLAs with continuous heads score **86.5 (ReCogDrive's DiT) to 88.3 (DiffAdapterVLA on a frozen 8B VLM)** [V], with DriveVLA-W0's flow expert at 87.2 [V] ([ReCogDrive](https://arxiv.org/html/2506.08052), [DiffAdapterVLA](https://arxiv.org/html/2609.15322), [DriveVLA-W0](https://arxiv.org/html/2510.12796)).

After RL the families overlap. From starts of 86.5–86.7 the gains are **ReCogDrive +4.3, WAM-Diff +4.4, WAM-Flow +3.6 and DriveFine +2.9 with GRPO alone** [V/D] ([ReCogDrive README](https://github.com/xiaomi-research/recogdrive)). The starting score predicts the gain better than the family does [D]. Gains shrink as the start rises: +4.3 from 86.5 (ReCogDrive), +2.1 from 89.1 (DiffusionDriveV2 with its selector held fixed) ([DiffusionDriveV2](https://arxiv.org/html/2512.07745)), +1.0 from 89.4 (EponaV2) ([EponaV2](https://arxiv.org/html/2605.14696v1)) and +1.3 to +1.5 from 90.8 (DriveWLA) ([DriveWLA](https://arxiv.org/html/2610.17903v1)). Weak starts gain far more: +8.6 from MD-Drive's 80.3 ([MD-Drive](https://arxiv.org/html/2611.15238)) and +8.57 from AutoVLA's 80.54 ([AutoVLA](https://arxiv.org/html/2506.13757)).

**Table 1. Head families compared (NAVSIM v1 navtest PDMS unless marked)**

| Family (examples) | SFT-only PDMS | After RL or selection | Steps that hold quality | How K plans arise | RL likelihood | Main risk for our head |
|---|---|---|---|---|---|---|
| Masked discrete diffusion, in-LLM (WAM-Diff, DriveFine, ReflectDrive, ReflectDrive-2, MD-Drive) | 84.8–86.7 [V]; ReflectDrive-2 87.7 with extra losses [V]; MD-Drive 80.3 [V] | RL 88.9–91.0 [V]; DriveFine 91.8 with score-based RFT [V]; ReflectDrive 91.1 with scorer-guided reflection, no RL [V] | 3–5 for 16 tokens (ReflectDrive peaks at 5; DriveFine 90.47 at 4) [V]; no 1–2-step driving result | Goal token sampled first, BEV NMS, then inpainting; oracle best-of-6 94.2–94.8 [V] | No exact sequence likelihood in general; mean-field (d1, WAM-Diff) or realized unmasking path (DriveFine) | No separate-head driving precedent; must choose a likelihood estimator |
| Discrete flow matching (WAM-Flow) | 86.7 [V] | 90.3 [V] | 1 step 89.1, 5 steps 90.3 after RL [V] | Not reported | Per-token ratios; how the CTMC path likelihood is computed is not explained [V] | Only one driving paper; v2 EPDMS only 84.7 [V] |
| Soft-label discrete hybrid (Discrete-WAM) | No v1 SFT number; v2 EPDMS 89.1 [V] | v1 92.2 final [V]; v2 EPDMS 90.4 after RL [V] | Confidence-replace rounds [V] | 16 decision anchors best; more anchors weaken GRPO [V] | Categorical | Soft labels never isolated; rests on 200k-step nuPlan pretraining [V] |
| Gaussian DiT, ε-prediction (ReCogDrive, DiffAdapterVLA) | 86.5; 88.3 [V] | 90.8 (2B per README); 90.3 [V] | About 5; DiffAdapterVLA 1.92 at 1 step, 82.63 at 2 [V] | Not reported for VLA heads; vanilla diffusion diversity 11% [V] | Exact per-step Gaussian (DiffGRPO) [V] | Fails at 1–2 steps; near-duplicate samples |
| Rectified flow / flow matching (DriveVLA-W0, DriveWLA, GoalFlow) | W0 flow 87.2; DriveWLA imitation 90.8; GoalFlow base 85.6 [V] | DriveWLA 92.3 with head-only RL [V]; GoalFlow 90.3 with goal and trajectory scorers [V] | GoalFlow 88.9 at 1 step vs 90.3 at 5 [V]; VITAL 90.8 at 1 step [R] | Many samples from a narrow start; σ = 0.2 collapses to 49.0 [V] | Exact after ODE→SDE (Flow-GRPO) or learned noise (ReinFlow) | "Too sample-inefficient" at NAVSIM scale (W0) [V] |
| Truncated / anchored diffusion (DiffusionDrive, DiffusionDriveV2, ResAD; camera + LiDAR) | 88.1; ResAD 88.8 [V] | DiffusionDriveV2 91.2 with selector, of which RL is +2.1 [V/D] | 87.9 at 1 step, 88.1 at 2 [V] | 20 anchors in one pass, diversity 74% [V]; mean proposal only 60.3 [V] | Exact per-step Gaussian with η = 1 [V] | Needs anchor-aware GRPO; RL cuts diversity 42.3 → 30.3 [V] |
| One-step distilled or self-consistency (OneDP, shortcut, MeanFlow/MP1) | None on NAVSIM | None | 1 step matches 10–100-step teachers in robotics [V] | Not reported | Needs a learned noise head for RL | No driving evidence |
| Query regression (control, DriveVLA-W0) | 88.4 [V] | Not reported | One pass | None | None usable by GRPO | No K plans |

Swapping text for a head buys less as the text baseline gets stronger. ReCogDrive's README shows **+2.4 on its 2B VLM (84.1 → 86.5) but +0.1 on its 8B VLM (86.4 → 86.5)** [D from V] ([ReCogDrive README](https://github.com/xiaomi-research/recogdrive)). Our 85.6 text model sits between those cases, so the supervised step alone should buy little; the large gain arrives with RL.

A text-waypoint model can also be RL-trained. Curious-VLA, a Qwen2.5-VL-3B text-waypoint planner, goes **87.6 → 90.3** with GRPO [V] ([Curious-VLA](https://arxiv.org/html/2603.06049)). The head therefore earns its place through properties text cannot offer, not through raw PDMS. First, it skips text decoding: ReCogDrive's text output takes 0.5839 s per sample against 0.0750 s for its diffusion planner, so the planner is **7.8× faster** [V/D]. Second, it emits K plans in one batched pass. Third, it exposes a clean likelihood for RL. Text waypoints are still the weakest output wherever they are ablated: AutoVLA scores 71.31 with text waypoints against 80.54 with physical action tokens [V].

Controlled comparisons show that the liability is autoregressive decoding, not discreteness. DD-VLA's ablation is the cleanest found. It keeps one Qwen2.5-VL backbone, changes only the head, and scores AR 81.2, one-pass parallel discrete 94.2, continuous diffusion 94.9 and masked discrete diffusion 95.1 on LIBERO [V] ([DD-VLA](https://arxiv.org/html/2508.20072)). OpenVLA-OFT shows a larger continuous edge: parallel discrete 90.2 against continuous L1 95.3 and diffusion 95.4 [V]. Its discrete arm, however, decodes in one pass with no iterative refinement ([OpenVLA-OFT](https://arxiv.org/html/2502.19645)). Continuous heads win big only against naive binning or small, non-robot-pretrained policies. In Octo, the diffusion head scores 83% against 18% for the discretized head [V]. In RoboVLMs with PaliGemma, CALVIN average length is 4.04 continuous against 0.44 discrete [V] ([Octo](https://arxiv.org/html/2405.12213), [RoboVLMs](https://arxiv.org/html/2412.14058)).

DriveVLA-W0 is the only controlled head swap on NAVSIM. It ranks query regression 88.4, flow matching 87.2 and AR tokens 85.3 [V]. On its 70M-frame in-house set the order reverses and AR has the best ADE [V]. It never tests masked or parallel discrete decoding ([DriveVLA-W0](https://arxiv.org/html/2510.12796)). No paper compares masked and continuous heads on one backbone on NAVSIM.

**Table 2. Same-paper ablations that isolate the head** (C1 = same backbone, data and budget; only the head or output representation changes)

| Paper (control level) | Setting | Arms compared | Result |
|---|---|---|---|
| DD-VLA Table 13 [C1] | Qwen2.5-VL, LIBERO avg, 30k steps, 256 bins | AR / parallel discrete / FAST / continuous diffusion / masked diffusion | 81.2 / 94.2 / 91.1 / 94.9 / 95.1 [V] |
| OpenVLA-OFT [C1] | OpenVLA 7B, LIBERO avg | AR / parallel + chunked discrete / + continuous diffusion / + L1 | 76.5 / 90.2 / 95.4 / 95.3 [V] |
| DriveVLA-W0 Table 4 [C1] | Same pretrained VLA, NAVSIM v1 | Query regression / flow (10 steps) / AR tokens | 88.4 / 87.2 / 85.3 [V] |
| ReCogDrive Table 4 | Same VLM, NAVSIM v1 | Text / MLP / query / vanilla diffusion / refined DiT | 84.1 / 75.5 / 85.0 / 85.4 / 86.5 [V] |
| DiffusionDrive Table 2 [C1] | Transfuser, NAVSIM v1 | MLP / vanilla DDIM, 20 steps / truncated, 2 steps / cascade decoder | 84.0 / 84.6 / 85.7 / 88.1 [V] |
| DiffAdapterVLA Tables 4–5 | Frozen 8B VLM, NAVSIM v1 | Final cache only / late layers, trajectory outside VLM / trajectory inside VLM | 80.2 / 85.2 / 86.9 [V] |
| AutoVLA [C1] | Qwen2.5-VL-3B, NAVSIM v1 | Text waypoints / FAST tokens / K-disk tokens | 71.31 / 67.63 / 80.54 [V] |
| Alpamayo-R1 Table 12 [C1 for decoder] | In-house data | AR tokens vs flow (5 steps) | minADE6 0.6811 vs 0.6440; comfort 44.05% vs 97.38% [V] |
| DFM-VLA [C1] | CALVIN avg length | AR / masked diffusion / discrete flow | 4.18 / 4.32 / 4.42 [V] |
| SmolVLA | About 0.5B, LIBERO | L1 regression vs flow | 75.25 vs 80.25 [V] |

Rows above about 92 v1 PDMS do not compare to a single-camera 0.8B SFT model. FlowR2A reaches 92.8 with 60 proposals and a learned selector on camera plus LiDAR [V] ([FlowR2A](https://arxiv.org/html/2606.24231)). TransDiffuser reaches 94.85 with 30 candidates and rejection sampling [V] ([TransDiffuser](https://arxiv.org/html/2505.09315)). DP-VLA's 94.8 comes from RL trained on the navtest split itself [V] ([DP-VLA](https://arxiv.org/html/2601.00898)). Discrete-WAM's 92.2 has no supervised-only v1 number and rests on 200k steps of full-nuPlan pretraining [V] ([Discrete-WAM](https://arxiv.org/pdf/2606.05645)).

NAVSIM v2 EPDMS and navhard rank models differently, so they stay separate here. On v2 navtest EPDMS the reported scores are:

- Discrete-WAM 90.4 after RL [V]
- DriveWLA 90.1 [V]
- WAM-Diff 89.7 [V]
- DriveFine 87.1, or 89.7 on the bug-fixed release [V]
- DiffAdapterVLA 86.3 [V]
- DiffusionDriveV2 85.5 [V]
- WAM-Flow 84.7 [V]
- ReCogDrive 83.6 [V]
- VITAL 91.1 [R] ([VITAL](https://arxiv.org/html/2610.20604v1))

On navhard two-stage EPDMS, DriveFine scores 74.4 at stage 1 and 41.0 at stage 2, against 68.9 and 37.8 for ReCogDrive [V]. GuideFlow reaches 43.0 with a GTRS-Dense scorer [V] ([GuideFlow](https://arxiv.org/html/2511.18729v3)). WAM-Flow scores 90.3 on v1 but 84.7 on v2, so a v1 lead does not carry over automatically.

## The start distribution, not the solver, sets the step count

A continuous head needs few steps only when sampling starts near the answer. Pure-noise ε-prediction needs about five. DiffAdapterVLA scores 1.92 at 1 step, 82.63 at 2, 84.32 at 3, 87.00 at 5 and 87.07 at 10 [V] ([DiffAdapterVLA](https://arxiv.org/html/2609.15322)). In robotics one step fails outright. OpenVLA-OFT's diffusion head, trained with 50 steps, drops to 0.0 LIBERO-Long success at 1 step [V], and OneDP finds that naive 1-step DDIM scores 0.000 on all six robot tasks [V] ([OneDP](https://arxiv.org/html/2410.21257)).

Heads that start near the data hold up. DiffusionDrive's anchored x0-prediction head scores **87.9 at 1 step, 88.1 at 2 and 88.1 at 3** [V]. GoalFlow's rectified flow scores 88.9 at 1 step against 90.3 at 5 [V] ([DiffusionDrive](https://arxiv.org/html/2411.15139), [GoalFlow](https://arxiv.org/html/2503.05689)). VITAL reports 90.8 at 1 step, within 0.13 of its 10-step score [R] ([VITAL](https://arxiv.org/html/2610.20604v1)).

DiffusionDrive's "diffusion" is really a small perturbation of the anchors. From the code constants, inference noises the anchors at t = 8 of 1000, a std of about 0.90 m in x and 0.73 m in y. Training never goes past t = 49, about 2.7 m in x [D] ([DiffusionDrive code](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py)). GoalFlow shows the same thing for flow: start noise σ = 0.1 gives 90.3, while σ = 0.2 collapses to 49.0 and σ = 0.3 to 18.8 [V].

The prediction target matters too. On CARLA, DiffE2E's driving score drops from 82.9 to 20.1 when it switches from x0 to noise prediction [V] ([DiffE2E](https://arxiv.org/html/2505.19516)). Against vanilla 20-step DDIM with the same UNet, truncation adds +1.1 PDMS, runs **9.4× faster** and lifts diversity from 11% to 70% [V/D]. No NAVSIM paper compares DDIM and rectified flow on one architecture. Next to the start distribution and the x0 or velocity target, the solver is a minor variable.

When a pure-noise start cannot be avoided, purpose-built one-step methods recover quality in robotics, though none has a driving benchmark. OneDP's distilled 1-step policy matches its 100-step teacher (0.830 vs 0.829) after 20 distillation epochs. Consistency-policy distillation reaches only 0.672 at 1 step even after 450 epochs [V] ([OneDP](https://arxiv.org/html/2410.21257)). A shortcut policy scores 0.87 on Push-T at 1 step, where a 1-step diffusion policy scores 0.12 [V] ([Shortcut models](https://arxiv.org/html/2410.12557)). MP1 (MeanFlow) reaches 78.9% at 1 NFE against 68.7% for 10-step DP3 [V] ([MP1](https://arxiv.org/html/2507.10543)). For an 8-waypoint head these are only a fallback. Anchors give the same few-step behaviour and have direct NAVSIM evidence.

Masked heads need a few more steps, and the driving evidence stops at 3. For 16-token plans, ReflectDrive peaks at 5 steps and declines after [V]. DriveFine keeps 90.47 PDMS at 4 steps [V]. ReflectDrive-2 runs 1+3+3 steps (decision, draft, reflect) on full frames and 1+1 on lite frames [V]. WAM-Diff2 saturates at 8–16 steps [V] ([WAM-Diff2](https://arxiv.org/pdf/2608.01035)). WAM-Diff runs 32 iterations for 32 tokens, about one token per step [D], so it saves no steps over AR.

WAM-Flow is the exception: its discrete flow scores 89.1 at 1 step and 90.3 at 5 after RL [V]. Its corruption path, softmax(−β_t·d(x, x1)), keeps intermediate states on metrically nearby tokens [V], the discrete counterpart of a narrow start. No driving paper reports a masked head at 1–2 steps, so **4 steps** is the documented setting. For a 16-token plan that commits four tokens per step [D]. Commit tokens selectively. In Discrete-WAM, replacing every token each round raises 4 s L2 (+0.0084 m), while confidence-based replacement lowers it (−0.0040 m) [V].

Iterative discrete and continuous heads cost the same per step, because the trunk does the same work whatever the output. On an H800, DD-VLA's 12-step masked head takes 68.8 ms and OFT's 12-step continuous diffusion head 67.1 ms [V]. Small heads cost a few ms per step on desktop GPUs. DiffusionDrive's 60M decoder takes 3.8 ms/step for 20 batched samples on an RTX 4090 [V]. pi0's 300M expert takes 2.7 ms/step over 10 flow steps on an RTX 4090 [D from V] ([pi0](https://arxiv.org/html/2410.24164)).

Four masked steps against two anchored steps doubles the head passes. That is small next to the VLM prefill: in Alpamayo-R1, prefill takes 16.54 ms against 8.75 ms for its flow decoder on an RTX 6000 Pro Blackwell [V] ([Alpamayo-R1](https://arxiv.org/html/2511.00088)). Make K plans a batch dimension over one shared cached context. No paper measures latency against K for any driving head.

## Categorical tokens make K plans and PDMS RL straightforward

### Diversity comes from anchors or goal tokens, and RL narrows it

Gaussian diffusion trained on single demonstrations collapses. DiffusionDrive's vanilla baseline reaches **11% diversity** because "20 random noises denoised over 20 steps converge to similar trajectories" [V]. DiffusionDriveV2 measures its truncated, anchor-free baseline at Div 0.1 [V].

Anchors restore the spread (74% [V]), but most proposals are poor. DiffusionDrive's mean proposal scores **60.3 PDMS** [V] ([ResAD Table 5](https://arxiv.org/html/2510.08562v2)) with a std of 33.1 [R] ([FlowR2A Table 3](https://arxiv.org/html/2606.24231)), so the selector does the real work. Quality saturates near K = 20: 84.9 at 10 samples, 88.1 at 20 and 88.2 at 40 [V]. No continuous-head VLA (ReCogDrive, DiffAdapterVLA, DriveVLA-W0) reports multi-sample metrics at all.

Masked planners make diversity explicit. ReflectDrive samples terminal-waypoint goals, keeps K = 3 after 0.9 m BEV NMS, and inpaints each [V]. ReflectDrive-2 samples top-k goals with an NMS threshold of about 1.2 m and uses an RL group of 3 goals × 2 drafts [V]. Oracle best-of-6 reaches **94.2 (DriveFine) and 94.8 (ReflectDrive-2)** [V], against about 91 single-sample, so the selector has real headroom. No paper reports a quantitative diversity metric for masked driving plans or compares masked and continuous diversity directly.

RL narrows the spread a selector needs. Under GRPO, DiffusionDriveV2's 20 raw samples lose diversity (42.3 → 30.3, −28% relative [V/D]) while PDMS@10 rises from 75.3 to 84.4 [V]. The paper uses per-anchor advantage grouping to stop collapse onto "go straight" [V].

Supervised over-training has the same effect. MD-Drive's pass@8 peaks at 96.72% at checkpoint 5318 and falls to 95.99% by checkpoint 10636, while pass@1 keeps climbing [V]. Its over-fitted checkpoint gains only **84.3 → 85.8** from GRPO, against **80.3 → 88.9** for the earlier one [V]. This bears directly on the goal of fitting train near-perfectly without memorizing. Choose the checkpoint that goes to RL by validation pass@K or oracle best-of-K, not by train fit or top-1. Curious-VLA reports the same exploration collapse after imitation: its k = 8 sample spread (pADE) is only 0.090 before its diversity-oriented training [V].

One implementation trap: truncating Gumbel noise to 32-bit lowers the effective sampling temperature of masked models, with sentence entropy falling from 5.66 to 5.17 [R] ([Zheng et al.](https://arxiv.org/html/2409.02908v6)). Sample K plans with fp64 Gumbel noise or explicit categorical sampling.

The selector keeps its value after RL. In every paper that reports it, oracle best-of-6 still sits above the single sample after RL [D]: +3.0 for AutoVLA, +3.5 for DriveFine, +4.5 for Curious-VLA, and +2.8 and +3.6 for PaIR-Drive's two heads. A learned selector also stacks with RL: in DiffusionDriveV2 the selector adds +1.0 and RL then adds +2.1 [D].

### Both families have a working PDMS-RL recipe

Continuous heads get an exact likelihood for free once sampling is stochastic. Each denoising step is a Gaussian policy, so the chain log-probability is a sum of per-step terms [V]. The NAVSIM recipe is settled. ReCogDrive's DiffGRPO uses group-standardized PDMS advantages, a BC loss in place of KL, one on-policy update and no clipping. Its ablations favour a BC weight of 0.001 (90.8, against 89.3 at 0.1), a sampling-std floor of 0.02 (90.8, against 89.5 at 0.01) and a denoising discount of 0.6 (90.8, against 90.3 at 1.0) [V]. RL mostly moves EP (+6.4) and DAC (+2.6), while NC dips from 98.1 to 97.9 [V/D] ([ReCogDrive](https://arxiv.org/html/2506.08052)).

Anchored heads need extra care. In DiffusionDriveV2, intra-anchor grouping adds +0.9, inter-anchor truncation adds +0.6, and multiplicative exploration noise beats additive noise (90.1 vs 89.7) [V]. Flow heads need an ODE-to-SDE conversion or learned noise. π_RL's flow policy does best when RL runs at 2 steps (97.0 on LIBERO-Spatial) and worse at 1 step (88.5) [V] ([π_RL](https://arxiv.org/html/2510.25889)). ReinFlow fine-tunes at 1–4 steps by learning the noise [V] ([ReinFlow](https://arxiv.org/html/2505.22094)).

Masked heads have no exact sequence likelihood in general, and the field has not settled on an estimator [V/R]. d1 and WAM-Diff use a one-step mean-field estimate ([d1](https://arxiv.org/html/2504.12216v2)). DriveFine and TraceRL score the realized unmasking path ([TraceRL](https://arxiv.org/abs/2509.06949)). SPG bounds the likelihood ([SPG](https://arxiv.org/abs/2510.09541)), and wd1 avoids ratios altogether ([wd1](https://arxiv.org/abs/2507.08838)).

That problem shrinks for our head. With 16 tokens committed over about 4 steps, the exact log-probability of a sampled plan under its realized order is the sum of the categorical log-probs of the tokens committed at each step. This is DriveFine's idea and is cheap at this length. It is a design inference: no paper has run it on a separate head.

Driving masked RL works with small groups. WAM-Diff scores 88.9 at G = 2 and 91.0 at G = 3. WAM-Flow does worse at G = 4 than at G = 3 (89.6 vs 90.3) [V]. DriveFine's sweep, by contrast, keeps rising to 89.6 at G = 8 [V]. Use mean-only advantages: NoRD goes from 77.18 with standard GRPO to **85.62 with Dr. GRPO**, because std normalization starves hard turning scenes of gradient [V] ([NoRD](https://arxiv.org/html/2602.21172v2)).

Masked heads also get inpainting for free. ReflectDrive's scorer-guided re-inpainting adds **+6.3 with no RL** (84.8 → 91.1 [V/D]). ReflectDrive-2's AutoEdit adds +0.3 after SFT but +1.9 after RL [V/R].

RL on a head that reads cached states is cheap, but it carries one published warning. DriveFine argues that diffusion planners loosely coupled to a VLM reward-hack under PDMS RL and lose EPDMS. It shows the drop only in a figure, so the size is unverified (claim [V], magnitude [R]). Two results point the same way. DiffAdapterVLA's external DiT on a frozen VLM scores 76.4, or 81.9 with LoRA [V]. Knowledge Insulation found that a fully frozen backbone gave 0% on shirt folding [V] ([KI](https://arxiv.org/html/2505.23705)). Our cached states come from a model already fine-tuned on driving text, which lowers this risk but does not remove it.

Head-only RL never re-runs the VLM, so each sample costs a head pass plus a PDM-scorer call. One navtrain epoch is about 309K scorer calls at G = 3 and about 824K at G = 8 [D]. That favours parallel scoring on CPU workers, plus DriveWLA's filter of training only on scenes where imitation scores below 90 [V]. Watch v2 terms while optimizing v1: PaIR-Drive's RL cut v2 extended comfort from 87.7 to 74.2 [V] ([PaIR-Drive](https://arxiv.org/html/2603.13842v1)).

### Training stability favours plain classification

Supervised training is where the masked head is simplest. It is plain cross-entropy on 16 tokens, with no noise schedule, normalization range or anchor set to tune. Continuous heads train more slowly. OpenVLA-OFT's diffusion head needs **100–250K steps against 50–150K** for its other heads [V]. DriveVLA-W0 calls flow matching "too sample-inefficient to converge" at NAVSIM scale [V]. ReCogDrive trains its planner for 200 epochs at batch 512 [V]. NORA-1.5's flow expert underperforms AR on a 50K-frame set [V] ([NORA-1.5](https://arxiv.org/html/2511.14659v1)).

dVLM-AD reports masked instability ("notably unstable", and worse with extra action vocabularies [V]). That instability comes from fine-tuning a whole diffusion LLM, not a small classifier head ([dVLM-AD](https://arxiv.org/html/2512.04459v1)).

Data needs track the planning data rather than head capacity. Discrete-WAM scores about the same trained from scratch (89.8 v2 EPDMS) as with LoRA (90.0) [V], and E0's small-expert recipe trains in about 24 GPU-hours [V] ([E0](https://arxiv.org/html/2511.21542v2)). No paper reports multi-seed PDMS variance on the full navtest for masked planners. The cheapest stability default comes from DiscreteRTC, the only same-trunk discrete-vs-flow study: an fp32 head with a cosine LR, head lr 1e-4 and VLM lr 1e-5 [V] ([DiscreteRTC](https://arxiv.org/html/2604.25050)).

## Sample discrete tokens, learn everything else in metric space

Per-axis coordinate tokens, two per waypoint, are the converged design for 4 s, 8-waypoint plans. ReflectDrive-2 uses exactly 16 tokens with the final (x8, y8) pair as the goal [V], and heading can be derived from the waypoints [V].

Decode the goal first. WAM-Diff scores 91.0 reverse-causal, 90.0 random and 88.9 causal [V]. MD-Drive scores 86.5 reverse-causal against 81.4 causal, and its per-sample distance-aware order scores 88.9 against 84.3 for confidence-only ordering [V]. In robotics, confidence-ranked unmasking beats parallel unmasking (96.8 vs 95.6) and a decaying Gumbel temperature beats argmax (96.8 vs 96.2) [V]. TS-Mask VLA's temporal-then-spatial masking lifts LIBERO-Long from 85.0 to 91.6 [V] ([TS-Mask VLA](https://arxiv.org/html/2607.09818)). DCT/BPE compression is the wrong tool at 2 Hz: AutoVLA's FAST tokens score 67.63 against 80.54 for physical tokens [V].

Hard bins stop helping well before centimetre resolution. MD-Drive's supervised-only sweep scores 68.5 with no discretization, 78.2 at 0.5 m, 81.4 at 0.25 m and 80.8 at 0.1 m [V]. With uniform bins of width Δ and hard rounding, the 2D RMS error is Δ/√6 ≈ 0.408Δ [D]. That gives 0.10 m at Δ = 0.25 m, 0.041 m at 0.1 m, 0.020 m at 0.05 m and 0.004 m at WAM-Diff's 0.01 m.

MD-Drive's optimum at 0.25 m partly reflects the cost of extending an LLM vocabulary: +580 tokens at 0.25 m and +1300 at 0.1 m [V]. A separate head pays almost nothing for extra classes. Over DiffusionDrive's data range (x 56.9 m, y 46 m [V/D]), 0.05 m bins need about 1,138 + 920 = 2,058 classes and 0.1 m bins about 569 + 460 = 1,029 [D].

More classes still cost something in learning. MVLAD-AD's joint codebook got worse from 256 to 384 entries (L2 1.28 → 2.76 m) [V], and E0 saturates near 2,048 bins [R] ([E0 project page](https://doo-mon.github.io/e0web/)). **Per-axis bins of 0.05–0.1 m put quantization error at 2–4 cm RMS**, below anything PDMS resolves. Resolution is not where the remaining points are.

Continuity pays where geometry enters the model, and metric-aware embeddings matter most. WAM-Flow's metric-aligned embeddings add **+2.3 PDMS** (81.1 → 83.4) [V]. Random embeddings nearly double MVLAD-AD's L2 (1.28 → 2.39 m) [V] ([MVLAD-AD](https://arxiv.org/html/2602.20577)). Soft targets help modestly: E0's one-hot smoothing at α = 0.1 adds about 1.4% success [V]. No VLA ablates Gaussian or ordinal neighbour labels.

Discrete-WAM, the top masked v1 number at 92.2, stacks four continuous pieces [V]:

- bilinear soft labels over 4 neighbouring prototypes
- a 0.02 m auxiliary position classifier
- a trajectory regression loss
- "mode-aware decoding", which fits a 1–3-component mixture to the categorical, picks one mode and takes the expectation inside it, avoiding "decoding-induced mode averaging"

No ablation isolates any of these [V]. Its acceleration-space tokens explain why it needs them: a constant 0.067 m/s² rounding error integrates to about 0.53 m by 4 s [D], which argues for position tokens. DiscreteRTC trains its discrete head with cross-entropy plus an auxiliary L1 (λ = 0.1) [V].

The VQ-BeT authors call their continuous offset on a discrete code "quite important", but show its size only in a figure (statement [V]). VQ-BeT beats BeT 0.78 vs 0.39 on PushT [V] ([VQ-BeT](https://arxiv.org/html/2403.03181)). No driving paper tests a token plus residual on NAVSIM. If a residual is added, keep it deterministic so the RL likelihood stays purely categorical; this is a design suggestion, not a published result.

The continuous family reaches the same split from the other side. DiffusionDrive's anchors are a discrete choice of mode, refined continuously, and worth +3.5 over vanilla diffusion [D]. GoalFlow picks a goal from a 4,096–8,192-entry vocabulary and flows toward it, gaining +2.9 from the goal score map and +4.7 with the full scorer pipeline [D]. ResAD predicts residuals against an inertial reference, and its point-wise normalization adds +0.9 [V] ([ResAD](https://arxiv.org/html/2510.08562v2)). E0 puts Gaussian noise on one-hot discrete actions [V].

Every best-in-paper design uses discrete or vocabulary structure to place the mode and metric or continuous structure to refine it. A masked head over fine bins with metric embeddings is already that hybrid, approached from the discrete side.

**Table 3. The first head, specified from the evidence**

| Component | First-build choice | Evidence |
|---|---|---|
| Trunk | Cross-attention DiT, about 16 layers × 384 hidden (≈35M); AdaLN for ego state; past poses as extra tokens | ReCogDrive planner [V]; DiscreteRTC AdaLN DiT [V] |
| VLM read | Several cached Qwen layers via layerwise cross-attention, with a last-layer-only control | DiffAdapterVLA 80.2 vs 85.2 [V]; GR00T N1 middle layer [V]; TS-Mask all-layer bridge [V] |
| Tokens | 16 per-axis tokens (x, y per waypoint), no heading | ReflectDrive-2 [V] |
| Resolution | Uniform 0.05–0.1 m bins over a data-fitted range (about 1,000–2,000 classes) | MD-Drive sweep [V]; arithmetic [D] |
| Embedding | Linear-in-value or sinusoidal metric embedding, plus a [MASK] entry | WAM-Flow +2.3 [V]; MVLAD-AD [V] |
| Targets and losses | Cross-entropy with Gaussian or bilinear neighbour soft labels, plus auxiliary L1 on the expected position | E0 smoothing [V]; Discrete-WAM [V]; DiscreteRTC λ = 0.1 [V] |
| Readout | Expectation within the selected mode, for sub-bin precision | Discrete-WAM mode-aware decoding [V] |
| Decoding | Goal pair first; 4 confidence-ranked steps; decaying-temperature Gumbel in fp64; selective commit | WAM-Diff and MD-Drive orders [V]; DD-VLA [V]; Zheng et al. [R]; Discrete-WAM [V] |
| K plans | 3–6 goal samples, BEV NMS near 1 m, 1–2 drafts per goal, batched over one cache | ReflectDrive, ReflectDrive-2 [V] |
| RL | GRPO on exact realized-order token log-probs, mean-only advantages, G of 3–8, advantages grouped within a goal | DriveFine, WAM-Diff, NoRD [V]; the likelihood choice is a design inference |
| Numerics | Head in fp32, cosine LR, head lr 1e-4 | DiscreteRTC [V] |

## One shared-trunk swap decides the family

"DiT versus masked" is less of a fork than it looks. DiscreteRTC runs a discrete variant and a flow variant on one 36-block cross-attention DiT with AdaLN, where "only the input and output layers differ". Its discrete variant uses nn.Embedding(257, 2048) including [MASK], 256 bins and 8 MaskGIT steps [V]. Build the trunk once, with layerwise cross-attention to several cached Qwen layers, AdaLN for ego state and ReCogDrive's 16 × 384 sizing. The masked and continuous heads then become two pairs of input and output layers on shared code.

Settle how the head reads the VLM first. In DiffAdapterVLA, reading only the final cache scores 80.2 against 85.2 for late-layer conditions. That **+5.0** swing [D] is larger than any head-type gap on NAVSIM. Layer count is non-monotonic there: 4 layers give 73.0, 8 give 84.1, 12 give 86.9 and 16 give 82.1 [V]. GR00T N1 reads its 12th LLM layer for "faster inference speed and higher downstream policy success" [V] ([GR00T N1](https://arxiv.org/html/2503.14734)). SmolVLA improves from 75.0 to 80.3 as it reads more layers [V] ([SmolVLA](https://arxiv.org/html/2506.01844)). Our backbone was unfrozen during text SFT, so its last layer is already task-trained, but the probe is still cheap. No study compares AdaLN ego conditioning with token conditioning.

Arm A is the masked head specified in Table 3. Arm B is the continuous comparator, and it should not be a vanilla ε-prediction DiT started from N(0, I). That configuration fails two project requirements: it collapses below about 5 steps [V], and it gives near-duplicate samples [V]. Build arm B as an anchored, truncated x0-prediction head: about 20 k-means anchors, DiffusionDrive's noise levels and 2 DDIM steps. This setup has the only NAVSIM evidence for few-step RL (DiffusionDriveV2 88.1 → 91.2 with selector [V]). A rectified flow from a σ ≈ 0.1 start is the alternative. Train both arms on the same cached states, data, epochs and parameter budget.

Judge the swap on the full official navtest with NAVSIM's own scorer, in two stages. After SFT, record single-sample PDMS, oracle best-of-6 and best-of-20, and one diversity metric applied identically to both arms; expect single-sample PDMS within a point. Then run a matched short RL stage with the same G, epochs and scorer budget and mean-only advantages. Group advantages within a goal token for arm A and within an anchor for arm B. Record the same numbers again, plus the v2 EPDMS sub-terms.

The deciding readout is **post-RL oracle best-of-6 together with post-RL single-sample PDMS**. That pair measures what the learned selector will have to work with once RL has narrowed the distribution. Keep the masked head if it matches arm B within run-to-run noise on single-sample PDMS and holds equal or higher best-of-6 at 4 or fewer steps. Switch to arm B if arm B wins post-RL by more than noise, or if the masked head needs more than about 5 steps to match. The only published seed spread is MD-Drive's ±0.30 and ±0.23 over 8 seeds [V], and DriveVLA-W0's head comparison is a single run [V]. Treat single-run gaps under roughly half a point to a point as ties.

Inside arm A, a short ladder answers "how continuous". Start from hard one-hot labels at 0.1 m. Then add, one at a time: Gaussian neighbour labels; an auxiliary L1 on the expected position with a local expectation readout; and a deterministic per-token residual. Run 0.05 m and 0.25 m bins as side arms. Each rung is a loss or readout change on one trained trunk. None has a published isolating ablation in driving, so the ladder produces new data as well as checking the design.

The strongest case for building the continuous head first is ReCogDrive. It is nearly our exact design (a 35M DiT over VLM hidden states, AdaLN ego, 5 steps), it has public code and a NAVSIM RL recipe, and DriveWLA's frozen-VLM flow head reaches 92.3 after head-only RL [V]. If RL plumbing turns out to be the bottleneck, that path carries less risk. But unless it is anchored it fails the few-step and K-plan requirements, and the shared trunk makes switching cheap, so starting masked costs little either way.

Applied to our 85.6, the cross-paper pattern stacks three gains. The supervised gain is small (+0.1 to +2.4 in ReCogDrive), RL adds +2 to +4.4, and a learned selector adds about +1. That puts a single-sample post-RL target near **89–91 PDMS**, with oracle best-of-6 roughly 3–4.5 points higher. This is an extrapolation from 1.5–8B models, not a measurement. ReflectDrive-2, a 0.7B masked-diffusion LM plus a 0.1B ViT that reaches 91.0 after RL [V], shows the size class can get there. It did so from proprietary weights, with diffusion run through the whole model.

### Recorded source conflicts

The notes record the following conflicts. None reverses the recommendation, but several change which number or hyperparameter to copy.

| Source | Recorded conflict | Bearing here |
|---|---|---|
| ReCogDrive | Paper Table 1 puts 90.8 under InternVL3-8B; README gives 90.8 to the 2B Base and 90.4 to the 8B Large. Noise clip ±5.0 in the text vs ±3.0 in the table. BC weight 1e-2 in Appendix Table 12 vs 0.001 as the best ablation row. Group size G is never stated. | Which backbone set the continuous-head RL record; which RL settings to copy |
| DriveFine | 91.9 in the text vs 91.8 in the table; 90.8 in Table 4 vs 90.7 in Table 1; group size 10 used, but 8 reported as best; v2 EPDMS 87.1 in the table vs 86.7 in the text | Masked post-RL figures move by ±0.1 |
| WAM-Diff | Text claims +5.3 from GSPO; Table 5 gives +4.4 | Size of the masked RL gain |
| ReflectDrive-2 | HTML reports a step plateau at "33–55 steps" against a 1+3+3 setting; the page's digit-doubling pattern suggests 3–5 (inference) | Masked step budget |
| ReflectDrive | No RL stage could be found; the arXiv version is SFT plus inference-time reflection | Its 91.1 is search, not RL |
| DiffE2E | Its own Table 2 says 92.7; DIVER quotes 89.8 | Upper continuous row; excluded from like-for-like comparison |
| GoalFlow | "1.6% drop" in the intro vs a 1.4-point drop in the table; resolved as a relative drop (1.4/90.3) | None |
| DiffusionDrive | "64%" diversity gain in the text vs 63 points in the table; 3-step row EP 92.2 vs 82.1–82.2 elsewhere, likely a typo | None material |
| GuideFlow (navhard) | "+1.3 over prior best" contradicts its own DiffVLA row at 45.0; GitHub "Private NavHard" 49.4–51.3 [R] vs 43.0 in the paper | navhard only |
| Alpamayo-R1 | 5 flow steps in the latency table vs a default of about 10; AR uses "6 tokens" in the text vs 127 in Table 14; relative decode speed 1.16× vs 222/8.75 ≈ 25× | Flow-vs-token speed ratio |
| TrajHF | v1 abstract claims 93.95 [R] vs "comparable" in v2; meanADE rises 0.7734 → 2.5843 despite "without significant degradation" | Trust in RLHF stability claims |
| FAST vs KI | FAST calls pi0-FAST "comparable" on LIBERO; KI's table has LIBERO-Long 60.2 vs 85.2 for pi0 | AR tokens vs flow on small data |
| MP1 vs FlowPolicy | FlowPolicy reports 70.0% itself vs 71.6% in MP1's re-run | One-step flow baselines |
| Hydra-MDP vs DiffusionDrive | Transfuser listed at 78.0 vs 84.0, likely different metric versions | Cross-paper absolute values |
| E0 | Text on τ bias contradicts Eq. 4; LIBERO baseline cells equal the published openpi numbers despite a claimed "same setup" | Check the code before reimplementing continuized discrete diffusion |
| DD-VLA | "Highest score across seeds" in one place vs "3-seed mean" in another; OFT-Diffusion Goal cell 96.2 vs 95.5 in the OFT paper | Uncertainty in the main robotics C1 row |
| ReinFlow | Wall-time saving vs DPPO quoted as 82.63%, 62.82% and >50% | None |
| MD-Drive | Speed claim cites "a 2.0-point drop" with no stated reference or hardware | None |
| GR00T N1.7 | README says the DiT went from 32 to 16 layers; the benchmark checkpoint has 32 blocks | None |

## Conclusion

The evidence changes what the question is about. Discrete versus continuous looks like a choice of model family, but at 16 numbers per plan the variables that decide the outcome are elsewhere: where sampling starts, how the head reads the VLM, and how RL groups its samples. In each family, the best-in-paper variant is a hybrid that uses discrete structure to place the mode and metric structure to refine it. The prior of masked first and DiT second holds, with two amendments. The masked head should be metric in everything except its sampled tokens. The DiT comparator should be anchored, not vanilla; otherwise the comparison tests a configuration the literature has already ruled out for few steps and K plans.

This project can also produce data the literature lacks. No NAVSIM paper compares masked and continuous heads on one backbone. None puts a separate masked head on an autoregressive VLM, none isolates soft labels or a token residual for trajectories, and none measures diversity for masked plans. A same-trunk swap with matched RL answers all four at once. The warning to carry forward comes from MD-Drive, whose over-fitted checkpoint gained 1.5 points from RL where the earlier one gained 8.6. Police the "near-perfect train fit" target with validation pass@K, not train loss.
