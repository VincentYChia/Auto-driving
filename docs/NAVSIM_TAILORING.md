# Tailoring inputs and outputs to the NAVSIM score (2026-10-22)

**Objective [agreed 2026-10-22]:** as a starting point, get the highest NAVSIM score under constrained compute. Tailor inputs and outputs to NAVSIM, including the output rate (Hz) and smoothing.

**What "constrained compute" means here [proposed]:**
- The pilot is a small (~2B) model.
- Compute is counted as visual tokens + action tokens × denoising steps, plus measured latency.
- Scores are reported against compute. There is no fixed device budget (C3, agreed 2026-10-20).

Evidence marks (V / R / D) as in `docs/RESEARCH_DIRECTION.md`. Facts marked V were read in NAVSIM's `main` branch and in nuplan-devkit on 2026-10-22.

## 1. How NAVSIM turns a planned trajectory into a score (V)
1. **The agent returns poses** `(x, y, heading)` in the current rear-axle frame.
   - The default is 8 poses at 0.5 s (4 s). An agent may declare its own sampling, because the scorer reads the trajectory's `trajectory_sampling`.
2. **The poses are made absolute,** and the current ego state is prepended. Velocity and acceleration are set to zero: "velocity and acceleration ignored by LQR + bicycle model".
3. **The trajectory is interpolated to 10 Hz** (40 poses × 0.1 s) by nuPlan's `InterpolatedTrajectory`.
   - Positions use linear `interp1d`; headings use an angular interpolator.
   - Query times are clipped to the trajectory's end, so a horizon shorter than 4 s gets its last pose held.
4. **An LQR tracker plus a kinematic bicycle model simulate the plan** for 40 steps.
   - The tracker looks 1 s ahead.
   - Its reference speed and curvature are fitted to the poses, with small jerk and curvature-rate penalties (1e-4 and 1e-2).
   - Below 0.2 m/s it switches to a stopping P-controller.
5. **The simulated states are scored, not the raw poses:**

| Score | Formula (weights) |
|---|---|
| PDMS (v1) | NC × DAC × (5·TTC + 5·EP + 2·**C**) / 12 |
| EPDMS (v2) | NC × DAC × DDC × TLC × (5·EP + 5·TTC + 2·LK + 2·**HC** + 2·**EC**) / 16 |

**The comfort terms:**
- **C / HC (comfort):** every simulated step must stay inside these bounds. HC also prepends the human's past trajectory at 10 Hz, so the plan must continue the car's actual motion.

  | Quantity | Bound |
  |---|---|
  | Longitudinal acceleration | −4.05 to 2.40 m/s² |
  | Lateral acceleration | ±4.89 m/s² |
  | Jerk | ±8.37 m/s³ |
  | Longitudinal jerk | ±4.13 m/s³ |
  | Yaw rate | ±0.95 rad/s |
  | Yaw acceleration | ±1.93 rad/s² |
- **EC (extended comfort):** compares this frame's plan with the plan made 0.5 s earlier, over the overlapping 3.5 s.
  - The root-mean-square differences must stay within 0.7 m/s² (acceleration), 0.5 m/s³ (jerk), 0.1 rad/s (yaw rate) and 0.1 rad/s² (yaw acceleration).
  - It is all-or-nothing per frame.

**Weights are not where points are lost.** Published sub-scores and SimScale's official per-scene CSVs show:
- **Single-plan comfort is already saturated.**
  - v1 Comfort is 99.3–100 for almost every method (R).
  - v2 History Comfort is ~97–98 (R).
  - Smoothing one plan has little left to win.
- **Imitation planners lose most on Ego Progress and Drivable Area.** Recomputed from SimScale's per-scene CSVs (D):
  - LTF and DiffusionDrive lose ~3.9 EPDMS to EP (about the human ceiling, EP 87.4) and 2.5–3.8 to DAC.
  - They lose only ~1.1–1.2 to EC.
- **Extended Comfort (consistency between consecutive plans) is the comfort term that moves:**

  | Planner / setting | EC result |
  |---|---|
  | Human logs | Pass only 90.1% (R) |
  | Published navtest EC values | 59–92 (R) |
  | Vocabulary-scorer planners | EC is their largest single loss: 4.0–4.4 points, with 41–45% of frames failing (D) |
  | navhard | Stage-1 EC 31–77; stage-2 EC 31–43 and stage-2 Lane Keeping 58–62 are the weakest terms (D) |
- **Pin the devkit commit.** NAVSIM fixed a human-filter bug on 2025-09-29 (commit `359c7f7`, V). It moves EPDMS by +2 to +3.8, which is more than most gaps between methods (R/D). Report which evaluator was used.

## 2. What this means for the output (D unless marked)
- **Horizon: exactly 4 s.** MVLAD-AD predicts 10 waypoints (5 s), so 2 of its 10 action tokens are wasted compute. A shorter horizon gets its last pose held and reads as stopping.
- **Rate is not the bottleneck** (evidence above).
  - Most planners, including the strongest, output 8 poses × 0.5 s (V, code), and 40 × 0.1 s shows no comfort advantage (R, confounded).
  - Pick the representation for **tokens and accuracy**, not smoothness. A common budget is 8 poses × (x, y[, heading]) = 16–24 tokens. A smooth parameterisation could use fewer:

  | Output design | Action tokens | Notes |
  |---|---|---|
  | **A.** 8 poses at 2 Hz (NAVSIM default) | 8 (joint codebook) or 16 (separate x and y) | The baseline |
  | **B.** 4 poses at 1 Hz + a spline to 4 s | 4–8 | Fewer tokens; needs a curvature check |
  | **C.** Frequency domain: K DCT coefficients per axis (FAST-style) | 2K (K ≈ 3–6) | The collaborator's idea. On its own it doesn't buy consistency: DriveVLA-W0 uses FAST and has the lowest published EC (58.9), though that is confounded (R) |
  | **D.** Control space: acceleration + curvature per 0.5 s, integrated with a bicycle model | 16 | Feasible by construction |
  | **E.** 40 poses at 10 Hz | 40–80 | 5× the tokens; no shown benefit |

  Prior representations on NAVSIM, all R:
  - Bézier control points inside a vocabulary: +3.25 PDMS over text waypoints (GRAVA).
  - 2 latent tokens: no loss (WALT).
- **Denoising steps: 2–4 look viable** (R).
  - DriveFine reaches ~90.5 PDMS with 4 steps at 207 ms.
  - WAM-Flow defaults to 5 steps; MVLAD-AD uses 2–10.
  - This supports a ~16-token, 2–4-step head [proposed].
- **The first 0.5 s must match the current speed and acceleration.**
  - The scorer starts from the real ego state and runs a straight line to the first pose.
  - A first pose that implies a different speed creates an acceleration spike, which fails C and HC.
  - Options: predict residuals from a constant-velocity or constant-acceleration roll-out, or condition strongly on ego status.
- **Heading.** The tracker uses heading error, so headings must agree with the path. Derive them from the path, as `navsim_eval/predictions.py` does, or predict them consistently.
- **Post-processing smoothing** (the collaborator's idea).
  - It is cheap, but expected gains are small, because single-plan comfort is already ~99–100 (above).
  - It may still help a model whose raw waypoints are noisy, e.g. MVLAD-AD's codebook rounding, so it is worth one cheap check (§4.2).
  - Risk: cutting corners costs drivable-area and collision scores. Tune it on a navtrain dev split, never on navtest.
- **Consistency across frames (EC) is the comfort lever that pays.** Every measured gain makes the new plan agree with the previous one or with a single mode (R):
  - re-ranking candidates against the previous plan: EC 77.0 → 92.8, EPDMS +1.6;
  - an anchor: EC 35 → 75.5.

  For a masked-diffusion head [proposed]:
  - decode deterministically;
  - condition decoding on the previous plan, shifted by 0.5 s, as soft anchors or as an inpainting prompt;
  - train with a consistency loss between frames 0.5 s apart.

  Caveat: NAVSIM's `compute_trajectory` does not receive the agent's own previous output (V). Previous-plan conditioning needs an internal cache and frames processed in log order. Whether the leaderboard allows that is not checked.

## 3. What this means for the input (D)
- **Ego history: 4 poses at 2 Hz (1.5 s).** That is what NAVSIM gives; MVLAD-AD was trained on 10 poses. Alternatively, summarise it as speed, acceleration and yaw rate, which is fewer tokens and serves HC directly.
- **Ego status and command:** current speed, acceleration and the 4-way driving command, all test-time visible (V).
- **Image:** the front camera at a fixed visual-token budget. Ablate the 3-camera stitch.

## 4. Experiments, cheapest first (D)
1. **Ceiling of each output design, no training (CPU only).**
   - Encode the human trajectories of a navtrain dev split with each design: A–E, plus codebook sizes 256 / 1,024 / 2,048 and coarse + fine tokens.
   - Decode them and score the result with NAVSIM's scorer on a navtrain metric cache.
   - This measures what each representation costs before training anything. It includes the shipped 256-token codebook (0.54 m rounding).
2. **Post-processing on existing predictions.** Re-score MVLAD-AD's zero-shot NAVSIM predictions with and without smoothing and first-segment correction. Expect small gains, given saturated comfort, unless the raw waypoints are noisy.
3. **Train the small pilot with the best 2–3 designs from (1).** Report score vs action tokens × denoising steps.
4. **EC study:** deterministic vs sampled decoding, previous-plan conditioning, and a consistency loss. On navhard, EC and lane keeping are the weakest terms.
5. **Always** report the devkit commit, and both EPDMS variants (before and after the 2025-09-29 human-filter fix) when comparing with published numbers.

Steps 1 and 2 need only the NAVSIM devkit and a metric cache, not GPUs, and can run beside the workstream-A setup.

## 5. Collaborator suggestions, 2026-10-22 (context, not direction)
- **Research angle:** efficiency and explainability, rather than absolute planning performance, where competition is fiercer (WAM-Diff, DriveFine, ReflectDrive / -2).
  - Our efficiency work (C2, C3) and the reasoner's narration map onto this.
- **Smoothing post-processing:** experiment 2 above.
- **Frequency-domain trajectories:** design C above.
