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

**Why smoothness matters (D):**
- In PDMS, comfort is 2/12 (17%) of the weighted part.
- In EPDMS, HC + EC are 4/16 (25%), and both are pass/fail.
- In other words, smoothness and frame-to-frame consistency carry about a quarter of the weighted EPDMS.

## 2. What this means for the output (D unless marked)
- **Horizon: exactly 4 s.** MVLAD-AD predicts 10 waypoints (5 s), so 2 of its 10 action tokens are wasted compute. A shorter horizon gets its last pose held and reads as stopping.
- **Rate.** The scorer resamples to 10 Hz with straight lines between our poses, so finer output only helps if it carries real smoothness. A smooth parameterization can use **fewer** tokens than 8 poses:

  | Output design | Action tokens | Smooth? | Notes |
  |---|---|---|---|
  | **A.** 8 poses at 2 Hz (NAVSIM default) | 8 (joint codebook) or 16 (separate x and y) | Kinks at each pose | The baseline |
  | **B.** 4 poses at 1 Hz + a spline to 4 s | 4–8 | Yes | Fewer tokens; needs a curvature check |
  | **C.** Frequency domain: K DCT coefficients per axis (FAST-style) | 2K (K ≈ 3–6) | Yes, low-pass by construction | Matches the collaborator's frequency-domain idea; ties to C2 |
  | **D.** Control space: acceleration + curvature per 0.5 s, integrated with a bicycle model | 16 | Dynamically feasible by construction | Comfort and HC become directly controllable |
  | **E.** 40 poses at 10 Hz | 40–80 | Only if learned | 5× the tokens; probably not worth it |
- **The first 0.5 s must match the current speed and acceleration.**
  - The scorer starts from the real ego state and runs a straight line to the first pose.
  - A first pose that implies a different speed creates an acceleration spike, which fails C and HC.
  - Options: predict residuals from a constant-velocity or constant-acceleration roll-out, or condition strongly on ego status.
- **Heading.** The tracker uses heading error, so headings must agree with the path. Derive them from the path, as `navsim_eval/predictions.py` does, or predict them consistently.
- **Post-processing smoothing** (the collaborator's idea).
  - Methods:
    - a smoothing spline or Savitzky–Golay filter on the poses;
    - or a small optimisation that fits acceleration and curvature to the poses with a jerk penalty (the same idea as the tracker's reference fit).
  - It costs almost no compute.
  - Risk: over-smoothing cuts corners, which costs drivable-area and collision scores. Tune it on a navtrain dev split, never on navtest.
- **Consistency across frames (EC):**
  - decode deterministically: confidence-ordered unmasking with no random sampling;
  - train with a consistency loss between plans for frames 0.5 s apart (navtrain frames are consecutive);
  - test whether coarser tokens make plans steadier.

## 3. What this means for the input (D)
- **Ego history: 4 poses at 2 Hz (1.5 s).** That is what NAVSIM gives; MVLAD-AD was trained on 10 poses. Alternatively, summarise it as speed, acceleration and yaw rate, which is fewer tokens and serves HC directly.
- **Ego status and command:** current speed, acceleration and the 4-way driving command, all test-time visible (V).
- **Image:** the front camera at a fixed visual-token budget. Ablate the 3-camera stitch.

## 4. Experiments, cheapest first (D)
1. **Ceiling of each output design, no training (CPU only).**
   - Encode the human trajectories of a navtrain dev split with each design: A–E, plus codebook sizes 256 / 1,024 / 2,048 and coarse + fine tokens.
   - Decode them and score the result with NAVSIM's scorer on a navtrain metric cache.
   - This measures what each representation costs before training anything. It includes the shipped 256-token codebook (0.54 m rounding).
2. **Post-processing on existing predictions.** Re-score MVLAD-AD's zero-shot NAVSIM predictions (workstream A) with and without smoothing and first-segment correction. This shows how much C/HC/EC alone is worth.
3. **Train the small pilot with the best 2–3 designs from (1).** Report score vs action tokens × denoising steps.
4. **EC study:** deterministic vs sampled decoding, with and without the consistency loss.

Steps 1 and 2 need only the NAVSIM devkit and a metric cache, not GPUs, and can run beside the workstream-A setup.

## 5. Collaborator suggestions, 2026-10-22 (context, not direction)
- **Research angle:** efficiency and explainability, rather than absolute planning performance, where competition is fiercer (WAM-Diff, DriveFine, ReflectDrive / -2).
  - Our efficiency work (C2, C3) and the reasoner's narration map onto this.
- **Smoothing post-processing:** experiment 2 above.
- **Frequency-domain trajectories:** design C above.
