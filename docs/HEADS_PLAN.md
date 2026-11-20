# Path-generator study: plan to Mon 2026-11-30, with the reasoning behind each step

Proposed 2026-11-19, for the researcher's review. **No new training runs start until it is approved**; the two running layer-test jobs finish either way, because they are part of phase 1. Terms are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal:** the best single-path NAVSIM score we can reach, without breaking scientific integrity. Integrity means:
- every choice is made on data the final number is not measured on;
- every method gets the same budget and the same chance to be tuned;
- every result is reported, including the losing and negative ones;
- every claim is larger than the measured noise.

## Rules that hold for every phase

1. **Choices on a validation split, results on navtest.**
   - Every choice below is a selection, and making it on navtest would bias the final navtest number upward and make the comparison with published papers unfair. The choices: stored layers, configuration, epoch, generation steps, sampling temperature, method, RL checkpoint, and whether to use the selector.
   - So choices are made on **navdev**: the 2,000 dev-v2 scenes from 11 nuPlan drives that appear in no training set (checked 2026-11-19: 0 of the 158 training drives).
   - navtest is used only to report, and it is reported for **every** configuration, not only the winners.
2. **The same budget for every method:** the same 236,133 training scenes, network body, epochs, batch size, learning rate and checkpoint-selection rule. Any extra tuning given to one method is given to all, or reported as unequal.
3. **Noise is measured, not assumed.**
   - Within one run, paired bootstrap confidence intervals over scenes.
   - Across runs, the spread over 3 seeds.
   - A difference counts only if its paired interval excludes 0 and it is larger than the seed spread. Otherwise it is reported as "not resolved".
4. **Decision rules are committed before the results they decide.** The commit time is the proof.
5. **Leakage audit before any score counts:** no navdev or navtest drive in the training data or in the RL scenes, and no NAVSIM warm-up scene in training.
6. **Every model is kept** (`pilot_sft/MODELS.md`), and every reported number comes from a committed aggregate file.
6b. **The same definitions for every method.**
   - **Single path** is each method's deterministic best guess: A, the most confident tokens (no sampling); D, the top-scored refined anchor; B and C, generation from the zero-noise start, which needs a small code change before phase 1. A single random sample is also reported for B and C.
   - **Best-of-6** uses each method's natural way of sampling 6 different paths. The one sampling setting that matters, A's temperature for its end-point draw, is chosen on navdev from the same small grid for every method that has one. Diversity is always reported next to best-of-6.
7. **Known biases, stated rather than hidden:**
   - H1 itself was chosen on navtest among about 15 supervised runs, so the starting point is mildly optimistic.
   - navdev is small (see phase 0), so it can resolve only differences of about a point or more.

## Phase 0 — integrity groundwork (Fri Nov 20, CPU only)

- **What:**
  - Build the NAVSIM metric cache for navdev on Anvil's CPUs. It needs the trainval logs that the navtrain-cache job unpacks.
  - Apply navtest's difficulty filter to navdev: the NAVSIM paper drops scenes where constant velocity already scores high (to be checked in NAVSIM's code before use). That makes navdev measure the same kind of scenes; expected size 1,000–1,500.
  - Score H1's text output and constant velocity on navdev as references.
  - Script the leakage audit.
  - Commit the decision rules below.
- **Why:** without a validation split, every later choice leaks into the reported number.
- **What it might reveal:** if H1 ranks differently on navdev than on navtest, our earlier navtest-based choices carry more bias than assumed. That gets reported.
- **Limit:** with about 1,200 scenes, paired differences under about 1 PDMS point cannot be resolved on navdev. Such choices fall to a pre-set tie-break (the simpler or cheaper option) and are marked "within noise".

## Phase 1 — each method at each layer setting, plus a regression control (Fri 20 – Sat 21, 10 training runs)

- **What:**
  - The base configuration of each method, A1, B, C and D, trained twice: reading stored layer 24 only, and reading layers 12, 18 and 24 (8 runs; A1's two are the jobs running now).
  - A **regression control R**: the same body, outputting the 8 waypoints directly with an L1 loss, one path, no generation (2 runs, one per layer setting).
  - Every run: seed 0, 20 epochs.
  - Each run's epoch is chosen on navdev by one rule for all: the best single-path PDMS among epochs 5, 10, 15 and 20.
- **Why:**
  - The best layers may depend on the method. Fixing them from A alone would hand A an advantage the others don't get, so each method keeps its own best layer setting.
  - R answers the question under everything else: does generation help at all, or do H1's features and the body do the work? DriveVLA-W0 found plain regression ahead of flow matching (88.4 against 87.2).
- **What it might reveal:**
  - whether method and layer interact;
  - whether H1's last layer, already fine-tuned for writing paths, carries all the needed information;
  - whether any method beats H1's text output before RL. Published work expects it to be close (+0.1 to +2.4).

## Phase 2 — how continuous (Sun 22, about 9 training runs, method A at its phase-1 layers)

- **What:**
  - The cumulative series A0, A1, A2, A3, A4, each adding one feature.
  - Then **leave-one-out** from the best A: that configuration with each of its features removed in turn.
  - Then bin size 0.25 m and 0.05 m on the best A configuration.
- **Why:**
  - A cumulative series measures each feature only on top of the ones added before it, so the order confounds the effects; leaving one out of the full system measures each feature's own contribution.
  - Bin size matters most when no other continuity feature compensates, so it is tested where it decides something: on the best configuration.
- **What it might reveal:** how much of a continuous method's advantage, if any, a discrete method recovers with metric features. That is the "how continuous" answer.
- **Limit:** one seed per configuration. Effects smaller than the phase-3 seed spread are reported as not resolved.

## Phase 3 — seed spread (Mon 23, 8–10 training runs)

- **What:** seeds 1 and 2 for each method's best configuration (A, B, C, D, and R).
- **Why:** it gives a measured tie threshold for the method decision and for every claim in phases 1–2.
- **What it might reveal:** how reproducible supervised path generators are on navtest. No such number is published for a masked-diffusion planner.

## Phase 4 — matched short RL stage (Tue 24 – Wed 25)

- **What:**
  - Each generating method's best configuration (A, B, C, D) starts from its phase-1/2 checkpoint, chosen on navdev by best-of-6.
  - One-paper evidence (MD-Drive) says an over-fitted checkpoint gains much less from RL; that is why the starting checkpoint is chosen by best-of-6.
  - Everything is matched: the same navtrain scenes (none from navdev or navtest drives), scene order, number of updates, 8 candidates per scene, learning rate and regularisation, and the same reward, NAVSIM's PDM score.
  - A and D, the deciding pair, get 3 seeds each; B and C get 1.
- **Measured:** single-path PDMS and best-of-6 on navdev (for the choice) and navtest (for the report), plus every sub-score.
  - A rise in the total with a fall in collisions or drivable area is reward hacking, and gets reported.
  - RL trains on sampled paths, so the deterministic single path is checked separately.
- **Why:** before RL the published methods land within about a point of each other, and RL adds 2–4 points, so the comparison that matters is after RL.
- **What it might reveal:**
  - whether some methods take RL better than others: no paper compares masked and continuous generators under one RL recipe;
  - what RL costs in GPU-hours and scorer calls.

## Phase 5 — choice of method (Thu 26)

- **Rule:**
  - Primary: post-RL single-path PDMS on navdev, as the mean over seeds. Secondary: post-RL best-of-6.
  - D, B or C replaces A only if it is ahead by more than the measured seed spread with a paired interval excluding 0. Otherwise A stays, the pre-set preference for the simpler method.
- **Then:** report every method's navtest numbers. With a clear winner, go on to phase 6; on a tie, wait for the researcher.

## Phase 6 — full RL and the selector (Fri 27 – Sun 29, only with a clear winner)

- **What:**
  - Train the chosen configuration with a longer RL stage (1–2 passes over navtrain).
  - Train the selector on candidate paths from navtrain scenes. It is adopted only if it beats the path generator's own top choice on navdev.
  - Caveat: the generator was trained on those same scenes, so its candidates there are better than on new scenes. The selector's navdev result shows whether that matters.
- **Final number:** one navtest evaluation of the final system, reported with the full list of choices that led to it.

## Mon 30 — report

- An HTML report on the home PC.
- In the repository: every number in `pilot_sft/README.md`, `docs/STATUS.md` and `docs/GAPS.md`.

## Compute

- **Phases 1–3:** about 27 training runs at 40 min to 2 h each, 20–50 GPU-hours on the 4090. Scoring: navdev about 30 s per path set; navtest 220 s.
- **Phase 4:** about 8 RL stages of 3–5 h each, limited by scoring on the desktop's 16 CPU cores.
- **Phase 6:** 1–1.5 days.
- **Priority if time runs short:** phases 0–5 (the comparison, the contribution) come before phase 6 (the headline number). Phase 6 moves into the next week rather than being cut short.
