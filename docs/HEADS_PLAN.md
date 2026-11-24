# Path-generator study: plan to Mon 2026-11-30, with the reasoning behind each step

Revision 2 (2026-11-19 21:10), after the researcher's review, **approved 2026-11-19 ~21:00** with two clarifications. They take precedence over anything below. Revision 1 is in git history.

**Clarification 1, equal tuning until a clear winner:** "Equal tuning means they should just both proceed till a clear winner is shown."
- There is no fixed decision date. Every generating method that is not clearly beaten keeps going through the same stages, with the same treatment: seeds, the RL stage, and each RL improvement.
- A method drops out only when it is clearly behind the leader on navdev: the drive-bootstrap interval of leader minus method excludes 0 (R3), seed-averaged where seeds exist.
- The repeated looks are choices on navdev. navtest's confirmatory tests (R4) run once, at the end, so the repeated looks don't inflate them.

**Clarification 2, quality over schedule:** "I would rather come back to comprehensive high quality work that is behind schedule then behind schedule work that is messy and unusable."
- The dates below are a guide.
- When the RL stage can be improved round after round (more updates, a better recipe, starting the next round from the last round's best checkpoint), it is, past the schedule if need be.
- Every RL improvement goes to every method still in contention, under the same settings, and is measured on navdev.
- Work is reported when it is right, not when the calendar says.

The two running layer-test runs (jobs 362, 363) are part of phase 1. They match the phase-1 rules exactly: seed 0, 20 epochs, every epoch's weights saved, the shared optimiser settings below. Terms are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal:** the best single-path NAVSIM score we can reach, without breaking scientific integrity. Integrity means:
- every choice is made on data the final number is not measured on;
- every method gets the same tuning budget;
- the statistics are fixed before the results;
- every result is reported, including R or H1's text output being good enough.

## Rules (committed at the end of phase 0, before any result they decide)

**R1. Choices on navdev, results on navtest.**
- navdev is the 2,000 dev-v2 scenes from 11 nuPlan drives. Checked 2026-11-19: none of the 158 drives of v1-hard appear in it, and v1-hard is H1's own fine-tuning data and every path generator's training data.
- navdev gets the same metric as navtest: NAVSIM v1.1 PDMS, the same scorer code and default config.
- It also gets navtest's difficulty filter, taken from NAVSIM's paper and code. The filter is checked by applying it to navtest, where it must keep at least 99% of scenes.
- navtest is used only to report and for the pre-specified confirmatory tests (R4). It is reported for every configuration.
- Choosing on navdev never biases the navtest number; navdev's noise only makes a choice less reliable.

**R2. The same tuning budget for every method.**
- The method decision uses each method's **phase-1 configuration** only: two layer settings each, an epoch chosen from 4, steps chosen from that method's grid, and one sampling setting chosen from a 3-point grid.
- Phase 2 (A's continuity series) is **analysis only** and does not change A's entry in the decision.
- After the decision, the winning method gets a bounded tuning budget for the final system (phase 6), labelled as post-decision tuning.

**R3. Statistics.**
- The unit is the per-scene PDMS.
- Intervals are paired, and **cluster-bootstrapped by drive**: resample drives with replacement, keeping all their scenes; 10,000 resamples, seed 0, 95% percentile interval.
  - navdev has 11 drives. navtest's scenes are grouped by drive, the date-and-vehicle part of the log name.
  - Scene-level intervals are also given, to compare with earlier results.
- Across seeds, a hierarchical bootstrap: each resample also redraws every method's seeds with replacement.
  - A method with one seed (B and C after RL) gets an added normal draw whose SD is the larger of A's and D's measured between-seed SDs. That is a conservative stand-in for the seed noise it lacks.
- **Choices** (layers, epoch, steps, sampling setting, RL checkpoint) take the navdev point estimate. **Claims** need the interval to exclude 0.
- navdev's resolution is measured in phase 0, as the drive-bootstrap interval width of a paired comparison, and reported before any choice. With 11 drives it will be wider than the "about 1 point" first guessed.

**R4. Confirmatory comparisons, named in advance and Holm-corrected at family-wise α = 0.05 on navtest.** Everything else (layer × method, the continuity series, bin sizes, B against C, steps, sampling settings) is exploratory: intervals reported, no significance claimed.
- C1 (primary): A against D, single-path PDMS after the RL stage.
- C2: the navdev-chosen method after RL against H1's text output.
- C3: the navdev-chosen generating method before RL against regression R: does generation help?
- C4: the chosen method before against after RL: does RL help?
- **Implementation** (added 2026-11-21 09:44, before any phase-4 run had started; `pilot_sft/heads/confirmatory.py`). R16 came after R4 and made phase 4 run two recipes and R5 report only, so the wording is fixed here:
  - All four use navtest's single-path PDMS and R3's hierarchical bootstrap: drives with all their scenes, and each method's seeds; a one-seed path generator gets the larger of A's and D's between-seed SDs as an added draw.
  - "After the RL stage" is phase 4 under the registered recipe, RL4, each RL seed at its R11 checkpoint (update 0 allowed). RL6's results are reported beside them, not tested.
  - "The navdev-chosen method" is R5's leader under RL4 (`choices/decision_phase5.json`), computed as registered although R16 no longer lets it decide the final system.
  - C2's H1 is one fixed model, so it gets no seed draw. C3 compares phase 3's seeds with R's. C4 pairs each RL seed with the supervised seed it started from.
  - Each p-value is two-sided: twice the smaller bootstrap tail at 0 (10,000 resamples, so a resolution of 1e-4). Holm's step-down procedure holds the family-wise error at 0.05. Each test's 95% interval is reported beside its p-value.

**R5. Method decision** (navdev, single-path PDMS after the RL stage, seed-averaged). Clarification 1 governs: contenders continue, with the same treatment, until this rule shows a clear winner. A tie means more of the same work for every contender, not an early stop.
- The method with the highest point estimate continues automatically only if its interval against every other generating method excludes 0.
- Otherwise it is a tie: **no automatic preference for A**. I report, and the researcher decides.
- My recommendation then uses, in this order: best-of-6 after RL on navdev, then generation time per scene.

**R6. Leakage audit, scripted, before any score counts.** Drives in each split:
- navdev and navtest against H1's fine-tuning data (v1-hard) and against every path generator's training data;
- navdev and navtest against the RL scenes;
- NAVSIM warm-up scenes against training.

The pre-training data of the Qwen3.5 base model cannot be audited; that is stated.

**R7. The same definitions for every method.**
- Single path is the deterministic best guess: A, most-confident tokens without sampling; D, the top-scored refined anchor; B and C, generation from the zero-noise start. B and C's random-sample path is also reported.
- Best-of-6 uses each method's sampling, with its one sampling setting chosen on navdev by best-of-6 from a 3-point grid:
  - A: temperature 0.5 / 1.0 / 1.5;
  - B: start-noise scale 0.5 / 1.0 / 1.5;
  - C: start-noise scale 0.5 / 1 / 2 times its training value;
  - D: starting noise of the anchors 0.5 / 1 / 2 times its training value.
- Diversity is reported with every best-of-6.

**R8.** Every model is kept (`pilot_sft/MODELS.md`), and every number comes from a committed aggregate file.

**R9. Known biases, stated:**
- H1 was chosen on navtest among about 15 supervised runs.
- navdev has only 11 drives.
- The selector's training candidates come from scenes the generator saw. Phase 6 measures this.

**R10. Choice procedure, fixed before any navdev score exists** (committed 2026-11-19 ~21:45; `pilot_sft/heads/driver.py` runs it).
- **navdev:** navdev_all is the 2,000 dev-v2 scenes, which share one metric cache. navdev is the navdev_all scenes passing navtest's filter: constant velocity scores PDMS ≤ 0.8 and the human path scores ≥ 0.8. On navtest's own scores this filter keeps 12,129 of 12,146 scenes (99.86%).
- **Drives:** navdev's 2,000 scenes come from 11 drives, and navtest's 12,146 from 44. A drive is the date-and-vehicle part of the log name.
- **Per training run, in this order, on navdev:**
  1. **Epoch:** the single path at the method's default steps, at epochs 5, 10, 15 and 20. Highest PDMS wins; a tie goes to the earlier epoch.
  2. **Steps:** at that epoch, the single path over the method's step grid. Highest wins; a tie goes to fewer steps.
  3. **Sampling setting:** at that epoch and steps, best-of-6 over the method's 3 settings. Highest wins; a tie goes to the training setting (1). R has no sampling and skips this step.
- **Per method:** the layer setting whose run has the higher navdev single-path PDMS at its own choices. A tie (equal to 4 decimals) goes to layer 24 alone, the cheaper setting.
- **Seeds 1 and 2** (phase 3) reuse seed 0's epoch, steps and sampling setting. Nothing is re-chosen per seed.
- **Phase-2 settings:**
  - A0 = A1 without distance-aware embeddings;
  - A2 = A1 plus soft labels (Gaussian, σ = 1 bin, cut at 3σ);
  - A3 = A2 plus an L1 loss, weight 1.0 in metres, on the local expectation (±3 bins around the arg-max), with the same averaged readout at inference;
  - A4 = A3 plus the deterministic sub-bin offset.
  - Leave-one-out from the series' best removes each of its added features once. A configuration that already exists is not retrained.
  - The bin sizes 0.25 m and 0.05 m are run on the series' best.
  - Every phase-2 run uses the same choice procedure.
- **navtest** (reporting) is scored per run at its navdev choices: the single path at every step count, and 6 samples at the chosen steps and setting.

## Phase 0 — groundwork (Fri Nov 20, CPU only)

- **navdev scoring:**
  - Extract only navdev's 40 log files from the OpenScene trainval archive already on Anvil. No need to wait for the navtrain-cache job's full unpack.
  - Build navdev's metric cache with NAVSIM v1.1 (Anvil CPU, `cis220051`) and copy it to the desktop.
  - Apply the difficulty filter, checked on navtest as in R1.
- **References on navdev:** constant velocity, H1's text output, and A1 from the layer test.
- **Statistics:** measure navdev's resolution (R3), and run the leakage audit (R6).
- **Code before phase 1's B, C, D and R runs:** regression R; B and C's zero-noise start; scoring and the drive bootstrap for navdev.
- **Commit R1–R9 and the phase-4 settings.** The commit hash goes in `docs/STATUS.md`.

## Phase 1 — each method at its own best layers, plus regression (Fri 20 – Sat 21, 10 training runs)

**Runs:** A1, B, C, D and R, each reading stored layer 24 only and layers 12, 18 and 24.
- **Shared settings:** seed 0; 236,133 training scenes; 20 epochs; batch 256; AdamW, learning rate 1e-4, betas (0.9, 0.95), weight decay 0.01, 1,000 warm-up steps then cosine to 0; gradient clip 1.0; bf16; every epoch saved; 12 blocks × 384 wide.
- **A1:** masked diffusion; 0.1 m bins (x 951, y 901 values); distance-aware bin embeddings; hard labels.
- **B:** Gaussian diffusion predicting the clean path, cosine schedule, DDIM.
- **C:** rectified flow from a start of standard deviation 0.1 (in normalised units), Euler steps.
- **D:** 20 anchors (k-means of 50,000 training paths), truncated diffusion up to t = 0.05, scored per anchor.
- **R:** 8 query tokens, L1 loss on normalised coordinates, one path.

**Choices on navdev**, by point estimate (R3):
- the epoch, from 5, 10, 15 and 20;
- the steps: A {1, 2, 4, 8}, B {1, 2, 5, 10}, C {1, 2, 5}, D {1, 2};
- the sampling setting (R7);
- the layer setting.

Every run's navtest scores are reported.

**Why:**
- The best layers may depend on the method, so each method keeps its own.
- R tests whether generation is needed at all. One paper found plain regression ahead of flow (88.4 against 87.2).

**What it might reveal:**
- an interaction between method and layer;
- whether H1's text-tuned last layer already holds what's needed;
- whether any path generator matches H1's text output before RL.

## Phase 2 — how continuous: analysis only (Sun 22, about 9 training runs, A at its phase-1 layers)

- **Runs:**
  - the cumulative series A0, A2, A3, A4 (A1 exists);
  - leave-one-out from the series' best (each added feature removed once);
  - 0.25 m and 0.05 m bins on the series' best.
- **Why:** the cumulative order confounds the effects; leave-one-out measures each feature's own contribution.
- **Not used** in the method decision (R2). If A wins in phase 5, the series' best becomes A's candidate for post-decision tuning in phase 6.

**R17. Reading phase 2 against seed noise** (committed 2026-11-20 23:12, before any continuity-series run had started).
- **The scale of seed noise.** A's three phase-1 seeds measure the training noise of one configuration. The between-seed SD is 0.50 on navtest single path and 0.44 on navdev (A1, phase 3).
- **The threshold.** A difference between two single-seed runs therefore carries a seed SD of about sqrt(2) x 0.50 = 0.71 on navtest [D]. A continuity-series effect is each run against A1, paired on navtest with its drive interval. It is called a measured effect only if its point estimate exceeds 1.4 points (about two seed SDs of the difference) and its drive interval excludes 0. Smaller effects are reported as within seed noise.
- **Extra seeds for the series' best.** If the series' best configuration is not A1, it gets seeds 1 and 2, with the same settings at its own R10 choices. A1 and the best are then compared over three seeds each with R5's hierarchical bootstrap. These are jobs `43{s}_`, after the series' leave-one-out and bin runs.
- **Cost** [D]: two training runs, about 2 h each.

## Phase 3 — seed spread (Mon 23, 10 training runs)

- **Runs:** seeds 1 and 2 of each method's phase-1 configuration (A1, B, C, D, R at their chosen layers, epochs and settings).
- **Gives:** the seed SDs used in R3 and R5, and the first published seed spread for a masked-diffusion planner.

## Phase 4 — matched short RL stage (Tue 24 – Wed 25)

**Settings, identical for every method:**

| Setting | Value |
|---|---|
| Algorithm | group-relative policy gradient (GRPO-style); on-policy, one gradient step per batch, so no ratio clipping; no value network; no KL or imitation term |
| Starting point | each method's phase-1 configuration, seed s, at its navdev best-of-6 epoch (MD-Drive: an over-fitted start gains much less from RL) |
| Scenes | the 80,626 navtrain scenes in our training data (v1-nt, part of v1-hard by construction; NAVSIM's navtrain lists 103,288; all on training drives, audited), shuffled once with seed 0, the same order for every method |
| Budget | 1,500 updates × 32 scenes = 48,000 scenes, about 0.6 of a pass over the 80,626 |
| Candidates | 8 per scene, sampled at the method's R7 setting |
| Reward | NAVSIM v1.1 PDMS of each candidate, the same code and config as evaluation, from the navtrain metric cache |
| Advantage | (reward − group mean) / (group SD + 1e-4); scenes whose 8 rewards are equal give no gradient, and are counted |
| Optimiser | AdamW, learning rate 1e-5, betas (0.9, 0.95), weight decay 0, 50 warm-up steps then constant, gradient clip 1.0, bf16 |
| Checkpoints | every 250 updates; the navdev-best of the 6 is chosen by the same rule for every method |

**Seeds:** A and D, 3 runs each (RL seed s starts from supervised seed s, so the spread covers both); B and C, 1 run each.

**Log-probability of a candidate** (method-specific by nature):
- A: the exact sum of log-probabilities of each token at the step it was committed.
- D: log-probability of the anchor choice, plus the Gaussian log-density of each stochastic refinement step (DDPM, η = 1).
- B: the Gaussian log-density of each stochastic DDPM step (η = 1), as in DiffGRPO/DDPO.
- C: the Gaussian log-density of each step of Flow-GRPO's SDE sampler. Its noise level is to be read from Flow-GRPO's code and committed in phase 0.

**Before any comparison run:** a 200-update trial on A1 seed 0 measures scorer speed and checks that the reward rises. The settings may change once, before phase 4, and the change is committed. The trial is in no comparison.

**Measured:**
- single-path PDMS and best-of-6 on navdev (choice) and navtest (report);
- every sub-score: a higher total with more collisions or more off-road driving is reward hacking, and is reported;
- the deterministic single path, checked separately, since RL trains sampled paths.

**Compute:** 384,000 scored paths per run, about 2 h at the measured 55 paths/s; 8 runs ≈ 16 h.

**R11. Phase-4 procedure, fixed before any RL comparison result** (committed 2026-11-19 ~22:15; the 200-update trial on a1-l24 is in no comparison).
- **Starting checkpoint:** per method, the epoch with the highest navdev best-of-6, at the chosen steps and sampling setting, on seed 0's run, among epochs 5, 10, 15 and 20. A tie goes to the earlier epoch. Seeds 1 and 2 start from the same epoch of their own runs.
- **RL runs:**
  - A and D: RL seed s from supervised seed s, for s = 0, 1, 2.
  - B and C: s = 0, once their policies exist: B, stochastic DDPM steps with the noise SD floored at 0.01 as for D; C, Flow-GRPO's SDE, its noise level read from the Flow-GRPO paper and code first.
  - R: no RL, since it has no stochastic policy. It stays a supervised control.
- **Evaluation:** each RL run saves checkpoints at 250, 500, …, 1,500 updates, and each is scored on navdev: the single path (temperature 0) and best-of-6 at the chosen setting.
- **B's and C's RL samplers** (fixed 2026-11-19 ~23:10, before any RL result):
  - **B:** stochastic DDIM (eta 1) from `scale` × unit noise, with the per-step noise SD floored at 0.01 as for D. The log-density is that of every step.
  - **C:** Flow-GRPO's SDE (arXiv 2505.05470; read 2026-11-19 from its paper and its code `sde_step_with_logprob`), with noise level a = 0.7, the code default (paper §5.3: "a=0.7"). Two adaptations, both needed for our flow:
    - Our start noise has SD c = 0.1, not 1. The score becomes −(z − τv)/(c²(1 − τ)), and σ_t = a·c·√(t/(1−t)), which keeps the SDE scale-equivariant. Here t = 1 − τ is Flow-GRPO's time, with 1 = noise.
    - σ and the drift's coefficient are taken at each step's midpoint, since σ_t is infinite at the noise end (Flow-GRPO's code avoids that point too).
  - Evaluation for both uses the deterministic samplers (DDIM eta 0, Euler), as in Flow-GRPO.
- **The RL checkpoint** is the one with the highest navdev single-path PDMS; a tie goes to the earlier checkpoint.
- **The decision** (R5) uses each method's seed-averaged navdev single-path PDMS at its RL checkpoint. navtest scores are reported for every checkpoint used.
- **Measured:** the RL scoring service equals NAVSIM's official per-scene scores, with a maximum difference of 0.0 on 200 navtest scenes. It scores 71.8 paths/s with 4 workers at 8 candidates per scene, so a 1,500-update run needs about 1 h of scoring.

**R12. Choosing the RL recipe** (committed 2026-11-19 ~23:30, after the trial and before any recipe run; this is the one change R11 allows before phase 4).

**What the trial found:** 200 updates on a1-l24, over the 2,662 RL scenes of 3 training drives, recipe as registered.
- On its own scenes, the second pass's reward (0.825) was no better than the first's (0.82).
- The policy became more aggressive: no-collision 0.964 → 0.915, time-to-collision 0.887 → 0.782, progress 0.828 → 0.871.
- It lost diversity: best-of-8 0.953 → 0.906, and scenes whose 8 rewards are all equal rose from 26% to 52%.
- **On navdev**, the single path fell from 81.18 to 75.58 after 100 updates (paired −5.6, drive interval −11.5 to −1.8) and to 73.97 after 200 (−7.2, −13.3 to −2.6). Collisions, drivable area and time-to-collision all got worse.
- The 3-drive scene set may explain part of this, but the reward did not improve even on the scenes it trained on.

**The candidate recipes**, the same for every method:

| Recipe | Learning rate | KL to the starting policy | Candidates × scenes per update |
|---|---|---|---|
| RL0 (as registered) | 1e-5 | none | 8 × 32 |
| RL1 | 1e-5 | β = 0.04 | 8 × 32 |
| RL2 | 3e-6 | β = 0.04 | 8 × 32 |
| RL3 | 3e-6 | β = 0.04 | 16 × 16 (the same 256 scored paths) |

- **The KL term** is exact along each sampled path, not estimated: for A, the categorical KL of each committed token's distribution at its commit step; for D, the anchor-choice categorical plus the Gaussian KL of each refinement step; for B and C, the Gaussian KL of each step. With an equal SD, the Gaussian KL is ‖μ_θ − μ_ref‖² / (2σ²).
- **Each recipe is run on A and on D:** their phase-1 configurations, seed 0, from their R11 starting epochs, 500 updates on the full navtrain cache.
- **Each run is scored on navdev** at 0, 250 and 500 updates, single path and best-of-6.

**The choice:**
- The recipe whose better checkpoint (250 or 500) gives the highest navdev single-path change from the start, averaged over A and D. A tie goes to the earlier recipe in the table.
- If every recipe lowers both methods' navdev single path, that is reported as a result in its own right: short RL did not help these path generators under our budget.
- The recipe runs are in no method comparison. Phase 4 then runs the chosen recipe afresh: A and D with 3 seeds each, B and C with 1.

**Change to R11:** the RL checkpoints considered are 0 (the starting point, no RL), 250, 500, …, 1,500. A method that RL only hurts keeps its supervised path generator, and how often that happens is reported.

**R13. A second recipe round, from the published RL methods** (committed 2026-11-20 07:02: after reading R12's RL0 and RL1 navdev scores, before reading any RL2 or RL3 navdev score, and before any R13 run).

**What R12 had shown when this was written** (navdev single-path PDMS, start → 250 → 500 updates):
- **A** (start: epoch 20, 81.43):
  - RL0 75.72 → 77.19; RL1 77.58 → 78.76.
  - Under RL0, progress rose (73.5 → 78.3) while time-to-collision (93.9 → 84.4), no-collision (97.5 → 92.3) and drivable area (91.8 → 88.7) fell.
  - The best-of-6 candidates' end points went from 6.4 m apart to 2.0 m.
- **D** (start: epoch 5, 77.10):
  - RL0 60.93 → 64.23; RL1 75.27 → 75.28.
  - Under RL0, every sub-score fell (at 250: drivable area −13.2, time-to-collision −13.1, progress −7.3).
- **On the training scenes,** A's sampled reward rose under RL2 and RL3 (0.79 → 0.85), while best-of-8 fell and equal-reward groups rose: the policy sharpened.

**Three likely causes, each with a published remedy:**
1. **The group SD.** Dividing by it turns tiny progress differences between safe candidates into full-size advantages, which pushes progress up and safety down: the pattern A shows.
   - Remedy: subtract the group mean only ("Dr. GRPO").
   - Evidence: NoRD, navtest, discrete path tokens: supervised 76.66 → GRPO 77.18 → Dr. GRPO 85.62. DriveFine also leaves the SD out.
2. **No tie to the logged driving.**
   - Remedy: an imitation (behaviour-cloning, BC) term next to the RL term.
   - Evidence: every published RL stage on a separate NAVSIM path generator has one: ReCogDrive, DiffusionDriveV2 (weight 0.1), DriveWLA (0.1), TrajHF (0.01), EponaV2.
3. **D's last refinement step.** It ends at noise level 0, so its SD is always the 0.01 floor, and its log-density gradient is about 100 times that of the anchor choice. D's gradient norm was 300–550, against 21–33 for A.
   - Remedy: a larger SD inside the log-probability only.
   - Evidence: DiffusionDriveV2 floors the SD at 0.1 inside log π "to avoid large-magnitude gradients"; ReCogDrive's minimum log-variance SD is 0.10.

**The candidate recipes.** Each starts from RL0 (learning rate 1e-5, no KL, 8 candidates × 32 scenes, 500 updates) and changes only what the table says:

| Recipe | Advantage | BC term |
|---|---|---|
| RL4 | reward − group mean | none |
| RL5 | (reward − group mean) / (group SD + 1e-4), as RL0 | yes |
| RL6 | reward − group mean | yes |
| RL7 | as RL0 | none |

- **SD floor (all four recipes):** in the log-probability (and KL) of every Gaussian step, the SD is max(SD, 0.1) in normalised path units. Sampling is unchanged.
  - This applies to D, B and C.
  - A has no Gaussian steps, so for A, RL7 is RL0 and is not run again.
- **BC term:**
  - It is the method's own supervised training loss, the phase-1 loss, on the logged paths of the update's 32 scenes.
  - Its weight λ is fixed per run so that, at the starting weights, its gradient norm equals the RL term's. Each norm is the mean over the first 4 batches, measured before any update and with a separate noise seed, so the first update's candidates are the same as in the other recipes. λ is logged.
- **Runs:**
  - A: RL4, RL5 and RL6.
  - D: RL4 to RL7.
  - Starts, seed, scene order and navdev scoring are the same as in R12.

**The choice:**
- R12's rule over all eight recipes, RL0 to RL7: the highest navdev single-path change from the start (the better of 250 and 500 updates), averaged over A and D. A tie goes to the earlier recipe.
- If every recipe lowers both methods' navdev single path, phase 4 does not start. The result is reported, and the next step goes to the researcher.
- Phase 4 then runs the chosen recipe afresh, as in R12. For B and C, the SD floor applies when the chosen recipe has it.

**Also measured** (analysis only, in no decision):
- For A, the 2 × 2 of advantage form × BC: RL0, RL4, RL5, RL6.
- For D, the SD floor alone (RL7 against RL0), and the same 2 × 2 with the floor on: RL7, RL4, RL5, RL6.

## Phase 5 — choice of method (Thu 26)

R5 on navdev, then C1–C4 on navtest (R4). With a clear winner, phase 6; on a tie, the researcher decides.

## Phase 6 — final system (Fri 27 – Sun 29, only with a clear winner)

- **Post-decision tuning** of the winning method, labelled as such:
  - if A, the phase-2 best;
  - if D, B or C, the same number of runs over its own main settings.
- **Full RL stage:** the phase-4 settings with 5,000 updates (160,000 scenes, about 2 passes over the 80,626).
- **Selector, cross-fitted:**
  - Split the training drives into 2 halves (by drive, seed 0) and train the chosen configuration on each half. Each half's generator makes candidates for the other half's navtrain scenes.
  - The selector is trained on these out-of-fold candidates. A second selector trained on in-fold candidates measures the caveat in R9, both on navdev.
  - The selector is adopted only if it beats the generator's own top choice on navdev.
- **Final number:** one navtest evaluation of the final system, reported with the list of every choice behind it.

**R15. Post-decision tuning and the full RL stage, specified** (committed 2026-11-20 13:25, before any phase-4 result, so the runs cannot be fitted to the winner; it fills in the two phase-6 lines above).
- **The 4 tuning runs per method.** A's phase-2 series has 4 runs beyond A1 (A0, A2, A3, A4), so every method gets 4 runs over the settings our research notes give as decisive for its family (`docs/research/heads/head_research_report.md`, `notes/continuous_diffusion_navsim.md`). All are at the method's chosen layers, seed 0, with the phase-1 recipe and navdev grid:
  - **A:** the phase-2 series as run (no new runs).
  - **B (DDIM; phase 1: clean-path target, cosine schedule):** noise target with the cosine schedule; v target with the cosine schedule; clean-path target with DDPM's linear schedule (β from 1e-4 to 0.02); noise target with the linear schedule, close to the published VLA diffusion planners: ReCogDrive and DiffAdapterVLA predict the noise with DDPM [V]; their schedule is not checked. The target matters on driving: DiffE2E, CARLA, 82.9 with the clean-path target against 20.1 with the noise target [V].
  - **C (flow matching; phase 1: start noise SD 0.1, uniform time):** start noise SD 0.3; start noise SD 1.0, the standard Gaussian used by π0 and GR00T [R]; π0's time sampling, a Beta(1.5, 1) weighted toward the noisy end [R], at SD 0.1; SD3's logit-normal(0, 1) time sampling [R], at SD 0.1.
  - **D (anchored; phase 1: 20 anchors, truncation 0.05):** 10 anchors; 40 anchors; truncation 0.02; truncation 0.1. DiffusionDrive uses 20 anchors and DiffVLA 32 [V]; DiffusionDrive trains on the first 50 of 1,000 noise steps [V].
- **The tuned configuration:** each run's choices by R10, then the configuration with the highest navdev single-path PDMS among the phase-1 configuration and its 4 runs. A tie goes to the phase-1 configuration. It is reported as post-decision tuning, never as part of the method comparison.
- **Full RL stage:** from the tuned configuration's R11 starting epoch, with phase 4's recipe (RL4 unless the researcher switches), seed 0, 5,000 updates, saved every 500. The checkpoint is chosen among update 0 and the 10 saves by navdev single-path PDMS, a tie going to the earlier one (R11's rule). The chosen checkpoint is the deployed path generator; the driver then writes its selector spec (R14).
- **Code:** the B, C and D settings above need new training flags. They are written only for the method that wins, before its runs. The rule fixes what they are now.
- **Cost** [D]: 4 training runs (about 2 h each at three stored layers) and their navdev grids; the full RL stage about 3.5 h on the GPU, plus about 35 min of navdev scoring for its 10 saves.

**R16. Both RL recipes, and the final system chosen with its selector** (committed 2026-11-20 13:51 at the researcher's direction, before any phase-4 run had started). The researcher, 13:45: "keep RL4 as registered and also run RL6 on every method with the same seeds and 1,500 updates ... so neither recipe is dropped on single-path score alone. Then take at least A and D under both recipes into phase 6, build the cross-fitted selector for each, and choose the final system by its generator-plus-selector score on navdev, with R5's single-path ranking reported but no longer deciding." It overrides R5's role and phase 6's "only with a clear winner".
- **Phase 4 runs RL4 and RL6 on every method**, with the same seeds (A and D: 0, 1, 2; B and C: 0) and the same settings. That is 1,500 updates saved every 250, with each run's checkpoint chosen by R11's rule (update 0 allowed). RL6 is R13's Dr. GRPO plus the BC term, weighted to the RL term's gradient norm at the start, with the 0.1 SD floor.
- **R5 is reported, not deciding.** Its single-path ranking after RL (navdev, the hierarchical bootstrap) is computed for each recipe and reported.
- **Phase 6 contenders:**
  - A and D, each under RL4 and under RL6: four path generators.
  - B and C join, under both recipes, if after phase 4 either is not clearly below the best of A and D under its better recipe (by point estimate). The test is navdev single path or best-of-6: the paired drive interval of the difference must reach 0 or above. They queue after A's and D's.
- **Amendment (21:42, before any phase-4 run had started): the B and C gate uses every phase-4 seed.**
  - It uses R5's hierarchical bootstrap, resampling drives and seeds; a single-seed method gets the larger of A's and D's between-seed SDs as an added normal draw.
  - It is computed on navdev single path and best-of-6, at each run's chosen checkpoint, under each method's better recipe (point estimate).
  - B or C joins if the interval of (the best of A and D) minus it includes 0 on either measure, or if it leads.
  - Reason: A1's three seeds differ by up to 1.0 point on navtest (measured 19:11), so a seed-0-only comparison could admit or exclude a method on seed luck.
- **Each contender gets:**
  - R15's post-decision tuning of its method, shared by its two recipes;
  - the full RL stage with its own recipe (5,000 updates, saved every 500, R11's checkpoint rule);
  - R14's cross-fitted selector, whose half generators use the same recipe and updates, with the in-fold control, seed 1 and the ablations.
- **The final system:** the contender with the highest navdev PDMS of its system path. The system path is the selected path where its selector is adopted (R14), otherwise its own pick. The choice is the point estimate, as in R10. Every contender's interval against it is reported, and an exact tie goes to the researcher. The choice is recorded before any phase-6 navtest score.
- **navtest:** each contender's system path is then scored once on navtest. The final system's score is the headline; the others are reported beside it.
- **Cost** [D]:
  - Phase 4 grows from 8 to 16 RL runs, about 10 more GPU hours.
  - Each phase-6 contender takes about 16 GPU hours: about 3.5 h for the full RL stage, about 4.5 h for each half generator, and about 4 h for the selectors. That is about 64 h for A and D, and about 64 h more if B and C join.
  - The researcher: "we have excess in the week".

**R18. The full RL stage continues while navdev still rises** (committed 2026-11-21 07:44, before any full RL stage or phase-4 run had started). The researcher, 2026-11-19: "if we are doing RL and we can recursively improve then we should do so even if [it breaks the] schedule"; the standing direction since: quality over schedule, iterate RL past the dates. R15's full RL stage stops at 5,000 updates even when its last save is its best, which would leave a still-rising RL stage unfinished.
- **Rounds.** The full RL stage runs in rounds of 5,000 updates, saved every 500. After each round, R11's rule chooses among update 0 and every save so far, by navdev single-path PDMS, a tie going to the earlier one.
  - If the choice is the round's last save, the stage continues from it for another round: the same recipe, settings, seed and scene order (`rl_head.py --resume`; jobs `47{m}{r}r{k}_`).
  - Otherwise, or after 4 rounds (20,000 updates), the stage ends, and the choice is the deployed path generator.
- **Equal treatment.** Every contender follows the rule on its own navdev curve. Its selector chain's half generators then run its deployed update count (R16).
- **Phase 4 is unchanged.** Its 1,500 updates are the matched comparison between methods.
- **Selection noise.** More saves make the navdev maximum more optimistic. navtest stays untouched until the final system is chosen (R16), so the headline number is not biased by it.
- **Cost** [D]: each extra round is about 3.1 h of RL for the contender plus 10 navdev path sets, and about 2 × 3.1 h more for its two half generators: about 9 GPU hours. At most 3 extra rounds per contender, about 28 h.

**R19. When RL improves no save of the tuned configuration** (committed 2026-11-24 05:16, before any navdev score of B's, C's or D's full RL stages; prompted by B's training reward under RL4, below). The standing direction: judge contenders by the final system's measure, and never let a pre-RL measure drop one.
- **Trigger.** A contender's full RL stage ends (R18) with R11 choosing update 0: no save beats the start on navdev's single path. It can only happen where R15's tuned configuration is not the method's phase-1 configuration.
- **Then** the method's phase-1 configuration gets the same full RL stage: the same recipe, R18's rounds, from its own R11 starting epoch (phase 4's), as jobs `47{m}{r}_rlfull{r}-<phase-1 run>`.
- **The contender's deployed path generator** is the higher of the two R11 choices by navdev single path; a tie stays with the tuned configuration. Its decisive chain (R14) runs on that one. Both stages are reported.
- **Equal treatment.** It applies to every contender. A4 and A6 chose RL saves, so it cannot change them.
- **Why.** R15 picks the tuned configuration by navdev single path before RL, but R16 judges the system after RL. Without R19, a configuration that RL cannot improve would stand in for its method because of a pre-RL difference. B's tuned configuration (noise target, linear schedule) won R15 by 0.37 points (81.77 against 81.40). Under RL4 its training reward fell from 0.843 (updates 1-100) to 0.502 (updates 1,001-1,073); best of 8 fell from 0.941 to 0.638 and NC from 0.981 to 0.717. B's phase-1 configuration, under the same recipe in phase 4, rose from 0.897 to 0.926. B's navdev scores, which decide, were not made when this was written.
- **Cost** [D]: about 3.4 h of GPU per triggered stage at B's measured 2.4 s per update (one round; more under R18). The chain still runs once per contender.

**R14. The selector's design** (committed 2026-11-20 11:38, before any selector result and before its code runs on real data). Evidence: `docs/research/heads/notes/selector_design.md`, a review of about 25 published NAVSIM selectors.
- **The score it predicts, checked on our files:** NAVSIM v1.1's PDMS is NC × DAC × (5·TTC + 5·EP + 2·C)/12.
  - On one of our navtest score files (12,146 scenes), this formula equals the official score exactly.
  - Driving-direction compliance has weight 0: 392 scenes have it below 1, with no effect on the score.
  - EP is 0 wherever NC × DAC is 0 [V, ours].
- **Inputs:**
  - the stored features of the layers the chosen path generator reads, each projected to width 256 with a layer embedding, read by cross-attention;
  - the ego state (the path generator's 19 numbers) through AdaLN;
  - each candidate's decoded path only, never the path generator's hidden states. DrivoR: a separate scorer with stop-gradient 90.0, against a shared branch 84.7 (navval) [V].
- **Network:** the path generator's block design, 4 blocks of width 256, about 5M parameters [D]. Published selectors add 6–8M [V].
  - One token per candidate: the 8 (x, y) waypoints normalised with the generator's statistics, plus the heading, speed and acceleration derived per step, through Fourier features and a 2-layer MLP.
  - No attention across candidates, so a candidate's score does not depend on which others are present. DrNMT's set-level reranker equals its per-hypothesis one [V].
  - Training: AdamW, learning rate 1e-4, weight decay 0.01, dropout 0.1, up to 20 epochs. The epoch with the best selected-path PDMS on the held-out training drives (below) is kept.
- **Targets and loss:**
  - One output per sub-score: NC, DAC, TTC, EP, C. Each candidate is scored alone with its scene's PDM-Closed reference, as `pdm_server.py` does: EP is relative to that reference, so scoring candidates together would change it.
  - BCE with soft targets: NC's ½ for a static object stays ½. The EP term is left out where NC × DAC = 0.
  - The selection score is the v1.1 formula over the predicted sub-scores, with the official weights.
  - Evidence: the two published ablations favour predicting sub-scores over predicting the PDMS directly: Hydra-MDP 83.0 against 80.2, and DrivoR 90.0 against 88.2 (navval) [V].
- **Candidates:** exactly the deployed sampler's. The own pick (temperature 0) plus 6 samples at the method's chosen sampling setting: 7 per scene. The own pick carries no flag.
- **Own-pick margin δ:** take the candidate with the highest predicted score, but keep the own pick unless that candidate is predicted to be more than δ higher. δ ≥ 0 is chosen on the held-out training drives.
- **Cross-fitting** (as approved above, now specified):
  - **Halves:** `drive_map.drive_halves` over all 236,133 training scenes, seed 0.
  - **Each half's generator:** the chosen configuration with its chosen settings (`train_head.py --drive-half`), then the same RL recipe and number of updates as the deployed generator, on that half's RL scenes only. The deployed generator is RL-trained, and RL changes the candidates: A's end points went from 6.4 m apart to 0.8 m under RL4.
  - **Candidates:** each half's generator makes the 7 candidates for the other half's RL scenes (`predict_head.py --only-tokens`), and `score_candidates.py` scores them.
  - **Held-out training drives:** 10 of the 101 RL drives (seed 0) stay out of the selector's training. They choose the epoch, δ and the ablations below. No selector setting uses navdev.
- **In-fold control (R9):** a second selector, trained the same way on the deployed generator's own candidates for the RL scenes.
- **Adoption, on navdev:**
  - The measure is the paired single-path difference, selected path minus own pick, with its drive interval.
  - The selector is adopted if the point estimate is above 0 (R10's convention: choices take the point estimate). A gain is claimed only if the interval excludes 0.
  - Then comes one navtest evaluation.
  - The base selector trains with 2 seeds: seed 0 is the one adopted, and seed 1 gives the spread.
- **Measured:**
  - best-of-7, a random pick and the mean candidate;
  - the share of the best-of-7 gap the selector closes;
  - the sub-score changes;
  - the switch rate, with wins, losses and ties;
  - picks that fail NC or DAC;
  - exact and tie-aware top-1 accuracy;
  - calibration per sub-score;
  - the gain by bins of candidate spread;
  - in-fold against cross-fitted, on navdev;
  - the half generators against the deployed one, on navdev.
- **Ablations,** one run each, judged on the held-out training drives:
  - a direct PDMS target;
  - an added listwise term (weight 0.25);
  - an imitation term;
  - 4 boundary paths per training scene: perturbations of the logged path across the DAC and NC boundaries. In Versatile Samples (arXiv 2609.01799, WACV 2027) these add +0.2 PDMS on v1 over the plain scorer [V].
- **If a runner-up's selector is also built** (the question in `docs/STATUS.md`), it uses this same design.
- **Implementation** (fixed 2026-11-20 13:06 with the code, `pilot_sft/heads/selector.py`, `train_selector.py`, `apply_selector.py`, before any run on real data):
  - **Size, measured:** 7.2M parameters with three stored layers: 5.8M in the 4 blocks, 0.9M for the stored-layer inputs, 0.2M for the candidate token. The estimate above (about 5M) left out the AdaLN modulation, 0.4M per block. The design is unchanged.
  - **Candidate token:** per step, the normalised x and y, the heading of the step (0 below 0.1 m), the speed, and the acceleration (from the ego speed at the first step). Each goes through 8-frequency Fourier features, then the 2-layer MLP.
  - **Training:** 128 scenes per step; 1,000 warm-up steps, then a cosine decay, as for the path generators; dropout 0.1 on the candidate token and on each block's update; gradient clip 1.0.
  - **Margins tried:** δ in {0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1}. The (epoch, δ) with the highest mean true PDMS of the selected paths on the held-out training drives is kept. A tie goes to the earlier epoch, then the larger δ.
  - **Unscorable candidates** (NAVSIM error) are left out of the loss and count as PDMS 0 when picked.
  - **Ablation definitions:**
    - direct target: BCE on the PDMS;
    - listwise: KL between softmaxes over the scene's candidates of the min-max-normalised true and predicted PDMS at temperature 0.5, weight 0.25;
    - imitation: a sixth output trained by soft cross-entropy towards softmax(−ADE to the logged path / 1 m) over the scene's candidates, entering selection as predicted PDMS × q^0.1 (WoTE's weight).
    - boundary paths (`boundary_paths.py`, fixed 13:13, before any run on real data): 16 perturbations of each RL scene's logged path. Twelve are lateral, an offset along each waypoint's left normal that grows linearly to d at 4 s, d in {±0.5, ±1, ±1.5, ±2, ±3, ±4} m. Four are longitudinal, each waypoint moved to 0.6, 0.8, 1.2 or 1.4 times its distance along the path. All are scored by the PDM scoring service. On each lateral side and each longitudinal direction, the scorer finds the first perturbation that fails NC or DAC. Per scene, 4 are kept: the last passing and the first failing perturbation on the lateral side that fails first, and the same pair in the longitudinal direction that fails first. Without a failure, the outermost are kept. They are training-only candidates (`train_selector.py --extra-candidates`); selection never sees them. Cost [D]: 80,626 × 16 = 1.29M scored paths, about 1.8 h of desktop CPU at 200 paths/s.
- **Cost** [D]:
  - Each candidate set is about 80,626 × 7 = 564k paths, about 50 min at the 190–210 paths/s measured in our RL runs.
  - The two half generators add about 2 × 4 h of GPU time: about 1 h of training on half the data, and about 3 h for 5,000 RL updates (the recipe runs took 17–20 min per 500).
- **Rehearsal of the chain** (added 2026-11-21 05:55, before any selector chain had run):
  - **What:** one run of the chain on A's phase-1 configuration (seed 0) without RL, deployed at its R11 starting epoch, with two of the variants above: the cross-fitted seed-0 selector and the boundary ablation (`pilot_sft/heads/rehearsal_spec.py`).
  - **Why:** in queue order, the first contender's chain would run only after phases 2–4 and the R15 tuning runs, about two days from now. The rehearsal checks every job of the chain on real data at full scale: both half generators, out-of-fold candidates for every RL scene, their PDM scores, the shared boundary set, selector training and navdev evaluation. The CPU smoke test (Nov 20) checked only 1,000 scenes.
  - **Cost [D]:** about 4 GPU-hours and 3–4 CPU-hours. Its jobs go ahead of each queue: their files are named `0_<name>.sh`, which the runners take first.
  - **Role:** it is no contender (R16 reads only the contenders' specs), and its numbers are used for no choice. No selector setting changes because of them. A failure it finds is fixed in the code for every chain. Its boundary set is the shared one that every later chain uses.

## Mon 30 — report

- An HTML report on the home PC.
- Every number in `pilot_sft/README.md`, `docs/STATUS.md` and `docs/GAPS.md`.

## Schedule risks, and what goes first

- **Phase 4** is the tightest: 8 RL runs share the desktop's 16 CPU cores. If behind, drop C's RL run first, then B's; they are reported without RL.
- **Further cuts, in order:** phase 2's bin-size runs; then phase 6 moves to the next week. It is not shortened.
- **Dependencies:**
  - navdev's cache: phase 0, extract only its 40 logs.
  - The navtrain cache for phase 4: Anvil wide job 22320518, estimated start Nov 20 18:13.
  - Code for R and for B and C's zero-noise start: before Friday's B, C and R runs. D's code exists.

## Compute

- **Phases 1–3:** about 29 training runs: 41 min each reading one stored layer, about 2.3 h reading three (measured). That is 28–59 GPU-hours on the 4090.
- **Phase 4:** about 16 h, limited by scoring.
- **Phase 6:** about 1.5 days.
- **Scoring:** navtest takes 220 s per path set; navdev about 40 s.
