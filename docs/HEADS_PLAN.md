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
