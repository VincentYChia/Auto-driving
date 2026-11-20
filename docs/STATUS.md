# Status of the autonomous session

Written only by the Claude session on the home PC, and pushed whenever something changes. To ask for something, leave a note in `docs/INBOX.md`.

**Updated:** 2026-11-19 22:00 EST

Terms (path generator, configuration, training run, layer test, continuity series, single-path PDMS, best-of-6, RL stage, selector, decision rule) are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal to Mon 2026-11-30:** a controlled answer to which generation method the path generator should use (A, B, C or D) and how continuous its output should be. All on H1's same frozen features, the same training data and the same network body, measured by single-path PDMS and best-of-6 on the full navtest, before and after a matched RL stage. Then the best single-path PDMS from the chosen path generator after the full RL stage, with the selector, against H1's 85.6 today. Day-by-day plan: `pilot_sft/EXPERIMENTS.md`, "Compute plan to Monday 2026-11-30". Access from a new device: `docs/ACCESS.md`.

## Inbox
- Read up to: no notes yet.

## Running now
- **4090 desktop, GPU queue** (phase 1, queued 21:10, in run order: 363, then D, B and C at both layer settings, 364–369, about 9 h; R once its code exists):
  - Job 361, layer test part 1: the training run of configuration A1 reading stored layer 24 only. It trained all 20 epochs in 41 min, and every epoch's weights are saved. Its navtest prediction step failed on the one navtest scene with 420 tokens (the training scenes have at most 419); fixed in f08b281.
  - Job 362, layer test part 2: the training run of configuration A1 reading layers 12, 18 and 24. Epoch 12 of 20 at 20:47, 6.8 min per epoch (3.4 times the one-layer run, from reading 3 layers), about 2.3 h in all; it ends about 21:45.
  - Job 363: navtest paths from job 361's path generator, next in the queue.
- **4090 desktop, CPU queue:** scoring jobs 371 and 372 wait for those runs' paths, then score them on NAVSIM (about an hour each).
- **Anvil:** wide job 22320518 builds the navtrain metric cache, the reward for the RL stage. Estimated start Nov 20 19:05.

## Latest results
- **Text pilot H1: 85.6 PDMS**, still the best.
  - S3's earlier passes scored 77.3, 81.9 and 84.5; the final pass 84.7.
  - H2 (8 passes) ties H1. Details in `pilot_sft/README.md`.
- **Layer test, dev ADE so far** (2,000 dev scenes; a training check only, not used for choices):
  - A1 reading layer 24, final (epoch 20): dev ADE 0.765 m, dev best-of-6 ADE 0.514 m.
  - A1 reading layers 12, 18 and 24, at epoch 12: 0.759 m and 0.507 m.
  - H1's text output: 0.77 m.
  - NAVSIM scores to come (navtest, for the report; the choice will be made on navdev once it exists).
- **NAVSIM scoring now runs on the desktop.** H1 re-scored there matches Anvil on all 12,146 scenes: PDMS 0.856069 on both, with 53 ego-progress values off by at most 1.3e-11 from CPU rounding.

## Next (the live to-do list; the plan is `docs/HEADS_PLAN.md`)
1. ~~Code: temperature 0 means the deterministic best guess for every method; the generate argument is a per-method sampling scale for the R7 grids.~~ **Done** (bb05104).
2. ~~Code: regression R as a configuration kind, with tests; queue its two phase-1 training runs.~~ **Done** (bb05104; jobs 380, 381 queued).
3. Phase 0, navdev:
   - **Done:** the scene filter navdev_all (2,000 tokens, 34 logs, 11 drives) in `navsim_eval/navsim_configs`; split-aware scoring and summary with a drive-cluster bootstrap; drive maps on the desktop.
   - **Running:** the metric cache plus constant-velocity and human-path scores, Anvil job 22322227 on `wide` (est. start 22:07).
   - **Then:** copy to the desktop and rewrite the metadata paths.
   - **Finding:** navtest's 12,146 scenes come from only 44 drives (16–985 scenes each), so drive-level intervals on navtest will also be much wider than the scene-level ones reported so far.
4. Phase 0:
   - **Difficulty filter: verified.** navtest's own rule (constant velocity PDMS ≤ 0.8 and human PDMS ≥ 0.8) keeps 12,129 of 12,146 navtest scenes (99.86%). It is applied to navdev once its scores exist.
   - **Leakage audit: passes** (`docs/navsim_eval/results/heads_leakage_audit_2026-11-19.json`):
     - 0 drives shared between navdev or navtest and the training data (236,133 scenes, 158 drives) or the RL scenes (80,626 scenes, 101 drives);
     - all RL scenes are inside the training data;
     - 0 of the 563 NAVSIM warm-up scenes are in any training set.
   - **Still to do:** navdev's resolution; the references (constant velocity, H1's text output) once the cache exists. H1's navdev paths are job 363b.
5. ~~Commit the rules.~~ **Done:** R1–R9 and the phase-4 settings in a64e437 and 79df830; R10, the choice procedure, in **b6ba4f5**, all before any navdev score. Still to read and commit before phase 4: Flow-GRPO's SDE noise level for C.
6. ~~Script the phase-1 choices and the phase transitions.~~ **Done:** `pilot_sft/heads/driver.py` (85dcec1), running on the desktop since 21:55. Every 5 min it:
   - scores navdev in the R10 order and writes each run's choices;
   - makes each method's layer choice;
   - queues navtest paths at the choices, phase 3's seeds and phase 2's series;
   - scores navtest for the report.
   - Its status is `~/navsim_workspace/exp/heads/driver_status.json`; its log is `driver.log`.
   - Each phase-1 training job now ends with its navdev grid, `scripts/desk/navdev_grid.sh`.
7. Phase 1: navdev and navtest paths and scores for all 10 runs; the per-method layer choice.
8. RL code (phase 4), written and tested while phases 2–3 train.
9. **Report at 12:00 Fri Nov 20.**

## Waiting on the researcher
- Nothing. **The plan (`docs/HEADS_PLAN.md`, revision 2) was approved 2026-11-19 ~21:00, and the HOLD is lifted.** Two clarifications are in the plan's header:
  - equal tuning until a clear winner: contenders proceed together;
  - quality over schedule: improve RL round after round, even past the dates.
- **Schedule:**
  - The researcher is away from about 15:00–16:00 Fri Nov 20.
  - Report at **12:00 Fri Nov 20**; then one final check by the researcher. Work proceeds whether or not they reply.
  - Work must not stop when a usage limit is hit, so the desktop queues are kept deep (hours of queued GPU work) and phase transitions are scripted.
- Earlier answers (2026-11-19 evening):
  - **Spare GPU time:** no filler runs; only what a comprehensive comparison needs.
  - **The feature cache:** keep it all week, and ask before deleting it.
  - **The repo:** commit everything allowed, kept current and descriptive.

## Decisions made by rule (no input needed)
- **Desktop NAVSIM scores accepted.** They equal Anvil's to at most 1.3e-11 per scene, with identical PDMS.

## Where things are
- **Research and plan:** `docs/research/heads/` (head research report, test plan, five topic notes) and `docs/research/sft_literature_check/`.
- **Code:**
  - `pilot_sft/heads/` (features, model, train, predict, summarize);
  - `scripts/desk/score_head_run.sh`.
- **Run outputs:**
  - Desktop: `~/navsim_workspace/exp/heads/<run>/`.
  - Feature cache: `~/navsim_workspace/head_cache/h1_l12-18-24/`.
- **HTML reports and replay pages:** on the home PC only, in `navsim_demo_renders/` and `reports/`. They contain nuPlan camera images, which stay out of this public repository.
