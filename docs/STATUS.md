# Status of the autonomous session

Written only by the Claude session on the home PC, and pushed whenever something changes. To ask for something, leave a note in `docs/INBOX.md`.

**Updated:** 2026-11-19 23:05 EST

Terms (path generator, configuration, training run, layer test, continuity series, single-path PDMS, best-of-6, RL stage, selector, decision rule) are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal to Mon 2026-11-30:** a controlled answer to which generation method the path generator should use (A, B, C or D) and how continuous its output should be. All on H1's same frozen features, the same training data and the same network body, measured by single-path PDMS and best-of-6 on the full navtest, before and after a matched RL stage. Then the best single-path PDMS from the chosen path generator after the full RL stage, with the selector, against H1's 85.6 today. Day-by-day plan: `pilot_sft/EXPERIMENTS.md`, "Compute plan to Monday 2026-11-30". Access from a new device: `docs/ACCESS.md`.

## Inbox
- Read up to: no notes yet.

## Running now
- **4090 desktop, GPU queue** (one job at a time):
  - **Done:** the A1 layer test (2 training runs), D reading layer 24, and navtest paths for both A1 runs at their navdev choices.
  - **Running:** the RL trial on a1-l24 (job 355, started 22:53).
  - **Queued, about 26 h:** D reading 12/18/24; B and C at both layer settings; R at both; A1's seeds 1–2 at layers 12/18/24; A's continuity series A0, A2, A3, A4.
- **Desktop CPU, the driver** (`pilot_sft/heads/driver.py`, phases 1–5, restarted 22:55 with the phase-4 logic): scoring D reading layer 24 on navdev. The old navtest scoring job 371 is still running in queue2.
- **Anvil:** wide job 22320518 builds the navtrain metric cache for RL, estimated to start Nov 20 17:05. The RL runs are queued automatically once it is on the desktop.

## Latest results
- **A1's navdev choices (R10):**
  - layer 24: epoch 15, 4 steps, setting 1.0; single path **81.48**, best-of-6 89.52.
  - layers 12/18/24: epoch 15, 8 steps, setting 1.5; single path **81.86**, best-of-6 89.65.
  - **A reads layers 12/18/24** by the point-estimate rule. The 0.4-point gap is within noise, and is reported as such.
  - A best-of-6 about 8 points above the single path is the room a selector could use.
- **First navdev results** (743 scenes, 11 drives; intervals resample drives; for choices only, R1):
  - **H1's text output: 82.4** (80.6–86.0). Constant velocity: 19.8.
  - **A1 reading layer 24:** epochs 5 / 10 / 15 / 20 at 4 steps score 77.9 / 79.0 / **81.5** / 81.2. At epoch 15, 1 / 2 / 4 / 8 steps score 80.8 / 81.3 / 81.5 / 80.2.
  - **Paired, A1 (epoch 15) − H1: −0.9** (−2.4 to −0.4). So before RL, A1 is close to but measurably below H1's text output.
  - The sanity check passes: far above constant velocity, and dev ADE near H1's.
  - Paired differences on navdev resolve to about ±1 point.
- **RL scoring service:** equals NAVSIM's official scores on 200 navtest scenes, with a maximum difference of 0.0 in every sub-score. It scores 11.6 paths/s in one process and 71.8 paths/s with 4 workers at 8 candidates per scene.
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
3. ~~Phase 0, navdev.~~ **Done** (02a8d7d):
   - The cache was built on the desktop instead: Anvil's wide queue had no start estimate. It took 586 s, including the constant-velocity and human-path scores.
   - Of the 2,000 dev-v2 scenes, 1,990 can be scored. navtest's filter removes 1,071 near-trivial scenes (constant velocity > 0.8) and 217 noisy labels (human < 0.8).
   - **navdev = 743 scenes from 11 drives.** Choices made on it stay unbiased for navtest, but they will be noisy. The limit is the 11 drives, not the scene count, so more scenes from the same drives would barely narrow the per-drive intervals.
   - navtest's 12,146 scenes come from 44 drives.
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
7. Phase 1: navdev and navtest paths and scores for all 10 runs; each method's layer choice. **Under way:** the driver scores as the GPU jobs land.
8. RL code (phase 4), A and D first:
   - **Written and tested:**
     - the PDM scoring service (`pdm_server.py`), which scores each candidate alone next to the PDM-Closed reference, exactly as run_pdm_score does;
     - the policies (`Head.rl_sample` / `rl_logprob`): A's exact token log-probability, and D's anchor draw plus its stochastic refinement;
     - the trainer (`rl_head.py`).
   - **Next:** check the service against official scores and measure its speed; a 200-update trial on A1 (needs part of the navtrain cache); then B's and C's policies.
   - **To watch:** in a toy test, RL on A's 0.1 m bins (951 values for x) gave a weak signal over 25 updates, while 5 m bins learned at once. Each sampled path reinforces only its own bins. Phase 4 measures whether this costs A.
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
