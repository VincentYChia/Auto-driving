# Status of the autonomous session

Written only by the Claude session on the home PC, and pushed whenever something changes. To ask for something, leave a note in `docs/INBOX.md`.

**Updated:** 2026-11-20 09:22 EST (times are taken from the clock; earlier labels ran 7-31 min ahead, see below)

Terms (path generator, configuration, training run, layer test, continuity series, single-path PDMS, best-of-6, RL stage, selector, decision rule) are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal to Mon 2026-11-30:** a controlled answer to which generation method the path generator should use (A, B, C or D) and how continuous its output should be. All on H1's same frozen features, the same training data and the same network body, measured by single-path PDMS and best-of-6 on the full navtest, before and after a matched RL stage. Then the best single-path PDMS from the chosen path generator after the full RL stage, with the selector, against H1's 85.6 today. Day-by-day plan: `pilot_sft/EXPERIMENTS.md`, "Compute plan to Monday 2026-11-30". Access from a new device: `docs/ACCESS.md`.

## Inbox
- Read up to: no notes yet.

## Running now
- **4090 desktop, GPU queue** (one job at a time):
  - **Done since 05:30:** R12's 8 recipe runs (RL0-RL3 on A and D, 500 updates each, about 20 min per run); navtest paths of B at layers 12/18/24; C reading layer 24 (07:10-07:51) and its navtest paths; R13's RL4 on A and on D.
  - **Running:** R13's RL6 on A (since 09:11). RL4 and RL5 are done on A and D.
  - **Next:** R13's RL6 and RL7 on D, about 20 min each, so done by about 10:15.
  - **Then, about 26 h:**
    - C at layers 12/18/24;
    - R at both layer settings;
    - seeds 1-2 of A1, B and D at layers 12/18/24;
    - A's continuity series A0, A2, A3, A4.
- **Desktop CPU, the driver** (`pilot_sft/heads/driver.py`, now 007c58d): scoring each R13 run on navdev as it lands. It now writes `driver_health.json` after every pass, failed or not; `driver_status.json` is written only by a completed pass, so it had hidden the 06:24-06:52 failures.
  - **A bug stopped it from 06:24 to 06:52.** A saved RL checkpoint choice, read back from JSON, had string keys, so every pass failed with `KeyError: 250`. It is fixed (d46ae30), with a test that fails on the old code.
- **Anvil:** nothing running.

## Latest results
- **R12 is complete: every recipe lowered both methods' navdev single path** (743 scenes, 11 drives; start, then 250 and 500 updates; the change at the better checkpoint, with its drive interval where computed). R12 alone would have chosen RL3 (mean change -0.95); R13 now decides.

  | Recipe | A (start 81.43, epoch 20) | D (start 77.10, epoch 5) |
  |---|---|---|
  | RL0: as registered (lr 1e-5, no KL) | 75.72, 77.19 (-4.2, -10.4 to -0.5) | 60.93, 64.23 (-12.9, -16.4 to -8.4) |
  | RL1: KL beta 0.04 | 77.58, 78.76 (-2.7, -8.1 to +0.6) | 75.27, 75.28 (-1.8, -2.9 to 0.0) |
  | RL2: KL, lr 3e-6 | 79.64, 79.01 (-1.8) | 75.55, 75.80 (-1.3) |
  | RL3: KL, lr 3e-6, 16 x 16 | 80.68, 78.61 (-0.75) | 75.95, 74.90 (-1.15) |

  - **A trades safety for progress.** Under RL0: progress 73.5 to 78.3, time-to-collision 93.9 to 84.4, no-collision 97.5 to 92.3, drivable area 91.8 to 88.7. The best-of-6 candidates' end points went from 6.4 m apart to 2.0 m.
  - **D is damaged everywhere.** Under RL0 at 250 updates: drivable area -13.2, time-to-collision -13.1, progress -7.3.
  - **On the training scenes,** A's sampled reward rose with the gentler recipes (RL3: 0.79 to 0.85), while best-of-8 fell and equal-reward groups rose. The policy sharpens, which raises the sampled reward without improving the temperature-0 path that navdev scores.
- **R13, registered before any of its runs** (`docs/HEADS_PLAN.md`, 4fed42c; code 714cd6f, 73 tests pass). Three likely causes, each with a published remedy:
  1. The group SD inflates tiny progress gaps between safe candidates. Remedy: subtract the group mean only (Dr. GRPO). NoRD on navtest: GRPO +0.5, Dr. GRPO +9.0.
  2. Nothing ties the policy to the logged driving. Every published RL stage on a separate NAVSIM path generator adds a behaviour-cloning (BC) term.
  3. D's last refinement step sits at the 0.01 SD floor, so its gradient is about 100 times the anchor choice's (gradient norm 300-550, against 21-33 for A). Remedy: a 0.1 SD floor inside the log-probability only, as in DiffusionDriveV2.
  - **The recipes,** each from RL0 and all with the SD floor: RL4 Dr. GRPO; RL5 BC; RL6 Dr. GRPO + BC; RL7 neither. For A, the floor does nothing, so A's RL7 is RL0 and is not run again.
  - The BC weight is set so its gradient norm equals the RL term's at the start.
  - **The choice:** R12's rule over RL0-RL7. If every recipe still lowers both methods, phase 4 does not start, and the next step goes to you.
  - **First R13 results on navdev (these decide):**
    - **A, RL4 (Dr. GRPO): 81.43, then 85.08 at 250 updates and 86.30 at 500 (+4.9).** That is the first RL run to raise a navdev single path, and it is above H1's text output on navdev (82.4). Seed 0, recipe stage only; navtest comes with phase 4.
    - A, RL5 (BC, SD division kept): 81.99 at 250 (+0.6), 78.11 at 500.
    - D, RL4: 76.84 at 250 (-0.3), 70.39 at 500. No recipe has helped D yet.
  - **The same runs on the training scenes** (block of the first 50 updates, then the last 50; navdev scores are being computed):
    - **A, RL4 (Dr. GRPO):** reward 0.82 to 0.92, with time-to-collision 0.90 to 0.96 and progress 0.78 to 0.88 both rising; best-of-8 held (0.953 to 0.958). Under R12, safety fell whenever progress rose.
    - **A, RL5 (BC, SD division kept):** R12's pattern again, progress 0.77 to 0.87 with time-to-collision 0.87 to 0.83. Its measured BC weight is 2.50.
    - **D, RL4:** the SD floor brought the gradient norm from 300-550 down to 2.6, but D's reward still slips (0.853 to 0.815).
- **Phase 1 so far** (single path / best-of-6 at each training run's navdev choices; navtest reports only):

  | Method | navdev, layer 24 | navdev, layers 12/18/24 | Layer choice | navtest at the choice (vs H1's 85.6) |
  |---|---|---|---|---|
  | A1 (masked) | 81.5 / 89.5 | 81.9 / 89.7 | 12/18/24 | 84.7 / 91.8 (-0.9, -1.4 to -0.5) |
  | B (DDIM) | 81.0 / 88.2 | 81.4 / 89.3 | 12/18/24 | 84.6 / 91.9 (-1.05, -1.5 to -0.6) |
  | D (anchored) | 76.8 / 89.6 | 78.4 / 92.2 | 12/18/24 | 82.9 / **93.7** (-2.7, -3.5 to -1.8) |
  | C (flow) | 78.4 / 89.5 | training | - | layer 24: 83.5 / 91.6 (-2.1) |

  - **Layer test on navtest** (layers 12/18/24 minus 24 alone; single path; best-of-6):
    - A: -0.01 (-0.43 to +0.41); -0.03.
    - B: +0.13 (-0.25 to +0.51); +0.85 (+0.55 to +1.18).
    - D: +1.28 (+0.77 to +1.78); +2.17 (+1.73 to +2.55).
    - So the best layers depend on the method, the case the per-method layer test was there to catch.
  - **Method pairs on navtest, at the layer choices** (single path; best-of-6):
    - A - B: +0.11 (-0.34 to +0.54); -0.10 (-0.48 to +0.21). Indistinguishable, although B's candidates are much closer together (end points 1.4 m apart, against 6.6 m for A).
    - A - D: +1.75 (+0.93 to +2.60); -1.89 (-2.21 to -1.57).
    - B - D: +1.64 (+0.96 to +2.35); -1.79 (-2.18 to -1.36).
  - References: H1's text output scores 82.4 on navdev and 85.6 on navtest; constant velocity scores 19.8 on navdev.
  - Choices: A1 epoch 15, 8 steps, setting 1.5. B epoch 10. D epoch 10, 2 steps, setting 0.5. C (layer 24) epoch 10, 2 steps, setting 2.0. RL starts (R11): A epoch 20, B epoch 10, D epoch 5.
- **The RL trial** (200 updates on a1-l24 over 3 training drives, recipe as registered) lowered A1's navdev single path from 81.18 to 73.97 (-7.2, -13.3 to -2.6). That led to R12 (eca6ee6).
- **RL scoring service:** equals NAVSIM's official scores on 200 navtest scenes, with a maximum difference of 0.0 in every sub-score.
- **Text pilot H1: 85.6 PDMS on navtest**, still the best single path. H2 (8 passes) ties it.
- **NAVSIM scoring runs on the desktop.** H1 re-scored there matches Anvil on all 12,146 scenes (PDMS 0.856069 both).

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
7. Phase 1: navdev and navtest paths and scores for all 10 runs, and each method's layer choice. A, B and D are done; C and R are training.
8. **RL recipe rounds:** R12 is complete (every recipe lowered both methods). R13: RL4 and RL5 done on A and D (RL4 raised A by 4.9 on navdev), RL6 on A running, 2 runs to go.
   - **Phase 4 now waits for the diffusion comparison** (39e7c7a): its jobs sort after every phase-2 and phase-3 job, the plan's order. Before, B's and C's RL would have run ahead of the seeds and A's continuity series. The driver chooses over RL0-RL7 by itself; if a recipe helps, phase 4 starts with it, and if every recipe lowers both methods, phase 4 waits for you.
9. **Report at 12:00 Fri Nov 20:** an HTML overview on the home PC (results with drive intervals, R12 and as much of R13 as has landed, problems and fixes, next steps); the headline numbers go here.
10. ~~Copy the R12 recipe runs to Anvil and add their rows to `pilot_sft/MODELS.md`.~~ **Done:** all 8, then `c-l24` and R13's RL4 on A and D (08:55), all on both machines with matching checksums. The other R13 runs follow when they finish.

## Waiting on the researcher
- **A question for your 12:00 check; it blocks nothing.**
  - **The issue:** R5 chooses the method by its own single path after RL, and the selector (phase 6) is built only for the winner. So far, D has clearly the best candidates and clearly the weakest own pick (A, B and D on navtest; C and R are still training).
    - D − A, best-of-6: **+1.9** (drive interval +1.6 to +2.2).
    - D − A, single path: **−1.75** (−2.6 to −0.9).
  - **Why it matters:** the final system always has a selector, so R5 could choose a method that is not the best one for the final system.
  - **Options:**
    1. Keep R5 as it is. It answers "which method gives the best path on its own". This is the default.
    2. Also build phase 6's cross-fitted selector for the runner-up, when its best-of-6 is clearly higher. The final system then uses whichever generator-plus-selector scores higher on navdev.
  - **My recommendation is option 2.** It costs one more selector, and R5's answer stays as it is.
  - Nothing changes unless you say so.
- **For your information: R13, a second RL recipe round, added under your "improve RL round after round" clarification.** R12's recipes all lowered the navdev single path so far, and the sub-scores point to the advantage normalisation, the missing imitation term and D's step size. R13 tests the published remedies, on A and D, before phase 4. It moves phase 4 back by about 2.5 h of GPU time (7 runs of about 20 min; I first wrote 3.5 h). If none helps, phase 4 waits for your call rather than running a recipe that only lowers scores.
- **The plan (`docs/HEADS_PLAN.md`, revision 2) was approved 2026-11-19 ~21:00, and the HOLD is lifted.** Two clarifications are in the plan's header:
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

## Overnight review (asked 08:38)
- **No check ran from about 01:00 to 04:47.** The 02:47 heartbeat hit the session's 5-hour usage limit (reset 03:10), and a missed heartbeat is not re-run, so STATUS sat unchanged from 00:55 to 04:55. The desktop and Anvil automation carried the work: the training-scene cache landed at 02:52 and the driver queued R12 at 03:03.
- **The driver stalled from 06:24 to about 06:52** (`KeyError: 250`), which delayed R12's navdev scores by about 30 min. It is fixed, and `driver_health.json` now shows any failing pass.
- **R12 tested the wrong levers.** It varied KL strength, learning rate and group size. Dr. GRPO and the BC term were already in our research notes, and including them would have saved the time R13 adds.
- **STATUS time labels ran 7-31 min ahead of their commits** (for example "23:55" was committed at 23:24), and R13's registration was first written as about 07:40 instead of 07:02. Times from logs are right; the labels now come from the clock.
- **Held up:** the 4090 ran 29 jobs back to back from Thursday 19:23 with no gap over 2 min; no Anvil job failed; every navtest score set has all 12,146 scenes and every navdev set all 743; no desktop connection was left open.

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
