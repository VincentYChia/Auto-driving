# Status of the autonomous session

Written only by the Claude session on the home PC, and pushed whenever something changes. To ask for something, leave a note in `docs/INBOX.md`.

**Updated:** 2026-11-20 05:30 EST

Terms (path generator, configuration, training run, layer test, continuity series, single-path PDMS, best-of-6, RL stage, selector, decision rule) are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal to Mon 2026-11-30:** a controlled answer to which generation method the path generator should use (A, B, C or D) and how continuous its output should be. All on H1's same frozen features, the same training data and the same network body, measured by single-path PDMS and best-of-6 on the full navtest, before and after a matched RL stage. Then the best single-path PDMS from the chosen path generator after the full RL stage, with the selector, against H1's 85.6 today. Day-by-day plan: `pilot_sft/EXPERIMENTS.md`, "Compute plan to Monday 2026-11-30". Access from a new device: `docs/ACCESS.md`.

## Inbox
- Read up to: no notes yet.

## Running now
- **4090 desktop, GPU queue** (one job at a time):
  - **Done since 01:10:**
    - D reading layers 12/18/24;
    - B at both layer settings;
    - navtest paths at the navdev choices of D (both runs) and B (layer 24);
    - R12's RL0 on A.
  - **Running: R12's recipe runs.** The navtrain cache was ready at 02:52, and the driver queued all 8 at 03:03.
    - RL0 on A finished at 04:39 (19 min); RL0 on D started 04:39.
    - The other 6 take about 20 min each, so all should be done by about 07:00.
  - **Queued, about 22 h:**
    - C and R at both layer settings;
    - A1's seeds 1–2 and D's seeds 1–2, at layers 12/18/24;
    - A's continuity series A0, A2, A3, A4.
  - The driver queues the phase-4 RL runs once R12's recipe is chosen.
- **Desktop CPU, the driver** (`pilot_sft/heads/driver.py`, 995c3ae): scoring B's navtest paths and navdev grids, then each recipe run's navdev paths as it lands. R12's choice goes to `heads/choices/rl_recipe.json`.
- **Anvil:** nothing running.
  - The navtrain cache job (22320518) ran 01:26–02:42. It cached all 103,288 navtrain scenes (61 GB).
  - The watcher `~/xv/navtrain_to_desk.sh` copied the cache to the desktop in 510 s. NAVSIM's loader found every scene, and the watcher wrote `.ready` at 02:52. No manual step was needed.

## Latest results
- **Phase 1 so far**: single path / best-of-6 at each training run's navdev choices. navdev has 743 scenes from 11 drives; intervals come in the 12:00 report.

  | Method | navdev, layer 24 | navdev, layers 12/18/24 | Layer choice | navtest at the choices (report only) |
  |---|---|---|---|---|
  | A1 (masked) | 81.5 / 89.5 | 81.9 / 89.7 | 12/18/24 | 84.7 / 91.8 (both settings) |
  | D (anchored) | 76.8 / 89.6 | 78.4 / 92.2 | 12/18/24 | 81.6 / 91.5 (24); **82.9 / 93.7** (12/18/24) |
  | B (DDIM) | 81.0 / 88.2 | scoring | — | scoring |

  - References: H1's text output scores 82.4 on navdev and 85.6 on navtest; constant velocity scores 19.8 on navdev.
  - **D's candidates are the best so far, and its own pick is the weakest.** At its layer choice, D's best-of-6 is 2.5 points above A's on navdev and 1.9 on navtest. Its single path is 3.5 below A's on navdev and 1.8 on navtest. The single path is the top-scored refined anchor; R7 defines it.
  - Choices:
    - D at layers 12/18/24: epoch 10, 2 steps, setting 0.5;
    - D at layer 24: epoch 15, 2 steps, setting 1.0;
    - B at layer 24: epoch 20, 5 steps, setting 1.5.
- **RL starting epochs (R11, by navdev best-of-6):**
  - A: epoch 20 (86.8 / 87.7 / 89.7 / 90.3 at epochs 5 / 10 / 15 / 20);
  - D: epoch 5 (92.3, against 92.2 / 89.5 / 88.9).
- **R12, RL0 on A** (the registered recipe, 500 updates on the 80,626 navtrain scenes): navdev scores are pending.
  - On the training scenes, the mean reward stayed flat: 0.79 in the first 50 updates, 0.80–0.83 after.
  - Best-of-8 fell from 0.953 to 0.900, and equal-reward groups rose from 16% to 52%.
  - This is the trial's pattern again.
- **The RL trial changed the plan (R12, eca6ee6):** 200 updates on a1-l24 over 2,662 scenes from 3 training drives, with the recipe as registered (lr 1e-5, no KL, 8 × 32).
  - It runs cleanly: about 2.2 s per update, 100–150 scored paths/s, 0 scorer errors.
  - **It made A1 worse on navdev:** the single path went 81.18 → 75.58 → 73.97 at 100 and 200 updates (paired −7.2, drive interval −13.3 to −2.6). No-collision fell 0.976 → 0.914, drivable area 0.914 → 0.871, time-to-collision 0.943 → 0.847.
  - On its own scenes the reward did not improve. Diversity collapsed: best-of-8 0.953 → 0.906, and equal-reward groups rose 26% → 52%.
  - **R12, committed before any recipe run:** 4 recipes (as registered; KL β = 0.04; KL + lr 3e-6; KL + lr 3e-6 + 16 candidates × 16 scenes), each run on A and D for 500 updates on the full navtrain cache. The recipe with the best mean navdev gain over A and D is chosen, then phase 4 runs it afresh. Update 0 (no RL) is an allowed checkpoint.
  - The KL is exact along each sampled path, for all four methods. The driver runs the whole R12 stage by itself once the navtrain cache is ready.
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
8. **RL code: done and tested for all four methods.**
   - Policies with exact log-probabilities and exact KL; the trainer, with KL; the scoring service, which equals official scores.
   - The driver handles phases 4–5: the R12 recipe stage, then the matched RL runs, the RL checkpoint choice on navdev (update 0 allowed), navtest reporting, and R5's drive-and-seed bootstrap decision.
   - ~~Waiting on the navtrain cache.~~ **Done:** ready on the desktop at 02:52 (103,288 scenes). R12's 8 recipe runs are under way.
9. **Report at 12:00 Fri Nov 20:** results with drive intervals for every configuration so far (navdev for choices, navtest for the report), R12's recipe result if it has landed, problems and fixes, and next steps. An HTML overview stays on the home PC; the headline numbers go here.

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
