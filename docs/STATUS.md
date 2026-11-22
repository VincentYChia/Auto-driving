# Status of the autonomous session

Written only by the Claude session on the home PC, and pushed whenever something changes. To ask for something, write in `docs/CHAT.md` (two-way: replies appear below your message; read about every 30 minutes) or leave a note in `docs/INBOX.md`.

**The 12:00 report** (refreshed Nov 22 13:48: phase 4 on navtest for A, with RL6 against RL4): https://claude.ai/artifact/N8XRgjfSNwkzoyxfwVfxSQ, a private page that only you can open. It holds charts only, no camera images. Its numbers are in `docs/navsim_eval/results/heads_aggregate_2026-11-20.json`.

**Updated:** 2026-11-22 16:50 EST (times are taken from the clock; earlier labels ran 7-31 min ahead, see below)

Terms (path generator, configuration, training run, layer test, continuity series, single-path PDMS, best-of-6, RL stage, selector, decision rule) are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal to Mon 2026-11-30:** a controlled answer to which generation method the path generator should use (A, B, C or D) and how continuous its output should be. All on H1's same frozen features, the same training data and the same network body, measured by single-path PDMS and best-of-6 on the full navtest, before and after a matched RL stage. Then the best single-path PDMS from the chosen path generator after the full RL stage, with the selector, against H1's 85.6 today. Day-by-day plan: `pilot_sft/EXPERIMENTS.md`, "Compute plan to Monday 2026-11-30". Access from a new device: `docs/ACCESS.md`.

## Inbox
- Read up to: no notes yet in `docs/INBOX.md` or `docs/CHAT.md` (last read Nov 22 16:50). `docs/CHAT.md` was added 13:28 at your request; a check every 30 minutes reads both and also updates this page when the desktop logs a new result.

## Running now
- **4090 desktop, GPU queue** (one job at a time; read 14:51 Nov 22):
  - **Running: phase 6 (R15), D's second tuning run** (`46d1`: 40 anchors, since 14:59; ends about 17:25).
  - **The driver's pass (since about 10:07) decided R5 and R16's gate at 16:42 and now scores D's first tuning run on navdev (R10's grid).** It then queues B's and C's four tuning runs each (R15, `46b*`, `46c*`), which sort ahead of D's last two. GPU order from about 17:25: D's seed 2 navtest paths under RL6, B's and C's tuning (8 runs, about 19 h), D's last two tuning runs, then the full RL stages.
  - **Done 14:58: D's first tuning run** (`46d0`, 10 anchors), kept twice with its MODELS row.
  - **The driver still runs the 5349fbd code** (since 09:27). It is restarted on 2e5e0d7 (the phase-6 reorder) at the first check that finds it not scoring; no chain job can be queued before A's full RL ends, about a day away.
  - **Done 12:24: phase 4's last RL run** (D's seed 2 under RL4). All 16 phase-4 RL runs are trained and kept twice.
  - **Failed jobs:** none since Nov 20. The one in `failed/` is the selector smoke test of Nov 20 17:42 ("1257 scenes have no true scores" in its last step), replaced by the full-scale rehearsal, which ran without error.
  - **Done since Nov 21 20:11:** A4 (21:08) and phase 2's navtest paths; phase 4's RL runs for A (all 6), B and C (both recipes), D's seed 0 (both) and seed 1 under RL6. Every job ended rc 0.
  - **No checks ran from Nov 21 20:11 to Nov 22 09:12** (13 h); the desktop automation carried the work, apart from the navtest bug above.
- **Desktop CPU queue (`~/queue2`):** idle since 11:24, when the rehearsal's candidates were all scored.
  - **Done 09:52: the shared boundary set** (R14), the first CPU job of the chain at full scale. 16 perturbations of each of the 80,626 RL scenes' logged paths: 1,290,016 paths scored in 3.04 h (117.7 paths/s, 0 errors). A lateral NC or DAC limit lies within ±4 m in 75,992 scenes (94%); a longitudinal one within 0.6-1.4 times the path's length in only 9,517 (12%), so most scenes' longitudinal boundary pair is the outermost two, both passing. Every later chain reuses the set.
- **Queue reordered (14:49, phase 6): each contender's selector chain now runs right after its own full RL stage**, ahead of other contenders' full RL stages (job names `470<x>-6...`, which sort before `47...`; `pilot_sft/heads/driver.py` `chain_job`, with a test).
  - **Why:** R16's gate will very likely admit B and C, which makes phase 6 eight contenders. Before, every chain waited for all eight full RL stages, so the first generator-plus-selector system came about a day later, and the chain's one part never run at scale (half generators with RL; the rehearsal's halves had no RL) would first run near the end.
  - **What does not change:** every contender gets the same tuning, full RL stage and chain; the tuning runs still go first; the final system is still chosen only when every contender's selector is decided (R16). The total GPU time is the same.
  - **Second step (15:49): the decisive part of each chain first, the rest later.** Adoption, and so R16's final system, reads only the cross-fitted seed-0 selector (R14). The in-fold control, the seed-1 selector and the four ablations are reported, and no choice reads them, so their jobs (`48<x>-...`) now sort after every full RL stage and fill the GPU's gaps. With eight contenders that is about 1.4 days less GPU work before the final system [D: 8 x (6 selectors x about 40 min + the in-fold candidates)]. Every contender still gets all of them before the report. If you want the boundary selector to compete for adoption (the question below), its jobs move up again.
  - The rehearsal's finished jobs keep their names, so nothing is queued twice. The driver picks the change up at its next restart, when it is not scoring; no chain job can be queued before A's full RL ends (about a day away).
- **Queue reordered (14:16, 266b83e): phase 2's follow-up runs now go last.** The leave-one-out, bin-size and R17 seed runs of A's continuity series (up to 7 training runs, about 16 GPU hours) are queued as `z_<name>.sh`, so they run only when nothing else waits.
  - **Why:** they feed no choice. A's tuned configuration reads the series alone (R15), and R4 calls them exploratory. The rehearsal has shown the selector chain works at full scale, so the final system is limited only by compute. Phase 4, D's tuning and phase 6 now come about 16 GPU hours earlier.
  - **What does not change:** A0-A4 themselves (the "how continuous" series) run now, as planned. The follow-up runs still run, after the final system's work or in its gaps; the plan's cut order already drops the bin-size runs first if time runs short.
  - The driver runs it since 14:41 (65b02d8), well before the series' best is known (about 23:00).
- **Time to the final system, an estimate [D]** (redone 16:50 Nov 22 for eight contenders; one GPU job at a time):
  - Tuning: D's last two runs and B's and C's eight, about 24 h (to about Mon Nov 23 17:00).
  - Each contender's full RL stage, about 3.1 h per 5,000-update round, then its decisive chain: two half generators (about 1 h of training and 3.1 h of RL each) and the adopted selector (about 40 min): about 12 h per contender with one round, 96 h for eight.
  - So the final system is about Fri Nov 27 17:00 if every stage stops after one round. R18 adds about 9.3 h for each extra round a contender takes (its full stage and both halves). Then each contender's navtest score.
  - After it: the in-fold controls, seed-1 selectors and ablations (about 36 h), then phase 2's follow-up runs (about 16 h). Phase 6 is not shortened (HEADS_PLAN, "Schedule risks"); quality over schedule.
- **Desktop CPU, the driver** (`pilot_sft/heads/driver.py`): restarted on 6c341e1 (navtest for every configuration) at 15:41; its first pass was OK, with no failed passes. Both queue runners are up.
- **Anvil** (Nov 22 14:51): nothing of ours running or queued; the GPU work stays on the desktop. The gpu partition holds 697 pending and 56 running jobs (15 nodes partly in use, 1 out of service); ai 340 and 40. Our wait statistics are unchanged: gpu 36 started, median wait 42.5 h (12.3-133.4 h), 1,816.7 h waiting for 56.0 h running; ai 5 started, median 30.6 h; shared 45 started, median 0.3 h. SU left: CPU 1,058,912, GPU 1,370.7, AI 2,764.0. No desktop connections open (0 from login01, 0 control sockets, 0 local ssh). The navtrain cache job finished Nov 20; the navdev cache job was cancelled Nov 19 (the desktop built it).

## Latest results
- **Phase 4 is decided (16:42): R16's gate admits B and C, so phase 6 has eight contenders** (A, B, C and D, each under RL4 and RL6). On navdev, B's intervals against the best of A and D include 0 on both measures, and C is level with A on the single path and leads D on best-of-6 (`choices/phase6_gate_bc.json`). **R5, reported only:** A leads navdev under both recipes with no clear winner; under RL6, A - C +0.10 (-2.14 to +2.46), A - B +2.41 (-0.46 to +4.82), A - D +7.84 (+6.14 to +10.22).
- **Phase 4 on navtest, A, B and C complete** (16:47, `method_seeds.py`; A three seeds, B and C one; D's last three runs are scored tonight):

  | navtest after RL | A | B | C |
  |---|---|---|---|
  | RL4, single path | 87.88 | 87.02 | 88.84 |
  | RL6, single path | 88.63 | 87.37 | 88.50 |
  | RL4, best-of-6 | 91.17 | 92.65 | 92.52 |
  | RL6, best-of-6 | 92.98 | 92.90 | 93.22 |

  - **Under RL6, A and C cannot be told apart** (A - C +0.13, -0.76 to +0.98), both about 3 points above H1's 85.6; **B is behind both** (A - B +1.26, +0.32 to +2.14; B - C -1.12, -2.16 to -0.03). Best-of-6 under RL6: all three within 0.3 of each other.
  - **Flow matching gains the most from RL:** C's change from phase 3 is +4.50 (+3.76 to +5.25) under RL4 and +4.16 (+3.51 to +4.88) under RL6, against A's +3.19 / +3.94 and B's +2.46 / +2.81. Before RL, C was measurably behind A and B (B - C +0.79, +0.22 to +1.32); after RL6 it leads B.
  - **Under RL4, A's candidates fall behind B's and C's** (best-of-6 A - B -1.48, -2.72 to -0.29; A - C -1.35, -2.53 to -0.22); under RL6 they are level. RL6 - RL4 on best-of-6: A +1.82 (+1.15 to +2.57), C +0.70 (+0.41 to +0.98), B +0.25 (-0.35 to +0.87).
- **Phase 4 on navdev is complete: every method under both recipes, every seed (15:44, `method_seeds.py`).** Mean of the seeds at R11's checkpoints (A and D three seeds, B and C one):

  | navdev after RL | A | B | C | D |
  |---|---|---|---|---|
  | RL4, single path | 86.37 | 85.07 | 86.36 | 76.68 |
  | RL6, single path | 86.62 | 84.21 | 86.52 | 78.78 |
  | RL4, best-of-6 | 89.43 | 91.00 | 90.61 | 89.11 |
  | RL6, best-of-6 | 90.82 | 91.20 | 91.69 | 91.68 |

  - **RL6 keeps the candidates better than RL4 wherever it is measured with an interval:** RL6 - RL4 on best-of-6 is +2.58 (+0.68 to +4.52) for D on navdev (three seeds), +1.82 (+1.15 to +2.57) for A on navtest (three seeds), +1.08 (+0.26 to +1.96) for C on navdev (one seed). RL4 costs D -2.72 (-5.09 to -0.47) of best-of-6 against phase 3; RL6 -0.14 (-1.57 to +0.95).
  - **D after RL6 has candidates as good as any method on navdev** (best-of-6 A - D -0.87, -1.95 to +0.80; C - D +0.01, -1.99 to +2.73) and an own pick 7.8 points below A's (+6.12 to +10.23).
  - **R16's gate for B and C** reads these navdev numbers: on best-of-6, C (RL6 91.69) is level with the best of A and D (D under RL6, 91.68) and B's interval against it includes 0, so both look set to join phase 6. The driver decides after B's and C's navtest scoring, about 16:50.
- **B after RL4 on navtest (15:26): 87.02 single path, 92.65 best-of-6**, against 84.56 and 91.87 for the same seed before RL: +2.46 (+1.51 to +3.35) and +0.78 (+0.08 to +1.41), one seed, so drive intervals only. **B after RL6 (15:51): 87.37 single path, 92.90 best-of-6** (+2.81 and +1.03 over the same seed before RL). So on navtest, as on navdev, RL lifts B by about 2.5-3 points on the single path, and RL6 keeps its candidates a little better than RL4. **C after RL4 (16:17): 88.84 single path, 92.52 best-of-6** (+4.50 and +0.61 over the same seed before RL, 84.34 and 91.91): flow matching gains the most from RL4 of the four methods, and its one seed sits next to A's best (89.05). C under RL6 lands about 16:45, then R5's rankings and R16's gate in the same pass.
- **A on navtest after RL, with every seed (13:43, `method_seeds.py`; intervals redraw drives and seeds):**

  | A on navtest | Single path | Best-of-6 |
  |---|---|---|
  | RL4, change from phase 3 (each RL seed against its own supervised seed) | +3.19 (+1.69 to +4.88) | -0.69 (-1.47 to +0.15) |
  | RL6, change from phase 3 | +3.94 (+2.95 to +4.92) | +1.13 (+0.55 to +1.78) |
  | RL6 - RL4 (each RL6 seed against the RL4 seed from the same start) | +0.75 (-0.35 to +1.79) | **+1.82 (+1.15 to +2.57)** |

  - **RL6 keeps A's candidates measurably better than RL4** (best-of-6, interval excluding 0); its single-path lead is within noise. navdev agrees in direction: RL6 - RL4 on best-of-6 is +1.39 (-0.21 to +2.96) for A and +1.08 (+0.26 to +1.96) for C (one seed).
  - Reported only: R16 takes both recipes into phase 6, and the final system is chosen by its navdev generator-plus-selector score. The comparison is in the report (version 23) and in `docs/navsim_eval/results/heads_method_seeds_2026-11-21.json` (updated in place; phase 3 and phase 4 on navdev unchanged).
  - **D's seed 1 under RL4 is chosen (13:26): rl_250**, navdev 76.70 against 76.59 at the start; it fell to 57.64 at update 750 and ended at 72.87. Its navtest paths are queued (`350_navtest-d-l12-18-24-s1-rl`, after D's tuning run).
  - **D's seed 2 under RL4 is chosen (13:50): rl_500**, navdev 76.07 against 76.02 at the start (lowest 68.48 at update 250). **So RL4 leaves all three D seeds within 0.16 of their starts on navdev** (+0.16, +0.11, +0.05), and even that reading is optimistic, as R11 picks the checkpoint on navdev itself. Under RL6, D's seeds 0 and 1 gain +1.04 and +3.32. D's seed 2 under RL6 is the last phase-4 choice; R16's gate for B and C follows on the driver's next pass.
  - **D under RL6 on navtest, seeds 0 and 1 (14:15, 14:40): 83.42 and 83.47 single path, 93.88 and 93.52 best-of-6**, against 82.92 / 82.08 and 93.67 / 93.24 for the same seeds before RL (single path +0.50, +1.39; best-of-6 +0.21, +0.28). Under RL4, seed 0 moved -0.09 and -1.41. So RL6 lifts D a little and keeps its candidates, and **D's best-of-6 is the highest after RL so far** (93.88 on seed 0, against 93.09 for A's best seed under RL6; B and C are not scored on navtest yet), while its own pick stays about 5 points below A's. That gap is what D's selector (phase 6) works on. Seed 2's choice and the seed-aware intervals follow.
  - **D's seed 2 under RL6 is chosen (15:00): rl_1000, navdev 78.28 against 76.02 at the start (+2.26).** So on navdev, at R11's checkpoints, RL6 lifts D's three seeds by +1.04, +3.32 and +2.26 and RL4 by +0.16, +0.11 and +0.05 (point estimates; the seed-aware intervals come from `method_seeds.py` at the next check).
  - **All 16 phase-4 RL runs are now kept twice** (14:16): D's last four copied to Anvil with every checkpoint's SHA-256 matching the desktop's, and their rows added to `pilot_sft/MODELS.md` (seed 2 under RL6 marks its choice as pending).
- **Path generators above H1 on navtest (Nov 22): A after RL4 scores 87.27, 89.05 and 87.33 on navtest's single path (seeds 0-2; mean 87.88)**, against H1's text output at 85.6. Best-of-6: 90.62, 91.56, 91.33. **Under RL6: 88.92, 88.73, 88.24 (mean 88.63) single path, and 93.09, 93.05, 92.81 (mean 92.98) best-of-6** (12:41). So on navtest A after RL6 is ahead of A after RL4 on both measures (single path 88.63 against 87.88; best-of-6 92.98 against 91.17), and 3.0 above H1. **D after RL4, seed 0 (13:06): 82.83 single path and 92.26 best-of-6**, against 82.92 and 93.67 for the same seed before RL, so RL4 leaves D's single path where it was and costs it 1.4 points of best-of-6, as on navdev. **Added at 13:21 (5802cda), before B, C and D land:** each method's RL6 minus RL4, each RL6 seed against the RL4 seed from the same start (`method_seeds.py`, reported only; R16 keeps both recipes), drawn in the report under phase 4. A's comparison runs at the next check. The seed-aware intervals come once B, C and D are scored. The paired interval and the other 12 phase-4 runs follow as their navtest scores land (about 5 h). **The 20 s replay page for the best single path so far is built (12:55):** A after RL4, seed 1 (navtest 89.05), against H1 on the 10 replay drives. It is on the home PC at `navsim_demo_renders/replay_v2_a1s1rl4_vs_h1.html` (local only: nuPlan camera images). Of the two drives where H1 fails NAVSIM at t = 0, A passes both (1.00 at b9e84e37, where H1 collides; 0.98 at c2aa6f44, where H1 leaves the drivable area). Over the re-plans, A's plans sit farther from the human's path than H1's on every drive (per-drive means 0.36-1.83 m against 0.13-1.05 m) [V]. Whether RL caused that is not measured here: there is no page for A before RL. The stored drive labels named the wrong model at c2aa6f44 ("both models fail"), so the page now writes each drive's t = 0 note from the pair's own scores (7338233, with a test). A run that beats 89.05 gets its own page.
- **Phase 4 on navdev (Nov 22 09:12; 13 of its 16 RL runs, D's still running): RL lifts A, B and C by 2.8-5.8 points, and flow matching (C) catches up with masked diffusion (A).** navdev single path after RL, each seed at its R11 checkpoint, and the change from phase 3 with every RL seed paired with the supervised seed it started from (`method_seeds.py`):

  | Method | RL4: after (change) | RL6: after (change) |
  |---|---|---|
  | A (3 seeds) | 86.37 (+4.81, +1.86 to +6.69) | 86.62 (+5.06, +3.50 to +7.01) |
  | B (1 seed) | 85.07 (+3.67, +0.42 to +5.79) | 84.21 (+2.81, +1.61 to +4.60) |
  | C (1 seed) | 86.36 (+5.59, +2.73 to +6.98) | 86.52 (+5.75, +2.07 to +7.64) |
  | D (2 of 3 seeds so far) | seed 0: 77.26 | seeds 0, 1: 78.14, 79.91 |

  - **After RL, A and C cannot be told apart** (A - C +0.01, -2.58 to +2.22, under RL4; +0.10, -1.70 to +2.13, under RL6). B is 1.3-2.4 below A, with intervals reaching 0.
  - **RL6 keeps the candidates; RL4 does not.** A's best-of-6 changes by -0.18 (-2.98 to +2.04) under RL4 and +1.21 (-0.60 to +2.21) under RL6; B and C gain best-of-6 under both. That matters for the selector, which needs good candidates.
  - **For scale:** H1's text output scores 82.4 on navdev, so A, B and C after RL are 1.8-4.2 points above it there (point estimates). navtest comes next (below).
  - **D barely moves:** 77.10 to 77.26 (RL4) and 78.14 (RL6) on seed 0; seed 1 76.59 to 79.91 (RL6).
  - **R16's gate for B and C** decides once D's last runs are chosen (about 12:00). C is level with A, so it will very likely join phase 6.
- **A bug cost phase 4 its navtest paths overnight, now fixed (5349fbd).** My R18 refactor (a4288cc) left `rl_choose` reading a variable it no longer defined. Each pass that chose a phase-4 checkpoint stopped right after writing the choice, and the next pass skipped the navtest step, so none of the 11 RL runs chosen from Nov 21 22:32 got navtest paths. No RL run or choice was affected. The fix queues them for fresh and saved choices alike, with a test that fails on the old code. The driver restarted on it at 09:27 and queued all 13 navtest jobs at once; they run after the current RL run.
- **Phase 2 is complete: no step toward continuous values is beyond seed noise (R17).** Each run minus A1 on navtest's single path: A0 +0.27 (-0.10 to +0.63), A2 +0.51 (+0.20 to +0.85), A3 -0.07 (-1.05 to +0.82), A4 +0.89 (+0.59 to +1.19). A4's interval excludes 0 but stays below 1.4 points. The series' best on navdev is A3 (82.79, against A1's 81.86), so A's tuned configuration for phase 6 is A3 (R15), and its full RL stages are queued (`47a4`, `47a6`, from epoch 5). A3's leave-one-out, bin-size and R17 seed runs are queued last.
- **Phase 2, how continuous: R17's first readings** (Nov 21 17:11; each run minus A1, both at their own navdev choices, paired; R17 calls an effect measured only above 1.4 points with its drive interval excluding 0, on navtest's single path).
  - **A0, plain bins with no notion of distance:** navtest single path 84.94 against A1's 84.67: **+0.27 (-0.10 to +0.63), within seed noise.** Its navtest best-of-6 is 91.09 against A1's 91.77: **-0.69 (-0.94 to -0.43)**, an interval excluding 0 but below R17's 1.4 points. So A1's distance-aware embeddings may make slightly better candidates while leaving the own pick unchanged (one seed each). navdev: single path -0.05 (-1.86 to +0.87), best-of-6 -1.07 (-1.94 to +0.08). So far the distance-aware bin embeddings of A1 do not measurably help the own pick; they may help the candidates (best-of-6), but not beyond noise yet.
  - **A2, A1 plus soft labels:** navtest single path 85.19 against A1's 84.67: **+0.51 (+0.20 to +0.85)**, an interval excluding 0 but below R17's 1.4 points, so within seed noise (A1's own seeds span 84.20-85.20). navdev: single path +0.50 (-0.21 to +0.93), best-of-6 +1.03 (+0.13 to +1.97). Its navtest best-of-6 is 92.27 against A1's 91.77 (the paired interval comes with the next aggregate).
  - **A3, A2 plus the position-error loss and the averaged readout (navdev only so far):** its choices are epoch 5, 2 steps, setting 1.0; single path 82.79, +0.93 against A1 (+0.34 to +2.35); best-of-6 90.29, +0.65 (-0.68 to +2.49). Its navtest paths run after A4.
  - **So far:** each step toward continuous values looks slightly better on the single path (A0 +0.27, A2 +0.51 on navtest; A3 +0.93 on navdev), but none is beyond R17's seed-noise threshold yet. A4 (the continuous offset) trains now.
- **The selector works at full scale (the rehearsal, 12:09; no contender, used for no choice).** On A1 without RL, the cross-fitted selector (seed 0) lifts navdev from the own pick's 81.43 to **82.77: +1.35 (+0.48 to +3.65), the interval excluding 0.**
  - It closes 14% of the gap to the best of the 7 candidates (90.81). It switches in 38% of scenes: 145 wins, 84 losses, 56 ties. Its picks fail NC or DAC in 72 scenes, against 79 for the own pick.
  - Held-out training drives (10 drives, 9,737 scenes) chose epoch 10 and margin 0.02: selected 91.01 against the own pick's 89.28 there.
  - For scale: H1's text output scores 82.4 on navdev, so A1 plus this selector is level with or above it before any RL (point estimates; no paired test).
  - Training took 34.7 min (7.2M parameters, 70,889 training scenes from 91 drives, about 800 scenes/s).
  - **The boundary ablation did better (12:51):** the same selector trained with 4 boundary paths per training scene (training-only candidates that straddle the NC or DAC limit) gives **+2.08 on navdev (+0.93 to +5.15)**, closing 22% of the gap. It switches in 34% of scenes (128 wins, 75 losses, 48 ties), its picks fail NC or DAC in 67 scenes against 79, and held-out training drives agree (91.20 against 91.01). 41.7 min to train. **Paired on the same 743 scenes, the boundary version minus the main selector is +0.74 (-0.30 to +2.10)**: they pick differently in 34% of scenes, and the interval includes 0, so the boundary version's edge is suggestive, not established (`pilot_sft/heads/selector_pair.py`; `docs/navsim_eval/results/heads_selector_rehearsal_pair_2026-11-21.json`).
  - What it means for the final system: R14's selector adds a measured gain on top of a supervised generator, so phase 6's chains, where it is judged after RL, start from a working design.
- **Phase 3 is complete for all five, R included (Nov 21 12:12): three seeds each.** R's navtest single path 84.57, 84.88, 84.46 (mean 84.64). With three seeds on both sides, R is level with A (+0.05, -0.53 to +0.63) and B (-0.00, -0.37 to +0.36) and above C (-0.79, -1.36 to -0.19) and D (-2.45, -3.42 to -1.51).
- **Phase 3 is complete for A, B, C and D (Nov 21 06:42): three seeds each.** With every seed, D has the best candidates of any method (navtest best-of-6 93.47, against 91.35-91.99; each pair's interval excludes 0) and the worst own pick (82.20, against 83.85-84.69). A and B cannot be told apart; C is below both on the single path. R, the regression control with one path, is level with A and B on navtest's single path and above C and D, now with two seeds (A - R -0.04, -0.62 to +0.55; C - R -0.88, -1.45 to -0.25; D - R -2.53, -3.50 to -1.53); its seed 2 is queued. Intervals below ("Phase 3 is complete for A-D").
- **Phase 1 is complete (14:47).** A, B and C cannot be told apart on navtest (single path or best-of-6, at their layer choices). The regression control R is level with them on the single path: its own pick is not beaten by any path generator's before RL. D has the best candidates and the weakest own pick (table below).
- **R13 is complete. The driver chose RL4 (Dr. GRPO) at 10:17 by the registered rule, and phase 4 has started with it.** The rule (R12, R13) takes the recipe with the highest navdev single-path change from the start at its better checkpoint (250 or 500 updates), averaged over A and D. Numbers are on navdev (743 scenes, 11 drives), seed 0, with the change and its drive interval in brackets:

  | Recipe | A (start 81.43, epoch 20): 250, 500 updates | D (start 77.10, epoch 5): 250, 500 updates | Mean change |
  |---|---|---|---|
  | RL0: as registered (lr 1e-5, no KL, 8 x 32) | 75.72, 77.19 (-4.24, -10.43 to -0.53) | 60.93, 64.23 (-12.88, -16.44 to -8.39) | -8.56 |
  | RL1: KL beta 0.04 | 77.58, 78.76 (-2.67, -8.09 to +0.55) | 75.27, 75.28 (-1.82, -2.93 to 0.00) | -2.25 |
  | RL2: KL, lr 3e-6 | 79.64 (-1.79, -7.20 to +1.03), 79.01 | 75.55, 75.80 (-1.30, -2.86 to +0.47) | -1.54 |
  | RL3: KL, lr 3e-6, 16 x 16 | 80.68 (-0.75, -6.17 to +2.17), 78.61 | 75.95 (-1.16, -3.24 to +0.35), 74.90 | -0.95 |
  | **RL4: Dr. GRPO** | 85.08, **86.30 (+4.87, +3.42 to +7.74)** | 76.84 (-0.27, -2.11 to +3.26), 70.39 | **+2.31** |
  | RL5: BC | 81.99 (+0.56, -3.10 to +2.55), 78.11 | 67.55 (-9.55, -11.08 to -5.62), 66.25 | -4.50 |
  | RL6: Dr. GRPO + BC | 83.61, 84.75 (+3.32, +2.12 to +5.89) | 78.24 (+1.14, -0.41 to +2.46), 76.89 | +2.23 |
  | RL7: neither (SD floor only) | the RL0 run | 62.39, 62.53 (-14.57, -17.64 to -12.10) | -9.40 |

  - RL4-RL7 keep RL0's settings. For D they add a 0.1 SD floor inside the log-probability; A's steps are not Gaussian, so A's RL7 is its RL0 run. The BC weight is set so the BC gradient norm equals the RL term's at the start: 2.50 and 0.73 on A, 2.65 and 0.40 on D, for RL5 and RL6.
  - **A after RL4 is above H1's text output on navdev:** 86.30 against 82.4, +3.89 (+2.51 to +5.14). This is seed 0, from the recipe stage; navtest comes with phase 4 (3 seeds, 1,500 updates).
  - **RL4 pulls A's 6 candidates together.** Their end points go from 6.4 m apart to 0.8 m, and best-of-6 stays where it was (90.27 to 90.13). With BC (RL6) they stay 2.2 m apart, and best-of-6 is the highest of any run (90.93). Every R12 recipe lowered A's best-of-6 (to 84.0-87.7).
  - **D:** only RL6 raised it, and its interval includes 0. Dividing by the group SD does D the most harm: RL7 (the SD floor alone) -14.6 and RL5 (BC with the division) -9.6, against -0.3 for RL4.
  - **RL4 and RL6 are within 0.1 on the mean (+2.31 against +2.23).** The rule takes the point estimate, so phase 4 runs RL4 (see "Waiting on the researcher").
  - **On the training scenes** (mean of the first 50 updates, then the last 50): A under RL6, reward 0.81 to 0.89, with time-to-collision 0.89 to 0.95 and progress 0.77 to 0.84 both rising. D under RL6, reward 0.85 to 0.87. D under RL7, reward 0.77 to 0.67, with every sub-score falling. Under R12, A's safety fell whenever its progress rose.
  - **Why R13 was added** (registered 07:02, before any of its runs: `docs/HEADS_PLAN.md`, 4fed42c; code 714cd6f). R12's recipes all lowered both methods: A traded safety for progress (under RL0, progress 73.5 to 78.3, time-to-collision 93.9 to 84.4), and D was damaged everywhere. Each of the three likely causes has a published remedy:
    - the group-SD division: Dr. GRPO (NoRD on navtest: GRPO +0.5, Dr. GRPO +9.0);
    - no tie to the logged driving: a BC term, used in every published RL stage on a separate NAVSIM path generator;
    - D's tiny last-step SD: a 0.1 floor inside the log-probability, as in DiffusionDriveV2.
- **Phase 1 so far** (single path / best-of-6 at each training run's navdev choices; navtest reports only):

  | Method | navdev, layer 24 | navdev, layers 12/18/24 | Layer choice | navtest at the choice (vs H1's 85.6) |
  |---|---|---|---|---|
  | A1 (masked) | 81.5 / 89.5 | 81.9 / 89.7 | 12/18/24 | 84.7 / 91.8 (-0.9, -1.4 to -0.5) |
  | B (DDIM) | 81.0 / 88.2 | 81.4 / 89.3 | 12/18/24 | 84.6 / 91.9 (-1.05, -1.5 to -0.6) |
  | D (anchored) | 76.8 / 89.6 | 78.4 / 92.2 | 12/18/24 | 82.9 / **93.7** (-2.7, -3.5 to -1.8) |
  | C (flow) | 78.4 / 89.5 | 80.8 / 89.7 | 12/18/24 (12:04) | 84.3 / 91.9 (-1.27, -1.9 to -0.6) |
  | R (regression, one path) | 81.0 / - | 81.6 / - | 12/18/24 (14:47) | 84.6 (-1.03, -1.5 to -0.6); layer 24: 84.8 (-0.85, -1.4 to -0.3) |

  - **Layer test on navtest** (layers 12/18/24 minus 24 alone; single path; best-of-6):
    - A: -0.01 (-0.43 to +0.41); -0.03.
    - B: +0.13 (-0.25 to +0.51); +0.85 (+0.55 to +1.18).
    - D: +1.28 (+0.77 to +1.78); +2.17 (+1.73 to +2.55).
    - C: +0.83 (+0.44 to +1.19); +0.29 (-0.00 to +0.63).
    - So the best layers depend on the method, the case the per-method layer test was there to catch.
  - **Method pairs on navtest, at the layer choices** (single path; best-of-6):
    - A - B: +0.11 (-0.34 to +0.54); -0.10 (-0.48 to +0.21). Indistinguishable, although B's candidates are much closer together (end points 1.4 m apart, against 6.6 m for A).
    - A - C: +0.33 (-0.31 to +1.04); -0.13 (-0.61 to +0.28). B - C: +0.22 (-0.33 to +0.80); -0.03 (-0.30 to +0.23). **So A, B and C cannot be told apart** on navtest, single path or best-of-6. C's candidates are 1.4 m apart, like B's.
    - A - D: +1.75 (+0.93 to +2.60); -1.89 (-2.21 to -1.57).
    - B - D: +1.64 (+0.96 to +2.35); -1.79 (-2.18 to -1.36).
    - C - D: +1.42 (+0.77 to +1.99); -1.76 (-2.26 to -1.21).
  - **Generation steps (navtest, single path, layers 12/18/24, point estimates):** two steps come within 0.1 of each method's best step count. One step costs 0.1-0.8, and A loses 0.4-0.7 at 8 steps. navdev chose 8 steps for A1, which scores 84.67 on navtest against 85.10 at 4 steps. Generation from stored features takes 0.24-1.4 ms per scene on the 4090 (batch 128), 6 paths included. Details: `docs/GAPS.md` H7 and H8.
  - **R, the regression control with no generation, is level with the path generators' own picks** (navtest, 16:40).
    - R at layers 12/18/24 (its layer choice) scores 84.57, -1.03 against H1's text output (-1.48 to -0.58). At layer 24 it scores 84.76. R's layer test: -0.18 (-0.64 to +0.22), so no layer effect.
    - Paired on navtest, at each method's layer choice:
      - A - R +0.10 (-0.26 to +0.47);
      - B - R -0.02 (-0.32 to +0.32);
      - C - R -0.24 (-0.76 to +0.24);
      - D - R -1.65 (-2.42 to -0.90).
      - navdev agrees: A - R +0.28 (-1.14 to +1.76), D - R -3.16.
    - So before RL, no path generator's own pick beats plain regression. What generation adds so far is its candidates (best-of-6 91.8-93.7 on navtest) and a handle for RL. That is why R16 judges the final system with its selector.
  - **What the gap to H1's text output is made of** (navtest sub-scores, layers 12/18/24, point estimates; `docs/navsim_eval/results/heads_by_command_2026-11-20.json`):
    - Every path generator and R trails H1 most in drivable-area compliance (DAC): -0.6 to -0.9 points for A, B, C and R, and -2.3 for D. Then come time-to-collision (-0.4 to -1.5) and progress (-0.1 to -2.1).
    - DAC is a gate (a failed scene scores 0), so each point of DAC costs about 0.9 PDMS [D]. That is most of A's -0.94.
    - Turns carry the failures. DAC fails in 3.6% of H1's straight scenes (8,070) against 9.6-10.7% of its turns (2,501 left, 1,575 right).
      - D fails 5.2% straight and 13.7% at turns.
      - R loses about 1.5 PDMS to H1 at turns, against 0.8 going straight.
      - A stays about 1 point behind H1 in every command.
    - Turns are where a selector that predicts DAC per candidate has the most room.
  - **Phase 3 is complete for A-D: method pairs with every seed** (Nov 21 06:42; `pilot_sft/heads/method_seeds.py`, `docs/navsim_eval/results/heads_method_seeds_2026-11-21.json`).
    - **Method:** R5's hierarchical bootstrap. Each resample redraws drives (with all their scenes) and each method's seeds, at seed 0's choices. R has three seeds (Nov 21 12:12), resampled like the others.
    - **Seeds on navtest** (single path; best-of-6):
      - A: 84.67, 84.20, 85.20 (mean 84.69, SD 0.50); best-of-6 mean 91.85 (SD 0.35).
      - B: 84.56, 84.74, 84.61 (mean 84.64, SD 0.09); 91.99 (SD 0.13).
      - C: 84.34, 83.55, 83.65 (mean 83.85, SD 0.43); 91.35 (SD 0.49).
      - D: 82.92, 82.08, 81.59 (mean 82.20, SD 0.67); 93.47 (SD 0.22).
      - R (one path, no best-of-6): 84.57, 84.88, 84.46 (mean 84.64, SD 0.22). Seed 1 minus seed 0 +0.31 on navtest (-0.06 to +0.73); seed 2 minus seed 0 -0.11 (-0.48 to +0.31).
    - **Pairs on navtest** (difference of seed-averaged means, 95% interval over drives and seeds):

      | Pair | Single path | Best-of-6 |
      |---|---|---|
      | A - B | +0.06 (-0.56 to +0.68) | -0.14 (-0.67 to +0.33) |
      | A - C | +0.85 (+0.06 to +1.60) | +0.50 (-0.18 to +1.12) |
      | B - C | +0.79 (+0.22 to +1.32) | +0.64 (+0.09 to +1.13) |
      | A - D | +2.50 (+1.41 to +3.67) | -1.61 (-2.11 to -1.05) |
      | B - D | +2.44 (+1.56 to +3.31) | -1.47 (-1.94 to -0.95) |
      | C - D | +1.65 (+0.62 to +2.60) | -2.11 (-2.73 to -1.40) |
      | A - R | +0.05 (-0.53 to +0.63) | - |
      | B - R | -0.00 (-0.37 to +0.36) | - |
      | C - R | -0.79 (-1.36 to -0.19) | - |
      | D - R | -2.45 (-3.42 to -1.51) | - |

    - **navdev, where choices are made, agrees.** D's own pick is below A's, B's and C's by 2.78-3.75 points, and its best-of-6 is above theirs by 2.10-2.26. Every one of those intervals excludes 0. A, B and C cannot be told apart on navdev.
    - **What it means for the final system:** D has the best candidates of any method with every seed and the worst own pick. Its gap from own pick to best candidate is 11.3 points on navtest, against 7.2-7.5 for A, B and C [D]. That gap is what a selector can recover, which is why R16 takes D to phase 6 under both recipes. Phase 4's RL comes first.
    - **D's seed spread is the largest:** seed 2 is 1.33 below seed 0 on navtest (-1.89 to -0.78), a range of 1.33 points. So a one-seed comparison can be off by about a point.
    - Before RL, DDIM (B) beats rectified flow (C) on this body, masked diffusion (A) matches DDIM, and anchored diffusion (D) trades its own pick for its candidates. These are reports; the confirmatory tests C1-C4 come at the end, Holm-corrected.
  - **Seed spread: A1's three seeds (19:12; phase 3; `docs/navsim_eval/results/heads_seed_pairs_2026-11-20.json`).** At seed 0's choices, layers 12/18/24:
    - navtest single path 84.67, 84.20, 85.20: mean 84.69, between-seed SD 0.50. Paired against seed 0: seed 1 -0.47 (-0.72 to -0.19), seed 2 +0.53 (+0.20 to +0.90).
    - navtest best-of-6 91.77, 91.55, 92.24: mean 91.85, SD 0.35.
    - navdev single path 81.86, 81.05, 81.76: mean 81.56, SD 0.44.
    - B's three seeds (22:49): navtest single path 84.56, 84.74, 84.61, mean 84.64, between-seed SD 0.09; best-of-6 mean 91.99, SD 0.13. Paired against seed 0: seed 1 +0.18 (-0.23 to +0.62), seed 2 +0.06 (-0.22 to +0.37).
    - C's seed 1 (Nov 21 00:43): navtest single path 83.55, -0.79 against seed 0 (-1.43 to -0.15); best-of-6 91.13, -0.78 (-1.10 to -0.50). On navdev it is +0.72 (-3.97 to +3.35): the same seed ranks differently on navdev and navtest, so one-seed navdev choices are noisy.
    - C's seed 2 (Nov 21 02:43): navtest single path 83.65, -0.69 against seed 0 (-1.31 to +0.03); best-of-6 91.02, -0.88 (-1.32 to -0.47).
    - So B (continuous DDIM) is about 5 times more stable across seeds than A (masked; SD 0.50 against 0.09), while their means are the same (84.69 against 84.64).
    - So training noise alone spreads one configuration by a full point on navtest, more than A, B and C differ (0.1-0.3). Method claims wait for every method's three seeds; R5 and C1-C4 carry the between-seed spread.
  - **Selector headroom on drivable-area failures (20:12; navtest, seed 0, before RL, reporting only; `docs/navsim_eval/results/heads_dac_headroom_2026-11-20.json`):**
    - When the own pick leaves the drivable area, one of the 6 samples stays inside in 64-80% of those scenes: A 69%, B 64%, C 68%, D 80% (93% of D's straight-road failures).
    - Best of 7 (own pick plus 6 samples): A 92.3, B 91.9, C 92.0, D 93.8.
    - The average candidate is far worse than the own pick for A (73.0 against 84.7) and D (68.1 against 82.9), but close for B (80.5) and C (80.1). A's and D's candidates are diverse but often bad, so their selector must be precise; R14's own-pick margin guards against that.
    - RL changes the picture: RL4 pulls A's candidates together. That is why R16 takes both RL4 and RL6 into phase 6.
  - **What RL fixes (16:50; navdev, seed 0, 743 scenes, point estimates only; `docs/navsim_eval/results/heads_rl_by_command_2026-11-20.json`):**
    - RL4's +4.9 on A comes mostly from staying in the drivable area. A's drivable-area failures fall below H1's in every command:
      - straight 6.7% to 2.7% (H1 5.8%);
      - left turns 9.9% to 6.0% (H1 8.6%);
      - right turns 12.5% to 4.5% (H1 14.3%).
    - So RL4 targets the very weakness behind the gap to H1 on navtest.
    - RL6 fixes about half as much on A (3.8%, 8.0%, 8.9%).
    - RL4 damages D at turns: left-turn failures 13.9% to 28.5%. RL6 helps D slightly.
    - navdev's turns are only 151 left and 112 right scenes from 11 drives, so this is a pointer; phase 4 measures it on navtest with seeds.
  - References: H1's text output scores 82.4 on navdev and 85.6 on navtest; constant velocity scores 19.8 on navdev.
  - Choices: A1 epoch 15, 8 steps, setting 1.5. B epoch 20, 10 steps, setting 1.5 (earlier written here as "B epoch 10"; epoch 10 is B's RL start). D epoch 10, 2 steps, setting 0.5. C epoch 10, 2 steps, setting 2.0, at both layer settings. RL starts (R11): A epoch 20, B epoch 10, C epoch 10, D epoch 5.
  - **C's layer choice (12:04): layers 12/18/24.** navdev single path 80.77 against 78.41 for layer 24 alone; best-of-6 89.74 against 89.45. C's navdev single path by epoch: 76.2, 80.8, 79.2, 80.6 (epochs 5-20); by steps at epoch 10: 79.6, 80.8, 80.0 (1, 2, 5 steps). Its candidates were best at the widest sampling setting tried (2.0: 89.7, against 87.3 at 1.0 and 84.0 at 0.5), as at layer 24.
- **The RL trial** (200 updates on a1-l24 over 3 training drives, recipe as registered) lowered A1's navdev single path from 81.18 to 73.97 (-7.2, -13.3 to -2.6). That led to R12 (eca6ee6).
- **RL scoring service:** equals NAVSIM's official scores on 200 navtest scenes, with a maximum difference of 0.0 in every sub-score.
- **Text pilot H1: 85.6 PDMS on navtest**, still the best single path. H2 (8 passes) ties it.
- **NAVSIM scoring runs on the desktop.** H1 re-scored there matches Anvil on all 12,146 scenes (PDMS 0.856069 both).

## Next (the live to-do list; the plan is `docs/HEADS_PLAN.md`)
**Coming up, in run order** (desktop GPU queue, one job at a time; times are estimates):
1. **Phase 4's last navtest scores:** D's seeds 1 and 2 under RL4 and seed 2 under RL6 (tonight), then R4's confirmatory tests C1-C4 (`confirmatory.py`; R5's RL4 leader is A) and the report.
2. **Phase 6 for eight contenders:** the R15 tuning runs (D's, then B's and C's), each contender's full RL stage (R18 rounds) and decisive chain, then the final system on navdev (R16) and one navtest score per contender.
3. **Phase 6:** D's four tuning runs (R15), then each contender's full RL stage (rounds of 5,000 updates, R18) and selector chain (R14); the final system on navdev (R16), one navtest score per contender, and R4's confirmatory tests.
4. **The report after each result** (the private link at the top of this page).
5. **A 20 s replay page for each new best** (your standing rule). A after RL4, seed 1 (89.05) has its page since 12:55 (above). A later run that beats it gets one within the hour: `scripts/desk/replay_head_predict.sh`, then `scripts/anvil/helpers/run_replay_v2_head.sh`. Local only (nuPlan images).

### Done so far (details, oldest first)
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
5. ~~Commit the rules.~~ **Done:** R1–R9 and the phase-4 settings in a64e437 and 79df830; R10, the choice procedure, in **b6ba4f5**, all before any navdev score. C's RL noise level (Flow-GRPO's SDE, a = 0.7, adapted to C's start noise) was read and committed in 0830a74 (Thu 22:51), before any RL score; this line said "still to do" until 11:01.
6. ~~Script the phase-1 choices and the phase transitions.~~ **Done:** `pilot_sft/heads/driver.py` (85dcec1), running on the desktop since 21:55. Every 5 min it:
   - scores navdev in the R10 order and writes each run's choices;
   - makes each method's layer choice;
   - queues navtest paths at the choices, phase 3's seeds and phase 2's series;
   - scores navtest for the report.
   - Its status is `~/navsim_workspace/exp/heads/driver_status.json`; its log is `driver.log`.
   - Each phase-1 training job now ends with its navdev grid, `scripts/desk/navdev_grid.sh`.
7. ~~Phase 1: navdev and navtest paths and scores for all 10 runs, and each method's layer choice.~~ **Done 14:47** (R's navtest at 12/18/24 landed 16:40). Layer choices: 12/18/24 for every method (A, B, C, D, R).
8. **RL recipe rounds: done.** R12 and R13 are complete, and the driver chose RL4 at 10:17 (`choices/rl_recipe.json` on the desktop). Phase 4 runs RL4 on every method, after phases 2-3: B's seed-0 job is queued (45b0); A's and D's 3 seeds each follow their supervised seeds 1-2; C's follows its layer choice.
9. ~~Report at 12:00 Fri Nov 20.~~ **Done 11:18, refreshed 11:45** (the work set out phase by phase; R14 added): `navsim_demo_renders/heads_report_2026-11-20.html` on the home PC (charts with drive intervals, R12 and R13, problems and fixes, next steps, the two questions below). Its numbers are in `docs/navsim_eval/results/heads_aggregate_2026-11-20.json`, and the headline numbers are on this page.
10. ~~Copy the R12 recipe runs to Anvil and add their rows to `pilot_sft/MODELS.md`.~~ **Done:** all 15 R12 and R13 recipe runs and `c-l24`, on both machines with matching checksums (last copy 10:49). `c-l12-18-24` and `r-l24` too, with matching checksums (12:53), and `r-l12-18-24` (14:51): all 10 phase-1 training runs are on both machines. Phase 3: `a1-l12-18-24-s1` (16:54), `a1-l12-18-24-s2` (18:44), `b-l12-18-24-s1` (20:44), `b-l12-18-24-s2` (22:44), `c-l12-18-24-s1` (Nov 21 00:14), `c-l12-18-24-s2` (Nov 21 02:14), `d-l12-18-24-s1` (Nov 21 04:14), `d-l12-18-24-s2` (Nov 21 06:16), `r-l12-18-24-s1` (Nov 21 08:14), `r-l12-18-24-s2` (Nov 21 11:44); phase 2: `a0-l12-18-24` (Nov 21 14:50), `a2-l12-18-24` (Nov 21 16:44), `a3-l12-18-24` (Nov 21 19:14), `a4-l12-18-24` (Nov 22 09:34); phase 4: 12 RL runs (Nov 22 09:34).

11. **Phase 6 code, written while the GPU trains** (needed after phase 5):
    - **Done:** training on one half of the training drives (`train_head.py --drive-half`, 9531c27); each half's RL-scene tokens and out-of-fold predictions (`drive_map.py halves`, `predict_head.py --only-tokens`, 5de57e2); scoring every candidate with the PDM scoring service (`score_candidates.py`, 332d20c). 85 heads tests pass.
    - **Done 11:38: R14, the selector's design**, fixed in `docs/HEADS_PLAN.md` before any selector result. Its evidence is a review of about 25 published NAVSIM selectors (`docs/research/heads/notes/selector_design.md`).
      - It predicts the 5 PDM sub-scores and combines them with NAVSIM v1.1's formula. The formula equals the official score exactly on one of our navtest score files (12,146 scenes).
      - It reads only the decoded candidates, with the own pick among them, and keeps the own pick unless another candidate is predicted to be better by a margin set on held-out training drives.
    - **Done 13:06: the selector's code**, with its implementation details added to R14 before any run on real data:
      - `selector.py`: the model, the loss, the own-pick margin and R14's measures;
      - `train_selector.py`: training on scored candidates, with 10 training drives held out to choose the epoch and the margin;
      - `apply_selector.py`: writes the selected path per scene for NAVSIM's own scorer, and reports the measures with drive intervals.
      - Measured size: 7.2M parameters, against R14's estimate of about 5M. The estimate left out the AdaLN modulation; the design is unchanged.
      - 9 new tests; all 95 heads tests pass on CPU.
      - Also fixed: `score_candidates.py` divided by an elapsed time of 0 on a coarse clock. This failed its test 4 times in 6 on Windows; it could not affect desktop runs, where each request takes seconds.
    - **Done 13:13: R14's boundary paths** (`boundary_paths.py`, an ablation): 16 perturbations of each logged path, scored, and the 4 that straddle the nearest NC or DAC limit kept as training-only candidates (`train_selector.py --extra-candidates`). Their definition was added to R14 before any run on real data. 4 new tests; 99 heads tests pass.
    - **Done 13:23: the selector's driver steps** (`driver.py`, `selector_chain`). They start from a spec file per deployed path generator, `choices/selector_spec_<X>.json`, written after phase 6's full RL stage, or for a runner-up if you choose option 2 below. Each step is queued once, only when its inputs exist:
      - the two half generators and the in-fold candidates (GPU, jobs `60`/`61`);
      - scoring every candidate set (CPU, `~/queue2`, jobs `57`, on port 47412, apart from the RL jobs' 47411);
      - the selectors (GPU, jobs `62`): the cross-fitted one with 2 seeds, the in-fold control and the 4 ablations;
      - adoption on navdev by the point estimate, then one navtest evaluation by NAVSIM's own scorer.
      - 3 new tests; all 102 heads tests pass. Nothing runs before a spec exists.
    - **Done 13:25: R15** (`docs/HEADS_PLAN.md`), registered before any phase-4 result. The plan said "the same number of runs over its own main settings" for B, C or D without naming them; R15 names them. Each method gets 4 runs, as A's continuity series has 4:
      - B: its prediction target and noise schedule;
      - C: its start-noise SD and time sampling;
      - D: its anchor count and truncation.
      - The full RL stage: 5,000 updates saved every 500, its checkpoint chosen on navdev by R11's rule.
    - **Done 13:34: phase 6's driver steps** (`driver.py`, `phase6`). They run after phase 5's clear winner, or after your choice on a tie (`choices/phase6_method.json`):
      - R15's tuned configuration;
      - the full RL stage (5,000 updates, saved every 500);
      - its checkpoint, chosen by R11's rule;
      - the selector spec, which starts the selector's chain.
      - B's, C's and D's tuning runs wait for their training flags. Those are written only if that method wins.
      - 3 new tests; all 105 heads tests pass.
    - **R16 in the driver** (registered 13:51, code at 14:01):
      - phase 4 queues RL6 beside RL4 for every method and seed (jobs `45{m}{s}r6_`);
      - R5 is computed per recipe and marked as reported only;
      - phase 6 runs every contender (A and D under RL4 and RL6, B and C if admitted): tuning, the full RL stage per recipe (jobs `47{m}{r}_`) and the selector chain;
      - the final system is chosen on navdev, then every contender is scored once on navtest.
      - D's tuning runs (R15) need `--anchor-trunc`, which `train_head.py` now has. They are queued now and run after phase 4.
      - 4 new tests; all 106 heads tests pass.
    - **Done 16:43: B's and C's R15 training flags**, ready before the gate decides, so admitted methods start tuning the same day.
      - B: the network can predict the noise or v instead of the clean path (`--target`), with DDPM's linear schedule as an alternative (`--schedule`).
      - C: training times can follow pi0's Beta or SD3's logit-normal (`--time-sampling`).
      - Training, generation, RL sampling and RL log-probabilities all use the same clean-path estimate. The defaults reproduce phase 1 exactly.
      - 10 new tests; all 116 heads tests pass. The driver now holds no method back for missing flags.
    - **Done 16:47:** the gap to H1 by sub-score and route command (above; `pilot_sft/heads/by_command.py`).
    - **Done 16:49:** one boundary-path set shared by every selector chain (the paths depend on no generator), saving about 1.8 h of desktop CPU per extra contender.
    - **Done 17:11:** the driver restarted on 1766794 (R16, R15 flags, the shared boundary set); its first pass was OK.
    - **Done 17:17: training reads were the bottleneck, so the default is now 8 feature readers instead of 4.**
      - Measured on A1's seed 2: the 4 readers ran at 420% CPU, the 4090 was about 40% busy (8 one-second samples: 0-100%), and disk wait was 2-3%.
      - Data order is fixed by the seed and the readers only prefetch, so results do not change.
      - **Measured on B's seed 1 (18:42):** 773 rows/s against 632-638 for B's phase-1 run with 4 readers, about 21% faster. The 4090 is now 67-100% busy, so the GPU sets the pace. Over the roughly 15 training runs left, that saves about 6 GPU hours [D].
    - **Done 21:44: R16's gate for B and C now uses every phase-4 seed** (amendment registered 21:42, before any phase-4 run; driver 4d436d9, restarted 21:43).
      - It applies R5's hierarchical bootstrap over drives and seeds.
      - The seed-0-only version could admit or exclude a method on seed luck: A1's seeds differ by up to 1.0 point on navtest.
    - **Done 23:13: R17 (registered 23:12, before any continuity-series run): phase 2 is read against seed noise.** Driver e9ffb8f, restarted 23:13.
      - A series effect counts only above 1.4 points with its drive interval excluding 0. That is about two seed SDs of a single-seed difference, from A's between-seed SD of 0.50.
      - The series' best, if not A1, gets seeds 1 and 2 (jobs `43{s}_`), so the "how continuous" answer rests on three seeds each.
    - **Done 00:05 (Nov 21): the RL smoke test for B and C on real data passed** (`350_rlsmoke-b-c`, 00:02-00:05; 20 updates each, outputs in `exp/heads_smoke/`, used for no choice).
      - Every run had finite rewards and gradients, and no scoring errors.
      - At update 20: B under RL4 reward 0.804 (best of 8, 0.927), B under RL6 0.812, C under RL4 0.808 (best of 8, 0.950), C under RL6 0.798. Gradient norms 3.3-9.0. About 2.2 s per update, so a 1,500-update run takes about 55 min, as A's and D's do.
      - In 31% of B's groups all 8 candidates get the same reward, against 9% for C, so those groups give no learning signal. B's candidates sit close together (1.4 m apart), which likely explains it. Phase 4 will show whether it slows B's RL.
    - **Done 05:55 (Nov 21): a rehearsal of the selector chain at full scale, ahead of the queue** (registered in R14 before any chain ran; code a85fba3).
      - **Why:** in queue order, the first contender's chain runs only after phases 2-4 and the R15 tuning runs, about two days from now. It is the least tested part of the final system, and the smoke test (Nov 20) covered 1,000 scenes on the CPU.
      - **What:** the chain as phase 6 runs it, on A's phase-1 configuration (seed 0) without RL, at its R11 starting epoch: both half generators, out-of-fold candidates for every RL scene, their PDM scores, the shared boundary set (kept for every later chain), the cross-fitted seed-0 selector and its boundary ablation, and navdev evaluation.
      - **Role:** no contender and used for no choice; a failure it finds is fixed for every chain. Cost [D]: about 4 GPU-hours and 3-4 CPU-hours.
      - **How:** `queue_job(first=True)` names a job `0_<name>.sh`, which the runners take first, and a spec may name its selector variants (`pilot_sft/heads/rehearsal_spec.py`). 4 new tests; 120 heads tests pass.
    - **Done 06:55 (Nov 21): phase 2's reading (R17), before any continuity-series result** (57303f5).
      - `collect_results.py` now writes a `series` section: each continuity-series, leave-one-out and bin-size run minus A1, each at its own navdev choices, paired on the same scenes, with R17's test (more than 1.4 points and a drive interval excluding 0; R17's call is on navtest's single path).
      - `method_seeds.py` compares the series' best with A1 over three seeds each, once its seeds 1 and 2 are scored (R17).
      - The report has a phase-2 section: a chart, a table with R17's verdict, and the three-seed table.
      - 4 new tests; 126 heads tests pass.
    - **Done 07:14 (Nov 21): phase 4 with every seed, before any phase-4 run** (8090a17; `pilot_sft/heads/method_seeds.py`). Per recipe (RL4 and RL6), once a method's RL seeds are all chosen and scored:
      - each method after RL: its seeds at their chosen checkpoints (R11; update 0 means the start);
      - the method pairs after RL, as before RL: seed-averaged means, with drives and seeds resampled;
      - each method's change from phase 3 (R10's choices), with every RL seed paired with the supervised seed it started from. B and C have one RL seed, so their intervals hold drive noise only.
      - This is the study's before-and-after-RL comparison; the report gets its phase-4 section once the first method's seeds land.
    - **Done 07:44 (Nov 21): R18, registered before any full RL stage or phase-4 run** (`docs/HEADS_PLAN.md`; code a4288cc).
      - **What:** the full RL stage runs in rounds of 5,000 updates. It continues from a round's last save only while that save is R11's choice on navdev, up to 4 rounds (20,000 updates). Every contender follows it on its own navdev curve, and the half generators then run the deployed update count. Phase 4's 1,500 updates stay the matched comparison.
      - **Why:** your direction to iterate RL while it improves, past the dates. Without it, a full RL stage still rising at 5,000 updates would stop there.
      - 2 new tests; 129 heads tests pass. `rl_choose` was split into `checkpoint_scores` and `best_update`, with the same behaviour.
    - **Done 08:46 (Nov 21): training-log fix** (683e005). An epoch's first training-loss line divided a partial window by 50, so the logged loss swung in a sawtooth (A1's ran 0.02-1.08). It looked like the rehearsal's half-0 loss had risen from 0.26 to 1.14; it had not (its dev ADE tracks A1's). Lines now average the steps they cover (logged as `window`). Training never read these numbers, so no result changes.
    - **Done 09:18 (Nov 21): the selectors in the aggregate and the report, before the first one is trained** (ab215d8). The aggregate now has every trained selector's held-out choice, training and R14's measures on navdev. The report gets a phase-6 section per chain: the adoption verdict, selected minus own pick per variant with its drive interval, and the switches and failures behind it. The rehearsal's two selectors will fill it first (about 13:40).
    - **Done 09:44 (Nov 21): R4's confirmatory tests, their wording fixed before any phase-4 run** (`docs/HEADS_PLAN.md` R4, implementation note; code 0a1184d).
      - R16 came after R4, so two phrases needed a fixed reading. "After the RL stage" is phase 4 under the registered recipe (RL4, each seed at its R11 checkpoint); RL6 is reported beside it, not tested. "The navdev-chosen method" is R5's leader under RL4.
      - C1 A minus D after RL; C2 the chosen method after RL minus H1; C3 the chosen method before RL minus R; C4 the chosen method after minus before RL, seed-paired. All on navtest's single path, by R3's hierarchical bootstrap, with two-sided p-values and Holm's correction at family-wise 0.05.
      - `pilot_sft/heads/confirmatory.py` runs them once every input exists. 2 new tests.

## Waiting on the researcher
- **Two optional questions below (the boundary-path selector and a pooled-candidate system); nothing blocks on them.** Your 13:45 answers, now R16 in `docs/HEADS_PLAN.md`, registered 13:51 before any phase-4 run had started:
  - Phase 4 runs RL4 (as registered) and RL6 on every method, with the same seeds and 1,500 updates. Neither recipe is dropped on single-path score alone.
  - R5's single-path ranking after RL is reported, no longer deciding.
  - A and D, each under both recipes, go into phase 6. B and C join if phase 4 shows they are not clearly below. Each contender gets its tuning (R15), the full RL stage with its own recipe, and the cross-fitted selector (R14).
  - The final system is the contender with the highest navdev generator-plus-selector score. Then every contender is scored once on navtest.
  - Compute: "we have excess in the week". The queue is reordered or cut when a result is pertinent.
- **A question for you, not blocking: should held-out training drives choose between the selector with and without boundary paths?**
  - **What happened:** in the rehearsal, the selector trained with R14's boundary paths did better than the main selector: navdev +2.08 against +1.35 over the own pick, and held-out training drives 91.20 against 91.01. Paired on the same scenes, the difference is +0.74 (-0.30 to +2.10): suggestive, not established. R14 deploys the main selector and reports the boundary version as an ablation.
  - **Why I won't switch on my own:** I registered that the rehearsal's numbers change no selector setting. Changing the deployed variant now would rest on them.
  - **The option:** for every contender, let its 10 held-out training drives choose between the two cross-fitted selectors (the rule never reads navdev), and report both. Your call; without it, R14 stands as registered.
- **An idea for you, not blocking: a pooled-candidate system.** Nothing changes without your yes.
  - **Why:** R16's final system is one path generator plus its own selector. But D has the best candidates of any method (navtest best-of-6 93.5, three seeds) and A the best own pick. One of D's samples stays in the drivable area in 80% of the scenes where D's own pick leaves it. A selector choosing among A's own pick plus D's 6 samples could beat both A plus its selector and D plus its selector.
  - **How:** a fifth kind of contender, registered before any phase-6 result and judged by the same navdev rule (R16). Its candidates are each contender's existing out-of-fold candidates, pooled per scene, so it adds about one selector chain's selectors (about 4 GPU hours) and one CPU scoring pass, plus a little code.
  - **Your call:** it changes what R16 calls the final system. Say yes, and I register it before phase 6 starts.
- **For your information (Nov 21): two calls made under your standing directions, each registered before any result it touches.** Say if you want either changed.
  - **The selector chain's rehearsal, ahead of the queue** (05:55, a note in R14): the whole chain at full scale on A1 without RL, about 4 GPU-hours, used for no choice. It tests the least-tested part of the final system about two days before the first contender's chain.
  - **R18** (07:44): phase 6's full RL stage continues in rounds of 5,000 updates while a round's last save is the best on navdev, up to 20,000 updates, for every contender alike. It follows "iterate RL past the dates"; each extra round costs a contender about 9 GPU hours.
- **Your three questions at 13:27:** answered in the session (last push, `docs/CHAT.md`, the report's private link at the top of this page).
- **For your information: R13, a second RL recipe round, added under your "improve RL round after round" clarification.** R12's recipes all lowered the navdev single path so far, and the sub-scores point to the advantage normalisation, the missing imitation term and D's step size. R13 tests the published remedies, on A and D, before phase 4. It took about 2.3 h of GPU time (7 runs of 17-20 min; I first wrote 3.5 h). Two of its recipes helped (RL4, RL6), so phase 4 started by rule.
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
- **HTML reports and replay pages:** on the home PC, in `navsim_demo_renders/` and `reports/`. Replay pages contain nuPlan camera images, so they stay off the public repository and off the web.
  - The path-generator report has no images, so it is also a private page: https://claude.ai/artifact/N8XRgjfSNwkzoyxfwVfxSQ.
