# Status of the autonomous session

Written only by the Claude session on the home PC, and pushed whenever something changes. To ask for something, leave a note in `docs/INBOX.md`.

**Updated:** 2026-11-19 19:35 EST

## Inbox
- Read up to: no notes yet.

## Running now
- **4090 desktop, GPU queue:**
  - Job 361 is training head A rung 1 on the last layer only, about 2 min per pass, 20 passes; then its navtest paths.
  - Job 362 is next: the same head on layers 12, 18 and 24.
- **4090 desktop, CPU queue:** scoring jobs 371 and 372 wait for those runs' paths, then score them on NAVSIM (about an hour each).
- **Anvil:** wide job 22320518 builds the navtrain metric cache, the reward for the RL stage. Estimated start Nov 20 19:05.

## Latest results
- **Text pilot H1: 85.6 PDMS**, still the best.
  - S3's earlier passes scored 77.3, 81.9 and 84.5; the final pass 84.7.
  - H2 (8 passes) ties H1. Details in `pilot_sft/README.md`.
- **Head A, rung 1, last layer only:** dev ADE about 0.77–0.82 m after 8–11 passes (the text pilot: 0.77 m), best-of-6 about 0.52 m. NAVSIM scores to come.
- **NAVSIM scoring now runs on the desktop.** H1 re-scored there matches Anvil on all 12,146 scenes: PDMS 0.856069 on both, with 53 ego-progress values off by at most 1.3e-11 from CPU rounding.

## Next (the week plan: `pilot_sft/EXPERIMENTS.md`, "Path-generator heads: the week of 2026-11-19")
1. Layer probe: score both runs, then pick the layers by the rule set in advance.
2. Head A's continuity ladder, then D, B and C.
3. Second seeds.
4. A short matched RL stage.
5. Stop at the head choice and wait for the researcher.

## Waiting on the researcher
- See the questions of 2026-11-19 evening (asked in the home PC's terminal).

## Where things are
- **Code:**
  - `pilot_sft/heads/` (features, model, train, predict, summarize);
  - `scripts/desk/score_head_run.sh`.
- **Run outputs:**
  - Desktop: `~/navsim_workspace/exp/heads/<run>/`.
  - Feature cache: `~/navsim_workspace/head_cache/h1_l12-18-24/`.
- **HTML reports and replay pages:** on the home PC only, in `navsim_demo_renders/` and `reports/`. They contain nuPlan camera images, which stay out of this public repository.
