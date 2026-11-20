# Status of the autonomous session

Written only by the Claude session on the home PC, and pushed whenever something changes. To ask for something, leave a note in `docs/INBOX.md`.

**Updated:** 2026-11-19 20:25 EST

Terms (path generator, configuration, training run, layer test, continuity series, single-path PDMS, best-of-6, RL stage, selector, decision rule) are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions".

**Goal to Mon 2026-11-30:** a controlled, same-backbone answer to "which diffusion head, and how continuous", measured on the full navtest before and after a matched RL stage. Then the best single-path score with head + RL + selector, against 85.6 today. Day-by-day plan: `pilot_sft/EXPERIMENTS.md`, "Compute plan to Monday 2026-11-30". Access from a new device: `docs/ACCESS.md`.

## Inbox
- Read up to: no notes yet.

## Running now
- **4090 desktop, GPU queue:**
  - Job 361, layer test part 1: the training run of configuration A1 reading stored layer 24 only. It trained all 20 epochs in 41 min, and every epoch's weights are saved. Its navtest prediction step failed on the one navtest scene with 420 tokens (the training scenes have at most 419); fixed in f08b281.
  - Job 362, layer test part 2: the training run of configuration A1 reading layers 12, 18 and 24, now training.
  - Job 363: navtest paths from job 361's path generator, next in the queue.
- **4090 desktop, CPU queue:** scoring jobs 371 and 372 wait for those runs' paths, then score them on NAVSIM (about an hour each).
- **Anvil:** wide job 22320518 builds the navtrain metric cache, the reward for the RL stage. Estimated start Nov 20 19:05.

## Latest results
- **Text pilot H1: 85.6 PDMS**, still the best.
  - S3's earlier passes scored 77.3, 81.9 and 84.5; the final pass 84.7.
  - H2 (8 passes) ties H1. Details in `pilot_sft/README.md`.
- **Path generator A1 (masked diffusion, 0.1 m bins, distance-aware bin embeddings), reading layer 24 only:** dev ADE about 0.77–0.82 m after epochs 8–11 (H1's text output: 0.77 m), dev best-of-6 ADE about 0.52 m. NAVSIM scores to come.
- **NAVSIM scoring now runs on the desktop.** H1 re-scored there matches Anvil on all 12,146 scenes: PDMS 0.856069 on both, with 53 ego-progress values off by at most 1.3e-11 from CPU rounding.

## Next (the week plan: `pilot_sft/EXPERIMENTS.md`, "Path-generator heads: the week of 2026-11-19")
1. Layer test: score both training runs, then fix the stored layers by the rule set in advance.
2. Method A's continuity series (A0, A2, A3, A4, A-0.25m, A-0.05m), then configurations D, B and C.
3. Second seeds.
4. A short matched RL stage.
5. The choice of generation method by the decision rule. With a clear winner, the selector and the full RL stage; on a tie, wait for the researcher.

## Waiting on the researcher
- Nothing. The 2026-11-19 evening questions are answered:
  - **Spare GPU time:** no filler runs, only what a comprehensive comparison needs. Configuration A0 (plain bins, no notion of distance) was added to A's continuity series for that reason. The ego-as-tokens run and the backbone snapshot runs are dropped.
  - **After the choice of method:** continue to the selector and full RL if the rule gives a clear winner; on a tie, stop and wait.
  - **The ~650 GB feature cache:** keep it all week, and ask before deleting it.
  - **The repo:** commit everything allowed, and keep it current and descriptive. Pages with nuPlan camera images stay on the home PC.

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
