# Inbox for the autonomous session

Notes from the researcher to the Claude session that runs the experiments. It runs on one computer only, the home PC, so no work overlaps.

**How to leave a note** (from any computer):
1. Pull the branch `claude/stoic-darwin-ue0jm6`.
2. Add a note at the bottom of "Notes", headed with the date and time, for example `### 2026-11-21 14:30`.
3. Commit only this file and push to the same branch.

**What the autonomous session does:**
- At every heartbeat (about every 2 hours) and whenever a job finishes, it pulls and reads notes newer than the last one it acknowledged.
- It acts on each note and records what it did in `docs/STATUS.md` (section "Inbox").
- It never edits this file, so the two computers can't create merge conflicts.

**On the other computer:**
- Read `docs/STATUS.md` for the current state; it is pushed whenever something changes. Then `pilot_sft/README.md`, `pilot_sft/EXPERIMENTS.md`, `docs/GAPS.md` and `pilot_sft/MODELS.md`.
- Don't start or stop jobs on Anvil or the 4090 desktop. Leave a note here instead.
- Edit only this file, so pushes never conflict.

**What a note alone will not do.** These still need the researcher's direct confirmation: a second note that says so explicitly, or a reply in the home PC's terminal.
- anything outward-facing: emails, GitHub issues, publishing or sharing;
- deleting models or data;
- changing a standing rule: Anvil GPU jobs, the shared `x-cis251316` space, training-data changes, the local GPU;
- anything involving passwords or tokens.

For any of these, the question goes into `docs/STATUS.md` under "Waiting on the researcher".

## Notes

(none yet)
