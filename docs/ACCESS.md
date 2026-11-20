# Getting to the compute from a new device

How to reach Purdue Anvil and the lab RTX 4090 desktop from any computer with this repository. Written 2026-11-19 as a backup. Secrets are never in this public repository:
- the desktop password;
- the Anvil private key;
- the desktop's address (host name and IP);
- any API token.

The steps below say where each one comes from.

Only one computer runs the autonomous Claude session (the home PC). From any other computer, read `docs/STATUS.md` and leave requests in `docs/INBOX.md` (see that file). Don't start jobs from two places.

## 1. Anvil (Purdue RCAC)

- **Host and user:** `anvil.rcac.purdue.edu`, user `x-vchia`. Login is by SSH key; `anvil` round-robins the login nodes login00–07.
- **A new device needs its own key:**
  1. `ssh-keygen -t ed25519 -f ~/.ssh/anvil_ed25519` on the new device.
  2. Append the `.pub` line to `~/.ssh/authorized_keys` on Anvil. Use any existing login (the home PC: `ssh anvil 'cat >> ~/.ssh/authorized_keys' < new_key.pub`), or Anvil's Open OnDemand web portal (`https://ondemand.anvil.rcac.purdue.edu`, ACCESS login, then Clusters > Anvil Shell Access).
  3. Add to `~/.ssh/config`:
     ```
     Host anvil
         HostName anvil.rcac.purdue.edu
         User x-vchia
         IdentityFile ~/.ssh/anvil_ed25519
     ```
  4. Test with `ssh anvil hostname`. From inside Anvil, `ssh login07` reaches a specific login node; the autopilot runs there.
- **Allocations:**
  - CPU jobs use `-A cis220051`: NAVSIM scoring, metric caches, data builds.
  - The GPU allocation `cis251316-gpu` exists, but **no new Anvil GPU jobs** (researcher's rule; the queue waits median 42 h). Training runs on the 4090 desktop.
  - **Never use the shared project space `x-cis251316`.**
- **Partitions for CPU jobs:** `shared` (often stalls), `wide` and `wholenode` (whole 128-core nodes; `wide` has started score jobs within minutes), `debug` (short), `highmem`. Check with `sbatch --test-only ...` before submitting.
- **Where things are on Anvil:**
  - repo clone `~/Auto-driving`;
  - data and runs `/anvil/scratch/x-vchia/navsim_workspace/` (`dataset/`, `exp/`, `training_data/`);
  - the NAVSIM v1.1 devkit `~/navsim-v1.1` (commit 3e8291b) with its env `~/.conda/envs/navsim`;
  - helper scripts `~/xv/`, backed up in `scripts/anvil/helpers/`.
  - `scripts/anvil/common.sh` sets the environment (`NAVSIM_EXP_ROOT` and the rest).
- **Scratch is purged** of files left unused. Check the oldest data's access times by about 2026-12-05. The desktop holds a second copy of every model (`pilot_sft/MODELS.md`).

## 2. The lab RTX 4090 desktop

- **What it is:** shared lab hardware (Ubuntu 22.04, RTX 4090 24 GB, i9-14900KF, 1.9 TB NVMe). Other lab members use it too.
- **Reaching it:** it is reachable only inside Purdue's network, so go through Anvil: `ssh -J anvil <user>@<desktop host>`.
- **The address is not in this repository.** It is the one line of `~/.ssh/desk4090.host` on Anvil (`ssh anvil cat ~/.ssh/desk4090.host`), and it also appears as `DESK=` in Anvil's `~/xv/to_desk.sh`. Copy it to `~/.ssh/desk4090.host` on the new device.
- **Password only:** key login was not set up. The researcher knows the password. Put it as the only line of `~/.ssh/desk4090.pass` (`chmod 600`) on the new device, and never commit it or copy it anywhere else.
- **Helper:** `source scripts/access/desk.sh`, then `D 'bash -s' <<'X' ... X` runs one command there. It reads the two private files above, through `scripts/access/askpass.sh`. On Windows, use Git Bash.
- **Connection rules:** one login at a time, one password try (`NumberOfPasswordPrompts=1`), and never retry loops or parallel calls. Its sshd locks everyone out when half-open logins pile up ("Exceeded MaxStartups"). If it doesn't answer, wait about 30 min. Anvil's autopilot log `~/xv/overnight.log` records "desktop reachable again".
- **Jobs:** pin every job to the E-cores, `taskset -c 16-31`; the P-cores crash this CPU.
  - Two queues run job scripts in name order, one at a time: `~/queue` for GPU jobs (one GPU job at a time) and `~/queue2` for CPU-only NAVSIM scoring.
  - The runner is backed up as `scripts/desk/queue_runner.sh`.
  - Add a job by writing `~/queue/pending/<NNN>_<name>.sh`. Logs are in `~/queue/logs/`, the history in `~/queue/history.log`.
  - Restart a runner with `setsid -f nohup bash ~/queue/runner.sh >> ~/queue/runner.out 2>&1 < /dev/null`.
- **Environments:**
  - `~/miniforge3/envs/pilot`: training and prediction (Python 3.11, torch 2.14, transformers 5.22; HF cache `~/hf_cache`).
  - `~/miniforge3/envs/navsim`: NAVSIM scoring, a mirror of Anvil's, built by `scripts/desk/setup_navsim_env.sh` from `scripts/desk/navsim_requirements.txt`.
  - `scripts/anvil/config.local.sh` (git-ignored) points `DATA_HOME` at `~/navsim_workspace` and sets `CC=/usr/bin/gcc-11`.
- **Data:** `~/navsim_workspace/`:
  - `training_data/pilot_v1/<view>`;
  - `exp/pilot_sft/<run>` (pilot runs);
  - `exp/heads/<run>` (path generators);
  - `head_cache/` (stored backbone features, ~650 GB);
  - `exp/metric_cache_navtest_v1` (its metadata CSV paths rewritten for this machine).
- **NAVSIM on the desktop:** `scripts/desk/score_head_run.sh`. A copy of H1's score there matches Anvil's (PDMS 0.856069 on both).

## 3. Copying between Anvil and the desktop

Run these on an Anvil login node, with the password piped in so it never touches disk:

```
head -1 ~/.ssh/desk4090.pass | ssh anvil 'read -r DESK_PW; export DESK_PW; bash ~/xv/to_desk_paths.sh exp/<folder>'
```

- `~/xv/to_desk_paths.sh <paths under navsim_workspace>`: Anvil to the desktop.
- `~/xv/to_desk_models.sh <run>/<adapter|epoch_k>`: adapters.
- `~/xv/from_desk.sh <desktop path> <anvil path>`: the desktop to Anvil.

All share one login (ControlMaster) that closes 60 s after use.

## 4. What else a new device needs

- **Git:** `git clone https://github.com/VincentYChia/Auto-driving.git`, then check out the branch `claude/stoic-darwin-ue0jm6`.
- **Local data** (only for local data builds): nuPlan / OpenScene archives are downloaded by the scripts in `scripts/`, and the camera images stay out of this repository. Results reports and replay pages with camera images live only on the home PC (`navsim_demo_renders/`).
- **Rules that travel with the repo:** `CLAUDE.md`, `docs/RESEARCH_DIRECTION.md` (decision log), `pilot_sft/EXPERIMENTS.md` (current plan), `docs/GAPS.md`, `pilot_sft/MODELS.md`.
