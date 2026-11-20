#!/bin/bash
# Backup copy of to_desk_paths.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Copy folders under Anvil's navsim_workspace to the same place under the desktop's (2026-11-19: NAVSIM scorer
# data for head scoring and RL). Args: paths relative to /anvil/scratch/x-vchia/navsim_workspace.
# Password from $DESK_PW (piped in), never on disk; one shared login (ControlMaster), closed 60 s after use.
set -uo pipefail
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
A=/anvil/scratch/x-vchia/navsim_workspace
R=/home/vincent/navsim_workspace
for x in "$@"; do
  x=${x%/}
  $SSHOPT $DESK "mkdir -p $R/$(dirname $x)" || exit 1
  t0=$(date +%s)
  rsync -a -e "$SSHOPT" "$A/$x" "$DESK:$R/$(dirname $x)/" && echo "copied $x ($(du -sh $A/$x | cut -f1), $(( $(date +%s) - t0 )) s)" || echo "FAILED $x"
done
