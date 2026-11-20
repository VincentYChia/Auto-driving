#!/bin/bash
# Backup copy of to_desk_adapters.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Copy finished runs' adapter/ and run.json (+ eval.json) from Anvil to the 4090 desktop: bash to_desk_adapters.sh <run> ...
# The desktop password comes from $DESK_PW (piped in by the launcher), never written to disk.
set -uo pipefail
source ~/Auto-driving/scripts/anvil/common.sh
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
# (2026-11-11) one password try; dead connections dropped after 90 s; every ssh and rsync here shares one logged-in
# connection, closed 60 s after the last use, so the desktop sees one login, not one per command
R=/home/vincent/navsim_workspace/exp/pilot_sft
for r in "$@"; do
  src=$NAVSIM_EXP_ROOT/pilot_sft/./$r
  files=("$src/adapter" "$src/run.json"); [ -f "$src/eval.json" ] && files+=("$src/eval.json")
  rsync -a --relative -e "$SSHOPT" "${files[@]}" "$DESK:$R/" && echo "copied $r"
done
