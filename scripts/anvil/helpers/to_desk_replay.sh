#!/bin/bash
# Backup copy of to_desk_replay.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Copy a re-planning replay export (navsim_eval/replay_demo.py export) to the 4090 desktop: the front-camera frames
# and the model inputs with desktop paths (2026-11-12). Password from $DESK_PW (piped in by the launcher), never on disk.
# usage: to_desk_replay.sh <export name, e.g. v1>
set -euo pipefail
source ~/Auto-driving/scripts/anvil/common.sh
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
A=/anvil/scratch/x-vchia/navsim_workspace; R=/home/vincent/navsim_workspace
SRC=$A/exp/replay/$1; DST=$R/exp/replay/$1
sed "s#$A#$R#g" "$SRC/replay_inputs.json" > "$HOME/xv/replay_inputs_desk_$1.json"
$SSHOPT $DESK "mkdir -p $DST"
rsync -a -e "$SSHOPT" "$SRC/cam_f0" "$DESK:$DST/"
rsync -a -e "$SSHOPT" "$HOME/xv/replay_inputs_desk_$1.json" "$DESK:$DST/replay_inputs.json"
$SSHOPT $DESK "ls $DST/cam_f0 | wc -l; du -sh $DST"
$SSHOPT -O exit $DESK 2>/dev/null || true
