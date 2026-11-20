#!/bin/bash
# Backup copy of to_desk_models.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Second copy of every trained adapter (user 2026-11-19: "make sure we save all the models"): copy run folders'
# adapter/ and epoch_k/ (plus run.json) that exist only on Anvil scratch to the desktop, same layout.
# Args: <run>/<adapter|epoch_k> ... Password from $DESK_PW (piped in), never on disk.
set -uo pipefail
source ~/Auto-driving/scripts/anvil/common.sh
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
R=/home/vincent/navsim_workspace/exp/pilot_sft
A=$NAVSIM_EXP_ROOT/pilot_sft
for x in "$@"; do
  run=${x%%/*}; sub=${x#*/}
  $SSHOPT $DESK "mkdir -p $R/$run" || exit 1
  [ -f "$A/$run/run.json" ] && rsync -a -e "$SSHOPT" --ignore-existing "$A/$run/run.json" "$DESK:$R/$run/"
  rsync -a -e "$SSHOPT" "$A/$run/$sub" "$DESK:$R/$run/" && echo "copied $x ($(du -sh "$A/$run/$sub" | cut -f1))"
done
