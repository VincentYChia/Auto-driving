#!/bin/bash
# Backup copy of from_desk.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Pull a path from the 4090 desktop to Anvil: from_desk.sh <desktop path> <anvil path>. Needs $DESK_PW in the
# environment (piped in by the caller, never stored); ~/xv/askpass_env.sh hands it to ssh.
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
set -euo pipefail
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
# (2026-11-11) one password try; dead connections dropped after 90 s; every ssh and rsync here shares one logged-in
# connection, closed 60 s after the last use, so the desktop sees one login, not one per command
rsync -a -e "$SSHOPT" "$DESK:$1" "$2" 2> >(grep -v -E "xauth|X11" >&2)
