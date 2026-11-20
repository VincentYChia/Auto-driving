#!/bin/bash
# Backup copy of recover_desk.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Desktop recovery after a hang (2026-11-14): waits until the autopilot (overnight.sh) logs "desktop reachable again"
# after this script started, then, in ONE login: if queue 1's runner is not running, moves the job that was running
# back to pending/ (training resumes from its checkpoint.pt), notes it in history.log and starts the runner.
# Makes no login of its own until the autopilot has seen the desktop answer. Password from $DESK_PW (piped in by the
# launcher), never on disk. Stops by STOP_AT (epoch seconds; default Mon Nov 16 03:30 EST, before the Anvil outage).
set -uo pipefail
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
LOG=$HOME/xv/recover_desk.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
start_line=$(wc -l < $HOME/xv/overnight.log)
stop_at=${STOP_AT:-$(date -d '2026-11-16 03:30' +%s)}
log "watching overnight.log from line $start_line for 'desktop reachable again'"
while [ "$(date +%s)" -lt "$stop_at" ]; do
  if tail -n +"$((start_line + 1))" $HOME/xv/overnight.log | grep -q "desktop reachable again"; then
    out=$($SSHOPT $DESK 'bash -s' <<'EOF' 2>&1
uptime
if pgrep -f "queue/runner.sh" >/dev/null; then echo "runner already up"; exit 0; fi
moved=$(ls ~/queue/running/ 2>/dev/null | tr '\n' ' ')
[ -n "$moved" ] && mv ~/queue/running/* ~/queue/pending/
echo "$(date '+%F %T') desktop came back after a hang (~13:00-15:40 Nov 14): moved back to pending: ${moved:-nothing}; runner restarted by ~/xv/recover_desk.sh on Anvil" >> ~/queue/history.log
setsid -f nohup bash ~/queue/runner.sh >> ~/queue/runner.out 2>&1 < /dev/null
sleep 5; pgrep -f "queue/runner.sh" >/dev/null && echo "runner restarted; moved: ${moved:-nothing}" || echo "RUNNER DID NOT START"
EOF
)
    log "recovery: $(echo "$out" | tr '\n' ' ')"
    case "$out" in *"runner already up"*|*"runner restarted"*) exit 0 ;; esac
    tries=$(( ${tries:-0} + 1 )); [ "$tries" -ge 3 ] && { log "gave up after 3 tries"; exit 1; }
    sleep 1800; continue  # one gentle retry every 30 min, never a tight loop
  fi
  sleep 300
done
log "stopped at the deadline without seeing the desktop come back"
