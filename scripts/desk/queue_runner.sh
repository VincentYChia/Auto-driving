#!/bin/bash
# Backup copy of runner.sh from the lab desktop's home (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Sequential GPU job queue for this shared desktop: runs ~/queue/pending/*.sh in name order, one at a time.
# Before each job it waits while another user's process holds more than 1 GB of GPU memory.
# Add a job: write ~/queue/pending/<NN>_<name>.sh. History: ~/queue/history.log; output: ~/queue/logs/<name>.log
Q="$HOME/queue"
exec 9>"$Q/runner.lock"
flock -n 9 || { echo "runner already running"; exit 0; }
others_on_gpu() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    while IFS=', ' read -r pid mem; do
        owner=$(ps -o user= -p "$pid" 2>/dev/null)
        [ -n "$owner" ] && [ "$owner" != "$USER" ] && [ "${mem:-0}" -gt 1024 ] && echo "$owner:$pid:${mem}MiB"
    done
}
while true; do
    next=$(ls "$Q"/pending/*.sh 2>/dev/null | sort | head -1)
    if [ -z "$next" ]; then sleep 60; continue; fi
    busy=$(others_on_gpu)
    if [ -n "$busy" ]; then echo "$(date '+%F %T') waiting: GPU in use by $busy" >> "$Q/history.log"; sleep 300; continue; fi
    name=$(basename "$next" .sh)
    mv "$next" "$Q/running/$name.sh"
    echo "$(date '+%F %T') start $name" >> "$Q/history.log"
    bash "$Q/running/$name.sh" > "$Q/logs/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then mv "$Q/running/$name.sh" "$Q/done/"; else mv "$Q/running/$name.sh" "$Q/failed/"; fi
    echo "$(date '+%F %T') end $name rc=$rc" >> "$Q/history.log"
done
