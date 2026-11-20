#!/bin/bash
# Heads RL prerequisite (2026-11-20): wait for the navtrain metric-cache job, copy the cache to the 4090 desktop,
# point its metadata CSV at the desktop's paths, check that NAVSIM's own MetricCacheLoader finds every scene, then
# write the cache's .ready mark. The heads driver (pilot_sft/heads/driver.py) queues the RL runs once .ready exists.
#
# The Anvil original of the CSV is kept in metadata_anvil/, outside metadata/: NAVSIM's loader reads the first file
# in metadata/ whose name contains ".csv", so a backup left there could be picked instead.
#
# Runs on an Anvil login node. The desktop password comes from $DESK_PW (piped in), never from disk:
#   head -1 ~/.ssh/desk4090.pass | ssh anvil 'ssh login07 "read -r DESK_PW; export DESK_PW;
#     setsid nohup bash ~/xv/navtrain_to_desk.sh <job id> > /dev/null 2>&1 < /dev/null &"'
# Log: ~/xv/navtrain_to_desk.log. One shared desktop login (ControlMaster), closed 60 s after the last use; a failed
# copy waits 30 min before the next of 3 tries (the desktop is shared; never retry in a tight loop).
set -uo pipefail
JOB=${1:?job id of the navtrain metric-cache job}
LOG=$HOME/xv/navtrain_to_desk.log
X=exp/metric_cache_navtrain_v1
A=/anvil/scratch/x-vchia/navsim_workspace
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
[ -n "${DESK_PW:-}" ] || { log "no DESK_PW in the environment: stopping"; exit 1; }
log "watching job $JOB (pid $$)"
while :; do
  st=$(sacct -j "$JOB" -X -n -P -o State 2>/dev/null | head -1)
  case "$st" in
    COMPLETED) break ;;
    PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") sleep 300 ;;
    *) log "job $JOB ended $st: not copied (resubmit ~/xv/navtrain_cache.sh, which resumes, then restart this)"; exit 1 ;;
  esac
done
n=$(find "$A/$X" -name metric_cache.pkl | wc -l)
log "job $JOB COMPLETED: $n scene caches, $(du -sh "$A/$X" | cut -f1); copying to the desktop"
ok=0
for try in 1 2 3; do
  out=$(bash "$HOME/xv/to_desk_paths.sh" "$X" 2>&1); echo "$out" >> "$LOG"
  echo "$out" | grep -q "^copied" && { ok=1; break; }
  log "copy try $try failed; next try in 30 min"; sleep 1800
done
[ $ok = 1 ] || { log "copy failed 3 times: stopping"; exit 1; }
$SSHOPT "$DESK" "bash -s $n" >> "$LOG" 2>&1 <<'EOF'
set -euo pipefail
M=/home/vincent/navsim_workspace/exp/metric_cache_navtrain_v1
n=$(find $M -name metric_cache.pkl | wc -l)
[ "$n" = "$1" ] || { echo "desktop has $n scene caches, Anvil $1: not marked ready"; exit 1; }
mkdir -p $M/metadata_anvil
for f in $M/metadata/*.csv; do
  [ -f $M/metadata_anvil/$(basename $f) ] || cp -p $f $M/metadata_anvil/
  sed -i 's#/anvil/scratch/x-vchia/navsim_workspace/#/home/vincent/navsim_workspace/#g' $f
done
export NAVSIM_DEVKIT_ROOT=$HOME/navsim-v1.1 OPENSCENE_DATA_ROOT=$HOME/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=$HOME/navsim_workspace/dataset/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$HOME/navsim_workspace/exp
check() {
  ~/miniforge3/envs/navsim/bin/python - "$1" <<'P'
import sys
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
p = MetricCacheLoader(Path("/home/vincent/navsim_workspace/exp/metric_cache_navtrain_v1")).metric_cache_paths
missing = sum(not Path(v).exists() for v in p.values())
print(f"MetricCacheLoader: {len(p)} scenes, {missing} listed paths missing, {sys.argv[1]} files on disk")
sys.exit(0 if missing == 0 and len(p) == int(sys.argv[1]) else 1)
P
}
if ! check $n; then
  # the CSV does not list every cached scene (a resumed caching run, for example): rebuild it from the files
  for f in $M/metadata/*.csv; do mv $f $M/metadata_anvil/$(basename $f).rewritten; done
  { echo file_name; find $M -name metric_cache.pkl | sort; } > $M/metadata/metric_cache_navtrain_v1_metadata_node_0.csv
  echo "metadata CSV rebuilt from the $n files on disk"
  check $n
fi
touch $M/.ready
echo "navtrain cache ready on the desktop: $n scenes ($(du -sh $M | cut -f1))"
EOF
rc=$?
[ $rc = 0 ] && log "done: .ready written on the desktop; the heads driver queues the RL runs" \
            || log "desktop step failed (exit $rc): .ready not written"
exit $rc
