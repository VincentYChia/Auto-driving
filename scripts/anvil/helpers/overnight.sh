#!/bin/bash
# Backup copy of overnight.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Overnight autopilot (from 2026-11-09 23:20) while the local Claude session is off. Every 10 min, if one login
# to the desktop works (else it waits 30 min):
#  1. copy finished pilot_v1.2 views (v1-large, v1-nt, v1-nt-mix) to the 4090 desktop as <set>-p12, check sizes, mark .copied
#  2. resubmit the v1.0-pool build (and its views-v12 job) if it stopped without printing "done"
#  3. cancel the H100 duplicate of xlnd-ep4s (22104010 + 22104011) once the desktop run has its epoch_1 snapshot
#  4. pull finished desktop runs (navtest predictions >= 12,146) and submit their NAVSIM scores on shared
# Needs DESK_PW in the environment (piped in by the launcher; never on disk). Log: ~/xv/overnight.log; state: ~/xv/overnight_state/
set -u
source ~/Auto-driving/scripts/anvil/common.sh
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
# (2026-11-11) one password try; dead connections dropped after 90 s; every ssh and rsync here shares one logged-in
# connection, closed 60 s after the last use, so the desktop sees one login, not one per command
R=/home/vincent/navsim_workspace
ST=$HOME/xv/overnight_state; mkdir -p "$ST"
log() { echo "$(date '+%F %T') $*"; }
dsh() { $SSHOPT $DESK "$@" 2>/dev/null; }
sizes() { stat -c '%n %s' images.zip train.jsonl dev.jsonl view.json 2>/dev/null | sort | md5sum; }
end=$(( $(date +%s) + ${HOURS:-40} * 3600 ))
log "start"
down=0
while [ "$(date +%s)" -lt "$end" ]; do
  # one gentle check first (2026-11-11: the desktop's sshd refused logins with "Exceeded MaxStartups"); while it is
  # down, read its SSH banner once for the log and wait 30 min, instead of one login per run every 10 min
  if ! dsh true; then
    banner=$(timeout 20 bash -c "exec 3<>/dev/tcp/${DESK#*@}/22 && head -c 80 <&3" 2>/dev/null | tr -dc '[:print:]')
    log "desktop unreachable (banner: ${banner:-none}); next try in 30 min"; down=1; sleep 1800; continue
  fi
  if [ "$down" = 1 ]; then log "desktop reachable again"; down=0; fi
  for spec in $(cat $HOME/xv/autopilot_views.txt 2>/dev/null); do
    b=${spec%%:*}; s=${spec#*:}; src=$TRAINING_DATA_ROOT/$b/views/pilot_v1.2/$s
    [ -f "$ST/view-$s" ] || [ ! -f "$src/view.json" ] && continue
    log "copy view $s-p12"
    rsync -a -e "$SSHOPT" "$src/" "$DESK:$R/training_data/pilot_v1/$s-p12/" 2>/dev/null
    a=$(cd "$src" && sizes); r=$(dsh "cd $R/training_data/pilot_v1/$s-p12 && stat -c '%n %s' images.zip train.jsonl dev.jsonl view.json | sort | md5sum")
    if [ -n "$r" ] && [ "$a" = "$r" ]; then dsh "touch $R/training_data/pilot_v1/$s-p12/.copied"; touch "$ST/view-$s"; log "view $s-p12 copied and marked"
    else log "view $s-p12: sizes differ, retry next round"; fi
  done
  if [ ! -f "$ST/pool-done" ]; then
    pj=$(cat "$ST/pool-job" 2>/dev/null || echo 22121755); vj=$(cat "$ST/views-job" 2>/dev/null || echo 22121846)
    s=$(sacct -X -n -j "$pj" -o State%20 | head -1 | tr -d ' ')
    if [ -n "$s" ] && [ "$s" != PENDING ] && [ "$s" != RUNNING ]; then
      if grep -q "^done:" ~/Auto-driving/v1-pool-"$pj".out 2>/dev/null; then touch "$ST/pool-done"; log "pool build $pj done"
      else
        scancel "$vj" 2>/dev/null
        cd ~/Auto-driving
        nj=$(sbatch --parsable -A cis220051 -p shared --gpus-per-node=0 scripts/anvil/data_pipeline/build_v1_pool.sbatch)
        nv=$(sbatch --parsable -A cis220051 -p shared --gpus-per-node=0 --dependency=afterok:"$nj" scripts/anvil/data_pipeline/build_views_v12.sbatch)
        cd - >/dev/null; echo "$nj" > "$ST/pool-job"; echo "$nv" > "$ST/views-job"
        log "pool build $pj ended $s without done: resubmitted $nj, views-v12 $nv"
      fi
    fi
  fi
  if [ ! -f "$ST/h100-cancelled" ] && dsh "test -f $R/exp/pilot_sft/qwen35-0.8b-xlnd-ep4s/epoch_1/adapter_config.json"; then
    scancel 22104010 22104011; touch "$ST/h100-cancelled"; log "desktop xlnd-ep4s past epoch 1: cancelled H100 duplicate 22104010 + 22104011"
  fi
  for run in $(cat $HOME/xv/autopilot_runs.txt 2>/dev/null); do
    [ -f "$ST/scored-$run" ] && continue
    n=$(dsh "cat $R/exp/pilot_sft/$run/navtest_predictions/shard_*.jsonl 2>/dev/null | wc -l")
    [ "${n:-0}" -ge 12146 ] || continue
    dst=$NAVSIM_EXP_ROOT/pilot_sft/$run
    rsync -a -e "$SSHOPT" --exclude checkpoint.pt --exclude images "$DESK:$R/exp/pilot_sft/$run/" "$dst/" 2>/dev/null
    m=$(cat "$dst"/navtest_predictions/shard_*.jsonl 2>/dev/null | wc -l)
    if [ "$m" -ge 12146 ]; then
      cd ~/Auto-driving
      j=$(STEP=pilot NAME=$run PREDICTIONS=$dst/navtest_predictions sbatch --parsable -A cis220051 -p shared --gpus-per-node=0 --cpus-per-task=64 --time=00:30:00 -J "score-${run#qwen35-0.8b-}" scripts/anvil/navsim_eval/navsim_score.sbatch)
      cd - >/dev/null; touch "$ST/scored-$run"; log "$run: $m predictions pulled, score job $j"
    else log "$run: only $m predictions after copy"; fi
  done
  sleep 600
done
log "end"
