#!/bin/bash
# Backup copy of to_desk.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Copy what the 4090 desktop needs from Anvil scratch (2026-11-09). The desktop password comes from $DESK_PW
# in the environment (piped in by the launcher); it is never written to disk.
set -uo pipefail
source ~/Auto-driving/scripts/anvil/common.sh
export SSH_ASKPASS=$HOME/xv/askpass_env.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x
DESK=$(head -1 "$HOME/.ssh/desk4090.host")  # user@host of the lab desktop, kept out of the public repo
SSHOPT="ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-desk-%C -o ControlPersist=60"
# (2026-11-11) one password try; dead connections dropped after 90 s; every ssh and rsync here shares one logged-in
# connection, closed 60 s after the last use, so the desktop sees one login, not one per command
R=/home/vincent/navsim_workspace
step() { echo "== $1 $(date +%T)"; }
step mkdir
$SSHOPT $DESK "mkdir -p $R/training_data/pilot_v1 $R/exp/pilot_sft $R/exp/mvlad $R/dataset"
step run-large-ep3
rsync -a -e "$SSHOPT" --exclude navtest_predictions "$NAVSIM_EXP_ROOT/pilot_sft/qwen35-0.8b-large-ep3" "$DESK:$R/exp/pilot_sft/"
step navtest-images
python3 - > "$HOME/xv/navtest_imgs.txt" <<'PY'
import json
d = json.load(open('/anvil/scratch/x-vchia/navsim_workspace/exp/mvlad/navtest_inputs.json'))
pre = '/anvil/scratch/x-vchia/navsim_workspace/dataset/'
for r in d:
    assert r['image'].startswith(pre), r['image']
    print(r['image'][len(pre):])
PY
wc -l "$HOME/xv/navtest_imgs.txt"
rsync -a -e "$SSHOPT" --files-from="$HOME/xv/navtest_imgs.txt" "$OPENSCENE_DATA_ROOT/" "$DESK:$R/dataset/"
sed "s#/anvil/scratch/x-vchia/navsim_workspace#$R#g" "$NAVSIM_EXP_ROOT/mvlad/navtest_inputs.json" > "$HOME/xv/navtest_inputs_desk.json"
rsync -a -e "$SSHOPT" "$HOME/xv/navtest_inputs_desk.json" "$DESK:$R/exp/mvlad/navtest_inputs.json"
for v in v1-xl-nd v1-large v1-pilot; do
  step "view-$v"
  rsync -a -e "$SSHOPT" "$TRAINING_DATA_ROOT/pilot_v1/$v/" "$DESK:$R/training_data/pilot_v1/$v/"
done
step done
