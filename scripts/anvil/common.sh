# Sourced by every job script: loads config.local.sh if present, else config.sh.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_here/config.local.sh" ]; then
    # shellcheck disable=SC1091
    source "$_here/config.local.sh"
else
    # shellcheck disable=SC1091
    source "$_here/config.sh"
fi
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/train:${PYTHONPATH:-}"
mkdir -p "$NAVSIM_EXP_ROOT/logs"
