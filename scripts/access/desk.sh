# Source this file, then run one command on the lab RTX 4090 desktop through Anvil:
#     source scripts/access/desk.sh
#     D 'bash -s' <<'X'
#     tail -3 ~/queue/history.log
#     X
# Needs (see docs/ACCESS.md): a "Host anvil" entry in ~/.ssh/config with key login, ~/.ssh/desk4090.host (one line,
# user@host of the desktop) and ~/.ssh/desk4090.pass (one line, the desktop password, chmod 600). Neither file is
# ever committed. One login per call, one password try, dead connections dropped after 90 s: the desktop is shared
# and its sshd locks everyone out when half-open logins pile up, so never loop or run calls in parallel.
_DESK_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D() {
  SSH_ASKPASS="$_DESK_HERE/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=x ssh -J anvil \
    -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1 \
    -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    "$(head -1 "$HOME/.ssh/desk4090.host")" "$@"
}
