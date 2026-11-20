#!/bin/sh
# Backup copy of askpass_env.sh from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# prints the desktop password from the environment of the calling process (never stored on disk)
printf "%s\n" "$DESK_PW"
