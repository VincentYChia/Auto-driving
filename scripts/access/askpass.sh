#!/bin/sh
# SSH_ASKPASS helper for the lab desktop: prints the password from the private file ~/.ssh/desk4090.pass
# (one line, chmod 600). The password is never stored anywhere else and never committed.
head -1 "$HOME/.ssh/desk4090.pass"
