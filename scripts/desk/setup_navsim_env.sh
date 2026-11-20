#!/bin/bash
# Backup copy of setup_navsim_env.sh from the lab desktop's home (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# NAVSIM v1.1 scorer on the desktop, mirroring Anvil's env (Python 3.9.23, the same pinned packages,
# navsim at 3e8291b, nuplan-devkit at ce3c323), for head scoring and RL rewards (plan 2026-11-19).
set -e
cd ~
[ -d navsim-v1.1 ] || git clone -q https://github.com/autonomousvision/navsim.git navsim-v1.1
cd navsim-v1.1 && git checkout -q 3e8291bfa89ff247231e0227778840cd0a036896 && git log -1 --oneline && cd ~
[ -x ~/miniforge3/envs/navsim/bin/python ] || ~/miniforge3/bin/conda create -y -q -n navsim python=3.9.23
P=~/miniforge3/envs/navsim/bin/pip
taskset -c 16-31 nice -n 5 $P install -q -r ~/Auto-driving/scripts/desk/navsim_requirements.txt
taskset -c 16-31 nice -n 5 $P install -q --no-deps "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
taskset -c 16-31 nice -n 5 $P install -q --no-deps -e ~/navsim-v1.1
~/miniforge3/envs/navsim/bin/python -c "import navsim, nuplan, numpy, shapely; print('navsim env ok', numpy.__version__, shapely.__version__)"
$P list --format=freeze > ~/navsim_list_desk.txt
echo done
