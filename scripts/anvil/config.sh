# Shared settings for the Anvil job scripts. Copy to config.local.sh, edit, and the jobs
# will source config.local.sh instead of this file.
#
#   cp scripts/anvil/config.sh scripts/anvil/config.local.sh && $EDITOR scripts/anvil/config.local.sh

# --- Slurm ---------------------------------------------------------------------------
# sbatch reads the account from the *submitting* shell, so set it there (e.g. in ~/.bashrc):
#   export SBATCH_ACCOUNT=<your GPU allocation account>   # see `mybalance`
# and submit every job from the repository root.

# --- Code ----------------------------------------------------------------------------
export REPO_ROOT="${REPO_ROOT:-$HOME/Auto-driving}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$HOME/navsim}"     # git clone of autonomousvision/navsim
export NAVSIM_CONDA_ENV="${NAVSIM_CONDA_ENV:-navsim}"              # NAVSIM's env (Python 3.9)
export MVLAD_CONDA_ENV="${MVLAD_CONDA_ENV:-mvlad}"                 # this repo's model env

# --- Data (keep large data in project space if you have it; scratch is purged after 30 days unused)
export DATA_HOME="${DATA_HOME:-$SCRATCH/navsim_workspace}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_HOME/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_HOME/dataset/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DATA_HOME/exp}"
export TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-$DATA_HOME/training_data}"

# --- Hugging Face cache (LLaDA-V / SigLIP2 / teacher models) --------------------------
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"

activate_env() {
    # Anvil provides conda through a module; adjust if your setup differs.
    module load anaconda 2>/dev/null || module load conda 2>/dev/null || true
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$1"
}
