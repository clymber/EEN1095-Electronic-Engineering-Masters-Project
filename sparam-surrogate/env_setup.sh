# Runtime environment setup
# Usage: source env_setup.sh
#

export CONDA_ENV="${CONDA_ENV:-meng}"
export NOTEBOOK_KERNEL="${NOTEBOOK_KERNEL:-sparam-surrogate}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

if command -v conda >/dev/null 2>&1; then
    if [ -n "${BASH_VERSION:-}" ]; then
        __conda_shell="bash"
    elif [ -n "${ZSH_VERSION:-}" ]; then
        __conda_shell="zsh"
    else
        __conda_shell="posix"
    fi

    eval "$(conda "shell.${__conda_shell}" hook)"
    conda activate "${CONDA_ENV}"
    unset __conda_shell
else
    echo "conda command not found. Install Conda before sourcing env_setup.sh." >&2
    return 1 2>/dev/null || exit 1
fi
