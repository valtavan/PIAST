#!/bin/bash
#SBATCH --job-name=noenv_test
#SBATCH --output=noenv_test-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --partition=tesla
#SBATCH --gpus-per-node=1

# NOTE: this script deliberately does NOT set the UV_* variables.

echo "Node: $(hostname)"
echo "HOME=[$HOME]  USER=[$USER]"
echo "UV_PROJECT_ENVIRONMENT=[$UV_PROJECT_ENVIRONMENT]"
echo "UV_CACHE_DIR=[$UV_CACHE_DIR]"
echo "UV_PYTHON_INSTALL_DIR=[$UV_PYTHON_INSTALL_DIR]"
echo "uv found at: $(command -v uv || echo 'NOT FOUND')"

# Safety stop: without these, uv would build a new .venv and fill the home quota.
if [ -z "$UV_PROJECT_ENVIRONMENT" ] || [ -z "$UV_CACHE_DIR" ] || ! command -v uv > /dev/null; then
    echo "RESULT: variables or uv missing in the job, so they must be set in the script. Stopping."
    exit 1
fi

cd "$HOME/pl1358-01/project_data/code" || exit 1
uv run --frozen python -c "import torch; print('torch', torch.__version__, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
echo "RESULT: variables were inherited and torch ran."
