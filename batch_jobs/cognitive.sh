#!/bin/bash
#SBATCH --job-name=cognitive
#SBATCH --output=cognitive-%A_%a.log   # %A = job id, %a = array task id
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
# No GPU needed: this only reads SAS files and writes one CSV.

# Before submitting, check in your login shell that this prints a path:
#   echo $UV_PROJECT_ENVIRONMENT
# The UV_* variables are inherited from the shell you run sbatch in.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $HOME/pl1358-01/project_data/code/helpers
uv run --frozen python -u cognitive_all.py
