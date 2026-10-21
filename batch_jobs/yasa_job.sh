#!/bin/bash
#SBATCH --job-name=yasa
#SBATCH --output=yasa-%A_%a.log      # %A = job id, %a = array task id
#SBATCH --array=0-9                  # 10 tasks; each processes every 10th file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
# No GPU and no tesla partition: YASA runs on CPU.

# Before submitting, check in your login shell that this prints a path:
#   echo $UV_PROJECT_ENVIRONMENT
# The UV_* variables are inherited from the shell you run sbatch in.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $HOME/pl1358-01/project_data/code/staging
uv run --frozen python -u yasa_edf_raw.py
