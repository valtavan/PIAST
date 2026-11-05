#!/bin/bash
#SBATCH --job-name=sof_lz
#SBATCH --array=0-9                 # 30 tasks x ~15 recordings; must equal N_TASKS below
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G                     # one 8 h recording x 4 channels is ~1 GB in float64
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT

# Submit from the project directory (the one containing pyproject.toml, uv.lock, lz_sof.py):
#   mkdir -p logs
#   JOB=$(sbatch --parsable run_lz.sbatch)
#   sbatch --dependency=afterok:${JOB} --job-name=sof_lz_merge --wrap \
#       "cd $PWD && uv run --frozen python lz_sof.py merge"
# Re-submitting is safe: finished subjects are skipped (use --overwrite to redo).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

N_TASKS=10
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=1             # keep scipy/BLAS from oversubscribing

uv run --frozen python lz_sof.py \
    --out-dir ~/pl1358-01/project_data/worksof/lz_results \
    run --task-id "${SLURM_ARRAY_TASK_ID}" --n-tasks "${N_TASKS}"
