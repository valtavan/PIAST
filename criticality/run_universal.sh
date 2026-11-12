#!/bin/bash
#SBATCH --job-name=crit_run
#SBATCH --output=%x-%j.log      # %x = job-name, %j = job id (written in directory you run sbatch from)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
# The UV_* variables are inherited from the shell you run sbatch in.

# --- Configuration (edit these lines) ----------------------------------------
CODE_DIR="$HOME/pl1358-01/project_data/code/criticality"
SCRIPT="classify_dementia_multiq.py"

# Optional arguments for the script (leave empty to use the script's defaults)
ARGS=(

)
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$CODE_DIR" || exit 1

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: $CODE_DIR/$SCRIPT not found" >&2
    exit 1
fi

echo "=============== Running $SCRIPT ($(date)) ==============="
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID"
echo "Command: uv run --frozen python -u $SCRIPT ${ARGS[*]} $*"
echo "=========================================================="

uv run --frozen python -u "$SCRIPT" "${ARGS[@]}" "$@"
status=$?

echo "=============== Finished with exit code $status ($(date)) ==============="
exit $status
