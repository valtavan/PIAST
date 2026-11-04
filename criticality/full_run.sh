#!/bin/bash
#SBATCH --job-name=sof_mfdfa
#SBATCH --output=sof_mfdfa-%A_%a.log   # %A = job id, %a = array task id; written where you run sbatch
#SBATCH --array=0-11                   # 12 tasks, 0-based and contiguous (sof_mfdfa.py shards on this)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=02:30:00
# The UV_* variables are inherited from the shell you run sbatch in.
#
# Full run: 453 recordings over 12 tasks = 37 to 38 subjects per task.
# Runtime estimate from the pilot (detrend order 2): 0.0386 s per epoch, mean 1139 epochs
# per subject, so about 44 s per subject and about 28 min per task. 01:30:00 leaves
# a margin of about 3x for slower nodes and longer recordings.
#
# Safe to resubmit: subjects with an existing parquet are skipped, so a rerun only does
# subjects that were unfinished (e.g. after a timeout). Do NOT use --overwrite here.
# Do not change ORDER or SCALE_MIN_S for an existing OUT_DIR; use a new folder instead.
#
# Output per subject in OUT_DIR: {id}.parquet, {id}_curves.npz, logs/{id}.json.
# Check afterwards for subjects that did not finish:
#   grep -L '"outcome": "done"' "$OUT_DIR"/logs/*.json

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The MFDFA package has invalid escape sequences in its docstrings, which Python 3.14
# reports as SyntaxWarnings on first import. Harmless.
export PYTHONWARNINGS="ignore::SyntaxWarning"

CODE_DIR="$HOME/pl1358-01/project_data/code/criticality"
ORDER=2
SCALE_MIN_S=0.125
OUT_DIR="$HOME/pl1358-01/project_data/worksof/criticality/order${ORDER}_smin${SCALE_MIN_S}"

cd "$CODE_DIR" || exit 1

# Fail fast if an outdated sof_mfdfa.py is installed here (needs --scale-min-s and curve storage).
if ! grep -q 'SCRIPT_VERSION = "1.2"' sof_mfdfa.py; then
    echo "ERROR: $CODE_DIR/sof_mfdfa.py is not version 1.2; copy the latest script there." >&2
    exit 1
fi

# The exit status of python (1 if any subject failed unexpectedly) is the task's exit status.
uv run --frozen python -u sof_mfdfa.py \
    --out-dir "$OUT_DIR" \
    --detrend-order "$ORDER" \
    --scale-min-s "$SCALE_MIN_S"
