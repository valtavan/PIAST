#!/bin/bash
#SBATCH --job-name=high0.5_mfdfa
#SBATCH --output=high0.5_mfdfa_%A_%a.out
#SBATCH --error=high0.5_mfdfa_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

# Submit with:   sbatch run_high0.5.sh
# (from the folder that contains sof_mfdfa_high0.5.py; Slurm logs land there too)
#
# The array must be 0-based and contiguous: the python script shards the subject list
# with SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT. Change the number of tasks with e.g.
#   sbatch --array=0-19 run_high0.5.sh
#
# Optional overrides, e.g.:
#   PY_SCRIPT=/path/to/sof_mfdfa_high0.5.py sbatch --export=ALL run_high0.5.sh
#   EXTRA_ARGS="--subjects 6353 --overwrite" sbatch --export=ALL --array=0 run_high0.5.sh
#   EXTRA_ARGS="--detrend-order 1" sbatch --export=ALL run_high0.5.sh
#   EXTRA_ARGS="--highpass-hz 0.3" sbatch --export=ALL run_high0.5.sh
#
# Resubmitting is safe: subjects whose parquet already exists are skipped
# (add --overwrite via EXTRA_ARGS to recompute them).

set -euo pipefail

OUT="$HOME/pl1358-01/project_data/worksof/criticality/high0.5"

# Slurm runs a copy of this .sh from a spool dir, so don't rely on $0; default to the
# folder you submitted from, or set PY_SCRIPT.
PY_SCRIPT="${PY_SCRIPT:-${SLURM_SUBMIT_DIR:-$PWD}/sof_mfdfa_high0.5.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# uv project directory (contains pyproject.toml / uv.lock); `uv run` uses its env
PROJECT_DIR="$HOME/pl1358-01/project_data/code/criticality"

mkdir -p "$OUT/logs"

echo "Host:      $(hostname)"
echo "Started:   $(date)"
echo "Task:      ${SLURM_ARRAY_TASK_ID:-0} of ${SLURM_ARRAY_TASK_COUNT:-1} (job ${SLURM_ARRAY_JOB_ID:-none})"
echo "Script:    $PY_SCRIPT"
echo "Project:   $PROJECT_DIR"
echo "Output:    $OUT"

[ -f "$PY_SCRIPT" ] || { echo "ERROR: $PY_SCRIPT not found (set PY_SCRIPT=...)"; exit 1; }
[ -d "$PROJECT_DIR" ] || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

cd "$PROJECT_DIR"

# $EXTRA_ARGS is intentionally unquoted so it splits into separate arguments
uv run --frozen python "$PY_SCRIPT" \
    --out-dir "$OUT" \
    $EXTRA_ARGS

echo "Finished:  $(date)"
