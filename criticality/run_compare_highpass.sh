#!/bin/bash
#SBATCH --job-name=compare_hp
#SBATCH --output=compare_hp_%j.out
#SBATCH --error=compare_hp_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

# Scores the high-pass cutoffs in .../criticality/explore and writes .../explore/_comparison/
# Submit from the folder that holds compare_highpass_explore.py (after run_explore_highpass.sh has finished):
#     sbatch compare_explore_highpass.sh
# Optional (use with --export=ALL):
#     COG_FILE=/path/to/other.csv   (default: ~/pl1358-01/project_data/worksof/cognitive.csv, columns ID, V9TENG, V8AGE)
#     EXTRA_ARGS="--min-epochs 10"

set -euo pipefail

PROJECT_DIR="$HOME/pl1358-01/project_data/code/criticality"
PY_SCRIPT="${PY_SCRIPT:-${SLURM_SUBMIT_DIR:-$PWD}/compare_highpass_explore.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
COG_ARG=()
[ -n "${COG_FILE:-}" ] && COG_ARG=(--cog-file "$COG_FILE")

echo "Host:     $(hostname)"
echo "Started:  $(date)"
echo "Script:   $PY_SCRIPT"

[ -f "$PY_SCRIPT" ] || { echo "ERROR: $PY_SCRIPT not found (set PY_SCRIPT=...)"; exit 1; }
[ -d "$PROJECT_DIR" ] || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }
cd "$PROJECT_DIR"

uv run --frozen python "$PY_SCRIPT" ${COG_ARG[@]+"${COG_ARG[@]}"} $EXTRA_ARGS

echo "Finished: $(date)"
