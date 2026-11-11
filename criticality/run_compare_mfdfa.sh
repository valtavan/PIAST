#!/bin/bash
#SBATCH --job-name=compare_mfdfa
#SBATCH --output=compare_mfdfa_%j.out
#SBATCH --error=compare_mfdfa_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

# Submit with:   sbatch run_compare_mfdfa.sh
# (from any folder; Slurm logs land in the folder you submit from)
#
# Optional overrides, e.g.:
#   REFERENCE=noenvelope sbatch --export=ALL run_compare_mfdfa.sh
#   EXTRA_ARGS="--collapse" sbatch --export=ALL run_compare_mfdfa.sh

set -euo pipefail

ROOT="$HOME/pl1358-01/project_data/worksof/criticality"
OUT="$ROOT/_comparison"

# Path to the python script. Slurm runs a copy of this .sh from a spool dir, so
# don't rely on $0; default to the folder you submitted from, or set PY_SCRIPT.
PY_SCRIPT="${PY_SCRIPT:-${SLURM_SUBMIT_DIR:-$PWD}/compare_mfdfa.py}"
REFERENCE="${REFERENCE:-original}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# uv project directory (contains pyproject.toml / uv.lock); `uv run` uses its env
PROJECT_DIR="$HOME/pl1358-01/project_data/code/criticality"

mkdir -p "$OUT"

echo "Host:      $(hostname)"
echo "Started:   $(date)"
echo "Script:    $PY_SCRIPT"
echo "Project:   $PROJECT_DIR"
echo "Root:      $ROOT"
echo "Output:    $OUT"

[ -f "$PY_SCRIPT" ] || { echo "ERROR: $PY_SCRIPT not found (set PY_SCRIPT=...)"; exit 1; }
[ -d "$PROJECT_DIR" ] || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

cd "$PROJECT_DIR"

uv run --frozen python "$PY_SCRIPT" \
    --root "$ROOT" \
    --out "$OUT" \
    --reference "$REFERENCE" \
    $EXTRA_ARGS

echo "Finished:  $(date)"
