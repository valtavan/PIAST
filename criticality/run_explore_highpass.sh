#!/bin/bash
#SBATCH --job-name=explore_hp
#SBATCH --output=explore_hp_%A_%a.out
#SBATCH --error=explore_hp_%A_%a.err
#SBATCH --array=0-34
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

# MFDFA for 7 high-pass cutoffs x 50 subjects x every 2nd epoch (half of the night).
# One array task = one (cutoff, subject shard): 7 cutoffs x 5 shards = 35 tasks, ~10 subjects each.
# Output: ~/pl1358-01/project_data/worksof/criticality/explore/high<cutoff>/<id>.parquet
#
# Submit from the folder that holds explore_highpass.py:
#     sbatch run_explore_highpass.sh
# then compare when everything is done:
#     jid=$(sbatch --parsable run_explore_highpass.sh)
#     sbatch --dependency=afterok:$jid compare_explore_highpass.sh
# Resubmitting is safe: subjects with an existing parquet are skipped.
# If you change CUTOFFS or N_SHARDS, set --array=0-<len(CUTOFFS)*N_SHARDS-1>.
#
# Optional: PY_SCRIPT=/full/path/explore_highpass.py  EXTRA_ARGS="--save-curves"  (use with --export=ALL)

set -euo pipefail

CUTOFFS=(0.1 0.25 0.5 0.75 1 1.5 2)
N_SHARDS=5

PROJECT_DIR="$HOME/pl1358-01/project_data/code/criticality"   # uv project (pyproject.toml / uv.lock)
PY_SCRIPT="${PY_SCRIPT:-${SLURM_SUBMIT_DIR:-$PWD}/explore_highpass.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASK="${SLURM_ARRAY_TASK_ID:?run this with sbatch (array job)}"
if [ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne $(( ${#CUTOFFS[@]} * N_SHARDS )) ]; then
    echo "ERROR: array has ${SLURM_ARRAY_TASK_COUNT:-?} tasks, expected $(( ${#CUTOFFS[@]} * N_SHARDS )) (cutoffs x shards)"
    exit 1
fi
HP="${CUTOFFS[$(( TASK / N_SHARDS ))]}"
SHARD=$(( TASK % N_SHARDS ))

echo "Host:     $(hostname)"
echo "Started:  $(date)"
echo "Task:     $TASK  -> high-pass ${HP} Hz, shard ${SHARD}/${N_SHARDS}"
echo "Script:   $PY_SCRIPT"
echo "Project:  $PROJECT_DIR"

[ -f "$PY_SCRIPT" ] || { echo "ERROR: $PY_SCRIPT not found (set PY_SCRIPT=...)"; exit 1; }
[ -d "$PROJECT_DIR" ] || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }
cd "$PROJECT_DIR"

uv run --frozen python "$PY_SCRIPT" \
    --high-pass "$HP" \
    --shard "$SHARD" --n-shards "$N_SHARDS" \
    --n-subjects 50 --seed 0 --epoch-step 2 \
    $EXTRA_ARGS

echo "Finished: $(date)"
