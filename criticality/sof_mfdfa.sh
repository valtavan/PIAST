#!/bin/bash
#SBATCH --job-name=sof_mfdfa
#SBATCH --output=sof_mfdfa-%x-%A_%a.log   # %A = job id, %a = array task id; written where you run sbatch
#SBATCH --array=0-11                   # 12 tasks, 0-based and contiguous (the script shards on this)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=03:00:00                # guess: adjust after checking "seconds" in logs/*.json of a first run
#
# Two versions of the analysis, selected with the MODE variable (default: envelope):
#   envelope   -> np.abs(hilbert(window_data)) -> $HOME/pl1358-01/project_data/worksof/criticality/original
#   noenvelope -> window_data                  -> $HOME/pl1358-01/project_data/worksof/criticality/noenvelope
#
# Submit both:
#   sbatch --export=ALL,MODE=envelope   sof_mfdfa.sh
#   sbatch --export=ALL,MODE=noenvelope sof_mfdfa.sh
#
# Safe to resubmit: subjects with an existing parquet are skipped, so a rerun only does
# unfinished subjects (e.g. after a timeout). Do NOT add --overwrite unless you want to redo everything.
# Do not change the analysis parameters for an existing OUT_DIR; use a new folder instead.
#
# Output per subject in OUT_DIR: {id}.parquet and logs/{id}.json.
# Check afterwards for subjects that did not finish:
#   grep -L '"outcome": "done"' "$OUT_DIR"/logs/*.json

# The UV_* variables are inherited from the shell you run sbatch in.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The MFDFA package has invalid escape sequences in its docstrings, which Python reports
# as SyntaxWarnings on first import. Harmless.
export PYTHONWARNINGS="ignore::SyntaxWarning"

MODE="${MODE:-envelope}"
# High-pass cutoff in Hz, or "none" for no filtering (e.g. sbatch --export=ALL,MODE=envelope,LOW_FREQ=none ...)
LOW_FREQ="${LOW_FREQ:-30}"
BASE_OUT="$HOME/pl1358-01/project_data/worksof/criticality"
case "$MODE" in
    envelope)   OUT_DIR="$BASE_OUT/original";   ENV_FLAG="--envelope" ;;
    noenvelope) OUT_DIR="$BASE_OUT/noenvelope"; ENV_FLAG="--no-envelope" ;;
    *) echo "ERROR: MODE must be 'envelope' or 'noenvelope', got '$MODE'" >&2; exit 1 ;;
esac
# Results with a different filter must not mix with the default (30 Hz) ones: use a separate folder.
if [ "$LOW_FREQ" != "30" ]; then
    OUT_DIR="${OUT_DIR}_lf${LOW_FREQ}"
fi

CODE_DIR="$HOME/pl1358-01/project_data/code/criticality"
SCRIPT="sof_mfdfa_subjectwise.py"

cd "$CODE_DIR" || exit 1

# Fail fast if the script is missing or outdated.
if ! grep -q 'SCRIPT_VERSION = "1.0"' "$SCRIPT"; then
    echo "ERROR: $CODE_DIR/$SCRIPT is missing or not version 1.0; copy the latest script there." >&2
    exit 1
fi

echo "MODE=$MODE OUT_DIR=$OUT_DIR task=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"

# The exit status of python (1 if any subject failed) is the task's exit status.
uv run --frozen python -u "$SCRIPT" \
    --out-dir "$OUT_DIR" \
    $ENV_FLAG \
    --low-freq "$LOW_FREQ" \
    --window-divider 4
