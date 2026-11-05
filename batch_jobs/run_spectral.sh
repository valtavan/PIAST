#!/bin/bash
#SBATCH --job-name=sof_spectral
#SBATCH --output=sof_spectral-%A_%a.log   # %A = job id, %a = array task id; written where you run sbatch
#SBATCH --array=0-11                      # 12 tasks, 0-based and contiguous (sof_spectral.py shards on this)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
# The UV_* variables are inherited from the shell you run sbatch in.
#
# Per-epoch spectral features (relative band powers, log power, two log-log slopes) for all
# recordings, as covariates for the MFDFA analysis. 453 recordings over 12 tasks = 37 to 38
# subjects per task.
# Runtime: NOT measured. Rough guess is about 10 s per subject (EDF read, notch filter, Welch),
# i.e. about 6 min per task; 00:30:00 leaves a large margin. Check runtime_s in the logs after
# the first run and tighten it if you like.
#
# Safe to resubmit: subjects with an existing parquet are skipped. Do NOT use --overwrite here.
# Output per subject in OUT_DIR: {id}.parquet and logs/{id}.json. OUT_DIR must stay separate
# from the MFDFA output folder (the analysis scripts glob {id}.parquet in their input folder).
# Check afterwards for subjects that did not finish:
#   grep -L '"outcome": "done"' "$OUT_DIR"/logs/*.json

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The MFDFA package (imported through sof_mfdfa.py) has invalid escape sequences in its
# docstrings, which Python 3.14 reports as SyntaxWarnings on first import. Harmless.
export PYTHONWARNINGS="ignore::SyntaxWarning"

CODE_DIR="$HOME/pl1358-01/project_data/code/criticality"
OUT_DIR="$HOME/pl1358-01/project_data/worksof/criticality/spectral"

cd "$CODE_DIR" || exit 1

# sof_spectral.py imports sof_mfdfa.py, so both must be here, and sof_mfdfa.py must be current.
if [ ! -f sof_spectral.py ] || ! grep -q 'SCRIPT_VERSION = "1.2"' sof_mfdfa.py; then
    echo "ERROR: $CODE_DIR needs sof_spectral.py and sof_mfdfa.py version 1.2." >&2
    exit 1
fi

# The exit status of python (1 if any subject failed unexpectedly) is the task's exit status.
uv run --frozen python -u sof_spectral.py --out-dir "$OUT_DIR"
