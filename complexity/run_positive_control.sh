#!/bin/bash
#SBATCH --job-name=sof_posctrl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G                     # lz_epochs_all.csv is loaded into memory
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT

# Submit from the folder that contains lz_positive_control.py, lz_h3.py, lz_h3_firth_perm.py
# and lz_night_change.py (and the project's pyproject.toml / uv.lock):
#   mkdir -p logs
#   sbatch run_positive_control.sh
#
# Needs lz_epochs_all.csv (from `lz_sof.py merge`) and night_change_subject.csv with the
# PREMMSE, V9TENG, V8AGE and AHI columns already added.

set -euo pipefail

RESULTS=~/pl1358-01/project_data/worksof/lz_results
NIGHT="${RESULTS}/night_change"
export OMP_NUM_THREADS=1             # keep BLAS from oversubscribing the allocation

uv run --frozen python lz_positive_control.py \
    --results-dir "${RESULTS}" \
    --night-dir "${NIGHT}" \
    --teng-cut 78 \
    --sws-var sws_min \
    --awak-var awak_per_h \
    --ahi hp3r

# Outputs: ${NIGHT}/positive_control/
