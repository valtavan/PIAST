#!/bin/bash
#SBATCH --job-name=sof_pilot
#SBATCH --output=sof_pilot-%j.log   # %j = job id; written in the directory you run sbatch from
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
# The UV_* variables are inherited from the shell you run sbatch in.
#
# Pilot: runs sof_mfdfa.py on a few subjects at detrend orders 1, 2 and 3, then
# prints pilot_summary.py output for all three. Paste the end of the .log here.
#
# This is a plain job, NOT an array: sof_mfdfa.py shards subjects by
# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT, so an array here would split the
# pilot subjects across tasks instead of running them all in each.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CODE_DIR="$HOME/pl1358-01/project_data/code/criticality"
PILOT_ROOT="$HOME/pl1358-01/project_data/worksof/criticality_pilot"
SUBJECTS="4 20 21 5128 1745 4905 3837 2705 603"     # add more ids, space-separated; zero-padded to 5 digits by the script
ORDERS="1 2 3"

cd "$CODE_DIR" || exit 1
mkdir -p "$PILOT_ROOT"

failed=0
dirs=()
for order in $ORDERS; do
    out="$PILOT_ROOT/order$order"
    dirs+=("$out")
    echo "=============== detrend order $order -> $out ==============="
    # --overwrite so reruns of the pilot never reuse stale files from an earlier attempt
    uv run --frozen python -u sof_mfdfa.py \
        --subjects $SUBJECTS \
        --out-dir "$out" \
        --detrend-order "$order" \
        --overwrite || failed=1
done

echo
echo "=============== pilot summary ==============="
uv run --frozen python -u pilot_summary.py "${dirs[@]}" | tee "$PILOT_ROOT/pilot_summary.txt"

exit $failed
