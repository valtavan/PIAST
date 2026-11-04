#!/bin/bash
#SBATCH --job-name=sof_pilot_scale
#SBATCH --output=sof_pilot_scale-%j.log   # %j = job id; written in the directory you run sbatch from
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=01:30:00
# The UV_* variables are inherited from the shell you run sbatch in.
#
# Pilot 2: does the smallest MFDFA scale change how much detrend order 1 and 2 differ?
# Reruns the same 9 subjects at detrend orders 1 and 2 with larger smallest scales
# (0.25 s and 0.5 s), then summarizes them next to the earlier 0.125 s runs.
# Expected runtime: about 9 subjects x 4 runs x 40 s = 25 min.
#
# Plain job, NOT an array (sof_mfdfa.py would shard the subjects across array tasks).
# Needs the earlier pilot folders order1 and order2 (smallest scale 0.125 s) in PILOT_ROOT.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The MFDFA package has invalid escape sequences in its docstrings, which Python 3.14
# reports as SyntaxWarnings on first import. Harmless.
export PYTHONWARNINGS="ignore::SyntaxWarning"

CODE_DIR="$HOME/pl1358-01/project_data/code/criticality"
PILOT_ROOT="$HOME/pl1358-01/project_data/worksof/criticality_pilot"
SUBJECTS="4 20 21 603 1745 2705 3837 4905 5128"   # same subjects as the first pilot
ORDERS="1 2"
SCALE_MINS="0.25 0.5"                             # seconds; the first pilot used 0.125

cd "$CODE_DIR" || exit 1
mkdir -p "$PILOT_ROOT"

# Fail fast if an outdated sof_mfdfa.py (without --scale-min-s) is installed here.
if ! grep -q -- "--scale-min-s" sof_mfdfa.py; then
    echo "ERROR: $CODE_DIR/sof_mfdfa.py has no --scale-min-s option; copy the version 1.1 script there." >&2
    exit 1
fi

failed=0
dirs=("$PILOT_ROOT/order1" "$PILOT_ROOT/order2")   # earlier runs, smallest scale 0.125 s
for smin in $SCALE_MINS; do
    for order in $ORDERS; do
        out="$PILOT_ROOT/order${order}_smin${smin}"
        dirs+=("$out")
        echo "=============== detrend order $order, smallest scale $smin s -> $out ==============="
        uv run --frozen python -u sof_mfdfa.py \
            --subjects $SUBJECTS \
            --out-dir "$out" \
            --detrend-order "$order" \
            --scale-min-s "$smin" \
            --overwrite || failed=1
    done
done

echo
echo "=============== pilot summary ==============="
uv run --frozen python -u pilot_summary.py "${dirs[@]}" | tee "$PILOT_ROOT/pilot_summary_scale.txt"

exit $failed
