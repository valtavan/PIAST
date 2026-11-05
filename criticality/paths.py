"""
paths.py - folder locations used as defaults by the analysis scripts
(through_night.py, scale_refit.py, delta_models.py, check_hypnograms.py, stage_profile.py,
pilot_summary.py). Edit here if a folder is named differently; keep this file next to the scripts.

sof_mfdfa.py, sof_spectral.py and the sbatch scripts have their own copies of the EDF and output
paths (they are the ones that were run on the cluster) and do not read this file.
"""
from pathlib import Path

PROJECT = Path("~/pl1358-01/project_data").expanduser()

HYP_DIR = PROJECT / "worksof/hypnograms/expert"            # hypnogram_XXXXX.csv
CRITICALITY_DIR = PROJECT / "worksof/criticality"
RESULTS_DIR = CRITICALITY_DIR / "order2_smin0.125"          # full MFDFA run: {id}.parquet, {id}_curves.npz, logs/
SPECTRAL_DIR = CRITICALITY_DIR / "spectral"                 # sof_spectral.py output
PILOT_DIR = PROJECT / "worksof/criticality_pilot"           # order1, order2, order1_smin0.25, ...
