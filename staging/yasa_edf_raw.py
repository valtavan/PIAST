"""
Batch YASA single-derivation sleep staging for the SOF cohort.

Processes the .edf files in EDF_DIR, saving one epoch-level CSV per file
immediately after it's scored, and clearing memory before moving to the
next file.
"""

import os
import re
import glob
import gc
import time
import socket
import warnings
from datetime import datetime, timedelta

import mne
import numpy as np
import pandas as pd
import yasa
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
mne.set_log_level("ERROR")

# -----------------------------------------------------------------------------
# Configuration & Metadata
# -----------------------------------------------------------------------------
EDF_DIR = os.path.expanduser("~/pl1358-01/project_data/SOFdataset/sleep/polysomnography/edfs")
OUTPUT_DIR = os.path.expanduser("~/pl1358-01/project_data/worksof/hypnograms/yasaRAW")
AGE_CSV_PATH = os.path.expanduser("~/pl1358-01/project_data/SOFdataset/sleep/datasets/age-participant.csv")

# Slurm job array support (defaults make it work as a normal single run)
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
TASK_MIN = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
N_TASKS = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
SHARD = TASK_ID - TASK_MIN
MAX_FILES = int(os.environ.get("MAX_FILES", 0)) or None

RUN_TAG = f"task{SHARD:02d}of{N_TASKS}"
LOG_PATH = os.path.join(OUTPUT_DIR, f"_batch_run_log_{RUN_TAG}.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "_batch_run_summary.csv")

# Load per-subject age mapping (stripping leading zeros from nsrrid)
age_df = pd.read_csv(AGE_CSV_PATH, dtype={"nsrrid": str})
AGE_MAP = dict(zip(age_df["nsrrid"].str.strip().str.lstrip("0"), age_df["v8age"]))
DEFAULT_AGE = 80  # Fallback age if participant ID is not found or NaN

WINDOW_SIZE_S = 30
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
YASA_TO_STAGE_NAME = {"WAKE": "Wake", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM"}

REQUIRED_CHANNELS = ["C3", "A2", "LOC", "ROC"]

# Known (left, right) chin EMG electrode name pairs; first pair found in a file is used
CHIN_PAIRS = [("L Chin", "R Chin"), ("EMG/L", "EMG/R")]

SKIP_EXISTING = True  # resume-safe: skip a file if its output CSV already exists

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fmt(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def get_subject_id(edf_stem: str) -> str:
    """Return the last run of digits in the filename stem.

    e.g. "sof-visit-8-00004" -> "00004"
    """
    matches = re.findall(r"\d+", edf_stem)
    return matches[-1] if matches else edf_stem


def find_chin_pair(ch_names):
    """Return the (left, right) chin EMG channel names present in this file, or None."""
    for left, right in CHIN_PAIRS:
        if left in ch_names and right in ch_names:
            return left, right
    return None


def append_log(row: dict) -> None:
    """Append one row to this task's log CSV (survives the job being killed)."""
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", index=False,
                               header=not os.path.isfile(LOG_PATH))


def process_file(edf_path: str) -> dict:
    """Run the full staging pipeline on one EDF file and save its epoch CSV."""
    edf_stem = os.path.splitext(os.path.basename(edf_path))[0]
    subject_id = get_subject_id(edf_stem)
    output_stem = f"8_YASA_RAW_SingleCh_{edf_stem}_hypnogram"
    csv_path = os.path.join(OUTPUT_DIR, f"{output_stem}_epochs.csv")

    result = {"file": edf_stem, "subject_id": subject_id, "status": None,
              "n_epochs": None, "age_used": None, "age_from_csv": None,
              "chin_channels": None,
              "load_seconds": None, "stage_seconds": None, "error": None}

    if SKIP_EXISTING and os.path.isfile(csv_path):
        result["status"] = "skipped_existing"
        return result

    raw = raw_prep = sls = hypno = prob_df = epochs_df = None
    try:
        t_load = time.time()
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        missing = [ch for ch in REQUIRED_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise ValueError(f"Missing required channel(s): {missing}")

        chin = find_chin_pair(raw.ch_names)
        if chin is None:
            raise ValueError(f"No chin EMG pair found (looked for {CHIN_PAIRS})")
        chin_l, chin_r = chin
        result["chin_channels"] = f"{chin_l}/{chin_r}"

        # Construct Clinical Derivations
        raw_prep = mne.set_bipolar_reference(raw, anode="C3", cathode="A2",
                                             ch_name="C3-A2", drop_refs=False, verbose=False)
        raw_prep = mne.set_bipolar_reference(raw_prep, anode="LOC", cathode="ROC",
                                             ch_name="LOC-ROC", drop_refs=False, verbose=False)
        raw_prep = mne.set_bipolar_reference(raw_prep, anode=chin_l, cathode=chin_r,
                                             ch_name="EMG_Chin", drop_refs=False, verbose=False)

        # Channel Re-Typing
        raw_prep.set_channel_types({"LOC-ROC": "eog", "EMG_Chin": "emg"})
        result["load_seconds"] = round(time.time() - t_load, 1)

        # Match participant age (strip leading zeroes to match nsrrid)
        nsrrid_key = subject_id.lstrip("0") or "0"
        subject_age = AGE_MAP.get(nsrrid_key, np.nan)
        result["age_from_csv"] = bool(not pd.isna(subject_age))
        if pd.isna(subject_age):
            subject_age = DEFAULT_AGE
        result["age_used"] = float(subject_age)

        # SOF is a cohort of women, so male=0
        metadata = {"age": float(subject_age), "male": 0}

        # ---------------------------------------------------------------
        # Single-Derivation YASA Staging
        # ---------------------------------------------------------------
        t_stage = time.time()
        sls = yasa.SleepStaging(
            raw_prep,
            eeg_name="C3-A2",
            eog_name="LOC-ROC",
            emg_name="EMG_Chin",
            metadata=metadata,
        )
        hypno = sls.predict()
        y_pred = hypno.as_int().to_numpy()
        prob_df = hypno.proba.rename(columns=YASA_TO_STAGE_NAME)
        result["stage_seconds"] = round(time.time() - t_stage, 1)

        # ---------------------------------------------------------------
        # Export Epoch CSV
        # ---------------------------------------------------------------
        epoch_start_min = np.arange(len(y_pred)) * WINDOW_SIZE_S / 60.0
        epochs_df = pd.DataFrame({
            "subject_id": subject_id,
            "source_file": edf_stem,
            "epoch_start_time_min": epoch_start_min,
            "stage_index": y_pred,
            "stage_name": pd.Series(y_pred).map(dict(enumerate(STAGE_NAMES))),
            "confidence": prob_df.max(axis=1).to_numpy(),
        })
        for stage in STAGE_NAMES:
            if stage in prob_df.columns:
                epochs_df[f"prob_{stage}"] = prob_df[stage].to_numpy()

        # Write to a temp file first, then rename, so a job killed mid-write
        # never leaves a half-written CSV that SKIP_EXISTING would trust.
        tmp_path = csv_path + ".tmp"
        epochs_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)

        result["status"] = "ok"
        result["n_epochs"] = len(y_pred)
        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    finally:
        del raw, raw_prep, sls, hypno, prob_df, epochs_df
        gc.collect()


def main():
    t_start = time.time()
    started = datetime.now()

    all_files = sorted(set(glob.glob(os.path.join(EDF_DIR, "*.edf")))
                       | set(glob.glob(os.path.join(EDF_DIR, "*.EDF"))))
    edf_files = all_files[SHARD::N_TASKS]  # this task's share of the files
    if MAX_FILES:
        edf_files = edf_files[:MAX_FILES]

    print(f"[{RUN_TAG}] host {socket.gethostname()}, started {started:%Y-%m-%d %H:%M:%S}")
    print(f"[{RUN_TAG}] {len(edf_files)} file(s) assigned (of {len(all_files)} in {EDF_DIR})\n",
          flush=True)

    results = []
    work_times = []  # durations of files actually processed (not skipped)
    for i, edf_path in enumerate(edf_files, 1):
        fname = os.path.basename(edf_path)
        print(f"[{i}/{len(edf_files)}] {fname} ...", end=" ", flush=True)

        t0 = time.time()
        result = process_file(edf_path)
        dt = time.time() - t0
        elapsed = time.time() - t_start

        result["seconds"] = round(dt, 1)
        result["elapsed_total_s"] = round(elapsed, 1)
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        results.append(result)
        append_log(result)

        if result["status"] != "skipped_existing":
            work_times.append(dt)
        remaining = len(edf_files) - i
        eta = f" | ETA {fmt(np.mean(work_times) * remaining)}" if work_times and remaining else ""
        tail = f" - {result['error']}" if result["error"] else ""
        print(f"{result['status']} ({dt:.1f}s){tail} | elapsed {fmt(elapsed)}{eta}", flush=True)

    total = time.time() - t_start
    log_df = pd.DataFrame(results)
    counts = log_df["status"].value_counts() if len(log_df) else pd.Series(dtype=int)
    n_fallback_age = int((log_df["age_from_csv"] == False).sum()) if len(log_df) else 0  # noqa: E712

    summary = {
        "run_tag": RUN_TAG,
        "host": socket.gethostname(),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(total, 1),
        "total_hms": fmt(total),
        "n_assigned": len(edf_files),
        "n_ok": int(counts.get("ok", 0)),
        "n_failed": int(counts.get("failed", 0)),
        "n_skipped": int(counts.get("skipped_existing", 0)),
        "mean_seconds_per_processed_file": round(float(np.mean(work_times)), 1) if work_times else None,
        "n_default_age_used": n_fallback_age,
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, mode="a", index=False,
                                   header=not os.path.isfile(SUMMARY_PATH))

    print(f"\n--- Batch summary [{RUN_TAG}] ---")
    print(counts.to_string())
    print(f"Total time: {fmt(total)}"
          + (f" (mean {summary['mean_seconds_per_processed_file']}s per processed file)" if work_times else ""))
    if n_fallback_age:
        print(f"WARNING: default age ({DEFAULT_AGE}) was used for {n_fallback_age} file(s); "
              f"see 'age_from_csv' in the log.")
    failed = log_df[log_df["status"] == "failed"] if len(log_df) else log_df
    if len(failed):
        print("\nFailed files:")
        print(failed[["file", "error"]].to_string(index=False))
    print(f"\nRun log: {LOG_PATH}\nSummary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
