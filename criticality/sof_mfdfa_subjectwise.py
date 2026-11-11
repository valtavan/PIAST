#!/usr/bin/env python
# coding: utf-8
"""
MFDFA on SOF polysomnography EEG, one 30 s epoch at a time, for HPC (SLURM array) use.

Adapted from edf_to_mfdfa_subjectwise.py. Main changes:
  * paths point at the HPC project folders
  * subjects are sharded over SLURM array tasks (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT)
  * one parquet per subject ({id}.parquet) instead of appending to one shared file,
    so parallel tasks never write to the same file
  * subjects with an existing parquet are skipped, so resubmitting is safe
  * --envelope / --no-envelope switches np.abs(hilbert(window_data)) on or off

Output folders:
  envelope    (Hilbert amplitude envelope) -> criticality/original   (default)
  no envelope (raw band-passed EEG)        -> criticality/noenvelope
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from MFDFA import MFDFA
from scipy.signal import hilbert

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "1.0"

HYP_DIR = Path("~/pl1358-01/project_data/worksof/hypnograms/expert").expanduser()
EDF_DIR = Path("~/pl1358-01/project_data/SOFdataset/sleep/polysomnography/edfs").expanduser()
DEFAULT_OUT_DIR = Path("~/pl1358-01/project_data/worksof/criticality/original").expanduser()

EDF_PREFIX = "sof-visit-8-"
EEG_CHANNELS = ["C3A2", "C4A1", "C3A1", "C4A2"]
Q_LIST = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
WINDOW_DURATION_S = 30


def freq_or_none(value):
    return None if str(value).lower() in ("none", "null", "") else float(value)


_STAGE_TOKENS = {"w", "wake", "n1", "n2", "n3", "n4", "r", "rem", "s1", "s2", "s3", "s4",
                 "m", "mt", "?", "unscored", "movement", "nan", ""}
_STAGE_COLUMN_NAMES = ("sleep_stage", "stage", "hypnogram", "label", "value")


def _is_number(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def load_hypnogram(path):
    """Read hypnogram_{id}.csv and return a list with one sleep stage per 30 s epoch.

    ASSUMED format (check against a real file): one stage per row, with or without a header.
    A header is detected when the first row has a cell that is neither a number nor a
    known stage label. If there are several columns, a column named like sleep_stage/stage/
    hypnogram/label/value is used, otherwise the last column.
    """
    df = pd.read_csv(path, header=None, dtype=str, skip_blank_lines=True)
    first = [str(c).strip() for c in df.iloc[0]]
    has_header = any(not _is_number(c) and c.lower() not in _STAGE_TOKENS for c in first)
    if has_header:
        df.columns = [c.lower() for c in first]
        df = df.iloc[1:]
        col = next((c for c in df.columns if c in _STAGE_COLUMN_NAMES), df.columns[-1])
    else:
        col = df.columns[-1]
    values = df[col].str.strip()
    if len(values) == 0:
        raise ValueError(f"hypnogram has no epochs: {path}")
    if values.map(_is_number).all():
        return [int(float(v)) if float(v).is_integer() else float(v) for v in values]
    return values.tolist()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--edf-dir", type=Path, default=EDF_DIR)
    p.add_argument("--hyp-dir", type=Path, default=HYP_DIR,
                   help="Folder with hypnogram_{id}.csv files.")
    p.add_argument("--envelope", action=argparse.BooleanOptionalAction, default=True,
                   help="Run MFDFA on np.abs(hilbert(window_data)) (default) or on window_data (--no-envelope).")
    p.add_argument("--low-freq", type=freq_or_none, default=30,
                   help="High-pass cutoff in Hz, or 'none' for no high-pass (default 30).")
    p.add_argument("--high-freq", type=freq_or_none, default=None,
                   help="Low-pass cutoff in Hz, or 'none' (default). With both 'none', no filtering is done.")
    p.add_argument("--window-divider", type=int, default=4,
                   help="Largest scale = window_samples / window_divider.")
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--overwrite", action="store_true", help="Recompute subjects that already have a parquet.")
    return p.parse_args()


def single_subject_mfdfa(raw, subject_id, sleep_stages,
                         low_freq=None, high_freq=None, window_divider=10, envelope=True):
    """Return a DataFrame with one row per 30 s epoch."""
    if low_freq is not None or high_freq is not None:
        raw.filter(l_freq=low_freq, h_freq=high_freq, method="iir", verbose=False)

    rawC3 = raw.copy()
    rawC3.set_eeg_reference(ref_channels=["A2"], projection=False, verbose=False)
    rawC4 = raw.copy()
    rawC4.set_eeg_reference(ref_channels=["A1"], projection=False, verbose=False)

    sfreq = raw.info["sfreq"]
    window_samples = int(sfreq * WINDOW_DURATION_S)

    data = {
        "C3A2": rawC3.get_data(picks=["C3"], units="uV").flatten().astype(float),
        "C4A1": rawC4.get_data(picks=["C4"], units="uV").flatten().astype(float),
        "C3A1": rawC4.get_data(picks=["C3"], units="uV").flatten().astype(float),
        "C4A2": rawC3.get_data(picks=["C4"], units="uV").flatten().astype(float),
    }
    del rawC3, rawC4

    scales = np.unique(np.logspace(np.log10(16), np.log10(window_samples / window_divider),
                                   num=20, dtype=int))
    num_windows = len(sleep_stages)
    n_available = len(data["C3A2"]) // window_samples
    if num_windows == 0 or num_windows > n_available:
        raise ValueError(f"hypnogram has {num_windows} epochs but recording has {n_available} full 30 s windows")

    rows = []
    for w_idx in range(num_windows):
        start_sample = w_idx * window_samples
        end_sample = start_sample + window_samples

        row = {
            "subject_id": subject_id,
            "window_start_time_s": start_sample / sfreq,
            "sleep_stage": sleep_stages[w_idx],
        }

        for eeg_elc in EEG_CHANNELS:
            window_data = data[eeg_elc][start_sample:end_sample]

            if envelope:
                signal = np.abs(hilbert(window_data))
            else:
                signal = window_data

            lag, dfal = MFDFA(signal, lag=scales, order=1, q=Q_LIST)

            # h(q): slope of the log-log fit for each q
            for ii in range(len(Q_LIST)):
                slope, _ = np.polyfit(np.log(lag), np.log(dfal[:, ii]), 1)
                row[f"{eeg_elc}_h{ii}"] = slope

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)


    # Subjects = every EDF in the folder; the 5-digit ID is taken from the filename
    edf_files = sorted(args.edf_dir.glob(f"{EDF_PREFIX}*.edf"))
    subject_ids = [f.stem[len(EDF_PREFIX):] for f in edf_files]

    # Shard subjects over array tasks (0-based, contiguous task ids)
    my_idx = [i for i in range(len(subject_ids)) if i % args.n_tasks == args.task_id]
    print(f"[{SCRIPT_VERSION}] task {args.task_id}/{args.n_tasks}: {len(my_idx)} of {len(subject_ids)} subjects | "
          f"envelope={args.envelope} | out={args.out_dir}", flush=True)

    n_failed = 0
    for i in my_idx:
        subject_id = subject_ids[i]
        out_file = args.out_dir / f"{subject_id}.parquet"
        log_file = args.out_dir / "logs" / f"{subject_id}.json"

        if out_file.exists() and not args.overwrite:
            print(f"subject# = {subject_id}: already done, skipping", flush=True)
            continue

        t0 = time.time()
        info = {"subject_id": subject_id, "script_version": SCRIPT_VERSION, "envelope": args.envelope,
                "low_freq": args.low_freq, "high_freq": args.high_freq,
                "window_divider": args.window_divider, "outcome": "started"}
        try:
            edf_path = edf_files[i]

            # A missing hypnogram raises FileNotFoundError -> subject logged as failed, no parquet
            sleep_stages = load_hypnogram(args.hyp_dir / f"hypnogram_{subject_id}.csv")

            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            raw.pick("eeg", verbose=False)
            raw.load_data(verbose=False)

            print(f"subject# = {subject_id} | epochs = {len(sleep_stages)}", flush=True)
            df = single_subject_mfdfa(raw=raw, subject_id=subject_id, sleep_stages=sleep_stages,
                                      low_freq=args.low_freq,
                                      high_freq=args.high_freq, window_divider=args.window_divider,
                                      envelope=args.envelope)

            # Write atomically so a timeout never leaves a half-written parquet
            tmp = out_file.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, out_file)

            info.update(outcome="done", n_epochs=len(df))
            print(f"subject# = {subject_id}: done in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:
            n_failed += 1
            info.update(outcome="failed", error=repr(e))
            traceback.print_exc()
        info["seconds"] = round(time.time() - t0, 1)
        log_file.write_text(json.dumps(info, indent=2))

    print(f"task {args.task_id} finished, {n_failed} failed", flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
