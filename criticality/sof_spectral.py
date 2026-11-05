#!/usr/bin/env python
"""
sof_spectral.py - per-epoch spectral features of SOF visit-8 sleep EEG, to use as covariates
for the MFDFA analysis (delta_models.py).

Same input handling as sof_mfdfa.py (imported from it, so it must sit in the same folder):
C3, C4, A1, A2 read at their native rate, 60 Hz notch over the whole night, derivations
C3A1, C3A2, C4A1, C4A2 by subtraction, consecutive 30 s epochs (partial last epoch dropped).

Per epoch and derivation, a Welch PSD (4 s Hann segments, 50% overlap, 0.25 Hz resolution) gives:
  rel_delta 0.5-4 Hz, rel_theta 4-8, rel_alpha 8-12, rel_sigma 12-16, rel_beta 16-30, rel_gamma 30-45
      relative power = band power / power in 0.5-45 Hz  (lo <= f < hi)
  log10_power   log10 of the 0.5-45 Hz power (uV^2)
  slope_1_40    least-squares slope of log10 PSD vs log10 f over 1-40 Hz
  slope_20_45   the same over 20-45 Hz (above slow waves and most oscillations)
The slopes are negative for a falling spectrum; more negative = steeper. For a power-law signal
S(f) ~ f^-beta the slope is -beta and h(q=2) is about (1 + beta) / 2. These are simple
log-log fits that do not separate oscillatory peaks from the aperiodic part (specparam / IRASA
would); treat them as covariates, not as clean aperiodic exponents.
Features of the four derivations are averaged per epoch (NaN-aware). Epochs that are flat
(std below FLAT_STD_UV) or contain non-finite samples are NaN for that derivation.

Output: {out_dir}/{subject_id}.parquet (one row per epoch, sorted by epoch_idx) and
{out_dir}/logs/{subject_id}.json. Keep this folder SEPARATE from the MFDFA output.
Slurm array sharding, resumability and --overwrite work as in sof_mfdfa.py.

Usage:
    python sof_spectral.py [--out-dir DIR] [--subjects ID [ID ...]] [--overwrite]

Requires: mne, numpy, scipy, pandas, pyarrow (and sof_mfdfa.py with its dependencies)
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import logging
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch, welch

import sof_mfdfa as base

SCRIPT_VERSION = "1.0"
DEFAULT_OUT_DIR = Path("~/pl1358-01/project_data/worksof/criticality/spectral").expanduser()

WELCH_SEG_S = 4.0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0), "gamma": (30.0, 45.0)}
TOTAL_BAND = (0.5, 45.0)
SLOPE_RANGES = {"slope_1_40": (1.0, 40.0), "slope_20_45": (20.0, 45.0)}
FEATURES = [f"rel_{b}" for b in BANDS] + ["log10_power"] + list(SLOPE_RANGES)

log = logging.getLogger("sof_spectral")


def epoch_features(epochs, sfreq):
    """Spectral features of an (n_epochs, n_samples) array: dict name -> (n_epochs,) array."""
    nperseg = int(round(WELCH_SEG_S * sfreq))
    f, pxx = welch(epochs, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    df = f[1] - f[0]
    out = {}
    with np.errstate(all="ignore"):
        total = pxx[:, (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])].sum(axis=1) * df
        for band, (lo, hi) in BANDS.items():
            out[f"rel_{band}"] = pxx[:, (f >= lo) & (f < hi)].sum(axis=1) * df / total
        out["log10_power"] = np.log10(total)
        for name, (lo, hi) in SLOPE_RANGES.items():
            sel = (f >= lo) & (f <= hi)
            x = np.log10(f[sel])
            xc = x - x.mean()
            y = np.log10(pxx[:, sel])
            out[name] = ((y - y.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc ** 2).sum()
    return out


def analyze_subject(subject_id, edf_path, out_dir):
    labels, units, rates = base.read_edf_header(edf_path)
    info = {"channels_found": labels}
    lookup = {}
    for label, unit, rate in zip(labels, units, rates):
        lookup.setdefault(label.upper(), (label, unit, rate))
    missing = [c for c in base.CHANNELS_RAW if c not in lookup]
    if missing:
        info.update(outcome="skipped_no_channels", reason=f"missing channels: {missing}")
        return info
    info["native_channels"] = {c: {"label": lookup[c][0], "unit": lookup[c][1], "rate_hz": lookup[c][2]}
                               for c in base.CHANNELS_RAW}
    rate_set = {lookup[c][2] for c in base.CHANNELS_RAW}
    if len(rate_set) != 1:
        info.update(outcome="skipped_sfreq", reason=f"channels have different sampling rates: {sorted(rate_set)}")
        return info
    sfreq = float(rate_set.pop())
    info["sfreq"] = sfreq
    if sfreq < 2 * TOTAL_BAND[1] + 10:  # needs the 0.5-45 Hz band well below Nyquist (and the 60 Hz notch)
        info.update(outcome="skipped_sfreq", reason=f"sfreq {sfreq} Hz is too low")
        return info

    names = [lookup[c][0] for c in base.CHANNELS_RAW]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(edf_path, preload=False, include=names, verbose=False)
    if float(raw.info["sfreq"]) != sfreq:
        info.update(outcome="skipped_sfreq",
                    reason=f"MNE reports {raw.info['sfreq']} Hz but the EDF header says {sfreq} Hz")
        return info
    data = raw.get_data(picks=names, units="uV").astype(np.float64)

    n_samples = data.shape[1]
    epoch_samples = int(round(sfreq * base.EPOCH_S))
    n_epochs = n_samples // epoch_samples
    n_used = n_epochs * epoch_samples
    info.update(n_samples=n_samples, epoch_samples=epoch_samples, n_epochs=n_epochs)
    if n_epochs == 0:
        info.update(outcome="skipped_too_short", reason="recording shorter than one epoch")
        return info

    bad = ~np.isfinite(data)
    if bad.any():
        epoch_bad = bad[:, :n_used].reshape(len(base.CHANNELS_RAW), n_epochs, epoch_samples).any(axis=2)
        data = np.where(bad, 0.0, data)
    else:
        epoch_bad = np.zeros((len(base.CHANNELS_RAW), n_epochs), dtype=bool)
    del bad

    b, a = iirnotch(base.NOTCH_HZ, base.NOTCH_Q, fs=sfreq)
    for i in range(data.shape[0]):
        data[i] = filtfilt(b, a, data[i])

    idx = {c: i for i, c in enumerate(base.CHANNELS_RAW)}
    per_derivation = {name: [] for name in FEATURES}
    for name, (pos, ref) in base.DERIVATIONS.items():
        epochs = (data[idx[pos]] - data[idx[ref]])[:n_used].reshape(n_epochs, epoch_samples)
        invalid = epoch_bad[idx[pos]] | epoch_bad[idx[ref]] | (epochs.std(axis=1) < base.FLAT_STD_UV)
        feats = epoch_features(epochs, sfreq)
        for key in FEATURES:
            values = feats[key]
            values[invalid | ~np.isfinite(values)] = np.nan
            per_derivation[key].append(values)

    cols = {"subject_id": subject_id, "epoch_idx": np.arange(n_epochs, dtype=np.int32),
            "window_start_time_s": np.arange(n_epochs) * float(base.EPOCH_S)}
    with warnings.catch_warnings():  # epochs where every derivation is NaN
        warnings.simplefilter("ignore")
        for key in FEATURES:
            cols[key] = np.nanmean(np.vstack(per_derivation[key]), axis=0)
    df = pd.DataFrame(cols)

    final = out_dir / f"{subject_id}.parquet"
    tmp = out_dir / f"{subject_id}.parquet.tmp"
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    info.update(outcome="done", n_rows=len(df), n_epochs_all_nan=int(df[FEATURES].isna().all(axis=1).sum()),
                parquet=str(final))
    return info


def run_subject(subject_id, edf_path, out_dir):
    t0 = time.time()
    info = {"subject_id": subject_id, "edf_path": str(edf_path)}
    try:
        if not edf_path.exists():
            info.update(outcome="skipped_no_edf", reason="EDF file not found")
        else:
            info.update(analyze_subject(subject_id, edf_path, out_dir))
    except Exception:
        info.update(outcome="failed", traceback=traceback.format_exc())
    info.update(runtime_s=round(time.time() - t0, 1), finished=datetime.now().isoformat(timespec="seconds"),
                hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                settings={"script_version": SCRIPT_VERSION, "welch_seg_s": WELCH_SEG_S, "bands": BANDS,
                          "total_band": TOTAL_BAND, "slope_ranges": SLOPE_RANGES,
                          "notch_hz": base.NOTCH_HZ, "notch_q": base.NOTCH_Q, "flat_std_uv": base.FLAT_STD_UV})
    base.write_json(out_dir / "logs" / f"{subject_id}.json", info)
    return info["outcome"], info.get("reason") or info.get("traceback", "")


def parse_args():
    p = argparse.ArgumentParser(description="Per-epoch spectral features of SOF visit-8 sleep EEG.")
    p.add_argument("--out-dir", type=lambda s: Path(s).expanduser(), default=DEFAULT_OUT_DIR,
                   help=f"output directory, keep separate from the MFDFA output (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--subjects", nargs="+", type=base.subject_id_arg, metavar="ID",
                   help="only these subject ids (zero-padded to 5 digits)")
    p.add_argument("--overwrite", action="store_true", help="recompute subjects whose parquet already exists")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    mne.set_log_level("ERROR")
    if not base.EDF_DIR.is_dir():
        sys.exit(f"EDF directory not found: {base.EDF_DIR}")
    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)

    task_id, task_count = base.slurm_shard()
    mine = base.list_subjects(args.subjects)[task_id::task_count]
    log.info("task %d/%d: %d subjects, out dir %s", task_id, task_count, len(mine), args.out_dir)

    outcomes = {}
    for n, (subject_id, edf_path) in enumerate(mine, start=1):
        if (args.out_dir / f"{subject_id}.parquet").exists() and not args.overwrite:
            outcome, detail = "skipped_exists", ""
        else:
            outcome, detail = run_subject(subject_id, edf_path, args.out_dir)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        log.log(logging.ERROR if outcome == "failed" else logging.INFO, "[%d/%d] %s -> %s %s",
                n, len(mine), subject_id, outcome, detail.strip().splitlines()[-1] if detail.strip() else "")
    log.info("finished: %s", outcomes)
    return 1 if outcomes.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
