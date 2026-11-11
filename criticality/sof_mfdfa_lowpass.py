#!/usr/bin/env python
"""
sof_mfdfa_lowpass.py - per-epoch multifractal DFA (MFDFA) of SOF visit-8 sleep EEG,
30 Hz low-pass variant, epochs limited to the length of the expert hypnogram.

For every EDF found in EDF_DIR (or those given with --subjects):
  1. read the hypnogram hypnogram_{id}.csv from HYP_DIR (no hypnogram -> subject skipped)
  2. read ONLY the EEG channels C3, C4, A1, A2 (matched case-insensitively by name) with
     MNE, at their native sampling rate (must be 128 Hz, checked against the EDF header)
  3. apply a zero-phase 30 Hz low-pass (Butterworth, order 4, sosfiltfilt) to the whole
     night - the only filter (no notch: 60 Hz is far above the cutoff)
  4. derive C3A1, C3A2, C4A1, C4A2 by subtraction (in microvolts)
  5. cut into consecutive 30 s epochs; the number of epochs equals the hypnogram length
     (if the hypnogram is longer than the recording, it is truncated to the full epochs
     that exist, and this is recorded in the log)
  6. run MFDFA (detrend order 2 by default) on every epoch and derivation; store h(q) and
     the R^2 of each fit, and the full log F_q(s) curves (so any scale range can be
     refitted later)
  7. write {out_dir}/{subject_id}_curves.npz, {out_dir}/{subject_id}.parquet and
     {out_dir}/logs/{subject_id}.json (the parquet is written last, so its existence
     means the subject is complete)

Hypnogram: one stage per 30 s epoch, epoch 0 = start of the recording. Codes as usual
with REM = 4: 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM. The parquet has the raw
`sleep_stage` value and a readable `stage_label` (anything outside 0-4 -> "other").

Curves file: np.load(...) gives log_f (float32, shape n_epochs x 4 x n_scales x n_q,
natural log of F_q(s); NaN where not computed or not finite), scales_samples, q_list,
channels (axis-1 order: C3A1, C3A2, C4A1, C4A2), sfreq, detrend_order, lowpass_hz. A slope
over any scale range is np.polyfit(np.log(scales_samples[sel]), log_f[..., sel, :], 1)
after moving the scale axis first.

Epoch e covers samples [e*30*sfreq, (e+1)*30*sfreq), i.e. it starts at e*30 s.
No samples are ever deleted. Every epoch gets a row, with a `status` column:
  ok | flat | nonfinite_input | nonfinite_fit | error

Slurm array sharding is read from SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
(array indices must be 0-based and contiguous, e.g. --array=0-19). Without
those variables everything runs in one process. Subjects whose parquet already
exists are skipped unless --overwrite is given.

Usage:
    python sof_mfdfa_lowpass.py [--out-dir DIR] [--detrend-order N] [--scale-min-s S]
                                [--subjects ID [ID ...]] [--overwrite]

Requires: mne, numpy, scipy, pandas, pyarrow, MFDFA
"""
import os

# MFDFA works on small arrays; BLAS threading only adds overhead. Set before numpy import.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import logging
import re
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
from MFDFA import MFDFA
from scipy.signal import butter, sosfiltfilt

SCRIPT_VERSION = "2.0-lowpass"

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EDF_DIR = Path("~/pl1358-01/project_data/SOFdataset/sleep/polysomnography/edfs").expanduser()
HYP_DIR = Path("~/pl1358-01/project_data/worksof/hypnograms/expert").expanduser()
DEFAULT_OUT_DIR = Path("~/pl1358-01/project_data/worksof/criticality/low_pass").expanduser()
FILENAME_GLOB = "sof-visit-8-*.edf"
FILENAME_RE = re.compile(r"sof-visit-8-(\d+)\.edf$")
HYP_FILENAME = "hypnogram_{id}.csv"

EPOCH_S = 30
EXPECTED_SFREQ = 128.0  # Hz; subjects whose EEG is not sampled at this rate are skipped

LOWPASS_HZ = 30.0
LOWPASS_ORDER = 4  # Butterworth order; sosfiltfilt runs it forwards and backwards (zero phase)

CHANNELS_RAW = ("C3", "C4", "A1", "A2")
DERIVATIONS = {  # name: (positive channel, reference channel); signal = pos - ref
    "C3A1": ("C3", "A1"),
    "C3A2": ("C3", "A2"),
    "C4A1": ("C4", "A1"),
    "C4A2": ("C4", "A2"),
}

STAGE_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

Q_LIST = (-4, -3, -2, -1, 1, 2, 3, 4)  # sorted; q = 0 skipped
DEFAULT_DETREND_ORDER = 2
SCALE_MIN_S = 0.125  # default smallest scale, in seconds: 16 samples at 128 Hz (--scale-min-s)
SCALE_MAX_FRAC = 0.25  # largest scale = epoch length / 4
N_SCALES = 20

FLAT_STD_UV = 0.05  # epochs with std below this (in uV) are marked "flat"

log = logging.getLogger("sof_mfdfa_lowpass")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_scales(sfreq, epoch_samples, scale_min_s):
    """Log-spaced integer scales (in samples), at most N_SCALES unique values."""
    s_min = int(round(scale_min_s * sfreq))
    s_max = int(epoch_samples * SCALE_MAX_FRAC)
    return np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max), N_SCALES)).astype(int))


def mfdfa_exponents(x, scales, order):
    """MFDFA of one epoch. Returns (h, r2, log_f).

    h and r2 are arrays over Q_LIST (NaN where a fit is not finite); log_f is the
    log F_q(s) curve, shape (n_scales, n_q), with non-finite entries set to NaN.
    """
    n_q = len(Q_LIST)
    h = np.full(n_q, np.nan)
    r2 = np.full(n_q, np.nan)
    with np.errstate(all="ignore"):
        lag, fq = MFDFA(x, lag=scales, q=np.array(Q_LIST), order=order)
        lag = np.asarray(lag)
        fq = np.asarray(fq).reshape(len(lag), -1)
        log_s = np.log(lag)
        log_f = np.log(fq)
        ok = np.isfinite(log_f).all(axis=0)
        if ok.any():
            y = log_f[:, ok]
            slope, intercept = np.polyfit(log_s, y, 1)
            resid = y - (np.outer(log_s, slope) + intercept)
            ss_res = (resid ** 2).sum(axis=0)
            ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
            h[ok] = slope
            r2[ok] = 1.0 - ss_res / ss_tot
    log_f = np.where(np.isfinite(log_f), log_f, np.nan)
    return h, r2, log_f


def read_edf_header(path):
    """Minimal EDF/EDF+ header parser. Returns (labels, physical units, sampling rates in Hz).

    Needed because MNE gives a whole file the HIGHEST sampling rate of any channel and
    silently resamples the slower ones, so raw.info['sfreq'] says nothing about the EEG.
    """
    with open(path, "rb") as f:
        fixed = f.read(256)
        n_signals = int(fixed[252:256])
        record_s = float(fixed[244:252])
        rest = f.read(n_signals * 256)
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)  # label, transducer, unit, 4 ranges, prefilter, n_samples, reserved
    blocks, pos = [], 0
    for w in widths:
        blocks.append([rest[pos + i * w: pos + (i + 1) * w].decode("ascii", "replace").strip()
                       for i in range(n_signals)])
        pos += n_signals * w
    rates = [int(n) / record_s for n in blocks[8]]
    return blocks[0], blocks[2], rates


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
    hypnogram/label/value is used, otherwise the last column. Numeric codes come back as ints.
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


def stage_label(code):
    """Readable label for a hypnogram entry (0-4 as usual, REM = 4)."""
    if isinstance(code, str):
        return code
    return STAGE_LABELS.get(code, "other")


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def settings_dict(order, scale_min_s):
    return {
        "script_version": SCRIPT_VERSION,
        "detrend_order": order,
        "q_list": list(Q_LIST),
        "n_scales_requested": N_SCALES,
        "scale_min_s": scale_min_s,
        "scale_max_frac": SCALE_MAX_FRAC,
        "lowpass_hz": LOWPASS_HZ,
        "lowpass_order": LOWPASS_ORDER,
        "expected_sfreq": EXPECTED_SFREQ,
        "flat_std_uv": FLAT_STD_UV,
        "epoch_s": EPOCH_S,
        "channels_raw": list(CHANNELS_RAW),
        "derivations": {k: list(v) for k, v in DERIVATIONS.items()},
        "hypnogram_dir": str(HYP_DIR),
    }


# ----------------------------------------------------------------------------
# Per-subject analysis
# ----------------------------------------------------------------------------
def analyze_subject(subject_id, edf_path, out_dir, order, scale_min_s):
    """Returns a dict with at least `outcome`; raises on unexpected errors."""
    info = {}

    hyp_path = HYP_DIR / HYP_FILENAME.format(id=subject_id)
    if not hyp_path.exists():
        info.update(outcome="skipped_no_hypnogram", reason=f"hypnogram not found: {hyp_path}")
        return info
    stages = load_hypnogram(hyp_path)  # raises on an unreadable file -> subject logged as failed
    n_hyp = len(stages)
    info["n_hypnogram_epochs"] = n_hyp

    labels, units, rates = read_edf_header(edf_path)
    info["channels_found"] = labels
    lookup = {}
    for label, unit, rate in zip(labels, units, rates):
        lookup.setdefault(label.upper(), (label, unit, rate))
    missing = [c for c in CHANNELS_RAW if c not in lookup]
    if missing:
        info.update(outcome="skipped_no_channels", reason=f"missing channels: {missing}")
        return info

    info["native_channels"] = {c: {"label": lookup[c][0], "unit": lookup[c][1], "rate_hz": lookup[c][2]}
                               for c in CHANNELS_RAW}
    rate_set = {lookup[c][2] for c in CHANNELS_RAW}
    if len(rate_set) != 1:
        info.update(outcome="skipped_sfreq", reason=f"channels have different sampling rates: {sorted(rate_set)}")
        return info
    sfreq = float(rate_set.pop())
    info["sfreq"] = sfreq
    if sfreq != EXPECTED_SFREQ:
        info.update(outcome="skipped_sfreq", reason=f"EEG sfreq is {sfreq} Hz, expected {EXPECTED_SFREQ} Hz")
        return info

    # include= restricts the read to the four EEG channels, so MNE does not resample them
    # to the highest rate found elsewhere in the file.
    names = [lookup[c][0] for c in CHANNELS_RAW]
    with warnings.catch_warnings():  # MNE EDF header warnings are noise here
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(edf_path, preload=False, include=names, verbose=False)
    if float(raw.info["sfreq"]) != sfreq:
        info.update(outcome="skipped_sfreq",
                    reason=f"MNE reports {raw.info['sfreq']} Hz but the EDF header says {sfreq} Hz")
        return info

    data = raw.get_data(picks=names, units="uV").astype(np.float64)
    n_samples = data.shape[1]
    epoch_samples = int(round(sfreq * EPOCH_S))
    n_available = n_samples // epoch_samples
    n_epochs = min(n_hyp, n_available)  # epochs follow the hypnogram length
    n_used = n_epochs * epoch_samples
    info.update(n_samples=n_samples, epoch_samples=epoch_samples, n_epochs_available=n_available,
                n_epochs=n_epochs, hypnogram_truncated=bool(n_hyp > n_available),
                unused_samples=n_samples - n_used)
    if n_hyp > n_available:
        log.warning("%s: hypnogram has %d epochs but the recording only has %d full epochs; truncating",
                    subject_id, n_hyp, n_available)
    if n_epochs == 0:
        info.update(outcome="skipped_too_short", reason="recording shorter than one epoch")
        return info
    stages = stages[:n_epochs]

    # Non-finite samples are never deleted: flag their epochs, zero-fill so filtering works.
    bad = ~np.isfinite(data)
    if bad.any():
        epoch_bad = bad[:, :n_used].reshape(len(CHANNELS_RAW), n_epochs, epoch_samples).any(axis=2)
        data = np.where(bad, 0.0, data)
    else:
        epoch_bad = np.zeros((len(CHANNELS_RAW), n_epochs), dtype=bool)
    del bad

    sos = butter(LOWPASS_ORDER, LOWPASS_HZ, btype="lowpass", fs=sfreq, output="sos")
    for i in range(data.shape[0]):
        data[i] = sosfiltfilt(sos, data[i])

    scales = make_scales(sfreq, epoch_samples, scale_min_s)
    info["scales_samples"] = [int(s) for s in scales]
    idx = {c: i for i, c in enumerate(CHANNELS_RAW)}
    n_q = len(Q_LIST)
    curves = np.full((n_epochs, len(DERIVATIONS), len(scales), n_q), np.nan, dtype=np.float32)
    labels_per_epoch = [stage_label(s) for s in stages]

    frames = []
    std_median = {}
    first_error = None
    for d_i, (name, (pos, ref)) in enumerate(DERIVATIONS.items()):
        signal = data[idx[pos]] - data[idx[ref]]
        epochs = signal[:n_used].reshape(n_epochs, epoch_samples)
        stds = epochs.std(axis=1)
        std_median[name] = float(np.median(stds))
        nonfinite = epoch_bad[idx[pos]] | epoch_bad[idx[ref]]

        h_all = np.full((n_epochs, n_q), np.nan)
        r2_all = np.full((n_epochs, n_q), np.nan)
        status = np.full(n_epochs, "ok", dtype=object)

        for e in range(n_epochs):
            if nonfinite[e]:
                status[e] = "nonfinite_input"
            elif stds[e] < FLAT_STD_UV:
                status[e] = "flat"
            else:
                try:
                    h, r2, log_f = mfdfa_exponents(epochs[e], scales, order)
                    h_all[e], r2_all[e], curves[e, d_i] = h, r2, log_f
                except Exception as exc:  # one bad epoch must not kill the subject
                    status[e] = "error"
                    if first_error is None:
                        first_error = f"{name} epoch {e}: {type(exc).__name__}: {exc}"
                    continue
                if not np.isfinite(h).all():
                    status[e] = "nonfinite_fit"

        cols = {
            "subject_id": subject_id,
            "epoch_idx": np.arange(n_epochs, dtype=np.int32),
            "window_start_time_s": np.arange(n_epochs) * float(EPOCH_S),
            "sleep_stage": stages,
            "stage_label": labels_per_epoch,
            "channel": name,
            "status": status,
            "detrend_order": order,
            "scale_min_s": scale_min_s,
            "sfreq": sfreq,
            "lowpass_hz": LOWPASS_HZ,
            "epoch_std_uv": stds,
            "delta_h": h_all[:, 0] - h_all[:, -1],  # h(q=-4) - h(q=+4)
        }
        for k, q in enumerate(Q_LIST):
            cols[f"h_q{q:+d}"] = h_all[:, k]
        for k, q in enumerate(Q_LIST):
            cols[f"r2_q{q:+d}"] = r2_all[:, k]
        frames.append(pd.DataFrame(cols))
        log.info("%s %s done: %d epochs, %d not ok", subject_id, name, n_epochs, int((status != "ok").sum()))

    df = (pd.concat(frames, ignore_index=True)
          .sort_values(["epoch_idx", "channel"], kind="stable")
          .reset_index(drop=True))

    final = out_dir / f"{subject_id}.parquet"
    tmp = out_dir / f"{subject_id}.parquet.tmp"
    curves_final = out_dir / f"{subject_id}_curves.npz"
    curves_tmp = out_dir / f"{subject_id}_curves.npz.tmp"
    try:
        with open(curves_tmp, "wb") as f:  # file object, so numpy does not append ".npz" to the temp name
            np.savez_compressed(f, log_f=curves, scales_samples=scales.astype(np.int32),
                                q_list=np.array(Q_LIST), channels=np.array(list(DERIVATIONS)),
                                sfreq=sfreq, detrend_order=order, lowpass_hz=LOWPASS_HZ)
        os.replace(curves_tmp, curves_final)
        df.to_parquet(tmp, index=False)  # last: its existence marks the subject as done
        os.replace(tmp, final)
    finally:
        for leftover in (tmp, curves_tmp):
            if leftover.exists():
                leftover.unlink()

    info.update(
        outcome="done",
        n_rows=len(df),
        status_counts=df["status"].value_counts().to_dict(),
        status_counts_by_channel={ch: g["status"].value_counts().to_dict() for ch, g in df.groupby("channel")},
        stage_counts=pd.Series(labels_per_epoch).value_counts().to_dict(),
        epoch_std_uv_median=std_median,
        first_epoch_error=first_error,
        parquet=str(final),
        curves_file=str(curves_final),
    )
    return info


def run_subject(subject_id, edf_path, out_dir, order, scale_min_s):
    """Wraps analyze_subject; always writes logs/{subject_id}.json. Returns (outcome, detail)."""
    t0 = time.time()
    info = {"subject_id": subject_id, "edf_path": str(edf_path)}
    try:
        if not edf_path.exists():
            info.update(outcome="skipped_no_edf", reason="EDF file not found")
        else:
            info.update(analyze_subject(subject_id, edf_path, out_dir, order, scale_min_s))
    except Exception:
        info.update(outcome="failed", traceback=traceback.format_exc())
    info.update(
        runtime_s=round(time.time() - t0, 1),
        finished=datetime.now().isoformat(timespec="seconds"),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        settings=settings_dict(order, scale_min_s),
    )
    write_json(out_dir / "logs" / f"{subject_id}.json", info)
    return info["outcome"], info.get("reason") or info.get("traceback", "")


# ----------------------------------------------------------------------------
# Subject list, sharding, CLI
# ----------------------------------------------------------------------------
def list_subjects(requested):
    """[(subject_id, edf_path)], sorted by id. `requested` is a list of ids or None."""
    if requested:
        ids = sorted({s.zfill(5) for s in requested})
        return [(i, EDF_DIR / f"sof-visit-8-{i}.edf") for i in ids]
    found = []
    for path in sorted(EDF_DIR.glob(FILENAME_GLOB)):
        m = FILENAME_RE.search(path.name)
        if m:
            found.append((m.group(1), path))
    return found


def slurm_shard():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if not 0 <= task_id < task_count:
        sys.exit(f"SLURM_ARRAY_TASK_ID={task_id} is outside 0..{task_count - 1}; "
                 "use a 0-based contiguous array such as --array=0-19")
    return task_id, task_count


def subject_id_arg(text):
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"subject id must be digits, got {text!r}")
    return text


def parse_args():
    p = argparse.ArgumentParser(description="Per-epoch MFDFA of 30 Hz low-passed SOF visit-8 sleep EEG.")
    p.add_argument("--out-dir", type=lambda s: Path(s).expanduser(), default=DEFAULT_OUT_DIR,
                   help=f"output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--detrend-order", type=int, default=DEFAULT_DETREND_ORDER, choices=[1, 2, 3, 4],
                   help=f"polynomial order of the MFDFA detrending (default: {DEFAULT_DETREND_ORDER})")
    p.add_argument("--scale-min-s", type=float, default=SCALE_MIN_S, metavar="S",
                   help=f"smallest MFDFA scale in seconds, 0.1 to 2 (default: {SCALE_MIN_S})")
    p.add_argument("--subjects", nargs="+", type=subject_id_arg, metavar="ID",
                   help="only these subject ids (zero-padded to 5 digits), e.g. --subjects 4 20 21")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute subjects whose parquet already exists")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0.1 <= args.scale_min_s <= 2.0:
        sys.exit(f"--scale-min-s must be between 0.1 and 2 seconds, got {args.scale_min_s}")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    mne.set_log_level("ERROR")

    if not EDF_DIR.is_dir():
        sys.exit(f"EDF directory not found: {EDF_DIR}")
    if not HYP_DIR.is_dir():
        sys.exit(f"Hypnogram directory not found: {HYP_DIR}")
    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)

    task_id, task_count = slurm_shard()
    mine = list_subjects(args.subjects)[task_id::task_count]
    log.info("task %d/%d: %d subjects, low-pass %g Hz, detrend order %d, scale min %g s, out dir %s",
             task_id, task_count, len(mine), LOWPASS_HZ, args.detrend_order, args.scale_min_s, args.out_dir)

    outcomes = {}
    for n, (subject_id, edf_path) in enumerate(mine, start=1):
        if (args.out_dir / f"{subject_id}.parquet").exists() and not args.overwrite:
            outcome, detail = "skipped_exists", ""  # existing log is left untouched
        else:
            outcome, detail = run_subject(subject_id, edf_path, args.out_dir, args.detrend_order, args.scale_min_s)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        level = logging.ERROR if outcome == "failed" else logging.INFO
        log.log(level, "[%d/%d] %s -> %s %s", n, len(mine), subject_id, outcome,
                detail.strip().splitlines()[-1] if detail.strip() else "")

    log.info("finished: %s", outcomes)
    return 1 if outcomes.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
