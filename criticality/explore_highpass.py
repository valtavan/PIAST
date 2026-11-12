#!/usr/bin/env python
"""
explore_highpass.py - MFDFA of SOF visit-8 sleep EEG for ONE high-pass cutoff (exploration run).

Same recipe as sof_mfdfa.py: native sampling rate, 60 Hz zero-phase notch, derivations C3A1/C3A2/C4A1/C4A2
(in uV), order-2 detrending, 20 log-spaced scales from 0.125 s to 7.5 s, q = -4..-1, +1..+4,
same parquet schema (so compare_mfdfa.py can read it too). One extra step: a zero-phase Butterworth
high-pass (order 4, forward-backward = sosfiltfilt, like MNE's method="iir") at --high-pass Hz on the
whole night. --high-pass 0 means no high-pass.

Exploration settings
  * Subjects: a fixed random sample (--n-subjects, default 50; --seed) of the paper cohort. The cohort ids
    come from the criticality_results_*.parquet found in SOFdataset (or --ids-file). The chosen list is
    saved once as <out-root>/subjects_n<N>_seed<S>.txt, so every cutoff and every shard uses the same subjects.
  * Epochs: every --epoch-step-th epoch (default 2 = half of the night, spread over the whole night, so the
    stage mix is preserved). The same epochs are used for every cutoff.
  * epoch_std_uv and the 'flat' flag are computed BEFORE the high-pass, so the set of analysable epochs
    does not depend on the cutoff.
  * Slurm: subjects are sharded over --shard/--n-shards (the sbatch script maps array ids to cutoff+shard).

Output: <out-root>/high<cutoff>/<id>.parquet (+ logs/<id>.json, optionally <id>_curves.npz with --save-curves).
Subjects whose parquet exists are skipped unless --overwrite.

Usage:
    uv run --frozen python explore_highpass.py --high-pass 0.5 [--shard 0 --n-shards 5]

Requires: mne, numpy, scipy, pandas, pyarrow, MFDFA
"""
import os

# MFDFA works on small arrays; BLAS threading only adds overhead. Set before numpy import.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import logging
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from MFDFA import MFDFA
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

SCRIPT_VERSION = "1.0-explore"

PROJECT = Path("~/pl1358-01/project_data").expanduser()
SOF_DIR = PROJECT / "SOFdataset"
EDF_DIR = SOF_DIR / "sleep/polysomnography/edfs"
HYP_DIR = PROJECT / "worksof/hypnograms/expert"
DEFAULT_OUT_ROOT = PROJECT / "worksof/criticality/explore"

EPOCH_S = 30
NOTCH_HZ, NOTCH_Q = 60.0, 60.0
HP_ORDER = 4
CHANNELS_RAW = ("C3", "C4", "A1", "A2")
DERIVATIONS = {"C3A1": ("C3", "A1"), "C3A2": ("C3", "A2"), "C4A1": ("C4", "A1"), "C4A2": ("C4", "A2")}
Q_LIST = (-4, -3, -2, -1, 1, 2, 3, 4)
SCALE_MIN_S, SCALE_MAX_FRAC, N_SCALES = 0.125, 0.25, 20
DETREND_ORDER = 2
FLAT_STD_UV = 0.05

log = logging.getLogger("explore_highpass")


# ----------------------------------------------------------------------------
# MFDFA helpers (as in sof_mfdfa.py)
# ----------------------------------------------------------------------------
def make_scales(sfreq, epoch_samples, scale_min_s=SCALE_MIN_S):
    s_min = int(round(scale_min_s * sfreq))
    s_max = int(epoch_samples * SCALE_MAX_FRAC)
    return np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max), N_SCALES)).astype(int))


def mfdfa_exponents(x, scales, order):
    """Returns (h, r2, log_f); h and r2 over Q_LIST, log_f shape (n_scales, n_q)."""
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
    """Minimal EDF header parser -> (labels, units, sampling rates in Hz)."""
    with open(path, "rb") as f:
        fixed = f.read(256)
        n_signals = int(fixed[252:256])
        record_s = float(fixed[244:252])
        rest = f.read(n_signals * 256)
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    blocks, pos = [], 0
    for w in widths:
        blocks.append([rest[pos + i * w: pos + (i + 1) * w].decode("ascii", "replace").strip()
                       for i in range(n_signals)])
        pos += n_signals * w
    return blocks[0], blocks[2], [int(n) / record_s for n in blocks[8]]


def load_eeg(edf_path):
    """Return (data[4, n_samples] in uV for C3, C4, A1, A2 at the native rate, sfreq, info) or (None, None, info)."""
    import mne  # imported here so the rest of the module can be tested without MNE

    labels, units, rates = read_edf_header(edf_path)
    info = {"channels_found": labels}
    lookup = {}
    for label, unit, rate in zip(labels, units, rates):
        lookup.setdefault(label.upper(), (label, unit, rate))
    missing = [c for c in CHANNELS_RAW if c not in lookup]
    if missing:
        info.update(outcome="skipped_no_channels", reason=f"missing channels: {missing}")
        return None, None, info
    rate_set = {lookup[c][2] for c in CHANNELS_RAW}
    if len(rate_set) != 1:
        info.update(outcome="skipped_sfreq", reason=f"channels have different sampling rates: {sorted(rate_set)}")
        return None, None, info
    sfreq = float(rate_set.pop())
    names = [lookup[c][0] for c in CHANNELS_RAW]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # include= restricts the read to the four EEG channels, so MNE does not resample them
        raw = mne.io.read_raw_edf(edf_path, preload=False, include=names, verbose=False)
    if float(raw.info["sfreq"]) != sfreq:
        info.update(outcome="skipped_sfreq", reason=f"MNE reports {raw.info['sfreq']} Hz, EDF header {sfreq} Hz")
        return None, None, info
    info["native_channels"] = {c: {"label": lookup[c][0], "unit": lookup[c][1], "rate_hz": lookup[c][2]}
                               for c in CHANNELS_RAW}
    return raw.get_data(picks=names, units="uV").astype(np.float64), sfreq, info


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Per-subject analysis
# ----------------------------------------------------------------------------
def analyze_subject(subject_id, edf_path, out_dir, hp_hz, step, save_curves):
    data, sfreq, info = load_eeg(edf_path)
    if data is None:
        return info
    info["sfreq"] = sfreq
    if sfreq <= 2 * NOTCH_HZ:
        info.update(outcome="skipped_sfreq", reason=f"sfreq {sfreq} Hz too low for a {NOTCH_HZ} Hz notch")
        return info

    n_samples = data.shape[1]
    epoch_samples = int(round(sfreq * EPOCH_S))
    n_epochs = n_samples // epoch_samples
    n_used = n_epochs * epoch_samples
    info.update(n_samples=n_samples, epoch_samples=epoch_samples, n_epochs=n_epochs)
    if n_epochs == 0:
        info.update(outcome="skipped_too_short", reason="recording shorter than one epoch")
        return info

    bad = ~np.isfinite(data)
    if bad.any():
        epoch_bad = bad[:, :n_used].reshape(len(CHANNELS_RAW), n_epochs, epoch_samples).any(axis=2)
        data = np.where(bad, 0.0, data)
    else:
        epoch_bad = np.zeros((len(CHANNELS_RAW), n_epochs), dtype=bool)
    del bad

    b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=sfreq)
    for i in range(data.shape[0]):
        data[i] = filtfilt(b, a, data[i])
    notch_only = data  # used for std / flat flags, identical for every cutoff

    if hp_hz and hp_hz > 0:
        sos = butter(HP_ORDER, hp_hz, btype="highpass", fs=sfreq, output="sos")
        padlen = min(int(20 * sfreq), n_samples - 1)  # long odd padding keeps the start/end transients small
        filtered = np.vstack([sosfiltfilt(sos, row, padlen=padlen) for row in notch_only])
    else:
        filtered = notch_only

    scales = make_scales(sfreq, epoch_samples)
    info["scales_samples"] = [int(s) for s in scales]
    idx = {c: i for i, c in enumerate(CHANNELS_RAW)}
    todo = np.arange(0, n_epochs, step)
    n_q = len(Q_LIST)
    curves = np.full((len(todo), len(DERIVATIONS), len(scales), n_q), np.nan, dtype=np.float32) if save_curves else None

    frames, first_error = [], None
    for d_i, (name, (pos, ref)) in enumerate(DERIVATIONS.items()):
        sig_flat = (notch_only[idx[pos]] - notch_only[idx[ref]])[:n_used].reshape(n_epochs, epoch_samples)
        stds = sig_flat.std(axis=1)
        sig = (filtered[idx[pos]] - filtered[idx[ref]])[:n_used].reshape(n_epochs, epoch_samples)
        nonfinite = epoch_bad[idx[pos]] | epoch_bad[idx[ref]]

        h_all = np.full((len(todo), n_q), np.nan)
        r2_all = np.full((len(todo), n_q), np.nan)
        status = np.full(len(todo), "ok", dtype=object)
        for k, e in enumerate(todo):
            if nonfinite[e]:
                status[k] = "nonfinite_input"
            elif stds[e] < FLAT_STD_UV:
                status[k] = "flat"
            else:
                try:
                    h, r2, log_f = mfdfa_exponents(sig[e], scales, DETREND_ORDER)
                    h_all[k], r2_all[k] = h, r2
                    if save_curves:
                        curves[k, d_i] = log_f
                except Exception as exc:  # one bad epoch must not kill the subject
                    status[k] = "error"
                    first_error = first_error or f"{name} epoch {e}: {type(exc).__name__}: {exc}"
                    continue
                if not np.isfinite(h).all():
                    status[k] = "nonfinite_fit"

        cols = {
            "subject_id": subject_id,
            "epoch_idx": todo.astype(np.int32),
            "window_start_time_s": todo * float(EPOCH_S),
            "channel": name,
            "status": status,
            "detrend_order": DETREND_ORDER,
            "scale_min_s": SCALE_MIN_S,
            "high_pass_hz": float(hp_hz or 0.0),
            "epoch_step": step,
            "sfreq": sfreq,
            "epoch_std_uv": stds[todo],
            "delta_h": h_all[:, 0] - h_all[:, -1],
        }
        for k, q in enumerate(Q_LIST):
            cols[f"h_q{q:+d}"] = h_all[:, k]
        for k, q in enumerate(Q_LIST):
            cols[f"r2_q{q:+d}"] = r2_all[:, k]
        frames.append(pd.DataFrame(cols))

    df = (pd.concat(frames, ignore_index=True)
          .sort_values(["epoch_idx", "channel"], kind="stable").reset_index(drop=True))

    final = out_dir / f"{subject_id}.parquet"
    tmp = out_dir / f"{subject_id}.parquet.tmp"
    curves_final = out_dir / f"{subject_id}_curves.npz"
    curves_tmp = out_dir / f"{subject_id}_curves.npz.tmp"
    try:
        if save_curves:
            with open(curves_tmp, "wb") as f:
                np.savez_compressed(f, log_f=curves, epoch_idx=todo.astype(np.int32),
                                    scales_samples=scales.astype(np.int32), q_list=np.array(Q_LIST),
                                    channels=np.array(list(DERIVATIONS)), sfreq=sfreq, high_pass_hz=hp_hz)
            os.replace(curves_tmp, curves_final)
        df.to_parquet(tmp, index=False)  # last: its existence marks the subject as done
        os.replace(tmp, final)
    finally:
        for leftover in (tmp, curves_tmp):
            if leftover.exists():
                leftover.unlink()

    info.update(outcome="done", n_rows=len(df), n_epochs_analysed=len(todo),
                status_counts=df["status"].value_counts().to_dict(), first_epoch_error=first_error)
    return info


def run_subject(subject_id, edf_path, out_dir, hp_hz, step, save_curves):
    t0 = time.time()
    info = {"subject_id": subject_id, "edf_path": str(edf_path)}
    try:
        if not edf_path.exists():
            info.update(outcome="skipped_no_edf", reason="EDF file not found")
        else:
            info.update(analyze_subject(subject_id, edf_path, out_dir, hp_hz, step, save_curves))
    except Exception:
        info.update(outcome="failed", traceback=traceback.format_exc())
    info.update(runtime_s=round(time.time() - t0, 1), finished=datetime.now().isoformat(timespec="seconds"),
                hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                settings={"script_version": SCRIPT_VERSION, "high_pass_hz": hp_hz, "high_pass_order": HP_ORDER,
                          "detrend_order": DETREND_ORDER, "q_list": list(Q_LIST), "scale_min_s": SCALE_MIN_S,
                          "notch_hz": NOTCH_HZ, "epoch_step": step})
    write_json(out_dir / "logs" / f"{subject_id}.json", info)
    return info["outcome"], info.get("reason") or info.get("traceback", "")


# ----------------------------------------------------------------------------
# Subject sample
# ----------------------------------------------------------------------------
def find_cohort_parquet():
    cands = sorted(SOF_DIR.rglob("criticality_results_*.parquet"))
    preferred = [c for c in cands if "lf0.5" in c.name and "rawEEG" in c.name]
    return (preferred or cands or [None])[0]


def read_ids(path):
    return sorted(pd.read_csv(path, header=None, dtype=str).iloc[:, 0].str.strip().str.zfill(5).unique())


def choose_subjects(out_root, n_subjects, seed, ids_file):
    """The same list for every cutoff/shard: read from file if it exists, otherwise draw it with the seed."""
    list_file = out_root / f"subjects_n{n_subjects}_seed{seed}.txt"
    if list_file.exists():
        return read_ids(list_file)
    if ids_file:
        pool = read_ids(ids_file)
    else:
        src = find_cohort_parquet()
        if src is None:
            sys.exit(f"No criticality_results_*.parquet found under {SOF_DIR}; pass --ids-file")
        log.info("cohort ids from %s", src)
        pool = sorted(pd.read_parquet(src, columns=["subject_id"])["subject_id"].astype(str).str.strip()
                      .str.zfill(5).unique())
    eligible = [i for i in pool if (EDF_DIR / f"sof-visit-8-{i}.edf").exists() and (HYP_DIR / f"hypnogram_{i}.csv").exists()]
    log.info("%d ids in the cohort, %d with EDF and hypnogram", len(pool), len(eligible))
    if len(eligible) < n_subjects:
        sys.exit(f"only {len(eligible)} eligible subjects, need {n_subjects}")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.permutation(eligible)[:n_subjects].tolist())
    tmp = list_file.with_name(f"{list_file.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(chosen) + "\n")
    os.replace(tmp, list_file)  # identical content from every task, so a race is harmless
    return chosen


def parse_hp(text):
    return 0.0 if str(text).lower() in ("0", "none", "null", "") else float(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--high-pass", type=parse_hp, required=True, help="high-pass cutoff in Hz (0/none = no high-pass)")
    ap.add_argument("--out-root", type=lambda s: Path(s).expanduser(), default=DEFAULT_OUT_ROOT)
    ap.add_argument("--n-subjects", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids-file", type=lambda s: Path(s).expanduser(), help="cohort ids to sample from (one per line)")
    ap.add_argument("--epoch-step", type=int, default=2, help="analyse every N-th epoch (2 = half of the night)")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--save-curves", action="store_true", help="also store log F_q(s) per epoch (_curves.npz)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        sys.exit(f"--shard {args.shard} outside 0..{args.n_shards - 1}")

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    import mne
    mne.set_log_level("ERROR")

    args.out_root.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_root / f"high{args.high_pass:g}"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    subjects = choose_subjects(args.out_root, args.n_subjects, args.seed, args.ids_file)
    mine = subjects[args.shard::args.n_shards]
    log.info("high-pass %g Hz | shard %d/%d: %d of %d subjects | every %d-th epoch | out %s",
             args.high_pass, args.shard, args.n_shards, len(mine), len(subjects), args.epoch_step, out_dir)

    outcomes = {}
    for n, sid in enumerate(mine, start=1):
        if (out_dir / f"{sid}.parquet").exists() and not args.overwrite:
            outcome, detail = "skipped_exists", ""
        else:
            outcome, detail = run_subject(sid, EDF_DIR / f"sof-visit-8-{sid}.edf", out_dir, args.high_pass,
                                          args.epoch_step, args.save_curves)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        log.log(logging.ERROR if outcome == "failed" else logging.INFO, "[%d/%d] %s -> %s %s", n, len(mine), sid,
                outcome, detail.strip().splitlines()[-1] if detail.strip() else "")
    log.info("finished: %s", outcomes)
    return 1 if outcomes.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
