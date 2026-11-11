#!/usr/bin/env python
"""
compare_hurst_variants.py - which processing choice changes h(q=2)? Runs the same epochs through the
Colab-notebook recipe, the sof_mfdfa.py recipe, and the steps in between, and prints the median h(2)
per sleep stage for every variant.

Factors that differ between the notebook and sof_mfdfa.py
    rate     notebook reads the EDF without include=, so MNE gives every channel the file's HIGHEST rate
             (256 Hz in your log) and resamples the EEG; sof_mfdfa.py reads the EEG at its native rate (128 Hz)
    scales   notebook: logspace(0.5, log10(N/4), 20) astype(int), >= 4 samples  (about 0.016 s to 7.5 s)
             repo:     20 log-spaced scales from 0.125 s to N/4 = 7.5 s
    order    notebook: MFDFA default (1); repo full run: 2
    signal   notebook: one raw channel (C3); repo: C3 - A2 (and three more derivations, averaged)
    notch    notebook: none; repo: 60 Hz zero-phase notch over the whole night

Variants
    0 notebook                      max rate, notebook scales, order 1, raw C3, no notch
    1 + native rate                 EEG read at its native rate
    2 + repo scales                 0.125 s ... 7.5 s
    3 + order 2
    4 + C3-A2 derivation
    5 + 60 Hz notch                 = sof_mfdfa.py for the C3-A2 derivation
    5 but ...                       variant 5 with one factor put back to the notebook choice
                                    (order 1 / notebook scales / raw C3 / upsampled rate)

The SAME epochs are used for every variant (random sample per stage, seeded), so differences are due to
the processing only. Output: median over subjects of each subject's median h(2), per stage.

Usage:
    uv run --frozen python helpers/compare_hurst_variants.py 00004 00020 00021
    uv run --frozen python helpers/compare_hurst_variants.py 00004 --max-epochs-per-stage 100 --csv out.csv

Needs: mne, MFDFA, scipy, numpy, pandas (as in the pyproject environment).
"""
import argparse
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from MFDFA import MFDFA
from scipy.signal import filtfilt, iirnotch

PROJECT = Path("~/pl1358-01/project_data").expanduser()
EDF_DIR = PROJECT / "SOFdataset/sleep/polysomnography/edfs"
HYP_DIR = PROJECT / "worksof/hypnograms/expert"  # hypnogram_XXXXX.csv: epoch, stage_code, start_sec

EPOCH_S = 30
STAGES = {0: "wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
CHANNELS = ("C3", "A2")
FLAT_STD_UV = 0.05
NOTCH_HZ, NOTCH_Q = 60.0, 60.0

BASE = dict(rate="max", scales="notebook", order=1, signal="C3", notch=False)
VARIANTS = {}
_cur = dict(BASE)
for _name, _change in [
    ("0 notebook", {}),
    ("1 + native rate", dict(rate="native")),
    ("2 + repo scales", dict(scales="repo")),
    ("3 + order 2", dict(order=2)),
    ("4 + C3-A2", dict(signal="C3-A2")),
    ("5 + notch (= repo)", dict(notch=True)),
]:
    _cur = {**_cur, **_change}
    VARIANTS[_name] = dict(_cur)
_REPO = dict(VARIANTS["5 + notch (= repo)"])
for _name, _change in [
    ("5 but order 1", dict(order=1)),
    ("5 but notebook scales", dict(scales="notebook")),
    ("5 but raw C3", dict(signal="C3", notch=False)),
    ("5 but upsampled rate", dict(rate="max")),
]:
    VARIANTS[_name] = {**_REPO, **_change}


# ---------------------------------------------------------------- scales
def notebook_scales(epoch_samples):
    """The notebook recipe: logspace(0.5, log10(N/4), 20), truncated to int, >= 4 samples."""
    max_scale = max(4, int(epoch_samples / 4))
    s = np.unique(np.logspace(0.5, np.log10(max_scale), 20).astype(int))
    return s[s >= 4]


def repo_scales(sfreq, epoch_samples, scale_min_s=0.125, n_scales=20):
    """sof_mfdfa.make_scales: log-spaced from scale_min_s to epoch/4, rounded integer samples."""
    s_min = int(round(scale_min_s * sfreq))
    s_max = int(epoch_samples * 0.25)
    return np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max), n_scales)).astype(int))


def get_scales(kind, sfreq, epoch_samples):
    return notebook_scales(epoch_samples) if kind == "notebook" else repo_scales(sfreq, epoch_samples)


# ---------------------------------------------------------------- data
def load_channels(edf_path, native):
    """C3 and A2 in uV and the sampling rate. native=False mimics the notebook (MNE default read)."""
    labels = mne.io.read_raw_edf(edf_path, preload=False, verbose=False).ch_names
    names = {c: next(l for l in labels if l.upper() == c) for c in CHANNELS}
    raw = mne.io.read_raw_edf(edf_path, preload=False, include=list(names.values()) if native else None,
                              verbose=False)
    data = raw.get_data(picks=list(names.values()), units="uV").astype(np.float64)
    return {c: data[i] for i, c in enumerate(CHANNELS)}, float(raw.info["sfreq"])


def build_signal(chan, sfreq, signal, notch):
    x = {c: chan[c].copy() for c in CHANNELS}
    if notch:
        b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=sfreq)
        x = {c: filtfilt(b, a, v) for c, v in x.items()}
    return x["C3"] - x["A2"] if signal == "C3-A2" else x["C3"]


def hurst(x, scales, order):
    """h(q=2): slope of log F_2(s) vs log s."""
    with np.errstate(all="ignore"):
        lag, dfa = MFDFA(x, lag=scales, q=2, order=order)
        lag = np.asarray(lag, dtype=float)
        f = np.asarray(dfa, dtype=float).reshape(len(lag), -1)[:, 0]
        if not np.isfinite(np.log(f)).all():
            return np.nan
        return float(np.polyfit(np.log(lag), np.log(f), 1)[0])


# ---------------------------------------------------------------- per subject
def pick_epochs(hyp_file, n_epochs, per_stage, rng):
    hyp = pd.read_csv(hyp_file)
    stage_at = dict(zip(hyp["start_sec"].round().astype(int), hyp["stage_code"].astype(int)))
    picked = {}
    for code in STAGES:
        idx = np.array([e for e in range(n_epochs) if stage_at.get(e * EPOCH_S) == code], dtype=int)
        if len(idx):
            picked[code] = np.sort(rng.choice(idx, size=min(per_stage, len(idx)), replace=False))
    return picked


def run_subject(sid, edf_dir, hyp_dir, per_stage, seed):
    edf = edf_dir / f"sof-visit-8-{sid}.edf"
    hyp = hyp_dir / f"hypnogram_{sid}.csv"
    if not edf.exists() or not hyp.exists():
        print(f"[skip] {sid}: missing {'EDF' if not edf.exists() else 'hypnogram'}")
        return [], {}
    loaded = {"native": load_channels(edf, True), "max": load_channels(edf, False)}
    n_epochs = min(len(next(iter(chan.values()))) // int(round(sf * EPOCH_S)) for chan, sf in loaded.values())
    picked = pick_epochs(hyp, n_epochs, per_stage, np.random.default_rng([int(sid), seed]))
    info = {"sfreq_native": loaded["native"][1], "sfreq_max": loaded["max"][1], "n_epochs": n_epochs,
            "epochs_per_stage": {STAGES[c]: len(v) for c, v in picked.items()}}

    rows, cache = [], {}
    for vname, v in VARIANTS.items():
        chan, sfreq = loaded[v["rate"]]
        key = (v["rate"], v["signal"], v["notch"])
        if key not in cache:
            cache[key] = build_signal(chan, sfreq, v["signal"], v["notch"])
        x = cache[key]
        n = int(round(sfreq * EPOCH_S))
        scales = get_scales(v["scales"], sfreq, n)
        for code, idx in picked.items():
            for e in idx:
                seg = x[e * n:(e + 1) * n]
                if len(seg) < n or not np.isfinite(seg).all() or seg.std() < FLAT_STD_UV:
                    continue
                rows.append((sid, vname, STAGES[code], hurst(seg, scales, v["order"])))
        info.setdefault("scales", {})[vname] = (f"{sfreq:g} Hz, {len(scales)} scales, "
                                                f"{scales[0] / sfreq:.3f}-{scales[-1] / sfreq:.2f} s, order {v['order']}")
    return rows, info


def main():
    ap = argparse.ArgumentParser(description="Compare h(2) under the notebook, repo and in-between settings.")
    ap.add_argument("subjects", nargs="+", help="subject ids, e.g. 00004 00020")
    ap.add_argument("--edf-dir", type=lambda s: Path(s).expanduser(), default=EDF_DIR)
    ap.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=HYP_DIR)
    ap.add_argument("--max-epochs-per-stage", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=lambda s: Path(s).expanduser(), help="also write the epoch-level results here")
    args = ap.parse_args()
    mne.set_log_level("ERROR")
    warnings.simplefilter("ignore")

    all_rows, first_info = [], None
    for sid in args.subjects:
        sid = sid.zfill(5)
        print(f"processing {sid} ...", flush=True)
        rows, info = run_subject(sid, args.edf_dir, args.hyp_dir, args.max_epochs_per_stage, args.seed)
        if rows and first_info is None:
            first_info = info
            print(f"  EDF rates: EEG native {info['sfreq_native']:g} Hz, MNE default read {info['sfreq_max']:g} Hz; "
                  f"epochs per stage {info['epochs_per_stage']}")
        all_rows += rows
    if not all_rows:
        raise SystemExit("no epochs processed")

    df = pd.DataFrame(all_rows, columns=["subject", "variant", "stage", "h2"])
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)

    per_subject = df.groupby(["variant", "stage", "subject"])["h2"].median().reset_index()
    table = (per_subject.groupby(["variant", "stage"])["h2"].median().unstack("stage")
             .reindex(index=list(VARIANTS), columns=[s for s in STAGES.values() if s in set(df["stage"])]))
    table["all epochs"] = df.groupby("variant")["h2"].median().reindex(table.index)
    pd.set_option("display.width", 200)
    print(f"\nmedian h(q=2) over {df['subject'].nunique()} subject(s) (median of per-subject medians):\n")
    print(table.round(3).to_string())
    print("\nsettings of each variant (first subject):")
    for vname, text in first_info["scales"].items():
        print(f"  {vname:<24} {text}")
    print("\nreading: beta ~ 2*h - 1 for a power-law spectrum (h = 1 -> 1/f, h = 1.5 -> 1/f^2).")


if __name__ == "__main__":
    main()
