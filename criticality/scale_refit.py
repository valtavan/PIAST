#!/usr/bin/env python
"""
scale_refit.py - is the early-vs-late change in h(q) in the short or in the long scales?

Usage:
    python scale_refit.py RESULTS_DIR [--hyp-dir DIR] [--out-dir DIR]

Uses the stored log F_q(s) curves ({id}_curves.npz from sof_mfdfa.py), so nothing is recomputed
from the EEG. For q = +2 and q = -2 the exponent is refitted over three scale ranges (seconds;
low < scale <= high):
    short  (0, 1]      full  (0, inf)      long  (1, inf)
At 128 Hz and the default scales this is 10 short scales (0.125-0.87 s) and 10 long scales
(1.1-7.5 s). `full` is the same fit as h_q+2 / h_q-2 in the parquet (checked on the first subject).
Also analyzed: h2_long_minus_short, the long-scale slope minus the short-scale slope of each epoch
(how much the log F vs log s curve bends); a change in it means the scaling is changing shape.

The four derivations are averaged per epoch; epochs with a non-finite curve are dropped. Windows,
sleep onset/offset and statistics are exactly those of through_night.py (same constants).

How to read the result: if the late - early shift is present in `long` but not `short`, it is a
large-scale effect (slow drift, slow waves, electrode changes); if in `short`, it is in faster
activity (spindles, muscle, noise); if equal, the whole curve moves.

Writes to OUT_DIR: scale_refit_subjects.csv, scale_refit_subject_medians.csv, scale_refit_summary.csv
(the medians file can be passed to delta_models.py).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import through_night as tn

SCALE_RANGES = {"short": (0.0, 1.0), "long": (1.0, np.inf), "full": (0.0, np.inf)}  # seconds, low < s <= high
Q_PREFIX = {2: "h2", -2: "hm2"}
MEASURES = ([f"h2_{r}" for r in SCALE_RANGES] + ["h2_long_minus_short"]
            + [f"hm2_{r}" for r in SCALE_RANGES])  # first one is shown first in the tables
MIN_SCALES = 4


def refit_slopes(log_f, scales, sfreq, q_index, low, high):
    """Slope of log F vs log s over the selected scales: array (n_epochs, n_derivations)."""
    seconds = scales / sfreq
    sel = (seconds > low) & (seconds <= high)
    if sel.sum() < MIN_SCALES:
        raise ValueError(f"only {int(sel.sum())} scales in ({low}, {high}] s")
    x = np.log(scales[sel].astype(np.float64))
    xc = x - x.mean()
    y = log_f[:, :, sel, q_index].astype(np.float64)
    return ((y - y.mean(axis=-1, keepdims=True)) * xc).sum(axis=-1) / (xc ** 2).sum()


def subject_epochs(npz_path):
    d = np.load(npz_path)
    log_f, scales, sfreq = d["log_f"], d["scales_samples"], float(d["sfreq"])
    q_list = [int(q) for q in d["q_list"]]
    cols = {}
    for q, prefix in Q_PREFIX.items():
        for name, (low, high) in SCALE_RANGES.items():
            slopes = refit_slopes(log_f, scales, sfreq, q_list.index(q), low, high)
            with warnings.catch_warnings():  # epochs with all derivations NaN
                warnings.simplefilter("ignore")
                cols[f"{prefix}_{name}"] = np.nanmean(slopes, axis=1)
    cols["h2_long_minus_short"] = cols["h2_long"] - cols["h2_short"]
    df = pd.DataFrame(cols)
    df["t"] = np.arange(len(df)) * tn.EPOCH_S
    return df.dropna().reset_index(drop=True)[["t"] + MEASURES]


def sanity_check(npz_path, epochs):
    """The full-range refit must reproduce h_q+2 stored in the parquet."""
    parquet = npz_path.with_name(npz_path.name.replace("_curves.npz", ".parquet"))
    if not parquet.exists():
        return "no parquet to compare with"
    stored = pd.read_parquet(parquet, columns=["window_start_time_s", "status", "h_q+2"])
    stored = stored[stored["status"] == "ok"].groupby("window_start_time_s")["h_q+2"].mean()
    mine = epochs.set_index("t")["h2_full"]
    both = stored.index.round().astype(int).intersection(mine.index)
    diff = (mine.loc[both].to_numpy() - stored.set_axis(stored.index.round().astype(int)).loc[both].to_numpy())
    return f"max |refit - stored h_q+2| over {len(both)} epochs = {np.abs(diff).max():.2e}"


def fmt(row):
    if pd.isna(row.get("mean_diff")):
        return "n/a"
    return f"{row['mean_diff']:+.3f} [{row['ci95_low']:+.3f},{row['ci95_high']:+.3f}]"


def show(summary, measures, title):
    sub = summary[summary["measure"].isin(measures)]
    table = pd.DataFrame({m: sub[sub["measure"] == m].set_index("stage").apply(fmt, axis=1) for m in measures})
    table.insert(0, "n_subjects", sub[sub["measure"] == measures[0]].set_index("stage")["n_subjects"])
    print(f"\n-- {title}: mean late - early [bootstrap 95% CI]")
    print(table.to_string())


def main():
    parser = argparse.ArgumentParser(description="Refit h(q) over short and long scales from stored curves.")
    parser.add_argument("results_dir", type=lambda s: Path(s).expanduser())
    parser.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=tn.HYP_DIR)
    parser.add_argument("--out-dir", type=lambda s: Path(s).expanduser(), help="default: RESULTS_DIR/scale_refit")
    args = parser.parse_args()
    if not args.results_dir.is_dir():
        sys.exit(f"Results directory not found: {args.results_dir}")
    curve_files = sorted(args.results_dir.glob("[0-9]*_curves.npz"))
    if not curve_files:
        sys.exit(f"No {{subject_id}}_curves.npz files in {args.results_dir}")
    out_dir = args.out_dir or args.results_dir / "scale_refit"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables, infos = [], []
    for i, npz in enumerate(curve_files):
        sid = npz.name.split("_")[0]
        epochs = subject_epochs(npz)
        if i == 0:
            print("sanity check on", sid + ":", sanity_check(npz, epochs))
        table, info = tn.assign_windows(epochs, args.hyp_dir / f"hypnogram_{sid}.csv", {"subject_id": sid})
        infos.append(info)
        if table is not None:
            tables.append(table)
    subjects = pd.DataFrame(infos).set_index("subject_id")
    if "excluded" not in subjects.columns:  # nobody excluded
        subjects["excluded"] = np.nan
    subjects.to_csv(out_dir / "scale_refit_subjects.csv")
    if not tables:
        print("No subject could be analyzed. Reasons:\n", subjects["excluded"].value_counts().to_string())
        return
    used = subjects[subjects["excluded"].isna()]
    print(f"{len(used)} of {len(subjects)} subjects analyzed"
          + (f"; excluded: {subjects['excluded'].value_counts().to_dict()}" if len(used) < len(subjects) else ""))

    medians = tn.stage_medians(pd.concat(tables, ignore_index=True), MEASURES)
    medians.to_csv(out_dir / "scale_refit_subject_medians.csv", index=False)
    summary = tn.stage_comparison(medians, MEASURES, np.random.default_rng(0))
    summary.to_csv(out_dir / "scale_refit_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n(>= {tn.MIN_EPOCHS} epochs per window and subject; first {tn.WINDOW_H:g} h of sleep vs last {tn.WINDOW_H:g} h)")
    show(summary, ["h2_short", "h2_long", "h2_full"], "h(q=+2) by scale range")
    show(summary, ["h2_long_minus_short"], "h(q=+2) long-scale slope minus short-scale slope (curve bending)")
    show(summary, ["hm2_short", "hm2_long", "hm2_full"], "h(q=-2) by scale range")
    print("\nHolm-adjusted p over stages, h(q=+2):")
    p = summary[summary["measure"].isin(["h2_short", "h2_long", "h2_full", "h2_long_minus_short"])]
    print(p.pivot(index="stage", columns="measure", values="p_holm_over_stages")
          .reindex(list(tn.STAGES.values())).round(4).to_string())
    print(f"\nfiles written to {out_dir}")


if __name__ == "__main__":
    main()
