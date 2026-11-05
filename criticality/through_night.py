#!/usr/bin/env python
"""
through_night.py - early vs late sleep: does MFDFA h(q) change between the first and the last
1.5 h of the sleep period, within each stage?

Usage:
    python through_night.py [RESULTS_DIR] [--hyp-dir DIR] [--out-dir DIR] [--min-r2 R]

With no arguments it uses the folders in paths.py (RESULTS_DIR = the full MFDFA run).

RESULTS_DIR is a sof_mfdfa.py output folder ({id}.parquet files). Hypnograms are read from
{hyp-dir}/hypnogram_{id}.csv (epoch, stage_code, start_sec) and joined by TIME
(start_sec == window_start_time_s). stage_code: 0 wake, 1 N1, 2 N2, 3 slow-wave sleep, 4 REM.

Definitions (constants below):
  sleep onset    start of the first run of >= ONSET_RUN_EPOCHS consecutive sleep epochs (codes 1-4)
  sleep offset   end of the last such run (final awakening, or end of recording if still asleep)
  early window   [onset, onset + WINDOW_H hours)
  late window    [offset - WINDOW_H hours, offset)
  Subjects whose sleep period is shorter than 2 * WINDOW_H are excluded (windows would overlap).
  Wake inside the windows (after onset / before final awakening) is analyzed as stage 0.

Method: the four derivations are averaged per epoch (status ok only). Per subject, stage and
window, take the median of each measure, requiring >= MIN_EPOCHS epochs (a plain count, not
consecutive). For each stage, the paired difference late - early across subjects is summarized
with the mean difference, a bootstrap 95% CI over subjects, a Wilcoxon signed-rank p value, and
a Holm-adjusted p across the stages (per measure). h_q+2 is the primary measure.

The subjects table also records slow-wave sleep (stage 3) epoch counts per subject
(n_sws_total, n_sws_before_late, n_sleep_epochs) for the dose-response analysis in delta_models.py.

The functions assign_windows, stage_medians and stage_comparison are reused by scale_refit.py
and delta_models.py.

Writes to OUT_DIR: through_night_subjects.csv, through_night_subject_medians.csv,
through_night_summary.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import paths

EPOCH_S = 30
WINDOW_H = 1.5
ONSET_RUN_EPOCHS = 10  # 5 min of CONSECUTIVE sleep epochs defines onset / final awakening
MIN_EPOCHS = 10  # per subject, stage and window: a plain count of epochs, not consecutive
N_BOOT = 5000
STAGES = {0: "wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
MEASURES = ["h_q+2", "h_q-2", "delta_h", "epoch_std_uv"]  # first one is primary
HYP_DIR = paths.HYP_DIR


def sleep_period(t, stage, k):
    """(onset_s, offset_s) from runs of >= k consecutive sleep epochs (offset exclusive), or None."""
    t0 = t.min()
    grid_len = int(round((t.max() - t0) / EPOCH_S)) + 1
    sleep = np.zeros(grid_len, dtype=bool)
    sleep[np.round((t - t0) / EPOCH_S).astype(int)] = (stage >= 1) & (stage <= 4)
    edges = np.diff(np.concatenate([[False], sleep, [False]]).astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    keep = (ends - starts) >= k
    if not keep.any():
        return None
    return t0 + starts[keep][0] * EPOCH_S, t0 + ends[keep][-1] * EPOCH_S


def holm(p):
    p = np.asarray(p, dtype=float)
    adj = np.full(p.shape, np.nan)
    valid = np.where(np.isfinite(p))[0]
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (len(valid) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def assign_windows(epochs, hyp_file, info):
    """Join stages to a per-epoch table and label the early / late windows.

    epochs: DataFrame with an integer column t (epoch start, seconds) and measure columns.
    info:   dict with at least subject_id; filled with sleep period, window counts and SWS counts.
    Returns (table of epochs inside a window with columns stage_code, window, subject_id; or None, info).
    """
    if not hyp_file.exists():
        info["excluded"] = "no hypnogram file"
        return None, info
    hyp = pd.read_csv(hyp_file)
    hyp["t"] = hyp["start_sec"].round().astype(int)
    period = sleep_period(hyp["t"].to_numpy(), hyp["stage_code"].to_numpy(), ONSET_RUN_EPOCHS)
    if period is None:
        info["excluded"] = f"no run of {ONSET_RUN_EPOCHS} sleep epochs"
        return None, info
    onset, offset = period
    w = WINDOW_H * 3600
    in_period = (hyp["t"] >= onset) & (hyp["t"] < offset)
    info.update(onset_h=onset / 3600, offset_h=offset / 3600, sleep_period_h=(offset - onset) / 3600,
                n_sleep_epochs=int((in_period & hyp["stage_code"].between(1, 4)).sum()),
                n_sws_total=int((in_period & (hyp["stage_code"] == 3)).sum()),
                n_sws_before_late=int(((hyp["t"] >= onset) & (hyp["t"] < offset - w)
                                       & (hyp["stage_code"] == 3)).sum()))
    if offset - onset < 2 * w:
        info["excluded"] = f"sleep period under {2 * WINDOW_H:g} h"
        return None, info

    merged = epochs.merge(hyp[["t", "stage_code"]], on="t", how="inner")
    merged["window"] = np.select(
        [(merged["t"] >= onset) & (merged["t"] < onset + w), (merged["t"] >= offset - w) & (merged["t"] < offset)],
        ["early", "late"], default="")
    merged = merged[merged["window"] != ""].copy()
    merged["stage_code"] = merged["stage_code"].astype(int)
    merged["subject_id"] = info["subject_id"]
    info["n_epochs_early"] = int((merged["window"] == "early").sum())
    info["n_epochs_late"] = int((merged["window"] == "late").sum())
    return merged, info


def load_subject(parquet, hyp_dir, min_r2):
    """MFDFA parquet -> per-epoch table (channels averaged, status ok) -> assign_windows."""
    sid = parquet.stem
    cols = ["window_start_time_s", "channel", "status", "r2_q+2"] + MEASURES
    df = pd.read_parquet(parquet, columns=list(dict.fromkeys(cols)))
    df = df[df["status"] == "ok"]
    if min_r2 > 0:
        df = df[df["r2_q+2"] >= min_r2]
    epochs = df.groupby("window_start_time_s", as_index=False)[MEASURES].mean()
    epochs["t"] = epochs["window_start_time_s"].round().astype(int)
    return assign_windows(epochs, hyp_dir / f"hypnogram_{sid}.csv", {"subject_id": sid})


def stage_medians(windowed, measures):
    """Median of each measure per subject, stage and window, with the epoch count."""
    medians = (windowed.groupby(["subject_id", "stage_code", "window"])
               .agg(n_epochs=("t", "size"), **{m: (m, "median") for m in measures}).reset_index())
    medians["stage"] = medians["stage_code"].map(STAGES)
    return medians


def paired_stats(early, late, rng):
    diff = late - early
    boots = rng.choice(diff, size=(N_BOOT, len(diff)), replace=True).mean(axis=1)
    try:
        p = wilcoxon(diff).pvalue if len(diff) >= 6 else np.nan
    except ValueError:  # all differences zero
        p = np.nan
    return {"n_subjects": len(diff), "median_early": np.median(early), "median_late": np.median(late),
            "mean_diff": diff.mean(), "ci95_low": np.percentile(boots, 2.5), "ci95_high": np.percentile(boots, 97.5),
            "p_wilcoxon": p}


def stage_comparison(medians, measures, rng):
    """Paired late - early statistics per stage and measure, with Holm over stages per measure."""
    rows = []
    for code, name in STAGES.items():
        sub = medians[(medians["stage_code"] == code) & (medians["n_epochs"] >= MIN_EPOCHS)]
        wide = (sub.pivot(index="subject_id", columns="window", values=measures)
                .reindex(columns=pd.MultiIndex.from_product([measures, ["early", "late"]])).dropna())
        for m in measures:
            if len(wide) < 3:
                rows.append({"stage": name, "measure": m, "n_subjects": len(wide)})
                continue
            rows.append({"stage": name, "measure": m, **paired_stats(wide[(m, "early")].to_numpy(),
                                                                     wide[(m, "late")].to_numpy(), rng)})
    summary = pd.DataFrame(rows)
    summary["p_holm_over_stages"] = np.nan
    for m in measures:
        sel = summary["measure"] == m
        summary.loc[sel, "p_holm_over_stages"] = holm(summary.loc[sel, "p_wilcoxon"].to_numpy())
    return summary


def main():
    parser = argparse.ArgumentParser(description="Early vs late sleep comparison of MFDFA measures.")
    parser.add_argument("results_dir", nargs="?", default=paths.RESULTS_DIR, type=lambda s: Path(s).expanduser(),
                        help=f"sof_mfdfa.py output folder (default: {paths.RESULTS_DIR})")
    parser.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=HYP_DIR)
    parser.add_argument("--out-dir", type=lambda s: Path(s).expanduser(),
                        help="default: RESULTS_DIR/through_night")
    parser.add_argument("--min-r2", type=float, default=0.0,
                        help="sensitivity check: drop epoch-channel rows whose r2_q+2 is below this (default: off)")
    args = parser.parse_args()
    if not args.results_dir.is_dir():
        sys.exit(f"Results directory not found: {args.results_dir}. Pass the folder or edit paths.py.")
    parquets = sorted(args.results_dir.glob("[0-9]*.parquet"))
    if not parquets:
        sys.exit(f"No {{subject_id}}.parquet files in {args.results_dir}. "
                 "Point RESULTS_DIR at the sof_mfdfa.py output folder (e.g. .../criticality/order2_smin0.125).")
    out_dir = args.out_dir or args.results_dir / "through_night"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables, infos = [], []
    for parquet in parquets:
        table, info = load_subject(parquet, args.hyp_dir, args.min_r2)
        infos.append(info)
        if table is not None:
            tables.append(table)
    subjects = pd.DataFrame(infos).set_index("subject_id")
    if "excluded" not in subjects.columns:  # nobody excluded
        subjects["excluded"] = np.nan
    subjects.to_csv(out_dir / "through_night_subjects.csv")
    if not tables:
        print("No subject could be analyzed. Reasons:\n", subjects["excluded"].value_counts().to_string())
        return
    epochs = pd.concat(tables, ignore_index=True)

    used = subjects[subjects["excluded"].isna()]
    print(f"{len(used)} of {len(subjects)} subjects analyzed"
          + (f"; excluded: {subjects['excluded'].value_counts().to_dict()}" if len(used) < len(subjects) else ""))
    print(f"sleep period (h), median [min-max]: {used['sleep_period_h'].median():.1f} "
          f"[{used['sleep_period_h'].min():.1f}-{used['sleep_period_h'].max():.1f}]; "
          f"sleep onset after recording start (h), median: {used['onset_h'].median():.1f}")

    comp = (epochs.groupby(["subject_id", "window"])["stage_code"].value_counts(normalize=True)
            .unstack(fill_value=0).reindex(columns=list(STAGES), fill_value=0)
            .groupby("window").mean().rename(columns=STAGES))
    print("\nstage composition of each window (mean share across subjects; shows why stages are compared separately):")
    print(comp.round(3).to_string())

    medians = stage_medians(epochs, MEASURES)
    medians.to_csv(out_dir / "through_night_subject_medians.csv", index=False)
    summary = stage_comparison(medians, MEASURES, np.random.default_rng(0))
    summary.to_csv(out_dir / "through_night_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nlate minus early, within stage (>= {MIN_EPOCHS} epochs per window and subject; "
          f"first {WINDOW_H:g} h of sleep vs last {WINDOW_H:g} h):")
    for m in MEASURES:
        print(f"\n-- {m}" + ("  (primary)" if m == MEASURES[0] else ""))
        print(summary[summary["measure"] == m].drop(columns="measure").set_index("stage").round(4).to_string())
    print(f"\nfiles written to {out_dir}")


if __name__ == "__main__":
    main()
