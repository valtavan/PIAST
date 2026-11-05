#!/usr/bin/env python
"""
delta_models.py - what explains the late - early change in h(q)?
  (1) does the amount of slow-wave sleep (N3, stage_code 3) earlier in the night predict the change?
  (2) does the change survive adjustment for the change in spectral properties?

Usage (folders come from paths.py, so no arguments are needed):
    python delta_models.py                         # h_q+2 from through_night.py output
    python delta_models.py --source scale_refit    # h2_short, h2_long, h2_full from scale_refit.py output
    python delta_models.py --no-spectral           # skip the spectral models
Full form:
    python delta_models.py [--source through_night|scale_refit] [--medians CSV] [--subjects CSV]
                           [--spectral-dir DIR | --no-spectral] [--hyp-dir DIR] [--measures M ...] [--out-dir DIR]

Inputs (defaults in brackets)
  --source    which earlier analysis to read: its subject medians and subjects table are taken from
              RESULTS_DIR/{source}/{source}_subject_medians.csv and {source}_subjects.csv [through_night]
  --medians / --subjects   explicit paths overriding --source (the subjects table supplies
              n_sws_before_late = SWS epochs between sleep onset and the start of the late window,
              n_sws_total and sleep_period_h; rerun through_night.py v2 if these columns are missing)
  --measures  columns of the medians file [h_q+2 for through_night; h2_short h2_long h2_full for scale_refit]
  spectral    used automatically when SPECTRAL_DIR (paths.py) holds sof_spectral.py output; the late - early
              change of each spectral feature is then used as covariates. Windows and stages are assigned
              exactly as in through_night.py. --spectral-dir overrides the folder, --no-spectral skips it.

Unit of analysis: one row per subject and stage, y = (median late) - (median early) of the measure,
requiring >= MIN_EPOCHS epochs of that stage in both windows (same rule as through_night.py).
Per stage, OLS models with bootstrap (over subjects) 95% CIs:
  baseline      y ~ 1                         intercept = mean change (as in through_night.py)
  n3            y ~ n3_30min + sleep_period_h
  spectral      y ~ d_slope_1_40 + d_slope_20_45 + d_rel_delta + d_rel_sigma + d_log10_power
  n3+spectral   both sets
n3_30min (SWS before the late window, units of 30 min) and sleep_period_h are centered. The spectral
changes are scaled to SD 1 but NOT centered. So in the spectral models the intercept is the predicted
change in h when the spectral features do not change (and N3 is average): an intercept that stays
clearly > 0 means the rise is not explained by the spectral changes; one that drops to about 0 means
it is. The n3_30min coefficient is the change in the h change per extra 30 min of SWS. Also reported: Spearman rho between
SWS count and the change, and the mean change by tertile of SWS count.

Caveats: this is observational and the SWS count is also a marker of sleep quality and age; subjects
who have enough epochs of a stage in both windows are a selected group (small for SWS and N1); SWS
count and sleep period length are correlated; spectral covariates are simple log-log fits (see
sof_spectral.py). No multiplicity correction is applied across models, so read the CIs, not p values.

Writes delta_models.csv (all coefficients) to OUT_DIR (default: folder of the medians file).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import paths
import through_night as tn

SPECTRAL = ["slope_1_40", "slope_20_45", "rel_delta", "rel_sigma", "log10_power"]
N_BOOT_MODEL = 2000
MIN_SUBJECTS = 20
N3_UNIT_EPOCHS = 60  # 30 min


def window_deltas(medians, measures, prefix=""):
    """Per (stage_code, subject_id): late - early of each measure; both windows need >= MIN_EPOCHS epochs."""
    sub = medians[medians["n_epochs"] >= tn.MIN_EPOCHS]
    wide = (sub.pivot(index=["stage_code", "subject_id"], columns="window", values=measures)
            .reindex(columns=pd.MultiIndex.from_product([measures, ["early", "late"]])))
    out = pd.DataFrame(index=wide.index)
    for m in measures:
        out[prefix + m] = wide[(m, "late")] - wide[(m, "early")]
    return out.dropna(how="all")


def spectral_medians(spectral_dir, hyp_dir):
    tables = []
    for parquet in sorted(spectral_dir.glob("[0-9]*.parquet")):
        sid = parquet.stem
        df = pd.read_parquet(parquet, columns=["window_start_time_s"] + SPECTRAL).dropna()
        df["t"] = df["window_start_time_s"].round().astype(int)
        table, _ = tn.assign_windows(df, hyp_dir / f"hypnogram_{sid}.csv", {"subject_id": sid})
        if table is not None:
            tables.append(table)
    if not tables:
        sys.exit(f"No usable spectral parquet files in {spectral_dir}")
    return tn.stage_medians(pd.concat(tables, ignore_index=True), SPECTRAL)


def ols_bootstrap(X, y, rng):
    """OLS coefficients, bootstrap 95% CI (resampling subjects) and R^2."""
    n = len(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ss_tot if X.shape[1] > 1 and ss_tot > 0 else 0.0
    boots = np.empty((N_BOOT_MODEL, X.shape[1]))
    for b in range(N_BOOT_MODEL):
        i = rng.integers(0, n, n)
        boots[b] = np.linalg.lstsq(X[i], y[i], rcond=None)[0]
    return beta, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0), r2


def fmt(coef, lo, hi):
    return f"{coef:+.3f} [{lo:+.3f},{hi:+.3f}]"


def main():
    parser = argparse.ArgumentParser(description="SWS dose-response and spectral adjustment of the late - early change.")
    parser.add_argument("--source", choices=["through_night", "scale_refit"], default="through_night")
    parser.add_argument("--medians", type=lambda s: Path(s).expanduser())
    parser.add_argument("--subjects", type=lambda s: Path(s).expanduser())
    parser.add_argument("--spectral-dir", type=lambda s: Path(s).expanduser(), default=paths.SPECTRAL_DIR)
    parser.add_argument("--no-spectral", action="store_true", help="skip the spectral models")
    parser.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=tn.HYP_DIR)
    parser.add_argument("--measures", nargs="+")
    parser.add_argument("--out-dir", type=lambda s: Path(s).expanduser())
    args = parser.parse_args()

    source_dir = paths.RESULTS_DIR / args.source
    args.medians = args.medians or source_dir / f"{args.source}_subject_medians.csv"
    args.subjects = args.subjects or source_dir / f"{args.source}_subjects.csv"
    for f in (args.medians, args.subjects):
        if not f.exists():
            sys.exit(f"{f} not found. Run {args.source}.py first (or pass --medians/--subjects, or edit paths.py).")
    args.measures = args.measures or (["h2_short", "h2_long", "h2_full"] if args.source == "scale_refit" else ["h_q+2"])
    if args.no_spectral:
        args.spectral_dir = None
    elif not (args.spectral_dir and args.spectral_dir.is_dir() and any(args.spectral_dir.glob("[0-9]*.parquet"))):
        print(f"[no spectral features found in {args.spectral_dir}; running without the spectral models]")
        args.spectral_dir = None
    else:
        print(f"[using spectral features from {args.spectral_dir}]")
    out_dir = args.out_dir or args.medians.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    medians = pd.read_csv(args.medians, dtype={"subject_id": str})
    subjects = pd.read_csv(args.subjects, dtype={"subject_id": str})
    missing = [m for m in args.measures if m not in medians.columns]
    if missing:
        sys.exit(f"measures not in {args.medians.name}: {missing}; available: "
                 f"{[c for c in medians.columns if c not in ('subject_id', 'stage_code', 'window', 'n_epochs', 'stage')]}")
    for col in ("n_sws_before_late", "sleep_period_h"):
        if col not in subjects.columns:
            sys.exit(f"{args.subjects.name} has no column {col}; rerun through_night.py / scale_refit.py (current versions)")

    data = window_deltas(medians, args.measures)
    spectral_cols = []
    if args.spectral_dir:
        spec = window_deltas(spectral_medians(args.spectral_dir, args.hyp_dir), SPECTRAL, prefix="d_")
        data = data.join(spec, how="inner")
        spectral_cols = ["d_" + s for s in SPECTRAL]
    data = (data.reset_index()
            .merge(subjects[["subject_id", "n_sws_before_late", "sleep_period_h"]], on="subject_id", how="inner"))
    data["n3_30min"] = data["n_sws_before_late"] / N3_UNIT_EPOCHS

    rng = np.random.default_rng(0)
    rows, summaries = [], []
    models = {"baseline": [], "n3": ["n3_30min", "sleep_period_h"]}
    if spectral_cols:
        models.update({"spectral": spectral_cols, "n3+spectral": ["n3_30min", "sleep_period_h"] + spectral_cols})

    for measure in args.measures:
        for code, stage in tn.STAGES.items():
            d = data[data["stage_code"] == code].dropna(subset=[measure, "n3_30min", "sleep_period_h"] + spectral_cols)
            if len(d) < MIN_SUBJECTS:
                summaries.append({"measure": measure, "stage": stage, "n": len(d)})
                continue
            d = d.copy()
            for c in spectral_cols:  # SD 1, deliberately NOT centered: intercept = change in h at zero spectral change
                d[c] = d[c] / d[c].std()
            for c in ("n3_30min", "sleep_period_h"):  # centered, natural units
                d[c] = d[c] - d[c].mean()
            y = d[measure].to_numpy()
            result = {}
            for name, terms in models.items():
                X = np.column_stack([np.ones(len(d))] + [d[t].to_numpy() for t in terms])
                beta, lo, hi, r2 = ols_bootstrap(X, y, rng)
                result[name] = dict(zip(["intercept"] + terms, zip(beta, lo, hi)))
                for term, b_, l_, h_ in zip(["intercept"] + terms, beta, lo, hi):
                    rows.append({"measure": measure, "stage": stage, "model": name, "term": term, "coef": b_,
                                 "ci95_low": l_, "ci95_high": h_, "n": len(d), "r2": r2})
            rho, p = spearmanr(d["n3_30min"], y)
            counts = d["n_sws_before_late"]
            tertile = pd.qcut(counts.rank(method="first"), 3, labels=["low", "mid", "high"])
            by_tertile = " | ".join(f"{lab} {y[(tertile == lab).to_numpy()].mean():+.3f}" for lab in ["low", "mid", "high"])
            summaries.append({"measure": measure, "stage": stage, "n": len(d),
                              "sws_epochs_median": counts.median(),
                              "spearman": f"{rho:+.2f} (p={p:.3g})", "mean_change_by_sws_tertile": by_tertile,
                              "n3_coef_n3_model": fmt(*result["n3"]["n3_30min"]),
                              "n3_coef_adjusted": fmt(*result["n3+spectral"]["n3_30min"]) if spectral_cols else "",
                              "intercept_baseline": fmt(*result["baseline"]["intercept"]),
                              "intercept_n3": fmt(*result["n3"]["intercept"]),
                              "intercept_spectral": fmt(*result["spectral"]["intercept"]) if spectral_cols else "",
                              "intercept_n3_spectral": fmt(*result["n3+spectral"]["intercept"]) if spectral_cols else ""})

    pd.DataFrame(rows).to_csv(out_dir / "delta_models.csv", index=False)
    summary = pd.DataFrame(summaries)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for measure in args.measures:
        s = summary[summary["measure"] == measure].drop(columns="measure").set_index("stage")
        print(f"\n===== {measure}: late - early change per subject, within stage "
              f"(>= {tn.MIN_EPOCHS} epochs per window; SWS counted before the late window) =====")
        print("\n-- slow-wave sleep and the change")
        print(s[[c for c in ["n", "sws_epochs_median", "spearman", "mean_change_by_sws_tertile", "n3_coef_n3_model",
                             "n3_coef_adjusted"] if c in s.columns]].to_string())
        print("\n-- intercept = mean change (CI); does it survive adjustment?")
        print(s[[c for c in ["intercept_baseline", "intercept_n3", "intercept_spectral", "intercept_n3_spectral"]
                 if c in s.columns]].to_string())
        if spectral_cols:
            sp = pd.DataFrame(rows)
            sp = sp[(sp["measure"] == measure) & (sp["model"] == "spectral") & (sp["term"] != "intercept")]
            if len(sp):
                sp = sp.assign(cell=sp.apply(lambda r: f"{r['coef']:+.3f}" + ("*" if r["ci95_low"] * r["ci95_high"] > 0 else ""), axis=1))
                print("\n-- spectral model coefficients (change in h change per SD of the spectral change; * = CI excludes 0)")
                print(sp.pivot(index="term", columns="stage", values="cell")
                      .reindex(columns=[v for v in tn.STAGES.values() if v in set(sp["stage"])]).to_string())
    print(f"\nall coefficients written to {out_dir / 'delta_models.csv'}")


if __name__ == "__main__":
    main()
