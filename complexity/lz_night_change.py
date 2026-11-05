#!/usr/bin/env python
"""Early-to-late-night change in Lempel-Ziv complexity (LZC) within each sleep stage.

Input : lz_epochs_all.csv written by `lz_sof.py merge`
Output: (in --out-dir) night_change_subject.csv, night_change_group.csv,
        night_change_sws_dependence.csv, [night_change_mixed.csv], night_change_curve.png

Definitions
  sleep period : persistent sleep onset (first run of >= --persist-run consecutive sleep epochs)
                 to persistent final awakening (end of the last such run); brief sleep
                 fragments after the final sustained awakening are excluded
  t_rel        : (epoch start - onset) / (offset - onset), 0..1
  early / late : halves (t_rel < 0.5 / >= 0.5)  or  thirds (first / last third; middle dropped)
  delta        : mean LZ(late) - mean LZ(early), per subject x stage x channel,
                 only if BOTH periods have >= --min-epochs clean epochs of that stage

  H1: group-level test of delta != 0 within each stage
  H2: relation of delta to total SWS (N3) minutes, adjusted for sleep-period length
      (+ optional epoch-level mixed model of LZ vs t_rel within each stage, --mixed)

Usage
  uv run --frozen python lz_night_change.py
  uv run --frozen python lz_night_change.py --split thirds --min-epochs 15 --mixed
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

EPOCH_SEC = 30
STAGES = ["W", "N1", "N2", "SWS", "REM"]
DEFAULT_RES = Path.home() / "pl1358-01" / "project_data" / "worksof" / "lz_results"


# ------------------------------------------------------------------ sleep period
def sleep_periods(d: pd.DataFrame, onset_run: int) -> pd.DataFrame:
    """Per-subject sleep onset/offset and SWS volume, from the hypnogram epochs in the file."""
    ep = (d.drop_duplicates(["subject", "epoch"])[["subject", "start_sec", "stage_code"]]
            .sort_values(["subject", "start_sec"]))
    rows = []
    for sid, g in ep.groupby("subject", sort=False):
        asleep = g["stage_code"].isin([1, 2, 3, 4]).to_numpy().astype(int)
        if asleep.sum() == 0:
            continue
        # fwd[i] = sleep epochs in asleep[i : i + onset_run]      -> persistent onset
        # bwd[i] = sleep epochs in asleep[i - onset_run + 1 : i + 1] -> persistent offset
        fwd = pd.Series(asleep[::-1]).rolling(onset_run).sum().to_numpy()[::-1]
        bwd = pd.Series(asleep).rolling(onset_run).sum().to_numpy()
        hit_on = np.flatnonzero(fwd == onset_run)
        hit_off = np.flatnonzero(bwd == onset_run)
        if hit_on.size == 0 or hit_off.size == 0:
            continue
        start = g["start_sec"].to_numpy()
        onset = start[hit_on[0]]
        end = start[hit_off[-1]] + EPOCH_SEC   # end of last epoch in the final sustained sleep run
        inside = (start >= onset) & (start < end)
        sws = (g["stage_code"].to_numpy() == 3) & inside
        mid = onset + (end - onset) / 2
        rows.append({
            "subject": sid, "onset_sec": onset, "end_sec": end,
            "sp_min": (end - onset) / 60,
            "tst_min": (asleep.astype(bool) & inside).sum() * EPOCH_SEC / 60,
            "sws_min_total": sws.sum() * EPOCH_SEC / 60,
            "sws_min_first_half": (sws & (start < mid)).sum() * EPOCH_SEC / 60,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ tests
def group_tests(sub: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (stage, ch), g in sub.groupby(["stage", "channel"]):
        x = g["delta"].dropna().to_numpy()
        if len(x) < 5:
            continue
        sd = x.std(ddof=1)
        t = stats.ttest_1samp(x, 0.0)
        w = stats.wilcoxon(x)
        half = t.confidence_interval(0.95)
        out.append({
            "stage": stage, "channel": ch, "n_subjects": len(x),
            "mean_early": g["mean_early"].mean(), "mean_late": g["mean_late"].mean(),
            "delta_mean": x.mean(), "delta_sd": sd, "delta_median": np.median(x),
            "ci95_lo": half.low, "ci95_hi": half.high,
            "cohen_dz": x.mean() / sd, "pct_positive": 100 * (x > 0).mean(),
            "t": t.statistic, "p_t": t.pvalue, "p_wilcoxon": w.pvalue,
        })
    res = pd.DataFrame(out)
    if not res.empty:
        res["p_t_holm"] = multipletests(res["p_t"], method="holm")[1]
        res["p_wilcoxon_holm"] = multipletests(res["p_wilcoxon"], method="holm")[1]
    return res


def sws_dependence(sub: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    """H2: does delta vary with total SWS minutes (adjusting for sleep-period length)?"""
    m = sub.dropna(subset=["delta"]).merge(per, on="subject")
    out = []
    for (stage, ch), g in m.groupby(["stage", "channel"]):
        if len(g) < 15:
            continue
        rho, p_rho = stats.spearmanr(g["sws_min_total"], g["delta"])
        fit = smf.ols("delta ~ sws_min_total + sp_min", data=g).fit()
        out.append({
            "stage": stage, "channel": ch, "n_subjects": len(g),
            "spearman_rho": rho, "p_spearman": p_rho,
            "beta_sws_per_min": fit.params["sws_min_total"],
            "se": fit.bse["sws_min_total"], "p_adj": fit.pvalues["sws_min_total"],
            "beta_sp_min": fit.params["sp_min"],
            "sws_min_median_in_sample": g["sws_min_total"].median(),
        })
    return pd.DataFrame(out)


def mixed_models(clean: pd.DataFrame) -> pd.DataFrame:
    """Epoch-level: lz_norm ~ t_rel with subject random intercept (+ slope if it converges)."""
    out = []
    for (stage, ch), g in clean.groupby(["stage", "channel"]):
        if g["subject"].nunique() < 15:
            continue
        kind = "intercept+slope"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = smf.mixedlm("lz_norm ~ t_rel", g, groups=g["subject"],
                                  re_formula="~t_rel").fit(reml=True)
            except Exception:
                kind = "intercept"
                fit = smf.mixedlm("lz_norm ~ t_rel", g, groups=g["subject"]).fit(reml=True)
        out.append({"stage": stage, "channel": ch, "random_effects": kind,
                    "n_subjects": g["subject"].nunique(), "n_epochs": len(g),
                    "slope_per_night": fit.params["t_rel"], "se": fit.bse["t_rel"],
                    "p": fit.pvalues["t_rel"], "converged": bool(fit.converged)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- plot
def plot_curve(clean: pd.DataFrame, path: Path, n_bins: int = 10, min_n: int = 3) -> None:
    c = clean.copy()
    c["bin"] = np.minimum((c["t_rel"] * n_bins).astype(int), n_bins - 1)
    g = (c.groupby(["subject", "stage", "bin"])["lz_norm"]
           .agg(["mean", "size"]).reset_index())
    g = g[g["size"] >= min_n]
    s = g.groupby(["stage", "bin"])["mean"].agg(["mean", "sem", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for st in STAGES:
        z = s[s["stage"] == st]
        if z.empty:
            continue
        x = (z["bin"] + 0.5) / n_bins
        ax.errorbar(x, z["mean"], yerr=z["sem"], marker="o", ms=3, capsize=2, label=st)
    ax.set_xlabel("relative time in sleep period (0 = onset, 1 = final sleep epoch)")
    ax.set_ylabel("LZ complexity (normalised)")
    ax.set_title(f"LZC across the night (mean ± SEM across subjects; bins with >= {min_n} epochs/subject)")
    ax.title.set_fontsize(9)
    ax.legend(title="stage", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RES)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <results-dir>/night_change")
    ap.add_argument("--split", choices=["halves", "thirds"], default="halves")
    ap.add_argument("--min-epochs", type=int, default=10, help="clean epochs required in BOTH periods")
    ap.add_argument("--persist-run", type=int, default=10, dest="onset_run",
                    help="consecutive sleep epochs defining persistent sleep onset AND offset (10 = 5 min)")
    ap.add_argument("--mixed", action="store_true", help="also fit epoch-level mixed models (slower)")
    args = ap.parse_args()
    out = args.out_dir or (args.results_dir / "night_change")
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.results_dir / "lz_epochs_all.csv", dtype={"subject": str})
    per = sleep_periods(d, args.onset_run)
    print(f"{len(per)}/{d['subject'].nunique()} subjects with a persistent sleep onset")
    per.to_csv(out / "sleep_periods.csv", index=False)

    d = d.merge(per[["subject", "onset_sec", "end_sec"]], on="subject")
    d["t_rel"] = (d["start_sec"] - d["onset_sec"]) / (d["end_sec"] - d["onset_sec"])
    d = d[(d["t_rel"] >= 0) & (d["t_rel"] < 1)]
    if args.split == "halves":
        d["period"] = np.where(d["t_rel"] < 0.5, "early", "late")
    else:
        d["period"] = np.select([d["t_rel"] < 1 / 3, d["t_rel"] >= 2 / 3], ["early", "late"], default="")

    clean = d[(~d["artifact"]) & d["lz_norm"].notna() & d["stage"].notna()]

    # subject x stage x channel early/late means
    sel = clean[clean["period"].isin(["early", "late"])]
    agg = (sel.groupby(["subject", "stage", "channel", "period"])["lz_norm"]
              .agg(["mean", "size"]).unstack("period"))
    agg.columns = [f"{'n' if a == 'size' else 'mean'}_{b}" for a, b in agg.columns]
    sub = agg.reset_index()
    ok = (sub["n_early"] >= args.min_epochs) & (sub["n_late"] >= args.min_epochs)
    sub["delta"] = np.where(ok, sub["mean_late"] - sub["mean_early"], np.nan)
    sub.to_csv(out / "night_change_subject.csv", index=False)

    grp = group_tests(sub)
    grp.insert(0, "split", args.split)
    grp.to_csv(out / "night_change_group.csv", index=False)
    dep = sws_dependence(sub, per)
    dep.to_csv(out / "night_change_sws_dependence.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4g}".format)
    print(f"\n== H1: late - early LZ, {args.split}, >= {args.min_epochs} clean epochs per period ==")
    print(grp[["stage", "channel", "n_subjects", "mean_early", "mean_late", "delta_mean",
               "cohen_dz", "p_t_holm", "p_wilcoxon_holm"]].to_string(index=False))
    print("\n== H2: delta vs total SWS minutes (adjusted for sleep-period length) ==")
    print(dep[["stage", "channel", "n_subjects", "spearman_rho", "p_spearman",
               "beta_sws_per_min", "p_adj"]].to_string(index=False))

    if args.mixed:
        mm = mixed_models(clean)
        mm.to_csv(out / "night_change_mixed.csv", index=False)
        print("\n== epoch-level mixed model: lz_norm ~ t_rel ==")
        print(mm.to_string(index=False))

    plot_curve(clean, out / "night_change_curve.png")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
