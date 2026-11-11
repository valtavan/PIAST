#!/usr/bin/env python
"""Positive control: do conventional sleep measures predict the Visit 9 Teng score?

Keep this file next to lz_h3.py, lz_h3_firth_perm.py and lz_night_change.py (it imports from them).

Predictors (per subject; all z-scored, so effects are 'per 1 SD')
    SWS      sws_min (slow-wave minutes in the sleep period), sws_pct (% of total sleep time)
    Awakening n_awak (wake runs >= --min-wake-epochs inside the sleep period), awak_per_h (per hour of sleep),
             waso_min (all wake minutes after onset)
    AHI      ahi_hp3r (nsrr_ahi_hp3r_aasm15) and ahi_hp4u (nsrr_ahi_hp4u_aasm15: >=30% flow reduction + >=4% desaturation, no arousal criterion)
    Reference age and baseline MMSE, to show how sensitive this outcome is to a known strong predictor
    Counts, WASO and AHI enter as log(1 + x) because they are right-skewed.

Outcomes (baseline MMSE >= --baseline-min, with a Visit 9 score)
    continuous V9TENG  OLS, robust (HC3) SEs
    decline            V9TENG <= cut vs stable; Firth logistic regression (cut 80 standard, 88 comparison:
                      --teng-cut 80 88; one output folder per cut)

Models
    1  one predictor at a time, adjusted for age + baseline MMSE (also unadjusted), with a quadratic check
    2  mutually adjusted: SWS + awakenings + AHI together (+ age + MMSE), VIFs, joint block test

Outputs in <night-dir>/positive_control/cut80/ and cut88/ (csv + png).

Usage
  uv run --frozen python lz_positive_control.py
  uv run --frozen python lz_positive_control.py --sws-var sws_pct --awak-var n_awak --ahi hp4u
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

from lz_h3 import DEFAULT_CUTS, DEFAULT_NIGHT, cut_dir, decline_flag, zs
from lz_h3_firth_perm import firth_dz, firth_fit
from lz_night_change import DEFAULT_RES, EPOCH_SEC, sleep_periods

# own copy, so this script does not depend on which version of lz_h3.py is installed
AHI_COLS = {"hp3r": "nsrr_ahi_hp3r_aasm15", "hp4u": "nsrr_ahi_hp4u_aasm15"}

# name -> (label, transform)
SLEEP_PREDS = {
    "sws_min": ("SWS minutes", None),
    "sws_pct": ("SWS % of sleep", None),
    "n_awak": ("Awakenings (count)", "log1p"),
    "awak_per_h": ("Awakenings per hour", "log1p"),
    "waso_min": ("WASO minutes", "log1p"),
    "ahi_hp3r": ("AHI hp3r (3% or arousal)", "log1p"),
    "ahi_hp4u": ("AHI hp4u (4% desat)", "log1p"),
}
REF_PREDS = {"V8AGE": ("Age (reference)", None, ["c_mmse"]), "PREMMSE": ("Baseline MMSE (reference)", None, ["c_age"])}
COVS = ["c_age", "c_mmse"]


# ================================================================== sleep measures
def sleep_measures(ep: pd.DataFrame, per: pd.DataFrame, min_wake: int) -> pd.DataFrame:
    eg = {k: v for k, v in ep.groupby("subject")}
    rows = []
    for r in per.itertuples():
        g = eg[r.subject]
        g = g[(g["start_sec"] >= r.onset_sec) & (g["start_sec"] < r.end_sec)].sort_values("start_sec")
        # wake runs on the 30 s time grid: absent (unscored) epochs are NOT wake and break a run,
        # so two wake bouts separated by a gap are not merged into one awakening
        k = np.rint(g["start_sec"].to_numpy(float) / EPOCH_SEC).astype(int)
        wake = np.zeros(k.max() - k.min() + 1, dtype=int) if len(k) else np.zeros(0, dtype=int)
        if len(k):
            wake[k - k.min()] = (g["stage_code"].to_numpy() == 0)
        edges = np.diff(np.r_[0, wake, 0])
        lens = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        rows.append({"subject": r.subject, "sws_min": r.sws_min_total, "tst_min": r.tst_min, "sp_min": r.sp_min,
                     "n_awak": int((lens >= min_wake).sum()), "waso_min": wake.sum() * EPOCH_SEC / 60})
    out = pd.DataFrame(rows)
    out["sws_pct"] = 100 * out["sws_min"] / out["tst_min"]
    out["awak_per_h"] = out["n_awak"] / (out["tst_min"] / 60)
    return out


def find_ahi_columns(cols, args) -> dict[str, str]:
    """Locate the AHI columns; tolerate a missing hp4u column or a slightly different name."""
    cols = list(cols)
    print("columns mentioning AHI in the file:", [c for c in cols if "ahi" in c.lower()])
    found = {}
    for key, override in (("hp3r", args.ahi_hp3r_col), ("hp4u", args.ahi_hp4u_col)):
        if override:
            if override not in cols:
                raise SystemExit(f"--ahi-{key}-col {override!r} is not a column of {args.subject_file}")
            found[key] = override
        elif AHI_COLS[key] in cols:
            found[key] = AHI_COLS[key]
        else:
            cand = [c for c in cols if re.search(rf"ahi.*{key}", c, re.I)]
            if len(cand) == 1:
                found[key] = cand[0]
                print(f"  using {cand[0]!r} for AHI {key}")
            elif len(cand) > 1:
                raise SystemExit(f"several candidate columns for AHI {key}: {cand}; pass --ahi-{key}-col")
            else:
                print(f"WARNING: no column found for AHI {key}; continuing without it")
    if args.ahi not in found:
        raise SystemExit(f"--ahi {args.ahi} requested but no matching column; "
                         f"pass --ahi-{args.ahi}-col or choose --ahi with an available column")
    return found


def load_cohort(args, meas: pd.DataFrame):
    raw = pd.read_csv(args.night_dir / args.subject_file, dtype={"subject": str})
    raw["subject"] = raw["subject"].str.zfill(5)
    core = ["PREMMSE", "V9TENG", "V8AGE"]
    miss = [c for c in core if c not in raw.columns]
    if miss:
        raise SystemExit(f"{args.subject_file} lacks columns {miss}")
    ahi_cols = find_ahi_columns(raw.columns, args)
    for key in ("hp3r", "hp4u"):
        if key not in ahi_cols:
            SLEEP_PREDS.pop(f"ahi_{key}", None)
    cov = raw.groupby("subject")[core + list(ahi_cols.values())].first().rename(
        columns={v: f"ahi_{k}" for k, v in ahi_cols.items()})
    df = cov.join(meas.set_index("subject"), how="left").reset_index()
    n_all = len(df)
    df = df.dropna(subset=["V8AGE", "PREMMSE"])
    df = df[df["PREMMSE"] >= args.baseline_min].copy()
    n_base = len(df)
    df["c_age"], df["c_mmse"] = zs(df["V8AGE"]), zs(df["PREMMSE"])
    for name, (_, tf) in SLEEP_PREDS.items():
        x = df[name].astype(float)
        df[f"z_{name}"] = zs(np.log1p(x) if tf == "log1p" else x)
    df["z_V8AGE"], df["z_PREMMSE"] = df["c_age"], df["c_mmse"]
    fu = df[df["V9TENG"].notna()].copy()
    fu["case"] = decline_flag(fu["V9TENG"], args.teng_cut).astype(float)
    print(f"subjects in file {n_all} -> baseline-normal {n_base} -> with V9TENG {len(fu)} "
          f"({int(fu['case'].sum())} decline <= {args.teng_cut:g}, {int((1 - fu['case']).sum())} stable)")
    print(f"V9TENG range {fu.V9TENG.min():g}-{fu.V9TENG.max():g}, mean {fu.V9TENG.mean():.1f}, SD {fu.V9TENG.std():.1f}")
    return df, fu


# ==================================================================== model 1
def single_models(fu: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    specs = {n: (lab, COVS) for n, (lab, _) in SLEEP_PREDS.items()}
    specs.update({n: (lab, cv) for n, (lab, _, cv) in REF_PREDS.items()})
    for name, (label, covs) in specs.items():
        d = fu.dropna(subset=[f"z_{name}", "V9TENG"] + covs).copy()
        d["z"], d["z2"] = d[f"z_{name}"], d[f"z_{name}"] ** 2
        y = d["V9TENG"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = sm.OLS(y, sm.add_constant(d[covs])).fit()
            un = sm.OLS(y, sm.add_constant(d[["z"]])).fit(cov_type="HC3")
            adj = sm.OLS(y, sm.add_constant(d[["z"] + covs])).fit(cov_type="HC3")
            quad = sm.OLS(y, sm.add_constant(d[["z", "z2"] + covs])).fit(cov_type="HC3")
            X = np.column_stack([np.ones(len(d)), d["z"], d[covs]])
            _, fr = firth_dz(X, d["case"].to_numpy(float))
        ci = adj.conf_int().loc["z"]
        rows.append({"predictor": name, "label": label, "reference": name in REF_PREDS, "n": len(d),
                     "events": int(d["case"].sum()), "beta_unadj": un.params["z"], "p_unadj": un.pvalues["z"],
                     "beta_adj": adj.params["z"], "se": adj.bse["z"], "ci_lo": ci[0], "ci_hi": ci[1],
                     "p_adj": adj.pvalues["z"], "delta_r2": adj.rsquared - base.rsquared,
                     "mde_80pct_power": 2.8 * adj.bse["z"],
                     "p_joint_lin_quad": float(quad.wald_test("z = 0, z2 = 0", scalar=True).pvalue),
                     "or_decline": fr["or"], "or_lo": fr["or_lo"], "or_hi": fr["or_hi"],
                     "p_firth_lr": fr["p_firth_lr"]})
    res = pd.DataFrame(rows)
    s = ~res["reference"]
    for pc in ["p_adj", "p_firth_lr", "p_joint_lin_quad"]:
        res[pc + "_holm"] = np.nan
        res.loc[s, pc + "_holm"] = multipletests(res.loc[s, pc], method="holm")[1]
    return res


# ==================================================================== model 2
def mutual_model(fu: pd.DataFrame, args):
    ahi = f"ahi_{args.ahi}"
    preds = [args.sws_var, args.awak_var, ahi]
    zc = [f"z_{p}" for p in preds]
    d = fu.dropna(subset=zc + COVS + ["V9TENG"]).copy()
    names = {z: p for z, p in zip(zc, preds)}
    X = sm.add_constant(d[zc + COVS])
    out, extra = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sm.OLS(d["V9TENG"], sm.add_constant(d[COVS])).fit()
        full = sm.OLS(d["V9TENG"], X).fit(cov_type="HC3")
        block = full.wald_test(", ".join(f"{z} = 0" for z in zc), scalar=True)
        vif = {c: variance_inflation_factor(X.to_numpy(), i) for i, c in enumerate(X.columns) if c != "const"}
        ci = full.conf_int()
        for z in zc:
            out.append({"outcome": "V9TENG (points per SD)", "predictor": names[z], "estimate": full.params[z],
                        "ci_lo": ci.loc[z, 0], "ci_hi": ci.loc[z, 1], "p": full.pvalues[z], "vif": vif[z]})
        extra["ols"] = {"n": len(d), "delta_r2_block": full.rsquared - base.rsquared,
                        "p_block": float(block.pvalue), "r2_full": full.rsquared, "r2_base": base.rsquared}
        y = d["case"].to_numpy(float)
        Xb = np.column_stack([np.ones(len(d)), d[zc].to_numpy(), d[COVS].to_numpy()])
        fb = firth_fit(Xb, y)
        rb = firth_fit(Xb, y, fixed={1: 0.0, 2: 0.0, 3: 0.0})
        se = np.sqrt(np.diag(fb["cov"]))
        for i, z in enumerate(zc, start=1):
            b = fb["beta"][i]
            out.append({"outcome": "decline (OR per SD)", "predictor": names[z], "estimate": np.exp(b),
                        "ci_lo": np.exp(b - 1.96 * se[i]), "ci_hi": np.exp(b + 1.96 * se[i]),
                        "p": 2 * stats.norm.sf(abs(b / se[i])), "vif": vif[z]})
        extra["firth"] = {"n": len(d), "events": int(y.sum()),
                          "p_block_lr": stats.chi2.sf(max(2 * (fb["pll"] - rb["pll"]), 0), 3)}
    return pd.DataFrame(out), extra, preds


# ===================================================================== plots
def plot_forest(res: pd.DataFrame, cut: float, path: Path) -> None:
    r = res.iloc[::-1].reset_index(drop=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    y = np.arange(len(r))
    colr = np.where(r["reference"], "#9D9D9D", "#4C78A8")
    for i, row in r.iterrows():
        a.errorbar(row["beta_adj"], i, xerr=[[row["beta_adj"] - row["ci_lo"]], [row["ci_hi"] - row["beta_adj"]]],
                   fmt="o", color=colr[i], capsize=2, ms=5)
        b.errorbar(row["or_decline"], i, xerr=[[row["or_decline"] - row["or_lo"]], [row["or_hi"] - row["or_decline"]]],
                   fmt="o", color=colr[i], capsize=2, ms=5)
        a.text(1.01, i, f"p={row['p_adj']:.3f}", transform=a.get_yaxis_transform(), fontsize=7, va="center")
        b.text(1.01, i, f"p={row['p_firth_lr']:.3f}", transform=b.get_yaxis_transform(), fontsize=7, va="center")
    a.axvline(0, color="k", ls="--", lw=0.8)
    b.axvline(1, color="k", ls="--", lw=0.8)
    b.set_xscale("log")
    a.set_yticks(y)
    a.set_yticklabels(r["label"])
    a.set_xlabel("V9TENG points per 1 SD (adjusted for age, baseline MMSE)")
    b.set_xlabel(f"odds ratio for decline (<={cut:g}) per 1 SD (Firth)")
    a.set_title("Continuous Teng score", fontsize=10)
    b.set_title("Decline vs stable", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_quintiles(fu: pd.DataFrame, names: list[str], cut: float, path: Path) -> None:
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 6.5), squeeze=False)
    for j, name in enumerate(names):
        d = fu.dropna(subset=[name, "V9TENG"] + COVS).copy()
        d["res"] = sm.OLS(d["V9TENG"], sm.add_constant(d[COVS])).fit().resid
        d["q"] = pd.qcut(d[name].rank(method="first"), 5, labels=False)
        g = d.groupby("q")
        xm = g[name].median()
        m, se = g["res"].mean(), g["res"].sem()
        axes[0, j].errorbar(range(len(m)), m, yerr=se, marker="o", capsize=3, color="#4C78A8")
        axes[0, j].axhline(0, color="k", lw=0.6, ls=":")
        axes[0, j].set_ylabel("V9TENG residual (age, MMSE removed)")
        k, n = g["case"].sum(), g["case"].size()
        lo, hi = proportion_confint(k, n, method="wilson")
        p = k / n
        axes[1, j].errorbar(range(len(p)), 100 * p, yerr=[100 * (p - lo), 100 * (hi - p)], marker="o", capsize=3,
                            color="#E45756")
        axes[1, j].set_ylabel(f"% decline (<={cut:g})")
        for ax in axes[:, j]:
            ax.set_xticks(range(len(xm)))
            ax.set_xticklabels([f"{v:.3g}" for v in xm], fontsize=8)
            ax.set_xlabel(f"{SLEEP_PREDS[name][0]} (quintile median)")
    fig.suptitle("Dose-response by quintile (mean ± SEM; Wilson 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ======================================================================= main
def run(args, meas: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df, fu = load_cohort(args, meas)
    df.to_csv(out / "positive_control_dataset.csv", index=False)

    cols = [c for c in SLEEP_PREDS]
    cor = fu[cols].corr(method="spearman")
    cor.to_csv(out / "predictor_spearman.csv")
    print("\nSpearman correlations between predictors (followed women):")
    print(cor.round(2).to_string())

    res = single_models(fu, args)
    res.to_csv(out / "positive_control_single.csv", index=False)
    print("\n== MODEL 1: one predictor at a time, adjusted for age + baseline MMSE (per 1 SD) ==")
    print(res[["label", "n", "events", "beta_unadj", "beta_adj", "ci_lo", "ci_hi", "p_adj", "p_adj_holm",
               "delta_r2", "mde_80pct_power", "p_joint_lin_quad", "or_decline", "or_lo", "or_hi", "p_firth_lr",
               "p_firth_lr_holm"]].to_string(index=False))
    print("(mde_80pct_power = smallest |beta| detectable with 80% power at alpha 0.05, = 2.8 x SE)")

    mut, extra, preds = mutual_model(fu, args)
    mut.to_csv(out / "positive_control_mutual.csv", index=False)
    print(f"\n== MODEL 2: {', '.join(preds)} together, + age + baseline MMSE ==")
    print(mut.to_string(index=False))
    print(f"OLS n={extra['ols']['n']}: block of 3 adds R2 {extra['ols']['delta_r2_block']:.4f} "
          f"(base {extra['ols']['r2_base']:.3f} -> {extra['ols']['r2_full']:.3f}), joint p = {extra['ols']['p_block']:.4f}")
    print(f"Firth n={extra['firth']['n']}, events={extra['firth']['events']}: block LR p = {extra['firth']['p_block_lr']:.4f}")

    plot_forest(res, args.teng_cut, out / "fig1_forest_positive_control.png")
    plot_quintiles(fu, preds, args.teng_cut, out / "fig2_quintiles.png")
    print(f"\nwritten to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RES)
    ap.add_argument("--night-dir", type=Path, default=DEFAULT_NIGHT)
    ap.add_argument("--subject-file", default="night_change_subject.csv")
    ap.add_argument("--persist-run", type=int, default=10)
    ap.add_argument("--min-wake-epochs", type=int, default=2, help="wake run length that counts as an awakening (2 = 1 min)")
    ap.add_argument("--teng-cut", type=float, nargs="+", default=list(DEFAULT_CUTS),
                    help="decline = V9TENG <= cut; first value = standard (80), others = comparison (88)")
    ap.add_argument("--baseline-min", type=float, default=24)
    ap.add_argument("--sws-var", choices=["sws_min", "sws_pct"], default="sws_min")
    ap.add_argument("--awak-var", choices=["n_awak", "awak_per_h", "waso_min"], default="awak_per_h")
    ap.add_argument("--ahi", choices=["hp3r", "hp4u"], default="hp3r")
    ap.add_argument("--ahi-hp3r-col", default=None, help="exact column name if it differs from nsrr_ahi_hp3r_aasm15")
    ap.add_argument("--ahi-hp4u-col", default=None, help="exact column name if it differs from nsrr_ahi_hp4u_aasm15")
    args = ap.parse_args()
    base_out = args.night_dir / "positive_control"
    base_out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.4g}".format)

    ep = pd.read_csv(args.results_dir / "lz_epochs_all.csv", dtype={"subject": str},
                     usecols=["subject", "epoch", "start_sec", "stage_code"]).drop_duplicates(["subject", "epoch"])
    per = sleep_periods(ep, args.persist_run)
    meas = sleep_measures(ep, per, args.min_wake_epochs)
    for i, cut in enumerate(args.teng_cut):
        a = argparse.Namespace(**vars(args))
        a.teng_cut = cut
        print(f"\n{'#' * 70}\n# cutoff {cut:g} ({'standard' if i == 0 else 'comparison'})\n{'#' * 70}")
        run(a, meas, cut_dir(base_out, cut))


if __name__ == "__main__":
    main()
