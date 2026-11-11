#!/usr/bin/env python
"""H3: does the early-to-late-night change in LZC (delta, 'H') relate to cognition at Visit 9?

Input  : night_change_subject.csv (from lz_night_change.py) with the added columns
         PREMMSE, V9TENG, V8AGE, nsrr_ahi_hp3r_aasm15, nsrr_ahi_hp4u_aasm15
         (+ sleep_periods.csv in the same folder, used for the optional M3 covariates)
Output : <night-dir>/h3/cut80/ and <night-dir>/h3/cut88/  h3_continuous.csv, h3_binary.csv, h3_analysis_dataset.csv
         (the continuous-outcome results do not depend on the cutoff; only the binary ones do)

Predictors : delta (late - early) for every stage x channel, z-scored within each model.
Cutoffs    : binary outcome = V9TENG <= cut. 80 is the standard (SOF Visit 9 manual: "80 or less" is the screening
             cut point); 88 is run for comparison (--teng-cut 80 88). The first value is the standard one.
Outcomes   : (1) PRIMARY  continuous V9TENG, adjusted for covariates (OLS, HC3 robust SEs)
             (2) SECONDARY binary 'impairment proxy' at Visit 9 among baseline-normal women
Terms      : linear delta ('attenuated' = shifted) and delta + delta^2 joint test ('abnormal' in
             either direction)
Covariate sets (nested):
             M1 age + baseline MMSE
             M2 M1 + log(1+AHI)
             M3 M2 + SWS minutes + sleep-period length   (does delta add beyond SWS volume?)

NOTE: this is not a clinical MCI diagnosis. No adjudication, functional status or education
is available, so the binary outcome is a cognitive-score proxy and education is unadjusted.

Usage
  uv run --frozen python lz_h3.py
  uv run --frozen python lz_h3.py --teng-cut 80 88 --ahi hp4u --adjust-early
  uv run --frozen python lz_h3.py --mci-def z --z-cut -1.5     # age-adjusted z definition instead of a fixed cut
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

DEFAULT_NIGHT = Path.home() / "pl1358-01" / "project_data" / "worksof" / "lz_results" / "night_change"
AHI_COLS = {"hp3r": "nsrr_ahi_hp3r_aasm15", "hp4u": "nsrr_ahi_hp4u_aasm15"}  # the two columns add_AHI_scores.py writes
DEFAULT_CUTS = (80.0, 88.0)  # first = standard, rest = comparison


def decline_flag(score, cut):
    """Decline = Visit 9 3MS score at or below the cutoff (the SOF manual's 'X or less' screening rule)."""
    return score <= cut


def cut_dir(base: Path, cut: float) -> Path:
    return base / f"cut{cut:g}"


def zs(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=1)


# ------------------------------------------------------------------------- data
def load_wide(args) -> tuple[pd.DataFrame, list[str]]:
    ahi = AHI_COLS[args.ahi]
    sub = pd.read_csv(args.night_dir / args.subject_file, dtype={"subject": str})
    sub["subject"] = sub["subject"].str.zfill(5)
    need = ["PREMMSE", "V9TENG", "V8AGE", ahi]
    missing = [c for c in need if c not in sub.columns]
    if missing:
        raise SystemExit(f"{args.subject_file} lacks columns: {missing}")

    nun = sub.groupby("subject")[need].nunique(dropna=True).max()
    if (nun > 1).any():
        print(f"WARNING: covariates vary within subject: {nun[nun > 1].to_dict()}")
    cov = sub.groupby("subject")[need].first()

    wide = sub.set_index(["subject", "stage", "channel"])[["delta", "mean_early"]].unstack(["stage", "channel"])
    wide.columns = [re.sub(r"\W", "_", f"{'d' if v == 'delta' else 'e'}_{s}_{c}") for v, s, c in wide.columns]
    df = cov.join(wide).reset_index()

    sp = args.night_dir / "sleep_periods.csv"
    if sp.exists():
        per = pd.read_csv(sp, dtype={"subject": str})
        df = df.merge(per[["subject", "sws_min_total", "sp_min"]], on="subject", how="left")
    preds = [c for c in df.columns if c.startswith("d_")]
    return df.rename(columns={ahi: "AHI"}), preds


def define_outcome(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()
    if args.mci_def == "fixed":
        df["case"] = decline_flag(df["V9TENG"], args.teng_cut).astype(int)
        label = f"V9TENG <= {args.teng_cut:g}"
    else:
        fit = smf.ols("V9TENG ~ V8AGE", df).fit()
        df["teng_age_z"] = fit.resid / np.sqrt(fit.scale)
        df["case"] = (df["teng_age_z"] <= args.z_cut).astype(int)
        label = f"age-adjusted V9TENG z <= {args.z_cut}"
    print(f"binary outcome: {label}  ->  {df['case'].sum()} cases / {len(df)} "
          f"({100 * df['case'].mean():.1f}%)")
    return df


# ------------------------------------------------------------------------ models
def _rhs(covs: list[str], adjust_early: bool) -> str:
    return " + ".join(covs + (["ez"] if adjust_early else []))


def fit_continuous(d: pd.DataFrame, covs: list[str], ae: bool) -> dict:
    rhs = _rhs(covs, ae)
    base = smf.ols(f"V9TENG ~ {rhs}", d).fit()
    lin = smf.ols(f"V9TENG ~ dz + {rhs}", d).fit(cov_type="HC3")
    quad = smf.ols(f"V9TENG ~ dz + dz2 + {rhs}", d).fit(cov_type="HC3")
    ci = lin.conf_int().loc["dz"]
    return {"n": int(lin.nobs), "beta_per_sd": lin.params["dz"], "se": lin.bse["dz"],
            "ci95_lo": ci[0], "ci95_hi": ci[1], "p": lin.pvalues["dz"],
            "r2_base": base.rsquared, "delta_r2": lin.rsquared - base.rsquared,
            "beta_quad": quad.params["dz2"],
            "p_joint_lin_quad": float(quad.wald_test("dz = 0, dz2 = 0", scalar=True).pvalue)}


def fit_binary(d: pd.DataFrame, covs: list[str], ae: bool) -> dict:
    rhs = _rhs(covs, ae)
    n_par = len(covs) + int(ae) + 1
    out = {"n": len(d), "events": int(d["case"].sum()), "epv": d["case"].sum() / n_par}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb = smf.logit(f"case ~ {rhs}", d).fit(disp=0, maxiter=200)
            lf = smf.logit(f"case ~ dz + {rhs}", d).fit(disp=0, maxiter=200)
            lq = smf.logit(f"case ~ dz + dz2 + {rhs}", d).fit(disp=0, maxiter=200)
    except Exception as exc:  # separation / singular fits
        out["note"] = type(exc).__name__
        return out
    ci = np.exp(lf.conf_int().loc["dz"])
    out.update({"or_per_sd": float(np.exp(lf.params["dz"])), "or_lo": ci[0], "or_hi": ci[1],
                "p_wald": lf.pvalues["dz"],
                "p_lr": stats.chi2.sf(2 * (lf.llf - lb.llf), 1),
                "p_joint_lin_quad": stats.chi2.sf(2 * (lq.llf - lb.llf), 2),
                "converged": bool(lf.mle_retvals.get("converged", True))})
    return out


def holm(res: pd.DataFrame, pcols: list[str]) -> pd.DataFrame:
    for pc in pcols:
        res[f"{pc}_holm"] = np.nan
        for _, idx in res.groupby("model_set").groups.items():
            p = res.loc[idx, pc]
            ok = p.notna()
            if ok.sum():
                res.loc[p[ok].index, f"{pc}_holm"] = multipletests(p[ok], method="holm")[1]
    return res


# -------------------------------------------------------------------------- main
def analyse(core: pd.DataFrame, preds: list[str], args, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = define_outcome(core, args)
    df["c_age"] = zs(df["V8AGE"])
    df["c_mmse"] = zs(df["PREMMSE"])
    df["c_ahi"] = zs(np.log1p(df["AHI"]))
    has_sws = "sws_min_total" in df.columns
    if has_sws:
        df["c_sws"] = zs(df["sws_min_total"])
        df["c_sp"] = zs(df["sp_min"])
    sets = {"M1": ["c_age", "c_mmse"], "M2": ["c_age", "c_mmse", "c_ahi"]}
    if has_sws:
        sets["M3"] = sets["M2"] + ["c_sws", "c_sp"]
    df.to_csv(out / "h3_analysis_dataset.csv", index=False)

    # ---- models
    cont, binr = [], []
    for sname, covs in sets.items():
        for pred in preds:
            early = "e_" + pred[2:]
            need = ["V9TENG", pred] + covs + ([early] if args.adjust_early else [])
            d = df.dropna(subset=need).copy()
            if len(d) < args.min_n:
                continue
            d["dz"] = zs(d[pred])
            d["dz2"] = d["dz"] ** 2
            if args.adjust_early:
                d["ez"] = zs(d[early])
            key = {"model_set": sname, "predictor": pred[2:]}
            cont.append({**key, **fit_continuous(d, covs, args.adjust_early)})
            if d["case"].sum() >= args.min_events:
                binr.append({**key, **fit_binary(d, covs, args.adjust_early)})

    cont = holm(pd.DataFrame(cont), ["p", "p_joint_lin_quad"])
    cont.to_csv(out / "h3_continuous.csv", index=False)
    print("\n== PRIMARY: V9TENG ~ delta (per 1 SD) + covariates; OLS, HC3 ==")
    print(cont[["model_set", "predictor", "n", "beta_per_sd", "ci95_lo", "ci95_hi", "p", "p_holm",
                "delta_r2", "p_joint_lin_quad", "p_joint_lin_quad_holm"]].to_string(index=False))

    if binr:
        binr = holm(pd.DataFrame(binr), ["p_lr", "p_joint_lin_quad"])
        binr["low_epv"] = binr["epv"] < 10
        binr.to_csv(out / "h3_binary.csv", index=False)
        print("\n== SECONDARY: impairment proxy ~ delta (per 1 SD) + covariates; logistic ==")
        print(binr[["model_set", "predictor", "n", "events", "epv", "or_per_sd", "or_lo", "or_hi",
                    "p_lr", "p_lr_holm", "p_joint_lin_quad", "low_epv"]].to_string(index=False))
        if binr["low_epv"].any():
            print("\nWARNING: events-per-variable < 10 in some models; treat odds ratios as unstable.")
    else:
        print(f"\nbinary models skipped: fewer than {args.min_events} cases")
    print(f"\nwritten to {out}")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night-dir", type=Path, default=DEFAULT_NIGHT)
    ap.add_argument("--subject-file", default="night_change_subject.csv")
    ap.add_argument("--ahi", choices=list(AHI_COLS), default="hp3r")
    ap.add_argument("--baseline-min", type=float, default=24,
                    help="exclude baseline MMSE below this (0 = keep all); default 24")
    ap.add_argument("--mci-def", choices=["z", "fixed"], default="fixed",
                    help="z: age-adjusted V9TENG z-score <= --z-cut; fixed: V9TENG < --teng-cut")
    ap.add_argument("--z-cut", type=float, default=-1.5)
    ap.add_argument("--teng-cut", type=float, nargs="+", default=list(DEFAULT_CUTS),
                    help="cutoff(s) for the binary outcome (decline = V9TENG <= cut); first = standard (80), "
                         "others = comparison (88). Ignored with --mci-def z")
    ap.add_argument("--adjust-early", action="store_true",
                    help="also adjust for early-night LZ of the same stage/channel (separates change from level)")
    ap.add_argument("--min-n", type=int, default=60, help="minimum subjects to fit a model")
    ap.add_argument("--min-events", type=int, default=10, help="minimum cases to fit the logistic model")
    args = ap.parse_args()
    base_out = args.night_dir / "h3"
    base_out.mkdir(parents=True, exist_ok=True)

    df, preds = load_wide(args)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.4g}".format)

    # ---- cohort flow, attrition, scale check
    print(f"subjects in file: {len(df)}")
    df["has_v9"] = df["V9TENG"].notna()
    print("\nattrition check (means by availability of V9TENG):")
    print(df.groupby("has_v9")[[c for c in ["V8AGE", "PREMMSE", "AHI", "sws_min_total"] if c in df]]
            .agg(["mean", "count"]).T.to_string())
    core = df.dropna(subset=["V9TENG", "V8AGE", "PREMMSE"])
    print(f"\nwith V9TENG, age, baseline MMSE: {len(core)}")
    if args.baseline_min > 0:
        core = core[core["PREMMSE"] >= args.baseline_min]
        print(f"after excluding baseline MMSE < {args.baseline_min:g}: {len(core)}")
    print(f"ranges  PREMMSE {core.PREMMSE.min():g}-{core.PREMMSE.max():g}   "
          f"V9TENG {core.V9TENG.min():g}-{core.V9TENG.max():g}   "
          f"age {core.V8AGE.min():g}-{core.V8AGE.max():g}")
    if core["V9TENG"].max() > 30:
        print("NOTE: V9TENG is on a different scale from MMSE (0-30), so change scores are NOT "
              "computed; baseline MMSE enters as a covariate.")

    if args.mci_def == "z":
        a = argparse.Namespace(**vars(args))
        analyse(core, preds, a, base_out / "z")
    else:
        for i, cut in enumerate(args.teng_cut):
            a = argparse.Namespace(**vars(args))
            a.teng_cut = cut
            print(f"\n{'#' * 70}\n# cutoff {cut:g} ({'standard' if i == 0 else 'comparison'})\n{'#' * 70}")
            analyse(core, preds, a, cut_dir(base_out, cut))


if __name__ == "__main__":
    main()
