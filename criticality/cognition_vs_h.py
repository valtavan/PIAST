#!/usr/bin/env python
"""
cognition_vs_h.py - does the level of H(q=+2) (0.5 Hz high-pass, full night) relate to follow-up cognition?

Complements the through-night analysis (which tests the early-to-late CHANGE of H): this one tests the
LEVEL of H in each sleep stage against the continuous follow-up 3MS (V9TENG), and against decline.

Input
  --h-dir   folder of <id>.parquet with the full 0.5 Hz run. Long format (one row per epoch and derivation:
            channel, status, window_start_time_s, h_q+2, delta_h or h_q-4/h_q+4) or the wide format
            (<DERIV>_h0..h9 + window_start_time_s, q order from --wide-q).
  --cog-file  worksof/cognitive.csv  (ID, COHORT, PREMMSE, PRETRB, V9TENG, V8AGE, nsrr_ahi_...)
  --hyp-dir   hypnogram_<id>.csv (start_sec, stage_code); stage is always taken from here
  --spectral-dir  optional; relative delta power per epoch, used as a covariate (model M3)

Exposures (per participant and stage, median over the night's epochs of that stage; >= --min-epochs epochs)
  H        H(q=+2), mean of the four derivations (epochs with all four fits valid)
  delta_h  multifractal width h(q=-4) - h(q=+4), same averaging
Stages: N2, N3 (called SWS in some outputs) and REM by default; N2 / H is the single pre-specified primary test.

Cohort: participants with the exposure, V9TENG and the covariates; baseline PREMMSE > --premmse-min (default 24,
as in the paper; 0 switches it off).

Models (complete cases per stage and measure, so the models are comparable)
  continuous  V9TENG ~ exposure (per 1 SD) + covariates, OLS with HC3 standard errors
      M1 age | M2 age + baseline MMSE + AHI | M3 M2 + relative delta power of that stage (does H add to delta?)
  binary      decline = V9TENG < cutoff (80 and 88), Firth logistic, OR per SD, same covariates (M2, M3), EPV reported
  plus the unadjusted Spearman correlation (bootstrap CI) and the AUC of H for decline.
Holm correction over the stage x measure grid (within model and outcome). The primary test is NOT corrected.
Attrition: exposure in participants with vs without a V9TENG score.
A power note gives the smallest correlation detectable with 80 % power at the achieved n.

Outputs in <h-dir>/cognition_vs_h: participant_table.csv, spearman.csv, continuous_models.csv,
binary_models.csv, attrition.csv, summary.txt, scatter_H_vs_V9TENG.png, forest_continuous.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, rankdata, t as tdist

PROJECT = Path("~/pl1358-01/project_data").expanduser()
CRIT = PROJECT / "worksof/criticality"
STAGE_NAMES = {0: "wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CHANNELS = ["C3A1", "C3A2", "C4A1", "C4A2"]


# ----------------------------------------------------------------------------- loading
def epoch_table(df, fmt, wide_q):
    """-> DataFrame(index window_start_time_s) with H and delta_h averaged over the four derivations."""
    if fmt == "long":
        d = df.copy()
        ok = d["status"] == "ok"
        if "delta_h" not in d:
            d["delta_h"] = d["h_q-4"] - d["h_q+4"]
        d.loc[~ok, ["h_q+2", "delta_h"]] = np.nan
        h = d.pivot(index="window_start_time_s", columns="channel", values="h_q+2")
        dh = d.pivot(index="window_start_time_s", columns="channel", values="delta_h")
        valid = h.notna().all(axis=1) & (h.shape[1] >= 4)
        return pd.DataFrame({"H": h.mean(axis=1), "delta_h": dh.mean(axis=1)})[valid]
    k2, km4, kp4 = wide_q.index(2), wide_q.index(-4), wide_q.index(4)
    hs = np.column_stack([df[f"{c}_h{k2}"] for c in CHANNELS])
    ds = np.column_stack([df[f"{c}_h{km4}"] - df[f"{c}_h{kp4}"] for c in CHANNELS])
    out = pd.DataFrame({"H": hs.mean(axis=1), "delta_h": ds.mean(axis=1)}, index=df["window_start_time_s"].to_numpy())
    return out[np.isfinite(hs).all(axis=1)]


def stage_of(times, hyp_file):
    h = pd.read_csv(hyp_file)
    m = dict(zip(h["start_sec"].round().astype(int), h["stage_code"].astype(int)))
    return pd.Series([STAGE_NAMES.get(m.get(int(round(t)), -1)) for t in times])


def build_participant_table(args):
    files = sorted(args.h_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files in {args.h_dir}")
    first = pd.read_parquet(files[0])
    fmt = "long" if "channel" in first.columns else "wide"
    if fmt == "long":
        need = ["window_start_time_s", "channel", "status", "h_q+2"] + \
               (["delta_h"] if "delta_h" in first else ["h_q-4", "h_q+4"])
    else:
        need = ["window_start_time_s"] + [c for c in first.columns if c.endswith(tuple(f"_h{k}" for k in range(10)))]
    print(f"[load] {len(files)} files in {args.h_dir}, {fmt} format")
    rows = []
    for f in files:
        pid = f.stem
        hyp = args.hyp_dir / f"hypnogram_{pid}.csv"
        if not hyp.exists():
            continue
        e = epoch_table(pd.read_parquet(f, columns=need), fmt, args.wide_q)
        if e.empty:
            continue
        e["stage"] = stage_of(e.index, hyp).to_numpy()
        if args.spectral_dir is not None and (args.spectral_dir / f"{pid}.parquet").exists():
            s = pd.read_parquet(args.spectral_dir / f"{pid}.parquet", columns=["window_start_time_s", "rel_delta"])
            s = s.drop_duplicates("window_start_time_s").set_index("window_start_time_s")["rel_delta"]
            e["rel_delta"] = s.reindex(e.index.to_numpy()).to_numpy()
        else:
            e["rel_delta"] = np.nan
        for stage, g in e.dropna(subset=["stage"]).groupby("stage"):
            rows.append(dict(id=pid, stage=stage, n=len(g), H=g["H"].median(), delta_h=g["delta_h"].median(),
                             rel_delta=g["rel_delta"].median()))
    long = pd.DataFrame(rows)
    return long


def load_cog(args):
    df = pd.read_csv(args.cog_file, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    for need in (args.id_col, args.teng_col, args.age_col):
        if need.lower() not in cols:
            sys.exit(f"{args.cog_file}: no column {need}; columns are {list(df.columns)}")
    out = pd.DataFrame({
        "id": df[cols[args.id_col.lower()]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5),
        "teng": pd.to_numeric(df[cols[args.teng_col.lower()]], errors="coerce"),
        "age": pd.to_numeric(df[cols[args.age_col.lower()]], errors="coerce"),
    })
    out["teng"] = out["teng"].where((out["teng"] >= 0) & (out["teng"] <= 100))
    for name, c in (("mmse", args.premmse_col), ("ahi", args.ahi_col)):
        out[name] = pd.to_numeric(df[cols[c.lower()]], errors="coerce") if c and c.lower() in cols else np.nan
    return out.drop_duplicates("id").set_index("id")


# ----------------------------------------------------------------------------- statistics
def ols_hc3(y, X, j):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    V = XtXi @ (X.T * ((e / (1 - h)) ** 2)) @ X @ XtXi
    se = np.sqrt(np.diag(V))
    tc = tdist.ppf(0.975, n - k)
    r2 = 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))
    return b[j], se[j], b[j] - tc * se[j], b[j] + tc * se[j], 2 * tdist.sf(abs(b[j] / se[j]), n - k), r2


def firth_logit(y, X, j, maxit=200):
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        Ii = np.linalg.inv(X.T @ (X * W[:, None]))
        hat = np.einsum("ij,jk,ik->i", X, Ii, X) * W
        step = Ii @ (X.T @ (y - p + hat * (0.5 - p)))
        m = np.abs(step).max()
        if m > 5:
            step *= 5 / m
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]))))
    z = beta[j] / se[j]
    return np.exp(beta[j]), np.exp(beta[j] - 1.96 * se[j]), np.exp(beta[j] + 1.96 * se[j]), 2 * norm.sf(abs(z))


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman_boot(x, y, rng, n_boot):
    def rho(a, b):
        ra, rb = rankdata(a), rankdata(b)
        return float(np.corrcoef(ra, rb)[0, 1])
    r = rho(x, y)
    n = len(x)
    bs = [rho(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # two-sided p from the t approximation
    tt = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, lo, hi, 2 * tdist.sf(abs(tt), n - 2)


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(order) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def min_detectable_r(n):
    return float(np.tanh((norm.ppf(0.975) + norm.ppf(0.8)) / np.sqrt(max(n - 3, 1))))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h-dir", type=lambda s: Path(s).expanduser(), default=CRIT / "high0.5")
    ap.add_argument("--cog-file", type=lambda s: Path(s).expanduser(), default=PROJECT / "worksof/cognitive.csv")
    ap.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=PROJECT / "worksof/hypnograms/expert")
    ap.add_argument("--spectral-dir", type=lambda s: Path(s).expanduser(), default=CRIT / "spectral")
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--teng-col", default="V9TENG")
    ap.add_argument("--age-col", default="V8AGE")
    ap.add_argument("--premmse-col", default="PREMMSE")
    ap.add_argument("--premmse-min", type=float, default=24.0, help="keep baseline MMSE > this (0 = no filter)")
    ap.add_argument("--ahi-col", default="nsrr_ahi_hp3r_aasm15")
    ap.add_argument("--stages", nargs="*", default=["N2", "N3", "REM"])
    ap.add_argument("--primary-stage", default="N2")
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--cutoffs", nargs="*", type=float, default=[80, 88])
    ap.add_argument("--wide-q", default="-5,-4,-3,-2,-1,1,2,3,4,5")
    ap.add_argument("--out", type=lambda s: Path(s).expanduser(), default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    args.wide_q = [int(x) for x in args.wide_q.split(",")]
    if not args.spectral_dir.is_dir():
        args.spectral_dir = None
        print("[spectral] folder not found: model M3 is skipped")
    out = args.out or (args.h_dir / "cognition_vs_h")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    long = build_participant_table(args)
    cog = load_cog(args)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    ids_all = long["id"].nunique()
    long = long[(long["n"] >= args.min_epochs) & long["stage"].isin(args.stages)]
    eligible = cog.index
    if args.premmse_min and cog["mmse"].notna().any():
        eligible = cog.index[cog["mmse"] > args.premmse_min]
        log(f"cohort: {ids_all} participants with data; baseline {args.premmse_col} > {args.premmse_min:g}: "
            f"{int(long['id'].isin(eligible).groupby(long['id']).first().sum())} of them")
    elif args.premmse_min:
        log(f"[warn] {args.premmse_col} not found: no baseline-MMSE filter applied")
    long = long[long["id"].isin(eligible)]

    wide = long.pivot(index="id", columns="stage", values=["H", "delta_h", "rel_delta", "n"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    P = wide.join(cog, how="left")
    P.to_csv(out / "participant_table.csv")
    log(f"participants with a score: {int(P['teng'].notna().sum())} of {len(P)} (any stage); "
        f"V9TENG bands: <80 {int((P['teng'] < 80).sum())}, 80-88 {int(((P['teng'] >= 80) & (P['teng'] < 88)).sum())}, "
        f">=88 {int((P['teng'] >= 88).sum())}")

    cov_sets = {"M1": ["age"], "M2": ["age", "mmse", "ahi"]}
    if args.spectral_dir is not None:
        cov_sets["M3"] = ["age", "mmse", "ahi", "REL"]
    if not P["mmse"].notna().any():
        cov_sets["M2"] = ["age", "ahi"]
        if "M3" in cov_sets:
            cov_sets["M3"] = ["age", "ahi", "REL"]
    if not P["ahi"].notna().any():
        cov_sets = {k: [c for c in v if c != "ahi"] for k, v in cov_sets.items()}

    sp_rows, ols_rows, bin_rows, att_rows = [], [], [], []
    samples = {}
    for stage in args.stages:
        for meas, label in (("H", "h_q+2"), ("delta_h", "delta_h")):
            xcol = f"{meas}_{stage}"
            if xcol not in P:
                continue
            P["REL"] = P.get(f"rel_delta_{stage}")
            need_all = [xcol, "teng"] + sorted({c for v in cov_sets.values() for c in v})
            if "REL" in need_all and P["REL"].isna().all():
                need_all.remove("REL")
            D = P.dropna(subset=need_all)
            samples[(stage, meas)] = D
            n = len(D)
            has_x = P[xcol].notna()
            a_with, a_without = P.loc[has_x & P["teng"].notna(), xcol], P.loc[has_x & P["teng"].isna(), xcol]
            att_rows.append(dict(stage=stage, measure=label, n_with_v9=len(a_with), n_without_v9=len(a_without),
                                 median_with=a_with.median(), median_without=a_without.median(),
                                 p_mwu=mannwhitneyu(a_with, a_without).pvalue if len(a_with) > 2 and len(a_without) > 2 else np.nan))
            if n < 20:
                log(f"[skip] {stage} {label}: only {n} complete cases")
                continue
            r, lo, hi, p = spearman_boot(D[xcol].to_numpy(), D["teng"].to_numpy(), rng, args.n_boot)
            sp_rows.append(dict(stage=stage, measure=label, n=n, spearman=r, ci_low=lo, ci_high=hi, p=p,
                                min_detectable_r_80pct_power=min_detectable_r(n)))
            z = (D[xcol] - D[xcol].mean()) / D[xcol].std(ddof=1)
            y = D["teng"].to_numpy()
            for mname, covs in cov_sets.items():
                covs = [c for c in covs if c in D and D[c].notna().all() and not (c == "REL" and D[c].isna().all())]
                X = np.column_stack([np.ones(n), z.to_numpy()] + [D[c].to_numpy(float) for c in covs])
                b, se, lo_, hi_, p_, r2 = ols_hc3(y, X, 1)
                r2_red = ols_hc3(y, np.delete(X, 1, axis=1), 0)[5]
                ols_rows.append(dict(stage=stage, measure=label, model=mname, n=n, beta_per_sd=b, ci_low=lo_,
                                     ci_high=hi_, p=p_, delta_r2=r2 - r2_red, covariates="+".join(covs)))
            for cut in args.cutoffs:
                yb = (D["teng"] < cut).to_numpy().astype(float)
                ev = int(yb.sum())
                if ev < 5 or ev > n - 5:
                    continue
                a = auc(D.loc[yb == 1, xcol].to_numpy(), D.loc[yb == 0, xcol].to_numpy())
                for mname in [m for m in cov_sets if m != "M1"]:
                    covs = [c for c in cov_sets[mname] if c in D and D[c].notna().all()]
                    X = np.column_stack([np.ones(n), z.to_numpy()] + [(D[c] - D[c].mean()).to_numpy(float) / D[c].std(ddof=1) for c in covs])
                    orr, olo, ohi, pf = firth_logit(yb, X, 1)
                    bin_rows.append(dict(stage=stage, measure=label, model=mname, cutoff=cut, n=n, events=ev,
                                         epv=ev / X.shape[1] if ev <= n - ev else (n - ev) / X.shape[1],
                                         auc_unadjusted=a, or_per_sd=orr, or_low=olo, or_high=ohi, p_firth=pf))

    SP, OLS, BIN, ATT = (pd.DataFrame(r) for r in (sp_rows, ols_rows, bin_rows, att_rows))
    if not OLS.empty:
        OLS["p_holm"] = np.nan
        for m, g in OLS.groupby("model"):
            OLS.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    if not BIN.empty:
        BIN["p_holm"] = np.nan
        for (m, c), g in BIN.groupby(["model", "cutoff"]):
            BIN.loc[g.index, "p_holm"] = holm(g["p_firth"].to_numpy())
    for name, df in (("spearman", SP), ("continuous_models", OLS), ("binary_models", BIN), ("attrition", ATT)):
        df.to_csv(out / f"{name}.csv", index=False)

    pd.set_option("display.width", 220)
    f = lambda d, **k: d.round(3).to_string(index=False, **k)
    log("\n== Spearman, participant median exposure vs V9TENG (bootstrap 95% CI) ==")
    log(f(SP) if not SP.empty else "none")
    if not OLS.empty:
        log("\n== CONTINUOUS: V9TENG ~ exposure (per 1 SD) + covariates [OLS, HC3]; Holm over stage x measure per model ==")
        log(f(OLS.drop(columns="covariates")))
        pr = OLS[(OLS.stage == args.primary_stage) & (OLS.measure == "h_q+2")]
        log(f"\nPRE-SPECIFIED PRIMARY TEST ({args.primary_stage}, h_q+2), no multiplicity correction:")
        log(f(pr[["model", "n", "beta_per_sd", "ci_low", "ci_high", "p", "delta_r2"]]))
        if not SP.empty:
            n_p = int(SP[(SP.stage == args.primary_stage) & (SP.measure == "h_q+2")]["n"].iloc[0])
            log(f"\nAt n = {n_p} a correlation of |r| >= {min_detectable_r(n_p):.2f} is detectable with 80% power; "
                f"with a standard deviation of V9TENG of {samples[(args.primary_stage, 'H')]['teng'].std():.1f} points that is "
                f"about {min_detectable_r(n_p) * samples[(args.primary_stage, 'H')]['teng'].std():.1f} points per SD of H.")
    if not BIN.empty:
        log("\n== BINARY: decline = V9TENG < cutoff [Firth logistic, Wald; OR per SD; Holm within model and cutoff] ==")
        log(f(BIN))
    log("\n== ATTRITION: exposure in participants with vs without a V9TENG score (p < 0.05 = informative dropout) ==")
    log(f(ATT))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    if not args.no_plots and not SP.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib missing; no plots")
            return 0
        fig, axes = plt.subplots(1, len(args.stages), figsize=(4.3 * len(args.stages), 4), squeeze=False)
        for ax, stage in zip(axes[0], args.stages):
            D = samples.get((stage, "H"))
            if D is None or len(D) < 20:
                ax.axis("off")
                continue
            dec = D["teng"] < args.cutoffs[0]
            ax.scatter(D.loc[~dec, f"H_{stage}"], D.loc[~dec, "teng"], s=14, alpha=.6, label=f">= {args.cutoffs[0]:g}")
            ax.scatter(D.loc[dec, f"H_{stage}"], D.loc[dec, "teng"], s=18, alpha=.8, color="C3", label=f"< {args.cutoffs[0]:g}")
            c = np.polyfit(D[f"H_{stage}"], D["teng"], 1)
            xs = np.linspace(D[f"H_{stage}"].min(), D[f"H_{stage}"].max(), 20)
            ax.plot(xs, np.polyval(c, xs), "k-", lw=1)
            r = SP[(SP.stage == stage) & (SP.measure == "h_q+2")]
            ax.set_title(f"{stage}: Spearman {r['spearman'].iloc[0]:.2f} (n={len(D)})", fontsize=10)
            ax.set_xlabel("median H(q=2)")
            ax.set_ylabel("V9TENG")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "scatter_H_vs_V9TENG.png", dpi=150)
        plt.close(fig)
        if not OLS.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
            for ax, (meas, ttl) in zip(axes, (("h_q+2", "H(q=2)"), ("delta_h", "delta_h"))):
                g = OLS[OLS.measure == meas]
                ypos, labels = 0, []
                for stage in args.stages:
                    for mname in cov_sets:
                        r = g[(g.stage == stage) & (g.model == mname)]
                        if r.empty:
                            continue
                        ax.errorbar(r["beta_per_sd"], [ypos], xerr=[r["beta_per_sd"] - r["ci_low"], r["ci_high"] - r["beta_per_sd"]],
                                    fmt="o", color=f"C{list(cov_sets).index(mname)}", capsize=2)
                        labels.append((ypos, f"{stage} {mname}"))
                        ypos -= 1
                    ypos -= .5
                ax.axvline(0, color="k", lw=.5)
                ax.set_yticks([y for y, _ in labels], [l for _, l in labels], fontsize=8)
                ax.set_title(f"V9TENG points per SD of {ttl}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "forest_continuous.png", dpi=150)
            plt.close(fig)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
