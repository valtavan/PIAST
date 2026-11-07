#!/usr/bin/env python
"""H3 v2 - binary 'decline vs stable' analysis with small-sample methods, attrition handling, and plots.

Keep this file in the same folder as lz_h3.py (it imports its loaders).

Groups (baseline MMSE >= --baseline-min only)
    Stable    V9TENG >= --teng-cut   (default 78)
    Decline   V9TENG <  --teng-cut
    No V9     no Visit 9 score (death / drop-out / not tested; the data cannot say which)

Methods for Stable vs Decline, per stage x channel x covariate set (M1/M2/M3 as in lz_h3.py)
    Firth       penalised logistic regression (Jeffreys prior): OR per SD, profile-likelihood CI,
                penalised LR p-value.  Built for few events and separation.
    Permutation score-test p-value; delta is residualised on the covariates and shuffled
                (Freedman-Lane style).  No distributional assumptions.
    ML          ordinary logistic regression, for comparison only.
    IPW         ML logistic with inverse-probability-of-follow-up weights (sensitivity for No V9).
    Attrition   does delta predict having a Visit 9 score at all?

Outputs in <night-dir>/h3_v2/: tables (csv) and figures (png).

Usage
  uv run --frozen python lz_h3_firth_perm.py --teng-cut 78
  uv run --frozen python lz_h3_firth_perm.py --teng-cut 78 --n-perm 20000 --focus REM_C3_A2
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
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

from lz_h3 import AHI_COLS, DEFAULT_NIGHT, load_wide, zs

STAGES = ["W", "N1", "N2", "SWS", "REM"]
CHANNELS = ["C3_A2", "C4_A1"]
COL = {"Stable": "#4C78A8", "Decline": "#E45756", "No V9": "#9D9D9D"}


# ============================================================ Firth logistic regression
def _pll(X, y, beta):
    eta = X @ beta
    w = expit(eta) * (1 - expit(eta))
    ll = np.sum(y * eta - np.logaddexp(0, eta))
    return ll + 0.5 * np.linalg.slogdet(X.T @ (X * w[:, None]))[1]


def firth_fit(X, y, fixed=None, max_iter=200):
    """Maximise the Firth-penalised log-likelihood; `fixed` = {column: value} holds coefficients."""
    fixed = fixed or {}
    p = X.shape[1]
    free = np.array([j for j in range(p) if j not in fixed])
    beta = np.zeros(p)
    for j, v in fixed.items():
        beta[j] = v
    cur = _pll(X, y, beta)
    converged = False
    for _ in range(max_iter):
        pi = expit(X @ beta)
        w = pi * (1 - pi)
        info = X.T @ (X * w[:, None])
        inv = np.linalg.inv(info)
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        U = X.T @ (y - pi + h * (0.5 - pi))
        step = np.zeros(p)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], U[free])
        if np.abs(step).max() > 5:
            step *= 5 / np.abs(step).max()
        t = 1.0
        for _h in range(30):
            new = beta + t * step
            val = _pll(X, y, new)
            if val >= cur - 1e-10:
                break
            t *= 0.5
        beta, cur = new, val
        if np.abs(t * step).max() < 1e-8:
            converged = True
            break
    pi = expit(X @ beta)
    cov = np.linalg.inv(X.T @ (X * (pi * (1 - pi))[:, None]))
    return {"beta": beta, "pll": cur, "cov": cov, "converged": converged}


def firth_dz(X, y, j=1, level=0.95):
    """Firth estimate, penalised-LR p-value and profile-likelihood CI for column j."""
    full = firth_fit(X, y)
    red = firth_fit(X, y, fixed={j: 0.0})
    p_lr = stats.chi2.sf(max(2 * (full["pll"] - red["pll"]), 0.0), 1)
    b, se = full["beta"][j], np.sqrt(full["cov"][j, j])
    crit = stats.chi2.ppf(level, 1)

    def g(bv):
        return 2 * (full["pll"] - firth_fit(X, y, fixed={j: bv})["pll"]) - crit

    def edge(sign):
        step = sign * se
        x = b + step
        for _ in range(40):
            if g(x) > 0:
                lo, hi = sorted([b, x])
                return optimize.brentq(g, lo, hi, xtol=1e-6)
            step *= 1.6
            x = b + step
        return np.nan

    return full, {"beta": b, "or": np.exp(b), "or_lo": np.exp(edge(-1)), "or_hi": np.exp(edge(+1)),
                  "p_firth_lr": p_lr, "converged": full["converged"]}


# ======================================================================= permutation test
def perm_score_test(y, X, n_perm, rng, j=1, return_null=False):
    """Permutation p-value for column j given the others (delta residualised on covariates)."""
    Z = np.delete(X, j, axis=1)
    x = X[:, j]
    fit0 = sm.GLM(y, Z, family=sm.families.Binomial()).fit()
    pi = fit0.fittedvalues.to_numpy() if hasattr(fit0.fittedvalues, "to_numpy") else fit0.fittedvalues
    w = pi * (1 - pi)
    r = y - pi
    ZWZinv = np.linalg.inv(Z.T @ (Z * w[:, None]))

    def stat(M):  # M: (B, n) candidate versions of the predictor
        U = M @ r
        A = (M * w) @ Z
        V = (M ** 2 * w).sum(1) - np.einsum("bk,kl,bl->b", A, ZWZinv, A)
        return U ** 2 / V

    obs = stat(x[None, :])[0]
    gamma = np.linalg.lstsq(Z, x, rcond=None)[0]
    fitted = Z @ gamma
    resid = x - fitted
    null, done = [], 0
    while done < n_perm:
        m = min(2000, n_perm - done)
        idx = np.argsort(rng.random((m, len(y))), axis=1)
        null.append(stat(fitted + resid[idx]))
        done += m
    null = np.concatenate(null)
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return (p, obs, null) if return_null else p


# ===================================================================== data preparation
def build(args):
    df, preds = load_wide(args)
    n_all = len(df)
    df = df.dropna(subset=["V8AGE", "PREMMSE"]).copy()
    n_cov = len(df)
    if args.baseline_min > 0:
        df = df[df["PREMMSE"] >= args.baseline_min].copy()
    n_base = len(df)

    df["has_v9"] = df["V9TENG"].notna().astype(int)
    df["case"] = np.where(df["V9TENG"].notna(), (df["V9TENG"] < args.teng_cut).astype(float), np.nan)
    df["group"] = np.select([df["V9TENG"].isna(), df["V9TENG"] < args.teng_cut], ["No V9", "Decline"], "Stable")

    df["c_age"] = zs(df["V8AGE"])
    df["c_mmse"] = zs(df["PREMMSE"])
    df["c_ahi"] = zs(np.log1p(df["AHI"]))
    sets = {"M1": ["c_age", "c_mmse"], "M2": ["c_age", "c_mmse", "c_ahi"]}
    if "sws_min_total" in df.columns:
        df["c_sws"] = zs(df["sws_min_total"])
        df["c_sp"] = zs(df["sp_min"])
        sets["M3"] = sets["M2"] + ["c_sws", "c_sp"]

    # inverse-probability-of-follow-up weights (stabilised, truncated, mean 1 among followed)
    wcovs = sets.get("M3", sets["M2"])
    wd = df.dropna(subset=wcovs)
    ps = sm.Logit(wd["has_v9"], sm.add_constant(wd[wcovs])).fit(disp=0).predict()
    w = np.minimum(wd["has_v9"].mean() / ps, 10.0)
    df["w"] = np.nan
    df.loc[wd.index, "w"] = w
    fu_w = df.loc[df["has_v9"] == 1, "w"]
    df.loc[df["has_v9"] == 1, "w"] = fu_w / fu_w.mean()
    flow = {"n_all": n_all, "n_missing_cov": n_all - n_cov, "n_excl_baseline": n_cov - n_base,
            "n_base": n_base, "n_v9": int(df["has_v9"].sum()), "n_no_v9": int((1 - df["has_v9"]).sum()),
            "n_stable": int((df["group"] == "Stable").sum()), "n_decline": int((df["group"] == "Decline").sum())}
    return df, preds, sets, flow


def prepare(fu, pred, covs, adjust_early):
    early = "e_" + pred[2:]
    need = ["case", pred] + covs + ([early] if adjust_early else [])
    d = fu.dropna(subset=need).copy()
    d["dz"] = zs(d[pred])
    cols = covs[:]
    if adjust_early:
        d["ez"] = zs(d[early])
        cols = cols + ["ez"]
    X = np.column_stack([np.ones(len(d)), d["dz"].to_numpy(), d[cols].to_numpy()])
    return d, X, d["case"].to_numpy(float)


def holm(res: pd.DataFrame, pcols: list[str]) -> pd.DataFrame:
    for pc in pcols:
        res[f"{pc}_holm"] = np.nan
        for _, idx in res.groupby("model_set").groups.items():
            ok = res.loc[idx, pc].notna()
            ids = res.loc[idx][ok].index
            if len(ids):
                res.loc[ids, f"{pc}_holm"] = multipletests(res.loc[ids, pc], method="holm")[1]
    return res


# ====================================================================== group / attrition
def group_table(df, preds):
    rows = []
    for pred in preds:
        g = {k: df.loc[df["group"] == k, pred].dropna() for k in ["Stable", "Decline", "No V9"]}
        if min(len(v) for v in g.values()) < 5:
            continue
        rows.append({"predictor": pred[2:], **{f"n_{k}": len(v) for k, v in g.items()},
                     **{f"median_{k}": v.median() for k, v in g.items()},
                     "p_kruskal_3groups": stats.kruskal(*g.values()).pvalue,
                     "p_decline_vs_stable": stats.mannwhitneyu(g["Decline"], g["Stable"]).pvalue,
                     "p_noV9_vs_followed": stats.mannwhitneyu(
                         g["No V9"], pd.concat([g["Stable"], g["Decline"]])).pvalue})
    out = pd.DataFrame(rows)
    for pc in ["p_kruskal_3groups", "p_decline_vs_stable", "p_noV9_vs_followed"]:
        out[pc + "_holm"] = multipletests(out[pc], method="holm")[1]
    return out


def attrition_table(df, preds, covs):
    rows = []
    for pred in preds:
        d = df.dropna(subset=[pred] + covs).copy()
        d["dz"] = zs(d[pred])
        X = sm.add_constant(d[["dz"] + covs])
        try:
            r = sm.Logit(d["has_v9"], X).fit(disp=0)
        except Exception:
            continue
        ci = np.exp(r.conf_int().loc["dz"])
        rows.append({"predictor": pred[2:], "n": len(d), "n_with_v9": int(d["has_v9"].sum()),
                     "or_having_v9_per_sd": np.exp(r.params["dz"]), "ci_lo": ci[0], "ci_hi": ci[1],
                     "p": r.pvalues["dz"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out


# ============================================================================== plots
def _split(name):
    stage, ch = name.split("_", 1)
    return stage, ch


def plot_forest(res, set_name, cut, path):
    r = res[res["model_set"] == set_name]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    off = {"C3_A2": -0.15, "C4_A1": 0.15}
    mark = {"C3_A2": "o", "C4_A1": "s"}
    colr = {"C3_A2": "#4C78A8", "C4_A1": "#F58518"}
    for _, row in r.iterrows():
        stage, ch = _split(row["predictor"])
        y = STAGES.index(stage) + off[ch]
        ax.errorbar(row["or_firth"], y, xerr=[[row["or_firth"] - row["or_lo"]], [row["or_hi"] - row["or_firth"]]],
                    fmt=mark[ch], color=colr[ch], capsize=2, ms=5,
                    label=ch.replace("_", "-") if stage == STAGES[0] else None)
        ax.text(1.02, y, f"p_perm={row['p_perm']:.3f}", transform=ax.get_yaxis_transform(), fontsize=7, va="center")
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(STAGES)))
    ax.set_yticklabels(STAGES)
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio for decline per 1 SD of early-to-late LZC change (log scale)")
    ax.set_title(f"Decline (V9TENG < {cut:g}) vs stable; Firth OR, profile-likelihood 95% CI; model {set_name}", fontsize=9)
    ax.legend(title="channel", fontsize=8, loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_groups(df, cut, path):
    labels = {"Stable": f"Stable\n(≥{cut:g})", "Decline": f"Decline\n(<{cut:g})", "No V9": "No V9\nscore"}
    fig, axes = plt.subplots(len(CHANNELS), len(STAGES), figsize=(15, 6.2))
    rng = np.random.default_rng(0)
    for i, ch in enumerate(CHANNELS):
        for j, st in enumerate(STAGES):
            ax = axes[i, j]
            col = f"d_{st}_{ch}"
            data = [df.loc[df["group"] == k, col].dropna().to_numpy() for k in COL]
            bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True)
            for patch, k in zip(bp["boxes"], COL):
                patch.set(facecolor=COL[k], alpha=0.35)
            for med in bp["medians"]:
                med.set_color("k")
            for x, (k, v) in enumerate(zip(COL, data), start=1):
                ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=6, color=COL[k], alpha=0.7, zorder=3)
            ax.axhline(0, color="k", lw=0.5, ls=":")
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels([f"{labels[k]}\nn={len(v)}" for k, v in zip(COL, data)], fontsize=7)
            ax.set_title(f"{st}  {ch.replace('_', '-')}", fontsize=9)
            if j == 0:
                ax.set_ylabel("late − early LZC")
    fig.suptitle("Early-to-late LZC change by Visit 9 outcome group (baseline MMSE-normal women)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_flow(f, cut, path):
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    def box(x, y, text, c="white"):
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc=c, ec="k"))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->"))

    box(5, 9.2, f"Visit 8 PSG subjects with LZ data\nn = {f['n_all']}")
    box(5, 7.0, f"Baseline MMSE normal, age available\nn = {f['n_base']}")
    box(8.6, 8.1, f"Excluded\nmissing age/MMSE: {f['n_missing_cov']}\nbaseline MMSE low: {f['n_excl_baseline']}", "#F2F2F2")
    arrow(5, 8.6, 5, 7.6); arrow(5.9, 8.2, 7.4, 8.1)
    box(2.2, 4.4, f"Visit 9 score available\nn = {f['n_v9']}")
    box(7.8, 4.4, f"No Visit 9 score\nn = {f['n_no_v9']}", COL["No V9"])
    arrow(4.3, 6.5, 2.6, 5.0); arrow(5.7, 6.5, 7.5, 5.0)
    box(0.9, 1.5, f"Stable\n(≥{cut:g})\nn = {f['n_stable']}", COL["Stable"])
    box(3.6, 1.5, f"Decline\n(<{cut:g})\nn = {f['n_decline']}", COL["Decline"])
    arrow(2.0, 3.8, 1.1, 2.3); arrow(2.5, 3.8, 3.3, 2.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_prob(d, full, pred, cut, path):
    sd, mu = d[pred].std(ddof=1), d[pred].mean()
    zgrid = np.linspace(d["dz"].quantile(0.02), d["dz"].quantile(0.98), 100)
    b, C = full["beta"], full["cov"]
    eta = b[0] + b[1] * zgrid                      # covariates held at their mean (z = 0)
    se = np.sqrt(C[0, 0] + zgrid ** 2 * C[1, 1] + 2 * zgrid * C[0, 1])
    fig, (ax, rug) = plt.subplots(2, 1, figsize=(6.5, 5), gridspec_kw={"height_ratios": [4, 1]}, sharex=True)
    x = mu + sd * zgrid
    ax.plot(x, expit(eta), color="#E45756")
    ax.fill_between(x, expit(eta - 1.96 * se), expit(eta + 1.96 * se), color="#E45756", alpha=0.2)
    ax.set_ylabel(f"P(decline: V9TENG < {cut:g})")
    ax.set_title(f"{pred[2:].replace('_', '-')}: Firth-predicted risk (covariates at mean)", fontsize=10)
    for val, colr, yy in [(0, COL["Stable"], 0.3), (1, COL["Decline"], 0.7)]:
        rug.scatter(d.loc[d["case"] == val, pred], np.full((d["case"] == val).sum(), yy), s=8, color=colr, alpha=0.6)
    rug.set_yticks([0.3, 0.7])
    rug.set_yticklabels(["stable", "decline"], fontsize=7)
    rug.set_ylim(0, 1)
    rug.set_xlabel("late − early LZC")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_null(null, obs, p, pred, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(null, bins=60, color="#BBBBBB")
    ax.axvline(obs, color="#E45756", lw=2)
    ax.set_xlabel("permutation score statistic")
    ax.set_ylabel("count")
    ax.set_title(f"{pred[2:].replace('_', '-')}: observed vs permutation null (p = {p:.4f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night-dir", type=Path, default=DEFAULT_NIGHT)
    ap.add_argument("--subject-file", default="night_change_subject.csv")
    ap.add_argument("--ahi", choices=list(AHI_COLS), default="hp3r")
    ap.add_argument("--baseline-min", type=float, default=24)
    ap.add_argument("--teng-cut", type=float, default=78.0)
    ap.add_argument("--adjust-early", action="store_true")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--min-events", type=int, default=8)
    ap.add_argument("--primary-set", default="M2")
    ap.add_argument("--focus", default="auto", help="predictor for plots 4/5, e.g. REM_C3_A2 (auto = smallest Firth p)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = args.night_dir / "h3_v2"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.4g}".format)

    df, preds, sets, flow = build(args)
    fu = df[df["has_v9"] == 1]
    print("flow:", flow)
    print(f"Decline = V9TENG < {args.teng_cut:g}: {flow['n_decline']} / {len(fu)} followed "
          f"({100 * flow['n_decline'] / max(len(fu), 1):.1f}%)")
    w = df.loc[df["has_v9"] == 1, "w"].dropna()
    print(f"IPW weights among followed: min {w.min():.2f}, max {w.max():.2f}, "
          f"effective n {w.sum() ** 2 / (w ** 2).sum():.0f} of {len(w)}")

    # ---- the women who did not show up
    grp = group_table(df, preds)
    grp.to_csv(out / "h3v2_groups.csv", index=False)
    att = attrition_table(df, preds, sets["M2"])
    att.to_csv(out / "h3v2_attrition.csv", index=False)
    print("\n== delta by group (Kruskal 3 groups; decline vs stable; no-V9 vs followed) ==")
    print(grp[["predictor", "n_Stable", "n_Decline", "n_No V9", "median_Stable", "median_Decline", "median_No V9",
               "p_kruskal_3groups_holm", "p_decline_vs_stable_holm", "p_noV9_vs_followed_holm"]].to_string(index=False))
    print("\n== does delta predict having a Visit 9 score? (logistic, M2 covariates) ==")
    print(att.to_string(index=False))

    # ---- Stable vs Decline models
    rows = []
    for sname, covs in sets.items():
        for pred in preds:
            d, X, y = prepare(fu, pred, covs, args.adjust_early)
            if len(d) < args.min_n or y.sum() < args.min_events:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, fr = firth_dz(X, y)
                p_perm = perm_score_test(y, X, args.n_perm, rng)
                row = {"model_set": sname, "predictor": pred[2:], "n": len(d), "events": int(y.sum()),
                       "epv": y.sum() / (X.shape[1] - 1), "or_firth": fr["or"], "or_lo": fr["or_lo"],
                       "or_hi": fr["or_hi"], "p_firth_lr": fr["p_firth_lr"], "p_perm": p_perm,
                       "firth_converged": fr["converged"]}
                try:
                    ml = sm.Logit(y, X).fit(disp=0, maxiter=200)
                    row.update({"or_ml": np.exp(ml.params[1]), "p_ml_wald": ml.pvalues[1]})
                except Exception:
                    pass
                try:
                    wt = d["w"].to_numpy()
                    ok = ~np.isnan(wt)
                    ip = sm.GLM(y[ok], X[ok], family=sm.families.Binomial(),
                                freq_weights=wt[ok]).fit(cov_type="HC0")
                    row.update({"or_ipw": np.exp(ip.params[1]), "p_ipw": ip.pvalues[1]})
                except Exception:
                    pass
            rows.append(row)
    res = holm(pd.DataFrame(rows), ["p_firth_lr", "p_perm"])
    res["low_epv"] = res["epv"] < 10
    res.to_csv(out / "h3v2_binary.csv", index=False)
    print(f"\n== STABLE vs DECLINE (<{args.teng_cut:g}) per 1 SD of delta ==")
    print(res[["model_set", "predictor", "n", "events", "epv", "or_firth", "or_lo", "or_hi", "p_firth_lr",
               "p_firth_lr_holm", "p_perm", "p_perm_holm", "or_ml", "or_ipw", "p_ipw"]].to_string(index=False))

    # ---- figures
    plot_flow(flow, args.teng_cut, out / "fig1_flow.png")
    plot_groups(df, args.teng_cut, out / "fig2_delta_by_group.png")
    plot_forest(res, args.primary_set, args.teng_cut, out / "fig3_forest_firth.png")
    prim = res[res["model_set"] == args.primary_set]
    focus = args.focus if args.focus != "auto" else prim.sort_values("p_firth_lr").iloc[0]["predictor"]
    pred = "d_" + focus
    d, X, y = prepare(fu, pred, sets[args.primary_set], args.adjust_early)
    full, fr = firth_dz(X, y)
    p, obs, null = perm_score_test(y, X, args.n_perm, rng, return_null=True)
    plot_prob(d, full, pred, args.teng_cut, out / f"fig4_risk_curve_{focus}.png")
    plot_null(null, obs, p, pred, out / f"fig5_permutation_null_{focus}.png")
    print(f"\nfocus predictor for fig4/fig5: {focus} (set {args.primary_set})")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
