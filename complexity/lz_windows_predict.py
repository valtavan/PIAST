#!/usr/bin/env python
"""Does LZC in the first / last 1.5 h of sleep predict the Visit 9 outcome group at all?

Keep this file next to lz_h3.py, lz_h3_firth_perm.py and lz_night_change.py (it imports from them).

Windows (per subject, inside the persistent sleep period used in lz_night_change.py)
    early = first --window-min minutes      late = last --window-min minutes
    --window-mode elapsed : clock time from persistent sleep onset / back from final awakening
    --window-mode sleep   : until the Nth epoch of *sleep* has been accumulated (wake does not count)
    Subjects whose two windows would overlap (sleep period < 2 x window) are dropped.

Features per stage x channel  (stages: W, N1, N2, SWS, REM and SLEEP = all sleep epochs pooled)
    early, late (mean LZ in each window) and delta = late - early; each needs >= --min-epochs clean epochs.
    NB 'SLEEP' mixes stages, so it partly reflects stage composition; the stage-specific rows do not.

Analyses  (Stable vs Decline; Decline = V9TENG < --teng-cut among baseline MMSE >= --baseline-min)
    1. Univariate: each feature, adjusted for covariates -> Firth OR per SD, penalised-LR p, permutation p,
       plus the continuous V9TENG slope.  Holm and Benjamini-Hochberg across all tests.
    2. Omnibus ("at all?"): ridge-logistic with cross-validated AUC. Covariates alone vs covariates + a
       block of LZ features (channels averaged); permutation p-value for the AUC gain.

Outputs in <night-dir>/windows_1p5h/ (csv + png)

Usage
  uv run --frozen python lz_windows_predict.py
  uv run --frozen python lz_windows_predict.py --window-mode sleep --window-min 90 --cv-perm 1000
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
from scipy import stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

from lz_h3 import AHI_COLS, DEFAULT_NIGHT, zs
from lz_h3_firth_perm import COL, firth_dz, perm_score_test
from lz_night_change import DEFAULT_RES, EPOCH_SEC, sleep_periods

STAGES = ["W", "N1", "N2", "SWS", "REM", "SLEEP"]
CHANNELS = ["C3_A2", "C4_A1"]
FEATS = ["early", "late", "delta"]
FEAT_TITLE = {"early": "First window", "late": "Last window", "delta": "Last − first"}


# ================================================================== windows & features
def window_bounds(d: pd.DataFrame, per: pd.DataFrame, win_min: float, mode: str) -> pd.DataFrame:
    per = per.copy()
    win = win_min * 60
    if mode == "elapsed":
        per["first_end"] = per["onset_sec"] + win
        per["last_start"] = per["end_sec"] - win
    else:
        n = int(win // EPOCH_SEC)
        ep = d.drop_duplicates(["subject", "epoch"])[["subject", "start_sec", "stage_code"]]
        eg = {k: v for k, v in ep.groupby("subject")}
        fe, ls = [], []
        for r in per.itertuples():
            g = eg[r.subject]
            g = g[(g["start_sec"] >= r.onset_sec) & (g["start_sec"] < r.end_sec)]
            s = g["start_sec"].to_numpy()[g["stage_code"].isin([1, 2, 3, 4]).to_numpy()]
            fe.append(s[n - 1] + EPOCH_SEC if len(s) >= n else np.nan)
            ls.append(s[-n] if len(s) >= n else np.nan)
        per["first_end"], per["last_start"] = fe, ls
    per["usable"] = per["first_end"] <= per["last_start"]  # NaN -> False
    return per


def window_features(d: pd.DataFrame, per: pd.DataFrame, min_epochs: int) -> pd.DataFrame:
    use = per[per["usable"]][["subject", "onset_sec", "end_sec", "first_end", "last_start"]]
    d = d.merge(use, on="subject")
    d["window"] = np.select(
        [(d["start_sec"] >= d["onset_sec"]) & (d["start_sec"] < d["first_end"]),
         (d["start_sec"] >= d["last_start"]) & (d["start_sec"] < d["end_sec"])],
        ["early", "late"], default="")
    clean = d[(~d["artifact"]) & d["lz_norm"].notna() & d["stage"].notna() & d["window"].isin(["early", "late"])]
    sleep = clean[clean["stage"].isin(["N1", "N2", "SWS", "REM"])].assign(stage="SLEEP")
    clean = pd.concat([clean, sleep], ignore_index=True)
    agg = clean.groupby(["subject", "stage", "channel", "window"])["lz_norm"].agg(["mean", "size"]).unstack("window")
    agg.columns = [f"{'n' if a == 'size' else 'mean'}_{b}" for a, b in agg.columns]
    sub = agg.reset_index()
    ok_e, ok_l = sub["n_early"] >= min_epochs, sub["n_late"] >= min_epochs
    sub.loc[~ok_e, "mean_early"] = np.nan
    sub.loc[~ok_l, "mean_late"] = np.nan
    sub["delta"] = np.where(ok_e & ok_l, sub["mean_late"] - sub["mean_early"], np.nan)
    return sub


def to_wide(sub: pd.DataFrame) -> pd.DataFrame:
    w = sub.set_index(["subject", "stage", "channel"])[["mean_early", "mean_late", "delta"]].unstack(["stage", "channel"])
    short = {"mean_early": "early", "mean_late": "late", "delta": "delta"}
    w.columns = [f"{short[v]}_{s}_{c.replace('-', '_')}" for v, s, c in w.columns]
    return w


# ===================================================================== cohort
def load_cohort(args, wide: pd.DataFrame, per: pd.DataFrame):
    ahi = AHI_COLS[args.ahi]
    raw = pd.read_csv(args.night_dir / args.subject_file, dtype={"subject": str})
    raw["subject"] = raw["subject"].str.zfill(5)
    need = ["PREMMSE", "V9TENG", "V8AGE", ahi]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise SystemExit(f"{args.subject_file} lacks columns {miss}")
    cov = raw.groupby("subject")[need].first().rename(columns={ahi: "AHI"})
    df = cov.join(wide, how="left").join(per.set_index("subject")[["sws_min_total", "sp_min"]], how="left")
    df = df.reset_index()
    n_all = len(df)
    df = df.dropna(subset=["V8AGE", "PREMMSE"])
    df = df[df["PREMMSE"] >= args.baseline_min].copy()
    df["case"] = np.where(df["V9TENG"].notna(), (df["V9TENG"] < args.teng_cut).astype(float), np.nan)
    df["group"] = np.select([df["V9TENG"].isna(), df["V9TENG"] < args.teng_cut], ["No V9", "Decline"], "Stable")
    df["c_age"], df["c_mmse"] = zs(df["V8AGE"]), zs(df["PREMMSE"])
    df["c_ahi"] = zs(np.log1p(df["AHI"]))
    df["c_sws"], df["c_sp"] = zs(df["sws_min_total"]), zs(df["sp_min"])
    sets = {"M1": ["c_age", "c_mmse"], "M2": ["c_age", "c_mmse", "c_ahi"],
            "M3": ["c_age", "c_mmse", "c_ahi", "c_sws", "c_sp"]}
    for feat, tag in [("early", "E"), ("late", "L")]:
        for st in STAGES:
            cols = [f"{feat}_{st}_{c}" for c in CHANNELS if f"{feat}_{st}_{c}" in df.columns]
            df[f"{tag}_{st}"] = df[cols].mean(axis=1, skipna=True) if cols else np.nan
    return df, sets[args.covset], n_all


# ================================================================ univariate analysis
def univariate(fu, preds, covs, args, rng) -> pd.DataFrame:
    rows = []
    for pred in preds:
        feat, stage, ch = pred.split("_", 2)
        d = fu.dropna(subset=["case", pred] + covs).copy()
        y = d["case"].to_numpy(float)
        if len(d) < args.min_n or y.sum() < args.min_events:
            continue
        d["dz"] = zs(d[pred])
        X = np.column_stack([np.ones(len(d)), d["dz"].to_numpy(), d[covs].to_numpy()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, fr = firth_dz(X, y)
            p_perm = perm_score_test(y, X, args.n_perm, rng)
            ols = sm.OLS(d["V9TENG"].to_numpy(float), X).fit(cov_type="HC3")
        rows.append({"feature": feat, "stage": stage, "channel": ch, "n": len(d), "events": int(y.sum()),
                     "epv": y.sum() / (X.shape[1] - 1), "or_firth": fr["or"], "or_lo": fr["or_lo"],
                     "or_hi": fr["or_hi"], "p_firth_lr": fr["p_firth_lr"], "p_perm": p_perm,
                     "beta_V9TENG_per_sd": ols.params[1], "p_V9TENG": ols.pvalues[1]})
    res = pd.DataFrame(rows)
    for pc in ["p_firth_lr", "p_perm", "p_V9TENG"]:
        res[pc + "_holm"] = multipletests(res[pc], method="holm")[1]
        res[pc.replace("p_", "q_") + "_bh"] = multipletests(res[pc], method="fdr_bh")[1]
    return res


# ================================================================ omnibus CV analysis
def make_folds(y, k, repeats, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = []
    for _ in range(repeats):
        f = np.empty(len(y), int)
        for idx in (pos, neg):
            p = rng.permutation(idx)
            f[p] = np.arange(len(p)) % k
        out.append(f)
    return out


def ridge_logit(X, y, pen, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ b)
        W = p * (1 - p)
        g = X.T @ (y - p) - pen * b
        H = X.T @ (X * W[:, None]) + np.diag(pen) + 1e-8 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-6:
            break
    return b


def oof_predict(Xc, Xl, y, folds, lam):
    m = 0 if Xl is None else Xl.shape[1]
    pen = np.r_[0.0, np.zeros(Xc.shape[1]), np.full(m, lam)]
    out = []
    for f in folds:
        pr = np.zeros(len(y))
        for k in np.unique(f):
            tr, te = f != k, f == k
            A = Xc[tr] if m == 0 else np.hstack([Xc[tr], Xl[tr]])
            B = Xc[te] if m == 0 else np.hstack([Xc[te], Xl[te]])
            mu, sd = A.mean(0), A.std(0, ddof=1)
            sd[sd == 0] = 1
            A = np.column_stack([np.ones(len(A)), (A - mu) / sd])
            B = np.column_stack([np.ones(len(B)), (B - mu) / sd])
            pr[te] = expit(B @ ridge_logit(A, y[tr], pen))
        out.append(pr)
    return np.array(out)


def auc(y, p):
    r = stats.rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def roc(y, p):
    o = np.argsort(-p)
    yy = y[o]
    return np.r_[0, np.cumsum(1 - yy) / (1 - yy).sum()], np.r_[0, np.cumsum(yy) / yy.sum()]


def omnibus(fu, covs, args, rng):
    ok_stages = []
    print("\ncoverage among followed women (both windows valid, channel-averaged):")
    for st in STAGES:
        frac = (fu[f"E_{st}"].notna() & fu[f"L_{st}"].notna()).mean()
        print(f"  {st:5s} {100 * frac:5.1f}%")
        if frac >= args.omnibus_min_frac:
            ok_stages.append(st)
    print(f"stages used in omnibus (>= {100 * args.omnibus_min_frac:.0f}% coverage): {ok_stages}")
    blocks = {"early": [f"E_{s}" for s in ok_stages], "late": [f"L_{s}" for s in ok_stages],
              "early+late": [f"{t}_{s}" for s in ok_stages for t in "EL"]}
    rows, store = [], {}
    for name, cols in blocks.items():
        d = fu.dropna(subset=["case"] + covs + cols)
        y = d["case"].to_numpy(float)
        if not cols or y.sum() < 10:
            print(f"  block {name}: skipped ({int(y.sum())} events)")
            continue
        Xc, Xl = d[covs].to_numpy(float), d[cols].to_numpy(float)
        folds = make_folds(y, args.cv_folds, args.cv_repeats, rng)
        base = oof_predict(Xc, None, y, folds, args.ridge)
        full = oof_predict(Xc, Xl, y, folds, args.ridge)
        a_base = np.mean([auc(y, p) for p in base])
        a_full = np.mean([auc(y, p) for p in full])
        null = np.empty(args.cv_perm)
        for b in range(args.cv_perm):
            pr = oof_predict(Xc, Xl[rng.permutation(len(y))], y, folds, args.ridge)
            null[b] = np.mean([auc(y, p) for p in pr]) - a_base
        d_obs = a_full - a_base
        rows.append({"block": name, "n": len(d), "events": int(y.sum()), "n_features": len(cols),
                     "auc_covariates": a_base, "auc_with_lz": a_full, "delta_auc": d_obs,
                     "p_perm_delta_auc": (1 + np.sum(null >= d_obs)) / (1 + args.cv_perm)})
        store[name] = {"y": y, "base": base.mean(0), "full": full.mean(0), "null": null, "d_obs": d_obs}
    return pd.DataFrame(rows), store


# ======================================================================== plots
def plot_forest(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    off, mk, cl = {"C3_A2": -0.15, "C4_A1": 0.15}, {"C3_A2": "o", "C4_A1": "s"}, {"C3_A2": "#4C78A8", "C4_A1": "#F58518"}
    for ax, feat in zip(axes, FEATS):
        r = res[res["feature"] == feat]
        for _, row in r.iterrows():
            y = STAGES.index(row["stage"]) + off[row["channel"]]
            ax.errorbar(row["or_firth"], y, xerr=[[row["or_firth"] - row["or_lo"]], [row["or_hi"] - row["or_firth"]]],
                        fmt=mk[row["channel"]], color=cl[row["channel"]], capsize=2, ms=4,
                        label=row["channel"].replace("_", "-") if row["stage"] == r["stage"].iloc[0] else None)
            ax.text(1.01, y, f"{row['p_perm']:.3f}", transform=ax.get_yaxis_transform(), fontsize=6, va="center")
        ax.axvline(1, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(FEAT_TITLE[feat], fontsize=10)
        ax.set_xlabel("OR for decline per 1 SD")
    axes[0].set_yticks(range(len(STAGES)))
    axes[0].set_yticklabels(STAGES)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, title="channel", frameon=False)
    fig.suptitle("LZC in first / last window and decline (Firth OR, profile 95% CI; numbers = permutation p)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_groups(df, cut, win_min, path):
    names = {"Stable": f"Stable (≥{cut:g})", "Decline": f"Decline (<{cut:g})", "No V9": "No V9 score"}
    fig, axes = plt.subplots(1, len(STAGES), figsize=(15, 3.8))
    for ax, st in zip(axes, STAGES):
        for k, off in zip(COL, (-0.06, 0.0, 0.06)):
            g = df[df["group"] == k]
            m, s, n = [], [], []
            for tag in "EL":
                v = g[f"{tag}_{st}"].dropna()
                m.append(v.mean()); s.append(v.sem()); n.append(len(v))
            ax.errorbar(np.array([0, 1]) + off, m, yerr=s, marker="o", ms=4, capsize=2, color=COL[k],
                        label=f"{names[k]} n={n[0]}/{n[1]}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["first", "last"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(st, fontsize=10)
        ax.legend(fontsize=6, frameon=False, loc="best")
    axes[0].set_ylabel("LZC (mean of C3-A2, C4-A1) ± SEM")
    fig.suptitle(f"LZC in the first and last {win_min:g} min of sleep by Visit 9 outcome group", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cv(store, table, path, name="early+late"):
    if name not in store:
        name = next(iter(store))
    s = store[name]
    row = table[table["block"] == name].iloc[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for lab, p, c in [("covariates only", s["base"], "#9D9D9D"), ("covariates + LZC", s["full"], "#E45756")]:
        fpr, tpr = roc(s["y"], p)
        a.plot(fpr, tpr, color=c, label=f"{lab} (AUC {auc(s['y'], p):.2f})")
    a.plot([0, 1], [0, 1], "k:", lw=0.8)
    a.set_xlabel("false positive rate")
    a.set_ylabel("true positive rate")
    a.set_title("Cross-validated ROC (out-of-fold, averaged over repeats)", fontsize=9)
    a.legend(fontsize=8, frameon=False)
    b.hist(s["null"], bins=40, color="#BBBBBB")
    b.axvline(s["d_obs"], color="#E45756", lw=2)
    b.set_xlabel("gain in cross-validated AUC")
    b.set_ylabel("permutations")
    b.set_title(f"Block '{name}': gain {row['delta_auc']:+.3f}, permutation p = {row['p_perm_delta_auc']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ======================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RES)
    ap.add_argument("--night-dir", type=Path, default=DEFAULT_NIGHT)
    ap.add_argument("--subject-file", default="night_change_subject.csv", help="file holding the covariates")
    ap.add_argument("--window-min", type=float, default=90.0)
    ap.add_argument("--window-mode", choices=["elapsed", "sleep"], default="elapsed")
    ap.add_argument("--persist-run", type=int, default=10)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--teng-cut", type=float, default=78.0)
    ap.add_argument("--baseline-min", type=float, default=24)
    ap.add_argument("--ahi", choices=list(AHI_COLS), default="hp3r")
    ap.add_argument("--covset", choices=["M1", "M2", "M3"], default="M2")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--min-events", type=int, default=8)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=5)
    ap.add_argument("--cv-perm", type=int, default=300)
    ap.add_argument("--ridge", type=float, default=10.0, help="ridge penalty on LZ features (standardised)")
    ap.add_argument("--omnibus-min-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = args.night_dir / "windows_1p5h"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.4g}".format)

    d = pd.read_csv(args.results_dir / "lz_epochs_all.csv", dtype={"subject": str})
    per = sleep_periods(d, args.persist_run)
    per = window_bounds(d, per, args.window_min, args.window_mode)
    print(f"{len(per)} subjects with a persistent sleep period; "
          f"{int((~per['usable']).sum())} dropped (windows of {args.window_min:g} min would overlap or "
          f"too little sleep); {int(per['usable'].sum())} usable")
    sub = window_features(d, per, args.min_epochs)
    sub.rename(columns={}).to_csv(out / "night_change_subject_1p5h.csv", index=False)
    wide = to_wide(sub)

    df, covs, n_all = load_cohort(args, wide, per)
    fu = df[df["V9TENG"].notna()]
    print(f"cohort: {n_all} subjects in file -> {len(df)} baseline-normal with covariates -> "
          f"{len(fu)} with V9TENG ({int((fu['case'] == 1).sum())} decline < {args.teng_cut:g}, "
          f"{int((fu['case'] == 0).sum())} stable); {int((df['V9TENG'].isna()).sum())} without V9TENG")

    preds = [c for c in df.columns if c.split("_")[0] in FEATS and c.count("_") >= 2]
    res = univariate(fu, preds, covs, args, rng)
    res.to_csv(out / "windows_univariate.csv", index=False)
    print(f"\n== univariate: decline vs stable per 1 SD, covariates {args.covset} ({len(res)} tests) ==")
    print(res.sort_values("p_perm")[["feature", "stage", "channel", "n", "events", "epv", "or_firth", "or_lo", "or_hi",
                                     "p_firth_lr", "p_perm", "p_perm_holm", "q_perm_bh", "beta_V9TENG_per_sd",
                                     "p_V9TENG"]].head(20).to_string(index=False))
    print("(top 20 by permutation p; full table in windows_univariate.csv)")

    table, store = omnibus(fu, covs, args, rng)
    table.to_csv(out / "windows_cv_auc.csv", index=False)
    print("\n== omnibus: cross-validated AUC, covariates vs covariates + LZC block ==")
    print(table.to_string(index=False))

    plot_forest(res, out / "fig1_forest_windows.png")
    plot_groups(df, args.teng_cut, args.window_min, out / "fig2_lzc_by_group.png")
    if store:
        plot_cv(store, table, out / "fig3_cv_auc.png")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
