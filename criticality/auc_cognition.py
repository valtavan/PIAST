#!/usr/bin/env python
"""
auc_cognition.py - does H(q=+2) separate N3 from the other stages differently in normal / MCI / dementia?

Reads <h-dir>/<id>.parquet (MFDFA output; columns epoch_idx, window_start_time_s, channel, status, h_q+2, r2_q+2),
averages H over the four derivations (epochs must be finite in all of them), joins the expert hypnogram and
computes, PER PARTICIPANT (within-participant, so between-person offsets in H do not leak in):

    auc_rest      N3 vs all non-N3 epochs (probability a random N3 epoch has higher H than a random non-N3 epoch)
    auc_wake      N3 vs wake
    auc_N1        N3 vs N1
    auc_N2        N3 vs N2
    auc_REM       N3 vs REM
    auc_balanced  mean of the four pairwise AUCs (independent of stage proportions; needs all four stages)
    auc_balanced_avail  same, but mean over the pairs that are available (secondary / sensitivity)

Ties count 0.5. Groups from the follow-up 3MS (V9TENG): dementia < 80, MCI 80 <= x < 88, normal >= 88.

Inclusion: N3 >= --min-epochs and comparison class >= --min-stage-epochs (default = --min-epochs).
All inference is at participant level (epochs within a night are not independent).

Per outcome:  group descriptives (+ bootstrap CI of the median, Wilcoxon vs 0.5), Kruskal-Wallis, pairwise
Mann-Whitney (Holm), Cliff's delta and median difference with stratified bootstrap CI, ordinal trend (Spearman of
AUC with group rank), rank-based ANCOVA for group adjusted for age, partial Spearman (group rank | age),
Spearman with raw 3MS (plain and partial for age).

Outputs in <out-dir> (default <h-dir>/auc_cognition):
    auc_by_participant.csv, auc_by_group_summary.csv, auc_group_tests.csv, exclusions_by_group.csv,
    auc_by_group.png, summary.txt

Usage:
    uv run --frozen python auc_cognition.py [--h-dir DIR] [--out-dir DIR] [--cog-file FILE]
Sensitivity examples (give each its own --out-dir):
    --min-epochs 5 | --min-epochs 20 | --channels C3A1 | --min-r2 0.8 | --dementia-cut 80 --mci-cut 90
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

PROJECT = Path("~/pl1358-01/project_data").expanduser()
CRIT_DIR = PROJECT / "worksof/criticality"
H_DIR = CRIT_DIR / "high0.5"
COG_FILE = PROJECT / "worksof/cognitive.csv"
HYP_DIR = PROJECT / "worksof/hypnograms/expert"

STAGES = ["wake", "N1", "N2", "N3", "REM"]
OTHERS = ["wake", "N1", "N2", "REM"]
CODE = {0: "wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CHANNELS = ["C3A1", "C3A2", "C4A1", "C4A2"]
HCOL, RCOL = "h_q+2", "r2_q+2"
GROUPS = ["Normal", "MCI", "Dementia"]
OUTCOMES = ["auc_rest", "auc_balanced", "auc_wake", "auc_N1", "auc_N2", "auc_REM", "auc_balanced_avail"]
LABEL = {"auc_rest": "N3 vs rest", "auc_balanced": "N3 vs stages (balanced)", "auc_wake": "N3 vs wake",
         "auc_N1": "N3 vs N1", "auc_N2": "N3 vs N2", "auc_REM": "N3 vs REM",
         "auc_balanced_avail": "balanced (available pairs)"}


# ----------------------------------------------------------------------------- loading
def load_participant(f, channels, min_r2):
    """Epoch-level mean H over the channels (all must be finite). Index epoch_idx; columns H, t."""
    df = pd.read_parquet(f, columns=["epoch_idx", "window_start_time_s", "channel", "status", HCOL, RCOL])
    ok = df["status"] == "ok"
    if min_r2 > 0:
        ok &= df[RCOL] >= min_r2
    df.loc[~ok, HCOL] = np.nan
    df = df[df["channel"].isin(channels)]
    if df.empty:
        return None
    wide = df.pivot(index="epoch_idx", columns="channel", values=HCOL).reindex(columns=channels).dropna()
    if wide.empty:
        return None
    t = df.drop_duplicates("epoch_idx").set_index("epoch_idx")["window_start_time_s"].reindex(wide.index)
    return pd.DataFrame({"H": wide.mean(axis=1), "t": t})


def add_stage(df, sid, hyp_dir):
    f = hyp_dir / f"hypnogram_{sid}.csv"
    if not f.exists():
        return None
    h = pd.read_csv(f)
    m = dict(zip(h["start_sec"].round().astype(int), h["stage_code"].astype(int)))
    df = df.copy()
    df["stage"] = [CODE.get(m.get(int(round(t)), -1)) for t in df["t"].to_numpy()]
    return df.dropna(subset=["stage"])


def load_cog(path, col, id_col, age_col):
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    for need in (col, id_col) + ((age_col,) if age_col else ()):
        if need.lower() not in cols:
            sys.exit(f"{path}: no column {need}; columns are {list(df.columns)}")
    ids = df[cols[id_col.lower()]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    y = pd.to_numeric(df[cols[col.lower()]], errors="coerce")
    out = pd.DataFrame({"teng": y.where((y >= 0) & (y <= 100)).to_numpy()}, index=ids)
    out["age"] = pd.to_numeric(df[cols[age_col.lower()]], errors="coerce").to_numpy() if age_col else np.nan
    return out[~out.index.duplicated()]


# ----------------------------------------------------------------------------- statistics
def auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cliff(a, b):
    return 2 * auc(a, b) - 1


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    run, m = 0.0, len(order)
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        out[i] = min(1.0, run)
    return out


def pctl(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return (np.nan, np.nan) if len(arr) < 50 else tuple(np.percentile(arr, [2.5, 97.5]))


def boot(fun, samples, rng, n_boot):
    """Stratified bootstrap: `samples` is a list of arrays (one per group), resampled within group."""
    res = []
    for _ in range(n_boot):
        res.append(fun(*[s[rng.integers(0, len(s), len(s))] for s in samples]))
    return pctl(res)


def partial_spearman(x, y, Z=None):
    """(rho, p). Rank-partial given Z (n x k) if provided; p from the t distribution with n-2-k df."""
    m = np.isfinite(x) & np.isfinite(y)
    if Z is not None:
        m &= np.isfinite(Z).all(axis=1)
    n = int(m.sum())
    if n < 8:
        return np.nan, np.nan, n
    rx, ry = rankdata(x[m]), rankdata(y[m])
    k = 0
    if Z is not None:
        rz = np.column_stack([rankdata(Z[m][:, j]) for j in range(Z.shape[1])])
        X = np.column_stack([np.ones(n), rz])
        rx = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
        ry = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
        k = Z.shape[1]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    t = r * np.sqrt((n - 2 - k) / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), n - 2 - k)), n


def rank_ancova(y, grp, age):
    """F test for group in a rank-based ANCOVA (rank(y) ~ rank(age) + group dummies). Returns (F, p, n)."""
    m = np.isfinite(y) & np.isfinite(age) & pd.notna(grp)
    y, age, grp = y[m], age[m], np.asarray(grp)[m]
    levels = [g for g in GROUPS if (grp == g).sum() > 0]
    n = len(y)
    if len(levels) < 2 or n < len(levels) + 4:
        return np.nan, np.nan, n
    ry, ra = rankdata(y), rankdata(age)
    X0 = np.column_stack([np.ones(n), ra])
    X1 = np.column_stack([X0] + [(grp == g).astype(float) for g in levels[1:]])
    rss = lambda X: float(np.sum((ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]) ** 2))
    r0, r1 = rss(X0), rss(X1)
    d1, d2 = X1.shape[1] - X0.shape[1], n - X1.shape[1]
    if d2 <= 0 or r1 == 0:
        return np.nan, np.nan, n
    F = ((r0 - r1) / d1) / (r1 / d2)
    return float(F), float(stats.f.sf(F, d1, d2)), n


# ----------------------------------------------------------------------------- per participant
def participant_row(sid, df, args):
    H, st = df["H"].to_numpy(float), df["stage"].to_numpy()
    row = {"id": sid, "n_epochs": len(df)}
    for s in STAGES:
        row[f"n_{s}"] = int((st == s).sum())
    n3 = H[st == "N3"]
    rest = H[st != "N3"]
    ok3 = len(n3) >= args.min_epochs
    row["auc_rest"] = auc(n3, rest) if ok3 and len(rest) >= args.min_stage_epochs else np.nan
    for s in OTHERS:
        x = H[st == s]
        row[f"auc_{s}"] = auc(n3, x) if ok3 and len(x) >= args.min_stage_epochs else np.nan
    vals = np.array([row[f"auc_{s}"] for s in OTHERS])
    row["n_stages_avail"] = int(np.isfinite(vals).sum())
    row["auc_balanced"] = float(vals.mean()) if np.isfinite(vals).all() else np.nan
    row["auc_balanced_avail"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    return row


# ----------------------------------------------------------------------------- group analysis
def analyse(P, outcome, rng, n_boot):
    """Returns (summary rows, test rows) for one outcome."""
    d = P[P["group"].notna() & P[outcome].notna()]
    arrs = {g: d.loc[d["group"] == g, outcome].to_numpy(float) for g in GROUPS}
    summ, tests = [], []

    for g in GROUPS:
        a = arrs[g]
        n_all = int((P["group"] == g).sum())
        row = dict(outcome=outcome, group=g, n_total=n_all, n_included=len(a))
        if len(a):
            lo, hi = boot(lambda x: np.median(x), [a], rng, n_boot)
            row.update(median=float(np.median(a)), q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)),
                       mean=float(a.mean()), median_lo=lo, median_hi=hi)
            if len(a) >= 6 and np.any(a != 0.5):
                try:
                    w = stats.wilcoxon(a - 0.5)
                    row.update(wilcoxon_vs_0p5_stat=float(w.statistic), wilcoxon_vs_0p5_p=float(w.pvalue))
                except ValueError:
                    pass
        summ.append(row)

    def T(test, comparison, n, stat=np.nan, p=np.nan, eff=np.nan, lo=np.nan, hi=np.nan, effect_name=""):
        tests.append(dict(outcome=outcome, test=test, comparison=comparison, n=n, statistic=stat, p=p,
                          effect_name=effect_name, effect=eff, effect_lo=lo, effect_hi=hi))

    present = [g for g in GROUPS if len(arrs[g]) >= 2]
    if len(present) >= 2:
        kw = stats.kruskal(*[arrs[g] for g in present])
        T("kruskal_wallis", " vs ".join(present), sum(len(arrs[g]) for g in present), float(kw.statistic), float(kw.pvalue))

    pw_start = len(tests)
    for g1, g2 in combinations(GROUPS, 2):
        a, b = arrs[g1], arrs[g2]
        if len(a) < 3 or len(b) < 3:
            T("mann_whitney", f"{g1} vs {g2}", len(a) + len(b))
            continue
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        cd = cliff(a, b)
        lo, hi = boot(cliff, [a, b], rng, n_boot)
        T("mann_whitney", f"{g1} vs {g2}", len(a) + len(b), float(mw.statistic), float(mw.pvalue), cd, lo, hi,
          "cliffs_delta (positive: first group higher)")
        md = float(np.median(a) - np.median(b))
        mlo, mhi = boot(lambda x, y: np.median(x) - np.median(y), [a, b], rng, n_boot)
        T("median_difference", f"{g1} - {g2}", len(a) + len(b), eff=md, lo=mlo, hi=mhi, effect_name="median AUC difference")
    ph = holm([t["p"] for t in tests[pw_start:] if t["test"] == "mann_whitney"])
    k = 0
    for t in tests[pw_start:]:
        if t["test"] == "mann_whitney":
            t["p_holm"] = ph[k]
            k += 1

    y = d[outcome].to_numpy(float)
    g_ord = d["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy(float)
    age = d["age"].to_numpy(float)
    teng = d["teng"].to_numpy(float)
    Za = age[:, None]
    r, p, n = partial_spearman(g_ord, y)
    T("trend_spearman_group_rank", "AUC vs group (Normal<MCI<Dementia)", n, p=p, eff=r, effect_name="rho")
    r, p, n = partial_spearman(g_ord, y, Za)
    T("trend_partial_age", "AUC vs group | age", n, p=p, eff=r, effect_name="partial rho")
    F, p, n = rank_ancova(y, d["group"].to_numpy(), age)
    T("rank_ancova_age", "group effect adjusted for age", n, F, p)
    r, p, n = partial_spearman(teng, y)
    T("spearman_3MS", "AUC vs 3MS", n, p=p, eff=r, effect_name="rho")
    r, p, n = partial_spearman(teng, y, Za)
    T("spearman_3MS_partial_age", "AUC vs 3MS | age", n, p=p, eff=r, effect_name="partial rho")
    return summ, tests


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ex = lambda s: Path(s).expanduser()
    ap.add_argument("--h-dir", type=ex, default=H_DIR)
    ap.add_argument("--out-dir", type=ex, default=None, help="default <h-dir>/auc_cognition")
    ap.add_argument("--hyp-dir", type=ex, default=HYP_DIR)
    ap.add_argument("--cog-file", type=ex, default=COG_FILE)
    ap.add_argument("--cog-col", default="V9TENG")
    ap.add_argument("--cog-id-col", default="ID")
    ap.add_argument("--age-col", default="V8AGE")
    ap.add_argument("--dementia-cut", type=float, default=80, help="3MS < this: dementia")
    ap.add_argument("--mci-cut", type=float, default=88, help="dementia-cut <= 3MS < this: MCI; >= this: normal")
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--min-r2", type=float, default=0.0, help="drop epochs with q=+2 fit R^2 below this (0 = off)")
    ap.add_argument("--min-epochs", type=int, default=10, help="min N3 epochs per participant")
    ap.add_argument("--min-stage-epochs", type=int, default=None, help="min epochs in the comparison class (default = --min-epochs)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    if args.min_stage_epochs is None:
        args.min_stage_epochs = args.min_epochs
    out = args.out_dir or (args.h_dir / "auc_cognition")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(args.h_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files in {args.h_dir}")
    cog = load_cog(args.cog_file, args.cog_col, args.cog_id_col, args.age_col)

    # ---- per participant
    rows, skipped = [], []
    for f in files:
        sid = f.stem.zfill(5)
        d = load_participant(f, args.channels, args.min_r2)
        if d is None:
            skipped.append((sid, "no valid epochs"))
            continue
        d = add_stage(d, sid, args.hyp_dir)
        if d is None or d.empty:
            skipped.append((sid, "no hypnogram / no labelled epochs"))
            continue
        rows.append(participant_row(sid, d, args))
    P = pd.DataFrame(rows).set_index("id")
    P = P.join(cog, how="left")
    tot = P[[f"n_{s}" for s in STAGES]].sum(axis=1)
    for s in STAGES:
        P[f"frac_{s}"] = P[f"n_{s}"] / tot
    P["group"] = np.select([P["teng"] < args.dementia_cut, P["teng"] < args.mci_cut, P["teng"] >= args.mci_cut],
                           ["Dementia", "MCI", "Normal"], default="")
    P["group"] = P["group"].replace("", np.nan)
    P["in_primary"] = P["group"].notna() & P["auc_rest"].notna()
    P["in_balanced"] = P["group"].notna() & P["auc_balanced"].notna()
    P.reset_index().to_csv(out / "auc_by_participant.csv", index=False)

    # ---- exclusions
    ex_rows = []
    for g in GROUPS:
        sub = P[P["group"] == g]
        row = dict(group=g, n_with_data=len(sub), n_in_auc_rest=int(sub["auc_rest"].notna().sum()),
                   n_in_balanced=int(sub["auc_balanced"].notna().sum()),
                   n_lacking_N3=int((sub["n_N3"] < args.min_epochs).sum()))
        for s in OTHERS:
            row[f"n_lacking_{s}"] = int((sub[f"n_{s}"] < args.min_stage_epochs).sum())
        for s in STAGES:
            row[f"median_frac_{s}_included"] = float(sub.loc[sub["auc_rest"].notna(), f"frac_{s}"].median())
            row[f"median_frac_{s}_excluded"] = float(sub.loc[sub["auc_rest"].isna(), f"frac_{s}"].median()) \
                if sub["auc_rest"].isna().any() else np.nan
        ex_rows.append(row)
    pd.DataFrame(ex_rows).to_csv(out / "exclusions_by_group.csv", index=False)

    # ---- statistics
    S, T = [], []
    for o in OUTCOMES:
        s_, t_ = analyse(P, o, rng, args.n_boot)
        S += s_
        T += t_
    S = pd.DataFrame(S)
    T = pd.DataFrame(T)
    S.to_csv(out / "auc_by_group_summary.csv", index=False)
    T.to_csv(out / "auc_group_tests.csv", index=False)

    # ---- text summary
    pd.set_option("display.width", 220)
    L = [f"h-dir: {args.h_dir}", f"channels: {args.channels}, min_r2: {args.min_r2}, min N3 epochs: {args.min_epochs}, "
         f"min comparison-stage epochs: {args.min_stage_epochs}",
         f"groups (3MS {args.cog_col}): Dementia < {args.dementia_cut:g} <= MCI < {args.mci_cut:g} <= Normal",
         f"parquet files: {len(files)}, analysed: {len(P)}, skipped: {len(skipped)}"
         + (f" ({skipped[:5]}{'...' if len(skipped) > 5 else ''})" if skipped else ""),
         f"with a valid 3MS: {int(P['teng'].notna().sum())}; with a group: {int(P['group'].notna().sum())}; "
         f"missing age among grouped: {int(P.loc[P['group'].notna(), 'age'].isna().sum())}", "",
         "=== participants per group and inclusion ===", pd.DataFrame(ex_rows).set_index("group").iloc[:, :8].to_string(), "",
         "=== AUC by group (median [IQR], n) ==="]
    for o in OUTCOMES:
        parts = []
        for g in GROUPS:
            r = S[(S.outcome == o) & (S.group == g)].iloc[0]
            parts.append(f"{g}: {r['median']:.3f} [{r['q25']:.3f}-{r['q75']:.3f}] n={int(r['n_included'])}"
                         if r["n_included"] else f"{g}: n=0")
        L.append(f"{LABEL[o]:28s} " + " | ".join(parts))
    L += ["", "=== key tests (see auc_group_tests.csv for all) ==="]
    key = T[T["test"].isin(["kruskal_wallis", "rank_ancova_age", "trend_partial_age", "spearman_3MS_partial_age"])]
    L.append(key[["outcome", "test", "n", "statistic", "p", "effect"]].round(4).to_string(index=False))
    L += ["", "Caveats: inference is at participant level; groups (especially dementia) may be small; participants "
          "lacking N3 or a comparison stage are excluded (see exclusions_by_group.csv); the 'rest' AUC mixes stages whose "
          "proportions differ by group, the balanced AUC does not (it requires all four comparison stages)."]
    txt = "\n".join(L)
    print(txt)
    (out / "summary.txt").write_text(txt + "\n")

    # ---- plot
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib missing; no plots")
            return 0
        outs = OUTCOMES[:-1]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
        for ax, o in zip(axes.ravel(), outs):
            data = [P.loc[(P["group"] == g) & P[o].notna(), o].to_numpy(float) for g in GROUPS]
            ax.boxplot(data, positions=range(3), widths=0.5, showfliers=False)
            for i, a in enumerate(data):
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(a)), a, s=14, alpha=0.6, zorder=3)
            ax.axhline(0.5, color="grey", ls=":", lw=1)
            ax.set_xticks(range(3), [f"{g}\n(n={len(a)})" for g, a in zip(GROUPS, data)])
            ax.set_title(LABEL[o], fontsize=10)
        for ax in axes[:, 0]:
            ax.set_ylabel("AUC (within participant)")
        fig.suptitle(f"AUC of H(q=+2), N3 vs other stages, by cognitive group (3MS cut-offs {args.dementia_cut:g}/{args.mci_cut:g})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "auc_by_group.png", dpi=150)
        plt.close(fig)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
