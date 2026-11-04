#!/usr/bin/env python3
"""
YASA vs expert hypnograms, compared between participants with TENG (V9TENG) < cutoff and >= cutoff.

Per participant:
  * YASA staging accuracy against the expert hypnogram (accuracy, kappa, per-stage recall/precision/F1)
  * YASA confidence (overall, and on correct vs wrong epochs)
  * Bias per stage: YASA minutes minus expert minutes
  * Sleep architecture from both scorers (TST, SE, SOL, WASO, REM latency, stage %)
Group comparison: Mann-Whitney U (+ FDR) and an age-adjusted linear model (metric ~ low-group + V8AGE),
plus Spearman correlation with the continuous V9TENG score.

Usage:
    python compare_hypnograms_expert.py
    python compare_hypnograms_expert.py --cutoff 80 --out results_expert
Requires: pandas numpy scipy matplotlib   (scikit-learn optional, for kappa)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["W", "N1", "N2", "N3", "REM"]
NAME_MAP = {"W": "W", "WAKE": "W", "N1": "N1", "N2": "N2", "N3": "N3", "N4": "N3", "SWS": "N3",
            "R": "REM", "REM": "REM", "S1": "N1", "S2": "N2", "S3": "N3", "S4": "N3"}
# numeric coding used if the expert file only has numbers (printed on first file so you can verify)
INT_MAP = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def parse_args():
    base = os.path.expanduser("~/pl1358-01/project_data/worksof")
    p = argparse.ArgumentParser()
    p.add_argument("--cog", default=os.path.join(base, "cognitive.csv"))
    p.add_argument("--yasa-dir", default=os.path.join(base, "hypnograms", "yasaRAW"))
    p.add_argument("--expert-dir", default=os.path.join(base, "hypnograms", "expert"))
    p.add_argument("--score-col", default="V9TENG")
    p.add_argument("--age-col", default="V8AGE")
    p.add_argument("--cutoff", type=float, default=88)
    p.add_argument("--epoch-sec", type=float, default=30)
    p.add_argument("--out", default="results_expert")
    p.add_argument("--expert-map", default=None,
                   help="Numeric expert coding, e.g. '0:W,1:N1,2:N2,3:N3,4:REM,5:REM'. Codes not listed are treated as unscored.")
    return p.parse_args()


# ----------------------------------------------------------------- loading
def to_stage(series):
    if pd.api.types.is_numeric_dtype(series):
        return np.array([INT_MAP.get(int(v)) if pd.notna(v) else None for v in series], dtype=object)
    return np.array([NAME_MAP.get(str(v).strip().upper()) for v in series], dtype=object)


def load_yasa(path):
    df = pd.read_csv(path)
    stage = to_stage(df["stage_name"]) if "stage_name" in df else to_stage(df["stage_index"])
    out = {"stage": stage}
    if "confidence" in df:
        out["conf"] = df["confidence"].to_numpy(float)
    return out


def load_expert(path, warn):
    df = pd.read_csv(path)
    low = {str(c).lower(): c for c in df.columns}
    col = next((low[k] for k in ("stage_code", "stage_name", "stage", "stages", "hypnogram", "label", "sleep_stage", "stage_index", "value") if k in low), None)
    if col is None:  # last non-time column
        cands = [c for c in df.columns if "time" not in str(c).lower() and "epoch" not in str(c).lower() and "id" not in str(c).lower()]
        col = cands[-1] if cands else df.columns[-1]
        warn(f"expert stage column guessed: '{col}'")
    return {"stage": to_stage(df[col]), "raw": df[col]}


# ----------------------------------------------------------------- metrics
def bouts(h):
    """Run-length encode a stage array -> list of (stage, n_epochs, start_index); None runs are skipped."""
    out, i, n = [], 0, len(h)
    while i < n:
        j = i
        while j + 1 < n and h[j + 1] == h[i]:
            j += 1
        if h[i] is not None:
            out.append((h[i], j - i + 1, i))
        i = j + 1
    return out


def frag_metrics(h, epoch_sec):
    """Fragmentation / transition metrics between first and last sleep epoch."""
    ep = epoch_sec / 60
    idx = [i for i, x in enumerate(h) if x in ("N1", "N2", "N3", "REM")]
    nan = {k: np.nan for k in ["Trans_per_h", "Awakenings_per_h", "WakeBouts_ge2min_per_h", "N1shifts_per_h",
                               "Arousal_idx_per_h", "MeanBout_W_min", "MeanBout_N1_min", "MeanBout_N2_min",
                               "MeanBout_N3_min", "MeanBout_REM_min", "SleepBouts_per_h"]}
    nan["Trans_prob"] = None
    if len(idx) < 4:
        return nan
    seg = h[idx[0]: idx[-1] + 1]
    b = bouts(seg)
    tst_h = len(idx) * ep / 60
    trans = [(b[k][0], b[k + 1][0]) for k in range(len(b) - 1)]
    awk = [k for k in range(1, len(b)) if b[k][0] == "W"]            # wake bouts after sleep onset
    awk2 = [k for k in awk if b[k][1] * ep >= 2]
    n1 = sum(1 for a, c in trans if c == "N1" and a != "W")           # shifts into N1 from deeper/REM sleep
    m = {"Trans_per_h": len(trans) / tst_h,
         "Awakenings_per_h": len(awk) / tst_h,
         "WakeBouts_ge2min_per_h": len(awk2) / tst_h,
         "N1shifts_per_h": n1 / tst_h,
         "Arousal_idx_per_h": (len(awk) + n1) / tst_h,
         "SleepBouts_per_h": sum(1 for s, _, _ in b if s != "W") / tst_h}
    for st in STAGES:
        lens = [n * ep for s, n, _ in b if s == st]
        m[f"MeanBout_{st}_min"] = float(np.mean(lens)) if lens else np.nan
    # row-normalised stage-to-stage transition probabilities (excluding staying in the same stage)
    P = {}
    for a in STAGES:
        outs = [c for x, c in trans if x == a]
        for c in STAGES:
            if c != a:
                P[f"P_{a}_to_{c}"] = outs.count(c) / len(outs) if outs else np.nan
    m["Trans_prob"] = P
    return m


def near_transition_mask(expert, width=1):
    """True for epochs within `width` epochs of an expert stage change."""
    n = len(expert)
    m = np.zeros(n, bool)
    for i in range(1, n):
        if expert[i] != expert[i - 1] and expert[i] is not None and expert[i - 1] is not None:
            m[max(0, i - width): i + width] = True
    return m


def sleep_metrics(h, epoch_sec, prefix):
    ep = epoch_sec / 60
    valid = np.array([x is not None for x in h])
    asleep = np.array([x in ("N1", "N2", "N3", "REM") for x in h])
    tib, tst = valid.sum() * ep, asleep.sum() * ep
    m = {"TIB_min": tib, "TST_min": tst, "SE_pct": 100 * tst / tib if tib else np.nan}
    if asleep.any():
        first = int(np.argmax(asleep))
        last = len(h) - 1 - int(np.argmax(asleep[::-1]))
        m["SOL_min"] = first * ep
        m["WASO_min"] = sum(1 for x in h[first:last + 1] if x == "W") * ep
        rem = [i for i, x in enumerate(h) if x == "REM"]
        m["REM_latency_min"] = (rem[0] - first) * ep if rem else np.nan
    else:
        m.update(SOL_min=np.nan, WASO_min=np.nan, REM_latency_min=np.nan)
    for st in STAGES:
        mins = sum(1 for x in h if x == st) * ep
        m[f"{st}_min"] = mins
        if st == "W":
            m["W_pct_TIB"] = 100 * mins / tib if tib else np.nan
        else:
            m[f"{st}_pct_TST"] = 100 * mins / tst if tst else np.nan
    fr = frag_metrics(h, epoch_sec)
    tp = fr.pop("Trans_prob")
    m.update(fr)
    if tp:
        m.update(tp)
    return {f"{prefix}_{k}": v for k, v in m.items()}


def kappa(y, x):
    try:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(y, x)
    except ImportError:
        labs = sorted(set(y) | set(x))
        n = len(y)
        po = np.mean(y == x)
        pe = sum(np.mean(y == l) * np.mean(x == l) for l in labs)
        return (po - pe) / (1 - pe) if pe < 1 else np.nan


def agreement(yasa, expert, conf, epoch_sec):
    """yasa/expert: object arrays of stage labels (same length). Returns metrics + confusion counts."""
    ok = np.array([a is not None and b is not None for a, b in zip(yasa, expert)])
    a, b = yasa[ok], expert[ok]
    out = {"n_epochs_compared": len(a)}
    cm = pd.DataFrame(0, index=STAGES, columns=STAGES)  # rows expert, cols yasa
    if len(a) == 0:
        return out, cm
    out["Accuracy"] = float(np.mean(a == b))
    out["Kappa"] = kappa(b, a)
    for st in STAGES:
        t, p = b == st, a == st
        out[f"Recall_{st}"] = float(np.mean(a[t] == st)) if t.any() else np.nan
        out[f"Precision_{st}"] = float(np.mean(b[p] == st)) if p.any() else np.nan
        out[f"F1_{st}"] = 2 * (t & p).sum() / (t.sum() + p.sum()) if (t.sum() + p.sum()) else np.nan
        out[f"Bias_{st}_min"] = (p.sum() - t.sum()) * epoch_sec / 60  # YASA minus expert
    for e, y in zip(b, a):
        cm.loc[e, y] += 1
    near = near_transition_mask(expert)[ok]
    out["Acc_near_transition"] = float(np.mean((a == b)[near])) if near.any() else np.nan
    out["Acc_stable_epochs"] = float(np.mean((a == b)[~near])) if (~near).any() else np.nan
    out["Frac_epochs_near_transition"] = float(np.mean(near))
    if conf is not None:
        c = conf[ok]
        out["Conf_mean"] = float(np.mean(c))
        out["Conf_correct"] = float(np.mean(c[a == b])) if (a == b).any() else np.nan
        out["Conf_wrong"] = float(np.mean(c[a != b])) if (a != b).any() else np.nan
        hi = c >= 0.9
        out["Acc_conf>=0.9"] = float(np.mean((a == b)[hi])) if hi.any() else np.nan
        out["Acc_conf<0.9"] = float(np.mean((a == b)[~hi])) if (~hi).any() else np.nan
        out["Frac_conf>=0.9"] = float(np.mean(hi))
    return out, cm


# ----------------------------------------------------------------- stats
def bh_fdr(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    r = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    res = np.empty_like(pv)
    res[order] = np.minimum(r, 1)
    out[ok] = res
    return out


def ols_group_effect(y, low_flag, age):
    """y ~ 1 + low_group + age. Returns (beta_low, p). Needs >=5 complete rows."""
    ok = ~(np.isnan(y) | np.isnan(age))
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    X = np.column_stack([np.ones(ok.sum()), low_flag[ok], age[ok]])
    yy = y[ok]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    dof = len(yy) - X.shape[1]
    s2 = np.sum((yy - X @ beta) ** 2) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = beta[1] / se[1] if se[1] > 0 else np.nan
    return beta[1], 2 * stats.t.sf(abs(t), dof), int(ok.sum())


def compare(df, metrics, low, high, score_col, age_col):
    rows = []
    isl = (df.group == low).to_numpy(float)
    age = df[age_col].to_numpy(float)
    for m in metrics:
        y = df[m].to_numpy(float)
        lo, hi = y[(df.group == low).to_numpy()], y[(df.group == high).to_numpy()]
        lo, hi = lo[~np.isnan(lo)], hi[~np.isnan(hi)]
        if len(lo) < 2 or len(hi) < 2:
            continue
        p = stats.mannwhitneyu(lo, hi, alternative="two-sided").pvalue
        sd = np.sqrt(((len(lo) - 1) * lo.var(ddof=1) + (len(hi) - 1) * hi.var(ddof=1)) / (len(lo) + len(hi) - 2))
        bl, pa, na = ols_group_effect(y, isl, age)
        okc = ~np.isnan(y) & ~df[score_col].isna().to_numpy()
        rho, prho = stats.spearmanr(y[okc], df[score_col].to_numpy(float)[okc]) if okc.sum() > 3 else (np.nan, np.nan)
        rows.append({"metric": m, "n_low": len(lo), "n_high": len(hi),
                     "mean_low": lo.mean(), "mean_high": hi.mean(), "diff_low_minus_high": lo.mean() - hi.mean(),
                     "cohens_d": (lo.mean() - hi.mean()) / sd if sd > 0 else np.nan, "p_mwu": p,
                     "adj_diff_age": bl, "p_adj_age": pa, "n_adj": na,
                     "spearman_rho_score": rho, "p_spearman": prho})
    res = pd.DataFrame(rows)
    if not res.empty:
        for c in ("p_mwu", "p_adj_age", "p_spearman"):
            res[c.replace("p_", "fdr_")] = bh_fdr(res[c].values)
    return res


def logit_cluster(y, X, groups, names, max_iter=50):
    """Logistic regression (IRLS) with cluster-robust (sandwich) SEs, clustered by person.
    This is a population-averaged model, equivalent to GEE with an independence working correlation."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-9, None)
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    codes, uniq = pd.factorize(groups)
    scores = np.zeros((len(uniq), X.shape[1]))
    for j in range(X.shape[1]):
        scores[:, j] = np.bincount(codes, weights=X[:, j] * (y - p), minlength=len(uniq))
    G, k = len(uniq), X.shape[1]
    cov = bread @ (scores.T @ scores) @ bread * (G / (G - 1)) * ((len(y) - 1) / max(len(y) - k, 1))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    return pd.DataFrame({"term": names, "OR": np.exp(beta), "CI_low": np.exp(beta - 1.96 * se),
                         "CI_high": np.exp(beta + 1.96 * se), "p": pv, "n_epochs": len(y), "n_people": G})


def epoch_models(ep, low, score_col):
    """Epoch-level odds of YASA being correct. ep columns: id, group, age, score, exp_stage, correct."""
    ep = ep.dropna(subset=["age"]).sort_values("id").reset_index(drop=True)
    if ep["id"].nunique() < 8:
        return {}
    ep["low"] = (ep.group == low).astype(float)
    ep["age_c10"] = (ep.age - ep.age.mean()) / 10          # per 10 years
    ep["score_c10"] = (ep.score - ep.score.mean()) / 10    # per 10 score points
    y = ep.correct.to_numpy(float)
    res = {}
    dummies = pd.get_dummies(ep.exp_stage).astype(float)
    ref = "N2"
    dcols = [s for s in STAGES if s != ref and s in dummies]
    stage_X = dummies[dcols].to_numpy()

    X = np.column_stack([np.ones(len(ep)), ep.low, ep.age_c10, stage_X])
    res["A_overall_group"] = logit_cluster(y, X, ep.id.to_numpy(),
                                           ["intercept", f"group {low} (vs high)", "age (per 10 y)"] + [f"expert stage {s} (vs {ref})" for s in dcols])
    X = np.column_stack([np.ones(len(ep)), ep.score_c10, ep.age_c10, stage_X])
    res["B_continuous_score"] = logit_cluster(y, X, ep.id.to_numpy(),
                                              ["intercept", f"{score_col} (per 10 points)", "age (per 10 y)"] + [f"expert stage {s} (vs {ref})" for s in dcols])
    # group x stage: separate models within each expert stage
    rows = []
    for s in STAGES:
        sub = ep[ep.exp_stage == s]
        if sub["id"].nunique() < 8 or sub.low.nunique() < 2:
            continue
        Xs = np.column_stack([np.ones(len(sub)), sub.low, sub.age_c10])
        r = logit_cluster(sub.correct.to_numpy(float), Xs, sub.id.to_numpy(), ["intercept", f"group {low}", "age"]).iloc[1]
        rows.append({"expert_stage": s, **r.to_dict()})
    res["C_group_effect_by_expert_stage"] = pd.DataFrame(rows)
    return res


# ----------------------------------------------------------------- plots
def boxes(df, metrics, low, high, path, title):
    metrics = [m for m in metrics if m in df]
    n = len(metrics)
    if n == 0:
        return
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes.ravel(), metrics):
        data = [df.loc[df.group == g, m].dropna() for g in (low, high)]
        ax.boxplot(data, tick_labels=[low, high], showfliers=False)
        for i, d in enumerate(data, 1):
            ax.scatter(rng.normal(i, 0.05, len(d)), d, s=8, alpha=0.5)
        ax.set_title(m, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_plot(cms, low, high, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, g in zip(axes, (low, high)):
        cm = cms[g]
        norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
        ax.imshow(norm.values, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(5), STAGES)
        ax.set_yticks(range(5), STAGES)
        for i in range(5):
            for j in range(5):
                v = norm.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else "black", fontsize=8)
        ax.set_xlabel("YASA")
        ax.set_ylabel("Expert")
        ax.set_title(f"{g}  (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.expert_map:
        INT_MAP.clear()
        for pair in a.expert_map.split(","):
            k, v = pair.split(":")
            INT_MAP[int(k)] = NAME_MAP[v.strip().upper()]
        print("Expert numeric map:", INT_MAP)
    low, high = f"<{a.cutoff:g}", f">={a.cutoff:g}"
    warned = set()

    def warn(msg):
        if msg not in warned:
            warned.add(msg)
            print("WARNING:", msg, file=sys.stderr)

    cog = pd.read_csv(a.cog)
    cog = cog.rename(columns={cog.columns[0]: "ID"})
    cog[a.score_col] = pd.to_numeric(cog[a.score_col], errors="coerce")
    cog[a.age_col] = pd.to_numeric(cog[a.age_col], errors="coerce")
    cog = cog.dropna(subset=[a.score_col]).copy()
    cog["id5"] = cog["ID"].astype(float).astype(int).astype(str).str.zfill(5)
    cog["group"] = np.where(cog[a.score_col] < a.cutoff, low, high)
    print(f"Participants with {a.score_col}: {len(cog)} ({(cog.group == low).sum()} {low}, {(cog.group == high).sum()} {high}); "
          f"missing {a.age_col} among them: {cog[a.age_col].isna().sum()}")

    rows, cms = [], {low: pd.DataFrame(0, index=STAGES, columns=STAGES), high: pd.DataFrame(0, index=STAGES, columns=STAGES)}
    epoch_parts = []
    no_y, no_e, shown = [], [], False
    for _, r in cog.iterrows():
        pid = r["id5"]
        yf = glob.glob(os.path.join(a.yasa_dir, f"*visit-8-{pid}_hypnogram_epochs.csv"))
        ef = os.path.join(a.expert_dir, f"hypnogram_{pid}.csv")
        if not yf:
            no_y.append(pid)
            continue
        if not os.path.exists(ef):
            no_e.append(pid)
            continue
        try:
            y = load_yasa(yf[0])
            e = load_expert(ef, warn)
        except Exception as ex:
            print(f"  skipped {pid}: {ex}", file=sys.stderr)
            continue
        if not shown:
            print("Expert raw stage values in first file:", sorted(pd.Series(e["raw"]).astype(str).unique()))
            print("Expert values mapped to None (unscored/unknown):", int(np.sum([s is None for s in e["stage"]])), "epochs in first file")
            shown = True
        ny, ne = len(y["stage"]), len(e["stage"])
        k = min(ny, ne)
        ys, es = y["stage"][:k], e["stage"][:k]
        conf = y.get("conf")
        conf = conf[:k] if conf is not None else None
        ag, cm = agreement(ys, es, conf, a.epoch_sec)
        cms[r["group"]] += cm
        okm = np.array([x is not None and z is not None for x, z in zip(ys, es)])
        epoch_parts.append(pd.DataFrame({
            "id": pid, "group": r["group"], "age": r[a.age_col], "score": r[a.score_col],
            "epoch": np.nonzero(okm)[0], "exp_stage": es[okm], "yasa_stage": ys[okm],
            "correct": (ys[okm] == es[okm]).astype(int),
            "conf": conf[okm] if conf is not None else np.nan}))
        rows.append({"id": pid, "group": r["group"], a.score_col: r[a.score_col], a.age_col: r[a.age_col],
                     "len_yasa": ny, "len_expert": ne, **ag,
                     **sleep_metrics(ys, a.epoch_sec, "YASA"), **sleep_metrics(es, a.epoch_sec, "EXP")})

    print(f"Matched participants: {len(rows)}   missing YASA file: {len(no_y)}   missing expert file: {len(no_e)}")
    if not rows:
        sys.exit("Nothing matched - check the directories / file names.")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_subject_metrics.csv"), index=False)
    print(f"Per group: {df.group.value_counts().to_dict()}")
    dl = (df.len_yasa - df.len_expert)
    print(f"Length check (YASA - expert, epochs): identical in {(dl == 0).sum()}/{len(df)}; "
          f"range {dl.min()} to {dl.max()}; max {dl.abs().max() / df[['len_yasa', 'len_expert']].min(axis=1)[dl.abs().idxmax()] * 100:.2f}% of that recording. "
          f"Files were aligned from the first epoch and cut to the shorter one.")
    if dl.abs().max() > 5:
        warn(f"some recordings differ by more than 5 epochs (max {dl.abs().max()}); check those ids in per_subject_metrics.csv")

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    show = ["metric", "mean_low", "mean_high", "diff_low_minus_high", "p_mwu", "fdr_mwu", "adj_diff_age", "p_adj_age", "fdr_adj_age"]

    acc_m = (["Accuracy", "Kappa"] + [f"{p}_{s}" for s in STAGES for p in ("Recall", "Precision", "F1")] +
             [f"Bias_{s}_min" for s in STAGES] +
             ["Conf_mean", "Conf_correct", "Conf_wrong", "Acc_conf>=0.9", "Acc_conf<0.9", "Frac_conf>=0.9",
              "Acc_near_transition", "Acc_stable_epochs", "Frac_epochs_near_transition"])
    r_acc = compare(df, [m for m in acc_m if m in df], low, high, a.score_col, a.age_col)
    r_acc.to_csv(os.path.join(a.out, "group_comparison_agreement.csv"), index=False)
    print("\n=== YASA vs expert: agreement, per-stage performance, bias (YASA - expert), confidence ===")
    print(r_acc[show].to_string(index=False))

    arch = ["SE_pct", "TST_min", "SOL_min", "WASO_min", "REM_latency_min", "W_pct_TIB",
            "N1_pct_TST", "N2_pct_TST", "N3_pct_TST", "REM_pct_TST"]
    for who in ("EXP", "YASA"):
        res = compare(df, [f"{who}_{m}" for m in arch], low, high, a.score_col, a.age_col)
        res.to_csv(os.path.join(a.out, f"group_comparison_architecture_{who}.csv"), index=False)
        print(f"\n=== Sleep architecture, {who} scoring ===")
        print(res[show].to_string(index=False))

    # ---- transitions / fragmentation (expert, YASA, and YASA minus expert)
    frag = ["Trans_per_h", "Awakenings_per_h", "WakeBouts_ge2min_per_h", "N1shifts_per_h", "Arousal_idx_per_h",
            "SleepBouts_per_h"] + [f"MeanBout_{s}_min" for s in STAGES]
    for f in frag:
        if f"YASA_{f}" in df and f"EXP_{f}" in df:
            df[f"Diff_{f}"] = df[f"YASA_{f}"] - df[f"EXP_{f}"]
    for tag, cols in (("EXP", [f"EXP_{f}" for f in frag]), ("YASA", [f"YASA_{f}" for f in frag]),
                      ("YASA_minus_EXP", [f"Diff_{f}" for f in frag])):
        res = compare(df, [c for c in cols if c in df], low, high, a.score_col, a.age_col)
        res.to_csv(os.path.join(a.out, f"group_comparison_fragmentation_{tag}.csv"), index=False)
        print(f"\n=== Fragmentation / transitions: {tag} ===")
        print(res[show].to_string(index=False))
    tp_cols = [c for c in df.columns if c.startswith("EXP_P_")]
    if tp_cols:
        res = compare(df, tp_cols, low, high, a.score_col, a.age_col)
        res.to_csv(os.path.join(a.out, "group_comparison_transition_probabilities_EXP.csv"), index=False)
        print("\n=== Stage-to-stage transition probabilities (expert), top 8 by p ===")
        print(res.sort_values("p_mwu")[show].head(8).to_string(index=False))
    df.to_csv(os.path.join(a.out, "per_subject_metrics.csv"), index=False)
    boxes(df, ["EXP_Trans_per_h", "YASA_Trans_per_h", "Diff_Trans_per_h", "EXP_Awakenings_per_h", "YASA_Awakenings_per_h",
               "Diff_Awakenings_per_h", "Acc_near_transition", "Acc_stable_epochs"], low, high,
          os.path.join(a.out, "fragmentation_boxplots.png"), "Fragmentation and accuracy around transitions")

    # ---- epoch-level model: P(YASA correct) ~ group + age + expert stage, clustered by person
    ep = pd.concat(epoch_parts, ignore_index=True)
    ep.to_csv(os.path.join(a.out, "epoch_level.csv.gz"), index=False)
    models = epoch_models(ep, low, a.score_col)
    if models:
        print("\n=== Epoch-level logistic model, P(YASA epoch correct), person-clustered robust SEs ===")
        print(f"(n={ep.dropna(subset=['age'])['id'].nunique()} people with age; OR<1 means lower odds of a correct YASA epoch)")
        for name, t in models.items():
            t.to_csv(os.path.join(a.out, f"epoch_model_{name}.csv"), index=False)
            print(f"\n-- {name}")
            print(t.to_string(index=False))
    else:
        print("\nEpoch-level model skipped (too few people with age).")

    boxes(df, ["Accuracy", "Kappa"] + [f"F1_{s}" for s in STAGES], low, high,
          os.path.join(a.out, "agreement_boxplots.png"), "YASA vs expert agreement by group")
    boxes(df, [f"Bias_{s}_min" for s in STAGES] + ["Conf_mean", "Conf_wrong"], low, high,
          os.path.join(a.out, "bias_confidence_boxplots.png"), "YASA minus expert minutes, and YASA confidence")
    boxes(df, [f"EXP_{m}" for m in arch], low, high, os.path.join(a.out, "architecture_expert_boxplots.png"),
          "Sleep architecture (expert scoring)")
    confusion_plot(cms, low, high, os.path.join(a.out, "confusion_matrices.png"))
    for g, cm in cms.items():
        cm.to_csv(os.path.join(a.out, f"confusion_{'low' if g == low else 'high'}.csv"))

    # Age check: does age differ between groups, and does it relate to accuracy?
    d = df.dropna(subset=[a.age_col])
    if len(d) > 5:
        lo, hi = d[d.group == low][a.age_col], d[d.group == high][a.age_col]
        print(f"\nAge: {low} mean {lo.mean():.1f} (n={len(lo)}), {high} mean {hi.mean():.1f} (n={len(hi)}), "
              f"MWU p={stats.mannwhitneyu(lo, hi).pvalue:.3g}")
        if "Accuracy" in d:
            rho, p = stats.spearmanr(d[a.age_col], d["Accuracy"], nan_policy="omit")
            print(f"Age vs accuracy: Spearman rho={rho:.2f}, p={p:.3g}")
    print(f"\nOutputs in {a.out}/")


if __name__ == "__main__":
    main()
