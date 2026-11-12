#!/usr/bin/env python
"""
generate_final_report.py
Final reporting pipeline for:
    Model: h_q+4(auc_N2) + frac_N3 + age
    Cohort: min_epochs = 5, dementia_cut = 78 (3MS < 78), mci_cut = 88

Generates in ~/pl1358-01/project_data/worksof/criticality/high0.5/final_model_ep5_cut78/:
    - final_composite_2x2.png  (300 DPI 2x2 publication figure)
    - final_composite_2x2.pdf  (vector format for manuscript submission)
    - final_report_summary.txt
    - final_odds_ratios.csv
    - final_operating_points.csv
    - final_cv_comparison.csv
    - final_multiq_scan.csv
    - participants_ep5_cut78.csv

Usage:
    cd ~/pl1358-01/project_data/code
    uv run --frozen python generate_final_report.py
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path("~/pl1358-01/project_data").expanduser()
H_DIR = PROJECT / "worksof/criticality/high0.5"
COG_FILE = PROJECT / "worksof/cognitive.csv"
HYP_DIR = PROJECT / "worksof/hypnograms/expert"

CODE = {0: "wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CHANNELS = ["C3A1", "C3A2", "C4A1", "C4A2"]
GROUPS = ["Normal", "MCI", "Dementia"]
PALETTE = {"Normal": "#0072B2", "MCI": "#E69F00", "Dementia": "#D55E00"}  # Okabe-Ito colorblind-safe


# ----------------------------------------------------------------------------- helpers
def parse_q_val(col):
    m = re.search(r"h_q([+-]?\d+(?:\.\d+)?)", col)
    return float(m.group(1)) if m else np.nan


def epoch_auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc(y_true, scores, rng, n_boot=2000):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    raw_auc = epoch_auc(pos, neg)
    flip = raw_auc < 0.5
    auc_val = 1.0 - raw_auc if flip else raw_auc
    mw_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        bp = pos[rng.integers(0, len(pos), len(pos))]
        bn = neg[rng.integers(0, len(neg), len(neg))]
        val = epoch_auc(bp, bn)
        boots[b] = 1.0 - val if flip else val
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc_val, float(lo), float(hi), mw_p, ("lower" if flip else "higher")


def corrected_resampled_ttest(diffs, n_splits=5):
    diffs = np.asarray(diffs, float)
    m = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    if var_d <= 1e-16:
        return mean_d, np.nan
    se = np.sqrt((1.0 / m + 1.0 / (n_splits - 1)) * var_d)
    t_stat = mean_d / se
    return mean_d, float(2 * stats.t.sf(abs(t_stat), df=m - 1))


def load_cog(path, col="V9TENG", id_col="ID", age_col="V8AGE"):
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    ids = df[cols[id_col.lower()]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    y = pd.to_numeric(df[cols[col.lower()]], errors="coerce")
    out = pd.DataFrame({"teng": y.where((y >= 0) & (y <= 100)).to_numpy()}, index=ids)
    out["age"] = pd.to_numeric(df[cols[age_col.lower()]], errors="coerce").to_numpy()
    return out[~out.index.duplicated()]


def extract_all_participants(h_dir, hyp_dir, channels, min_epochs=5):
    files = sorted(h_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"No parquet files found in {h_dir}")
    sample_cols = pd.read_parquet(files[0]).columns
    h_cols = sorted([c for c in sample_cols if c.startswith("h_q")], key=parse_q_val)

    rows = []
    for f in files:
        sid = f.stem.zfill(5)
        hf = hyp_dir / f"hypnogram_{sid}.csv"
        if not hf.exists():
            continue

        df = pd.read_parquet(f, columns=["epoch_idx", "window_start_time_s", "channel", "status"] + h_cols)
        df = df[df["channel"].isin(channels)].copy()
        if df.empty:
            continue

        ok = df["status"] == "ok"
        q_series = {}
        for hcol in h_cols:
            vals = df[hcol].where(ok, np.nan)
            wide = (
                df[["epoch_idx", "channel"]]
                .assign(val=vals)
                .pivot(index="epoch_idx", columns="channel", values="val")
                .reindex(columns=channels)
                .dropna()
            )
            if not wide.empty:
                q_series[hcol] = wide.mean(axis=1)

        if not q_series:
            continue

        ep_df = pd.DataFrame(q_series)
        t = df.drop_duplicates("epoch_idx").set_index("epoch_idx")["window_start_time_s"].reindex(ep_df.index)
        hyp = pd.read_csv(hf)
        stage_map = dict(zip(hyp["start_sec"].round().astype(int), hyp["stage_code"].astype(int)))
        ep_df["stage"] = [CODE.get(stage_map.get(int(round(ts)), -1)) for ts in t.to_numpy()]
        ep_df = ep_df.dropna(subset=["stage"])
        if ep_df.empty:
            continue

        st = ep_df["stage"].to_numpy()
        n_tot = len(st)
        m_n3 = st == "N3"
        m_n2 = st == "N2"

        row = {
            "id": sid,
            "n_epochs": n_tot,
            "n_N3": int(m_n3.sum()),
            "n_N2": int(m_n2.sum()),
            "frac_N3": float(m_n3.sum() / n_tot),
            "frac_N2": float(m_n2.sum() / n_tot),
        }
        for hcol in h_cols:
            vals = ep_df[hcol].to_numpy(float)
            n3 = vals[m_n3 & np.isfinite(vals)]
            n2 = vals[m_n2 & np.isfinite(vals)]
            if len(n3) >= min_epochs and len(n2) >= min_epochs:
                med_n3 = float(np.median(n3))
                med_n2 = float(np.median(n2))
                row[f"{hcol}__H_N3_med"] = med_n3
                row[f"{hcol}__H_N2_med"] = med_n2
                row[f"{hcol}__H_N3_minus_N2"] = med_n3 - med_n2
                row[f"{hcol}__auc_N2"] = epoch_auc(n3, n2)
            else:
                row[f"{hcol}__H_N3_med"] = np.nan
                row[f"{hcol}__H_N2_med"] = np.nan
                row[f"{hcol}__H_N3_minus_N2"] = np.nan
                row[f"{hcol}__auc_N2"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("id"), h_cols


def fit_logistic_with_or(df, target_col, feature_specs):
    cols = [f[0] for f in feature_specs]
    sub = df[[target_col] + cols].dropna().copy()
    y = sub[target_col].to_numpy(float)
    X_raw = sub[cols].to_numpy(float)

    means = X_raw.mean(axis=0)
    sds = X_raw.std(axis=0, ddof=1)
    Z = (X_raw - means) / sds
    Z_design = np.column_stack([np.ones(len(Z)), Z])

    lr = LogisticRegression(C=1e9, max_iter=2000, solver="lbfgs").fit(Z, y)
    beta_std = np.concatenate([[lr.intercept_[0]], lr.coef_[0]])
    probs = lr.predict_proba(Z)[:, 1]
    W = probs * (1.0 - probs)
    cov_std = np.linalg.inv((Z_design.T * W) @ Z_design)
    se_std = np.sqrt(np.diag(cov_std))

    out = []
    for j, (col, unit_step, unit_label) in enumerate(feature_specs, start=1):
        b_s, se_s = beta_std[j], se_std[j]
        z_val = b_s / se_s
        p_val = float(2 * stats.norm.sf(abs(z_val)))
        scale_factor = unit_step / sds[j - 1]
        b_c = b_s * scale_factor
        se_c = se_s * abs(scale_factor)
        out.append({
            "feature": col,
            "std_OR": float(np.exp(b_s)),
            "std_OR_lo": float(np.exp(b_s - 1.96 * se_s)),
            "std_OR_hi": float(np.exp(b_s + 1.96 * se_s)),
            "clinical_step": unit_label,
            "clin_OR": float(np.exp(b_c)),
            "clin_OR_lo": float(np.exp(b_c - 1.96 * se_c)),
            "clin_OR_hi": float(np.exp(b_c + 1.96 * se_c)),
            "p_value": p_val,
        })
    return pd.DataFrame(out)


def evaluate_operating_points(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    youden = tpr + spec - 1.0

    idx_youden = int(np.argmax(youden))
    sens_cand = np.where(tpr >= 0.80)[0]
    idx_sens80 = int(sens_cand[np.argmax(spec[sens_cand])]) if len(sens_cand) else idx_youden
    spec_cand = np.where(spec >= 0.80)[0]
    idx_spec80 = int(spec_cand[np.argmax(tpr[spec_cand])]) if len(spec_cand) else idx_youden

    rows = []
    for name, idx in [
        ("Youden Optimal (max Sens+Spec-1)", idx_youden),
        ("High-Sensitivity Screen (Sens >= 80%)", idx_sens80),
        ("High-Specificity Rule-In (Spec >= 80%)", idx_spec80),
    ]:
        th = float(thresholds[idx])
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        se = tp / max(1, tp + fn)
        sp = tn / max(1, tn + fp)
        ppv = tp / max(1, tp + fp)
        npv = tn / max(1, tn + fn)
        rows.append({
            "operating_point": name,
            "prob_threshold": th,
            "sensitivity": float(se),
            "specificity": float(sp),
            "ppv": float(ppv),
            "npv": float(npv),
            "balanced_acc": float(0.5 * (se + sp)),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ex = lambda s: Path(s).expanduser()
    ap.add_argument("--h-dir", type=ex, default=H_DIR)
    ap.add_argument("--hyp-dir", type=ex, default=HYP_DIR)
    ap.add_argument("--cog-file", type=ex, default=COG_FILE)
    ap.add_argument("--out-dir", type=ex, default=None)
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--dementia-cut", type=float, default=78)
    ap.add_argument("--mci-cut", type=float, default=88)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.h_dir / f"final_model_ep{args.min_epochs}_cut{int(args.dementia_cut)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Extracting multi-q features (min_epochs={args.min_epochs}, dementia_cut<{args.dementia_cut:g})...")
    cog = load_cog(args.cog_file)
    feat_df, h_cols = extract_all_participants(args.h_dir, args.hyp_dir, CHANNELS, min_epochs=args.min_epochs)
    P = feat_df.join(cog, how="left")
    P["group"] = np.select(
        [P["teng"] < args.dementia_cut, P["teng"] < args.mci_cut, P["teng"] >= args.mci_cut],
        ["Dementia", "MCI", "Normal"],
        default="",
    )
    P["group"] = P["group"].replace("", np.nan)
    P.reset_index().to_csv(out_dir / "participants_ep5_cut78.csv", index=False)

    # Complete-case cohort for final 3-feature model
    model_cols = ["h_q+4__auc_N2", "frac_N3", "age"]
    comp = P[P["group"].notna() & P[model_cols].notna().all(axis=1)].copy()
    n_norm = int((comp["group"] == "Normal").sum())
    n_mci = int((comp["group"] == "MCI").sum())
    n_dem = int((comp["group"] == "Dementia").sum())

    # 1. Multi-q Diagnostic Scan (Dementia vs Normal across all q)
    scan_rows = []
    dn_all = P[P["group"].isin(["Dementia", "Normal"])]
    for hcol in h_cols:
        q_val = parse_q_val(hcol)
        for metric in ["auc_N2", "H_N3_minus_N2", "H_N2_med", "H_N3_med"]:
            col = f"{hcol}__{metric}"
            sub = dn_all[dn_all[col].notna()]
            y_dn = (sub["group"] == "Dementia").astype(int).to_numpy()
            auc_v, lo_v, hi_v, p_v, dir_v = bootstrap_auc(y_dn, sub[col].to_numpy(float), rng, args.n_boot)
            scan_rows.append({
                "dimension": hcol,
                "q": q_val,
                "metric": metric,
                "n": len(sub),
                "Dem_vs_Norm_AUC": auc_v,
                "CI_lo": lo_v,
                "CI_hi": hi_v,
                "p_value": p_v,
                "direction": dir_v,
            })
    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(out_dir / "final_multiq_scan.csv", index=False)

    # 2. Repeated 5-Fold Stratified CV Comparison
    y_multi = comp["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
    y_dem = (comp["group"] == "Dementia").astype(int).to_numpy()
    n_splits, n_repeats = 5, 20
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=args.seed)

    cv_models = {
        "Age only": ["age"],
        "h_q+4(auc_N2) only": ["h_q+4__auc_N2"],
        "h_q+4(auc_N2) + age": ["h_q+4__auc_N2", "age"],
        "h_q+4(auc_N2) + frac_N3 + age": ["h_q+4__auc_N2", "frac_N3", "age"],
    }
    fpr_grid = np.linspace(0, 1, 101)
    roc_store = {}
    fold_dn_store = {}
    oof_p_dem = np.zeros(len(comp))
    oof_cnt = np.zeros(len(comp))

    cv_rows = []
    for mname, cols in cv_models.items():
        X = comp[cols].to_numpy(float)
        dn_aucs, dr_aucs, ovo_aucs, bals, tprs = [], [], [], [], []
        for tr, te in rskf.split(X, y_multi):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
            clf.fit(X[tr], y_multi[tr])
            probs = clf.predict_proba(X[te])
            preds = clf.predict(X[te])

            dr_aucs.append(roc_auc_score(y_dem[te], probs[:, 2]))
            ovo_aucs.append(roc_auc_score(y_multi[te], probs, multi_class="ovo", average="macro"))
            bals.append(balanced_accuracy_score(y_multi[te], preds))

            m_dn = np.isin(y_multi[te], [0, 2])
            y_dn_te = (y_multi[te][m_dn] == 2).astype(int)
            p_dn_te = probs[m_dn, 2] / np.clip(probs[m_dn, 0] + probs[m_dn, 2], 1e-12, None)
            dn_aucs.append(roc_auc_score(y_dn_te, p_dn_te))

            fpr, tpr, _ = roc_curve(y_dn_te, p_dn_te)
            tprs.append(np.interp(fpr_grid, fpr, tpr))

            if mname == "h_q+4(auc_N2) + frac_N3 + age":
                oof_p_dem[te] += probs[:, 2]
                oof_cnt[te] += 1

        fold_dn_store[mname] = np.array(dn_aucs)
        roc_store[mname] = (np.mean(tprs, axis=0), np.std(tprs, axis=0), float(np.mean(dn_aucs)))

    for mname in cv_models:
        d_age, p_age = corrected_resampled_ttest(fold_dn_store[mname] - fold_dn_store["Age only"], n_splits)
        cv_rows.append({
            "model": mname,
            "Dem_vs_Norm_AUC": float(np.mean(fold_dn_store[mname])),
            "Gain_over_Age": d_age if mname != "Age only" else 0.0,
            "p_vs_Age": p_age if mname != "Age only" else np.nan,
        })
    # Add full metrics row table
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "final_cv_comparison.csv", index=False)

    comp["oof_prob_dementia"] = oof_p_dem / oof_cnt

    # 3. Exact Odds Ratios
    feature_specs = [
        ("h_q+4__auc_N2", -0.05, "per -0.05 drop in N3-N2 AUC"),
        ("frac_N3", +0.05, "per +5% (+0.05) increase in N3 fraction"),
        ("age", +5.0, "per +5 years of age"),
    ]
    dn_comp = comp[comp["group"].isin(["Normal", "Dementia"])].copy()
    dn_comp["y_dem"] = (dn_comp["group"] == "Dementia").astype(int)
    or_dn = fit_logistic_with_or(dn_comp, "y_dem", feature_specs)
    or_dn.insert(0, "outcome", "Dementia vs Normal")

    all_comp = comp.copy()
    all_comp["y_dem"] = (all_comp["group"] == "Dementia").astype(int)
    or_dr = fit_logistic_with_or(all_comp, "y_dem", feature_specs)
    or_dr.insert(0, "outcome", "Dementia vs Non-Dementia")
    or_table = pd.concat([or_dn, or_dr], ignore_index=True)
    or_table.to_csv(out_dir / "final_odds_ratios.csv", index=False)

    # 4. Operating Points
    op_dn = evaluate_operating_points(
        (dn_comp["group"] == "Dementia").astype(int).to_numpy(),
        dn_comp["oof_prob_dementia"].to_numpy(),
    )
    op_dn.insert(0, "comparison", "Dementia vs Normal")

    op_dr = evaluate_operating_points(
        (comp["group"] == "Dementia").astype(int).to_numpy(),
        comp["oof_prob_dementia"].to_numpy(),
    )
    op_dr.insert(0, "comparison", "Dementia vs Non-Dementia")
    op_table = pd.concat([op_dn, op_dr], ignore_index=True)
    op_table.to_csv(out_dir / "final_operating_points.csv", index=False)

    # Save & print summary
    auc_n2_scan = scan_df[scan_df["metric"] == "auc_N2"].sort_values("q")
    lines = [
        f"FINAL MODEL SPECIFICATION: h_q+4(auc_N2) + frac_N3 + age",
        f"Parameters: min_epochs = {args.min_epochs}, dementia_cut < {args.dementia_cut:g}, mci_cut < {args.mci_cut:g}",
        f"Sample Size: n = {len(comp)} (Normal = {n_norm}, MCI = {n_mci}, Dementia = {n_dem})",
        "",
        "=== 1. Repeated 5-Fold Stratified CV Comparison (Dementia vs. Normal) ===",
        cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 2. Multivariable Odds Ratios ===",
        or_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 3. Out-of-Fold Classification Operating Points ===",
        op_table[[
            "comparison", "operating_point", "sensitivity", "specificity",
            "ppv", "npv", "balanced_acc", "TP", "FP", "TN", "FN",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 4. Multi-q Diagnostic Profile (auc_N2, Dementia vs. Normal) ===",
        auc_n2_scan[["dimension", "q", "n", "Dem_vs_Norm_AUC", "CI_lo", "CI_hi", "p_value"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ),
    ]
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "final_report_summary.txt").write_text(txt + "\n")

    # ------------------------------------------------------------------------- 2x2 COMPOSITE FIGURE
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

    # PANEL A: Within-Night Stage Convergence (h_q+4: N2 -> N3)
    ax = axes[0, 0]
    for idx, g in enumerate(GROUPS):
        gsub = comp[comp["group"] == g]
        x0, x1 = idx - 0.17, idx + 0.17
        for _, r in gsub.iterrows():
            ax.plot([x0, x1], [r["h_q+4__H_N2_med"], r["h_q+4__H_N3_med"]], color=PALETTE[g], alpha=0.14, lw=0.8)

        med_n2, med_n3 = gsub["h_q+4__H_N2_med"].median(), gsub["h_q+4__H_N3_med"].median()
        q25_n2, q75_n2 = np.percentile(gsub["h_q+4__H_N2_med"], [25, 75])
        q25_n3, q75_n3 = np.percentile(gsub["h_q+4__H_N3_med"], [25, 75])

        ax.plot([x0, x1], [med_n2, med_n3], color=PALETTE[g], lw=3.0, marker="o", markersize=7.5, zorder=5)
        ax.errorbar([x0], [med_n2], yerr=[[med_n2 - q25_n2], [q75_n2 - med_n2]], color=PALETTE[g], lw=1.8, capsize=3, zorder=6)
        ax.errorbar([x1], [med_n3], yerr=[[med_n3 - q25_n3], [q75_n3 - med_n3]], color=PALETTE[g], lw=1.8, capsize=3, zorder=6)

        med_auc = gsub["h_q+4__auc_N2"].median()
        ax.text(
            idx, max(q75_n3, med_n3) + 0.025,
            f"Median auc_N2\n= {med_auc:.3f}",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=PALETTE[g],
        )

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{g} (n={int((comp['group']==g).sum())})\nN2  →  N3" for g in GROUPS])
    ax.set_ylabel("Epoch-Median Hurst Exponent H(q = +4)")
    ax.set_ylim(0.50, 0.96)
    ax.set_title("A. Within-Night Stage Convergence (N2 → N3)", loc="left", fontweight="bold")

    # PANEL B: Multi-q Diagnostic Profile (Dementia vs Normal AUC across q)
    ax = axes[0, 1]
    profile_specs = [
        ("auc_N2", "Within-night N3 vs. N2 AUC (auc_N2)", "#009E73", "o", "-"),
        ("H_N3_minus_N2", "Median difference (H_N3 − H_N2)", "#CC79A7", "s", "--"),
        ("H_N2_med", "Raw N2 median H(q)", "#56B4E9", "v", "-."),
        ("H_N3_med", "Raw N3 median H(q)", "#888888", "^", ":"),
    ]
    for metric, label, color, marker, ls in profile_specs:
        mdf = scan_df[scan_df["metric"] == metric].sort_values("q")
        ax.plot(mdf["q"], mdf["Dem_vs_Norm_AUC"], marker=marker, ls=ls, lw=2.1, color=color, label=label)
        if metric == "auc_N2":
            ax.fill_between(mdf["q"], mdf["CI_lo"], mdf["CI_hi"], color=color, alpha=0.14)
            peak_row = mdf[mdf["q"] == 4.0].iloc[0]
            ax.annotate(
                f"q = +4 peak\nAUC = {peak_row['Dem_vs_Norm_AUC']:.3f}\n(p = {peak_row['p_value']:.4f})",
                xy=(4.0, peak_row["Dem_vs_Norm_AUC"]),
                xytext=(1.7, 0.745),
                arrowprops=dict(arrowstyle="->", color="#009E73", lw=1.2),
                fontsize=8.5, fontweight="bold", color="#00664B",
            )

    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_xticks([-4, -3, -2, -1, 1, 2, 3, 4])
    ax.set_xlabel("Multifractal Moment Order (q)")
    ax.set_ylabel("Dementia vs. Normal ROC-AUC")
    ax.set_ylim(0.42, 0.82)
    ax.set_title("B. Multi-q Diagnostic Profile (3MS < 78)", loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.3, frameon=False)

    # PANEL C: 2D Feature Synergy (h_q+4__auc_N2 vs frac_N3)
    ax = axes[1, 0]
    for g in GROUPS:
        gsub = comp[comp["group"] == g]
        ax.scatter(
            gsub["h_q+4__auc_N2"], gsub["frac_N3"] * 100.0,
            color=PALETTE[g], alpha=0.45, s=26, edgecolors="none", label=f"{g} (n={len(gsub)})",
        )
        # Group median + IQR crossbars
        mx, my = gsub["h_q+4__auc_N2"].median(), gsub["frac_N3"].median() * 100.0
        qx_lo, qx_hi = np.percentile(gsub["h_q+4__auc_N2"], [25, 75])
        qy_lo, qy_hi = np.percentile(gsub["frac_N3"] * 100.0, [25, 75])
        ax.errorbar(
            [mx], [my],
            xerr=[[mx - qx_lo], [qx_hi - mx]],
            yerr=[[my - qy_lo], [qy_hi - my]],
            color=PALETTE[g], elinewidth=2.0, capsize=3, zorder=5,
        )
        ax.scatter([mx], [my], color=PALETTE[g], edgecolor="black", s=115, marker="D", zorder=6)

    ax.set_xlabel("Within-Night N3 vs. N2 Separation (h_q+4 auc_N2)")
    ax.set_ylabel("Scored N3 Sleep Proportion (frac_N3, %)")
    ax.set_title("C. 2D Feature Synergy (Median & IQR Crossbars)", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.8, frameon=False)

    # PANEL D: Out-of-Fold Cross-Validated ROC Curves (Dementia vs Normal)
    ax = axes[1, 1]
    p_vs_age_final = cv_df.loc[cv_df["model"] == "h_q+4(auc_N2) + frac_N3 + age", "p_vs_Age"].iloc[0]
    roc_plot_specs = [
        ("Age only", "#888888", "--", "Age only"),
        ("h_q+4(auc_N2) only", "#0072B2", "-.", "h_q+4(auc_N2) only"),
        ("h_q+4(auc_N2) + frac_N3 + age", "#D55E00", "-", "h_q+4(auc_N2) + frac_N3 + age"),
    ]
    for mname, col, ls, short_lbl in roc_plot_specs:
        mean_tpr, sd_tpr, mean_auc = roc_store[mname]
        extra = f", p={p_vs_age_final:.3f} vs age" if "frac_N3" in mname else ""
        ax.plot(fpr_grid, mean_tpr, color=col, ls=ls, lw=2.4, label=f"{short_lbl} (AUC = {mean_auc:.3f}{extra})")
        if "frac_N3" in mname:
            ax.fill_between(
                fpr_grid,
                np.clip(mean_tpr - sd_tpr, 0, 1),
                np.clip(mean_tpr + sd_tpr, 0, 1),
                color=col,
                alpha=0.13,
            )

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("D. Out-of-Fold CV ROC (Dementia vs. Normal)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.3, frameon=False)

    fig.tight_layout(pad=1.6)
    out_png = out_dir / "final_composite_2x2.png"
    out_pdf = out_dir / "final_composite_2x2.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nSaved composite figure & tables to:\n  {out_png}\n  {out_pdf}\n  {out_dir / 'final_report_summary.txt'}")


if __name__ == "__main__":
    sys.exit(main())
