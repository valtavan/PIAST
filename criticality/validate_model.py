#!/usr/bin/env python
"""
validate_final_model.py
Final validation and reporting checks for the 3-feature model:
    h_q+4(auc_N2) + frac_N3 + age

1. Exact Odds Ratios (standardized per 1 SD & clinically scaled) with 95% CIs and Wald p-values.
2. Out-of-fold sensitivity, specificity, PPV, NPV at Youden optimal, >=80% Sens, and >=80% Spec cut-offs.
3. Sensitivity analysis across MFDFA fit quality (r2 >= 0.0 vs 0.8) and epoch thresholds (min-epochs 5, 10, 20).

Usage:
    cd ~/pl1358-01/project_data/code
    uv run --frozen python validate_final_model.py
    # Or shift the primary OR/operating-point cohort to min-epochs 5:
    uv run --frozen python validate_final_model.py --min-epochs 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
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


# ----------------------------------------------------------------------------- data loading
def load_cog(path, col="V9TENG", id_col="ID", age_col="V8AGE"):
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    ids = df[cols[id_col.lower()]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    y = pd.to_numeric(df[cols[col.lower()]], errors="coerce")
    out = pd.DataFrame({"teng": y.where((y >= 0) & (y <= 100)).to_numpy()}, index=ids)
    out["age"] = pd.to_numeric(df[cols[age_col.lower()]], errors="coerce").to_numpy()
    return out[~out.index.duplicated()]


def load_all_epochs(h_dir, hyp_dir, channels, q_col="h_q+4", r_col="r2_q+4"):
    """Load epoch-level H and R2 once so filtering regimes run in milliseconds."""
    cache = {}
    for f in sorted(h_dir.glob("*.parquet")):
        sid = f.stem.zfill(5)
        hf = hyp_dir / f"hypnogram_{sid}.csv"
        if not hf.exists():
            continue
        df = pd.read_parquet(f, columns=["epoch_idx", "window_start_time_s", "channel", "status", q_col, r_col])
        df = df[df["channel"].isin(channels)].copy()
        if df.empty:
            continue

        ok = df["status"] == "ok"
        df["h_val"] = df[q_col].where(ok, np.nan)
        df["r_val"] = df[r_col].where(ok, np.nan)

        wide_h = df.pivot(index="epoch_idx", columns="channel", values="h_val").reindex(columns=channels)
        wide_r = df.pivot(index="epoch_idx", columns="channel", values="r_val").reindex(columns=channels)
        valid = wide_h.notna().all(axis=1) & wide_r.notna().all(axis=1)
        if not valid.any():
            continue

        wide_h = wide_h.loc[valid]
        wide_r = wide_r.loc[valid]
        t = df.drop_duplicates("epoch_idx").set_index("epoch_idx")["window_start_time_s"].reindex(wide_h.index)

        h_hyp = pd.read_csv(hf)
        m = dict(zip(h_hyp["start_sec"].round().astype(int), h_hyp["stage_code"].astype(int)))
        stages = np.array([CODE.get(m.get(int(round(ts)), -1), "") for ts in t.to_numpy()])
        keep = stages != ""
        if not keep.any():
            continue

        cache[sid] = {
            "H": wide_h.mean(axis=1).to_numpy(float)[keep],
            "min_r2": wide_r.min(axis=1).to_numpy(float)[keep],
            "stage": stages[keep],
        }
    return cache


def epoch_auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_cohort(cache, cog, min_r2=0.0, min_epochs=10, dem_cut=80, mci_cut=88):
    rows = []
    for sid, d in cache.items():
        if min_r2 > 0:
            m = d["min_r2"] >= min_r2
            H, st = d["H"][m], d["stage"][m]
        else:
            H, st = d["H"], d["stage"]

        n_tot = len(st)
        if n_tot == 0:
            continue
        n3 = H[st == "N3"]
        n2 = H[st == "N2"]
        auc_n2 = epoch_auc(n3, n2) if (len(n3) >= min_epochs and len(n2) >= min_epochs) else np.nan
        rows.append({
            "id": sid,
            "n_epochs": n_tot,
            "n_N3": len(n3),
            "n_N2": len(n2),
            "frac_N3": len(n3) / n_tot,
            "h_q+4__auc_N2": auc_n2,
        })

    df = pd.DataFrame(rows).set_index("id").join(cog, how="left")
    df["group"] = np.select(
        [df["teng"] < dem_cut, df["teng"] < mci_cut, df["teng"] >= mci_cut],
        ["Dementia", "MCI", "Normal"],
        default="",
    )
    df["group"] = df["group"].replace("", np.nan)
    return df


# ----------------------------------------------------------------------------- statistical helpers
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
            "mean": float(means[j - 1]),
            "sd": float(sds[j - 1]),
            "std_beta": float(b_s),
            "std_OR": float(np.exp(b_s)),
            "std_OR_lo": float(np.exp(b_s - 1.96 * se_s)),
            "std_OR_hi": float(np.exp(b_s + 1.96 * se_s)),
            "clinical_step": unit_label,
            "clin_OR": float(np.exp(b_c)),
            "clin_OR_lo": float(np.exp(b_c - 1.96 * se_c)),
            "clin_OR_hi": float(np.exp(b_c + 1.96 * se_c)),
            "wald_z": float(z_val),
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


def run_cv_regime(df, n_splits=5, n_repeats=20, seed=0):
    cols = ["h_q+4__auc_N2", "frac_N3", "age"]
    sub = df[df["group"].notna() & df[cols].notna().all(axis=1)].copy()
    y_multi = sub["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
    y_dem = (sub["group"] == "Dementia").astype(int).to_numpy()

    X_base = sub[cols].to_numpy(float)
    X_age = sub[["age"]].to_numpy(float)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    dn_base, dn_age, dr_base, ovo_base = [], [], [], []
    oof_p_dem_sum = np.zeros(len(sub))
    oof_counts = np.zeros(len(sub))

    for tr, te in rskf.split(sub, y_multi):
        m_dn = np.isin(y_multi[te], [0, 2])
        y_dn_te = (y_multi[te][m_dn] == 2).astype(int)

        clf_a = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
        clf_a.fit(X_age[tr], y_multi[tr])
        pa = clf_a.predict_proba(X_age[te])
        dn_age.append(roc_auc_score(y_dn_te, pa[m_dn, 2] / np.clip(pa[m_dn, 0] + pa[m_dn, 2], 1e-12, None)))

        clf_b = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
        clf_b.fit(X_base[tr], y_multi[tr])
        pb = clf_b.predict_proba(X_base[te])
        dn_base.append(roc_auc_score(y_dn_te, pb[m_dn, 2] / np.clip(pb[m_dn, 0] + pb[m_dn, 2], 1e-12, None)))
        dr_base.append(roc_auc_score(y_dem[te], pb[:, 2]))
        ovo_base.append(roc_auc_score(y_multi[te], pb, multi_class="ovo", average="macro"))

        oof_p_dem_sum[te] += pb[:, 2]
        oof_counts[te] += 1

    sub["oof_prob_dementia"] = oof_p_dem_sum / oof_counts
    delta_age, p_vs_age = corrected_resampled_ttest(np.array(dn_base) - np.array(dn_age), n_splits)

    dn_sub = sub[sub["group"].isin(["Dementia", "Normal"])]
    pos = dn_sub.loc[dn_sub["group"] == "Dementia", "h_q+4__auc_N2"].to_numpy(float)
    neg = dn_sub.loc[dn_sub["group"] == "Normal", "h_q+4__auc_N2"].to_numpy(float)
    raw_u = epoch_auc(pos, neg)
    uni_auc = max(raw_u, 1.0 - raw_u)
    uni_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)

    metrics = {
        "n_total": len(sub),
        "n_Norm": int((sub["group"] == "Normal").sum()),
        "n_MCI": int((sub["group"] == "MCI").sum()),
        "n_Dem": int((sub["group"] == "Dementia").sum()),
        "Uni_h_q4_AUC": uni_auc,
        "Uni_p": uni_p,
        "CV_Age_DemNorm_AUC": float(np.mean(dn_age)),
        "CV_Base_DemNorm_AUC": float(np.mean(dn_base)),
        "Gain_over_Age": delta_age,
        "p_vs_Age": p_vs_age,
        "CV_Base_DemRest_AUC": float(np.mean(dr_base)),
        "CV_Base_3Class_OvO": float(np.mean(ovo_base)),
    }
    return sub, metrics


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ex = lambda s: Path(s).expanduser()
    ap.add_argument("--h-dir", type=ex, default=H_DIR)
    ap.add_argument("--hyp-dir", type=ex, default=HYP_DIR)
    ap.add_argument("--cog-file", type=ex, default=COG_FILE)
    ap.add_argument("--out-dir", type=ex, default=None)
    ap.add_argument("--min-r2", type=float, default=0.0, help="primary cohort min R2 (default 0.0)")
    ap.add_argument("--min-epochs", type=int, default=10, help="primary cohort min epochs (default 10)")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.h_dir / "final_model_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading epoch-level MFDFA & hypnogram data...")
    cog = load_cog(args.cog_file)
    cache = load_all_epochs(args.h_dir, args.hyp_dir, CHANNELS, q_col="h_q+4", r_col="r2_q+4")
    print(f"Cached {len(cache)} participants with valid 4-channel h_q+4 & r2_q+4 epochs.\n")

    # 1. Exact Odds Ratios on Primary Cohort
    base_df = build_cohort(cache, cog, min_r2=args.min_r2, min_epochs=args.min_epochs)
    feature_specs = [
        ("h_q+4__auc_N2", -0.05, "per -0.05 drop in N3-N2 AUC"),
        ("frac_N3", +0.05, "per +5% (+0.05) increase in N3 fraction"),
        ("age", +5.0, "per +5 years of age"),
    ]

    dn_df = base_df[base_df["group"].isin(["Normal", "Dementia"])].copy()
    dn_df["y_dem"] = (dn_df["group"] == "Dementia").astype(int)
    or_dn = fit_logistic_with_or(dn_df, "y_dem", feature_specs)
    or_dn.insert(0, "outcome", "Dementia vs Normal")

    all_df = base_df[base_df["group"].notna()].copy()
    all_df["y_dem"] = (all_df["group"] == "Dementia").astype(int)
    or_dr = fit_logistic_with_or(all_df, "y_dem", feature_specs)
    or_dr.insert(0, "outcome", "Dementia vs Non-Dementia (Normal+MCI)")

    or_table = pd.concat([or_dn, or_dr], ignore_index=True)
    or_table.to_csv(out_dir / "model_odds_ratios.csv", index=False)

    # 2. Out-of-Fold Sensitivity / Specificity Operating Thresholds
    oof_sub, _ = run_cv_regime(base_df, n_splits=5, n_repeats=20, seed=0)
    oof_dn = oof_sub[oof_sub["group"].isin(["Normal", "Dementia"])]
    op_dn = evaluate_operating_points(
        (oof_dn["group"] == "Dementia").astype(int).to_numpy(),
        oof_dn["oof_prob_dementia"].to_numpy(),
    )
    op_dn.insert(0, "comparison", "Dementia vs Normal")

    op_dr = evaluate_operating_points(
        (oof_sub["group"] == "Dementia").astype(int).to_numpy(),
        oof_sub["oof_prob_dementia"].to_numpy(),
    )
    op_dr.insert(0, "comparison", "Dementia vs Non-Dementia")

    op_table = pd.concat([op_dn, op_dr], ignore_index=True)
    op_table.to_csv(out_dir / "operating_thresholds_oof.csv", index=False)

    # 3. Robustness Grid: min-epochs 5, 10, 20 x min-r2 0.0, 0.8
    regimes = [
        ("1_Lenient_Epochs (r2>=0.0, epochs>=5)", 0.0, 5),
        ("2_Baseline (r2>=0.0, epochs>=10)", 0.0, 10),
        ("3_Strict_Epochs (r2>=0.0, epochs>=20)", 0.0, 20),
        ("4_Fit0.8_Epochs5 (r2>=0.8, epochs>=5)", 0.8, 5),
        ("5_Fit0.8_Epochs10 (r2>=0.8, epochs>=10)", 0.8, 10),
        ("6_Fit0.8_Epochs20 (r2>=0.8, epochs>=20)", 0.8, 20),
    ]
    rob_rows = []
    for label, r2_cut, ep_cut in regimes:
        rdf = build_cohort(cache, cog, min_r2=r2_cut, min_epochs=ep_cut)
        _, m = run_cv_regime(rdf, n_splits=5, n_repeats=20, seed=0)
        m["regime"] = label
        m["short_label"] = f"r2>={r2_cut:g}\nep>={ep_cut}"
        rob_rows.append(m)

    rob_df = pd.DataFrame(rob_rows)
    cols_order = [
        "regime", "n_Norm", "n_MCI", "n_Dem", "Uni_h_q4_AUC", "Uni_p",
        "CV_Age_DemNorm_AUC", "CV_Base_DemNorm_AUC", "Gain_over_Age", "p_vs_Age",
        "CV_Base_DemRest_AUC", "CV_Base_3Class_OvO",
    ]
    rob_df[cols_order].to_csv(out_dir / "robustness_r2_epochs.csv", index=False)

    # Print formatted summary
    lines = [
        f"Primary Cohort Settings: min_r2={args.min_r2:g}, min_epochs={args.min_epochs}",
        "",
        "=== 1. Exact Multivariable Odds Ratios: h_q+4(auc_N2) + frac_N3 + age ===",
        or_table[[
            "outcome", "feature", "std_OR", "std_OR_lo", "std_OR_hi",
            "clinical_step", "clin_OR", "clin_OR_lo", "clin_OR_hi", "p_value",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 2. Out-of-Fold Classification Operating Points (Repeated 5-Fold CV) ===",
        op_table[[
            "comparison", "operating_point", "sensitivity", "specificity",
            "ppv", "npv", "balanced_acc", "TP", "FP", "TN", "FN",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 3. Robustness Across Fit Quality (r2>=0.0 vs 0.8) & Epoch Thresholds (5, 10, 20) ===",
        rob_df[cols_order].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
    ]
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "final_validation_summary.txt").write_text(txt + "\n")

    # 4. 3-Panel Validation Figure
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel A: Standardized Odds Ratio Forest Plot
    ax = axes[0]
    y_pos = np.arange(len(or_dn))
    ax.errorbar(
        or_dn["std_OR"], y_pos,
        xerr=[or_dn["std_OR"] - or_dn["std_OR_lo"], or_dn["std_OR_hi"] - or_dn["std_OR"]],
        fmt="o", color="#D55E00", ecolor="#D55E00", elinewidth=2, capsize=4, markersize=7,
    )
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["h_q+4 auc_N2\n(per +1 SD)", "frac_N3\n(per +1 SD)", "Age\n(per +1 SD)"])
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted Odds Ratio (95% CI)")
    ax.set_title("A. Multivariable ORs (Dem vs. Norm)", loc="left", fontweight="bold")
    for i, r in or_dn.iterrows():
        ax.text(r["std_OR_hi"] + 0.05, i, f"OR={r['std_OR']:.2f}\np={r['p_value']:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0.35, max(or_dn["std_OR_hi"]) + 0.55)

    # Panel B: Out-of-Fold ROC with Operating Thresholds
    ax = axes[1]
    y_dn_true = (oof_dn["group"] == "Dementia").astype(int).to_numpy()
    y_dn_prob = oof_dn["oof_prob_dementia"].to_numpy()
    fpr, tpr, _ = roc_curve(y_dn_true, y_dn_prob)
    oof_auc = roc_auc_score(y_dn_true, y_dn_prob)
    ax.plot(fpr, tpr, color="#0072B2", lw=2.3, label=f"OOF ROC (AUC = {oof_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    markers = [("o", "#D55E00", "Youden"), ("s", "#009E73", "Sens>=80%"), ("^", "#CC79A7", "Spec>=80%")]
    for (_, row), (mk, col, short_lbl) in zip(op_dn.iterrows(), markers):
        ax.scatter(
            [1.0 - row["specificity"]], [row["sensitivity"]],
            marker=mk, color=col, s=75, zorder=5,
            label=f"{short_lbl}: Se={row['sensitivity']*100:.0f}%, Sp={row['specificity']*100:.0f}%",
        )
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("B. OOF ROC & Operating Cut-offs", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # Panel C: Robustness Across r2 and min-epochs (5, 10, 20)
    ax = axes[2]
    x = np.arange(len(rob_df))
    ax.plot(x, rob_df["CV_Base_DemNorm_AUC"], marker="o", lw=2.3, color="#D55E00", label="Base 3-Feature Model")
    ax.plot(x, rob_df["CV_Age_DemNorm_AUC"], marker="^", lw=1.8, ls="--", color="#777777", label="Age Only")
    ax.axvline(2.5, color="#CCCCCC", ls=":", lw=1.2)
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(rob_df["short_label"], fontsize=8)
    ax.set_ylabel("Dementia vs. Normal CV AUC")
    ax.set_ylim(0.50, 0.78)
    ax.set_title("C. Robustness: Fit R² & Min Epochs (5/10/20)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    out_png = out_dir / "final_model_validation.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"\nSaved all validation tables and plots to:\n  {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
