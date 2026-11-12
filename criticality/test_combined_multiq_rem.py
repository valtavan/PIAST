#!/usr/bin/env python
"""
test_combined_multiq_rem.py
Evaluates combining h_q+4 + h_q-2 + auc_REM + frac_N3 + age on existing multiq_features_by_participant.csv.

Outputs in <out-dir> (default <h-dir>/classify_dementia_multiq):
    combined_multiq_rem_cv.csv
    combined_multiq_rem_plot.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("~/pl1358-01/project_data/worksof/criticality/high0.5/classify_dementia_multiq").expanduser()
GROUPS = ["Normal", "MCI", "Dementia"]


def corrected_resampled_ttest(diffs, n_splits=5):
    """Nadeau & Bengio (2003) corrected paired t-test for repeated k-fold CV."""
    diffs = np.asarray(diffs, float)
    m = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    if var_d <= 1e-16:
        return mean_d, np.nan, np.nan
    test_train_ratio = 1.0 / (n_splits - 1)
    se = np.sqrt((1.0 / m + test_train_ratio) * var_d)
    t_stat = mean_d / se
    p_val = float(2 * stats.t.sf(abs(t_stat), df=m - 1))
    return mean_d, float(t_stat), p_val


def main():
    P = pd.read_csv(OUT_DIR / "multiq_features_by_participant.csv")

    models = {
        "1_Age_only": ["age"],
        "2_Base: h_q+4(auc_N2) + frac_N3 + age": ["h_q+4__auc_N2", "frac_N3", "age"],
        "3_Add_q-2: h_q+4 + h_q-2 + frac_N3 + age": ["h_q+4__auc_N2", "h_q-2__auc_N2", "frac_N3", "age"],
        "4_Add_REM: h_q+4(auc_N2 + auc_REM) + frac_N3 + age": ["h_q+4__auc_N2", "h_q+4__auc_REM", "frac_N3", "age"],
        "5_Combined: h_q+4 + h_q-2 + auc_REM + frac_N3 + age": [
            "h_q+4__auc_N2", "h_q-2__auc_N2", "h_q+4__auc_REM", "frac_N3", "age"
        ],
        "6_Full_MultiQ_REM: + h_q-2(auc_REM)": [
            "h_q+4__auc_N2", "h_q-2__auc_N2", "h_q+4__auc_REM", "h_q-2__auc_REM", "frac_N3", "age"
        ],
        "7_EEG_only (no age): h_q+4 + h_q-2 + auc_REM + frac_N3": [
            "h_q+4__auc_N2", "h_q-2__auc_N2", "h_q+4__auc_REM", "frac_N3"
        ],
    }

    all_cols = list(dict.fromkeys([c for cols in models.values() for c in cols]))
    comp = P[P["group"].notna() & P[all_cols].notna().all(axis=1)].copy()
    y_multi = comp["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
    y_dem = (comp["group"] == "Dementia").astype(int).to_numpy()
    y_imp = (comp["group"].isin(["MCI", "Dementia"])).astype(int).to_numpy()

    print(f"Complete-case sample across all features: n={len(comp)} "
          f"(Normal={(comp['group']=='Normal').sum()}, "
          f"MCI={(comp['group']=='MCI').sum()}, "
          f"Dementia={(comp['group']=='Dementia').sum()})\n")

    # Feature correlation check
    print("=== Feature Spearman Correlations ===")
    corr_cols = ["h_q+4__auc_N2", "h_q-2__auc_N2", "h_q+4__auc_REM", "frac_N3", "age"]
    print(comp[corr_cols].corr(method="spearman").round(3).to_string())
    print()

    # Repeated 5-Fold CV
    n_splits, n_repeats = 5, 20
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    fold_dn, fold_dr, fold_ovo, fold_imp, fold_bal = {k: [] for k in models}, {k: [] for k in models}, {k: [] for k in models}, {k: [] for k in models}, {k: [] for k in models}
    fpr_grid = np.linspace(0, 1, 101)
    roc_tprs = {k: [] for k in models}

    for tr, te in rskf.split(comp, y_multi):
        for name, cols in models.items():
            X = comp[cols].to_numpy(float)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
            clf.fit(X[tr], y_multi[tr])
            probs = clf.predict_proba(X[te])
            preds = clf.predict(X[te])

            fold_ovo[name].append(roc_auc_score(y_multi[te], probs, multi_class="ovo", average="macro"))
            fold_dr[name].append(roc_auc_score(y_dem[te], probs[:, 2]))
            fold_imp[name].append(roc_auc_score(y_imp[te], probs[:, 1] + probs[:, 2]))
            fold_bal[name].append(balanced_accuracy_score(y_multi[te], preds))

            m_dn = np.isin(y_multi[te], [0, 2])
            y_dn = (y_multi[te][m_dn] == 2).astype(int)
            p_dn = probs[m_dn, 2] / np.clip(probs[m_dn, 0] + probs[m_dn, 2], 1e-12, None)
            fold_dn[name].append(roc_auc_score(y_dn, p_dn))

            fpr, tpr, _ = roc_curve(y_dn, p_dn)
            roc_tprs[name].append(np.interp(fpr_grid, fpr, tpr))

    base_name = "2_Base: h_q+4(auc_N2) + frac_N3 + age"
    summary_rows = []
    for name in models:
        d_dn, t_dn, p_dn = corrected_resampled_ttest(np.array(fold_dn[name]) - np.array(fold_dn[base_name]), n_splits)
        d_dr, _, p_dr = corrected_resampled_ttest(np.array(fold_dr[name]) - np.array(fold_dr[base_name]), n_splits)
        summary_rows.append({
            "model": name,
            "Dem_vs_Norm_AUC": float(np.mean(fold_dn[name])),
            "Delta_vs_Base": d_dn if name != base_name else 0.0,
            "p_vs_Base": p_dn if name != base_name else np.nan,
            "Dem_vs_Rest_AUC": float(np.mean(fold_dr[name])),
            "Imp_vs_Norm_AUC": float(np.mean(fold_imp[name])),
            "3Class_OvO_AUC": float(np.mean(fold_ovo[name])),
            "3Class_Bal_Acc": float(np.mean(fold_bal[name])),
        })

    res_df = pd.DataFrame(summary_rows)
    res_df.to_csv(OUT_DIR / "combined_multiq_rem_cv.csv", index=False)
    print("=== Repeated 5-Fold Stratified CV Results ===")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Standardized Logistic Regression Coefficients for Model 5 (Dementia vs Normal)
    comb_cols = models["5_Combined: h_q+4 + h_q-2 + auc_REM + frac_N3 + age"]
    dn_sub = comp[comp["group"].isin(["Normal", "Dementia"])].copy()
    X_dn = StandardScaler().fit_transform(dn_sub[comb_cols].to_numpy(float))
    y_dn_all = (dn_sub["group"] == "Dementia").astype(int).to_numpy()
    lr_dn = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X_dn, y_dn_all)
    coefs = lr_dn.coef_[0]

    print("\n=== Standardized Betas & Odds Ratios (Model 5: Dementia vs Normal) ===")
    for c, b in zip(comb_cols, coefs):
        print(f"{c:18s} | beta = {b:+.4f} | OR per 1 SD = {np.exp(b):.3f}")

    # Plot ROC + Standardized Weights
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    plot_specs = [
        ("1_Age_only", "#999999", "--", "Age only"),
        ("2_Base: h_q+4(auc_N2) + frac_N3 + age", "#0072B2", "-.", "Base: h_q+4(N2) + frac_N3 + age"),
        ("5_Combined: h_q+4 + h_q-2 + auc_REM + frac_N3 + age", "#D55E00", "-", "Combined: h_q+4 + h_q-2 + REM + frac_N3 + age"),
    ]
    for mname, col, ls, label in plot_specs:
        mean_tpr = np.mean(roc_tprs[mname], axis=0)
        mean_auc = float(np.mean(fold_dn[mname]))
        ax.plot(fpr_grid, mean_tpr, color=col, ls=ls, lw=2.2, label=f"{label} (AUC={mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. Out-of-Fold CV ROC (Dementia vs. Normal)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    ax = axes[1]
    y_pos = np.arange(len(comb_cols))
    colors = ["#D55E00" if b > 0 else "#0072B2" for b in coefs]
    ax.barh(y_pos, coefs, color=colors, alpha=0.85, height=0.55)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comb_cols)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized Log-Odds Coefficient (Beta)")
    ax.set_title("B. Combined Model Feature Weights (Dementia vs. Normal)", loc="left", fontweight="bold")

    fig.tight_layout()
    out_png = OUT_DIR / "combined_multiq_rem_plot.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"\nSaved table & figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
