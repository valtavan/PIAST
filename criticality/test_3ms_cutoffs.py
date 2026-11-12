#!/usr/bin/env python
"""
test_3ms_cutoffs.py
Run from ~/pl1358-01/project_data/code using already-extracted features in:
    ~/pl1358-01/project_data/worksof/criticality/high0.5/classify_dementia_multiq/multiq_features_by_participant.csv

Usage:
    cd ~/pl1358-01/project_data/code
    uv run --frozen python test_3ms_cutoffs.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path("~/pl1358-01/project_data").expanduser()
DEFAULT_DATA_DIR = PROJECT / "worksof/criticality/high0.5/classify_dementia_multiq"
GROUPS = ["Normal", "MCI", "Dementia"]


def epoch_auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ex = lambda s: Path(s).expanduser()
    ap.add_argument("--data-dir", type=ex, default=DEFAULT_DATA_DIR, help="dir with multiq_features_by_participant.csv")
    ap.add_argument("--out-dir", type=ex, default=None, help="output dir (default: same as --data-dir)")
    args = ap.parse_args()

    in_csv = args.data_dir / "multiq_features_by_participant.csv"
    if not in_csv.exists():
        sys.exit(f"Could not find {in_csv}. Pass --data-dir to point to the folder containing it.")

    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    P = pd.read_csv(in_csv)
    req_cols = ["teng", "h_q+4__auc_N2", "h_q+2__auc_N2", "frac_N3", "age"]
    comp = P[P[req_cols].notna().all(axis=1)].copy()
    print(f"Loaded {in_csv} (complete cases: n={len(comp)})\n")

    cutoff_grid = [
        (76, 88, "Strict (<76 / 88)"),
        (78, 88, "Teng-Chui (<78 / 88)"),
        (80, 88, "Baseline (<80 / 88)"),
        (80, 90, "Alt MCI (<80 / 90)"),
        (82, 88, "Broad-82 (<82 / 88)"),
        (84, 90, "Broad-84 (<84 / 90)"),
        (85, 90, "Broad-85 (<85 / 90)"),
    ]

    rows = []
    n_splits, n_repeats = 5, 20

    for dem_cut, mci_cut, label in cutoff_grid:
        df = comp.copy()
        df["group"] = np.select(
            [df["teng"] < dem_cut, df["teng"] < mci_cut, df["teng"] >= mci_cut],
            ["Dementia", "MCI", "Normal"],
            default="",
        )
        n_norm = int((df["group"] == "Normal").sum())
        n_mci = int((df["group"] == "MCI").sum())
        n_dem = int((df["group"] == "Dementia").sum())

        if min(n_norm, n_mci, n_dem) < n_splits:
            continue

        # Univariate Dementia vs Normal AUC for h_q+4__auc_N2
        dn = df[df["group"].isin(["Dementia", "Normal"])]
        pos = dn.loc[dn["group"] == "Dementia", "h_q+4__auc_N2"].to_numpy(float)
        neg = dn.loc[dn["group"] == "Normal", "h_q+4__auc_N2"].to_numpy(float)
        raw_u = epoch_auc(pos, neg)
        uni_auc = max(raw_u, 1.0 - raw_u)
        uni_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)

        # Repeated 5-Fold Stratified CV: Age_only vs Base (h_q+4 auc_N2 + frac_N3 + age)
        y_multi = df["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
        y_dem = (df["group"] == "Dementia").astype(int).to_numpy()
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

        dn_age, dn_base, dr_base, ovo_base = [], [], [], []
        X_age = df[["age"]].to_numpy(float)
        X_base = df[["h_q+4__auc_N2", "frac_N3", "age"]].to_numpy(float)

        for tr, te in rskf.split(df, y_multi):
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

        delta_age, p_vs_age = corrected_resampled_ttest(np.array(dn_base) - np.array(dn_age), n_splits)

        rows.append({
            "cutoff_label": label,
            "dem_cut": dem_cut,
            "mci_cut": mci_cut,
            "n_Norm": n_norm,
            "n_MCI": n_mci,
            "n_Dem": n_dem,
            "Uni_h_q4_AUC": uni_auc,
            "Uni_p": uni_p,
            "CV_Age_DemNorm_AUC": float(np.mean(dn_age)),
            "CV_Base_DemNorm_AUC": float(np.mean(dn_base)),
            "Gain_over_Age": delta_age,
            "p_vs_Age": p_vs_age,
            "CV_Base_DemRest_AUC": float(np.mean(dr_base)),
            "CV_Base_3Class_OvO": float(np.mean(ovo_base)),
        })

    res = pd.DataFrame(rows)
    out_csv = out_dir / "cutoff_sensitivity.csv"
    res.to_csv(out_csv, index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot sensitivity across cutoffs
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(res))
    ax.plot(x, res["CV_Base_DemNorm_AUC"], marker="o", lw=2.3, color="#D55E00", label="CV Base: h_q+4(auc_N2) + frac_N3 + age")
    ax.plot(x, res["Uni_h_q4_AUC"], marker="s", lw=1.8, ls="-.", color="#009E73", label="Univariate: h_q+4(auc_N2) alone")
    ax.plot(x, res["CV_Age_DemNorm_AUC"], marker="^", lw=1.8, ls="--", color="#777777", label="CV Age alone")
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.cutoff_label}\n(n_Dem={r.n_Dem})" for _, r in res.iterrows()], fontsize=8.5)
    ax.set_ylabel("Dementia vs. Normal ROC-AUC")
    ax.set_ylim(0.50, 0.80)
    ax.set_title("Sensitivity of Dementia Classification Across 3MS Cutoffs", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    out_png = out_dir / "cutoff_sensitivity.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"\nWritten to:\n  {out_csv}\n  {out_png}")


if __name__ == "__main__":
    main()
