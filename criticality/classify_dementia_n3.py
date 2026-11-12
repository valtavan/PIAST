#!/usr/bin/env python
"""
classify_dementia_n3.py - Classify Normal vs MCI vs Dementia using N3 H(q=+2) features and stage contrasts.

Outputs in <out-dir> (default <h-dir>/classify_dementia_n3):
    features_by_participant.csv
    univariate_classification_auc.csv
    cv_multiclass_results.csv
    roc_n3_classification.png
    summary.txt
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path("~/pl1358-01/project_data").expanduser()
H_DIR = PROJECT / "worksof/criticality/high0.5"
COG_FILE = PROJECT / "worksof/cognitive.csv"
HYP_DIR = PROJECT / "worksof/hypnograms/expert"

CODE = {0: "wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CHANNELS = ["C3A1", "C3A2", "C4A1", "C4A2"]
HCOL, RCOL = "h_q+2", "r2_q+2"
GROUPS = ["Normal", "MCI", "Dementia"]


def load_participant(f, channels, min_r2):
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
    ids = df[cols[id_col.lower()]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    y = pd.to_numeric(df[cols[col.lower()]], errors="coerce")
    out = pd.DataFrame({"teng": y.where((y >= 0) & (y <= 100)).to_numpy()}, index=ids)
    out["age"] = pd.to_numeric(df[cols[age_col.lower()]], errors="coerce").to_numpy() if age_col else np.nan
    return out[~out.index.duplicated()]


def epoch_auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def extract_features(sid, df, min_epochs):
    H = df["H"].to_numpy(float)
    st = df["stage"].to_numpy()
    n_tot = len(df)
    n3 = H[st == "N3"]
    n2 = H[st == "N2"]
    rem = H[st == "REM"]
    wake = H[st == "wake"]

    row = {
        "id": sid,
        "n_epochs": n_tot,
        "n_N3": len(n3),
        "frac_N3": len(n3) / n_tot if n_tot > 0 else np.nan,
    }
    if len(n3) >= min_epochs:
        row["H_N3_median"] = float(np.median(n3))
        row["H_N3_mean"] = float(np.mean(n3))
        row["H_N3_iqr"] = float(np.percentile(n3, 75) - np.percentile(n3, 25))
        row["H_N3_q90"] = float(np.percentile(n3, 90))
    else:
        for k in ["H_N3_median", "H_N3_mean", "H_N3_iqr", "H_N3_q90"]:
            row[k] = np.nan

    if len(n3) >= min_epochs and len(n2) >= min_epochs:
        row["H_N2_median"] = float(np.median(n2))
        row["H_N3_minus_N2"] = row["H_N3_median"] - row["H_N2_median"]
        row["auc_N2"] = epoch_auc(n3, n2)
    else:
        row["H_N2_median"] = np.nan
        row["H_N3_minus_N2"] = np.nan
        row["auc_N2"] = np.nan

    row["auc_REM"] = epoch_auc(n3, rem) if (len(n3) >= min_epochs and len(rem) >= min_epochs) else np.nan
    row["auc_wake"] = epoch_auc(n3, wake) if (len(n3) >= min_epochs and len(wake) >= min_epochs) else np.nan
    return row


def bootstrap_binary_auc(y_true, scores, rng, n_boot=2000):
    """Returns (auc_oriented, raw_auc, ci_lo, ci_hi, p_val)."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    raw_auc = epoch_auc(pos, neg)
    # Orient so AUC >= 0.5 reflects discriminative power regardless of sign
    flip = raw_auc < 0.5
    s_orient = -scores if flip else scores
    auc_val = 1.0 - raw_auc if flip else raw_auc
    mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")

    boots = []
    for _ in range(n_boot):
        bp = pos[rng.integers(0, len(pos), len(pos))]
        bn = neg[rng.integers(0, len(neg), len(neg))]
        b_auc = epoch_auc(bp, bn)
        boots.append(1.0 - b_auc if flip else b_auc)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    direction = "lower in positive class" if flip else "higher in positive class"
    return auc_val, raw_auc, float(lo), float(hi), float(mw.pvalue), direction


def evaluate_cv_models(df, feature_sets, rng_seed=0, n_splits=5, n_repeats=20):
    """Repeated stratified CV for 3-class (Normal, MCI, Dementia) and binary Dementia vs Non-Dementia."""
    y_multi = df["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
    y_dem = (df["group"] == "Dementia").astype(int).to_numpy()
    y_imp = (df["group"].isin(["MCI", "Dementia"])).astype(int).to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rng_seed)
    rows = []

    for name, cols in feature_sets.items():
        X = df[cols].to_numpy(float)
        ovr_aucs, ovo_aucs, dem_aucs, imp_aucs, bal_accs = [], [], [], [], []

        for train_idx, test_idx in rskf.split(X, y_multi):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
            )
            clf.fit(X[train_idx], y_multi[train_idx])
            probs = clf.predict_proba(X[test_idx])
            preds = clf.predict(X[test_idx])

            ovr_aucs.append(roc_auc_score(y_multi[test_idx], probs, multi_class="ovr", average="macro"))
            ovo_aucs.append(roc_auc_score(y_multi[test_idx], probs, multi_class="ovo", average="macro"))
            dem_aucs.append(roc_auc_score(y_dem[test_idx], probs[:, 2]))
            imp_aucs.append(roc_auc_score(y_imp[test_idx], probs[:, 1] + probs[:, 2]))
            bal_accs.append(balanced_accuracy_score(y_multi[test_idx], preds))

        rows.append({
            "model": name,
            "features": ", ".join(cols),
            "n": len(df),
            "cv_3class_macro_ovr_auc": float(np.mean(ovr_aucs)),
            "cv_3class_ovr_sd": float(np.std(ovr_aucs)),
            "cv_3class_hand_till_ovo_auc": float(np.mean(ovo_aucs)),
            "cv_dementia_vs_rest_auc": float(np.mean(dem_aucs)),
            "cv_impaired_vs_normal_auc": float(np.mean(imp_aucs)),
            "cv_3class_balanced_acc": float(np.mean(bal_accs)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ex = lambda s: Path(s).expanduser()
    ap.add_argument("--h-dir", type=ex, default=H_DIR)
    ap.add_argument("--out-dir", type=ex, default=None)
    ap.add_argument("--hyp-dir", type=ex, default=HYP_DIR)
    ap.add_argument("--cog-file", type=ex, default=COG_FILE)
    ap.add_argument("--cog-col", default="V9TENG")
    ap.add_argument("--cog-id-col", default="ID")
    ap.add_argument("--age-col", default="V8AGE")
    ap.add_argument("--dementia-cut", type=float, default=80)
    ap.add_argument("--mci-cut", type=float, default=88)
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--min-r2", type=float, default=0.0)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = args.out_dir or (args.h_dir / "classify_dementia_n3")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cog = load_cog(args.cog_file, args.cog_col, args.cog_id_col, args.age_col)
    rows = []
    for f in sorted(args.h_dir.glob("*.parquet")):
        sid = f.stem.zfill(5)
        d = load_participant(f, args.channels, args.min_r2)
        if d is None:
            continue
        d = add_stage(d, sid, args.hyp_dir)
        if d is None or d.empty:
            continue
        rows.append(extract_features(sid, d, args.min_epochs))

    P = pd.DataFrame(rows).set_index("id").join(cog, how="left")
    P["group"] = np.select(
        [P["teng"] < args.dementia_cut, P["teng"] < args.mci_cut, P["teng"] >= args.mci_cut],
        ["Dementia", "MCI", "Normal"],
        default="",
    )
    P["group"] = P["group"].replace("", np.nan)
    P.reset_index().to_csv(out / "features_by_participant.csv", index=False)

    # 1. Univariate between-group ROC-AUCs (N3_normal vs N3_MCI vs N3_dementia)
    cand_features = ["H_N3_median", "H_N3_mean", "H_N3_iqr", "H_N3_minus_N2", "auc_N2", "auc_REM", "frac_N3", "age"]
    contrasts = [
        ("Dementia vs Normal", "Dementia", ["Normal"]),
        ("MCI vs Normal", "MCI", ["Normal"]),
        ("Dementia vs MCI", "Dementia", ["MCI"]),
        ("Dementia vs Non-Dementia", "Dementia", ["Normal", "MCI"]),
        ("Impaired (MCI+Dem) vs Normal", ["MCI", "Dementia"], ["Normal"]),
    ]

    uni_rows = []
    for feat in cand_features:
        for cname, pos_g, neg_g in contrasts:
            pos_list = [pos_g] if isinstance(pos_g, str) else pos_g
            sub = P[P["group"].isin(pos_list + neg_g) & P[feat].notna()].copy()
            y_bin = sub["group"].isin(pos_list).astype(int).to_numpy()
            scores = sub[feat].to_numpy(float)
            auc_val, raw_auc, lo, hi, pval, direction = bootstrap_binary_auc(y_bin, scores, rng, args.n_boot)
            uni_rows.append({
                "feature": feat,
                "contrast": cname,
                "n_pos": int(y_bin.sum()),
                "n_neg": int((1 - y_bin).sum()),
                "roc_auc": auc_val,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_mann_whitney": pval,
                "direction": direction,
            })
    uni_df = pd.DataFrame(uni_rows)
    uni_df.to_csv(out / "univariate_classification_auc.csv", index=False)

    # 2. Multivariable Repeated Stratified CV (complete cases across core features)
    core_cols = ["H_N3_median", "H_N3_iqr", "H_N3_minus_N2", "auc_N2", "frac_N3", "age"]
    complete = P[P["group"].notna() & P[core_cols].notna().all(axis=1)].copy()
    feature_sets = {
        "1_N3_raw_only": ["H_N3_median", "H_N3_iqr"],
        "2_N3_within_night_contrast": ["auc_N2", "H_N3_minus_N2"],
        "3_N3_raw_plus_contrast": ["H_N3_median", "H_N3_iqr", "auc_N2"],
        "4_N3_quantity_only": ["frac_N3"],
        "5_Age_only": ["age"],
        "6_Age_plus_N3_EEG": ["age", "H_N3_median", "H_N3_iqr", "auc_N2"],
    }
    cv_df = evaluate_cv_models(complete, feature_sets, rng_seed=args.seed)
    cv_df.to_csv(out / "cv_multiclass_results.csv", index=False)

    # Summary text
    lines = [
        f"Participants with group & >= {args.min_epochs} N3 epochs: {int((P['group'].notna() & P['H_N3_median'].notna()).sum())}",
        f"Complete-case sample for CV models: {len(complete)} "
        f"(Normal={int((complete['group']=='Normal').sum())}, "
        f"MCI={int((complete['group']=='MCI').sum())}, "
        f"Dementia={int((complete['group']=='Dementia').sum())})",
        "",
        "=== Univariate ROC-AUC: Dementia vs Normal ===",
        uni_df[uni_df["contrast"] == "Dementia vs Normal"]
        .sort_values("roc_auc", ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== Univariate ROC-AUC: Pairwise N3_normal vs N3_MCI vs N3_dementia (H_N3_median & auc_N2) ===",
        uni_df[uni_df["feature"].isin(["H_N3_median", "H_N3_minus_N2", "auc_N2"])]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== Repeated 5-Fold Stratified CV (Multiclass & Binary ROC-AUC) ===",
        cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
    ]
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt + "\n")

    # ROC Plot for Normal vs MCI vs Dementia
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        pair_specs = [
            ("Dementia", "Normal", "Dementia vs Normal"),
            ("MCI", "Normal", "MCI vs Normal"),
            ("Dementia", "MCI", "Dementia vs MCI"),
        ]
        for ax, feat, title in zip(axes, ["H_N3_median", "auc_N2"], ["Raw N3 H(q=+2)", "Within-Night N3 vs N2 AUC"]):
            for g_pos, g_neg, label in pair_specs:
                sub = P[P["group"].isin([g_pos, g_neg]) & P[feat].notna()]
                y_b = (sub["group"] == g_pos).astype(int).to_numpy()
                s = -sub[feat].to_numpy(float)  # lower values indicate impairment
                fpr, tpr, _ = roc_curve(y_b, s)
                val = roc_auc_score(y_b, s)
                ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {val:.3f})")
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            ax.set_title(title)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "roc_n3_classification.png", dpi=150)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
