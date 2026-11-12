#!/usr/bin/env python
"""
classify_dementia_multiq.py
1. Decomposes raw N2 vs N3 H(q) shifts across Normal, MCI, and Dementia.
2. Evaluates clean CV models: auc_N2 + frac_N3 + age (with and without H_N3_minus_N2).
3. Scans all q orders (and multifractal width delta_H) present in the MFDFA parquet files.

Outputs in <out-dir> (default <h-dir>/classify_dementia_multiq):
    stage_h_decomposition.csv
    cv_clean_models.csv
    multiq_classification_scan.csv
    multiq_auc_profile.png
    summary.txt
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path("~/pl1358-01/project_data").expanduser()
H_DIR = PROJECT / "worksof/criticality/high0.5"
COG_FILE = PROJECT / "worksof/cognitive.csv"
HYP_DIR = PROJECT / "worksof/hypnograms/expert"

CODE = {0: "wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CHANNELS = ["C3A1", "C3A2", "C4A1", "C4A2"]
GROUPS = ["Normal", "MCI", "Dementia"]


def parse_q_val(col):
    m = re.search(r"h_q([+-]?\d+(?:\.\d+)?)", col)
    return float(m.group(1)) if m else np.nan


def detect_q_columns(sample_file, requested=None):
    df0 = pd.read_parquet(sample_file)
    all_h = [c for c in df0.columns if c.startswith("h_q")]
    all_h = sorted(all_h, key=parse_q_val)
    if requested:
        missing = [c for c in requested if c not in all_h]
        if missing:
            sys.exit(f"Requested q columns {missing} not found. Available: {all_h}")
        all_h = sorted(requested, key=parse_q_val)
    r_map = {h: h.replace("h_q", "r2_q") if h.replace("h_q", "r2_q") in df0.columns else None for h in all_h}
    return all_h, r_map


def load_participant_multiq(f, channels, h_cols, r_map, min_r2):
    need_cols = ["epoch_idx", "window_start_time_s", "channel", "status"] + h_cols
    if min_r2 > 0:
        need_cols += [r for r in r_map.values() if r is not None]
    df = pd.read_parquet(f, columns=list(dict.fromkeys(need_cols)))
    df = df[df["channel"].isin(channels)].copy()
    if df.empty:
        return None

    base_ok = df["status"] == "ok"
    out_dict = {}
    for hcol in h_cols:
        ok = base_ok.copy()
        rcol = r_map.get(hcol)
        if min_r2 > 0 and rcol is not None:
            ok &= df[rcol] >= min_r2
        vals = df[hcol].where(ok, np.nan)
        tmp = df[["epoch_idx", "channel"]].assign(val=vals)
        wide = tmp.pivot(index="epoch_idx", columns="channel", values="val").reindex(columns=channels).dropna()
        if not wide.empty:
            out_dict[hcol] = wide.mean(axis=1)

    if not out_dict:
        return None
    res = pd.DataFrame(out_dict)
    t = df.drop_duplicates("epoch_idx").set_index("epoch_idx")["window_start_time_s"].reindex(res.index)
    res["t"] = t
    return res


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


def partial_spearman(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(m.sum())
    if n < 10:
        return np.nan, np.nan
    rx, ry, rz = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    X = np.column_stack([np.ones(n), rz])
    rx = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ry = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(rx, ry)[0, 1])
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), n - 3))


def extract_participant_multiq(sid, df, dim_cols, min_epochs):
    st = df["stage"].to_numpy()
    n_tot = len(df)
    m_n3 = st == "N3"
    m_n2 = st == "N2"
    m_rem = st == "REM"
    row = {
        "id": sid,
        "n_epochs": n_tot,
        "n_N3": int(m_n3.sum()),
        "n_N2": int(m_n2.sum()),
        "frac_N3": float(m_n3.sum() / n_tot) if n_tot > 0 else np.nan,
        "frac_N2": float(m_n2.sum() / n_tot) if n_tot > 0 else np.nan,
    }
    for col in dim_cols:
        vals = df[col].to_numpy(float)
        n3 = vals[m_n3 & np.isfinite(vals)]
        n2 = vals[m_n2 & np.isfinite(vals)]
        rem = vals[m_rem & np.isfinite(vals)]

        row[f"{col}__H_N3_med"] = float(np.median(n3)) if len(n3) >= min_epochs else np.nan
        row[f"{col}__H_N2_med"] = float(np.median(n2)) if len(n2) >= min_epochs else np.nan
        if len(n3) >= min_epochs and len(n2) >= min_epochs:
            row[f"{col}__H_N3_minus_N2"] = row[f"{col}__H_N3_med"] - row[f"{col}__H_N2_med"]
            row[f"{col}__auc_N2"] = epoch_auc(n3, n2)
        else:
            row[f"{col}__H_N3_minus_N2"] = np.nan
            row[f"{col}__auc_N2"] = np.nan
        row[f"{col}__auc_REM"] = epoch_auc(n3, rem) if (len(n3) >= min_epochs and len(rem) >= min_epochs) else np.nan
    return row


def bootstrap_auc(y_true, scores, rng, n_boot=2000):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    raw_auc = epoch_auc(pos, neg)
    flip = raw_auc < 0.5
    auc_val = 1.0 - raw_auc if flip else raw_auc
    mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    boots = []
    for _ in range(n_boot):
        bp = pos[rng.integers(0, len(pos), len(pos))]
        bn = neg[rng.integers(0, len(neg), len(neg))]
        b = epoch_auc(bp, bn)
        boots.append(1.0 - b if flip else b)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc_val, float(lo), float(hi), float(mw.pvalue), ("lower" if flip else "higher")


def run_cv_suite(df, feature_sets, seed=0, n_splits=5, n_repeats=20):
    y_multi = df["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
    y_dem = (df["group"] == "Dementia").astype(int).to_numpy()
    y_imp = (df["group"].isin(["MCI", "Dementia"])).astype(int).to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for name, cols in feature_sets.items():
        X = df[cols].to_numpy(float)
        ovr_aucs, ovo_aucs, dem_rest_aucs, dem_norm_aucs, imp_aucs, bals = [], [], [], [], [], []
        for tr, te in rskf.split(X, y_multi):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
            )
            clf.fit(X[tr], y_multi[tr])
            probs = clf.predict_proba(X[te])
            preds = clf.predict(X[te])

            ovr_aucs.append(roc_auc_score(y_multi[te], probs, multi_class="ovr", average="macro"))
            ovo_aucs.append(roc_auc_score(y_multi[te], probs, multi_class="ovo", average="macro"))
            dem_rest_aucs.append(roc_auc_score(y_dem[te], probs[:, 2]))
            imp_aucs.append(roc_auc_score(y_imp[te], probs[:, 1] + probs[:, 2]))
            bals.append(balanced_accuracy_score(y_multi[te], preds))

            # Binary Dementia vs Normal subset within fold
            m_dn = np.isin(y_multi[te], [0, 2])
            if len(np.unique(y_multi[te][m_dn])) == 2:
                y_dn = (y_multi[te][m_dn] == 2).astype(int)
                p_dn = probs[m_dn, 2] / np.clip(probs[m_dn, 0] + probs[m_dn, 2], 1e-12, None)
                dem_norm_aucs.append(roc_auc_score(y_dn, p_dn))

        rows.append({
            "model": name,
            "features": " + ".join(cols),
            "n": len(df),
            "cv_3class_ovr_auc": float(np.mean(ovr_aucs)),
            "cv_3class_ovo_auc": float(np.mean(ovo_aucs)),
            "cv_dem_vs_norm_auc": float(np.mean(dem_norm_aucs)),
            "cv_dem_vs_rest_auc": float(np.mean(dem_rest_aucs)),
            "cv_imp_vs_norm_auc": float(np.mean(imp_aucs)),
            "cv_3class_bal_acc": float(np.mean(bals)),
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
    ap.add_argument("--q-cols", nargs="+", default=None, help="specific h_q* cols (default: all in parquet)")
    ap.add_argument("--min-r2", type=float, default=0.0)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = args.out_dir or (args.h_dir / "classify_dementia_multiq")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(args.h_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"No parquet files found in {args.h_dir}")
    h_cols, r_map = detect_q_columns(files[0], args.q_cols)
    dim_cols = list(h_cols)
    if len(h_cols) >= 2:
        dim_cols.append("delta_H")  # h(q_min) - h(q_max): multifractal spectrum width

    cog = load_cog(args.cog_file, args.cog_col, args.cog_id_col, args.age_col)
    rows = []
    for f in files:
        sid = f.stem.zfill(5)
        d = load_participant_multiq(f, args.channels, h_cols, r_map, args.min_r2)
        if d is None:
            continue
        if len(h_cols) >= 2:
            d["delta_H"] = d[h_cols[0]] - d[h_cols[-1]]
        d = add_stage(d, sid, args.hyp_dir)
        if d is None or d.empty:
            continue
        rows.append(extract_participant_multiq(sid, d, dim_cols, args.min_epochs))

    P = pd.DataFrame(rows).set_index("id").join(cog, how="left")
    P["group"] = np.select(
        [P["teng"] < args.dementia_cut, P["teng"] < args.mci_cut, P["teng"] >= args.mci_cut],
        ["Dementia", "MCI", "Normal"],
        default="",
    )
    P["group"] = P["group"].replace("", np.nan)
    P.reset_index().to_csv(out / "multiq_features_by_participant.csv", index=False)

    # 1. N2 vs N3 Decomposition for q=+2 (or first positive q if h_q+2 absent)
    ref_q = "h_q+2" if "h_q+2" in h_cols else h_cols[-1]
    decomp_feats = [
        f"{ref_q}__H_N3_med",
        f"{ref_q}__H_N2_med",
        f"{ref_q}__H_N3_minus_N2",
        f"{ref_q}__auc_N2",
        "frac_N3",
        "frac_N2",
    ]
    decomp_rows = []
    for feat in decomp_feats:
        sub = P[P["group"].notna() & P[feat].notna()]
        vals_by_g = {g: sub.loc[sub["group"] == g, feat].to_numpy(float) for g in GROUPS}
        kw = stats.kruskal(*[vals_by_g[g] for g in GROUPS])
        dn_sub = sub[sub["group"].isin(["Dementia", "Normal"])]
        auc_dn, lo, hi, p_dn, dir_dn = bootstrap_auc(
            (dn_sub["group"] == "Dementia").astype(int).to_numpy(),
            dn_sub[feat].to_numpy(float),
            rng,
            args.n_boot,
        )
        decomp_rows.append({
            "feature": feat,
            "Normal_med": float(np.median(vals_by_g["Normal"])),
            "MCI_med": float(np.median(vals_by_g["MCI"])),
            "Dementia_med": float(np.median(vals_by_g["Dementia"])),
            "KW_p": float(kw.pvalue),
            "Dem_vs_Norm_AUC": auc_dn,
            "Dem_vs_Norm_p": p_dn,
            "Dem_direction": dir_dn,
        })
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(out / "stage_h_decomposition.csv", index=False)

    # 2. Multi-q Scan across all dimensions
    scan_rows = []
    for col in dim_cols:
        q_num = parse_q_val(col)
        for metric in ["auc_N2", "H_N3_minus_N2", "H_N3_med", "H_N2_med", "auc_REM"]:
            feat = f"{col}__{metric}"
            sub = P[P["group"].notna() & P[feat].notna()]
            if len(sub) < 30:
                continue
            vals_by_g = [sub.loc[sub["group"] == g, feat].to_numpy(float) for g in GROUPS]
            kw = stats.kruskal(*vals_by_g)
            g_ord = sub["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy(float)
            rho_g, p_g = partial_spearman(g_ord, sub[feat].to_numpy(float), sub["age"].to_numpy(float))
            rho_3ms, p_3ms = partial_spearman(sub["teng"].to_numpy(float), sub[feat].to_numpy(float), sub["age"].to_numpy(float))

            dn_sub = sub[sub["group"].isin(["Dementia", "Normal"])]
            auc_dn, lo_dn, hi_dn, p_dn, dir_dn = bootstrap_auc(
                (dn_sub["group"] == "Dementia").astype(int).to_numpy(),
                dn_sub[feat].to_numpy(float),
                rng,
                args.n_boot,
            )
            imp_sub = sub[sub["group"].isin(["MCI", "Dementia", "Normal"])]
            auc_imp, _, _, p_imp, _ = bootstrap_auc(
                imp_sub["group"].isin(["MCI", "Dementia"]).astype(int).to_numpy(),
                imp_sub[feat].to_numpy(float),
                rng,
                args.n_boot,
            )
            scan_rows.append({
                "dimension": col,
                "q": q_num,
                "metric": metric,
                "n": len(sub),
                "Dem_vs_Norm_AUC": auc_dn,
                "Dem_vs_Norm_CI_lo": lo_dn,
                "Dem_vs_Norm_CI_hi": hi_dn,
                "Dem_vs_Norm_p": p_dn,
                "Imp_vs_Norm_AUC": auc_imp,
                "Imp_vs_Norm_p": p_imp,
                "KW_p": float(kw.pvalue),
                "partial_rho_group_age": rho_g,
                "partial_p_group_age": p_g,
                "partial_rho_3MS_age": rho_3ms,
                "partial_p_3MS_age": p_3ms,
                "direction_in_dem": dir_dn,
            })
    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(out / "multiq_classification_scan.csv", index=False)

    # Identify best q dimension for auc_N2
    auc_n2_scan = scan_df[scan_df["metric"] == "auc_N2"].sort_values("Dem_vs_Norm_AUC", ascending=False)
    best_dim = auc_n2_scan.iloc[0]["dimension"] if not auc_n2_scan.empty else ref_q

    # 3. Clean Cross-Validation Models (testing exact user requests)
    core_cols = [
        f"{ref_q}__auc_N2",
        f"{ref_q}__H_N3_minus_N2",
        f"{ref_q}__H_N3_med",
        f"{ref_q}__H_N2_med",
        f"{best_dim}__auc_N2",
        "frac_N3",
        "age",
    ]
    core_cols = list(dict.fromkeys(core_cols))
    complete = P[P["group"].notna() & P[core_cols].notna().all(axis=1)].copy()

    feature_sets = {
        "1_Age_only": ["age"],
        "2_Raw_N3_vs_Raw_N2": [f"{ref_q}__H_N3_med", f"{ref_q}__H_N2_med"],
        "3_auc_N2_only": [f"{ref_q}__auc_N2"],
        "4_auc_N2 + age": [f"{ref_q}__auc_N2", "age"],
        "5_auc_N2 + frac_N3 + age": [f"{ref_q}__auc_N2", "frac_N3", "age"],
        "6_auc_N2 + H_N3_minus_N2 + frac_N3 + age": [f"{ref_q}__auc_N2", f"{ref_q}__H_N3_minus_N2", "frac_N3", "age"],
    }
    if best_dim != ref_q:
        feature_sets[f"7_BestQ({best_dim})_auc_N2 + frac_N3 + age"] = [f"{best_dim}__auc_N2", "frac_N3", "age"]
        feature_sets[f"8_MultiQ({ref_q}+{best_dim}) + frac_N3 + age"] = [
            f"{ref_q}__auc_N2",
            f"{best_dim}__auc_N2",
            "frac_N3",
            "age",
        ]
    cv_df = run_cv_suite(complete, feature_sets, seed=args.seed)
    cv_df.to_csv(out / "cv_clean_models.csv", index=False)

    # Text summary
    lines = [
        f"Detected q columns in parquet: {h_cols}",
        f"Complete-case sample for CV models: {len(complete)} "
        f"(Normal={int((complete['group']=='Normal').sum())}, "
        f"MCI={int((complete['group']=='MCI').sum())}, "
        f"Dementia={int((complete['group']=='Dementia').sum())})",
        "",
        f"=== 1. Raw N2 vs Raw N3 Decomposition ({ref_q}) ===",
        decomp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 2. Clean Cross-Validation Models (Out-of-Fold ROC-AUC) ===",
        cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 3. Multi-q Scan: auc_N2 (Within-Night N3 vs N2 AUC) Across All q Dimensions ===",
        auc_n2_scan[[
            "dimension", "q", "n", "Dem_vs_Norm_AUC", "Dem_vs_Norm_CI_lo", "Dem_vs_Norm_CI_hi",
            "Dem_vs_Norm_p", "Imp_vs_Norm_AUC", "partial_rho_group_age", "partial_p_group_age",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== 4. Top 10 Dimension x Metric Combinations (Ranked by Dementia vs Normal AUC) ===",
        scan_df.sort_values("Dem_vs_Norm_AUC", ascending=False)
        .head(10)[["dimension", "metric", "Dem_vs_Norm_AUC", "Dem_vs_Norm_p", "Imp_vs_Norm_AUC", "partial_p_group_age"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}"),
    ]
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt + "\n")

    # Plot AUC across q values
    if not args.no_plots and len(h_cols) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        q_sub = scan_df[scan_df["q"].notna()].copy()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for metric, label, marker in [
            ("auc_N2", "Within-night N3 vs N2 AUC (auc_N2)", "o"),
            ("H_N3_minus_N2", "Median difference (H_N3 - H_N2)", "s"),
            ("H_N3_med", "Raw N3 median H(q)", "^"),
            ("H_N2_med", "Raw N2 median H(q)", "v"),
        ]:
            m_df = q_sub[q_sub["metric"] == metric].sort_values("q")
            if not m_df.empty:
                ax.plot(m_df["q"], m_df["Dem_vs_Norm_AUC"], marker=marker, lw=2, label=label)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xlabel("Multifractal Moment Order (q)")
        ax.set_ylabel("Dementia vs Normal ROC-AUC")
        ax.set_title("Dementia Classification Performance Across MFDFA q Orders")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "multiq_auc_profile.png", dpi=150)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
