#!/usr/bin/env python3
"""
Compare MFDFA variants stored as parquet, one folder per variant.

Layout:  <root>/<variant>/<5-digit id>.parquet

Two MFDFA formats are normalised to the same canonical form
(index id + window_start_time_s, columns '<DERIV>_q<+n>' and 'MEAN_q<+n>'):
  wide  subject_id, window_start_time_s, sleep_stage, <DERIV>_h0..h9      (original*, noenvelope*)
        -> h index k is mapped to q with --wide-q (ASSUMED, check against sof_mfdfa.py)
  long  one row per (epoch, channel): h_q-4 ... h_q+4, status, ...          (order2_smin0.125, ...)
Folders with no h columns (e.g. 'spectral': rel_* band powers, slopes) are treated as
feature tables: they are related to h(q) per epoch, not compared as MFDFA.
Empty folders are skipped.

Stage always comes from <hyp-dir>/hypnogram_<id>.csv (start_sec, stage_code), because
sleep_stage in the wide files looked like a copy of window_start_time_s.

Outputs in --out (default <root>/_comparison):
  coverage.csv, rows_per_participant.csv, schema.csv
  summary_median_by_stage.csv       median over participants of participant medians
  agreement_epoch_vs_ref.csv        epoch-level agreement of each MFDFA variant vs --reference
  agreement_participant_pairs.csv   participant-level agreement, all MFDFA variant pairs
  matrix_*_by_stage.csv, heatmap_*.png, scatter_*.png
  spectral_vs_h.csv                 within-participant Spearman: spectral features vs h(focus q)
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ["original", "original_lfnone", "noenvelope", "noenvelope_lfnone",
            "order2_smin0.125", "through_order2_smin0.125", "spectral"]
STAGES = {0: "wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
TIME, STAGE = "window_start_time_s", "sleep_stage"
WIDE_RE = re.compile(r"^(?P<d>.+)_h(?P<k>\d+)$")
LONG_RE = re.compile(r"^h_q(?P<q>[+-]\d+)$")
PROJECT = Path("~/pl1358-01/project_data").expanduser()


def qname(q: int) -> str:
    return f"q{int(q):+d}"


# ----------------------------------------------------------------------------- loading
def load_variant(vdir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(vdir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df = df.drop(columns=[c for c in ("subject_id",) if c in df.columns])
        df.insert(0, "id", f.stem)
        if TIME in df:
            df[TIME] = df[TIME].round().astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def canonicalise(v, df, qmap):
    """Return (kind, frame). kind in {'mfdfa', 'features'}; frame has columns id, TIME, ..."""
    lcols = [c for c in df.columns if LONG_RE.match(c)]
    wcols = [c for c in df.columns if WIDE_RE.match(c)]
    if lcols and "channel" in df:  # long format
        d = df
        if "status" in d:
            bad = (d["status"] != "ok").mean()
            if bad:
                print(f"[note] {v}: {bad:.1%} rows have status != ok -> set to NaN")
            d = d.copy()
            d.loc[d["status"] != "ok", lcols] = np.nan
        d = d.drop_duplicates(["id", TIME, "channel"])
        w = d.pivot(index=["id", TIME], columns="channel", values=lcols)
        w.columns = [f"{ch}_{qname(int(LONG_RE.match(h)['q']))}" for h, ch in w.columns]
        return "mfdfa", w.reset_index()
    if wcols:  # wide format
        ks = sorted({int(WIDE_RE.match(c)["k"]) for c in wcols})
        if len(ks) != len(qmap):
            sys.exit(f"{v}: {len(ks)} h indices but --wide-q has {len(qmap)} values")
        ren = {c: f"{WIDE_RE.match(c)['d']}_{qname(qmap[int(WIDE_RE.match(c)['k'])])}" for c in wcols}
        if STAGE in df:
            print(f"[note] {v}: {STAGE} == {TIME} in {(df[STAGE] == df[TIME]).mean():.0%} of rows")
        keep = ["id", TIME] + wcols
        return "mfdfa", df[keep].drop_duplicates(["id", TIME]).rename(columns=ren)
    num = [c for c in df.columns if c not in ("id", TIME, "epoch_idx") and pd.api.types.is_numeric_dtype(df[c])]
    return "features", df[["id", TIME] + num].drop_duplicates(["id", TIME])


def hyp_stage_labels(df, hyp_dir):
    out = np.full(len(df), "unscored", dtype=object)
    t_all = df[TIME].to_numpy()
    for sid, idx in df.groupby("id").indices.items():
        f = hyp_dir / f"hypnogram_{sid}.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        m = dict(zip(h["start_sec"].round().astype(int), h["stage_code"].astype(int)))
        out[idx] = [STAGES.get(m.get(int(t), -1), "other" if m.get(int(t), -1) >= 0 else "unscored")
                    for t in t_all[idx]]
    return out


# ----------------------------------------------------------------------------- stats
def icc_2_1(x, y) -> float:
    d = np.column_stack([x, y])
    n, k = d.shape
    if n < 3:
        return np.nan
    gm = d.mean()
    msr = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def pair_stats(a, b) -> dict:
    """a = reference, b = variant (diffs are b - a)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    out = dict(n=n, pearson=np.nan, spearman=np.nan, icc21=np.nan, mean_diff=np.nan,
               sd_diff=np.nan, median_abs_diff=np.nan, loa_low=np.nan, loa_high=np.nan)
    if n < 3:
        return out
    d = b - a
    sd = d.std(ddof=1)
    out.update(mean_diff=d.mean(), sd_diff=sd, median_abs_diff=np.median(np.abs(d)),
               loa_low=d.mean() - 1.96 * sd, loa_high=d.mean() + 1.96 * sd)
    if a.std() > 0 and b.std() > 0:
        out["pearson"] = np.corrcoef(a, b)[0, 1]
        out["spearman"] = np.corrcoef(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())[0, 1]
        out["icc21"] = icc_2_1(a, b)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PROJECT / "worksof/criticality"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--reference", default="original")
    ap.add_argument("--hyp-dir", default=str(PROJECT / "worksof/hypnograms/expert"))
    ap.add_argument("--wide-q", default="-5,-4,-3,-2,-1,1,2,3,4,5",
                    help="q value of wide columns h0..h9, comma separated. ASSUMED default; verify.")
    ap.add_argument("--focus-q", type=int, default=2, help="q shown in per-stage tables / plots")
    ap.add_argument("--min-epochs", type=int, default=20, help="min epochs for within-participant correlations")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    qmap = [int(x) for x in args.wide_q.split(",")]
    if args.wide_q == "-5,-4,-3,-2,-1,1,2,3,4,5":
        print("[WARN] wide h0..h9 -> q mapping is ASSUMED to be -5..-1,+1..+5 (set --wide-q to override)")

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else root / "_comparison"
    out.mkdir(parents=True, exist_ok=True)
    hyp_dir = Path(args.hyp_dir).expanduser()
    if not hyp_dir.is_dir():
        sys.exit(f"Hypnogram dir not found: {hyp_dir}")

    # ---- load + canonicalise
    raw_cols, mf, feat, ids_by_var, rows_by_var = {}, {}, {}, {}, {}
    for v in args.variants:
        vdir = root / v
        if not vdir.is_dir():
            print(f"[warn] missing folder: {vdir}", file=sys.stderr)
            continue
        df = load_variant(vdir)
        n_id = df["id"].nunique() if len(df) else 0
        print(f"[load] {v:28s} participants={n_id:5d} rows={len(df)}")
        if not len(df):
            print(f"[skip] {v}: no data")
            continue
        raw_cols[v] = df.dtypes.astype(str)
        ids_by_var[v] = set(df["id"])
        rows_by_var[v] = df.groupby("id").size()
        kind, frame = canonicalise(v, df, qmap)
        (mf if kind == "mfdfa" else feat)[v] = frame
        print(f"       -> {kind}, {frame.shape[1] - 2} columns")
    if args.reference not in mf:
        sys.exit(f"Reference '{args.reference}' not available as an MFDFA variant.")
    variants = [args.reference] + [v for v in mf if v != args.reference]
    if len(variants) < 2:
        sys.exit("Need at least two MFDFA variants.")

    # ---- coverage
    ids = sorted(set().union(*ids_by_var.values()))
    cov = pd.DataFrame({v: [i in ids_by_var[v] for i in ids] for v in ids_by_var}, index=ids)
    cov.index.name = "id"
    cov.to_csv(out / "coverage.csv")
    print(f"\n[coverage] {len(ids)} ids; in every loaded variant: {int(cov.all(axis=1).sum())}")
    pd.DataFrame(rows_by_var).to_csv(out / "rows_per_participant.csv")
    pd.DataFrame(raw_cols).to_csv(out / "schema.csv")

    # ---- metrics, MEAN derivation, stage
    derivs = sorted({c.rsplit("_q", 1)[0] for v in variants for c in mf[v].columns if "_q" in c})
    qs = sorted({int(c.rsplit("_q", 1)[1]) for v in variants for c in mf[v].columns if "_q" in c})
    print(f"[detect] derivations={derivs}  q values={qs}")
    metrics = [f"{d}_{qname(q)}" for d in derivs for q in qs] + [f"MEAN_{qname(q)}" for q in qs]
    mean_cols = [f"MEAN_{qname(q)}" for q in qs]
    fcol = f"MEAN_{qname(args.focus_q)}"

    frames = {}
    for v in variants:
        df = mf[v].copy()
        newcols = {}
        for q in qs:
            cols = [f"{d}_{qname(q)}" for d in derivs if f"{d}_{qname(q)}" in df.columns]
            newcols[f"MEAN_{qname(q)}"] = df[cols].mean(axis=1) if cols else np.nan
        df = pd.concat([df, pd.DataFrame(newcols)], axis=1)
        df["stage"] = hyp_stage_labels(df, hyp_dir)
        frames[v] = df[["id", TIME, "stage"] + [m for m in metrics if m in df.columns]]
    ref_stage = frames[args.reference]["stage"]
    print("[stage] epochs per stage (reference):")
    print(ref_stage.value_counts().to_string())
    stage_order = ["all"] + [s for s in STAGES.values() if s in set(ref_stage)]

    # ---- participant medians
    pm = {}
    for v, df in frames.items():
        mm = [m for m in metrics if m in df.columns]
        g = df.groupby(["id", "stage"])[mm].median()
        a = df.groupby("id")[mm].median()
        a["stage"] = "all"
        pm[v] = pd.concat([g, a.reset_index().set_index(["id", "stage"])])

    summ = pd.concat({v: p.groupby(level="stage").median() for v, p in pm.items()}, names=["variant", "stage"])
    summ.to_csv(out / "summary_median_by_stage.csv")
    pd.set_option("display.width", 220)
    print("\n[summary] MEAN derivation, all epochs: median over participants of participant medians")
    print(summ.xs("all", level="stage")[[c for c in mean_cols if c in summ]].round(3).to_string())
    if fcol in summ:
        print(f"\n[summary] {fcol} per stage")
        print(summ[fcol].unstack("stage").reindex(columns=stage_order).round(3).to_string())

    # ---- epoch-level agreement vs reference
    ref_ix = frames[args.reference].set_index(["id", TIME])
    rows = []
    for v in variants[1:]:
        vix = frames[v].set_index(["id", TIME])
        idx = ref_ix.index.intersection(vix.index)
        A, B = ref_ix.reindex(idx), vix.reindex(idx)
        st = A["stage"].to_numpy()
        print(f"[epoch] {v}: {len(idx)} matched epochs (reference has {len(ref_ix)}, variant {len(vix)})")
        for stage in stage_order:
            msk = np.ones(len(idx), bool) if stage == "all" else st == stage
            for m in metrics:
                if m in A and m in B:
                    rows.append(dict(variant=v, stage=stage, metric=m,
                                     **pair_stats(A[m].to_numpy(float)[msk], B[m].to_numpy(float)[msk])))
    pd.DataFrame(rows).to_csv(out / "agreement_epoch_vs_ref.csv", index=False)

    # ---- participant-level agreement, all pairs (reference first -> diffs are variant - reference)
    prow = []
    for a, b in itertools.combinations(variants, 2):
        for stage in stage_order:
            try:
                X, Y = pm[a].xs(stage, level="stage"), pm[b].xs(stage, level="stage")
            except KeyError:
                continue
            idx = X.index.intersection(Y.index)
            for m in metrics:
                if m in X and m in Y:
                    prow.append(dict(a=a, b=b, stage=stage, metric=m,
                                     **pair_stats(X.loc[idx, m].to_numpy(float), Y.loc[idx, m].to_numpy(float))))
    pairs = pd.DataFrame(prow)
    pairs.to_csv(out / "agreement_participant_pairs.csv", index=False)

    vs = pairs[(pairs.a == args.reference) & (pairs.stage == "all") & pairs.metric.isin(mean_cols)]
    print(f"\n[participant-level vs {args.reference}] MEAN derivation, all epochs, median over shared q")
    print(vs.groupby("b")[["spearman", "icc21", "mean_diff", "median_abs_diff"]].median().round(3).to_string())
    f = pairs[(pairs.a == args.reference) & (pairs.metric == fcol)]
    if len(f):
        print(f"\n[participant-level vs {args.reference}] {fcol} per stage: Spearman / mean diff (variant - ref)")
        t = f.pivot(index="b", columns="stage", values="spearman").reindex(columns=stage_order)
        d = f.pivot(index="b", columns="stage", values="mean_diff").reindex(columns=stage_order)
        print(pd.concat({"spearman": t, "mean_diff": d}, axis=1).round(3).to_string())

    # ---- matrices
    mats = {}
    for stat in ("spearman", "icc21"):
        for stage in stage_order:
            g = pairs[(pairs.stage == stage) & pairs.metric.isin(mean_cols)]
            mat = pd.DataFrame(np.nan, index=variants, columns=variants)
            for (a, b), gg in g.groupby(["a", "b"]):
                mat.loc[a, b] = mat.loc[b, a] = gg[stat].mean()
            for v_ in variants:
                mat.loc[v_, v_] = 1.0
            mats[(stat, stage)] = mat
        pd.concat({s: mats[(stat, s)] for s in stage_order}, names=["stage", "variant"]).to_csv(
            out / f"matrix_{stat}_by_stage.csv")
    print("\n[spearman matrix] participant medians, MEAN derivation, stage=all, mean over shared q")
    print(mats[("spearman", "all")].round(3).to_string())

    # ---- feature tables (e.g. spectral) vs h(focus q), within participant
    srows = []
    for fv, fdf in feat.items():
        fdf = fdf.copy()
        fdf["stage"] = hyp_stage_labels(fdf, hyp_dir)
        fdf = fdf.set_index(["id", TIME])
        fcols = [c for c in fdf.columns if c != "stage"]
        for v in variants:
            if fcol not in frames[v]:
                continue
            j = fdf.join(frames[v].set_index(["id", TIME])[[fcol]], how="inner")
            for stage in stage_order:
                sub = j if stage == "all" else j[j["stage"] == stage]
                per = []
                for sid, g in sub.groupby(level="id"):
                    if len(g) >= args.min_epochs:
                        per.append(g[fcols + [fcol]].corr(method="spearman")[fcol][fcols])
                if per:
                    med = pd.concat(per, axis=1).T.median()
                    for c in fcols:
                        srows.append(dict(feature_table=fv, feature=c, variant=v, stage=stage,
                                          n_participants=len(per), median_spearman=med[c]))
    if srows:
        sp = pd.DataFrame(srows)
        sp.to_csv(out / "spectral_vs_h.csv", index=False)
        print(f"\n[features vs {fcol}] median within-participant Spearman, stage=all")
        print(sp[sp.stage == "all"].pivot(index="feature", columns="variant", values="median_spearman")
              .round(2).to_string())

    # ---- plots
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib missing; skipping plots", file=sys.stderr)
            plt = None
        if plt:
            for stat in ("spearman", "icc21"):
                mat = mats[(stat, "all")]
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(mat.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
                ax.set_xticks(range(len(variants)), variants, rotation=45, ha="right")
                ax.set_yticks(range(len(variants)), variants)
                for i in range(len(variants)):
                    for j in range(len(variants)):
                        ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center",
                                color="w" if mat.iloc[i, j] < .6 else "k", fontsize=8)
                ax.set_title(f"{stat}, participant medians, MEAN derivation")
                fig.colorbar(im)
                fig.tight_layout()
                fig.savefig(out / f"heatmap_{stat}.png", dpi=150)
                plt.close(fig)
            if fcol in pm[args.reference]:
                others = variants[1:]
                fig, axes = plt.subplots(1, len(others), figsize=(3.2 * len(others), 3.4), squeeze=False)
                x = pm[args.reference].xs("all", level="stage")[fcol]
                for ax, v in zip(axes[0], others):
                    y = pm[v].xs("all", level="stage")[fcol]
                    j = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
                    ax.scatter(j.x, j.y, s=6, alpha=.5)
                    if len(j):
                        lo, hi = j.to_numpy().min(), j.to_numpy().max()
                        ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
                    ax.set_title(v, fontsize=8)
                    ax.set_xlabel(args.reference, fontsize=8)
                axes[0][0].set_ylabel("variant", fontsize=8)
                fig.suptitle(f"{fcol}, participant medians (all epochs)")
                fig.tight_layout()
                fig.savefig(out / f"scatter_{fcol}.png", dpi=130)
                plt.close(fig)

    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
