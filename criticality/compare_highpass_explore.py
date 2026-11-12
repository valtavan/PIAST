#!/usr/bin/env python
"""
compare_highpass_explore.py - which high-pass cutoff separates the sleep stages best?

Reads <explore>/high<cutoff>/<id>.parquet (written by explore_highpass.py), uses H(q=+2) averaged over the
four derivations, joins the expert hypnograms (wake, N1, N2, N3, REM) and scores every cutoff.

Epochs: only epochs that are finite in all four derivations for EVERY cutoff are used, so all cutoffs are
scored on exactly the same epochs and participants.

Criteria (all on participant-level medians unless stated; --min-epochs per participant and stage)
  1. Gates (must pass to be ranked)
       ok_frac >= 0.95 (epochs with a valid fit), median R^2 of the q=+2 fit >= 0.8,
       N3 has the highest group-median H and N3 > N2 in >= 80 % of participants,
       median within-participant Spearman(H, relative delta power) > 0 (if the spectral folder exists).
  2. PRIMARY: stage separation = mean |d_z| over all stage pairs (paired across participants, 10 pairs,
       pairs with fewer than --min-pair-n participants are left out for every cutoff).
     Secondary: AUC of H for N3 vs the other stages, pooled over epochs (fixed threshold) and averaged
       within participant (removes between-person offsets).
  3. Label-free quality: split-half reliability of the participant medians (odd vs even epochs of a stage,
       Spearman-Brown corrected), averaged over stages. Interleaved halves share slow state, so the
       absolute value is optimistic; use it to compare cutoffs.
  4. Cognitive check (not used for the choice): Spearman between participant median H (per stage) and the
       continuous follow-up 3MS (V9TENG), plain, partial for age (V8AGE) and partial for age plus the relative
       delta power of that stage. With 50 participants the intervals are wide.
Participants are bootstrapped (same resamples for all cutoffs, so differences are paired).
Choice: best primary score among the cutoffs that pass the gates; every cutoff whose paired difference to
the best has a 95 % interval including 0 is "equivalent"; among those the one with the highest split-half
reliability is recommended.

Outputs in <explore>/_comparison: summary_by_cutoff.csv, stage_medians.csv, stage_pair_dz.csv,
reliability_by_stage.csv, cognition_by_stage.csv, bootstrap_vs_best.csv, recommendation.txt,
metrics_vs_cutoff.png

Usage:
    uv run --frozen python compare_highpass_explore.py [--explore-dir DIR] [--cog-file FILE]
Defaults: --cog-file ~/pl1358-01/project_data/worksof/cognitive.csv (columns ID, V9TENG, V8AGE, ...).
"""
import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT = Path("~/pl1358-01/project_data").expanduser()
SOF_DIR = PROJECT / "SOFdataset"
CRIT_DIR = PROJECT / "worksof/criticality"
COG_FILE = PROJECT / "worksof/cognitive.csv"
HYP_DIR = PROJECT / "worksof/hypnograms/expert"

STAGES = ["wake", "N1", "N2", "N3", "REM"]
CODE = {0: "wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CHANNELS = ["C3A1", "C3A2", "C4A1", "C4A2"]
HCOL, RCOL = "h_q+2", "r2_q+2"
ID_NAMES = ("id", "subject_id", "sof_id", "subject", "sid", "idno", "newid")


# ----------------------------------------------------------------------------- loading
def cutoff_dirs(explore):
    out = []
    for d in explore.iterdir():
        m = re.fullmatch(r"high(\d+(?:\.\d+)?)", d.name)
        if d.is_dir() and m:
            out.append((float(m.group(1)), d))
    return sorted(out)


def load_cutoff(d):
    frames = []
    for f in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["epoch_idx", "window_start_time_s", "channel", "status", HCOL, RCOL])
        df["id"] = f.stem
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    ok = df["status"] == "ok"
    df.loc[~ok, [HCOL, RCOL]] = np.nan
    n_ok = float(ok.mean())
    wide = df.pivot(index=["id", "epoch_idx"], columns="channel", values=HCOL)
    r2 = df.pivot(index=["id", "epoch_idx"], columns="channel", values=RCOL)
    tmap = df.drop_duplicates(["id", "epoch_idx"]).set_index(["id", "epoch_idx"])["window_start_time_s"]
    return dict(h=wide, r2=r2, t=tmap, ok_frac=n_ok)


def hypnogram_stage(ids_times, hyp_dir):
    """Series of stage names aligned to a (id, epoch_idx)-indexed Series of window start times."""
    out = pd.Series(np.nan, index=ids_times.index, dtype=object)
    for sid, g in ids_times.groupby(level="id"):
        f = hyp_dir / f"hypnogram_{sid}.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        m = dict(zip(h["start_sec"].round().astype(int), h["stage_code"].astype(int)))
        out.loc[g.index] = [CODE.get(m.get(int(round(t)), -1)) for t in g.to_numpy()]
    return out


def load_cog(path, col, id_col, age_col):
    """DataFrame indexed by 5-digit id with the 3MS score (and the age covariate, if given)."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    for need in (col, id_col) + ((age_col,) if age_col else ()):
        if need.lower() not in cols:
            sys.exit(f"{path}: no column {need}; columns are {list(df.columns)}")
    ids = df[cols[id_col.lower()]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    y = pd.to_numeric(df[cols[col.lower()]], errors="coerce")
    out = pd.DataFrame({"teng": y.where((y >= 0) & (y <= 100)).to_numpy()}, index=ids)
    if age_col:
        out["age"] = pd.to_numeric(df[cols[age_col.lower()]], errors="coerce").to_numpy()
    out = out[~out.index.duplicated()]
    return out


# ----------------------------------------------------------------------------- statistics
def auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def dz(a, b):
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return np.nan
    return float(d.mean() / d.std(ddof=1))


def primary_score(P, idx, pairs):
    X = P[idx]
    vals = [abs(v) for v in (dz(X[:, i], X[:, j]) for i, j in pairs) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def split_half(A, B, idx, min_n=15):
    """Spearman-Brown corrected split-half reliability for each stage column."""
    out = []
    for s in range(A.shape[1]):
        a, b = A[idx, s], B[idx, s]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_n or a[m].std() == 0 or b[m].std() == 0:
            out.append(np.nan)
            continue
        r = np.corrcoef(a[m], b[m])[0, 1]
        out.append(2 * r / (1 + r) if r > -1 else np.nan)
    return np.array(out)


def spearman_partial(x, y, Z=None):
    """Spearman correlation of x and y and, if Z (n x k) is given, the rank-partial correlation given Z."""
    m = np.isfinite(x) & np.isfinite(y)
    if Z is not None:
        m &= np.isfinite(Z).all(axis=1)
    if m.sum() < 8:
        return np.nan, np.nan
    rx, ry = rankdata(x[m]), rankdata(y[m])
    plain = np.nan if rx.std() == 0 or ry.std() == 0 else float(np.corrcoef(rx, ry)[0, 1])
    if Z is None:
        return plain, np.nan
    rz = np.column_stack([rankdata(Z[m][:, k]) for k in range(Z.shape[1])])
    if (rz.std(axis=0) == 0).any():
        return plain, np.nan
    X = np.column_stack([np.ones(m.sum()), rz])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    partial = np.nan if ex.std() == 0 or ey.std() == 0 else float(np.corrcoef(ex, ey)[0, 1])
    return plain, partial


def ci(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return (np.nan, np.nan) if len(arr) < 20 else tuple(np.percentile(arr, [2.5, 97.5]))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explore-dir", type=lambda s: Path(s).expanduser(), default=CRIT_DIR / "explore")
    ap.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=HYP_DIR)
    ap.add_argument("--spectral-dir", type=lambda s: Path(s).expanduser(), default=CRIT_DIR / "spectral")
    ap.add_argument("--cog-file", type=lambda s: Path(s).expanduser(), default=COG_FILE)
    ap.add_argument("--cog-col", default="V9TENG")
    ap.add_argument("--cog-id-col", default="ID")
    ap.add_argument("--age-col", default="V8AGE", help="covariate for the partial correlations ('' = none)")
    ap.add_argument("--min-epochs", type=int, default=10, help="min epochs per participant and stage")
    ap.add_argument("--min-pair-n", type=int, default=15, help="min participants for a stage pair / reliability")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = args.explore_dir / "_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load all cutoffs
    raw = {}
    for fc, d in cutoff_dirs(args.explore_dir):
        r = load_cutoff(d)
        if r is None:
            print(f"[skip] {d.name}: no parquet files")
            continue
        raw[fc] = r
        print(f"[load] high{fc:g}: {r['h'].index.get_level_values('id').nunique()} participants, "
              f"{len(r['h'])} epochs, ok fraction {r['ok_frac']:.3f}")
    if len(raw) < 2:
        sys.exit("need at least two cutoffs with data")
    cutoffs = sorted(raw)
    sl = sorted(args.explore_dir.glob("subjects_n*_seed*.txt"))
    if sl:
        expected = set(pd.read_csv(sl[0], header=None, dtype=str).iloc[:, 0].str.zfill(5))
        for fc in cutoffs:
            have = set(raw[fc]["h"].index.get_level_values("id"))
            if have != expected:
                print(f"[warn] high{fc:g}: {len(have)} of {len(expected)} expected participants "
                      f"(missing {len(expected - have)}), runs unfinished?")

    # ---- common epochs: finite in all four derivations for every cutoff
    common = None
    for fc in cutoffs:
        h = raw[fc]["h"].reindex(columns=CHANNELS).dropna()
        common = h.index if common is None else common.intersection(h.index)
    print(f"[common] {len(common)} epochs, {common.get_level_values('id').nunique()} participants "
          f"valid in all four derivations at all cutoffs")
    tmap = raw[cutoffs[0]]["t"].reindex(common)
    stage = hypnogram_stage(tmap, args.hyp_dir)
    E = pd.DataFrame({"stage": stage, "t": tmap}).dropna(subset=["stage"])
    print("[stage] epochs per stage:", E["stage"].value_counts().reindex(STAGES).fillna(0).astype(int).to_dict())
    idx = E.index
    for fc in cutoffs:
        E[f"H{fc:g}"] = raw[fc]["h"].reindex(idx)[CHANNELS].mean(axis=1)
    E = E.reset_index()
    E["pos"] = E.groupby(["id", "stage"]).cumcount()
    E["half"] = E["pos"] % 2

    pids = sorted(E["id"].unique())
    n_p = len(pids)
    counts = E.groupby(["id", "stage"]).size().unstack().reindex(index=pids, columns=STAGES).fillna(0)
    enough = counts >= args.min_epochs
    print("[participants per stage with >= %d epochs]" % args.min_epochs, enough.sum().to_dict())

    # ---- spectral and cognition inputs
    spec = None
    if args.spectral_dir.is_dir():
        frames = []
        for pid in pids:
            f = args.spectral_dir / f"{pid}.parquet"
            if f.exists():
                s = pd.read_parquet(f, columns=["window_start_time_s", "rel_delta", "slope_1_40"])
                s["id"] = pid
                frames.append(s)
        if frames:
            spec = pd.concat(frames, ignore_index=True)
            spec["tkey"] = spec["window_start_time_s"].round().astype(int)
            E["tkey"] = E["t"].round().astype(int)
            E = E.merge(spec[["id", "tkey", "rel_delta", "slope_1_40"]], on=["id", "tkey"], how="left")
    if spec is None:
        print("[spectral] folder missing or empty: spectral gate and partial correlation skipped")

    teng, age = None, None
    if args.cog_file.exists():
        cog = load_cog(args.cog_file, args.cog_col, args.cog_id_col, args.age_col).reindex(pids)
        teng = cog["teng"]
        age = cog["age"].to_numpy(float) if "age" in cog else None
        print(f"[cognition] {args.cog_col} from {args.cog_file}: {int(teng.notna().sum())} of {n_p} participants"
              + (f", {args.age_col} for {int(np.isfinite(age).sum())}" if age is not None else ""))
        if teng.notna().sum() < 10:
            print("[cognition] fewer than 10 participants with a score: skipped")
            teng = None
    else:
        print(f"[cognition] {args.cog_file} not found; pass --cog-file. Cognitive check skipped")

    # ---- per-cutoff participant-level tables
    P, HA, HB, AUCW, AUCP, medians_long, spec_rho, extra = {}, {}, {}, {}, {}, [], {}, {}
    delta_med = None
    if "rel_delta" in E:
        delta_med = E.groupby(["id", "stage"])["rel_delta"].median().unstack().reindex(index=pids, columns=STAGES)
        delta_med = delta_med.where(enough)
    for fc in cutoffs:
        col = f"H{fc:g}"
        med = E.groupby(["id", "stage"])[col].median().unstack().reindex(index=pids, columns=STAGES).where(enough)
        P[fc] = med.to_numpy(float)
        half = E.groupby(["id", "stage", "half"])[col].median().unstack("half").reindex(columns=[0, 1])
        ha = half[0].unstack().reindex(index=pids, columns=STAGES).where(enough & (counts >= 2 * (args.min_epochs // 2)))
        hb = half[1].unstack().reindex(index=pids, columns=STAGES).where(enough & (counts >= 2 * (args.min_epochs // 2)))
        HA[fc], HB[fc] = ha.to_numpy(float), hb.to_numpy(float)
        n3 = E["stage"] == "N3"
        AUCP[fc] = auc(E.loc[n3, col].to_numpy(), E.loc[~n3, col].to_numpy())
        aw = []
        for pid, g in E.groupby("id"):
            pos, neg = g.loc[g["stage"] == "N3", col].to_numpy(), g.loc[g["stage"] != "N3", col].to_numpy()
            aw.append(auc(pos, neg) if len(pos) >= args.min_epochs and len(neg) >= args.min_epochs else np.nan)
        AUCW[fc] = pd.Series(aw, index=sorted(E["id"].unique())).reindex(pids).to_numpy(float)
        for s in STAGES:
            medians_long.append(dict(cutoff=fc, stage=s, median_H=float(np.nanmedian(med[s])),
                                     n_participants=int(med[s].notna().sum())))
        if spec is not None:
            r_d = E.groupby("id").apply(lambda g: g[[col, "rel_delta"]].corr(method="spearman").iloc[0, 1],
                                        include_groups=False)
            r_s = E.groupby("id").apply(lambda g: g[[col, "slope_1_40"]].corr(method="spearman").iloc[0, 1],
                                        include_groups=False)
            spec_rho[fc] = (float(r_d.median()), float(r_s.median()))
        # gate inputs
        m32 = np.isfinite(P[fc][:, 3]) & np.isfinite(P[fc][:, 2])
        fr = float(np.mean(P[fc][m32, 3] > P[fc][m32, 2])) if m32.any() else np.nan
        gmed = {s: float(np.nanmedian(P[fc][:, i])) for i, s in enumerate(STAGES)}
        r2 = raw[fc]["r2"].reindex(idx)[CHANNELS].to_numpy(float)
        extra[fc] = dict(ok_frac=raw[fc]["ok_frac"], median_r2=float(np.nanmedian(r2)),
                         n3_highest=bool(max(gmed, key=gmed.get) == "N3"), frac_n3_gt_n2=fr)

    # ---- stage pairs available in the full sample (identical for all cutoffs)
    any_c = cutoffs[0]
    pairs = []
    for (i, a), (j, b) in combinations(list(enumerate(STAGES)), 2):
        if int((np.isfinite(P[any_c][:, i]) & np.isfinite(P[any_c][:, j])).sum()) >= args.min_pair_n:
            pairs.append((i, j))
    print(f"[pairs] {len(pairs)} of 10 stage pairs have >= {args.min_pair_n} participants: "
          + ", ".join(f"{STAGES[i]}-{STAGES[j]}" for i, j in pairs))
    if not pairs:
        sys.exit("no stage pair with enough participants; lower --min-pair-n or --min-epochs")

    full = np.arange(n_p)
    boots = rng.integers(0, n_p, size=(args.n_boot, n_p))

    # ---- tables
    dz_rows, rel_rows = [], []
    for fc in cutoffs:
        for i, j in pairs:
            dz_rows.append(dict(cutoff=fc, pair=f"{STAGES[i]}-{STAGES[j]}", dz=dz(P[fc][:, i], P[fc][:, j])))
        rel = split_half(HA[fc], HB[fc], full, args.min_pair_n)
        for s, v in zip(STAGES, rel):
            rel_rows.append(dict(cutoff=fc, stage=s, reliability=v))
    pd.DataFrame(dz_rows).to_csv(out / "stage_pair_dz.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "reliability_by_stage.csv", index=False)
    pd.DataFrame(medians_long).to_csv(out / "stage_medians.csv", index=False)

    # ---- bootstrap
    prim_b, rel_b, aucw_b = {}, {}, {}
    for fc in cutoffs:
        prim_b[fc] = np.array([primary_score(P[fc], b, pairs) for b in boots])
        rel_b[fc] = np.array([np.nanmean(split_half(HA[fc], HB[fc], b, args.min_pair_n)) for b in boots])
        aucw_b[fc] = np.array([np.nanmean(AUCW[fc][b]) for b in boots])

    summ = []
    for fc in cutoffs:
        ps = primary_score(P[fc], full, pairs)
        rel = float(np.nanmean(split_half(HA[fc], HB[fc], full, args.min_pair_n)))
        aw = float(np.nanmean(AUCW[fc]))
        row = dict(cutoff=fc, n_participants=n_p, n_epochs=len(E), **extra[fc],
                   primary_mean_abs_dz=ps, primary_lo=ci(prim_b[fc])[0], primary_hi=ci(prim_b[fc])[1],
                   auc_n3_pooled=AUCP[fc], auc_n3_within=aw, auc_within_lo=ci(aucw_b[fc])[0],
                   auc_within_hi=ci(aucw_b[fc])[1], split_half_reliability=rel, rel_lo=ci(rel_b[fc])[0],
                   rel_hi=ci(rel_b[fc])[1])
        if fc in spec_rho:
            row["spearman_H_rel_delta"], row["spearman_H_slope_1_40"] = spec_rho[fc]
        summ.append(row)
    S = pd.DataFrame(summ).set_index("cutoff")

    gates = (S["ok_frac"] >= 0.95) & (S["median_r2"] >= 0.8) & S["n3_highest"] & (S["frac_n3_gt_n2"] >= 0.8)
    if "spearman_H_rel_delta" in S:
        gates &= S["spearman_H_rel_delta"] > 0
    S["passes_gates"] = gates

    # ---- recommendation
    lines = []
    cand = S[S["passes_gates"]]
    if cand.empty:
        lines.append("No cutoff passes all gates; see summary_by_cutoff.csv (gate columns).")
        best, equivalent, rec = None, [], None
        vs_best = pd.DataFrame()
    else:
        best = float(cand["primary_mean_abs_dz"].idxmax())
        rows = []
        equivalent = []
        for fc in cand.index:
            d = prim_b[best] - prim_b[fc]
            lo, hi = ci(d)
            rows.append(dict(cutoff=fc, primary=S.loc[fc, "primary_mean_abs_dz"],
                             diff_to_best=S.loc[best, "primary_mean_abs_dz"] - S.loc[fc, "primary_mean_abs_dz"],
                             diff_lo=lo, diff_hi=hi))
            if fc == best or not (lo > 0):
                equivalent.append(float(fc))
        vs_best = pd.DataFrame(rows)
        rec = max(equivalent, key=lambda c: (np.nan_to_num(S.loc[c, "split_half_reliability"], nan=-1)))
        lines += [f"Cutoffs passing the gates: {[float(c) for c in cand.index]}",
                  f"Best primary score (mean |d_z| over stage pairs): {best:g} Hz "
                  f"({S.loc[best, 'primary_mean_abs_dz']:.3f}, 95% CI {S.loc[best, 'primary_lo']:.3f}-{S.loc[best, 'primary_hi']:.3f})",
                  f"Statistically equivalent to the best (paired 95% interval of the difference includes 0): {equivalent}",
                  f"Recommended (highest split-half reliability among the equivalent ones): {rec:g} Hz"]
        if 0.5 in S.index:
            lines.append(f"0.5 Hz: primary {S.loc[0.5, 'primary_mean_abs_dz']:.3f}, "
                         f"{'in' if 0.5 in equivalent else 'NOT in'} the equivalent set")
        lines.append("Plateau width matters more than the maximum: if the equivalent set is wide, the choice within it is not critical.")
    vs_best.to_csv(out / "bootstrap_vs_best.csv", index=False)

    # ---- cognition
    cog_rows = []
    if teng is not None:
        y = teng.to_numpy(float)
        Za = age[:, None] if age is not None else None
        for fc in cutoffs:
            for i, s_ in enumerate(STAGES):
                d_ = delta_med[s_].to_numpy(float) if delta_med is not None else None
                Zad = np.column_stack([age, d_]) if (age is not None and d_ is not None) else None
                x = P[fc][:, i]
                plain, p_age = spearman_partial(x, y, Za)
                _, p_ad = spearman_partial(x, y, Zad)
                bs = [(spearman_partial(x[b], y[b], Za[b] if Za is not None else None),
                       spearman_partial(x[b], y[b], Zad[b] if Zad is not None else None)[1]) for b in boots[:300]]
                cog_rows.append(dict(cutoff=fc, stage=s_, n=int((np.isfinite(x) & np.isfinite(y)).sum()),
                                     rho=plain, rho_lo=ci([b[0][0] for b in bs])[0], rho_hi=ci([b[0][0] for b in bs])[1],
                                     partial_age=p_age, partial_age_lo=ci([b[0][1] for b in bs])[0],
                                     partial_age_hi=ci([b[0][1] for b in bs])[1],
                                     partial_age_delta=p_ad, partial_age_delta_lo=ci([b[1] for b in bs])[0],
                                     partial_age_delta_hi=ci([b[1] for b in bs])[1]))
        C = pd.DataFrame(cog_rows)
        C.to_csv(out / "cognition_by_stage.csv", index=False)
        lines.append("")
        lines.append(f"Cognitive check, Spearman of participant median H with {args.cog_col} (not used for the choice):")
        lines.append(C.pivot(index="cutoff", columns="stage", values="rho").reindex(columns=STAGES).round(2).to_string())
        if age is not None:
            lines.append("")
            lines.append("... partial for age (V8AGE):")
            lines.append(C.pivot(index="cutoff", columns="stage", values="partial_age").reindex(columns=STAGES).round(2).to_string())

    S.to_csv(out / "summary_by_cutoff.csv")
    pd.set_option("display.width", 250)
    show = ["n_epochs", "passes_gates", "median_r2", "frac_n3_gt_n2", "primary_mean_abs_dz", "auc_n3_pooled",
            "auc_n3_within", "split_half_reliability"] + [c for c in ("spearman_H_rel_delta",) if c in S]
    print("\n=== summary by cutoff ===")
    print(S[show].round(3).to_string())
    print("\n=== median H(q=2) per stage (participant medians) ===")
    print(pd.DataFrame(medians_long).pivot(index="cutoff", columns="stage", values="median_H")[STAGES].round(3).to_string())
    print("\n" + "\n".join(lines))
    (out / "recommendation.txt").write_text("\n".join(lines) + "\n")

    # ---- plots
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib missing; no plots")
            return 0
        x = np.arange(len(cutoffs))
        lab = [f"{c:g}" for c in cutoffs]

        def panel(ax, col, lo, hi, title):
            y_ = S[col].to_numpy(float)
            ax.errorbar(x, y_, yerr=[y_ - S[lo].to_numpy(float), S[hi].to_numpy(float) - y_], fmt="o-", capsize=3)
            ax.set_title(title, fontsize=9)

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        panel(axes[0, 0], "primary_mean_abs_dz", "primary_lo", "primary_hi", "PRIMARY: mean |d_z| over stage pairs")
        panel(axes[0, 1], "auc_n3_within", "auc_within_lo", "auc_within_hi", "AUC N3 vs rest (within participant)")
        axes[0, 1].plot(x, S["auc_n3_pooled"], "s--", label="pooled")
        axes[0, 1].legend(fontsize=8)
        panel(axes[0, 2], "split_half_reliability", "rel_lo", "rel_hi", "split-half reliability")
        if "spearman_H_rel_delta" in S:
            axes[1, 0].plot(x, S["spearman_H_rel_delta"], "o-", label="rel. delta")
            axes[1, 0].plot(x, S["spearman_H_slope_1_40"], "s-", label="slope 1-40 Hz")
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].set_title("within-participant Spearman of H with spectra", fontsize=9)
        for s in STAGES:
            axes[1, 1].plot(x, [np.nanmedian(P[fc][:, STAGES.index(s)]) for fc in cutoffs], "o-", label=s)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_title("median H(q=2) per stage", fontsize=9)
        if teng is not None:
            for s in ("N2", "N3", "REM"):
                g = C[C.stage == s].set_index("cutoff").loc[cutoffs]
                axes[1, 2].errorbar(x, g["rho"], yerr=[g["rho"] - g["rho_lo"], g["rho_hi"] - g["rho"]], fmt="o-",
                                    capsize=2, label=s)
            axes[1, 2].axhline(0, color="k", lw=.5)
            axes[1, 2].legend(fontsize=8)
            axes[1, 2].set_title(f"Spearman with {args.cog_col}", fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks(x, lab)
            ax.set_xlabel("high-pass cutoff (Hz)", fontsize=8)
            if best is not None and ax is axes[0, 0]:
                ax.axvline(cutoffs.index(best), color="grey", ls=":")
        fig.tight_layout()
        fig.savefig(out / "metrics_vs_cutoff.png", dpi=150)
        plt.close(fig)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
