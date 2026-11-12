#!/usr/bin/env python
"""through_night_cognition.py - does the through-night change in criticality relate to cognitive decline?

Put this file in the criticality/ folder (next to through_night.py and delta_models.py). It reuses their
windows, hypnogram handling and functions, and the Firth / permutation helpers of complexity/lz_h3_firth_perm.py.

What it does
  1. Reads the MFDFA parquet files ({subject_id}.parquet from sof_mfdfa.py) and builds, per subject and stage,
     the median of each measure in the EARLY window (first 1.5 h of the sleep period) and the LATE window
     (last 1.5 h):  delta = late - early.  (Same windows, onset rule and MIN_EPOCHS as through_night.py.)
  2. Joins cognition (cognitive.csv: V9TENG, V8AGE, PREMMSE and an AHI column).
  3. Tests delta against the outcome, per stage x measure:
       continuous  V9TENG ~ delta (per 1 SD) + covariates, OLS with HC3 robust errors, plus the R^2 gained
       binary      decline = V9TENG < cutoff, for EACH cutoff in --teng-cuts (default 80 and 88):  Mann-Whitney AUC (unadjusted; the chance that a woman with decline
                   has a HIGHER delta than a stable one, so < 0.5 means decline goes with a lower delta), Firth
                   logistic OR per SD with profile-likelihood CI and a Freedman-Lane permutation p
  4. Covariate sets (M1 -> M3 each add to the one before):
       M1  age (V8AGE) + baseline MMSE (PREMMSE) + AHI
       M2  M1 + the EARLY value of the same measure (guards against regression to the mean)
       M3  M2 + change in spectral features (--spectral-dir) and/or extra change covariates (--extra-csv, e.g.
           change in arousal / event rate).  This is the "beyond the spectrum / beyond SDB" test.
     The two default cutoffs answer different questions: < 88 is "MCI or worse" (many cases, more power), < 80 is
     closer to "dementia-range" (fewer cases). A delta that works at 88 but not at 80 points to subtle decline;
     the reverse points to a threshold effect. The continuous V9TENG model does not depend on any cutoff.
  5. Holm correction across the stage x measure grid, separately for each outcome, cutoff and covariate set; the
     pre-specified primary test (--primary-stage / --primary-measure) is flagged and also reported on its own.
  6. Attrition check (does delta differ between women with and without a Visit-9 score?) and a figure.

Outputs in --out-dir (default <results_dir>/through_night_cognition/):
  subject_deltas.csv, continuous_models.csv, binary_models.csv, attrition.csv, sleep_period_info.csv,
  figure_primary_cut<cutoff>.png (one per cutoff)

Usage
  uv run --frozen python through_night_cognition.py
  uv run --frozen python through_night_cognition.py --teng-cuts 78 80 88
  uv run --frozen python through_night_cognition.py --spectral-dir ~/pl1358-01/project_data/worksof/criticality/spectral
  uv run --frozen python through_night_cognition.py --extra-csv event_deltas.csv --extra-covs d_arousal_rate d_event_rate
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "complexity"))  # Firth / permutation helpers

import paths  # noqa: E402
import through_night as tn  # noqa: E402
import delta_models as dm  # noqa: E402

try:
    from lz_h3_firth_perm import firth_dz, perm_score_test  # noqa: E402

    HAVE_FIRTH, FIRTH_ERR = True, ""
except Exception as exc:  # the binary Firth / permutation columns are then left empty
    HAVE_FIRTH, FIRTH_ERR = False, str(exc)

STAGE_CODE = {name: code for code, name in tn.STAGES.items()}  # 'N2' -> 2
DEFAULT_AHI = "nsrr_ahi_hp3r_aasm15"


# ------------------------------------------------------------------------------------------- data
def zs(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x * np.nan


def load_cognitive(path: Path, ahi_col: str) -> pd.DataFrame:
    cog = pd.read_csv(path)
    need = ["ID", "V9TENG", "V8AGE", "PREMMSE", ahi_col]
    missing = [c for c in need if c not in cog.columns]
    if missing:
        sys.exit(f"{path.name} lacks columns {missing}. Columns: {list(cog.columns)}")
    ids = pd.to_numeric(cog["ID"], errors="coerce").astype("Int64")
    cog["subject_id"] = ids.map(lambda x: f"{x:05d}" if pd.notna(x) else None)
    cog = cog.dropna(subset=["subject_id"]).drop_duplicates("subject_id")
    return cog[["subject_id", "V9TENG", "V8AGE", "PREMMSE", ahi_col]].rename(columns={ahi_col: "AHI"})


def windowed_medians(results_dir: Path, hyp_dir: Path, measures: list[str], min_r2: float):
    parquets = sorted(results_dir.glob("[0-9]*.parquet"))
    if not parquets:
        sys.exit(f"No {{subject_id}}.parquet files in {results_dir}")
    tables, infos = [], []
    for pq in parquets:
        table, info = tn.load_subject(pq, hyp_dir, min_r2)
        infos.append(info)
        if table is not None:
            tables.append(table)
    if not tables:
        sys.exit("No subject had a usable sleep period / hypnogram.")
    medians = tn.stage_medians(pd.concat(tables, ignore_index=True), measures)
    return medians, pd.DataFrame(infos)


def delta_table(medians: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    """One row per (stage_code, subject_id): d_<m> = late - early and e_<m> = early median."""
    deltas = dm.window_deltas(medians, measures, prefix="d_")
    early = (medians[(medians["n_epochs"] >= tn.MIN_EPOCHS) & (medians["window"] == "early")]
             .set_index(["stage_code", "subject_id"])[measures].add_prefix("e_"))
    return deltas.join(early, how="left").reset_index()


def add_spectral(table: pd.DataFrame, spectral_dir: Path, hyp_dir: Path) -> pd.DataFrame:
    spec = dm.window_deltas(dm.spectral_medians(spectral_dir, hyp_dir), dm.SPECTRAL, prefix="s_")
    return table.merge(spec.reset_index(), on=["stage_code", "subject_id"], how="left")


def add_extra(table: pd.DataFrame, csv: Path, cols: list[str]) -> pd.DataFrame:
    extra = pd.read_csv(csv, dtype={"subject_id": str, "subject": str})
    if "subject_id" not in extra.columns and "subject" in extra.columns:
        extra = extra.rename(columns={"subject": "subject_id"})
    extra["subject_id"] = extra["subject_id"].str.split(".").str[0].str.zfill(5)
    missing = [c for c in cols if c not in extra.columns]
    if missing:
        sys.exit(f"{csv.name} lacks columns {missing}")
    keys = ["subject_id"] + (["stage_code"] if "stage_code" in extra.columns else [])
    return table.merge(extra[keys + cols], on=keys, how="left")


# ------------------------------------------------------------------------------------- modelling
def design(d: pd.DataFrame, pred: str, early: str, extra: list[str], model: str):
    """Rows with complete data and the z-scored columns used by the model. Returns (frame, covariate names)."""
    covs = ["V8AGE", "PREMMSE", "AHI"]
    if model in ("M2", "M3"):
        covs.append(early)
    if model == "M3":
        covs += extra
    use = d.dropna(subset=["V9TENG", pred] + covs).copy()
    use["z_d"] = zs(use[pred])
    names = []
    for i, c in enumerate(covs):
        use[f"z_c{i}"] = zs(use[c])
        names.append(f"z_c{i}")
    return use.dropna(subset=["z_d"] + names), names


def fit_continuous(use: pd.DataFrame, names: list[str]) -> dict:
    rhs = " + ".join(names)
    full = smf.ols(f"V9TENG ~ z_d + {rhs}", use).fit(cov_type="HC3")
    r2_full = smf.ols(f"V9TENG ~ z_d + {rhs}", use).fit().rsquared
    r2_base = smf.ols(f"V9TENG ~ {rhs}", use).fit().rsquared
    lo, hi = full.conf_int().loc["z_d"]
    return {"n": len(use), "beta_per_sd": full.params["z_d"], "se": full.bse["z_d"], "ci_low": lo, "ci_high": hi,
            "p": full.pvalues["z_d"], "delta_r2": r2_full - r2_base}


def fit_binary(use: pd.DataFrame, names: list[str], cut: float, n_perm: int, min_events: int, rng) -> dict:
    y = (use["V9TENG"] < cut).astype(float).to_numpy()
    out = {"n": len(use), "events": int(y.sum()), "epv": y.sum() / max(1, len(names) + 1)}
    if y.sum() < min_events or (len(y) - y.sum()) < min_events:
        out["note"] = f"fewer than {min_events} events or non-events"
        return out
    if not HAVE_FIRTH:
        out["note"] = f"Firth helpers unavailable ({FIRTH_ERR})"
        return out
    X = np.column_stack([np.ones(len(use)), use["z_d"].to_numpy()] + [use[n].to_numpy() for n in names])
    try:
        _, f = firth_dz(X, y, j=1)
        out.update(or_per_sd=f["or"], or_low=f["or_lo"], or_high=f["or_hi"], p_firth=f["p_firth_lr"])
        out["p_perm"] = perm_score_test(y, X, n_perm, rng, j=1)
    except Exception as exc:
        out["note"] = f"model failed: {exc}"
    return out


def auc_unadjusted(use: pd.DataFrame, cut: float) -> dict:
    y = use["V9TENG"] < cut
    a, b = use.loc[y, "z_d"], use.loc[~y, "z_d"]
    if len(a) < 5 or len(b) < 5:
        return {}
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"auc_unadj": u.statistic / (len(a) * len(b)), "p_mwu": u.pvalue}


def holm_within(df: pd.DataFrame, pcol: str, by: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(by).groups.items():
        out.loc[idx] = tn.holm(df.loc[idx, pcol].to_numpy(float))
    return out


# ------------------------------------------------------------------------------------------ figure
def plot_primary(table, measure, stages, cut, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(stages), figsize=(max(3.6 * len(stages), 6.5), 4), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        d = table[(table["stage_code"] == STAGE_CODE[stage])].dropna(subset=["V9TENG", f"d_{measure}"])
        groups = [d.loc[d["V9TENG"] >= cut, f"d_{measure}"], d.loc[d["V9TENG"] < cut, f"d_{measure}"]]
        ax.boxplot(groups, widths=0.5, showfliers=False)
        for i, g in enumerate(groups, start=1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(g)), g, s=8, alpha=0.5)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xticks([1, 2], [f"stable\nn={len(groups[0])}", f"decline\nn={len(groups[1])}"])
        ax.set_title(stage)
        ax.set_ylabel(f"late - early  {measure}" if stage == stages[0] else "")
    fig.suptitle(f"Through-night change of {measure} by outcome (decline = V9TENG < {cut:g})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default=paths.RESULTS_DIR, type=lambda s: Path(s).expanduser())
    ap.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=paths.HYP_DIR)
    ap.add_argument("--cognitive", type=lambda s: Path(s).expanduser(), default=paths.PROJECT / "worksof/cognitive.csv")
    ap.add_argument("--out-dir", type=lambda s: Path(s).expanduser())
    ap.add_argument("--ahi-col", default=DEFAULT_AHI, help=f"AHI column in cognitive.csv (default {DEFAULT_AHI})")
    ap.add_argument("--stages", nargs="+", default=["N2", "SWS", "REM"], choices=list(STAGE_CODE))
    ap.add_argument("--measures", nargs="+", default=["h_q+2", "delta_h"], choices=tn.MEASURES)
    ap.add_argument("--primary-stage", default="N2", choices=list(STAGE_CODE))
    ap.add_argument("--primary-measure", default="h_q+2")
    ap.add_argument("--teng-cuts", type=float, nargs="+", default=[80.0, 88.0],
                    help="decline = V9TENG < cutoff; one set of binary models per cutoff (default 80 and 88)")
    ap.add_argument("--baseline-min", type=float, default=24.0, help="exclude PREMMSE below this (0 = off)")
    ap.add_argument("--min-r2", type=float, default=0.0)
    ap.add_argument("--min-subjects", type=int, default=40, help="minimum women for a model")
    ap.add_argument("--min-events", type=int, default=10, help="minimum decline events for the binary models")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spectral-dir", type=lambda s: Path(s).expanduser(), help="sof_spectral.py output folder")
    ap.add_argument("--spectral-covs", nargs="+", default=["s_slope_1_40", "s_log10_power"],
                    help="spectral change covariates for M3 (names: s_ + a column of delta_models.SPECTRAL)")
    ap.add_argument("--extra-csv", type=lambda s: Path(s).expanduser(),
                    help="CSV with subject_id (and optionally stage_code) plus extra change covariates")
    ap.add_argument("--extra-covs", nargs="*", default=[], help="column names in --extra-csv used in M3")
    a = ap.parse_args()

    if a.primary_measure not in a.measures:
        sys.exit("--primary-measure must be one of --measures")
    if a.primary_stage not in a.stages:
        a.stages.append(a.primary_stage)
    out_dir = a.out_dir or a.results_dir / "through_night_cognition"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    a.teng_cuts = sorted(set(a.teng_cuts))
    warnings.filterwarnings("ignore")

    # ---- through-night deltas ------------------------------------------------------------------
    medians, infos = windowed_medians(a.results_dir, a.hyp_dir, a.measures, a.min_r2)
    infos.to_csv(out_dir / "sleep_period_info.csv", index=False)
    table = delta_table(medians, a.measures)
    extra = []
    if a.spectral_dir:
        table = add_spectral(table, a.spectral_dir, a.hyp_dir)
        extra += a.spectral_covs
    if a.extra_csv:
        table = add_extra(table, a.extra_csv, a.extra_covs)
        extra += a.extra_covs
    table = table.merge(load_cognitive(a.cognitive, a.ahi_col), on="subject_id", how="left")
    table["has_v9"] = table["V9TENG"].notna()
    table["stage"] = table["stage_code"].map(tn.STAGES)
    if a.baseline_min > 0:
        table = table[table["PREMMSE"].isna() | (table["PREMMSE"] >= a.baseline_min)]
    table.to_csv(out_dir / "subject_deltas.csv", index=False)

    print(f"subjects with a usable sleep period: {infos['subject_id'].nunique() - infos.get('excluded', pd.Series(dtype=str)).notna().sum()}"
          f" of {infos['subject_id'].nunique()}")
    print(f"windows: first/last {tn.WINDOW_H:g} h of the sleep period; >= {tn.MIN_EPOCHS} epochs per stage and window")
    for stage in a.stages:
        t = table[(table["stage_code"] == STAGE_CODE[stage])]
        n_d = t[f"d_{a.primary_measure}"].notna().sum()
        n_c = t.dropna(subset=[f"d_{a.primary_measure}", "V9TENG"])
        counts = ";  ".join(f"V9TENG < {c:g}: {int((n_c['V9TENG'] < c).sum())}" for c in a.teng_cuts)
        print(f"  {stage:<4} women with a delta: {n_d:>4};  with V9TENG too: {len(n_c):>4};  decline cases  {counts}")
    core = table.drop_duplicates("subject_id").dropna(subset=["V9TENG"])
    edges = [-np.inf] + a.teng_cuts + [np.inf]
    bands = pd.cut(core["V9TENG"], bins=edges, right=False)
    print("V9TENG bands (all women with a score, any stage): "
          + ";  ".join(f"{str(k)} n={v}" for k, v in bands.value_counts().sort_index().items()))
    if not a.spectral_dir and not a.extra_csv:
        print("NOTE: no --spectral-dir / --extra-csv given, so M3 equals M2.")

    # ---- models -----------------------------------------------------------------------------------
    models = ["M1", "M2"] + (["M3"] if extra else [])
    cont_rows, bin_rows, att_rows = [], [], []
    for stage in a.stages:
        d_stage = table[table["stage_code"] == STAGE_CODE[stage]]
        for m in a.measures:
            pred, early = f"d_{m}", f"e_{m}"
            base = {"stage": stage, "measure": m, "primary": stage == a.primary_stage and m == a.primary_measure}
            # attrition: delta in women with vs without a Visit-9 score
            g = d_stage.dropna(subset=[pred])
            if g["has_v9"].nunique() == 2:
                x, y = g.loc[g["has_v9"], pred], g.loc[~g["has_v9"], pred]
                u = stats.mannwhitneyu(x, y)
                att_rows.append({**base, "n_with_v9": len(x), "n_without_v9": len(y), "median_delta_with": x.median(),
                                 "median_delta_without": y.median(), "p_mwu": u.pvalue})
            for model in models:
                use, names = design(d_stage, pred, early, extra, model)
                if len(use) < a.min_subjects:
                    continue
                cont_rows.append({**base, "model": model, **fit_continuous(use, names)})
                for cut in a.teng_cuts:
                    row = {**base, "model": model, "cutoff": cut,
                           **fit_binary(use, names, cut, a.n_perm, a.min_events, rng)}
                    if model == "M1":
                        row.update(auc_unadjusted(use, cut))
                    bin_rows.append(row)

    cont, binary = pd.DataFrame(cont_rows), pd.DataFrame(bin_rows)
    if len(cont):
        cont["p_holm"] = holm_within(cont, "p", ["model"])
        cont.to_csv(out_dir / "continuous_models.csv", index=False)
    if len(binary) and "p_perm" in binary:
        binary["p_perm_holm"] = holm_within(binary.assign(_p=binary["p_perm"]), "_p", ["model", "cutoff"])
    if len(binary):
        binary.to_csv(out_dir / "binary_models.csv", index=False)
    if att_rows:
        pd.DataFrame(att_rows).to_csv(out_dir / "attrition.csv", index=False)

    # ---- report -------------------------------------------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3g}".format)
    if len(cont):
        print("\n== CONTINUOUS: V9TENG ~ delta (per 1 SD) + covariates  [OLS, HC3]  Holm over the stage x measure grid ==")
        print(cont[["stage", "measure", "model", "n", "beta_per_sd", "ci_low", "ci_high", "p", "p_holm", "delta_r2"]]
              .to_string(index=False))
        prim = cont[cont["primary"]]
        if len(prim):
            print(f"\nPRE-SPECIFIED PRIMARY TEST ({a.primary_stage}, {a.primary_measure}) - no multiplicity correction:")
            print(prim[["model", "n", "beta_per_sd", "ci_low", "ci_high", "p", "delta_r2"]].to_string(index=False))
    if len(binary):
        cols = [c for c in ["stage", "measure", "model", "cutoff", "n", "events", "epv", "auc_unadj", "p_mwu", "or_per_sd",
                            "or_low", "or_high", "p_firth", "p_perm", "p_perm_holm", "note"] if c in binary]
        print(f"\n== BINARY: decline = V9TENG < {' and < '.join(f'{c:g}' for c in a.teng_cuts)}  "
              "[Firth OR per SD; permutation p; Holm within model and cutoff] ==")
        print(binary[cols].to_string(index=False))
    if att_rows:
        print("\n== ATTRITION: delta in women with vs without a Visit-9 score (p < 0.05 means informative dropout) ==")
        print(pd.DataFrame(att_rows)[["stage", "measure", "n_with_v9", "n_without_v9", "median_delta_with",
                                      "median_delta_without", "p_mwu"]].to_string(index=False))
    for cut in a.teng_cuts:
        try:
            plot_primary(table, a.primary_measure, a.stages, cut, out_dir / f"figure_primary_cut{cut:g}.png")
        except Exception as exc:
            print(f"(figure for cutoff {cut:g} skipped: {exc})")
    print(f"\nresults written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
