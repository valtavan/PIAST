#!/usr/bin/env python3
"""
Which criticality_results_*.parquet reproduces Figure 1 of the paper?

For every file it computes the median H(q=2) per stage group (Pre-sleep, N1, N2, N3, REM,
Nocturnal wake, Post-sleep) on the same participants as the paper, and compares the pattern
with the values read off Figure 1 (top-left panel, C3A2; approximate, edit PAPER below).

File names are parsed too:  lf = high-pass cutoff, hf = low-pass cutoff, Nby4 = window divider,
hilbert / rawEEG = envelope or not.

Usage
    uv run --frozen python identify_paper_variant.py
        searches ~/pl1358-01/project_data/SOFdataset (recursively) for criticality_results_*.parquet.
        The files are assumed to contain only the paper's participants (N=290); the participant and
        epoch counts are printed so you can check this against the paper (290 / 347,232).
    uv run --frozen python identify_paper_variant.py DIR_OR_FILES... [--ids-file ids.txt]
        other location, and/or restrict to a list of ids.

Assumes the wide layout seen in the other parquets: subject_id, window_start_time_s, sleep_stage,
<CHANNEL>_h0..h9 with q = -5..-1, +1..+5 (so q=+2 is h6; change with --k).
If sleep_stage is not a usable stage (e.g. it just repeats window_start_time_s) the stage is
rebuilt from <hyp-dir>/hypnogram_<id>.csv (start_sec, stage_code).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("~/pl1358-01/project_data").expanduser()
CHANNELS = ["C3A2", "C4A1", "C3A1", "C4A2"]
GROUPS = ["Pre-sleep", "N1", "N2", "N3", "REM", "Nocturnal", "Post-sleep"]
# Approximate medians read off Figure 1, C3A2 panel (rendered image). Edit if you have the real values.
PAPER = {"Pre-sleep": 0.66, "N1": 0.66, "N2": 0.75, "N3": 0.87, "REM": 0.68, "Nocturnal": 0.64, "Post-sleep": 0.60}
PAPER_N_EPOCHS, PAPER_N_SUBJECTS = 347232, 290

DEFAULT_MAP = {"0": "wake", "1": "N1", "2": "N2", "3": "N3", "4": "REM"}
TEXT_MAP = {"w": "wake", "wake": "wake", "n1": "N1", "s1": "N1", "n2": "N2", "s2": "N2",
            "n3": "N3", "s3": "N3", "n4": "N3", "s4": "N3", "sws": "N3", "r": "REM", "rem": "REM"}


def parse_name(name):
    lf = re.search(r"lf([\d.]+?)_", name)
    hf = re.search(r"hf(None|[\d.]+?)(?:_|\.parquet)", name)
    return dict(lf=lf.group(1) if lf else "?", hf=hf.group(1) if hf else "?",
                envelope="hilbert" if "hilbert" in name else ("raw" if "rawEEG" in name else "?"))


def find_files(paths):
    out = []
    for p in map(lambda s: Path(s).expanduser(), paths):
        out += sorted(p.rglob("criticality_results_*.parquet")) if p.is_dir() else [p]
    return out


def normalise_stage(series, user_map):
    s = series.astype(str).str.strip()
    out = s.map(lambda v: user_map.get(v, TEXT_MAP.get(v.lower(), DEFAULT_MAP.get(v.split(".")[0], "other"))))
    return out


def stage_from_hypnogram(df, hyp_dir):
    out = np.full(len(df), "other", dtype=object)
    for sid, idx in df.groupby("subject_id").indices.items():
        f = hyp_dir / f"hypnogram_{sid}.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        m = dict(zip(h["start_sec"].round().astype(int), h["stage_code"].astype(int)))
        out[idx] = [DEFAULT_MAP.get(str(m.get(int(round(t)), -1)), "other") for t in df["window_start_time_s"].to_numpy()[idx]]
    return pd.Series(out, index=df.index)


def split_wake(df):
    """wake -> Pre-sleep / Nocturnal / Post-sleep by position relative to first / last sleep epoch."""
    df = df.sort_values(["subject_id", "window_start_time_s"]).copy()
    group = df["stage"].copy()
    for sid, g in df.groupby("subject_id"):
        sleep_t = g.loc[~g["stage"].isin(["wake", "other"]), "window_start_time_s"]
        if sleep_t.empty:
            continue
        first, last = sleep_t.min(), sleep_t.max()
        w = g["stage"] == "wake"
        group.loc[g.index[w & (g["window_start_time_s"] < first)]] = "Pre-sleep"
        group.loc[g.index[w & (g["window_start_time_s"] > last)]] = "Post-sleep"
        group.loc[g.index[w & (g["window_start_time_s"] >= first) & (g["window_start_time_s"] <= last)]] = "Nocturnal"
    df["group"] = group
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[str(PROJECT / "SOFdataset")],
                    help="files or folders (default: ~/pl1358-01/project_data/SOFdataset)")
    ap.add_argument("--ids-file", help="text/CSV with participant ids (first column), the paper's N=290")
    ap.add_argument("--k", type=int, default=6, help="h index of q=+2 (default 6)")
    ap.add_argument("--stage-map", default="", help='override, e.g. "0=wake,1=N1,2=N2,3=N3,4=REM"')
    ap.add_argument("--hyp-dir", default=str(PROJECT / "worksof/hypnograms/expert"))
    ap.add_argument("--out", default="paper_variant_comparison.csv")
    args = ap.parse_args()

    files = find_files(args.paths)
    if not files:
        sys.exit("no criticality_results_*.parquet files found in: " + ", ".join(args.paths))
    print(f"[files] {len(files)} found: " + ", ".join(f.name for f in files))
    user_map = dict(kv.split("=") for kv in args.stage_map.split(",") if "=" in kv)
    ids = None
    if args.ids_file:
        ids = set(pd.read_csv(args.ids_file, header=None, dtype=str).iloc[:, 0].str.strip().str.zfill(5))
        ids.discard("ID")
        print(f"[ids] {len(ids)} participant ids from {args.ids_file}")
    else:
        print("[ids] no --ids-file: using all participants in each file (assumed to be the paper's N=290)")

    rows, tables = [], {}
    for f in files:
        meta = parse_name(f.name)
        df = pd.read_parquet(f)
        df["subject_id"] = df["subject_id"].astype(str).str.strip().str.zfill(5)
        if ids is not None:
            df = df[df["subject_id"].isin(ids)]
        df = df.reset_index(drop=True)
        broken = "window_start_time_s" in df and (df["sleep_stage"] == df["window_start_time_s"]).mean() > 0.5
        if broken:
            print(f"[{f.name}] sleep_stage looks like start_sec -> rebuilding stage from hypnograms")
            df["stage"] = stage_from_hypnogram(df, Path(args.hyp_dir).expanduser())
        else:
            df["stage"] = normalise_stage(df["sleep_stage"], user_map)
        print(f"\n== {f.name}\n   lf={meta['lf']} hf={meta['hf']} envelope={meta['envelope']} | "
              f"{df['subject_id'].nunique()} participants, {len(df)} epochs (paper: {PAPER_N_SUBJECTS} / {PAPER_N_EPOCHS})")
        print("   stage values:", df["stage"].value_counts().to_dict())
        df = split_wake(df)

        cols = {c: f"{c}_h{args.k}" for c in CHANNELS if f"{c}_h{args.k}" in df}
        if not cols:
            print("   no wide-format h columns found, skipping")
            continue
        med = {c: df.groupby("group")[col].median() for c, col in cols.items()}
        tab = pd.DataFrame(med).reindex(GROUPS)
        tab["mean_4ch"] = tab[list(cols)].mean(axis=1)
        tables[f.name] = tab
        c3 = tab["C3A2"] if "C3A2" in tab else tab["mean_4ch"]
        ref = pd.Series(PAPER)
        ok = c3.notna()
        mad = float((c3[ok] - ref[ok]).abs().mean())
        rank_corr = float(c3[ok].rank().corr(ref[ok].rank()))
        n3_max = bool(c3.idxmax() == "N3")
        rows.append(dict(file=f.name, **meta, n_subjects=df["subject_id"].nunique(), n_epochs=len(df),
                         n3_is_highest=n3_max, rank_corr_with_paper=rank_corr, mean_abs_diff=mad,
                         **{g: c3.get(g, np.nan) for g in GROUPS}))
        print(tab.round(3).to_string())

    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("nothing compared")
    res = res.sort_values("mean_abs_diff")  # closest to the paper first; rank corr is noisy with near-tied groups
    pd.set_option("display.width", 250)
    print("\n=== summary (C3A2 median H(q=2) per stage group) ===")
    ref_row = pd.DataFrame([dict(file="PAPER (approx. from figure)", **{g: PAPER[g] for g in GROUPS})])
    print(pd.concat([ref_row, res]).drop(columns=["n_subjects", "n_epochs"], errors="ignore")
          .round(3).to_string(index=False))
    res.to_csv(args.out, index=False)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
