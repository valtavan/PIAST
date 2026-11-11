#!/usr/bin/env python3
"""
create_age_file.py - build age-participant.csv (columns: nsrrid, v8age) by JOINING ON PARTICIPANT ID,
and audit whether the earlier row-position version (pd.concat(..., axis=1)) put wrong ages on anyone.

Why: the old script paired df1["nsrrid"] (harmonized dataset) with df2["v8age"] (raw dataset) purely by
row position. That is only right if both files list the same participants in the same order.

What this script does, in order
  1. Reads the participant ID and age from the two source CSVs.
  2. Reproduces what the OLD position-based pairing gives, and counts how many ages differ from the
     correct ID-based join  -> tells you whether the old file could have been wrong.
  3. Checks the age-participant.csv files currently on disk (the ones YASA and merge_age.py read).
  4. Cross-checks against the harmonized `nsrr_age` column, if the file has one (independent of the join).
  5. Checks downstream copies of age: V8AGE in cognitive.csv, and age_used in the YASA batch logs.
  6. Writes a report (age_mismatches.csv, affected_subjects.csv) and a verdict.
  7. Unless --check-only: writes the corrected file(s), backing up any existing file first.

If the raw file has NO ID column (the SOF visit-8 file does not), the files cannot be joined. Then:
  - default: the script checks the row alignment by comparing raw v8age with harmonized nsrr_age row by row
    (aligned files agree within a year on nearly every row; misaligned ones only at chance level)
  - --use-nsrr-age: builds the age file from the harmonized file alone (nsrrid and nsrr_age sit in the
    same row, so no join is needed) and audits everything else against it. Use this if the alignment
    check fails, or if you prefer a file that cannot be misaligned by construction.

Exit status: 0 = nothing was wrong, 2 = at least one wrong age found, 1 = error / cannot join safely.

Usage
  python create_age_file.py --check-only      # audit only, writes the report but changes no age file
  python create_age_file.py                   # audit, then write the corrected age file(s)
"""
from __future__ import annotations

import argparse
import glob
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("~/pl1358-01/project_data").expanduser()
DATASETS = BASE / "SOFdataset" / "sleep" / "datasets"
WORKSOF = BASE / "worksof"

HARMONIZED = DATASETS / "sof-visit-8-harmonized-dataset-0.8.0.csv"
RAW = DATASETS / "sof-visit-8-dataset-0.8.0.csv"
# yasa_edf_raw.py reads the first location, merge_age.py reads the second. Both are written so that
# neither consumer sees a stale copy.
DEFAULT_OUTS = [DATASETS / "age-participant.csv", WORKSOF / "age-participant.csv"]
COGNITIVE = WORKSOF / "cognitive.csv"
YASA_DIR = WORKSOF / "hypnograms" / "yasaRAW"
REPORT_DIR = WORKSOF / "age_check"

NSRR_AGE_TOL = 1.0  # years; harmonized nsrr_age may be rounded or computed differently
NSRR_AGE_TOPCODE = 89  # rows with nsrr_age >= this are skipped in the cross-check (possible top-coding)


# ------------------------------------------------------------------------------------------ helpers
def to_id(series: pd.Series) -> pd.Series:
    """'00004', '4', '4.0', 4 -> 4 (nullable int), so IDs from different files compare equal."""
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return pd.to_numeric(text, errors="coerce").astype("Int64")


def differs(a: pd.Series, b: pd.Series, tol: float = 1e-6) -> pd.Series:
    """True where a and b are different; two missing values count as equal."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return ~((a.isna() & b.isna()) | ((a - b).abs() <= tol))


def resolve_column(path: Path, name: str, required: bool = True) -> str | None:
    """Actual header spelling of `name` (case-insensitive), or None / exit if absent."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    found = {str(c).strip().lower(): c for c in header}.get(name.lower())
    if found is None and required:
        sys.exit(f"ERROR: {path.name} has no column '{name}'. Columns are: {list(header)[:40]}"
                 f"{' ...' if len(header) > 40 else ''}\n"
                 f"       Without a shared ID column the two files cannot be joined safely; "
                 f"use --id-col / --raw-id-col to name it.")
    return found


def truth_lookup(frame: pd.DataFrame) -> dict[int, float]:
    return {int(i): a for i, a in zip(frame["_id"], frame["v8age"]) if pd.notna(i)}


def compare_to_truth(source: str, ids: pd.Series, ages: pd.Series, truth: dict[int, float],
                     tol: float = 1e-6, skip_unknown: bool = False) -> tuple[int, pd.DataFrame]:
    """(rows compared, rows whose age differs from the ID-joined age)."""
    df = pd.DataFrame({"_id": to_id(ids), "age_in_source": pd.to_numeric(ages, errors="coerce")})
    df = df.dropna(subset=["_id"]).drop_duplicates("_id")
    df["age_correct"] = df["_id"].map(lambda x: truth.get(int(x), np.nan))
    df = df[df["age_correct"].notna()] if skip_unknown else df[df["age_in_source"].notna() | df["age_correct"].notna()]
    bad = df[differs(df["age_in_source"], df["age_correct"], tol)].copy()
    bad.insert(0, "source", source)
    return len(df), bad


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harmonized", type=Path, default=HARMONIZED, help="file the ID column comes from")
    p.add_argument("--raw", type=Path, default=RAW, help="file the v8age column comes from")
    p.add_argument("--id-col", default="nsrrid", help="ID column in --harmonized")
    p.add_argument("--raw-id-col", default="nsrrid", help="ID column in --raw")
    p.add_argument("--age-col", default="v8age", help="age column in --raw")
    p.add_argument("--out", type=Path, action="append", help="output file; repeat for several "
                   f"(default: {', '.join(str(o) for o in DEFAULT_OUTS)})")
    p.add_argument("--cognitive", type=Path, default=COGNITIVE, help="cognitive.csv to audit (V8AGE column)")
    p.add_argument("--yasa-dir", type=Path, default=YASA_DIR, help="folder with _batch_run_log_*.csv to audit")
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--use-nsrr-age", action="store_true",
                   help="take ages from the harmonized file's nsrr_age (same row as nsrrid; no join needed)")
    p.add_argument("--check-only", action="store_true", help="audit and report, but do not write any age file")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    outs = a.out or DEFAULT_OUTS
    for f in (a.harmonized, a.raw):
        if not f.exists():
            sys.exit(f"ERROR: {f} not found")

    # ---- 1. read the two sources --------------------------------------------------------------
    id1 = resolve_column(a.harmonized, a.id_col)
    id2 = resolve_column(a.raw, a.raw_id_col, required=False)
    age2 = resolve_column(a.raw, a.age_col)
    nsrr_age = resolve_column(a.harmonized, "nsrr_age", required=False)
    no_id = id2 is None
    use_nsrr = a.use_nsrr_age
    if use_nsrr and not nsrr_age:
        sys.exit("ERROR: --use-nsrr-age needs an 'nsrr_age' column in the harmonized file.")
    if no_id and not nsrr_age:
        header = pd.read_csv(a.raw, nrows=0, encoding="utf-8-sig").columns
        idlike = [c for c in header if re.search(r"id", str(c).lower())]
        sys.exit(f"ERROR: {a.raw.name} has no '{a.raw_id_col}' column and the harmonized file has no 'nsrr_age' "
                 f"to check the row order against.\n       Columns containing 'id' in the raw file: {idlike}\n"
                 f"       Name the right one with --raw-id-col, or find another file that has both an ID and an age.")

    d1 = pd.read_csv(a.harmonized, usecols=[id1] + ([nsrr_age] if nsrr_age else []), encoding="utf-8-sig")
    d2 = pd.read_csv(a.raw, usecols=[age2] + ([id2] if id2 else []), encoding="utf-8-sig")
    d1["_id"] = to_id(d1[id1])
    d2 = d2.rename(columns={age2: "v8age"})
    if id2:
        d2["_id"] = to_id(d2[id2])

    say("=" * 78)
    say("AGE FILE AUDIT")
    say("=" * 78)
    say(f"{a.harmonized.name}: {len(d1)} rows, {d1['_id'].nunique()} unique IDs, {int(d1['_id'].isna().sum())} unreadable IDs")
    if id2:
        say(f"{a.raw.name}: {len(d2)} rows, {d2['_id'].nunique()} unique IDs, {int(d2['_id'].isna().sum())} unreadable IDs")
    else:
        header = pd.read_csv(a.raw, nrows=0, encoding="utf-8-sig").columns
        say(f"{a.raw.name}: {len(d2)} rows, NO ID COLUMN "
            f"(columns containing 'id': {[c for c in header if re.search('id', str(c).lower())]})")

    problems = []
    for name, d in (("harmonized", d1),) + ((("raw", d2),) if id2 else ()):
        dup = d[d["_id"].notna() & d["_id"].duplicated(keep=False)]
        if len(dup):
            problems.append(f"{name} file has {dup['_id'].nunique()} IDs that appear more than once")
    if problems:
        sys.exit("ERROR: cannot join safely: " + "; ".join(problems) + ". Resolve duplicates first.")

    # ---- 1b. no ID in the raw file: check the row alignment against nsrr_age ----------------------
    if no_id and not use_nsrr:
        m = min(len(d1), len(d2))
        x = pd.to_numeric(d1[nsrr_age].iloc[:m], errors="coerce").to_numpy(float)
        y = pd.to_numeric(d2["v8age"].iloc[:m], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y) & (x < NSRR_AGE_TOPCODE)
        frac = float((np.abs(x - y)[ok] <= NSRR_AGE_TOL).mean()) if ok.any() else float("nan")
        exact = float((np.abs(x - y)[ok] < 1e-9).mean()) if ok.any() else float("nan")
        rng = np.random.default_rng(0)
        chance = float(np.mean([(np.abs(x[ok] - rng.permutation(y[ok])) <= NSRR_AGE_TOL).mean() for _ in range(200)]))
        say(f"\nNo shared ID, so row alignment is checked through age: raw v8age vs harmonized {nsrr_age}, row by row.")
        say(f"  row counts: {len(d1)} vs {len(d2)}  ({'equal' if len(d1) == len(d2) else 'DIFFERENT'})")
        say(f"  rows compared: {int(ok.sum())};  within {NSRR_AGE_TOL:g} y: {100 * frac:.1f}%;  exactly equal: {100 * exact:.1f}%")
        say(f"  expected if the row order were unrelated (shuffled): {100 * chance:.1f}% within {NSRR_AGE_TOL:g} y")
        aligned = len(d1) == len(d2) and frac >= 0.97
        broken = len(d1) != len(d2) or frac < 0.90
        say("\n" + "=" * 78)
        if aligned:
            say("VERDICT: ALIGNED. The two files agree on age row by row, so the old position-based pairing was correct")
            say("         (to the extent age can show it). No analysis needs rerunning because of this.")
            if not a.check_only:
                out_df = pd.concat([d1["_id"].reset_index(drop=True), d2["v8age"].reset_index(drop=True)], axis=1)
                out_df.columns = ["nsrrid", "v8age"]
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for path in outs:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        shutil.copy2(path, path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}"))
                    tmp = path.with_suffix(".tmp")
                    out_df.to_csv(tmp, index=False)
                    tmp.replace(path)
                    say(f"wrote {path}  ({len(out_df)} rows; position-paired, alignment verified above)")
            say("=" * 78)
            return 0
        if broken:
            say("VERDICT: NOT ALIGNED. The old position-based file very likely holds wrong ages.")
        else:
            say("VERDICT: INCONCLUSIVE. Agreement is neither near-perfect nor at chance level; do not trust the old file.")
        say("  Rebuild it without any join: rerun with --use-nsrr-age (ages from the harmonized file's nsrr_age),")
        say("  which also audits cognitive.csv and the YASA logs and tells you what to rerun.")
        say("=" * 78)
        return 2

    # ---- 2. correct ID-based join; reproduce the old position-based pairing --------------------
    d1c = d1.dropna(subset=["_id"])
    if use_nsrr:
        corr = d1c[["_id"]].copy()
        corr["v8age"] = pd.to_numeric(d1c[nsrr_age], errors="coerce")
        n_top = int((corr["v8age"] >= NSRR_AGE_TOPCODE).sum())
        # ages from nsrr_age are only used as the reference below where they are not possibly top-coded
        ref_ok = corr[corr["v8age"] < NSRR_AGE_TOPCODE]
        truth = truth_lookup(ref_ok.rename(columns={}))
        say(f"\n--use-nsrr-age: ages taken from '{nsrr_age}' in the harmonized file (same row as the ID). "
            f"{n_top} participants have nsrr_age >= {NSRR_AGE_TOPCODE} (possibly top-coded): kept in the output, "
            f"excluded from the audit. Audit tolerance: {NSRR_AGE_TOL:g} y.")
        cmp_kw = dict(tol=NSRR_AGE_TOL, skip_unknown=True)
    else:
        d2c = d2.dropna(subset=["_id"])
        corr = d1c[["_id"]].merge(d2c[["_id", "v8age"]], on="_id", how="outer", indicator=True)
        only1, only2 = int((corr["_merge"] == "left_only").sum()), int((corr["_merge"] == "right_only").sum())
        truth = truth_lookup(corr)
        cmp_kw = {}
        say(f"\nID-based join: {int((corr['_merge'] == 'both').sum())} IDs in both files, "
            f"{only1} only in harmonized (age will be missing), {only2} only in raw (kept, with their age)")
        say(f"missing v8age after join: {int(corr['v8age'].isna().sum())}")

    n = max(len(d1), len(d2))
    old_pair = pd.concat([d1["_id"].reset_index(drop=True), d2["v8age"].reset_index(drop=True)], axis=1)
    if id2:
        pos_id2 = d2["_id"].reset_index(drop=True).reindex(range(len(old_pair)))
        pos_mismatch = int((old_pair["_id"].reset_index(drop=True) != pos_id2).fillna(True).sum())
        say(f"\nRow-position check: {pos_mismatch} of {n} row positions hold a DIFFERENT participant in the two files"
            f" (lengths {len(d1)} vs {len(d2)}).")
        if pos_mismatch == 0 and len(d1) == len(d2):
            say("  -> the two files list the same participants in the same order; the old pairing was correct.")

    frames, summary = [], []
    n_cmp, bad = compare_to_truth("old pairing, recomputed from the source files", old_pair["_id"],
                                  old_pair["v8age"], truth, **cmp_kw)
    frames.append(bad)
    summary.append(("old pairing, recomputed from source files", n_cmp, len(bad)))

    # ---- 3. age files currently on disk -------------------------------------------------------
    for path in dict.fromkeys(list(outs) + DEFAULT_OUTS):
        if path.exists():
            cur = pd.read_csv(path, encoding="utf-8-sig")
            cols = {str(c).strip().lower(): c for c in cur.columns}
            if "nsrrid" in cols and "v8age" in cols:
                n_cmp, bad = compare_to_truth(f"on disk: {path}", cur[cols["nsrrid"]], cur[cols["v8age"]], truth, **cmp_kw)
                frames.append(bad)
                summary.append((f"on disk: {path}", n_cmp, len(bad)))
            else:
                summary.append((f"on disk: {path} (no nsrrid/v8age columns; skipped)", 0, 0))

    # ---- 4. independent cross-check with the harmonized nsrr_age (does not depend on the join) -
    if nsrr_age and not use_nsrr and id2:
        ref = pd.DataFrame({"_id": d1["_id"], "nsrr_age": pd.to_numeric(d1[nsrr_age], errors="coerce")})
        ref = ref[ref["nsrr_age"].notna() & (ref["nsrr_age"] < NSRR_AGE_TOPCODE)]
        ref_truth = dict(zip(ref["_id"].astype(int), ref["nsrr_age"]))
        say(f"\nIndependent check against harmonized '{nsrr_age}' (|difference| > {NSRR_AGE_TOL:g} y counts as a "
            f"disagreement; {len(ref)} participants with nsrr_age < {NSRR_AGE_TOPCODE}):")
        for label, frame in (("correct ID-based join", corr[["_id", "v8age"]]),
                             ("old position pairing", old_pair.rename(columns={"_id": "_id"})[["_id", "v8age"]])):
            f = frame.dropna(subset=["_id"])
            f = f[f["_id"].astype(int).isin(ref_truth)]
            diff = (pd.to_numeric(f["v8age"], errors="coerce").to_numpy()
                    - np.array([ref_truth[int(i)] for i in f["_id"]]))
            n_bad = int((np.abs(diff) > NSRR_AGE_TOL).sum())
            say(f"  {label:<24} {n_bad:>4} of {len(f)} disagree with nsrr_age"
                f"   (median |diff| {np.nanmedian(np.abs(diff)):.2f} y)")
        say("  If the correct join also shows many disagreements, nsrr_age and v8age measure different things; "
            "compare the two lines, not the absolute number.")
    elif not use_nsrr:
        say("\n(independent nsrr_age cross-check skipped)")

    # ---- 5. downstream copies of age ----------------------------------------------------------
    if a.cognitive.exists():
        cog = pd.read_csv(a.cognitive)
        col = next((c for c in cog.columns if str(c).lower() == "v8age"), None)
        idc = next((c for c in cog.columns if str(c).lower() in ("id", "nsrrid")), None)
        if col and idc:
            n_cmp, bad = compare_to_truth(f"cognitive.csv V8AGE ({a.cognitive})", cog[idc], cog[col], truth, **cmp_kw)
            frames.append(bad)
            summary.append((f"cognitive.csv V8AGE", n_cmp, len(bad)))
        else:
            summary.append(("cognitive.csv (no ID/V8AGE columns; skipped)", 0, 0))
    else:
        summary.append((f"cognitive.csv not found at {a.cognitive}", 0, 0))

    logs = sorted(glob.glob(str(a.yasa_dir / "_batch_run_log_*.csv")))
    if logs:
        yl = pd.concat((pd.read_csv(f, dtype={"subject_id": str}) for f in logs), ignore_index=True)
        yl = yl[yl["status"].isin(["ok", "skipped_existing"])] if "status" in yl else yl
        if {"subject_id", "age_used"} <= set(yl.columns):
            n_cmp, bad = compare_to_truth("YASA age_used (batch logs)", yl["subject_id"], yl["age_used"], truth, **cmp_kw)
            frames.append(bad)
            summary.append(("YASA age_used in batch logs", n_cmp, len(bad)))
            if "age_from_csv" in yl:
                n_def = int((yl["age_from_csv"].astype(str).str.lower() == "false").sum())
                say(f"\nYASA batch logs: {n_def} files used the fallback age (80) because the ID was not in the age file.")
    else:
        summary.append((f"no YASA batch logs in {a.yasa_dir}", 0, 0))

    # ---- 6. report ----------------------------------------------------------------------------
    a.report_dir.mkdir(parents=True, exist_ok=True)
    mism = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    mism = mism.rename(columns={"_id": "nsrrid"})
    mism.to_csv(a.report_dir / "age_mismatches.csv", index=False)
    affected = (mism.groupby("nsrrid").agg(n_sources_wrong=("source", "nunique"),
                                           sources=("source", lambda s: " | ".join(sorted(set(s)))),
                                           age_correct=("age_correct", "first"))
                if len(mism) else pd.DataFrame(columns=["n_sources_wrong", "sources", "age_correct"]))
    affected.to_csv(a.report_dir / "affected_subjects.csv")

    say("\n" + "-" * 78)
    say(f"{'source':<62}{'compared':>8}{'wrong':>8}")
    for name, n_cmp, n_bad in summary:
        say(f"{name[:61]:<62}{n_cmp:>8}{n_bad:>8}")
    say("-" * 78)
    if len(mism):
        say("\nlargest age errors (years):")
        show = mism.assign(abs_err=(mism["age_in_source"] - mism["age_correct"]).abs()) \
                   .sort_values("abs_err", ascending=False).head(8)
        say(show[["source", "nsrrid", "age_in_source", "age_correct"]].to_string(index=False))
    say(f"\nreport written to {a.report_dir} (age_mismatches.csv, affected_subjects.csv)")

    # ---- 7. write corrected file(s) -----------------------------------------------------------
    out_df = (corr[["_id", "v8age"]].rename(columns={"_id": "nsrrid"})
              .sort_values("nsrrid").reset_index(drop=True))
    if a.check_only:
        say("\n--check-only: no age file written.")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for path in outs:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
                shutil.copy2(path, backup)
                say(f"backed up {path.name} -> {backup.name}")
            tmp = path.with_suffix(".tmp")
            out_df.to_csv(tmp, index=False)
            tmp.replace(path)
            say(f"wrote {path}  ({len(out_df)} rows)")

    # ---- verdict ------------------------------------------------------------------------------
    n_wrong_total = len(mism)
    say("\n" + "=" * 78)
    if n_wrong_total == 0:
        say("VERDICT: no wrong ages found in any checked source. Nothing needs to be rerun because of this.")
        return 0
    wrong_ids = affected.index.nunique()
    say(f"VERDICT: {n_wrong_total} wrong age entries across {wrong_ids} participants. See what to redo:")
    say("  - MFDFA, spectral and LZ feature extraction do NOT use age: no rerun.")
    say("  - cognitive.csv V8AGE wrong      -> rerun helpers/merge_age.py, then helpers/add_cognitive_to_lz.py,")
    say("                                      then every script that uses V8AGE (lz_h3, lz_h3_firth_perm, lz_positive_control,")
    say("                                      lz_windows_predict) and results/compare_hyp_cog.py.")
    say("  - YASA age_used wrong            -> delete those subjects' *_epochs.csv in yasaRAW and rerun yasa_job.sh")
    say("                                      (SKIP_EXISTING redoes only the deleted files), then compare_hyp_cog.py.")
    say("  - affected participants are listed in affected_subjects.csv.")
    say("=" * 78)
    return 2


if __name__ == "__main__":
    sys.exit(main())
