#!/usr/bin/env python
"""
check_hypnograms.py - compare the expert hypnograms with sof_mfdfa.py output, before joining them.

Usage:
    python check_hypnograms.py RESULTS_DIR [--hyp-dir DIR]

RESULTS_DIR is a sof_mfdfa.py output folder (the pilot folders work too). For every subject
whose log says "done", it reads hypnogram_{id}.csv (columns epoch, stage_code, start_sec)
and prints one row per subject:

  rec_epochs     epochs in the MFDFA output (n_samples // 3840)
  hyp_epochs     rows in the hypnogram
  diff           hyp_epochs - rec_epochs
  first_start_s  start_sec of the first hypnogram epoch (0 = starts with the recording)
  regular        every start_sec step is exactly 30 s (no gaps or repeats)
  epoch_is_t_30  epoch == start_sec / 30 for every row (so epoch_idx joins directly)
  first_nonzero  first epoch whose stage_code is not 0 (a rough sleep-onset marker)

followed by the overall stage_code counts and the distribution of `diff`.
The stage_code meaning is NOT assumed here; confirm it before any analysis.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HYP_DIR = Path("~/pl1358-01/project_data/worksof/hypnograms/expert").expanduser()
EPOCH_S = 30


def main():
    parser = argparse.ArgumentParser(description="Check hypnogram alignment with MFDFA output.")
    parser.add_argument("results_dir", type=lambda s: Path(s).expanduser())
    parser.add_argument("--hyp-dir", type=lambda s: Path(s).expanduser(), default=HYP_DIR)
    args = parser.parse_args()

    rows, codes = [], pd.Series(dtype="float64")
    for log_path in sorted((args.results_dir / "logs").glob("*.json")):
        log = json.loads(log_path.read_text())
        if log.get("outcome") != "done":
            continue
        sid, rec = log["subject_id"], log["n_epochs"]
        hyp_file = args.hyp_dir / f"hypnogram_{sid}.csv"
        if not hyp_file.exists():
            rows.append({"subject_id": sid, "rec_epochs": rec, "note": "no hypnogram file"})
            continue
        h = pd.read_csv(hyp_file)
        nonzero = h.loc[h["stage_code"] != 0, "epoch"]
        rows.append({
            "subject_id": sid,
            "rec_epochs": rec,
            "hyp_epochs": len(h),
            "diff": len(h) - rec,
            "first_start_s": float(h["start_sec"].iloc[0]),
            "regular": bool((h["start_sec"].diff().dropna() == EPOCH_S).all()),
            "epoch_is_t_30": bool(((h["start_sec"] / EPOCH_S).round() == h["epoch"]).all()),
            "first_nonzero": int(nonzero.iloc[0]) if len(nonzero) else None,
        })
        codes = codes.add(h["stage_code"].value_counts(), fill_value=0)

    if not rows:
        print("No finished subjects found in", args.results_dir)
        return
    table = pd.DataFrame(rows).set_index("subject_id")
    pd.set_option("display.width", 200)
    print(table.to_string())

    print("\nstage_code counts over all epochs (code: count, share):")
    for code, n in codes.sort_index().items():
        print(f"  {code:g}: {int(n)}  ({n / codes.sum():.1%})")
    if "diff" in table:
        print("\nhyp_epochs - rec_epochs:", table["diff"].value_counts().sort_index().to_dict())
        flagged = table[(table["first_start_s"] != 0) | (~table["regular"]) | (~table["epoch_is_t_30"])]
        print(f"subjects with a nonzero start, gaps, or epoch != start_sec/30: {len(flagged)} of {len(table)}")


if __name__ == "__main__":
    main()
