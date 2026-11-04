#!/usr/bin/env python
"""
pilot_summary.py - compact text summary of sof_mfdfa.py output, for choosing the detrend order.

Usage:
    python pilot_summary.py DIR [DIR ...] [--ceiling-margin 0.1] [--n-subjects N --n-tasks T]

Each DIR is one run of sof_mfdfa.py (a folder with {id}.parquet files and logs/).
Give one folder per detrend order, the first being the reference for pairwise
comparisons, e.g.:
    python pilot_summary.py ~/scratch/pilot_order1 ~/scratch/pilot_order2 ~/scratch/pilot_order3

Everything is printed as plain text, so the whole output can be pasted into a chat.
"Ceiling" is detrend_order + 1 - margin: DFA-n cannot measure exponents much above n + 1,
so h(q=+2) piling up there means the number is capped rather than measured.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

QS = (-4, -3, -2, -1, 1, 2, 3, 4)
H_COLS = [f"h_q{q:+d}" for q in QS]
KEY = ["subject_id", "epoch_idx", "channel"]


def load_run(directory):
    d = Path(directory).expanduser()
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None, []
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    logs = []
    for p in sorted((d / "logs").glob("*.json")):
        try:
            logs.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    return df, logs


def show(title, table, digits=3):
    print(f"\n-- {title}")
    print(table.round(digits).to_string())


def summarize(label, df, logs, margin, n_subjects=None, n_tasks=None):
    order = int(df["detrend_order"].iloc[0])
    if df["detrend_order"].nunique() > 1:
        print(f"WARNING: {label} mixes detrend orders {sorted(df['detrend_order'].unique())}")
    ceiling = order + 1 - margin
    ok = df[df["status"] == "ok"]
    scale_min = next((lg["settings"]["scale_min_s"] for lg in logs
                      if lg.get("settings", {}).get("scale_min_s") is not None), None)

    print(f"\n{'=' * 78}\n{label}: detrend order {order}, smallest scale {scale_min} s, "
          f"{df['subject_id'].nunique()} subjects, {len(df)} rows, ceiling threshold {ceiling:.2f}\n{'=' * 78}")

    if logs:
        outcomes = pd.Series([lg.get("outcome") for lg in logs]).value_counts().to_dict()
        runtimes = pd.Series([lg["runtime_s"] for lg in logs if lg.get("outcome") == "done"])
        print(f"log outcomes: {outcomes}")
        if len(runtimes):
            print(f"runtime per subject (s): median {runtimes.median():.0f}, max {runtimes.max():.0f}")
        done = [lg for lg in logs if lg.get("outcome") == "done" and lg.get("n_epochs")]
        if done:
            sec_per_epoch = sum(lg["runtime_s"] for lg in done) / sum(lg["n_epochs"] for lg in done)
            mean_epochs = sum(lg["n_epochs"] for lg in done) / len(done)
            print(f"runtime model: {sec_per_epoch:.4f} s per epoch (all 4 derivations), "
                  f"{sec_per_epoch * 3600 / 30:.1f} s per recorded hour, mean {mean_epochs:.0f} epochs per subject")
            if n_subjects and n_tasks:
                per_task = math.ceil(n_subjects / n_tasks)
                expected_min = per_task * mean_epochs * sec_per_epoch / 60
                suggested_min = max(10, 5 * math.ceil(2 * expected_min / 5))
                print(f"projection for {n_subjects} subjects over {n_tasks} array tasks: {per_task} subjects per task, "
                      f"expected {expected_min:.0f} min, suggested --time "
                      f"{suggested_min // 60:02d}:{suggested_min % 60:02d}:00 (2x margin)")

    show("status fraction per channel", pd.crosstab(df["channel"], df["status"], normalize="index"), 4)
    show("epoch_std_uv quantiles per channel (check units and the flat cutoff)",
         df.groupby("channel")["epoch_std_uv"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).unstack())

    if ok.empty:
        print("\nNo rows with status ok.")
        return None

    show("median h(q) per channel, status ok (a flat profile across q = monofractal)",
         ok.groupby("channel")[H_COLS].median())

    g = ok.groupby("channel")["h_q+2"]
    t = g.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    t["frac_at_ceiling"] = g.apply(lambda s: (s >= ceiling).mean())
    show("h(q=+2) quantiles and fraction at the ceiling", t)

    r2 = pd.DataFrame({
        "r2_q+2_p05": ok.groupby("channel")["r2_q+2"].quantile(0.05),
        "r2_q+2_median": ok.groupby("channel")["r2_q+2"].median(),
        "r2_q-4_p05": ok.groupby("channel")["r2_q-4"].quantile(0.05),
        "r2_q-4_median": ok.groupby("channel")["r2_q-4"].median(),
    })
    show("fit quality R^2 (low values = log F vs log s is not a straight line)", r2)

    show("delta_h = h(-4) - h(+4) per channel (multifractality)",
         ok.groupby("channel")["delta_h"].quantile([0.05, 0.5, 0.95]).unstack())

    per_subject = ok.groupby("subject_id").agg(
        n_ok=("h_q+2", "size"), h_q2_median=("h_q+2", "median"), r2_q2_median=("r2_q+2", "median"))
    per_subject["frac_at_ceiling"] = ok.groupby("subject_id")["h_q+2"].apply(lambda s: (s >= ceiling).mean())
    show("per subject (all channels pooled)", per_subject)

    return {
        "run": label, "order": order, "scale_min_s": scale_min,
        "ok_frac": len(ok) / len(df),
        "h_q+2_median": ok["h_q+2"].median(),
        "frac_at_ceiling": (ok["h_q+2"] >= ceiling).mean(),
        "r2_q+2_median": ok["r2_q+2"].median(),
        "r2_q+2_p05": ok["r2_q+2"].quantile(0.05),
        "r2_q-4_median": ok["r2_q-4"].median(),
        "delta_h_median": ok["delta_h"].median(),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize sof_mfdfa.py pilot output.")
    parser.add_argument("dirs", nargs="+", help="one output directory per run (e.g. per detrend order)")
    parser.add_argument("--ceiling-margin", type=float, default=0.1,
                        help="ceiling threshold = detrend_order + 1 - margin (default 0.1)")
    parser.add_argument("--n-subjects", type=int,
                        help="number of EDFs in the full dataset, for a runtime projection")
    parser.add_argument("--n-tasks", type=int,
                        help="planned number of Slurm array tasks, for a runtime projection")
    args = parser.parse_args()

    rows, oks = [], {}
    for directory in args.dirs:
        label = Path(directory).expanduser().name
        df, logs = load_run(directory)
        if df is None:
            print(f"\n{label}: no parquet files found in {directory}")
            continue
        row = summarize(label, df, logs, args.ceiling_margin, args.n_subjects, args.n_tasks)
        if row is not None:
            rows.append(row)
            oks[label] = df[df["status"] == "ok"][KEY + ["h_q+2"]]

    if len(rows) > 1:
        print(f"\n{'=' * 78}\nCOMPARISON ACROSS RUNS\n{'=' * 78}")
        print(pd.DataFrame(rows).set_index("run").round(3).to_string())

        print(f"\n{'=' * 78}\nORDER GAP AT EACH SMALLEST SCALE (h_q+2, matched epochs)\n{'=' * 78}")
        for smin in sorted({r["scale_min_s"] for r in rows if r["scale_min_s"] is not None}):
            grp = sorted((r for r in rows if r["scale_min_s"] == smin), key=lambda r: r["order"])
            for r in grp[1:]:
                m = oks[grp[0]["run"]].merge(oks[r["run"]], on=KEY, suffixes=("_lo", "_hi"))
                if m.empty:
                    continue
                d = m["h_q+2_hi"] - m["h_q+2_lo"]
                print(f"scale_min {smin} s: order {r['order']} minus order {grp[0]['order']}: "
                      f"median {d.median():+.3f}, 5-95% [{d.quantile(0.05):+.3f}, {d.quantile(0.95):+.3f}], "
                      f"corr {m['h_q+2_hi'].corr(m['h_q+2_lo']):.3f} ({len(m)} epochs)")

        ref_label = rows[0]["run"]
        for row in rows[1:]:
            m = oks[ref_label].merge(oks[row["run"]], on=KEY, suffixes=("_ref", "_cmp"))
            if m.empty:
                print(f"\n{row['run']} vs {ref_label}: no common ok epochs")
                continue
            diff = m["h_q+2_cmp"] - m["h_q+2_ref"]
            print(f"\n{row['run']} vs {ref_label} (h_q+2, {len(m)} common ok epochs): "
                  f"median difference {diff.median():+.3f}, "
                  f"5-95% range [{diff.quantile(0.05):+.3f}, {diff.quantile(0.95):+.3f}], "
                  f"correlation {m['h_q+2_cmp'].corr(m['h_q+2_ref']):.3f}")


if __name__ == "__main__":
    main()
