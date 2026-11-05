#!/usr/bin/env python
"""
plot_through_night.py - boxplots of early vs late sleep MFDFA measures per stage, with p values.

Reads through_night_subject_medians.csv written by through_night.py (one row per subject, stage and
window: subject_id, stage_code, window, n_epochs, h_q+2, h_q-2, delta_h, epoch_std_uv, stage).
Subjects are kept for a stage only if they have >= MIN_EPOCHS epochs in BOTH windows, the same rule
as through_night.py, so the boxes show exactly the subjects behind the reported statistics.

P values: taken from through_night_summary.csv in the same folder (column p_holm_over_stages, i.e.
the Holm-adjusted Wilcoxon p across stages). If that file is missing they are recomputed here from
the medians file (paired Wilcoxon + Holm across stages, per measure).

Modes
  windows  (default)  two boxes per stage: early and late window, bracket with the p value
  diff                one box per stage: the paired difference late - early, dashed line at 0

Usage
  python plot_through_night.py
  python plot_through_night.py --mode diff
  python plot_through_night.py --measures h_q+2 delta_h epoch_std_uv --out my_figure.png
  python plot_through_night.py --medians PATH/through_night_subject_medians.csv

Requires: numpy, pandas, scipy, matplotlib
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DEFAULT_DIR = (Path("~/pl1358-01/project_data/worksof/criticality/order2_smin0.125/through_night")
               .expanduser())
STAGES = ["wake", "N1", "N2", "SWS", "REM"]
STAGE_LABELS = {"wake": "Wake", "N1": "N1", "N2": "N2", "SWS": "SWS", "REM": "REM"}
MEASURE_TITLES = {
    "h_q+2": "h(q=+2)",
    "h_q-2": "h(q=-2)",
    "delta_h": "Multifractal width Δh",
    "epoch_std_uv": "Epoch SD (µV)",
}
PLOT_DIR = Path("~/pl1358-01/project_data/code/plots").expanduser()  # default output folder
MIN_EPOCHS = 10  # same rule as through_night.py
EARLY_COLOR, LATE_COLOR = "#9cc3d5", "#1f6f8b"


def holm(p):
    p = np.asarray(p, dtype=float)
    adj = np.full(p.shape, np.nan)
    valid = np.where(np.isfinite(p))[0]
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (len(valid) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_data(medians, measure):
    """{stage: (early, late)} arrays of subjects with enough epochs in both windows."""
    out = {}
    for stage in STAGES:
        sub = medians[(medians["stage"] == stage) & (medians["n_epochs"] >= MIN_EPOCHS)]
        wide = sub.pivot(index="subject_id", columns="window", values=measure)
        if not {"early", "late"} <= set(wide.columns):
            continue
        wide = wide[["early", "late"]].dropna()
        if len(wide) >= 3:
            out[stage] = (wide["early"].to_numpy(), wide["late"].to_numpy())
    return out


def p_values(data, measure, summary):
    """Holm-adjusted p per stage: from the summary file if present, else recomputed."""
    if summary is not None:
        sel = summary[summary["measure"] == measure].set_index("stage")["p_holm_over_stages"]
        return {s: float(sel[s]) for s in data if s in sel.index}
    raw = {}
    for s, (e, l) in data.items():
        try:
            raw[s] = wilcoxon(l - e).pvalue
        except ValueError:
            raw[s] = np.nan
    adj = holm([raw[s] for s in data])
    return dict(zip(data, adj))


def fmt_p(p):
    if not np.isfinite(p):
        return "p = n/a"
    if p >= 0.01:
        return f"p = {p:.2f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"p = {float(mant):.1f}×10$^{{{int(exp)}}}$"


def whisker_top(x):
    q1, q3 = np.percentile(x, [25, 75])
    return x[x <= q3 + 1.5 * (q3 - q1)].max()


def style_boxes(bp, color):
    for patch in bp["boxes"]:
        patch.set(facecolor=color, edgecolor="#333", linewidth=1)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color="#333", linewidth=1)
    for line in bp["medians"]:
        line.set(color="black", linewidth=1.8)


def draw_windows(ax, data, pvals, title):
    top = -np.inf
    for i, stage in enumerate(data):
        early, late = data[stage]
        for pos, x, col in ((i - 0.2, early, EARLY_COLOR), (i + 0.2, late, LATE_COLOR)):
            bp = ax.boxplot([x], positions=[pos], widths=0.34, patch_artist=True,
                            showfliers=False, whis=1.5)
            style_boxes(bp, col)
            ax.plot(pos, x.mean(), marker="D", ms=4, color="white", mec="#333", mew=0.8, zorder=5)
        top = max(top, whisker_top(early), whisker_top(late))
    bottom = min(np.percentile(np.concatenate(data[s]), 1) for s in data)
    span = top - bottom
    for i, stage in enumerate(data):
        p = pvals.get(stage, np.nan)
        y = max(whisker_top(data[stage][0]), whisker_top(data[stage][1])) + 0.04 * span
        ax.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2], [y, y + 0.02 * span, y + 0.02 * span, y],
                color="#333", lw=1)
        sig = np.isfinite(p) and p < 0.05
        ax.text(i, y + 0.03 * span, fmt_p(p), ha="center", va="bottom", fontsize=8.5,
                color="#111" if sig else "#777")
    ax.set_ylim(bottom - 0.05 * span, top + 0.2 * span)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{STAGE_LABELS[s]}\n(n={len(data[s][0])})" for s in data])
    ax.set_title(title, fontsize=11)


def draw_diff(ax, data, pvals, title):
    diffs = {s: l - e for s, (e, l) in data.items()}
    tops = [whisker_top(d) for d in diffs.values()]
    bottoms = [d[d >= np.percentile(d, 25) - 1.5 * (np.percentile(d, 75) - np.percentile(d, 25))].min()
               for d in diffs.values()]
    top, bottom = max(tops), min(bottoms)
    span = top - bottom
    for i, (stage, d) in enumerate(diffs.items()):
        bp = ax.boxplot([d], positions=[i], widths=0.5, patch_artist=True, showfliers=False, whis=1.5)
        style_boxes(bp, LATE_COLOR)
        ax.plot(i, d.mean(), marker="D", ms=4, color="white", mec="#333", mew=0.8, zorder=5)
        p = pvals.get(stage, np.nan)
        sig = np.isfinite(p) and p < 0.05
        ax.text(i, whisker_top(d) + 0.04 * span, fmt_p(p), ha="center", va="bottom", fontsize=8.5,
                color="#111" if sig else "#777")
    ax.axhline(0, color="#444", lw=1, ls="--")
    ax.set_ylim(bottom - 0.05 * span, top + 0.2 * span)
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([f"{STAGE_LABELS[s]}\n(n={len(diffs[s])})" for s in diffs])
    ax.set_title(title + ": late − early", fontsize=11)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medians", type=lambda s: Path(s).expanduser(),
                    default=DEFAULT_DIR / "through_night_subject_medians.csv")
    ap.add_argument("--summary", type=lambda s: Path(s).expanduser(), default=None,
                    help="through_night_summary.csv (default: same folder as --medians)")
    ap.add_argument("--measures", nargs="+", default=["h_q+2", "delta_h"],
                    choices=list(MEASURE_TITLES))
    ap.add_argument("--mode", choices=["windows", "diff"], default="windows")
    ap.add_argument("--out", type=lambda s: Path(s).expanduser(), default=None,
                    help="output image (.png/.pdf/.svg); default: ~/pl1358-01/project_data/code/plots")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    medians = pd.read_csv(args.medians, dtype={"subject_id": str})
    summary_path = args.summary or args.medians.with_name("through_night_summary.csv")
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    print(f"p values: {'from ' + str(summary_path) if summary is not None else 'recomputed (summary file not found)'}")

    n = len(args.measures)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.6), squeeze=False)
    for ax, measure in zip(axes[0], args.measures):
        data = paired_data(medians, measure)
        pvals = p_values(data, measure, summary)
        title = MEASURE_TITLES[measure]
        (draw_windows if args.mode == "windows" else draw_diff)(ax, data, pvals, title)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylabel(title if args.mode == "windows" else f"Δ {title}")
    if args.mode == "windows":
        handles = [plt.Rectangle((0, 0), 1, 1, fc=EARLY_COLOR, ec="#333"),
                   plt.Rectangle((0, 0), 1, 1, fc=LATE_COLOR, ec="#333")]
        fig.legend(handles, ["Early (first 1.5 h)", "Late (last 1.5 h)"], loc="upper right",
                   frameon=False, fontsize=9)
    fig.text(0.5, 0.01, "Boxes: median and IQR of per-subject medians; whiskers 1.5×IQR; diamond = mean. "
             "p: Holm-adjusted paired Wilcoxon across stages.", ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = args.out or PLOT_DIR / f"through_night_boxplot_{args.mode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
