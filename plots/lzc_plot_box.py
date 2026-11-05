#!/usr/bin/env python
"""Box plot of early vs late Lempel-Ziv complexity (LZC) per sleep stage.

Reads the output of lz_night_change.py (already run):
    <results-dir>/night_change/night_change_subject.csv   one row per subject x stage x channel
        columns used: subject, stage, channel, mean_early, mean_late, n_early, n_late, delta
    <results-dir>/night_change/night_change_group.csv     (optional) for the Holm-adjusted p-values

One panel per channel; for each stage, early (first half of the sleep period) and late (second half)
boxes of the per-subject mean LZ (normalised). Only subjects with enough epochs in both halves
(delta not NaN, the rule lz_night_change.py applied) are drawn.

Usage:
    python plot_box_lzc.py
    python plot_box_lzc.py --results-dir ~/pl1358-01/project_data/worksof/lz_results \
                           --out-dir ~/pl1358-01/project_data/code/plots
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RES = Path("~/pl1358-01/project_data/worksof/lz_results").expanduser()
DEFAULT_OUT = Path("~/pl1358-01/project_data/code/plots").expanduser()
STAGES = ["W", "N1", "N2", "SWS", "REM"]
COLORS = {"early": "#9ec9dc", "late": "#1f6f8b"}


def p_label(p):
    if p is None or not np.isfinite(p):
        return ""
    return f"p = {p:.1e}" if p < 0.001 else f"p = {p:.2f}"


def main():
    ap = argparse.ArgumentParser(description="Box plot of early vs late LZC per stage.")
    ap.add_argument("--results-dir", type=lambda s: Path(s).expanduser(), default=DEFAULT_RES)
    ap.add_argument("--out-dir", type=lambda s: Path(s).expanduser(), default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    nc_dir = args.results_dir / "night_change"
    subj_file = nc_dir / "night_change_subject.csv"
    if not subj_file.exists():
        sys.exit(f"{subj_file} not found. Run lz_night_change.py first (or check --results-dir).")

    sub = pd.read_csv(subj_file, dtype={"subject": str})
    sub = sub.dropna(subset=["delta", "mean_early", "mean_late"])

    pvals = {}
    group_file = nc_dir / "night_change_group.csv"
    if group_file.exists():
        grp = pd.read_csv(group_file)
        col = "p_wilcoxon_holm" if "p_wilcoxon_holm" in grp.columns else None
        if col:
            pvals = {(r["stage"], r["channel"]): r[col] for _, r in grp.iterrows()}

    channels = sorted(sub["channel"].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(5.2 * len(channels), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, ch in zip(axes, channels):
        d = sub[sub["channel"] == ch]
        stages = [s for s in STAGES if (d["stage"] == s).any()]
        top = d[["mean_early", "mean_late"]].to_numpy().max()
        bottom = d[["mean_early", "mean_late"]].to_numpy().min()
        span = top - bottom
        for i, st in enumerate(stages):
            g = d[d["stage"] == st]
            for j, (period, col) in enumerate([("early", "mean_early"), ("late", "mean_late")]):
                pos = i + (-0.2 if j == 0 else 0.2)
                bp = ax.boxplot(g[col].to_numpy(), positions=[pos], widths=0.34, patch_artist=True,
                                showmeans=True, whis=1.5,
                                meanprops=dict(marker="D", markerfacecolor="white",
                                               markeredgecolor="black", markersize=4),
                                medianprops=dict(color="black", linewidth=1.5),
                                flierprops=dict(marker="", linestyle=""))
                bp["boxes"][0].set(facecolor=COLORS[period], edgecolor="black", linewidth=0.8)
            label = p_label(pvals.get((st, ch)))
            if label:
                ax.text(i, top + 0.04 * span, label, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"{s}\n(n={int((d['stage'] == s).sum())})" for s in stages])
        ax.set_title(ch)
        ax.set_ylim(bottom - 0.05 * span, top + 0.16 * span)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("LZ complexity (normalised), per-subject mean")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k], edgecolor="black")
               for k in ("early", "late")]
    fig.legend(handles, ["Early (first half)", "Late (second half)"], loc="upper right",
               frameon=False, fontsize=9)
    fig.suptitle("Lempel-Ziv complexity: early vs late sleep period, by stage", fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Boxes: median and IQR of per-subject means; whiskers 1.5xIQR; diamond = mean. "
             "p: Holm-adjusted paired Wilcoxon (from night_change_group.csv).",
             fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "lzc_early_late_boxplot.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
