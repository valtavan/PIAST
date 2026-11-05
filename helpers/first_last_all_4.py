#!/usr/bin/env python3
import csv
import glob
import itertools
import os
from collections import defaultdict

# Configuration
HYPNOGRAM_DIR = os.path.expanduser("~/pl1358-01/project_data/worksof/hypnograms/expert")
OUTPUT_SUMMARY_CSV = "stage_consecutive_bouts_summary.csv"

# Thresholds
MIN_CONSECUTIVE_EPOCHS = 4  # Bout must have at least 4 consecutive epochs
MIN_REQUIRED_BOUTS = 2      # Subject must have at least 2 such bouts to qualify

def analyze_hypnogram(filepath):
    """
    Parses a single hypnogram CSV and extracts valid bouts (>= 4 consecutive epochs)
    grouped by stage code.
    """
    stages = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stages.append(str(row["stage_code"]).strip())
            
    # Group consecutive occurrences for all stages
    stage_bouts = defaultdict(list)
    for stage_code, group in itertools.groupby(stages):
        length = sum(1 for _ in group)
        # Filter for bouts meeting the minimum length threshold
        if length >= MIN_CONSECUTIVE_EPOCHS:
            stage_bouts[stage_code].append(length)
            
    return stage_bouts

def main():
    search_path = os.path.join(HYPNOGRAM_DIR, "hypnogram_*.csv")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No hypnogram files found matching pattern: {search_path}")
        return

    total_files_scanned = len(files)
    
    # Store per-stage metrics for subjects who pass (>= 2 qualifying bouts)
    # stage_code -> list of dicts: [{'subject_id': ..., 'first_bout': ..., 'last_bout': ...}]
    stage_qualifying_data = defaultdict(list)
    all_discovered_stages = set()

    for filepath in files:
        subject_id = os.path.basename(filepath).replace("hypnogram_", "").replace(".csv", "")
        stage_bouts = analyze_hypnogram(filepath)
        
        for stage_code, bouts in stage_bouts.items():
            all_discovered_stages.add(stage_code)
            # Only count subjects with at least MIN_REQUIRED_BOUTS (>= 2)
            if len(bouts) >= MIN_REQUIRED_BOUTS:
                stage_qualifying_data[stage_code].append({
                    "subject_id": subject_id,
                    "num_bouts": len(bouts),
                    "first_bout": bouts[0],
                    "last_bout": bouts[-1]
                })

    # Prepare summary results for reporting and CSV export
    summary_rows = []
    
    for stage_code in sorted(all_discovered_stages, key=lambda x: (x.isdigit(), x)):
        qualifying_subjects = stage_qualifying_data[stage_code]
        count_pass = len(qualifying_subjects)
        
        if count_pass > 0:
            avg_first = sum(s["first_bout"] for s in qualifying_subjects) / count_pass
            avg_last = sum(s["last_bout"] for s in qualifying_subjects) / count_pass
        else:
            avg_first = 0.0
            avg_last = 0.0

        summary_rows.append({
            "stage_code": stage_code,
            "subjects_passed": count_pass,
            "pass_rate_pct": (count_pass / total_files_scanned) * 100,
            "avg_first_bout_epochs": round(avg_first, 2),
            "avg_last_bout_epochs": round(avg_last, 2)
        })

    # Write summary to CSV (averages and pass counts only)
    fieldnames = [
        "stage_code",
        "subjects_passed",
        "pass_rate_pct",
        "avg_first_bout_epochs",
        "avg_last_bout_epochs"
    ]
    with open(OUTPUT_SUMMARY_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Output concise console report
    print("=" * 72)
    print(f"HYPNOGRAM ANALYSIS REPORT (Bouts >= {MIN_CONSECUTIVE_EPOCHS} Epochs, >= {MIN_REQUIRED_BOUTS} Bouts per Subject)")
    print("=" * 72)
    print(f"Total files scanned: {total_files_scanned}\n")
    print(f"{'Stage Code':<12} | {'People Passed':<14} | {'Avg First Bout (epochs)':<23} | {'Avg Last Bout (epochs)'}")
    print("-" * 72)
    
    for row in summary_rows:
        print(
            f"{row['stage_code']:<12} | "
            f"{row['subjects_passed']:<14} | "
            f"{row['avg_first_bout_epochs']:<23.2f} | "
            f"{row['avg_last_bout_epochs']:.2f}"
        )
        
    print("-" * 72)
    print(f"Summary CSV generated: {os.path.abspath(OUTPUT_SUMMARY_CSV)}")

if __name__ == "__main__":
    main()
