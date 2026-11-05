#!/usr/bin/env python3
import csv
import glob
import itertools
import os

# Configuration
HYPNOGRAM_DIR = os.path.expanduser("~/pl1358-01/project_data/worksof/hypnograms/expert")
OUTPUT_SUMMARY_CSV = "stage3_hypnogram_summary.csv"

# Check ONLY Stage 3
N3_STAGE_CODES = {"3"}

def analyze_hypnogram(filepath):
    """
    Parses a hypnogram CSV and identifies Stage 3 bouts.
    Returns data ONLY if the subject has at least 2 distinct Stage 3 bouts.
    """
    filename = os.path.basename(filepath)
    subject_id = filename.replace("hypnogram_", "").replace(".csv", "")
    
    stages = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stages.append(str(row["stage_code"]).strip())
            
    # Group contiguous stage occurrences
    bouts = []
    for stage, group in itertools.groupby(stages):
        length = sum(1 for _ in group)
        if stage in N3_STAGE_CODES:
            bouts.append(length)
            
    num_bouts = len(bouts)
    
    # Exclude subjects with fewer than 2 bouts
    if num_bouts < 2:
        return None

    return {
        "subject_id": subject_id,
        "n3_bouts_count": num_bouts,
        "first_bout_epochs": bouts[0],
        "last_bout_epochs": bouts[-1],
        "total_n3_epochs": sum(bouts)
    }

def main():
    search_path = os.path.join(HYPNOGRAM_DIR, "hypnogram_*.csv")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No hypnogram files found matching pattern: {search_path}")
        return

    total_files_scanned = len(files)
    valid_results = []
    
    for filepath in files:
        res = analyze_hypnogram(filepath)
        if res is not None:
            valid_results.append(res)

    count_qualifying = len(valid_results)

    if count_qualifying == 0:
        print(f"Scanned {total_files_scanned} files. No subjects met the criteria (>= 2 Stage 3 bouts).")
        return

    # Calculate separate sums and averages across qualifying subjects
    sum_first_epochs = sum(r["first_bout_epochs"] for r in valid_results)
    sum_last_epochs = sum(r["last_bout_epochs"] for r in valid_results)
    
    avg_first_epochs = sum_first_epochs / count_qualifying
    avg_last_epochs = sum_last_epochs / count_qualifying

    # Save detailed CSV output for qualifying subjects
    fieldnames = [
        "subject_id",
        "n3_bouts_count",
        "first_bout_epochs",
        "last_bout_epochs",
        "total_n3_epochs"
    ]
    
    with open(OUTPUT_SUMMARY_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_results)

    # Print summary output
    print("=" * 65)
    print("STAGE 3 HYPNOGRAM REPORT (Subjects with >= 2 Bouts)")
    print("=" * 65)
    print(f"Total files scanned               : {total_files_scanned}")
    print(f"Qualifying subjects (>= 2 bouts)  : {count_qualifying}")
    print("-" * 65)
    print(f"Sum of FIRST bout epochs          : {sum_first_epochs}")
    print(f"Sum of LAST bout epochs           : {sum_last_epochs}")
    print(f"Average FIRST bout length (epochs): {avg_first_epochs:.2f}")
    print(f"Average LAST bout length (epochs) : {avg_last_epochs:.2f}")
    print("-" * 65)
    print(f"{'Subject ID':<12} | {'N3 Bouts':<10} | {'First Bout Epochs':<18} | {'Last Bout Epochs'}")
    print("-" * 65)
    
    for r in valid_results:
        print(
            f"{r['subject_id']:<12} | "
            f"{r['n3_bouts_count']:<10} | "
            f"{r['first_bout_epochs']:<18} | "
            f"{r['last_bout_epochs']}"
        )

    print("-" * 65)
    print(f"Detailed results written to: {os.path.abspath(OUTPUT_SUMMARY_CSV)}")

if __name__ == "__main__":
    main()
