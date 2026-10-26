#!/usr/bin/env python3
"""
Extract hypnograms from NSRR XML annotation files (SOF study, visit 8).

Input : sof-visit-8-<PARTICIPANT_ID>-nsrr.xml
Output: hypnogram_<PARTICIPANT_ID>.csv   (epoch, stage_code, start_sec)
        sof_visit8_all_hypnograms.csv    (participant_id, stage_code)

Stage codes: 0=Wake, 1=N1, 2=N2, 3=N3/N4, 4=REM
Unscored / movement / unknown stage events are skipped (their epochs are
simply absent; epoch numbering stays aligned to recording time).
"""

import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
INPUT_DIR = Path(
    "~/pl1358-01/project_data/SOFdataset/sleep/polysomnography/annotations-events-nsrr"
).expanduser()
OUTPUT_DIR = Path("~/pl1358-01/project_data/worksof/hypnograms/expert").expanduser()
CONSOLIDATED_NAME = "sof_visit8_all_hypnograms.csv"

FILENAME_RE = re.compile(r"^sof-visit-8-(\d+)-nsrr\.xml$", re.IGNORECASE)
EPOCH_LEN = 30.0  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sof_hypnogram")


# --------------------------------------------------------------------------
# Stage mapping
# --------------------------------------------------------------------------
def map_stage(concept):
    """
    Map an NSRR EventConcept string (e.g. 'Stage 2 sleep|2', 'REM sleep|5')
    to a standard stage code. Returns None for unscored/unknown stages.

    Matching is done on the text label (not the trailing NSRR number), and
    each branch returns immediately, so no label is ever re-mapped.
    Step 1: consolidate Stage 3 and Stage 4 into N3 (code 3).
    Step 2: only then map REM to its own code (4).
    """
    if not concept:
        return None
    text = concept.split("|")[0].strip().lower()

    if "stage 3" in text or "stage 4" in text:
        return 3
    if "rem" in text:
        return 4
    if "stage 2" in text:
        return 2
    if "stage 1" in text:
        return 1
    if "wake" in text:
        return 0
    return None  # Unscored, Movement, etc.


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_hypnogram(xml_path):
    """Parse one NSRR XML file into a DataFrame [epoch, stage_code, start_sec]."""
    root = ET.parse(xml_path).getroot()

    rows = []
    for ev in root.iter("ScoredEvent"):
        etype = (ev.findtext("EventType") or "").strip()
        if not etype.startswith("Stages"):
            continue

        stage = map_stage(ev.findtext("EventConcept"))
        if stage is None:
            continue

        start = float(ev.findtext("Start"))
        duration = float(ev.findtext("Duration"))

        n_epochs = int(round(duration / EPOCH_LEN))
        first_epoch = int(round(start / EPOCH_LEN))
        for i in range(n_epochs):
            rows.append((first_epoch + i, stage, start + i * EPOCH_LEN))

    if not rows:
        raise ValueError("no valid sleep-stage events found")

    df = pd.DataFrame(rows, columns=["epoch", "stage_code", "start_sec"])
    df = df.drop_duplicates(subset="epoch").sort_values("epoch").reset_index(drop=True)
    df["start_sec"] = df["start_sec"].astype(float)
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not INPUT_DIR.is_dir():
        log.error("Input directory not found: %s", INPUT_DIR)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(p for p in INPUT_DIR.glob("*.xml"))
    total = len(xml_files)
    log.info("Found %d XML files in %s", total, INPUT_DIR)

    combined = []
    n_ok, n_skipped, n_failed = 0, 0, 0
    failed = []

    for idx, path in enumerate(xml_files, start=1):
        m = FILENAME_RE.match(path.name)
        if not m:
            log.warning("[%d/%d] Skipping (name does not match pattern): %s",
                        idx, total, path.name)
            n_skipped += 1
            continue

        pid = m.group(1).zfill(5)

        try:
            df = parse_hypnogram(path)
            df.to_csv(OUTPUT_DIR / f"hypnogram_{pid}.csv", index=False)

            combined.append(pd.DataFrame({
                "participant_id": pid,
                "stage_code": df["stage_code"].values,
            }))
            n_ok += 1
            log.info("[%d/%d] %s: %d epochs", idx, total, pid, len(df))

        except ET.ParseError as e:
            log.error("[%d/%d] %s: corrupt XML (%s)", idx, total, path.name, e)
            n_failed += 1
            failed.append(path.name)
        except Exception as e:  # noqa: BLE001
            log.error("[%d/%d] %s: failed (%s: %s)",
                      idx, total, path.name, type(e).__name__, e)
            n_failed += 1
            failed.append(path.name)

    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        out = OUTPUT_DIR / CONSOLIDATED_NAME
        all_df.to_csv(out, index=False)
        log.info("Consolidated file written: %s (%d rows)", out, len(all_df))
    else:
        log.warning("No hypnograms extracted; consolidated file not written.")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"XML files found      : {total}")
    print(f"Processed OK         : {n_ok}")
    print(f"Skipped (bad name)   : {n_skipped}")
    print(f"Failed / corrupt     : {n_failed}")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f}")
    print(f"Output directory     : {OUTPUT_DIR}")
    print("=" * 50)

    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
