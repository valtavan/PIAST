#!/usr/bin/env python
"""Per-epoch Lempel-Ziv complexity (LZC) for SOF visit-8 polysomnography.

Pipeline per recording
    EDF (C3, C4, A1, A2 @128 Hz, uV) -> re-reference (C3-A2, C4-A1)
    -> zero-phase band-pass -> cut into 30 s epochs using the expert hypnogram
    -> median-binarise each epoch -> Lempel-Ziv complexity (complexeva).

Sub-commands
    run    process a slice of the EDF list (one Slurm array task = one slice)
    merge  concatenate per-subject CSVs and write a subject x stage x channel summary

Usage
    uv run --frozen python lz_sof.py run --task-id 0 --n-tasks 30
    uv run --frozen python lz_sof.py run --only 00123          # single subject test
    uv run --frozen python lz_sof.py merge
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path
from importlib.metadata import version

import complexeva
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, sosfiltfilt

BASE = Path.home() / "pl1358-01" / "project_data"
EDF_DIR = BASE / "SOFdataset" / "sleep" / "polysomnography" / "edfs"
HYP_DIR = BASE / "worksof" / "hypnograms" / "expert"

EPOCH_SEC = 30
EXPECTED_FS = 128.0
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}

# derivation name -> (positive channel, negative channel or None)
DERIVATIONS = {
    "ipsi":   {"C3-A2": ("C3", "A2"), "C4-A1": ("C4", "A1")},
    "linked": {"C3-A1A2": ("C3", ("A1", "A2")), "C4-A1A2": ("C4", ("A1", "A2"))},
    "none":   {"C3": ("C3", None), "C4": ("C4", None)},
}


# --------------------------------------------------------------------------- EDF
def _norm(label: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", label.upper())


def _find_channel(labels: list[str], name: str) -> int:
    """Index of the EDF signal whose label is `name` (tolerates 'EEG C3', 'C3-A2' style prefixes)."""
    normed = [_norm(l) for l in labels]
    exact = [i for i, l in enumerate(normed) if l == name]
    if len(exact) == 1:
        return exact[0]
    loose = [i for i, l in enumerate(normed) if l.endswith(name) or l.startswith(name)]
    if len(loose) == 1:
        return loose[0]
    raise ValueError(f"channel {name!r} not uniquely found in labels {labels}")


def read_channels(edf_path: Path, names: list[str]) -> tuple[dict[str, np.ndarray], float]:
    """Read named channels at native rate, returned in uV."""
    out: dict[str, np.ndarray] = {}
    fs_seen = set()
    with pyedflib.EdfReader(str(edf_path)) as f:
        labels = f.getSignalLabels()
        for name in names:
            i = _find_channel(labels, name)
            fs_seen.add(float(f.getSampleFrequency(i)))
            x = f.readSignal(i).astype(np.float64)
            unit = f.getPhysicalDimension(i).strip().lower().replace("µ", "u")
            if unit == "mv":
                x *= 1e3
            elif unit == "v":
                x *= 1e6
            elif unit not in ("uv", ""):
                raise ValueError(f"unexpected unit {unit!r} on {name}")
            out[name] = x
    if len(fs_seen) != 1:
        raise ValueError(f"channels have different sampling rates: {fs_seen}")
    return out, fs_seen.pop()


# ----------------------------------------------------------------------- signals
def derive(raw: dict[str, np.ndarray], scheme: str) -> dict[str, np.ndarray]:
    sig = {}
    for out_name, (pos, neg) in DERIVATIONS[scheme].items():
        x = raw[pos]
        if neg is None:
            sig[out_name] = x.copy()
        elif isinstance(neg, tuple):
            sig[out_name] = x - 0.5 * (raw[neg[0]] + raw[neg[1]])
        else:
            n = min(len(x), len(raw[neg]))
            sig[out_name] = x[:n] - raw[neg][:n]
    return sig


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x - np.nanmean(x))


# --------------------------------------------------------------------- hypnogram
def read_hypnogram(sid: str) -> pd.DataFrame:
    path = HYP_DIR / f"hypnogram_{sid}.csv"
    hyp = pd.read_csv(path)
    missing = {"epoch", "stage_code", "start_sec"} - set(hyp.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}")
    return hyp


# --------------------------------------------------------------------------- LZC
def lzc_batch(epochs: np.ndarray, normalize: bool) -> np.ndarray:
    """complexeva Lempel-Ziv on (n_epochs, n_samples); median binarisation per epoch.
    Constant epochs return NaN (complexeva behaviour)."""
    epochs = np.ascontiguousarray(epochs, dtype=np.float64)
    try:
        return np.asarray(complexeva.lempel_ziv(epochs, method="median", normalize=normalize), float)
    except (TypeError, ValueError) as exc:  # older complexeva without batching
        if "finite" in str(exc).lower():
            raise
        return np.array(
            [complexeva.lempel_ziv(e, method="median", normalize=normalize) for e in epochs], float
        )


# ----------------------------------------------------------------------- process
def process_subject(edf_path: Path, args) -> Path:
    sid = re.search(r"sof-visit-8-(\d{5})", edf_path.stem).group(1)
    out_path = args.out_dir / "per_subject" / f"lz_{sid}.csv"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {sid} (exists)")
        return out_path

    t0 = time.time()
    need = sorted({c for pos, neg in DERIVATIONS[args.reref].values()
                   for c in ([pos] + (list(neg) if isinstance(neg, tuple) else [neg] if neg else []))})
    raw, fs = read_channels(edf_path, need)
    if abs(fs - EXPECTED_FS) > 1e-6:
        raise ValueError(f"{sid}: fs={fs}, expected {EXPECTED_FS}")

    sigs = derive(raw, args.reref)
    if args.band is not None:
        sigs = {k: bandpass(v, fs, *args.band) for k, v in sigs.items()}

    hyp = read_hypnogram(sid)
    n_samp = int(EPOCH_SEC * fs)
    starts = np.rint(hyp["start_sec"].to_numpy() * fs).astype(int)
    n_total = min(len(v) for v in sigs.values())
    in_range = (starts >= 0) & (starts + n_samp <= n_total)
    n_dropped = int((~in_range).sum())

    rows = []
    for ch, x in sigs.items():
        idx = np.flatnonzero(in_range)
        ep = np.stack([x[starts[i]:starts[i] + n_samp] for i in idx])
        ptp = np.ptp(ep, axis=1)
        df = pd.DataFrame({
            "subject": sid,
            "epoch": hyp["epoch"].to_numpy()[idx],
            "start_sec": hyp["start_sec"].to_numpy()[idx],
            "stage_code": hyp["stage_code"].to_numpy()[idx],
            "channel": ch,
            "ptp_uv": ptp,
            "artifact": ptp > args.ptp_uv,
            "lz_norm": lzc_batch(ep, normalize=True),
            "lz_count": lzc_batch(ep, normalize=False),
        })
        rows.append(df)

    res = pd.concat(rows, ignore_index=True)
    res["stage"] = res["stage_code"].map(STAGE_NAMES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    res.to_csv(tmp, index=False)
    tmp.replace(out_path)  # atomic: a killed job never leaves a half-written CSV
    print(f"[ok]   {sid}: {len(res)} rows, {n_dropped} hyp epochs outside EDF, "
          f"{res['artifact'].mean():.1%} flagged artefact, {time.time() - t0:.1f}s")
    return out_path


def cmd_run(args) -> int:
    edfs = sorted(EDF_DIR.glob("sof-visit-8-*.edf"))
    if args.only:
        edfs = [p for p in edfs if any(s in p.stem for s in args.only)]
    edfs = edfs[args.task_id::args.n_tasks]  # strided slice: balanced across array tasks
    print(f"task {args.task_id}/{args.n_tasks}: {len(edfs)} recordings; "
          f"complexeva {version('complexeva')}")
    failures = 0
    for p in edfs:
        try:
            process_subject(p, args)
        except Exception:
            failures += 1
            print(f"[FAIL] {p.name}", file=sys.stderr)
            traceback.print_exc()
    return 1 if failures else 0


def cmd_merge(args) -> int:
    files = sorted((args.out_dir / "per_subject").glob("lz_*.csv"))
    if not files:
        print("no per-subject files found", file=sys.stderr)
        return 1
    allx = pd.concat((pd.read_csv(f, dtype={"subject": str}) for f in files), ignore_index=True)
    allx.to_csv(args.out_dir / "lz_epochs_all.csv", index=False)

    clean = allx[(~allx["artifact"]) & allx["lz_norm"].notna() & allx["stage"].notna()]
    summ = (clean.groupby(["subject", "stage", "channel"])
                 .agg(n_epochs=("lz_norm", "size"),
                      lz_norm_mean=("lz_norm", "mean"),
                      lz_norm_median=("lz_norm", "median"),
                      lz_norm_sd=("lz_norm", "std"),
                      lz_count_mean=("lz_count", "mean"))
                 .reset_index())
    summ.to_csv(args.out_dir / "lz_subject_stage_summary.csv", index=False)
    print(f"merged {len(files)} subjects, {len(allx)} epoch-rows; "
          f"{len(summ)} subject/stage/channel summaries")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("lz_results"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--task-id", type=int, default=0)
    r.add_argument("--n-tasks", type=int, default=1)
    r.add_argument("--only", nargs="*", help="substring(s) of subject IDs to process")
    r.add_argument("--reref", choices=list(DERIVATIONS), default="ipsi",
                   help="ipsi: C3-A2,C4-A1 | linked: C3/C4 minus mean(A1,A2) | none: raw C3,C4")
    r.add_argument("--band", type=float, nargs=2, metavar=("LO", "HI"), default=(0.5, 40.0),
                   help="band-pass Hz (default 0.5 40); use --band 0 0 to disable")
    r.add_argument("--ptp-uv", type=float, default=500.0,
                   help="flag epochs with peak-to-peak above this (uV) as artefact")
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("merge")
    m.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    if getattr(args, "band", None) == [0.0, 0.0] or getattr(args, "band", None) == (0.0, 0.0):
        args.band = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
