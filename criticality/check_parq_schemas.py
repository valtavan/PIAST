#!/usr/bin/env python3
"""Check that all variant folders (and all files in them) share the same parquet schema.
Reads only parquet metadata, so it is fast. Usage:
    uv run --frozen python check_schemas.py [root] [reference]
"""
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "~/pl1358-01/project_data/worksof/criticality").expanduser()
ref_name = sys.argv[2] if len(sys.argv) > 2 else "original"
dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))

info = {}  # variant -> dict(sigs={sig: [files]}, rows={id: n})
for d in dirs:
    sigs, rows = defaultdict(list), {}
    for f in sorted(d.glob("*.parquet")):
        try:
            pf = pq.ParquetFile(f)
            sig = tuple((n, str(t)) for n, t in zip(pf.schema_arrow.names, pf.schema_arrow.types))
            sigs[sig].append(f.stem)
            rows[f.stem] = pf.metadata.num_rows
        except Exception as e:
            sigs[("UNREADABLE", str(e)[:80])].append(f.stem)
    info[d.name] = dict(sigs=sigs, rows=rows)

ref = info.get(ref_name) or next(iter(info.values()))
ref_sig = max(ref["sigs"], key=lambda s: len(ref["sigs"][s]))
ref_cols = dict(ref_sig)
print(f"root: {root}\nreference: {ref_name}  ({len(ref_cols)} columns in its most common schema)\n")


def describe(sig):
    cols = dict(sig)
    missing = [c for c in ref_cols if c not in cols]
    extra = [c for c in cols if c not in ref_cols]
    retyped = [f"{c}: {ref_cols[c]}->{cols[c]}" for c in cols if c in ref_cols and cols[c] != ref_cols[c]]
    order = (list(cols) != list(ref_cols)) and not missing and not extra
    return missing, extra, retyped, order


for name, d in info.items():
    n_files = sum(len(v) for v in d["sigs"].values())
    print(f"== {name}: {n_files} files, {len(d['sigs'])} distinct schema(s)")
    for sig, files in sorted(d["sigs"].items(), key=lambda kv: -len(kv[1])):
        if sig and sig[0][0] == "UNREADABLE":
            print(f"   {len(files)} unreadable files, e.g. {files[:3]}: {sig[0][1]}")
            continue
        missing, extra, retyped, order = describe(sig)
        same = not (missing or extra or retyped or order)
        tag = "same as reference" if same else "DIFFERS"
        print(f"   {len(files):5d} files, {len(sig)} cols -> {tag}   e.g. {files[:3]}")
        if missing: print(f"         missing: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
        if extra:   print(f"         extra:   {extra[:12]}{' ...' if len(extra) > 12 else ''}")
        if retyped: print(f"         dtype:   {retyped[:8]}{' ...' if len(retyped) > 8 else ''}")
        if order:   print("         same columns, different order")

# participants present / row counts across variants
all_ids = sorted(set().union(*[set(d["rows"]) for d in info.values()]))
partial = [i for i in all_ids if not all(i in d["rows"] for d in info.values())]
print(f"\nparticipants: {len(all_ids)} total, {len(partial)} not present in every folder")
for name, d in info.items():
    miss = [i for i in all_ids if i not in d["rows"]]
    if miss:
        print(f"   {name}: missing {len(miss)}  e.g. {miss[:8]}")
diff_rows = [i for i in all_ids if len({d["rows"][i] for d in info.values() if i in d["rows"]}) > 1]
print(f"participants with differing row counts across folders: {len(diff_rows)}")
for i in diff_rows[:8]:
    print("   ", i, {n: d["rows"].get(i) for n, d in info.items()})
