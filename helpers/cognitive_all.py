import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

DATA = Path('~/pl1358-01/project_data/SOFdataset').expanduser()
OUT = Path('~/pl1358-01/project_data/worksof/cognitive.csv').expanduser()
OUT.parent.mkdir(parents=True, exist_ok=True)


def read_sas_zip(name):
    """Read DATA/<stem>.zip (e.g. v8sleep.zip) without unzipping to disk."""
    stem = name.removesuffix('.sas7bdat')
    with zipfile.ZipFile(DATA / f'{stem}.zip') as z:
        members = [m for m in z.namelist() if m.lower().endswith('.sas7bdat')]
        if len(members) != 1:
            raise ValueError(f'{name}.zip: expected 1 .sas7bdat file, found {z.namelist()}')
        return pd.read_sas(BytesIO(z.read(members[0])), format='sas7bdat')


def load(name, rename, cohort):
    """Read one SAS file, keep only the columns in `rename`, rename them, tag the cohort."""
    df = read_sas_zip(name)
    df = df[list(rename)].rename(columns=rename)
    df['COHORT'] = cohort
    return df


# --- Log all column names in v6cogfxn ---
v6cols = list(read_sas_zip('v6cogfxn.sas7bdat').columns)
print(f'v6cogfxn: {len(v6cols)} columns', flush=True)
for c in v6cols:
    print(f'  {c}', flush=True)

# Same for the AA cohort's visit 6 file, to check the variable names
v6aacols = list(read_sas_zip('v6aacogfxn.sas7bdat').columns)
print(f'v6aacogfxn: {len(v6aacols)} columns', flush=True)
for c in v6aacols:
    print(f'  {c}', flush=True)

# --- Visit 8 sleep files: keep only participants who had PSG and passed ---
# V8PSGPI = did participant have polysomnography; V8PSGST = PSG pass/fail
psg_cols = {'ID': 'ID', 'V8PSGPI': 'PSG_DONE', 'V8PSGST': 'PSG_PASS'}
sleep = pd.concat([
    load('v8sleep.sas7bdat', psg_cols, 'MAIN'),
    load('v8aasleep.sas7bdat', psg_cols, 'AA'),
])
sleep = sleep[(sleep.PSG_DONE == 1) & (sleep.PSG_PASS == 1)]
base = sleep[['ID', 'COHORT']].drop_duplicates(subset='ID')

# --- Pre-PSG cognition (visit 6): MMSE and Trails ---
pre = pd.concat([
    load('v6cogfxn.sas7bdat', {'ID': 'ID', 'V6MMSE': 'PREMMSE', 'V6TRLNUM': 'PRETRB'}, 'MAIN'),
    load('v6aacogfxn.sas7bdat', {'ID': 'ID', 'AAMMSE': 'PREMMSE', 'AATRLNUM': 'PRETRB'}, 'AA'),
])

# --- Post-PSG cognition (visit 9): Teng score ---
post = pd.concat([
    load('v9cogfxn.sas7bdat', {'ID': 'ID', 'V9TENG': 'V9TENG'}, 'MAIN'),
    load('v9aacogfxn.sas7bdat', {'ID': 'ID', 'V9TENG': 'V9TENG'}, 'AA'),
])

# One row per ID in each table, otherwise merges silently duplicate rows
for name, t in {'sleep': sleep, 'pre': pre, 'post': post}.items():
    dups = t['ID'].duplicated().sum()
    if dups:
        print(f'WARNING: {dups} duplicated IDs in {name}')

# Left-merge onto the PSG-pass IDs: everyone who passed PSG stays in,
# with NaN where a visit (e.g. the visit 9 follow-up) is missing.
dffinal = base
for t in (pre, post):
    dffinal = dffinal.merge(t.drop(columns='COHORT'), on='ID', how='left')

print(dffinal.shape)
print(dffinal.groupby('COHORT').count())  # non-missing counts per variable, per cohort

dffinal.to_csv(OUT, index=False, header=True)
