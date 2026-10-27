from pathlib import Path

import pandas as pd

WORKSOF = Path('~/pl1358-01/project_data/worksof').expanduser()
COG = WORKSOF / 'cognitive.csv'
AGE = WORKSOF / 'age-participant.csv'
OUT = COG

cog = pd.read_csv(COG)
# sep=None sniffs the delimiter (comma, semicolon, tab); utf-8-sig drops a BOM if present
age = pd.read_csv(AGE, sep=None, engine='python', encoding='utf-8-sig')
age.columns = [str(c).strip() for c in age.columns]
print(f'{AGE.name} columns: {list(age.columns)}', flush=True)

# Accept any capitalisation of the ID and age columns and normalise them
age = age.rename(columns={c: 'ID' for c in age.columns if c.lower() in ('id', 'nsrrid')})
age = age.rename(columns={c: 'V8AGE' for c in age.columns if c.lower() == 'v8age'})
missing_cols = [c for c in ('ID', 'V8AGE') if c not in age.columns]
if missing_cols:
    raise SystemExit(f'{AGE.name} has no column(s) {missing_cols}; found {list(age.columns)}')
age = age[['ID', 'V8AGE']]

# If cognitive.csv already has V8AGE (from an earlier version of cognitive_all.py), replace it
cog = cog.drop(columns='V8AGE', errors='ignore')


def id_key(s):
    """Make IDs comparable across files (whitespace, '123.0' vs '123')."""
    return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


cog['_key'] = id_key(cog['ID'])
age['_key'] = id_key(age['ID'])

dups = age['_key'].duplicated().sum()
if dups:
    print(f'WARNING: {dups} duplicated IDs in {AGE.name}; keeping the first of each', flush=True)
    age = age.drop_duplicates(subset='_key')

# Left merge: every participant in cognitive.csv stays, V8AGE is NaN if not in the age file
merged = cog.merge(age[['_key', 'V8AGE']], on='_key', how='left').drop(columns='_key')

print(f'cognitive.csv: {len(cog)} rows; {AGE.name}: {len(age)} rows', flush=True)
print(f"V8AGE present for {merged['V8AGE'].notna().sum()} of {len(merged)} participants", flush=True)
missing = cog.loc[~cog['_key'].isin(age['_key']), 'ID']
if len(missing):
    print(f'{len(missing)} IDs in cognitive.csv have no age, e.g. {list(missing.head(5))}', flush=True)
only_age = (~age['_key'].isin(cog['_key'])).sum()
print(f'{only_age} IDs in {AGE.name} are not in cognitive.csv (dropped)', flush=True)

merged.to_csv(OUT, index=False)
print(f'Saved {merged.shape} to {OUT}', flush=True)
