import os
import pandas as pd

# File paths
source_path = os.path.expanduser('~/pl1358-01/project_data/SOFdataset/sleep/datasets/sof-visit-8-harmonized-dataset-0.8.0.csv')
target_path = os.path.expanduser('~/pl1358-01/project_data/worksof/cognitive.csv')

# AHI variables to merge
vars_to_add = ['nsrr_ahi_hp3r_aasm15', 'nsrr_ahi_hp4u_aasm15']

# Load datasets
df_target = pd.read_csv(target_path)
df_source = pd.read_csv(source_path, usecols=['nsrrid'] + vars_to_add)

# Remove any empty AHI columns from prior attempt if present
for col in vars_to_add:
    if col in df_target.columns:
        df_target.drop(columns=[col], inplace=True)

# Create normalized matching keys (converts "4.0" and "4" to integer 4)
df_target['_match_id'] = pd.to_numeric(df_target['ID'], errors='coerce').astype('Int64')
df_source['_match_id'] = pd.to_numeric(df_source['nsrrid'], errors='coerce').astype('Int64')

# Left join on normalized integer IDs
df_merged = df_target.merge(
    df_source[['_match_id'] + vars_to_add],
    on='_match_id',
    how='left'
).drop(columns=['_match_id'])

# Overwrite target CSV with populated values
df_merged.to_csv(target_path, index=False)
print("Merge complete: AHI columns successfully populated.")
