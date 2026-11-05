import os
import pandas as pd

# File paths
cog_path = os.path.expanduser('~/pl1358-01/project_data/worksof/cognitive.csv')
target_path = os.path.expanduser('~/pl1358-01/project_data/worksof/lz_results/night_change/night_change_subject.csv')

# Load datasets
df_cog = pd.read_csv(cog_path)
df_target = pd.read_csv(target_path)

# Columns to bring over from cognitive.csv (excludes 'ID')
cog_cols = [col for col in df_cog.columns if col != 'ID']

# Clean up overlapping columns in target if re-running the script
for col in cog_cols:
    if col in df_target.columns:
        df_target.drop(columns=[col], inplace=True)

# Format matching keys to 5-digit zero-padded strings (e.g., 4 or 4.0 -> "00004")
df_cog['_match_id'] = pd.to_numeric(df_cog['ID'], errors='coerce').astype('Int64').apply(
    lambda x: f"{x:05d}" if pd.notna(x) else None
)
df_target['_match_id'] = df_target['subject'].astype(str).str.split('.').str[0].str.zfill(5)

# Left join: keeps all rows and original row order of night_change_subject.csv
df_merged = df_target.merge(
    df_cog[['_match_id'] + cog_cols],
    on='_match_id',
    how='left'
).drop(columns=['_match_id'])

# Save updated dataset
df_merged.to_csv(target_path, index=False)
print("Merge complete: cognitive columns added to night_change_subject.csv")
