from pathlib import Path
import pandas as pd

# Define paths using ~ expansion
file1_path = Path("~/pl1358-01/project_data/SOFdataset/sleep/datasets/sof-visit-8-harmonized-dataset-0.8.0.csv").expanduser()
file2_path = Path("~/pl1358-01/project_data/SOFdataset/sleep/datasets/sof-visit-8-dataset-0.8.0.csv").expanduser()
output_path = Path("~/pl1358-01/project_data/SOFdataset/sleep/datasets/age-participant.csv").expanduser()

# Load the files
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Combine the two columns row-by-row
result = pd.concat([df1["nsrrid"], df2["v8age"]], axis=1)

# Export to CSV
result.to_csv(output_path, index=False)
