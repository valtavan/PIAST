from pathlib import Path
import pandas as pd

RES = Path.home() / "pl1358-01" / "project_data" / "worksof" / "lz_results"

d = pd.read_csv(RES / "lz_epochs_all.csv", dtype={"subject": str})
s = pd.read_csv(RES / "lz_subject_stage_summary.csv", dtype={"subject": str})

# --- checks from before ---
print(d.subject.nunique(), s.subject.nunique())
print(d.groupby("stage")["artifact"].mean())
print(d.groupby("stage")["ptp_uv"].quantile([.5, .95, .99]).unstack())
print(s.groupby("subject").stage.nunique().value_counts())
print(d["lz_norm"].isna().groupby(d.subject).sum().sort_values().tail())

# --- subjects missing stages: clean epochs per stage ---
n_stage = s.groupby("subject").stage.nunique()
odd = n_stage[n_stage < 5].index
print(s[s.subject.isin(odd)]
        .pivot_table(index="subject", columns="stage", values="n_epochs", aggfunc="first")
        .fillna(0).astype(int))

# --- minimum clean epochs per subject/stage; how many subjects survive ---
MIN_EPOCHS = 10
ok = s[s.n_epochs >= MIN_EPOCHS]
print(ok.groupby(["stage", "channel"]).subject.nunique().unstack())
