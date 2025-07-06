# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# %%
jkp_data = pd.read_csv(Path("../../data/crsp_raw") / "crsp_80s.csv")
jkp_data = jkp_data.rename(columns={c: c.lower() for c in jkp_data.columns})
jkp_data = jkp_data.dropna(subset=["permno"])
jkp_data["permno"] = jkp_data["permno"].astype(int)
jkp_data["date"] = pd.to_datetime(jkp_data["date"])
jkp_data = jkp_data.sort_values(["date", "permno"])
jkp_data = jkp_data.drop_duplicates(subset=["date", "permno"])
jkp_data = jkp_data.set_index(["date", "permno"])
jkp_data.head()
# %%
# mapping = jkp_data.reset_index()[["permno", "comnam"]].drop_duplicates()
# mapping = mapping.set_index("permno")
# %%
# mapping.to_csv(Path("../../data/output") / "crsp_mapping.csv")
# %%
CRSP_IGNORED = [-66, -77, -88, -99]

jkp_data = jkp_data[
    (jkp_data["ret"] != CRSP_IGNORED[0])
    & (jkp_data["ret"] != CRSP_IGNORED[1])
    & (jkp_data["ret"] != CRSP_IGNORED[2])
    & (jkp_data["ret"] != CRSP_IGNORED[3])
]
# %%
jkp_data["ret"] = jkp_data["ret"].replace("C", np.nan).astype(float)
# %%
# jkp_data["adj_prc"] = np.abs(jkp_data["prc"]) / jkp_data["cfacpr"].ffill().fillna(1).replace(0, 1.0)
# %%
jkp_data["mktcap"] = jkp_data["shrout"] * 1_000 * jkp_data["prc"]
# %%
dolvol = jkp_data.reset_index().pivot(index="date", columns="permno", values="mktcap")
# %%
dolvol = dolvol.resample("ME").last()
# %%
N_LARGEST = 500

presence_matrix = dolvol.apply(
    lambda x: x >= x.nlargest(N_LARGEST).min(), axis=1
).astype(float)
presence_matrix[presence_matrix == 0] = np.nan
# %%
presence_matrix = presence_matrix.dropna(axis=1, how="all")
presence_matrix.shape
# %%
pivoted_returns = (
    jkp_data.loc[
        jkp_data.index.get_level_values("permno").isin(presence_matrix.columns)
    ]
    .reset_index()
    .pivot_table(index="date", columns="permno", values="ret")
)
# %%
# last_selection = presence_matrix.iloc[-1]
# last_selection = last_selection[last_selection.notna()].index
#
# df_data = jkp_data.reset_index()
# df_data[df_data["permno"].isin(last_selection)]["comnam"].drop_duplicates()
# %%
full_df = pivoted_returns
# %%
valid_cols = presence_matrix.columns.intersection(full_df.columns)
len(valid_cols)
# %%
presence_matrix = presence_matrix.reset_index()
presence_matrix["date"] = pd.to_datetime(presence_matrix["date"])
presence_matrix = presence_matrix.set_index("date")
# %%
presence_matrix = presence_matrix.resample("D").ffill()
# %%
merged_index = full_df.merge(
    presence_matrix, left_index=True, right_index=True, how="inner"
).index
# %%
full_df = full_df.loc[merged_index]
presence_matrix = presence_matrix.loc[merged_index]
full_df.shape, presence_matrix.shape
# %%
full_df[valid_cols].to_csv(Path("../../data/output") / f"top{N_LARGEST}_data.csv")
# %%
presence_matrix[valid_cols].to_csv(
    Path("../../data/output") / f"top{N_LARGEST}_presence_matrix.csv"
)
# %%
pd.DataFrame(valid_cols).to_csv(
    Path("../../data/output") / f"top{N_LARGEST}_stocks_list.csv",
    index=False,
)
