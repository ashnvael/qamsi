from __future__ import annotations

from os import listdir

import numpy as np
import pandas as pd

from qamsi.config.experiment_config import BaseExperimentConfig
from qamsi.config.topn_experiment_config import TopNExperimentConfig


CRSP_MAPPING_FILENAME = "crsp_mapping.csv"
CRSP_IGNORED_PLACEHOLDERS = [-66, -77, -88, -99]


def _load_crsp_data(
    config: BaseExperimentConfig, store: bool = True, store_mapping: bool = False
) -> pd.DataFrame:
    crsp_data = pd.read_csv(config.CRSP_PATH / config.CRSP_FILENAME)
    crsp_data = crsp_data.rename(columns={c: c.lower() for c in crsp_data.columns})
    crsp_data = crsp_data.dropna(subset=["permno"])
    crsp_data["permno"] = crsp_data["permno"].astype(int)
    crsp_data["date"] = pd.to_datetime(crsp_data["date"])
    crsp_data = crsp_data.sort_values(["date", "permno"])
    crsp_data = crsp_data.drop_duplicates(subset=["date", "permno"])
    crsp_data = crsp_data.set_index(["date", "permno"])

    if store_mapping:
        mapping = crsp_data.reset_index()[["permno", "comnam"]].drop_duplicates()
        mapping = mapping.set_index("permno")
        mapping.to_csv(config.PATH_OUTPUT / CRSP_MAPPING_FILENAME)

    if store:
        crsp_data.to_csv(config.PATH_TMP / config.CRSP_FULL_TMP_FILENAME)

    return crsp_data


def _filter_invalid_crsp_returns(crsp_data: pd.DataFrame) -> pd.DataFrame:
    crsp_data = crsp_data[
        (crsp_data["ret"] != CRSP_IGNORED_PLACEHOLDERS[0])
        & (crsp_data["ret"] != CRSP_IGNORED_PLACEHOLDERS[1])
        & (crsp_data["ret"] != CRSP_IGNORED_PLACEHOLDERS[2])
        & (crsp_data["ret"] != CRSP_IGNORED_PLACEHOLDERS[3])
    ]
    crsp_data["ret"] = crsp_data["ret"].replace("C", np.nan).astype(float)

    return crsp_data


def _create_top_liquid_presence_matrix(
    crsp_data: pd.DataFrame, topn: int
) -> pd.DataFrame:
    crsp_data["mktcap"] = crsp_data["shrout"] * 1_000 * crsp_data["prc"]
    dolvol = crsp_data.reset_index().pivot(
        index="date", columns="permno", values="mktcap"
    )
    dolvol = dolvol.resample("ME").last()

    presence_matrix = dolvol.apply(
        lambda x: x >= x.nlargest(topn).min(), axis=1
    ).astype(float)
    presence_matrix[presence_matrix == 0] = np.nan

    presence_matrix = presence_matrix.dropna(axis=1, how="all")

    presence_matrix = presence_matrix.reset_index()
    presence_matrix["date"] = pd.to_datetime(presence_matrix["date"])
    presence_matrix = presence_matrix.set_index("date")
    presence_matrix = presence_matrix.resample("D").ffill()

    return presence_matrix


def _create_top_liquid_returns(
    config: BaseExperimentConfig, topn: int, store_mapping: bool = False
) -> None:
    try:
        crsp_data = pd.read_csv(config.PATH_TMP / config.CRSP_FULL_TMP_FILENAME)
    except FileNotFoundError:
        crsp_data = _load_crsp_data(config, store=True, store_mapping=store_mapping)

    crsp_data = _filter_invalid_crsp_returns(crsp_data)
    presence_matrix = _create_top_liquid_presence_matrix(crsp_data, topn)

    pivoted_returns = (
        crsp_data.loc[
            crsp_data.index.get_level_values("permno").isin(presence_matrix.columns)
        ]
        .reset_index()
        .pivot_table(index="date", columns="permno", values="ret")
    )

    pivoted_returns.to_csv(
        config.PATH_OUTPUT / (f"top{topn}_" + config.RAW_DATA_FILENAME)
    )
    presence_matrix.to_csv(
        config.PATH_OUTPUT / (f"top{topn}_" + config.PRESENCE_MATRIX_FILENAME)
    )


def _add_factors(
    crsp_returns: pd.DataFrame,
    config: BaseExperimentConfig,
    selected_factors: list[str],
) -> pd.DataFrame:
    jkp_factors = pd.read_csv(config.PATH_FACTORS / config.FACTORS_FILENAME)
    jkp_factors = jkp_factors[jkp_factors["name"].isin(selected_factors)]
    jkp_factors["date"] = pd.to_datetime(jkp_factors["date"])
    factors = jkp_factors.pivot_table(index="date", columns="name", values="ret")

    return crsp_returns.merge(factors, left_index=True, right_index=True, how="left")


def _add_rf_rate_and_market_index(
    crsp_returns: pd.DataFrame, config: BaseExperimentConfig
) -> pd.DataFrame:
    spx = pd.read_excel(config.PATH_MKT / config.MKT_FILENAME, skiprows=6)
    spx = spx.rename(columns={"Date": "date", "PX_LAST": "spx"})
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.set_index("date")
    spx = spx.sort_index()
    spx = spx[["spx"]].pct_change()

    rf = pd.read_excel(config.PATH_RF_RATE / config.RF_RATE_FILENAME, skiprows=6)
    rf = rf.rename(columns={"Date": "date", "RF": "rf"})
    rf["date"] = pd.to_datetime(rf["date"], format="%Y%m%d")
    rf = rf.set_index("date")
    rf = rf["rf"] / 100
    spx_rf = spx.merge(
        rf.rename("acc_rate"), left_index=True, right_index=True, how="left"
    )
    spx_rf["spx"] = spx_rf["spx"].sub(spx_rf["acc_rate"], axis=0)

    return crsp_returns.merge(spx_rf, left_index=True, right_index=True, how="left")


def _add_hedging_assets(
    crsp_returns: pd.DataFrame, config: BaseExperimentConfig
) -> pd.DataFrame:
    spx_fut = pd.read_excel(
        config.PATH_HEDGING_ASSETS / config.HEDGING_ASSETS_FILENAME, skiprows=6
    )
    spx_fut = spx_fut.rename(columns={"Date": "date", "PX_LAST": "price"})
    spx_fut["date"] = pd.to_datetime(spx_fut["date"])
    spx_fut = spx_fut.set_index("date")
    spx_fut = spx_fut.sort_index()
    spx_fut = spx_fut[["price"]]
    spx_fut["spx_fut"] = spx_fut["price"].pct_change()

    return crsp_returns.merge(spx_fut, left_index=True, right_index=True, how="left")


def create_top_liquid_dataset(
    config: TopNExperimentConfig, store_mapping: bool = False
) -> None:
    topn = config.topn
    raw_data_filename = f"top{topn}_" + config.RAW_DATA_FILENAME

    if raw_data_filename not in listdir(config.PATH_OUTPUT):
        _create_top_liquid_returns(config, topn, store_mapping=store_mapping)

    crsp_returns = pd.read_csv(config.PATH_OUTPUT / raw_data_filename)

    crsp_returns = _add_factors(
        crsp_returns, config=config, selected_factors=list(config.FACTORS)
    )
    crsp_returns = _add_rf_rate_and_market_index(crsp_returns, config=config)
    crsp_returns = _add_hedging_assets(crsp_returns, config=config)

    crsp_returns.to_csv(
        config.PATH_OUTPUT / (f"top{topn}_" + config.TRADE_DATASET_TMP_FILENAME)
    )


if __name__ == "__main__":
    from run import Dataset

    topn = 30
    dataset = Dataset.TOPN_US

    config = dataset.value(topn=topn)
    create_top_liquid_dataset(config=config)
