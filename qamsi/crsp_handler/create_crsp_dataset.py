from __future__ import annotations

from os import listdir
from typing import Callable

import numpy as np
import pandas as pd

from qamsi.config.base_experiment_config import BaseExperimentConfig
from qamsi.utils.data import read_csv


CRSP_MAPPING_FILENAME = "crsp_mapping.csv"
CRSP_IGNORED_PLACEHOLDERS = [-66, -77, -88, -99]


def _load_crsp_data(
    config: BaseExperimentConfig, store_mapping: bool = False
) -> pd.DataFrame:
    crsp_data = pd.read_csv(config.CRSP_PATH / config.CRSP_FILENAME)
    crsp_data = crsp_data.rename(columns={c: c.lower() for c in crsp_data.columns})
    crsp_data = crsp_data.dropna(subset=["permno"])
    crsp_data["permno"] = crsp_data["permno"].astype(int)
    crsp_data["date"] = pd.to_datetime(crsp_data["date"])
    crsp_data = crsp_data.sort_values(["date", "permno"])
    crsp_data = crsp_data.drop_duplicates(subset=["date", "permno"])
    crsp_data = crsp_data.set_index(["date", "permno"])

    if store_mapping and (CRSP_MAPPING_FILENAME not in listdir(config.CRSP_PATH)):
        mapping = crsp_data.reset_index()[["permno", "comnam"]].drop_duplicates()
        mapping = mapping.set_index("permno")
        mapping.to_csv(config.PATH_OUTPUT / CRSP_MAPPING_FILENAME)

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


def _create_presence_matrix(
    universe_builder_fn: Callable[
        [
            pd.DataFrame,
        ],
        pd.DataFrame,
    ],
    crsp_data: pd.DataFrame,
) -> pd.DataFrame:
    presence_matrix = universe_builder_fn(crsp_data)

    presence_matrix = presence_matrix.reset_index()
    presence_matrix["date"] = pd.to_datetime(presence_matrix["date"])
    presence_matrix = presence_matrix.set_index("date")
    presence_matrix = presence_matrix.resample("D").ffill()

    return presence_matrix


def _create_returns(
    config: BaseExperimentConfig,
    universe_builder_fn: Callable[
        [
            pd.DataFrame,
        ],
        pd.DataFrame,
    ],
    store_mapping: bool = False,
) -> None:
    try:
        crsp_data = pd.read_csv(config.PATH_TMP / config.CRSP_FULL_TMP_FILENAME)
        crsp_data["date"] = pd.to_datetime(crsp_data["date"])
        crsp_data = crsp_data.set_index(["date", "permno"])
    except FileNotFoundError:
        crsp_data = _load_crsp_data(config, store_mapping=store_mapping)

    crsp_data = _filter_invalid_crsp_returns(crsp_data)
    presence_matrix = _create_presence_matrix(universe_builder_fn, crsp_data)

    pivoted_returns = (
        crsp_data.loc[
            crsp_data.index.get_level_values("permno").isin(presence_matrix.columns)
        ]
        .reset_index()
        .pivot_table(index="date", columns="permno", values="ret")
    )

    pivoted_returns.to_csv(config.PATH_TMP / (config.PREFIX + config.RAW_DATA_FILENAME))
    presence_matrix.to_csv(
        config.PATH_OUTPUT / (config.PREFIX + config.PRESENCE_MATRIX_FILENAME)
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

    rf = pd.read_excel(config.PATH_RF_RATE / config.RF_RATE_FILENAME)
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

    return crsp_returns.merge(
        spx_fut["spx_fut"], left_index=True, right_index=True, how="left"
    )


def create_crsp_dataset(
    config: BaseExperimentConfig,
    universe_builder_fn: Callable[
        [
            pd.DataFrame,
        ],
        pd.DataFrame,
    ],
    store_mapping: bool = False,
) -> None:
    raw_full_data_filename = config.PREFIX + config.DF_FILENAME
    raw_presence_matrix_filename = config.PREFIX + config.PRESENCE_MATRIX_FILENAME
    if raw_full_data_filename not in listdir(
        config.PATH_OUTPUT
    ) or raw_presence_matrix_filename not in listdir(config.PATH_OUTPUT):
        raw_data_filename = config.PREFIX + config.RAW_DATA_FILENAME

        if raw_data_filename not in listdir(
            config.PATH_TMP
        ) or raw_presence_matrix_filename not in listdir(config.PATH_OUTPUT):
            _create_returns(
                config=config,
                universe_builder_fn=universe_builder_fn,
                store_mapping=store_mapping,
            )

        crsp_returns = read_csv(config.PATH_TMP, raw_data_filename)

        crsp_returns = _add_factors(
            crsp_returns, config=config, selected_factors=list(config.FACTORS)
        )
        crsp_returns = _add_rf_rate_and_market_index(crsp_returns, config=config)
        crsp_returns = _add_hedging_assets(crsp_returns, config=config)

        crsp_returns.to_csv(config.PATH_OUTPUT / (config.PREFIX + config.DF_FILENAME))


if __name__ == "__main__":
    from qamsi.crsp_handler.universe_builder_functions import (
        mkt_cap_topn_universe_builder_fn,
    )
    from run import Dataset

    TOP_N = 30
    dataset = Dataset.TOPN_US

    settings = dataset.value(topn=TOP_N)
    create_crsp_dataset(
        config=settings,
        universe_builder_fn=lambda cfg: mkt_cap_topn_universe_builder_fn(
            cfg, topn=TOP_N
        ),
    )
