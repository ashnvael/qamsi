from __future__ import annotations

from os import listdir
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from qamsi.crsp_handler.create_crsp_dataset import create_crsp_dataset
from qamsi.crsp_handler.universe_builder_functions import (
    mkt_cap_topn_universe_builder_fn,
    mkt_cap_quantile_universe_builder_fn,
)
from qamsi.features.dnk_features_targets import create_dnk_features_targets
from qamsi.utils.data import read_csv
from qamsi.config.base_experiment_config import BaseExperimentConfig
from qamsi.config.topn_experiment_config import TopNExperimentConfig
from qamsi.config.jkp_experiment_config import JKPExperimentConfig
from qamsi.config.spx_experiment_config import SPXExperimentConfig


@dataclass
class DatasetData:
    data: pd.DataFrame
    presence_matrix: pd.DataFrame
    mkt_cap: pd.DataFrame | None = None
    adj_price: pd.DataFrame | None = None


def build_dataset(
    config: BaseExperimentConfig,
    universe_builder_fn: Callable[
        [
            pd.DataFrame,
        ],
        pd.DataFrame,
    ],
    features_targets_fn: Callable[[BaseExperimentConfig, bool], None] | None = None,
    verbose: bool = False,
) -> DatasetData:
    if not config.PATH_OUTPUT.exists():
        raise FileNotFoundError(
            f"Output directory {config.PATH_OUTPUT} does not exist."
        )

    df_filename = config.PREFIX + config.DF_FILENAME
    pm_filename = config.PREFIX + config.PRESENCE_MATRIX_FILENAME
    available_files = listdir(config.PATH_OUTPUT)

    if df_filename not in available_files or pm_filename not in available_files:
        if verbose:
            print("Creating returns dataset...")
        create_crsp_dataset(config=config, universe_builder_fn=universe_builder_fn)

        if features_targets_fn is not None:
            if verbose:
                print("Calculating features and targets...")
            features_targets_fn(config, verbose)

    data_df = read_csv(config.PATH_OUTPUT, df_filename)
    presence_matrix = read_csv(config.PATH_OUTPUT, pm_filename)

    return DatasetData(data=data_df, presence_matrix=presence_matrix)


def build_dnk_dataset(
    config: TopNExperimentConfig, verbose: bool = False
) -> DatasetData:
    return build_dataset(
        config=config,
        universe_builder_fn=lambda data: mkt_cap_topn_universe_builder_fn(
            data, topn=config.TOPN
        ),
        features_targets_fn=create_dnk_features_targets,
        verbose=verbose,
    )


def build_jkp_dataset(
    config: JKPExperimentConfig, verbose: bool = False
) -> DatasetData:
    return build_dataset(
        config=config,
        universe_builder_fn=lambda data: mkt_cap_quantile_universe_builder_fn(
            data,
            quantile=config.MCAP_SELECTION_QUANTILE,
        ),
        features_targets_fn=None,
        verbose=verbose,
    )


def build_spx_dataset(
    config: SPXExperimentConfig,
    spx_presence_matrix: pd.DataFrame,
    verbose: bool = False,
) -> DatasetData:
    return build_dataset(
        config=config,
        universe_builder_fn=lambda data: spx_presence_matrix,
        features_targets_fn=None,
        verbose=verbose,
    )


if __name__ == "__main__":
    # from run import Dataset
    #
    # TOP_N = 30
    # dataset = Dataset.TOPN_US
    #
    # settings = dataset.value(topn=TOP_N)
    # dataset = build_dnk_dataset(settings, verbose=True)

    # from run import Dataset
    #
    # dataset = Dataset.JKP
    #
    # settings = dataset.value()
    # dataset = build_jkp_dataset(settings, verbose=True)

    from run import Dataset

    dataset = Dataset.SPX_US

    settings = dataset.value()

    pm = read_csv(
        settings.PATH_OUTPUT,
        settings.PREFIX + settings.PRESENCE_MATRIX_FILENAME,
        date_column="caldt",
        rename_column=True,
    )
    dataset = build_spx_dataset(settings, spx_presence_matrix=pm, verbose=True)
