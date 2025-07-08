from __future__ import annotations

from os import listdir
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from qamsi.crsp_handler.create_dataset import create_dataset
from qamsi.crsp_handler.universe_builder_functions import mkt_cap_topn_universe_builder_fn
from qamsi.features.dnk_features_targets import create_dnk_features_targets
from qamsi.utils.data import read_csv
from qamsi.config.experiment_config import BaseExperimentConfig
from qamsi.config.topn_experiment_config import TopNExperimentConfig


@dataclass
class DatasetData:
    data: pd.DataFrame
    presence_matrix: pd.DataFrame


def build_dataset(
    config: BaseExperimentConfig,
    prefix: str,
    universe_builder_fn: Callable[
        [
            pd.DataFrame,
        ],
        pd.DataFrame,
    ],
    features_targets_fn: Callable[[BaseExperimentConfig, bool], None],
    verbose: bool = False,
) -> DatasetData:
    if not config.PATH_OUTPUT.exists():
        raise FileNotFoundError(
            f"Output directory {config.PATH_OUTPUT} does not exist."
        )

    if config.DF_FILENAME not in listdir(
        config.PATH_OUTPUT
    ) or config.PRESENCE_MATRIX_FILENAME not in listdir(config.PATH_OUTPUT):
        if verbose:
            print("Creating returns dataset...")
        create_dataset(
            config=config, prefix=prefix, universe_builder_fn=universe_builder_fn
        )
        if verbose:
            print("Calculating features and targets...")
        features_targets_fn(config, verbose)

    data_df = read_csv(config.PATH_OUTPUT, config.DF_FILENAME)
    presence_matrix = read_csv(
        str(config.PATH_OUTPUT),
        config.PRESENCE_MATRIX_FILENAME,
    )

    return DatasetData(data=data_df, presence_matrix=presence_matrix)


def build_dnk_dataset(config: TopNExperimentConfig, verbose: bool = False) -> DatasetData:
    topn = config.topn
    prefix = f"top{topn}_"
    return build_dataset(
        config=config,
        prefix=prefix,
        universe_builder_fn=lambda data: mkt_cap_topn_universe_builder_fn(data, topn=topn),
        features_targets_fn=create_dnk_features_targets,
        verbose=verbose,
    )


if __name__ == "__main__":
    from run import Dataset

    TOP_N = 30
    dataset = Dataset.TOPN_US

    settings = dataset.value(topn=TOP_N)
    dataset = build_dnk_dataset(settings, verbose=True)
