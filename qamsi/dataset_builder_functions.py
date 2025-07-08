from __future__ import annotations

from os import listdir
from dataclasses import dataclass

import pandas as pd

from qamsi.crsp_handler.create_topn_returns import create_top_liquid_dataset
from qamsi.features.dnk_features_targets import create_dnk_features_targets
from qamsi.utils.data import read_csv
from qamsi.config.topn_experiment_config import TopNExperimentConfig


@dataclass
class DatasetData:
    data: pd.DataFrame
    presence_matrix: pd.DataFrame


def build_dataset(config: TopNExperimentConfig) -> Dataset:
    if not config.PATH_OUTPUT.exists():
        raise FileNotFoundError(
            f"Output directory {config.PATH_OUTPUT} does not exist."
        )

    data_df = read_csv(config.PATH_OUTPUT, config.DF_FILENAME)
    presence_matrix = read_csv(
        str(config.PATH_OUTPUT),
        config.PRESENCE_MATRIX_FILENAME,
    )

    return DatasetData(data=data_df, presence_matrix=presence_matrix)


def build_topn_dataset(config: TopNExperimentConfig, verbose: bool = False) -> Dataset:
    if not config.PATH_OUTPUT.exists():
        raise FileNotFoundError(
            f"Output directory {config.PATH_OUTPUT} does not exist."
        )

    if config.DF_FILENAME not in listdir(
        config.PATH_OUTPUT
    ) or config.PRESENCE_MATRIX_FILENAME not in listdir(config.PATH_OUTPUT):
        if verbose:
            print("Creating returns dataset...")
        create_top_liquid_dataset(config)
        if verbose:
            print("Calculating DNK features and targets...")
        create_dnk_features_targets(config, verbose=verbose)

    data_df = read_csv(config.PATH_OUTPUT, config.DF_FILENAME)
    presence_matrix = read_csv(
        str(config.PATH_OUTPUT),
        config.PRESENCE_MATRIX_FILENAME,
    )

    return DatasetData(data=data_df, presence_matrix=presence_matrix)


if __name__ == "__main__":
    from run import Dataset

    topn = 25
    dataset = Dataset.TOPN_US

    config = dataset.value(topn=topn)
    dataset = build_topn_dataset(config, verbose=True)
