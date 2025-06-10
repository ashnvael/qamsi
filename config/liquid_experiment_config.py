from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from qamsi.config.experiment_config import BaseExperimentConfig


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    # Strategy Settings
    FRACTIONAL_DIFFERENCING: float = field(
        default=0.5, metadata={"docs": "Fractional Difference Order"}
    )

    RANDOM_SEED: int = field(default=12, metadata={"docs": "Fix random seed"})

    # Folders
    PATH_FEATURES: Path = field(
        default=Path(__file__).resolve().parents[4] / "data" / "gw",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_MARKET_DATA: Path = field(
        default=Path(__file__).resolve().parents[4] / "data" / "spx_stocks",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_BETTER_MARKET_DATA: Path = field(
        default=Path(__file__).resolve().parents[4] / "data" / "spxc",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_INPUT: Path = field(
        default=Path(__file__).resolve().parents[1] / "data" / "input",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_OUTPUT: Path = field(
        default=Path(__file__).resolve().parents[1] / "data" / "output",
        metadata={"docs": "Relative path to data folder"},
    )

    SAVE_PATH: Path = field(
        default=Path(__file__).resolve().parents[1] / "backtests" / "runs",
        metadata={"docs": "Relative path to data folder"},
    )

    # Filenames
    INITIAL_DF_FILENAME: str = field(
        default="initial_df.csv", metadata={"docs": "Initial preprocessed data"}
    )

    JKP_DATA_FILENAME: str = field(
        default="jkp_data.csv", metadata={"docs": "JKP Stock Characteristics"}
    )

    STOCKS_LIST_FILENAME: str = field(
        default="liquid_stocks_list.csv", metadata={"docs": "Stocks list"}
    )

    INITIAL_FEATURES_FILENAME: str = field(
        default="initial_features_df.csv",
        metadata={"docs": "Initial preprocessed features"},
    )

    DF_FILENAME: str = field(
        default="liquid_data.csv",
        metadata={"docs": "Preprocessed data (after experiments)"},
    )

    RETURNS_FILENAME: str = field(
        default="returns_incl_div_consituents_w_name.csv",
        metadata={"docs": "Returns With Dividends"},
    )

    BETTER_RETURNS_FILENAME: str = field(
        default="returns_data_cleaned_better.parquet",
        metadata={"docs": "Returns With Dividends"},
    )

    PRESENCE_MATRIX_FILENAME: str = field(
        default="presence_matrix.csv",
        metadata={"docs": "Matrix of Presence in S&P500 for each day"},
    )

    # Experiment Settings
    START_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2000-02-08"),
        metadata={"docs": "Date to start training"},
    )

    END_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2024-12-31"),
        metadata={"docs": "Date to end train (as per paper by Paoella and co)"},
    )

    REBALANCE_FREQ: int | str | None = field(
        default=7,
        metadata={
            "docs": "Frequency of rebalancing in days (pass `int`) or pandas freq (pass `str`). "
            "Pass `None` for Buy & Hold portfolio",
        },
    )

    HEDGE_FREQ: int | str | None = field(
        default=1,
        metadata={
            "docs": "Frequency of hedging in days (pass `int`) or pandas freq (pass `str`). Pass `None` for Buy & Hold portfolio",
        },
    )

    N_LOOKBEHIND_PERIODS: int = field(
        default=252,
        metadata={
            "docs": "Number of rebalance periods to take into rolling regression"
        },
    )

    MIN_ROLLING_PERIODS: int = field(
        default=252,
        metadata={"docs": "Number of minimum rebalance periods to run the strategy"},
    )

    # Universe Setting
    ASSET_UNIVERSE: tuple[str] = field(
        default=("spx",),
        metadata={"docs": "Tradeable assets tuple"},
    )

    HEDGING_ASSETS: tuple[str] = field(
        default=("spx_fut",),
        metadata={"docs": "Tradeable assets tuple"},
    )

    FACTORS: tuple[str] = field(
        default=("spx",),
        metadata={"docs": "Tradeable factors tuple"},
    )

    RF_NAME: str = field(
        default="acc_rate",
        metadata={"docs": "Risk-Free rate column name"},
    )

    MKT_NAME: str = field(
        default="spx",
        metadata={"docs": "Market index column name"},
    )

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}:"
        for key, value in self.__dict__.items():
            string += f"\n* {key} = {value}"
        return string

    def __repr__(self) -> str:
        return self.__str__()
