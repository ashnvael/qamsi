from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentConfig:
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
        default=Path(__file__).resolve().parents[2] / "data" / "spx_stocks",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_BETTER_MARKET_DATA: Path = field(
        default=Path(__file__).resolve().parents[0] / "data" / "spxc",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_OUTPUT: Path = field(
        default=Path(__file__).resolve().parents[1] / "data" / "output",
        metadata={"docs": "Relative path to data folder"},
    )

    # Filenames
    INITIAL_DF_FILENAME: str = field(
        default="initial_features_df.csv",
        metadata={"docs": "Initial preprocessed data"},
    )

    STOCKS_LIST_FILENAME: str = field(
        default="stocks_list.csv", metadata={"docs": "Stocks list"}
    )

    INITIAL_FEATURES_FILENAME: str = field(
        default="initial_features_df.csv",
        metadata={"docs": "Initial preprocessed features"},
    )

    PRICES_DF_FILENAME: str = field(
        default="prices_df.csv", metadata={"docs": "Preprocessed price data"}
    )

    DF_FILENAME: str = field(
        default="data_df.csv",
        metadata={"docs": "Preprocessed data (after experiments)"},
    )

    RETURNS_FILENAME: str = field(
        default="returns_data_cleaned_better.parquet",
        metadata={"docs": "Returns With Dividends"},
    )

    # Experiment Settings
    TRAIN_START_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2000-02-08"),
        metadata={"docs": "Date to start training"},
    )

    TRAIN_END_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2022-03-31"),
        metadata={"docs": "Date to end train (as per paper by Paoella and co)"},
    )

    TEST_END_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2025-03-06"),
        metadata={"docs": "Date to end analysis"},
    )

    REBALANCE_FREQ_DAYS: int = field(
        default=3 * 5 * 4,
        metadata={"docs": "Frequency of rebalancing"},
    )

    LAG_DAYS: int = field(
        default=1,
        metadata={"docs": "Number of days to lag for feature observation"},
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

    FACTORS: tuple[str] = field(
        default=("spx",),
        metadata={"docs": "Tradeable factors tuple"},
    )

    RF_NAME: str = field(
        default="acc_rate",
        metadata={"docs": "Risk-Free rate column name"},
    )

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}:"
        for key, value in self.__dict__.items():
            string += f"\n* {key} = {value}"
        return string

    def __repr__(self) -> str:
        return self.__str__()
