from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class BaseExperimentConfig:
    RANDOM_SEED: int = field(default=12, metadata={"docs": "Fix random seed"})

    # Folders
    PATH_OUTPUT: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "output",
        metadata={"docs": "Relative path to data folder"},
    )

    SAVE_PATH: Path = field(
        default=Path(__file__).resolve().parents[1] / "backtests" / "runs",
        metadata={"docs": "Relative path to data folder"},
    )

    # Filename
    DF_FILENAME: str = field(
        default="data_df.csv", metadata={"docs": "Preprocessed data"}
    )

    # Experiment Settings
    START_DATE: pd.Timestamp | None = field(
        default=pd.to_datetime("1980-01-01"),
        metadata={"docs": "Date to start training"},
    )

    END_DATE: pd.Timestamp | None = field(
        default=pd.to_datetime("2024-01-01"),
        metadata={"docs": "Date to end train"},
    )

    REBALANCE_FREQ: int | str | None = field(
        default=21,
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

    N_LOOKBEHIND_PERIODS: int | None = field(
        default=None,
        metadata={
            "docs": "Number of rebalance periods to take into rolling regression"
        },
    )

    MIN_ROLLING_PERIODS: int = field(
        default=12,
        metadata={"docs": "Number of minimum rebalance periods to run the strategy"},
    )

    CAUSAL_WINDOW_SIZE: int | None = field(
        default=None,
        metadata={
            "docs": "Number of datapoints that are not available at rebalancing"
        },
    )

    # Universe Setting
    ASSET_UNIVERSE: tuple[str] = field(
        default=("MOEX_INDEX",),
        metadata={"docs": "Tradeable assets tuple"},
    )

    FACTORS: tuple[str] = field(
        default=("MOEX_INDEX",),
        metadata={"docs": "Tradeable factors tuple"},
    )

    TARGETS: tuple[str] = field(
        default=(),
        metadata={"docs": "ML Targets"},
    )

    HEDGING_ASSETS: tuple[str] = field(
        default=("spx_fut",),
        metadata={"docs": "Tradeable assets tuple"},
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
