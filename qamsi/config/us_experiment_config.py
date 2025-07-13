from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from qamsi.config.base_experiment_config import BaseExperimentConfig


@dataclass
class USExperimentConfig(BaseExperimentConfig):
    RANDOM_SEED: int = field(default=12, metadata={"docs": "Fix random seed"})

    # Experiment Settings
    START_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2000-12-18"),
        metadata={"docs": "Date to start training"},
    )

    END_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2024-07-31"),
        metadata={"docs": "Date to end train (as per paper by DNK)"},
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

    CAUSAL_WINDOW_SIZE: int | None = field(
        default=21,
        metadata={"docs": "Number of days that are not available at rebalancing"},
    )

    CAUSAL_WINDOW_END_DATE_FIELD: str | None = field(
        default="end_date",
        metadata={
            "docs": "Field name for last date, required for datapoint to be available. Overrides `CAUSIAL_WINDOW_SIZE` (!)"
        },
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
        default=(
            "low_risk",
            "momentum",
            "size",
            "quality",
            "value",
        ),
        metadata={"docs": "Tradeable factors tuple"},
    )

    TARGETS: tuple[str] = field(
        default=(
            "vol",
            "naive_vol",
            "target",
            "end_date",
        ),
        metadata={"docs": "ML Targets"},
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
