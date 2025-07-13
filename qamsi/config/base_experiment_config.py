from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class BaseExperimentConfig:
    PREFIX: str = field(
        default="crsp_",
        metadata={"docs": "Prefix for all output files to define dataset"},
    )

    RANDOM_SEED: int = field(default=12, metadata={"docs": "Fix random seed"})

    # Folders
    CRSP_PATH: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "crsp_raw",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_FACTORS: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "jkp_raw",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_RF_RATE: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "ff",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_MKT: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "gw_replication",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_HEDGING_ASSETS: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "hedge",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_TMP: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "tmp",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_OUTPUT: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "output",
        metadata={"docs": "Relative path to data folder"},
    )

    SAVE_PATH: Path = field(
        default=Path(__file__).resolve().parents[2] / "final_results" / "runs",
        metadata={"docs": "Relative path to data folder"},
    )

    # Filename
    CRSP_FULL_TMP_FILENAME: str = field(
        default="crsp_full_tmp.csv",
        metadata={"docs": "File with full CRSP stock returns data for reloading"},
    )

    TRADE_DATASET_TMP_FILENAME: str = field(
        default="crsp_factors_mkt_tmp.csv",
        metadata={"docs": "File with full CRSP stock returns data for reloading"},
    )

    CRSP_FILENAME: str = field(
        default="crsp_80s.csv", metadata={"docs": "File with CRSP stock returns data"}
    )

    RAW_DATA_FILENAME: str = field(
        default="raw_data.csv",
        metadata={"docs": "Initial filtered data for temporary storing"},
    )

    FACTORS_FILENAME: str = field(
        default="jkp_factors.csv", metadata={"docs": "Tradeable factors file"}
    )

    RF_RATE_FILENAME: str = field(
        default="FFDaily.xlsx", metadata={"docs": "Rf rate file"}
    )

    MKT_FILENAME: str = field(
        default="spx.xlsx", metadata={"docs": "Market index file"}
    )

    HEDGING_ASSETS_FILENAME: str = field(
        default="spx_fut.xlsx", metadata={"docs": "Tradeable hedging assets file"}
    )

    DF_FILENAME: str = field(
        default="data_df.csv", metadata={"docs": "Preprocessed data"}
    )

    DNK_FEATURES_TMP_FILENAME: str = field(
        default="dnk_features_tmp.csv",
        metadata={"docs": "Temporary storage of pre-computed features"},
    )

    PRESENCE_MATRIX_FILENAME: str | None = field(
        default="presence_matrix.csv", metadata={"docs": "Presence matrix (2d pivot)"}
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
        default=1,
        metadata={"docs": "Number of minimum rebalance periods to run the strategy"},
    )

    CAUSAL_WINDOW_SIZE: int | None = field(
        default=None,
        metadata={"docs": "Number of datapoints that are not available at rebalancing"},
    )

    CAUSAL_WINDOW_END_DATE_FIELD: str | None = field(
        default=None,
        metadata={
            "docs": "Field name for last date, required for datapoint to be available. Overrides `CAUSIAL_WINDOW_SIZE` (!)"
        },
    )

    # Universe Setting
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
        default=(),
        metadata={"docs": "ML Targets"},
    )

    HEDGING_ASSETS: tuple[str] = field(
        default=("spx_fut",),
        metadata={"docs": "Tradeable hedging assets tuple"},
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
