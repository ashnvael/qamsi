from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamsi.strategies.optimization_data import PredictionData, TrainingData

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseHedger(ABC):
    def __init__(self, market_name: str = "MKT") -> None:
        super().__init__()
        self._market_name = market_name

        self.hedge_assets = None

    def __call__(
        self, prediction_data: PredictionData, asset_weights: pd.DataFrame
    ) -> pd.DataFrame:
        return self.get_weights(
            prediction_data=prediction_data, asset_weights=asset_weights
        )

    def fit(self, training_data: TrainingData, hedge_assets: pd.DataFrame) -> None:
        self.hedge_assets = hedge_assets.columns.tolist()

        self._fit(training_data=training_data, hedge_assets=hedge_assets)

    @abstractmethod
    def _fit(self, training_data: TrainingData, hedge_assets: pd.DataFrame) -> None:
        raise NotImplementedError

    def get_weights(
        self, prediction_data: PredictionData, asset_weights: pd.DataFrame
    ) -> pd.DataFrame:
        rebal_date = prediction_data.features.index[-1]
        init_weights = pd.DataFrame(
            0.0, index=[rebal_date], columns=self.hedge_assets, dtype=np.float64
        )
        return self._get_weights(
            prediction_data=prediction_data,
            asset_weights=asset_weights,
            hedge_weights_=init_weights.copy(),
        )

    @abstractmethod
    def _get_weights(
        self,
        prediction_data: PredictionData,
        asset_weights: pd.DataFrame,
        hedge_weights_: pd.DataFrame,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def market_name(self) -> str:
        return self._market_name

    @market_name.setter
    def market_name(self, market_name: str) -> None:
        self._market_name = market_name
