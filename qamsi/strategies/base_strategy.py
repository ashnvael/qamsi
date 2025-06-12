from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from qamsi.strategies.optimization_data import PredictionData, TrainingData


class BaseStrategy(ABC):
    """An abstract base class representing a generic financial investment strategy.

    The Strategy class defines a template for implementing custom strategies
    with methods for fitting models, making predictions, and determining
    portfolio weights.

    """

    def __init__(self) -> None:
        super().__init__()

        self.all_assets = None
        self.available_assets = None
        self._weights_template = None

    def fit(self, training_data: TrainingData) -> None:
        simple_xs_r = training_data.simple_excess_returns

        available_stocks = simple_xs_r.loc[
            :, ~simple_xs_r.iloc[-1].isna()
        ].columns.tolist()
        self.all_assets = simple_xs_r.columns.tolist()
        self.available_assets = simple_xs_r[available_stocks].columns.tolist()

        training_data.simple_excess_returns = (
            training_data.simple_excess_returns[available_stocks]
            if training_data.simple_excess_returns is not None
            else None
        )
        training_data.log_excess_returns = (
            training_data.log_excess_returns[available_stocks]
            if training_data.log_excess_returns is not None
            else None
        )

        self._fit(training_data=training_data)

    @abstractmethod
    def _fit(self, training_data: TrainingData) -> None:
        raise NotImplementedError

    def get_weights(self, prediction_data: PredictionData) -> pd.DataFrame:
        rebal_date = prediction_data.features.index[-1]
        init_weights = pd.DataFrame(0.0, index=[rebal_date], columns=self.all_assets)
        return self._get_weights(
            prediction_data=prediction_data, weights_=init_weights.copy()
        )

    @abstractmethod
    def _get_weights(
        self, prediction_data: PredictionData, weights_: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError

    def __call__(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self.get_weights(prediction_data=prediction_data)
