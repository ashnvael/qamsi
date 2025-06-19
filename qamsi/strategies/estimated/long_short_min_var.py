from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamsi.strategies.optimization_data import PredictionData, TrainingData

import pandas as pd

from qamsi.config.trading_config import TradingConfig
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.strategies.base_strategy import BaseStrategy
from qamsi.strategies.heuristics.equally_weighted import EWStrategy
from qamsi.strategies.estimated.min_var import MinVariance


class LongShortMinVariance(BaseStrategy):
    PERCENTAGE_VALID_POINTS = 1.0

    def __init__(
        self,
        cov_estimator: BaseCovEstimator,
        trading_config: TradingConfig,
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.ew_strategy = EWStrategy()
        self.min_var_strategy = MinVariance(
            cov_estimator=cov_estimator,
            trading_config=trading_config,
            window_size=window_size,
        )

    def _fit(self, training_data: TrainingData) -> None:
        self.ew_strategy.fit(training_data)
        self.min_var_strategy.fit(training_data)

    def _get_weights(
        self, prediction_data: PredictionData, weights_: pd.DataFrame
    ) -> pd.DataFrame:
        ew_weights = self.ew_strategy.get_weights(prediction_data)
        mv_weights = self.min_var_strategy.get_weights(prediction_data)

        ls_weights = mv_weights.to_numpy() - ew_weights.to_numpy()
        weights_.loc[:, self.available_assets] = ls_weights

        return weights_
