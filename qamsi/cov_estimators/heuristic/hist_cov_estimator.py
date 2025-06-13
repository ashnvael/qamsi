from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from qamsi.strategies.optimization_data import PredictionData, TrainingData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class HistoricalCovEstimator(BaseCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._fitted_cov = None

    def _fit(self, training_data: TrainingData) -> None:
        self._fitted_cov = training_data.simple_excess_returns.cov()

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self._fitted_cov
