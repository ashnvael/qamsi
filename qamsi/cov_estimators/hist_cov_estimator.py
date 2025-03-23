from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class HistoricalCovEstimator(BaseCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._fitted_cov = None

    def _fit(self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame) -> None:
        self._fitted_cov = targets.cov()

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return self._fitted_cov
