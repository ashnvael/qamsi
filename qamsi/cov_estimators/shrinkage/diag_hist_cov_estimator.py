from __future__ import annotations


import numpy as np
import pandas as pd
from qamsi.strategies.optimization_data import PredictionData
from qamsi.cov_estimators.heuristic.hist_cov_estimator import HistoricalCovEstimator


class DiagHistoricalCovEstimator(HistoricalCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._fitted_cov = None

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        cov = super()._predict(prediction_data=prediction_data)
        return pd.DataFrame(np.diag(np.diag(cov)))
