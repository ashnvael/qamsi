from __future__ import annotations


import numpy as np
import pandas as pd
from qamsi.cov_estimators.hist_cov_estimator import HistoricalCovEstimator


class DiagHistoricalCovEstimator(HistoricalCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._fitted_cov = None

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        cov = super()._predict(features, factors)
        return pd.DataFrame(np.diag(np.diag(cov)))
