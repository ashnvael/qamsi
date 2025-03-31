from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
from qamsi.cov_estimators.hist_cov_estimator import HistoricalCovEstimator


class DiagHistoricalCovEstimator(HistoricalCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._fitted_cov = None

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        cov = super().predict(features, factors)
        return pd.DataFrame(np.diag(np.diag(cov)))
