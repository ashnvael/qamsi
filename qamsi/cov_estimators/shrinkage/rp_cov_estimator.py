from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import riskfolio as rp

from qamsi.strategies.optimization_data import PredictionData, TrainingData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class RiskfolioCovEstimator(BaseCovEstimator):
    def __init__(self, estimator_type: str = "hist", alpha: float = 0.1) -> None:
        super().__init__()

        self.estimator_type = estimator_type
        self.alpha = alpha

        self._fitted_cov = None

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns
        # ret = ret.fillna(ret.mean(axis=0))

        self._fitted_cov = rp.ParamsEstimation.covar_matrix(
            ret,
            method=self.estimator_type,
            d=0.94,
            alpha=self.alpha,
            bWidth=0.01,
            detone=False,
            mkt_comp=1,
            threshold=0.5,
        )

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self._fitted_cov
