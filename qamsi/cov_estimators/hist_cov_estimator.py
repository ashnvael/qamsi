from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd
from riskfolio import ParamsEstimation

from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class HistoricalCovEstimator(BaseCovEstimator):
    """
    A  wrapper around Riskfolio's covariance estimators.
    """

    def __init__(
        self,
        use_riskfolio: bool = True,
        method: str = "hist",
        d: float = 0.94,
        alpha: float = 0.1,
        bWidth: float = 0.01,
        detone: bool = False,
        mkt_comp: int = 1,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_riskfolio = use_riskfolio
        self.method = method
        self.d = d
        self.alpha = alpha
        self.bWidth = bWidth
        self.detone = detone
        self.mkt_comp = mkt_comp
        self.threshold = threshold

        self._fitted_cov = None

    def _fit(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None:

        if self.use_riskfolio:
            cov_array = ParamsEstimation.covar_matrix(
                X=targets,
                method=self.method,
                d=self.d,
                alpha=self.alpha,
                bWidth=self.bWidth,
                detone=self.detone,
                mkt_comp=self.mkt_comp,
                threshold=self.threshold,
            )
            idx = targets.columns
            self._fitted_cov = pd.DataFrame(cov_array, index=idx, columns=idx)

        else:
            self._fitted_cov = targets.cov()

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return self._fitted_cov
