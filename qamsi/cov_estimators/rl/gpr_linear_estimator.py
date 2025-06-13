from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class GPRLinearCovEstimator(BaseRLCovEstimator):
    def __init__(self, shrinkage_type: str, kernel=DotProduct()) -> None:
        super().__init__(shrinkage_type=shrinkage_type)

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=12,
            random_state=12,
        )

        self.last_pred = None
        self.encountered_nan = False

        self.shrinkage_mean = None

    def _transform_shrinkage_target(self, shrinkage_target: pd.Series) -> pd.Series:
        self.shrinkage_mean = shrinkage_target.mean()
        shrinkage_target = shrinkage_target - self.shrinkage_mean
        # return np.log(shrinkage_target) - np.log(1 - shrinkage_target)
        return shrinkage_target

    def _inv_transform_shrinkage_target(self, shrinkage_target: float) -> float:
        # return 1 / (1 + np.exp(-shrinkage_target)) + self.shrinkage_mean
        return shrinkage_target + self.shrinkage_mean

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        shrinkage_target = self._transform_shrinkage_target(shrinkage_target)
        if shrinkage_target.isna().any():
            self.encountered_nan = True
        else:
            self.gpr.fit(X=features, y=shrinkage_target)
            self.encountered_nan = False

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        if not self.encountered_nan:
            pred = self.gpr.predict(features).item()
            pred = self._inv_transform_shrinkage_target(pred)
            pred = np.clip(pred, 0, 1)
            self.last_pred = pred
            return pred

        return self.last_pred
