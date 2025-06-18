from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.preprocessing import StandardScaler

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class GPRLinearCovEstimator(BaseRLCovEstimator):
    def __init__(
        self, shrinkage_type: str, kernel=DotProduct(), window_size: int | None = None
    ) -> None:
        super().__init__(shrinkage_type=shrinkage_type, window_size=window_size)

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=12,
        )

        self.target_scaler = StandardScaler(with_std=False)

        self.last_pred = None
        self.encountered_nan = False

        self.shrinkage_mean = None

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        if shrinkage_target.isna().any():
            self.encountered_nan = True
        else:
            self.gpr.fit(X=features, y=shrinkage_target)
            self.encountered_nan = False

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        if not self.encountered_nan:
            pred = self.gpr.predict(features).item()
            pred = np.clip(pred, 0, 1)
            self.last_pred = pred
            return pred

        return self.last_pred
