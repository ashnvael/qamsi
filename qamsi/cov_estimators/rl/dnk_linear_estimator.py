from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class DNKLinearCovEstimator(BaseRLCovEstimator):
    def __init__(self, shrinkage_type: str) -> None:
        super().__init__(shrinkage_type=shrinkage_type)

        self.enet = ElasticNetCV(
            cv=TimeSeriesSplit(n_splits=5),
            alphas=[0.5, 1.0, 1.5, 2.0, 5.0],
            l1_ratio=[0.1, 0.25, 0.5, 0.75, 0.9],
            # max_iter=[500, 1500, 2000],
            # tol=[1e-3, 1e-4, 1e-5],
        )

        self.last_pred = None
        self.encountered_nan = False

    @staticmethod
    def _transform_shrinkage_target(shrinkage_target: pd.Series) -> pd.Series:
        shrinkage_target = shrinkage_target + 1e-9
        return np.log(shrinkage_target) - np.log(1 - shrinkage_target)

    @staticmethod
    def _inv_transform_shrinkage_target(shrinkage_target: float) -> float:
        return 1 / (1 + np.exp(-shrinkage_target))

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        shrinkage_target = self._transform_shrinkage_target(shrinkage_target)
        if shrinkage_target.isna().any():
            self.encountered_nan = True
        else:
            self.enet.fit(X=features, y=shrinkage_target)
            self.encountered_nan = False

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        if not self.encountered_nan:
            pred = self.enet.predict(features).item()
            pred = self._inv_transform_shrinkage_target(pred)
            self.last_pred = pred
            return pred

        return self.last_pred
