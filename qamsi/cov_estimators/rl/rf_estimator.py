from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class RandomForestCovEstimator(BaseRLCovEstimator):
    def __init__(self, shrinkage_type: str, window_size: int | None = None) -> None:
        super().__init__(shrinkage_type=shrinkage_type, window_size=window_size)

        self.last_pred = None
        self.encountered_nan = False

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        if shrinkage_target.isna().any():
            self.encountered_nan = True
            print(f"{features.index.min()}-{features.index.max()}: Encountered NaN in shrinkage target.")
        else:
            self.rf = RandomForestRegressor(
                n_estimators=30,
                max_depth=10,
                random_state=12,
            )
            self.rf.fit(X=features, y=shrinkage_target)
            self.encountered_nan = False

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        if not self.encountered_nan:
            pred = self.rf.predict(features).item()
            self.last_pred = pred
            return pred

        return self.last_pred
