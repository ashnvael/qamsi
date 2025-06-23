from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from qamsi.cov_estimators.rl.dl.dl_model import DeepLearningModel
from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class DLCovEstimator(BaseRLCovEstimator):
    def __init__(self, shrinkage_type: str, window_size: int | None = None) -> None:
        super().__init__(shrinkage_type=shrinkage_type, window_size=window_size)

        self.last_pred = None
        self.encountered_nan = False

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        if shrinkage_target.isna().any():
            self.encountered_nan = True
            print(
                f"{features.index.min()}-{features.index.max()}: Encountered NaN in shrinkage target."
            )
        else:
            self.dl_model = DeepLearningModel(n_features=features.shape[1])
            self.dl_model.fit(X=features, y=shrinkage_target)
            self.encountered_nan = False

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        if not self.encountered_nan:
            pred = self.dl_model.predict(features).item()
            self.last_pred = pred
            return pred

        return self.last_pred
