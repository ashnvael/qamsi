from __future__ import annotations

import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class DNKCovEstimator(BaseRLCovEstimator):
    def __init__(
        self,
        shrinkage_type: str,
        window_size: int | None = None,
        lag_target: int | None = None,
    ) -> None:
        super().__init__(shrinkage_type=shrinkage_type, window_size=window_size)

        self.lag_target = lag_target

        self.last_pred = None
        self.encountered_nan = False

        self.selected_features = None

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        if shrinkage_target.isna().any():
            self.encountered_nan = True
            print(
                f"{features.index.min()}-{features.index.max()}: Encountered NaN in shrinkage target."
            )
        else:
            self.enet = ElasticNetCV(
                cv=TimeSeriesSplit(n_splits=5),
                alphas=[0.5, 1.0, 1.5, 2.0, 5.0],
                l1_ratio=[0.1, 0.25, 0.5, 0.75, 0.9],
            )

            if self.lag_target is not None:
                shrinkage_target = shrinkage_target.iloc[self.lag_target :]
                features = features.iloc[: -self.lag_target]

            self.enet.fit(X=features, y=shrinkage_target)
            self.encountered_nan = False

            if self.selected_features is None:
                self.selected_features = pd.DataFrame(
                    index=[features.index[-1]], columns=features.columns
                )

            self.selected_features.loc[features.index[-1], features.columns] = (
                self.enet.coef_ != 0.0
            )

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        if not self.encountered_nan:
            pred = self.enet.predict(features).item()
            self.last_pred = pred
            return pred

        return self.last_pred
