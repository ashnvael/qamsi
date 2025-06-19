from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class RandomForestCovEstimator(BaseRLCovEstimator):
    SHRINKAGE_PRECISION = 100

    def __init__(self, shrinkage_type: str, trading_lag: int = 0, as_classifier: bool = False, window_size: int | None = None) -> None:
        super().__init__(shrinkage_type=shrinkage_type, window_size=window_size)

        self.trading_lag = trading_lag
        self.as_classifier = as_classifier

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
            if self.trading_lag > 0:
                features = features.shift(self.trading_lag).iloc[self.trading_lag:]
                shrinkage_target = shrinkage_target.iloc[self.trading_lag:]

            if self.as_classifier:
                discretizer = KBinsDiscretizer(n_bins=self.SHRINKAGE_PRECISION, encode="ordinal", strategy="uniform")
                shrinkage_transf = discretizer.fit_transform(shrinkage_target.values.reshape(-1, 1)).ravel()
                shrinkage_target = pd.Series(shrinkage_transf, index=shrinkage_target.index).astype(int)

                self.rf = RandomForestClassifier(
                    n_estimators=30,
                    max_depth=10,
                    random_state=12,
                )
                self.rf.fit(X=features, y=shrinkage_target)
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
