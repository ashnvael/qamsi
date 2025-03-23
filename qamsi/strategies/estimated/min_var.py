from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.optimization.constraints import Constraints
from qamsi.optimization.optimization import VarianceMinimizer
from qamsi.strategies.base_strategy import BaseStrategy


class MinVariance(BaseStrategy):
    def __init__(
        self,
        cov_estimator: BaseCovEstimator,
        max_exposure: float = 1.0,
        random_seed: int | None = None,
        ml_metrics: Optional[list[Callable]] = None,
    ) -> None:
        super().__init__(
            random_seed=random_seed,
            ml_metrics=ml_metrics,
        )

        self.cov_estimator = cov_estimator
        self.max_exposure = max_exposure

    def _fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        available_hist_stocks = set(
            targets.loc[:, ~targets.isna().any()].columns.tolist()
        )
        self.available_assets = list(set(self.available_assets) & available_hist_stocks)

        self.cov_estimator.fit(features, targets[self.available_assets])

    def _optimize(self, covmat: pd.DataFrame) -> dict[str, float]:
        constraints = Constraints(ids=self.available_assets)
        constraints.add_budget(rhs=self.max_exposure, sense="=")
        self.var_min = VarianceMinimizer(constraints=constraints)

        self.var_min.set_objective(covmat=covmat)
        self.var_min.solve()

        return self.var_min.results["weights"]

    def get_weights(self, features: pd.DataFrame) -> pd.DataFrame:
        covmat = self.cov_estimator.predict(features)
        weights = self._optimize(covmat)

        weights_df = pd.DataFrame(
            0.0, index=[features.index[-1]], columns=self.all_assets
        )
        weights_df.loc[:, self.available_assets] = weights

        # (!!!) Please, use only excess returns to apply this scaling correctly
        return weights_df
