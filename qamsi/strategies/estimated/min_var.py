from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamsi.strategies.optimization_data import PredictionData, TrainingData

import pandas as pd

from qamsi.config.trading_config import TradingConfig
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.optimization.constraints import Constraints
from qamsi.optimization.optimization import VarianceMinimizer
from qamsi.strategies.base_strategy import BaseStrategy


class MinVariance(BaseStrategy):
    PERCENTAGE_VALID_POINTS = 1.0

    def __init__(
        self,
        cov_estimator: BaseCovEstimator,
        trading_config: TradingConfig,
        window_size: int,
    ) -> None:
        super().__init__()

        self.cov_estimator = cov_estimator
        self.trading_config = trading_config
        self.window_size = window_size

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns[self.available_assets]

        start_date = ret.index[-1] - pd.Timedelta(days=self.window_size)
        ret = ret.loc[start_date:]

        n_valid_points = (~ret.isna()).sum(axis=0) / len(ret)
        valid_stocks = list(
            n_valid_points[n_valid_points >= self.PERCENTAGE_VALID_POINTS].index
        )

        self.available_assets = valid_stocks
        self.cov_estimator.available_assets = self.available_assets

        training_data.simple_excess_returns = training_data.simple_excess_returns[
            self.available_assets
        ]
        training_data.simple_excess_returns = training_data.simple_excess_returns.loc[
            start_date:
        ]

        training_data.log_excess_returns = training_data.log_excess_returns.loc[
            start_date:, self.available_assets
        ] if training_data.log_excess_returns is not None else None
        training_data.targets = (
            training_data.targets.loc[start_date:, self.available_assets]
            if training_data.targets is not None
            else None
        )
        training_data.features = training_data.features.loc[start_date:]
        training_data.factors = training_data.factors.loc[start_date:]

        self.cov_estimator.fit(training_data)

    def _optimize(self, covmat: pd.DataFrame) -> pd.Series[float]:
        constraints = Constraints(ids=self.available_assets)

        constraints.add_box(
            lower=self.trading_config.min_exposure,
            upper=self.trading_config.max_exposure,
        )
        constraints.add_budget(rhs=self.trading_config.total_exposure, sense="=")

        self.var_min = VarianceMinimizer(constraints=constraints)

        self.var_min.set_objective(covmat=covmat)
        self.var_min.solve()

        return pd.Series(self.var_min.results["weights"])

    def _get_weights(
        self, prediction_data: PredictionData, weights_: pd.DataFrame
    ) -> pd.DataFrame:
        covmat = self.cov_estimator.predict(prediction_data)
        weights = self._optimize(covmat)

        weights_.loc[:, weights.index] = weights.to_numpy()

        # (!!!) Please, use only excess returns to apply this scaling correctly
        return weights_
