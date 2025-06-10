from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from qamsi.strategies.optimization_data import PredictionData, TrainingData

import numpy as np

from qamsi.features.ols_betas import get_window_betas
from qamsi.hedge.base_hedger import BaseHedger


class MarketFuturesHedge(BaseHedger):
    def __init__(self, market_name: str = "MKT", window_days: int = 365) -> None:
        super().__init__(market_name=market_name)

        self.window_days = window_days

        self._betas = None
        self.hedge_assets = None

    def _fit(self, training_data: TrainingData, hedge_assets: pd.DataFrame) -> None:  # noqa: ARG002
        self._betas = get_window_betas(
            market_index=training_data.factors[self.market_name],
            targets=training_data.simple_excess_returns,
            window_days=self.window_days,
        )

    def _get_weights(
        self,
        prediction_data: PredictionData,  # noqa: ARG002
        asset_weights: pd.DataFrame,
        hedge_weights_: pd.DataFrame,
    ) -> pd.DataFrame:
        hedge_weights = -np.dot(self._betas.to_numpy().T, asset_weights.to_numpy().T)
        hedge_weights_.loc[:, self.hedge_assets] = hedge_weights

        return hedge_weights_

    @property
    def betas(self) -> pd.DataFrame:
        return self._betas
