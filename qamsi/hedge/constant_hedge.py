from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from qamsi.strategies.optimization_data import PredictionData, TrainingData

from qamsi.hedge.base_hedger import BaseHedger


class ConstantHedge(BaseHedger):
    def __init__(self, constant_weight: float | np.array, market_name: str = "MKT") -> None:
        super().__init__(market_name=market_name)

        self.constant_weight = constant_weight

    def _fit(self, training_data: TrainingData, hedge_assets: pd.DataFrame) -> None:
        pass

    def _get_weights(
        self,
        prediction_data: PredictionData,  # noqa: ARG002
        asset_weights: pd.DataFrame,  # noqa: ARG002
        hedge_weights_: pd.DataFrame,
    ) -> pd.DataFrame:
        hedge_weights_.loc[:, self.hedge_assets] = self.constant_weight

        return hedge_weights_
