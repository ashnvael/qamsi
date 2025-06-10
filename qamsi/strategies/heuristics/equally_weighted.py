from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qamsi.strategies.base_strategy import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd

    from qamsi.strategies.optimization_data import PredictionData, TrainingData


class EWStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()

    def _fit(self, training_data: TrainingData) -> None:
        pass

    def _get_weights(
        self, prediction_data: PredictionData, weights_: pd.DataFrame
    ) -> pd.DataFrame:  # noqa: ARG002
        n_assets = len(self.available_assets)
        weights_.loc[:, self.available_assets] = (
            np.ones((1, n_assets), dtype=float) / n_assets
        )
        return weights_
