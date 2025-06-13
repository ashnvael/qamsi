from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from qamsi.strategies.optimization_data import TrainingData

from enum import Enum


class WeightingScheme(Enum):
    EQUALLY_WEIGHTED = "equally_weighted"
    CAPITALIZATION_WEIGHTED = "cap_weighted"


class WeightingMixin:
    def __init__(self, weighting_scheme: str = "equally_weighted") -> None:
        self.weighting_scheme = WeightingScheme(weighting_scheme)

    def set_weights(
        self,
        data: TrainingData,
        weights_: pd.DataFrame,
        long_upper: list[str],
        short_lower: list[str],
    ) -> pd.DataFrame:
        if self.weighting_scheme == WeightingScheme.EQUALLY_WEIGHTED:
            weights_ = self.equal_weights(
                weights_=weights_,
                selection=long_upper,
                direction=1,
            )
            return self.equal_weights(
                weights_=weights_,
                selection=short_lower,
                direction=-1,
            )

        if self.weighting_scheme == WeightingScheme.CAPITALIZATION_WEIGHTED:
            weights_ = self.cap_weights(
                data=data,
                weights_=weights_,
                selection=long_upper,
                direction=1,
            )
            return self.cap_weights(
                data=data,
                weights_=weights_,
                selection=short_lower,
                direction=-1,
            )

        msg = f"Unknown weighting scheme: {self.weighting_scheme}"
        raise NotImplementedError(msg)

    @staticmethod
    def equal_weights(
        weights_: pd.DataFrame, selection: list[str], direction: int
    ) -> pd.DataFrame:
        if len(selection) == 0:
            return weights_
        weights_.loc[:, selection] = direction * 1 / len(selection)
        return weights_

    @staticmethod
    def cap_weights(
        data: TrainingData, weights_: pd.DataFrame, selection: list[str], direction: int
    ) -> pd.DataFrame:
        if len(selection) == 0:
            return weights_

        if data.market_cap is None:
            msg = "Market cap is not available"
            raise ValueError(msg)

        market_cap = data.market_cap.loc[:, selection]
        weights_.loc[:, selection] = direction * market_cap / market_cap.sum()

        return weights_
