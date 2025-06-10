from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Data:
    features: pd.DataFrame | None = None

    prices: pd.DataFrame | None = None
    market_cap: pd.DataFrame | None = None
    factors: pd.DataFrame | None = None

    def add_features(self, new_features: pd.DataFrame) -> None:
        self.features = pd.concat([self.features, new_features], axis=0)

    def add_factors(self, new_factors: pd.DataFrame) -> None:
        self.factors = pd.concat([self.factors, new_factors], axis=0)

    def add_prices(self, new_prices: pd.DataFrame) -> None:
        self.prices = pd.concat([self.prices, new_prices], axis=0)

    def add_market_caps(self, new_market_caps: pd.DataFrame) -> None:
        self.market_cap = pd.concat([self.market_cap, new_market_caps], axis=0)


@dataclass
class TrainingData(Data):
    simple_excess_returns: pd.DataFrame | None = None
    log_excess_returns: pd.DataFrame | None = None

    targets: pd.DataFrame | None = None


@dataclass
class PredictionData(Data):
    pass
