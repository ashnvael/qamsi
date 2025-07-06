from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityTargeting:
    def __init__(
        self,
        window_days: int = int(252 / 2),  # By [Barroso & Santa-Clara, 2015]
        trading_lag: int = 1,
        min_exposure: float = 0.0,
        max_exposure: float = 1.0,
    ):
        self.window_days = window_days
        self.trading_lag = trading_lag

        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

    def transform(
        self,
        strategy_excess_r: pd.Series,
        rebal_dates: pd.DatetimeIndex,
        baseline: pd.Series | None = None,
        target_vol: float | None = None,
    ) -> pd.Series:
        if baseline is None and target_vol is None:
            msg = "Either baseline or target_vol must be provided."
            raise ValueError(msg)

        strategy_rv = self.get_rolling_rv(strategy_excess_r, rebal_dates)
        if baseline is not None:
            target = np.sqrt(self.get_rolling_rv(baseline, rebal_dates))
        else:
            target = target_vol

        scaling = target / np.sqrt(strategy_rv)
        scaling = scaling.fillna(1).clip(
            lower=self.min_exposure, upper=self.max_exposure
        )

        scaled_strategy = strategy_excess_r.to_frame("strategy_xs_r").merge(
            scaling.rename("weight"), how="left", left_index=True, right_index=True
        )
        scaled_strategy["weight"] = scaled_strategy["weight"].ffill()

        return scaled_strategy["strategy_xs_r"].multiply(
            scaled_strategy["weight"], axis=0
        )

    def __call__(
        self,
        strategy_excess_r: pd.Series,
        rebal_dates: pd.DatetimeIndex,
        baseline: pd.Series | None = None,
        target_vol: float | None = None,
    ):
        return self.transform(
            strategy_excess_r=strategy_excess_r,
            rebal_dates=rebal_dates,
            baseline=baseline,
            target_vol=target_vol,
        )

    def get_rolling_rv(
        self, strategy: pd.Series, rebal_dates: pd.DatetimeIndex
    ) -> pd.Series:
        day_diff = strategy.index.diff().days
        factor_annual = round(np.nanmean(365 // day_diff))

        rv = []
        for date in rebal_dates:
            rv_t = (
                strategy.loc[
                    date - pd.Timedelta(days=self.window_days + 1) : date
                    - pd.Timedelta(days=1)
                ]
                .pow(2)
                .mean()
            )
            rv.append([date, rv_t])

        rv = pd.DataFrame(rv, columns=["date", "rv"]).set_index("date")["rv"]
        return factor_annual * rv
