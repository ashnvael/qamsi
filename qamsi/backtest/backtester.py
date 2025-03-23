from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from IPython.display import clear_output

if TYPE_CHECKING:
    from config.trading_config import TradingConfig
    from qamsi.backtest.transaction_costs_charger import TransactionCostCharger
    from qamsi.base.returns import Returns
    from qamsi.hedge.hedger import Hedger
    from qamsi.strategies.base_strategy import BaseStrategy

import numpy as np
import pandas as pd
from tqdm import tqdm


class Backtester:
    def __init__(  # noqa: PLR0913, PLR0913, RUF100
        self,
        stocks_returns: Returns,
        features: pd.DataFrame,
        rf_rate: pd.DataFrame,
        hedging_assets: pd.DataFrame,
        tc_charger: TransactionCostCharger,
        trading_config: TradingConfig,
        n_lookback_periods: int = 30 * 12,
        min_rolling_periods: int | None = 12,
        rebal_freq_days: int = 20,
        verbose: bool = False,  # noqa: FBT001, FBT002
        plot: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        super().__init__()

        self.stocks_returns = stocks_returns
        self.features = features
        self.rf_rate = rf_rate
        self.hedging_assets = hedging_assets
        self.tc_charger = tc_charger
        self.trading_config = trading_config
        self.n_lookback_periods = n_lookback_periods
        self.min_rolling_periods = (
            n_lookback_periods if min_rolling_periods is None else min_rolling_periods
        )
        self.rebal_freq_days = rebal_freq_days
        self.verbose = verbose
        self.plot = plot

        self._stocks_total_r = None
        self._stocks_excess_r = None

        self._acc_rf = None
        self._acc_factors = None
        self._strategy_total_r = None
        self._strategy_excess_r = None
        self._strategy_weights = None
        self._hedge_weights = None
        self._trailing_betas = None
        self._strategy_turnover = None

        self._rolling_tuples = None

        self._prepare()

    def __call__(self, strategy: BaseStrategy, hedger: Hedger | None = None) -> None:
        self.run(strategy, hedger)

    def run(self, strategy: BaseStrategy, hedger: Hedger | None = None) -> None:  # noqa: PLR0915
        rolling_dates = []
        rolling_stocks_total_r = []
        rolling_stocks_xs_r = []
        rolling_acc_rf = []
        rolling_factors = []
        rolling_weights = []
        rolling_hedge_weights = []
        rolling_betas = []
        rolling_turnover = []
        previous_weights = pd.DataFrame(
            np.zeros((1, self.stocks_returns.simple_returns.shape[1]))
        )
        last_rebal_date = None

        if self.verbose and not self.plot:
            print("Running backtest...")
            self._rolling_tuples = tqdm(self._rolling_tuples)

        i = 0
        for iter_tuple in self._rolling_tuples:
            (
                rolling_features,
                rolling_simple_r,
                rolling_log_r,
                rolling_hedge_simple_r,
                rolling_hedge_log_r,
                rolling_rf,
            ) = iter_tuple

            current_date = rolling_simple_r.index[-1]

            if last_rebal_date is None:
                rebal = True
            else:
                rebal = (
                    current_date - last_rebal_date
                ).days >= self.rebal_freq_days + self.trading_config.trading_days_lag
            if rebal:
                # Each slice => n - 1 goes into training -> 1 last for predict
                train_features = rolling_features.iloc[:-1]
                pred_features = (rolling_features.iloc[-1]).to_frame().T

                train_factors = (
                    rolling_hedge_log_r.iloc[:-1] - rolling_rf.iloc[:-1].to_numpy()
                )
                pred_factors = (
                    (rolling_hedge_log_r.iloc[-1] - rolling_rf.iloc[-1].to_numpy())
                    .to_frame()
                    .T
                )

                train_xs_r = rolling_log_r.iloc[:-1] - rolling_rf.iloc[:-1].to_numpy()

                if last_rebal_date is not None:
                    agg_stocks_total_r = rolling_simple_r.loc[
                        last_rebal_date:current_date
                    ]
                    agg_stocks_total_r = agg_stocks_total_r.fillna(0)
                    agg_stocks_total_r = (
                        agg_stocks_total_r.add(1).prod(axis=0).add(-1).to_numpy()
                    )

                    agg_rf = (
                        rolling_rf.loc[last_rebal_date:current_date]
                        .add(1)
                        .prod()
                        .add(-1)
                        .to_numpy()
                        .item()
                    )
                    agg_stocks_xs_r = agg_stocks_total_r - agg_rf
                    agg_hedge_total_r = (
                        rolling_hedge_simple_r.loc[last_rebal_date:current_date]
                        .add(1)
                        .prod(axis=0)
                        .add(-1)
                        .to_numpy()
                    )
                    agg_hedge_xs_r = agg_hedge_total_r - agg_rf

                    rolling_stocks_total_r.append(
                        [current_date, *agg_stocks_total_r.tolist()]
                    )
                    rolling_stocks_xs_r.append(
                        [current_date, *agg_stocks_xs_r.tolist()]
                    )
                    rolling_acc_rf.append([current_date, agg_rf])
                    rolling_factors.append([current_date, *agg_hedge_xs_r.tolist()])

                # Whether the strategy has a memory or retrains from scratch is handled inside the strategy obj
                # Still pass all targets, not only tradeable, as can provide info for the strategy
                strategy.fit(
                    features=train_features, factors=train_factors, targets=train_xs_r
                )
                weights = strategy(features=pred_features, factors=pred_factors)

                weights = np.minimum(weights, self.trading_config.max_exposure)
                weights = np.maximum(weights, self.trading_config.min_exposure)

                if hedger is not None and self.hedging_assets is not None:
                    hedger.fit(
                        features=train_features,
                        targets=train_xs_r,
                        rf_rate=rolling_rf.iloc[:-1].to_numpy(),
                        hedge_assets=rolling_hedge_simple_r.iloc[:-1],
                    )
                    hedge_weights = hedger(features=pred_features, weights=weights)
                    rolling_hedge_weights.append(
                        [current_date, *hedge_weights.to_numpy().flatten().tolist()]
                    )

                    if hasattr(hedger, "betas"):
                        rolling_betas.append(
                            [current_date, *hedger.betas.to_numpy().flatten().tolist()]
                        )

                rolling_dates.append(current_date)
                rolling_weights.append(
                    [current_date, *weights.to_numpy().flatten().tolist()]
                )
                rolling_turnover.append(
                    [current_date]
                    + np.abs(weights - previous_weights.to_numpy())
                    .to_numpy()
                    .tolist()[0]
                )

                if self.plot and i != 0 and i % 20 == 0:
                    self._plot_progress(
                        dates=np.array(rolling_stocks_xs_r)[:, 0].tolist(),
                        rolling_weights=np.array(rolling_weights)[:-1, 1:],
                        rolling_hedge_weights=np.array(rolling_hedge_weights)[:-1, 1:]
                        if len(rolling_hedge_weights) > 0
                        else None,
                        stocks_excess_r=np.array(rolling_stocks_xs_r)[:, 1:],
                        factors_excess_r=np.array(rolling_factors)[:, 1:],
                        rf_rate=np.array(rolling_acc_rf)[:, 1:].flatten(),
                    )

                previous_weights = weights
                last_rebal_date = current_date
                i += 1

        rolling_weights = rolling_weights[:-1]
        rolling_turnover = rolling_turnover[:-1]
        if len(rolling_hedge_weights) > 0:
            rolling_hedge_weights = rolling_hedge_weights[:-1]

        stocks_columns = ["date", *list(self.stocks_returns.simple_returns.columns)]
        hedge_columns = ["date", *list(self.hedging_assets.simple_returns.columns)]
        self._stocks_total_r = pd.DataFrame(
            rolling_stocks_total_r, columns=stocks_columns
        ).set_index("date")
        self._stocks_excess_r = pd.DataFrame(
            rolling_stocks_xs_r, columns=stocks_columns
        ).set_index("date")
        self._acc_factors = pd.DataFrame(
            rolling_factors, columns=hedge_columns
        ).set_index("date")
        self._hedge_weights = pd.DataFrame(
            rolling_hedge_weights, columns=hedge_columns
        ).set_index("date")
        self._trailing_betas = pd.DataFrame(
            rolling_betas, columns=stocks_columns
        ).set_index("date")
        self._acc_rf = pd.DataFrame(rolling_acc_rf, columns=["date", "rf"]).set_index(
            "date"
        )

        self._strategy_weights = pd.DataFrame(
            rolling_weights, columns=stocks_columns
        ).set_index("date")
        self._strategy_turnover = pd.DataFrame(
            rolling_turnover, columns=stocks_columns
        ).set_index("date")

        strategy_excess_r = (
            self._strategy_weights.to_numpy() * self._stocks_excess_r.to_numpy()
        ).sum(axis=1)

        if len(rolling_hedge_weights) > 0:
            self._hedge_weights = pd.DataFrame(
                rolling_hedge_weights, columns=hedge_columns
            ).set_index("date")
            hedge_excess_r = (
                self._hedge_weights.to_numpy() * self._acc_factors.to_numpy()
            ).sum(axis=1)
            strategy_excess_r = strategy_excess_r + hedge_excess_r

        self._strategy_excess_r = pd.DataFrame(strategy_excess_r, columns=["excess_r"])
        self._strategy_excess_r["date"] = self._stocks_total_r.index
        self._strategy_excess_r = self._strategy_excess_r.set_index("date")

        strategy_transac_costs = self.tc_charger(
            weights=self._strategy_weights, returns=self._stocks_total_r
        )
        self._strategy_excess_r = (
            self._strategy_excess_r - strategy_transac_costs.to_numpy()[:, np.newaxis]
        )

        self._strategy_total_r = self._strategy_excess_r + self._acc_rf.to_numpy()
        self._strategy_total_r = self._strategy_total_r.rename(
            columns={"excess_r": "total_r"}
        )

    @staticmethod
    def _plot_progress(
        rolling_weights: np.ndarray,
        stocks_excess_r: np.ndarray,
        factors_excess_r: np.ndarray,
        rf_rate: np.ndarray,
        rolling_hedge_weights: np.ndarray | None = None,
        dates: list[pd.Timestamp] | None = None,
    ) -> None:
        clear_output()

        dates = np.arange(1, len(rf_rate) + 1) if dates is None else dates

        unhedged_strategy_excess_r = (rolling_weights * stocks_excess_r).sum(axis=1)

        # Chart 1 - NAV
        if rolling_hedge_weights is not None:
            fig, axs = plt.subplots(1, 3, figsize=(20, 8))

            hedge_excess_r = (rolling_hedge_weights * factors_excess_r).sum(axis=1)
            hedged_strategy_excess_r = unhedged_strategy_excess_r + hedge_excess_r

            hedged_strategy_total_r = hedged_strategy_excess_r + rf_rate
            axs[0].plot(dates, np.cumprod(1 + hedged_strategy_total_r), label="Hedged")

            # Chart 2 - Hedging weights
            axs[2].plot(dates, rolling_hedge_weights)

            axs[2].set_xlabel("Date")
            axs[2].set_ylabel("Hedge Weights")
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 20))

        unhedged_strategy_total_r = unhedged_strategy_excess_r + rf_rate
        axs[0].plot(dates, np.cumprod(1 + unhedged_strategy_total_r), label="Unhedged")

        factor_total_r = np.mean(factors_excess_r, axis=1) + rf_rate
        axs[0].plot(dates, np.cumprod(1 + factor_total_r), label="EW Factors")

        axs[0].set_xlabel("Date")
        axs[0].set_ylabel("Total NAV")
        axs[0].legend()

        # Chart 3 - Outperformance
        sim_rel = (
            unhedged_strategy_excess_r.flatten() - factors_excess_r.flatten()
        ) / (1 + factors_excess_r.flatten())
        axs[1].plot(dates, np.log(1 + sim_rel.astype(float)).cumsum())

        axs[1].set_xlabel("Date")
        axs[1].set_ylabel("Outperformance")

        plt.show()

    def _prepare(self) -> None:
        rolling_features = self.get_rolling_arrays(data_df=self.features)

        rolling_simple_r = self.get_rolling_arrays(
            data_df=self.stocks_returns.simple_returns
        )
        rolling_log_r = self.get_rolling_arrays(data_df=self.stocks_returns.log_returns)

        rolling_rf = self.get_rolling_arrays(data_df=self.rf_rate)

        if self.hedging_assets is not None:
            rolling_hedge_simple_r = self.get_rolling_arrays(
                data_df=self.hedging_assets.simple_returns
            )
            rolling_hedge_log_r = self.get_rolling_arrays(
                data_df=self.hedging_assets.log_returns
            )

            self._rolling_tuples = zip(
                rolling_features,
                rolling_simple_r,
                rolling_log_r,
                rolling_hedge_simple_r,
                rolling_hedge_log_r,
                rolling_rf,
                strict=False,
            )
        else:
            self._rolling_tuples = zip(
                rolling_features,
                rolling_simple_r,
                rolling_log_r,
                rolling_rf,
                strict=False,
            )

    def get_rolling_arrays(self, data_df: pd.DataFrame) -> list[pd.DataFrame]:
        # TODO @V: Expanding or not
        # TODO @V: start not from Test start, but predict on Test start with further lookbehind back
        rolling_windows = data_df.rolling(
            self.n_lookback_periods + 1, min_periods=self.min_rolling_periods + 1
        )
        return [
            window
            for window in rolling_windows
            if not window.empty and window.shape[0] >= self.min_rolling_periods + 1
        ]

    @property
    def total_returns(self) -> pd.Series:
        return self._strategy_total_r

    @property
    def stocks_total_r(self) -> pd.DataFrame:
        return self._stocks_total_r

    @property
    def stocks_excess_r(self) -> pd.DataFrame:
        return self._stocks_excess_r

    @property
    def acc_rf_rate(self) -> pd.DataFrame:
        return self._acc_rf

    @property
    def acc_factors(self) -> pd.DataFrame:
        return self._acc_factors

    @property
    def excess_returns(self) -> pd.Series:
        return self._strategy_excess_r

    @property
    def weights(self) -> pd.DataFrame:
        return self._strategy_weights

    @property
    def turnover(self) -> pd.Series:
        return self._strategy_turnover.sum(axis=1)

    @property
    def hedge_weights(self) -> pd.DataFrame:
        return self._hedge_weights

    @property
    def trailing_betas(self) -> pd.DataFrame:
        return self._trailing_betas
