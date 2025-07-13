from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from qamsi.backtest.transaction_costs_charger import TransactionCostCharger
    from qamsi.base.returns import Returns
    from qamsi.config.trading_config import TradingConfig
    from qamsi.hedge.base_hedger import BaseHedger
    from qamsi.strategies.base_strategy import BaseStrategy

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from tqdm import tqdm

from qamsi.strategies.optimization_data import PredictionData, TrainingData


class Backtester:
    def __init__(  # noqa: PLR0913, PLR0913, RUF100
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        stocks_returns: Returns,
        targets: pd.DataFrame,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        mkt_caps: pd.DataFrame,
        rf: pd.Series,
        factors: pd.DataFrame,
        hedging_assets: Returns | None,
        tc_charger: TransactionCostCharger,
        trading_config: TradingConfig,
        n_lookback_periods: int,
        min_rolling_periods: int | None,
        rebal_freq: int | str | None,
        hedge_freq: int | str | None,
        presence_matrix: pd.DataFrame | None = None,
        causal_window_end_date_field: str | None = None,
        causal_window_size: int | None = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        super().__init__()

        self.start_date = start_date
        self.end_date = end_date

        self.stocks_returns = stocks_returns
        self.features = features
        self.targets = targets
        self.prices = prices
        self.mkt_caps = mkt_caps
        self.rf = rf
        self.factors = factors
        self.hedging_assets = hedging_assets
        self.tc_charger = tc_charger
        self.trading_config = trading_config
        self.trading_lag = trading_config.trading_lag_days
        self.n_lookback_periods = n_lookback_periods
        self.min_rolling_periods = (
            n_lookback_periods if min_rolling_periods is None else min_rolling_periods
        )
        self.rebal_freq = rebal_freq
        self.hedge_freq = hedge_freq
        self.presence_matrix = presence_matrix
        self.causal_window_end_date_field = causal_window_end_date_field
        if self.causal_window_end_date_field is not None:
            self.causal_window_size = None
        self.verbose = verbose

        self._strategy_total_r = None
        self._strategy_unhedged_total_r = None
        self._strategy_excess_r = None
        self._strategy_weights = None
        self._strategy_transac_costs = None

        self._rebal_weights = None
        self._first_rebal_date = None

        self._hedge_weights = None
        self._hedge_rebal_weights = None
        self._hedge_total_r = None
        self._hedge_excess_r = None

        self._rolling_strategy_tuples = None
        self._rolling_hedge_tuples = None

        self._prepare()

    def __call__(
        self, strategy: BaseStrategy, hedger: BaseHedger | None = None
    ) -> None:
        self.run(strategy, hedger)

    def generate_rebal_schedule(self, freq: int | str | None) -> pd.DatetimeIndex:
        features = self.features.loc[self.start_date : self.end_date]
        date_index = features.index

        if freq is None:
            schedule = date_index
        elif isinstance(freq, str):
            schedule = (
                features.groupby(date_index.to_period(freq.rstrip("E"))).tail(1).index
            )
        elif isinstance(freq, int):
            generated_dates = pd.date_range(
                start=date_index.min(), end=date_index.max(), freq=f"{freq}B"
            )

            closest_dates_indices = date_index.get_indexer(
                generated_dates, method="nearest"
            )

            schedule = date_index[closest_dates_indices]
        else:
            msg = f"Unknown rebalancing frequency type: {freq}."
            raise NotImplementedError(msg)

        if self.min_rolling_periods is not None:
            for i, date in enumerate(schedule):
                n_points = self.features.loc[:date].shape[0]
                if n_points >= self.min_rolling_periods:
                    schedule = schedule[i:]
                    break

        if schedule[-1] == self.end_date:
            schedule = schedule[:-1]

        if freq is None:
            schedule = schedule[:1]

        return schedule

    def _prepare(self) -> None:
        self.rebal_schedule = self.generate_rebal_schedule(freq=self.rebal_freq)
        self.hedge_schedule = self.generate_rebal_schedule(freq=self.hedge_freq)

        if self.verbose:
            print(
                f"Backtest on {self.rebal_schedule[0]} to {self.features.index.max()}"
            )  # noqa: T201
            print(f"Num Train Iterations: {len(self.rebal_schedule)}")  # noqa: T201
            print(
                f"Num OOS Daily Points: {len(self.features.loc[self.rebal_schedule[0] :])}"
            )

    @staticmethod
    def _accrue_returns(
        simple_returns: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Series:
        """Accrues simple returns from start_date to end_date (end date is included!)."""
        ret_series = simple_returns.loc[start_date:end_date]
        ret_series = ret_series.fillna(0)
        return ret_series.add(1).prod(axis=0).add(-1)

    def run(self, strategy: BaseStrategy, hedger: BaseHedger | None = None) -> None:
        strategy_weights = self.calc_rolling_weights(
            lambda pred_date: self.get_strategy_weights(
                strategy=strategy, pred_date=pred_date
            )
        )
        stocks_columns = ["date", *list(self.stocks_returns.simple_returns.columns)]
        strategy_weights = pd.DataFrame(
            strategy_weights, columns=stocks_columns
        ).set_index("date")

        self._rebal_weights = strategy_weights
        stocks_total_r = self.stocks_returns.simple_returns
        float_w_normalized, strategy_unhedged_total_r = self.float_weights(
            total_returns=stocks_total_r,
            weights=strategy_weights,
            rf=self.rf,
            add_total_r=None,
        )
        self._strategy_unhedged_total_r = strategy_unhedged_total_r
        start_date = strategy_unhedged_total_r.index.min()
        rf = self.rf.loc[start_date:]

        if hedger is not None:
            if self.verbose:
                print(f"Num Hedge Iterations: {len(self.hedge_schedule)}")  # noqa: T201

            hedger_weights = self.get_hedger_weights(hedger, float_w_normalized)
            hedge_float_w, strategy_hedged_total_r = self.float_weights(
                total_returns=self.hedging_assets.simple_returns.add(self.rf, axis=0),
                weights=hedger_weights,
                rf=self.rf,
                add_total_r=strategy_unhedged_total_r,
            )
            strategy_total_r = strategy_hedged_total_r

            self._hedge_weights = hedge_float_w
        else:
            strategy_total_r = strategy_unhedged_total_r

        self._strategy_weights = float_w_normalized

        strategy_transac_costs = self.tc_charger(
            weights=strategy_weights, returns=stocks_total_r.loc[start_date:]
        )
        strategy_total_r = strategy_total_r.sub(strategy_transac_costs, axis=0)

        strategy_excess_r = strategy_total_r.sub(rf, axis=0).rename(
            columns={"total_r": "excess_r"},
        )

        self._strategy_transac_costs = strategy_transac_costs
        self._strategy_total_r = strategy_total_r
        self._strategy_excess_r = strategy_excess_r

    def run_one_step(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        strategy: BaseStrategy,
        hedger: BaseHedger | None = None,
    ) -> pd.DataFrame:
        # TODO(@V): Add hedger (!!!)
        if hedger is not None:
            msg = "Hedged one step is not supported yet!"
            raise NotImplementedError(msg)

        pred_date = start_date - BusinessDay(n=self.trading_lag)

        weights = self.get_strategy_weights(strategy=strategy, pred_date=pred_date)
        strategy_weights = [[start_date, *weights.flatten().tolist()]]
        stocks_columns = ["date", *list(self.stocks_returns.simple_returns.columns)]
        strategy_weights = pd.DataFrame(
            strategy_weights, columns=stocks_columns
        ).set_index("date")

        stocks_total_r = self.stocks_returns.simple_returns.loc[:end_date]
        float_w_normalized, strategy_unhedged_total_r = self.float_weights(
            total_returns=stocks_total_r,
            weights=strategy_weights,
            rf=self.rf.loc[:end_date],
            add_total_r=None,
        )

        return strategy_unhedged_total_r

    def get_data(self, pred_date: pd.Timestamp) -> tuple[TrainingData, PredictionData]:
        # Each slice => n - lag goes into training -> 1 last for predict
        available_features = self.features.loc[:pred_date]
        train_features = available_features.shift(1).iloc[1:]
        pred_features = (available_features.iloc[-1]).to_frame().T

        available_prices = self.prices.loc[:pred_date]
        train_prices = available_prices.shift(1).iloc[1:]
        pred_prices = (available_prices.iloc[-1]).to_frame().T

        available_mkt_caps = self.mkt_caps.loc[:pred_date]
        train_mkt_caps = available_mkt_caps.shift(1).iloc[1:]
        pred_mkt_caps = (available_mkt_caps.iloc[-1]).to_frame().T

        train_factors = self.factors.loc[:pred_date].iloc[1:]
        train_targets = self.targets.loc[:pred_date]

        if self.causal_window_end_date_field is not None:
            train_targets[self.causal_window_end_date_field] = pd.to_datetime(
                train_targets[self.causal_window_end_date_field]
            )
            train_targets = train_targets.reset_index().set_index(
                self.causal_window_end_date_field
            )
            train_targets = train_targets[train_targets.index <= pred_date]
            train_targets = (
                train_targets.reset_index()
                .drop(columns=[self.causal_window_end_date_field])
                .set_index("date")
            )
        elif self.causal_window_size is not None:
            train_targets = train_targets.iloc[: -self.causal_window_size]

        train_targets = train_targets.iloc[1:]
        train_rf = self.rf.loc[:pred_date].iloc[1:]
        simple_train_xs_r = (
            self.stocks_returns.simple_returns.loc[:pred_date]
            .iloc[1:]
            .sub(train_rf, axis=0)
        )
        log_train_xs_r = (
            self.stocks_returns.log_returns.loc[:pred_date]
            .iloc[1:]
            .sub(train_rf, axis=0)
        )

        training_data = TrainingData(
            features=train_features,
            targets=train_targets,
            prices=train_prices,
            market_cap=train_mkt_caps,
            factors=train_factors,
            simple_excess_returns=simple_train_xs_r,
            log_excess_returns=log_train_xs_r,
        )

        prediction_data = PredictionData(
            features=pred_features,
            prices=pred_prices,
            market_cap=pred_mkt_caps,
        )

        return training_data, prediction_data

    def get_strategy_weights(
        self, strategy: BaseStrategy, pred_date: pd.Timestamp
    ) -> np.array:
        if self.presence_matrix is not None:
            curr_matrix = self.presence_matrix.loc[:pred_date].iloc[-1]
            strategy.universe = curr_matrix[curr_matrix == 1].index.tolist()

        training_data, prediction_data = self.get_data(pred_date)

        # Whether the strategy has a memory or retrains from scratch is handled inside the strategy obj
        strategy.fit(training_data=training_data)

        weights = strategy(prediction_data=prediction_data)
        weights = np.clip(
            weights,
            self.trading_config.min_exposure,
            self.trading_config.max_exposure,
        )
        return weights.to_numpy()

    def calc_rolling_weights(
        self, get_weights_fn: Callable[[pd.Timestamp], np.ndarray[float]]
    ) -> list[np.ndarray]:
        rolling_weights = []
        last_rebal_date = None
        n_rebals = 0
        for rebal_date in tqdm(
            self.rebal_schedule, desc="Computing Weights", disable=not self.verbose
        ):
            if last_rebal_date is None:
                self._first_rebal_date = rebal_date

            if self.rebal_freq is None:
                should_rebal = last_rebal_date is None
            elif isinstance(self.rebal_freq, int | float):
                if last_rebal_date is None:
                    should_rebal = True
                else:
                    n_days_change = (
                        (rebal_date - last_rebal_date).days
                        if last_rebal_date is not None
                        else 0
                    )
                    should_rebal = n_days_change >= self.rebal_freq
            elif isinstance(self.rebal_freq, str):
                should_rebal = rebal_date >= self.rebal_schedule[n_rebals]
            else:
                msg = f"Unknown rebalancing frequency type: {self.rebal_freq}."
                raise NotImplementedError(msg)

            if should_rebal:
                pred_date = rebal_date - BusinessDay(n=self.trading_lag)

                weights = get_weights_fn(pred_date)

                rolling_weights.append([rebal_date, *weights.flatten().tolist()])

                last_rebal_date = rebal_date
                n_rebals += 1

        return rolling_weights

    def get_hedger_weights(
        self, hedger: BaseHedger, strategy_weights: pd.DataFrame
    ) -> pd.DataFrame:
        # TODO(@V): Deprecate and use calc_rolling_weights(lambda pred_date: get_hedger_weights(...))
        rolling_weights = []
        last_hedge_date = None
        n_hedges = 0
        for rebal_date in tqdm(
            self.hedge_schedule, desc="Hedging", disable=not self.verbose
        ):
            if rebal_date < self._first_rebal_date:
                should_hedge = False
            elif self.hedge_freq is None:
                should_hedge = last_hedge_date is None
            elif isinstance(self.hedge_freq, int | float):
                if last_hedge_date is None:
                    should_hedge = True
                else:
                    n_days_change = (
                        (rebal_date - last_hedge_date).days
                        if last_hedge_date is not None
                        else 0
                    )
                    should_hedge = n_days_change >= self.hedge_freq
            elif isinstance(self.hedge_freq, str):
                should_hedge = rebal_date == self.rebal_schedule[n_hedges]
            else:
                msg = f"Unknown hedging frequency type: {self.hedge_freq}."
                raise NotImplementedError(msg)

            if should_hedge:
                period_end_weights = strategy_weights.loc[rebal_date].copy()
                picked_assets = period_end_weights[period_end_weights != 0].index

                pred_date = rebal_date - BusinessDay(n=self.trading_lag)

                # Each slice => n - lag goes into training -> 1 last for predict
                available_features = self.features.loc[:pred_date]
                train_features = available_features.iloc[:-1]
                pred_features = (available_features.iloc[-1]).to_frame().T

                available_prices = self.prices.loc[:pred_date]
                train_prices = available_prices.iloc[:-1]
                pred_prices = (available_prices.iloc[-1]).to_frame().T

                available_mkt_caps = self.mkt_caps.loc[:pred_date]
                train_mkt_caps = available_mkt_caps.iloc[:-1]
                pred_mkt_caps = (available_mkt_caps.iloc[-1]).to_frame().T

                train_factors = self.factors.loc[:pred_date].iloc[1:]
                train_rf = self.rf.loc[:pred_date].iloc[1:]

                simple_train_xs_r = (
                    self.stocks_returns.simple_returns.loc[:pred_date]
                    .iloc[1:]
                    .sub(train_rf, axis=0)
                )
                log_train_xs_r = (
                    self.stocks_returns.log_returns.loc[:pred_date]
                    .iloc[1:]
                    .sub(train_rf, axis=0)
                )

                train_hedging_assets_r = self.hedging_assets.simple_returns.loc[
                    :pred_date
                ].iloc[1:]

                training_data = TrainingData(
                    features=train_features,
                    prices=train_prices[picked_assets]
                    if len(train_prices) > 0
                    else train_prices,
                    market_cap=train_mkt_caps[picked_assets]
                    if len(train_mkt_caps) > 0
                    else train_mkt_caps,
                    factors=train_factors,
                    simple_excess_returns=simple_train_xs_r[picked_assets],
                    log_excess_returns=log_train_xs_r[picked_assets],
                )

                prediction_data = PredictionData(
                    features=pred_features,
                    prices=pred_prices[picked_assets]
                    if len(pred_prices) > 0
                    else pred_prices,
                    market_cap=pred_mkt_caps[picked_assets]
                    if len(pred_mkt_caps) > 0
                    else pred_mkt_caps,
                )

                hedger.fit(
                    training_data=training_data,
                    hedge_assets=train_hedging_assets_r,
                )
                hedge_weights = hedger(
                    prediction_data=prediction_data,
                    asset_weights=period_end_weights.loc[picked_assets],
                )

                rolling_weights.append(
                    [rebal_date, *hedge_weights.to_numpy().flatten().tolist()]
                )

                last_hedge_date = rebal_date
                n_hedges += 1

        hedge_columns = ["date", *list(self.hedging_assets.simple_returns.columns)]
        self._hedge_rebal_weights = pd.DataFrame(
            rolling_weights, columns=hedge_columns
        ).set_index("date")

        return self._hedge_rebal_weights

    @staticmethod
    def float_weights(
        total_returns: pd.DataFrame,
        weights: pd.DataFrame,
        rf: pd.Series,
        add_total_r: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        total_r = pd.DataFrame(
            index=total_returns.index, columns=["total_r"], dtype=np.float64
        )
        level_data = pd.DataFrame(
            index=total_returns.index, columns=["level_data"], dtype=np.float64
        )
        float_weights = pd.DataFrame(
            index=total_returns.index, columns=[*total_returns.columns.tolist()]
        )

        total_returns = (
            pd.concat([total_returns, rf], axis=1)
            if add_total_r is None
            else pd.concat([total_returns, add_total_r, rf], axis=1)
        )
        n_auxilary_cols = 1 if add_total_r is None else 2
        weights = weights.copy()
        if add_total_r is not None:
            weights["add"] = 1
        weights["rf"] = (1 - weights.sum(axis=1)).round(5)

        last_rebal_date = weights.index[0]

        total_r = total_r.loc[last_rebal_date:]
        float_weights = float_weights.loc[last_rebal_date:]
        level_data = level_data.loc[last_rebal_date:]

        total_r.loc[last_rebal_date] = np.float64(0.0)
        float_weights.loc[last_rebal_date] = np.float64(0.0)
        for rebal in [*weights.index[1:].tolist(), None]:
            w0 = weights.loc[last_rebal_date]
            start_date = last_rebal_date
            end_date = rebal if rebal else None

            sample_r = total_returns.loc[start_date:end_date].copy().fillna(0)
            r_mat = 1 + sample_r
            r_mat.iloc[0] = w0.fillna(0)
            float_w = r_mat.cumprod(axis=0).fillna(0)

            level = float_w.sum(axis=1)

            ret_tmp = level.pct_change(1).iloc[1:]

            total_r.loc[ret_tmp.index] = ret_tmp.to_frame()
            # TODO(@V): Check .div() by level for long-short
            normalized = float_w.iloc[:, :-n_auxilary_cols].div(level, axis=0)
            float_weights.loc[sample_r.index, :] = normalized
            level_data.loc[level.index] = level.to_frame()

            last_rebal_date = rebal

        return float_weights, total_r

    @property
    def strategy_total_r(self) -> pd.Series:
        return self._strategy_total_r

    @property
    def strategy_unhedged_total_r(self) -> pd.Series:
        return self._strategy_unhedged_total_r

    @property
    def strategy_excess_r(self) -> pd.Series:
        return self._strategy_excess_r

    @property
    def strategy_weights(self) -> pd.DataFrame:
        return self._strategy_weights

    @property
    def rebal_weights(self) -> pd.DataFrame:
        return self._rebal_weights

    @property
    def strategy_transaction_costs(self) -> pd.DataFrame:
        return self._strategy_transac_costs

    @property
    def hedge_total_r(self) -> pd.Series:
        return self._hedge_total_r

    @property
    def hedge_excess_r(self) -> pd.Series:
        return self._hedge_excess_r

    @property
    def turnover(self) -> pd.Series:
        return self.tc_charger.turnover

    @property
    def hedge_weights(self) -> pd.DataFrame:
        return self._hedge_weights

    @property
    def hedge_rebal_weights(self) -> pd.DataFrame:
        return self._hedge_rebal_weights
