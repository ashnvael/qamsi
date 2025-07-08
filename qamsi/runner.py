from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from qamsi.config.experiment_config import BaseExperimentConfig
    from qamsi.config.trading_config import TradingConfig
    from qamsi.features.preprocessor import Preprocessor
    from qamsi.hedge.base_hedger import BaseHedger
    from qamsi.strategies.base_strategy import BaseStrategy

import pandas as pd

from qamsi.backtest.assessor import Assessor, StrategyStatistics
from qamsi.backtest.backtester import Backtester
from qamsi.backtest.plot import (
    plot_cumulative_pnls,
    plot_histogram,
    plot_outperformance,
    plot_turnover,
)
from qamsi.backtest.transaction_costs_charger import TransactionCostCharger
from qamsi.base.returns import Returns
from qamsi.dataset_builder_functions import DatasetData, build_dataset


class Runner:
    def __init__(
        self,
        experiment_config: BaseExperimentConfig,
        trading_config: TradingConfig,
        dataset_builder_fn: Callable[
            [BaseExperimentConfig], DatasetData
        ] = build_dataset,
        ml_metrics: list[Callable] | None = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        self.dataset_builder_fn = dataset_builder_fn
        self.experiment_config = experiment_config
        self.trading_config = trading_config

        self.ml_metrics = ml_metrics
        self.verbose = verbose

        self.tc_charger = TransactionCostCharger(
            trading_config=self.trading_config,
        )

        self._is_hedged = None
        self.strategy_backtester = None
        self.strategy_total_r = None
        self.strategy_excess_r = None
        self.strategy_weights = None
        self.strategy_turnover = None

        self._prepare()

    def _prepare(self) -> None:
        dataset = self.dataset_builder_fn(self.experiment_config)
        asset_universe = dataset.presence_matrix.columns.tolist()

        self.data = dataset.data.loc[: self.experiment_config.END_DATE]
        self.data.columns = self.data.columns.astype(str)
        self.presence_matrix = dataset.presence_matrix.loc[
            : self.experiment_config.END_DATE
        ]

        if len(self.data) == 0:
            msg = "Backtesting data is empty!"
            raise ValueError(msg)

        # TODO(@V): Handle by BacktestBuilder on top
        # TODO(@V): Separate files
        prices_names = [stock + "_Price" for stock in asset_universe]
        if self.data.columns.isin(prices_names).any():
            self.prices = self.data.loc[:, prices_names]
            self.prices = self.prices.rename(
                columns={col: col.rstrip("_Price") for col in self.prices.columns}
            )
        else:
            self.prices = pd.DataFrame(index=self.data.index, columns=asset_universe)

        market_cap_names = [stock + "_Market_Cap" for stock in asset_universe]
        if self.data.columns.isin(market_cap_names).any():
            self.mkt_caps = self.data.loc[:, market_cap_names]
            self.mkt_caps = self.mkt_caps.rename(
                columns={col: col.rstrip("_Price") for col in self.mkt_caps.columns}
            )
        else:
            self.mkt_caps = pd.DataFrame(index=self.data.index, columns=asset_universe)

        self.returns = Returns(self.data.loc[:, asset_universe])
        self.rf = self.data[self.experiment_config.RF_NAME]

        self.targets = self.data[
            self.data.columns.intersection(set(self.experiment_config.TARGETS))
        ]

        # Factors are passed as excess returns
        self.factors = self.data.loc[:, self.experiment_config.FACTORS]

        # Hedging assets are passed as excess returns
        self.hedging_assets = (
            self.data.loc[:, self.experiment_config.HEDGING_ASSETS]
            if self.data.columns.isin(self.experiment_config.HEDGING_ASSETS).any()
            else pd.DataFrame(index=self.data.index)
        )

        exclude = [
            *asset_universe,
            *prices_names,
            *market_cap_names,
            self.experiment_config.RF_NAME,
            *self.experiment_config.FACTORS,
            *self.experiment_config.TARGETS,
            *self.experiment_config.HEDGING_ASSETS,
        ]
        self.features = self.data.drop(columns=exclude, errors="ignore")

        self.strategy_backtester = self.init_backtester()

    def available_features(self) -> list[str]:
        return self.features.columns.tolist()

    def init_backtester(self) -> Backtester:
        hedging_assets_ret = (
            Returns(simple_returns=self.hedging_assets)
            if self.hedging_assets is not None
            else self.hedging_assets
        )
        hedge_freq = (
            self.experiment_config.HEDGE_FREQ
            if self.experiment_config.HEDGE_FREQ is not None
            else self.experiment_config.REBALANCE_FREQ
        )

        return Backtester(
            start_date=self.experiment_config.START_DATE,
            end_date=self.experiment_config.END_DATE,
            stocks_returns=self.returns,
            features=self.features,
            targets=self.targets,
            prices=self.prices,
            mkt_caps=self.mkt_caps,
            rf=self.rf,
            factors=self.factors,
            tc_charger=self.tc_charger,
            trading_config=self.trading_config,
            n_lookback_periods=self.experiment_config.N_LOOKBEHIND_PERIODS,
            min_rolling_periods=self.experiment_config.MIN_ROLLING_PERIODS,
            rebal_freq=self.experiment_config.REBALANCE_FREQ,
            hedge_freq=hedge_freq,
            presence_matrix=self.presence_matrix,
            causal_window_size=self.experiment_config.CAUSAL_WINDOW_SIZE,
            verbose=self.verbose,
            hedging_assets=hedging_assets_ret,
        )

    def run(
        self,
        feature_processor: Preprocessor,
        strategy: BaseStrategy,
        hedger: BaseHedger | None = None,
    ) -> StrategyStatistics:
        if hedger is None:
            self._is_hedged = False
        else:
            self._is_hedged = True
            hedger.market_name = self.experiment_config.MKT_NAME

        self.strategy_backtester(strategy, hedger)

        self.strategy_total_r = self.strategy_backtester.strategy_total_r
        self.strategy_excess_r = self.strategy_backtester.strategy_excess_r
        self.strategy_weights = self.strategy_backtester.strategy_weights
        self.strategy_turnover = self.strategy_backtester.turnover

        start_date = self.strategy_total_r.index.min()
        end_date = self.strategy_total_r.index.max()

        assessor = Assessor(
            rf_rate=self.rf.loc[start_date:end_date],
            factors=self.factors.loc[start_date:end_date],
            mkt_name=self.experiment_config.MKT_NAME,
        )

        return assessor(self.strategy_total_r)

    def run_one_step(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        feature_processor: Preprocessor,
        strategy: BaseStrategy,
        hedger: BaseHedger | None = None,
    ) -> pd.DataFrame:
        return self.strategy_backtester.run_one_step(
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            hedger=hedger,
        )

    def __call__(
        self,
        feature_processor: Preprocessor,
        strategy: BaseStrategy,
        hedger: BaseHedger | None = None,
    ) -> StrategyStatistics:
        return self.run(
            feature_processor=feature_processor,
            strategy=strategy,
            hedger=hedger,
        )

    def plot_returns_histogram(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> None:
        if start_date is None:
            start_date = self.strategy_total_r.index.min()

        if end_date is None:
            end_date = self.strategy_total_r.index.max()

        plot_histogram(
            strategy_total=self.strategy_total_r.loc[start_date:end_date],
        )

    def plot_cumulative(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        include_factors: bool = False,  # noqa: FBT001, FBT002
        strategy_name: str | None = None,
    ) -> None:
        if start_date is None:
            start_date = self.strategy_total_r.index.min()

        if end_date is None:
            end_date = self.strategy_total_r.index.max()

        plot_cumulative_pnls(
            strategy_total=self.strategy_total_r.loc[start_date:end_date],
            buy_hold=self.factors.add(self.rf, axis=0).loc[start_date:end_date]
            if include_factors
            else None,
            plot_log=True,
            name_strategy=strategy_name if strategy_name is not None else "Strategy",
            mkt_name=self.experiment_config.MKT_NAME,
        )

    def plot_turnover(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> None:
        if start_date is None:
            start_date = self.strategy_total_r.index.min()

        if end_date is None:
            end_date = self.strategy_total_r.index.max()

        plot_turnover(self.strategy_turnover.loc[start_date:end_date])

    def plot_outperformance(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        mkt_only: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        if start_date is None:
            start_date = self.strategy_total_r.index.min()

        if end_date is None:
            end_date = self.strategy_total_r.index.max()

        strategy_total_r = self.strategy_total_r.loc[start_date:end_date]
        factors = self.factors.loc[start_date:end_date].add(
            self.rf.loc[start_date:end_date], axis=0
        )

        if mkt_only:
            plot_outperformance(
                strategy_total=strategy_total_r,
                baseline=factors[self.experiment_config.MKT_NAME],
                baseline_name=self.experiment_config.MKT_NAME,
            )
        else:
            for factor_name in factors.columns:
                plot_outperformance(
                    strategy_total=strategy_total_r,
                    baseline=factors[factor_name],
                    baseline_name=factor_name,
                )

    def save(self, strategy_name: str) -> None:
        if self.strategy_excess_r is None:
            msg = "Strategy is not backtested yet!"
            raise ValueError(msg)

        filename = strategy_name + ".csv"
        strategy_xs_r = self.strategy_excess_r.rename(
            columns={"excess_r": "strategy_xs_r"}
        )
        start, end = strategy_xs_r.index.min(), strategy_xs_r.index.max()
        factors = self.strategy_backtester.factors.loc[start:end]
        rf = self.strategy_backtester.rf.loc[start:end]

        rebal_bool = pd.Series(
            1, index=self.strategy_backtester.rebal_weights.index, name="rebal"
        )
        rebal_bool = (
            rebal_bool.reindex(self.strategy_weights.index).fillna(0).astype(bool)
        )

        sample = strategy_xs_r.merge(factors, left_index=True, right_index=True)
        sample = sample.merge(rf, left_index=True, right_index=True)
        sample = sample.merge(rebal_bool, left_index=True, right_index=True)

        sample.to_csv(self.experiment_config.SAVE_PATH / filename)
