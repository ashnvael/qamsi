from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.experiment_config import ExperimentConfig
    from config.trading_config import TradingConfig
    from qamsi.features.base_preprocessor import BasePreprocessor
    from qamsi.hedge.hedger import Hedger
    from qamsi.strategies.base_strategy import BaseStrategy

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qamsi.backtest.assessor import Assessor, StrategyStatistics
from qamsi.backtest.backtester import Backtester
from qamsi.backtest.plot import (
    plot_cumulative_pnls,
    plot_histogram,
    plot_histogram_vs_baseline,
    plot_turnover,
)
from qamsi.backtest.transaction_costs_charger import TransactionCostCharger
from qamsi.base.returns import Returns


@dataclass
class RunResult:
    strategy: StrategyStatistics
    baseline: StrategyStatistics


class Runner:
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        trading_config: TradingConfig,
        ml_metrics: list[Callable] | None = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        self.experiment_config = experiment_config
        self.trading_config = trading_config

        self.ml_metrics = ml_metrics
        self.verbose = verbose

        self.tc_charger = TransactionCostCharger(
            trading_config=self.trading_config,
        )

        self._prepare()

    def _prepare(self) -> None:
        data_df = pd.read_csv(
            self.experiment_config.PATH_OUTPUT / self.experiment_config.DF_FILENAME
        )
        data_df["date"] = pd.to_datetime(data_df["date"])
        data_df = data_df.set_index("date")

        self.train_data = data_df.loc[
            self.experiment_config.TRAIN_START_DATE : self.experiment_config.TRAIN_END_DATE
        ]
        self.test_data = data_df.loc[
            self.experiment_config.TRAIN_END_DATE : self.experiment_config.TEST_END_DATE
        ]

        if self.experiment_config.ASSET_UNIVERSE is None:
            self.experiment_config.ASSET_UNIVERSE = tuple(
                set(self.train_data.columns.tolist())
                - {self.experiment_config.RF_NAME}
                - set(self.experiment_config.FACTORS),
            )
        self.train_returns = Returns(
            self.train_data.loc[:, self.experiment_config.ASSET_UNIVERSE].iloc[1:]
        )
        self.test_returns = Returns(
            self.test_data.loc[:, self.experiment_config.ASSET_UNIVERSE].iloc[1:]
        )

        self.train_rf = self.train_data[[self.experiment_config.RF_NAME]].iloc[1:]
        self.test_rf = self.test_data[[self.experiment_config.RF_NAME]].iloc[1:]

        self.train_factors = self.train_data.loc[
            :, self.experiment_config.FACTORS
        ].iloc[1:]
        self.test_factors = self.test_data.loc[:, self.experiment_config.FACTORS].iloc[
            1:
        ]

        self.train_data = self.train_data.shift(1).iloc[1:]
        self.test_data = self.test_data.shift(1).iloc[1:]

        if self.verbose:
            print(
                f"Train data on {self.train_data.index.min()} to {self.train_data.index.max()}"
            )  # noqa: T201
            if len(self.test_data) > 0:
                print(
                    f"Test data on {self.test_data.index.min()} to {self.test_data.index.max()}"
                )  # noqa: T201
            print(f"Num Train Iterations: {len(self.train_data)}")  # noqa: T201

    def available_features(self) -> list[str]:
        return self.train_data.columns.tolist()

    def _run_backtest(  # noqa: PLR0913
        self,
        feature_processor: BasePreprocessor,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        returns: Returns,
        rf: pd.DataFrame,
        hedger: Hedger | None = None,
        hedging_assets: pd.DataFrame | None = None,
    ) -> Backtester:
        backtester = Backtester(
            stocks_returns=returns.truncate(feature_processor.truncation_len),
            features=feature_processor(data),
            rf_rate=rf.iloc[feature_processor.truncation_len :],
            tc_charger=self.tc_charger,
            trading_config=self.trading_config,
            n_lookback_periods=self.experiment_config.N_LOOKBEHIND_PERIODS,
            min_rolling_periods=self.experiment_config.MIN_ROLLING_PERIODS,
            rebal_freq_days=self.experiment_config.REBALANCE_FREQ_DAYS,
            verbose=self.verbose,
            hedging_assets=hedging_assets.truncate(feature_processor.truncation_len)
            if hedging_assets is not None
            else hedging_assets,
        )
        backtester(strategy, hedger)
        return backtester

    def _run(  # noqa: PLR0913
        self,
        feature_processor: BasePreprocessor,
        strategy: BaseStrategy,
        baseline_strategy: BaseStrategy,
        data: pd.DataFrame,
        returns: Returns,
        factors: pd.DataFrame,
        rf: pd.DataFrame,
        hedger: Hedger | None = None,
    ) -> RunResult:
        self.strategy_backtester = self._run_backtest(
            feature_processor,
            strategy,
            data,
            returns,
            rf,
            hedger,
            Returns(simple_returns=factors),
        )

        self.strategy_total_r = self.strategy_backtester.total_returns
        self.strategy_excess_r = self.strategy_backtester.excess_returns
        self.strategy_weights = self.strategy_backtester.weights
        self.strategy_turnover = self.strategy_backtester.turnover

        self.baseline_backtester = self._run_backtest(
            feature_processor,
            baseline_strategy,
            data,
            returns,
            rf,
            hedger,
            Returns(simple_returns=factors),
        )
        self.baseline_total_r = self.baseline_backtester.total_returns
        self.baseline_excess_r = self.baseline_backtester.excess_returns
        self.baseline_weights = self.baseline_backtester.weights

        self.factors_total_r = (
            self.strategy_backtester.acc_factors
            + self.strategy_backtester.acc_rf_rate.to_numpy().flatten()[:, np.newaxis]
        )

        assessor = Assessor(
            rf_rate=self.strategy_backtester.acc_rf_rate.iloc[:, 0],
            factors=self.strategy_backtester.acc_factors,
        )

        strategy_statistics = assessor(self.strategy_backtester.total_returns)
        baseline_statistics = assessor(self.baseline_backtester.total_returns)

        return RunResult(
            strategy=strategy_statistics,
            baseline=baseline_statistics,
        )

    def train(
        self,
        feature_processor: BasePreprocessor,
        strategy: BaseStrategy,
        baseline_strategy: BaseStrategy,
        hedger: Hedger | None = None,
    ) -> RunResult:
        return self._run(
            feature_processor=feature_processor,
            strategy=strategy,
            baseline_strategy=baseline_strategy,
            data=self.train_data,
            returns=self.train_returns,
            factors=self.train_factors,
            rf=self.train_rf,
            hedger=hedger,
        )

    def test(
        self,
        feature_processor: BasePreprocessor,
        strategy: BaseStrategy,
        baseline_strategy: BaseStrategy,
        hedger: Hedger | None = None,
    ) -> RunResult:
        return self._run(
            feature_processor=feature_processor,
            strategy=strategy,
            baseline_strategy=baseline_strategy,
            data=self.test_data,
            returns=self.test_returns,
            factors=self.test_factors,
            rf=self.test_rf,
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

    def plot_returns_histogram_vs_baseline(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> None:
        if start_date is None:
            start_date = self.strategy_total_r.index.min()

        if end_date is None:
            end_date = self.strategy_total_r.index.max()

        plot_histogram_vs_baseline(
            strategy_total=self.strategy_total_r.loc[start_date:end_date],
            baseline=self.baseline_total_r.loc[start_date:end_date],
        )

    def plot_cumulative(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        include_factors: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        if start_date is None:
            start_date = self.strategy_total_r.index.min()

        if end_date is None:
            end_date = self.strategy_total_r.index.max()

        plot_cumulative_pnls(
            strategy_total=self.strategy_total_r.loc[start_date:end_date],
            baseline=self.baseline_total_r.loc[start_date:end_date],
            buy_hold=self.factors_total_r.loc[start_date:end_date]
            if include_factors
            else None,
            plot_log=False,
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
