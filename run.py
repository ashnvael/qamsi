from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamsi.runner import RunResult

import pandas as pd
from config.experiment_config import ExperimentConfig
from config.trading_config import TradingConfig
from qamsi.hedge.market_futures_hedge import MarketFuturesHedge
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.cov_estimators.cov_estimators import CovEstimators
from qamsi.strategies.heuristics.equally_weighted import EWStrategy
from qamsi.features.preprocessor import Preprocessor

HEDGE = False


def run_backtest(cov_estimator: BaseCovEstimator) -> RunResult:
    experiment_config = ExperimentConfig()
    stocks = tuple(pd.read_csv(experiment_config.PATH_OUTPUT / experiment_config.STOCKS_LIST_FILENAME).columns)
    experiment_config.ASSET_UNIVERSE = stocks  # type: ignore  # noqa: PGH003

    experiment_config.N_LOOKBEHIND_PERIODS = 252
    experiment_config.REBALANCE_FREQ_DAYS = 1

    trading_config = TradingConfig(
        broker_fee=0.05 / 100,
        bid_ask_spread=0.03 / 100,  # Taken as average bid-ask spread
        min_exposure=0,
    )

    runner = Runner(
        experiment_config=experiment_config,
        trading_config=trading_config,
        verbose=True,
    )

    hedger = MarketFuturesHedge()

    # Handles the features
    preprocessor = Preprocessor(exclude_names=[*list(stocks), "acc_rate", "spx"])

    strategy = MinVariance(
        cov_estimator=cov_estimator,
    )

    baseline_strategy = EWStrategy()

    run_result = runner.train(
        feature_processor=preprocessor,
        strategy=strategy,
        baseline_strategy=baseline_strategy,
        hedger=hedger if HEDGE else None,
    )

    runner.plot_cumulative(include_factors=True)

    runner.plot_turnover()

    runner.plot_returns_histogram_vs_baseline()

    return run_result


if __name__ == "__main__":
    estimator = CovEstimators.HISTORICAL.value()

    run_result = run_backtest(estimator)

    print(run_result.strategy)

    print("***")

    print(run_result.baseline)
