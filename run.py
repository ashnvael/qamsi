from __future__ import annotations

from enum import Enum

import pandas as pd
from qamsi.config.trading_config import TradingConfig
from qamsi.backtest.assessor import StrategyStatistics
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from qamsi.features.preprocessor import Preprocessor

from config.liquid_experiment_config import ExperimentConfig as LiquidConfig
from config.spx_experiment_config import ExperimentConfig as SPXConfig


class Dataset(Enum):
    LIQUID_US = LiquidConfig
    SPX_US = SPXConfig


REBAL_FREQ = "ME"
DATASET = Dataset.SPX_US

ESTIMATION_WINDOW = 365 * 5
ESTIMATOR = CovEstimators.RISKFOLIO.value()

SAVE = True

def run_backtest() -> StrategyStatistics:
    experiment_config = DATASET.value()

    stocks = tuple(
        pd.read_csv(experiment_config.PATH_OUTPUT / experiment_config.STOCKS_LIST_FILENAME).iloc[:, 0].astype(str).tolist(),
    )
    experiment_config.ASSET_UNIVERSE = stocks  # type: ignore  # noqa: PGH003

    experiment_config.MIN_ROLLING_PERIODS = ESTIMATION_WINDOW + 1
    experiment_config.N_LOOKBEHIND_PERIODS = None
    experiment_config.REBALANCE_FREQ = REBAL_FREQ

    factors = pd.read_csv(experiment_config.PATH_OUTPUT / "factors.csv")
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.set_index("date")
    factor_names = tuple(factors.columns.astype(str).tolist())
    experiment_config.FACTORS = factor_names

    prices = [stock + "_Price" for stock in list(stocks)]
    preprocessor = Preprocessor(
        exclude_names=[*list(stocks), experiment_config.RF_NAME, *experiment_config.HEDGING_ASSETS, *factor_names, *prices],
    )

    trading_config = TradingConfig(
        broker_fee=0.05 / 100,
        bid_ask_spread=0.03 / 100,
        total_exposure=1,
        max_exposure=1,
        min_exposure=0,
        trading_lag_days=1,
    )

    strategy = MinVariance(
        cov_estimator=ESTIMATOR,
        trading_config=trading_config,
        window_size=ESTIMATION_WINDOW,
    )

    runner = Runner(
        experiment_config=experiment_config,
        trading_config=trading_config,
        verbose=True,
    )

    result = runner(
        feature_processor=preprocessor,
        strategy=strategy,
        hedger=None,
    )

    strategy_name = ESTIMATOR.__class__.__name__

    if SAVE:
        runner.save(DATASET.name + strategy_name + f"_rebal{REBAL_FREQ}")

    runner.plot_cumulative(
        strategy_name=strategy_name,
        include_factors=True,
    )

    runner.plot_turnover()

    runner.plot_outperformance(mkt_only=True)

    return result


if __name__ == "__main__":
    run_result = run_backtest()

    print(run_result)  # noqa: T201
