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
from config.topn_experiment_config import ExperimentConfig as TopNConfig


class Dataset(Enum):
    LIQUID_US = LiquidConfig
    SPX_US = SPXConfig
    TOPN_US = TopNConfig


REBAL_FREQ = "ME"

DATASET = Dataset.SPX_US

ESTIMATION_WINDOW = 365 * 3
ESTIMATOR = CovEstimators.GLASSO.value()

SAVE = True

TRADING_CONFIG = TradingConfig(
    broker_fee=0.05 / 100,
    bid_ask_spread=0.03 / 100,
    total_exposure=1,
    max_exposure=1,
    min_exposure=0,
    trading_lag_days=1,
)


def initialize(
        dataset: Dataset,
        start: str | None = None,
        end: str | None = None,
        trading_config: TradingConfig = TRADING_CONFIG,
        rebal_freq: str = REBAL_FREQ,
        topn: int | None = None,
) -> tuple[Preprocessor, Runner]:
    experiment_config = dataset.value(topn=topn) if topn else dataset.value()

    experiment_config.N_LOOKBEHIND_PERIODS = None
    experiment_config.REBALANCE_FREQ = rebal_freq

    if start is not None:
        experiment_config.START_DATE = pd.Timestamp(start)
    if end is not None:
        experiment_config.END_DATE = pd.Timestamp(end)

    factors = pd.read_csv(experiment_config.PATH_OUTPUT / "factors.csv")
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.set_index("date")
    factor_names = tuple(factors.columns.astype(str).tolist())
    experiment_config.FACTORS = factor_names

    preprocessor = Preprocessor()

    runner = Runner(
        experiment_config=experiment_config,
        trading_config=trading_config,
        verbose=True,
    )

    return preprocessor, runner


def run_backtest() -> StrategyStatistics:
    print("Running backtest...")
    print(f"Estimation window: {ESTIMATION_WINDOW}")

    preprocessor, runner = initialize(DATASET, TRADING_CONFIG, topn=100)
    trading_config = TRADING_CONFIG

    strategy = MinVariance(
        cov_estimator=ESTIMATOR,
        trading_config=trading_config,
        window_size=ESTIMATION_WINDOW,
    )

    result = runner(
        feature_processor=preprocessor,
        strategy=strategy,
        hedger=None,
    )

    strategy_name = ESTIMATOR.__class__.__name__

    if SAVE:
        runner.save(f"{DATASET.name}_" + strategy_name + f"_rebal{REBAL_FREQ}")

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
