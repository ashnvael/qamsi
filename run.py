from __future__ import annotations

from enum import Enum

import pandas as pd
from qamsi.config.trading_config import TradingConfig
from qamsi.backtest.assessor import StrategyStatistics
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.cov_estimators.cov_estimators import CovEstimators
from qamsi.features.preprocessor import Preprocessor
from qamsi.dataset_builder_functions import build_dnk_dataset

from qamsi.config.spx_experiment_config import SPXExperimentConfig as SPXConfig
from qamsi.config.topn_experiment_config import TopNExperimentConfig as TopNConfig
from qamsi.config.jkp_experiment_config import JKPExperimentConfig as JKPConfig


class Dataset(Enum):
    SPX_US = SPXConfig
    TOPN_US = TopNConfig
    JKP = JKPConfig


REBAL_FREQ = "ME"

TOP_N = 30
DATASET = Dataset.TOPN_US
SAVE = True
ESTIMATION_WINDOW = 365 * 1

TRADING_CONFIG = TradingConfig(
    broker_fee=0.05 / 100,
    bid_ask_spread=0.03 / 100,
    total_exposure=1,
    max_exposure=None,
    min_exposure=None,
    trading_lag_days=1,
)


def initialize(
    dataset: Dataset,
    with_causal_window: bool = True,
    start: str | None = None,
    end: str | None = None,
    trading_config: TradingConfig = TRADING_CONFIG,
    rebal_freq: str = REBAL_FREQ,
    topn: int | None = None,
    verbose: bool = True,
) -> tuple[Preprocessor, Runner]:
    experiment_config = dataset.value(topn=topn) if topn else dataset.value()

    experiment_config.N_LOOKBEHIND_PERIODS = None
    experiment_config.REBALANCE_FREQ = rebal_freq

    if not with_causal_window:
        experiment_config.CAUSAL_WINDOW_SIZE = None

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
        dataset_builder_fn=lambda config: build_dnk_dataset(config, verbose=verbose),
        experiment_config=experiment_config,
        trading_config=trading_config,
        verbose=verbose,
    )

    return preprocessor, runner


def run_backtest(
        estimator: BaseCovEstimator,
        dataset: Dataset = DATASET,
        rebal_freq: str = REBAL_FREQ,
        trading_config: TradingConfig = TRADING_CONFIG,
        estimation_window: int = ESTIMATION_WINDOW,
        topn: int | None = None,
        save: bool = SAVE,
) -> StrategyStatistics:
    preprocessor, runner = initialize(
        dataset=dataset,
        with_causal_window=True,
        trading_config=trading_config,
        rebal_freq=rebal_freq,
        topn=topn,
    )

    strategy = MinVariance(
        cov_estimator=estimator,
        trading_config=trading_config,
        window_size=estimation_window,
    )

    print("Running backtest...")
    print(f"Estimation window: {estimation_window}")

    result = runner(
        feature_processor=preprocessor,
        strategy=strategy,
        hedger=None,
    )

    strategy_name = estimator.__class__.__name__

    if save:
        runner.save(f"{dataset.name}_{topn}_" + strategy_name + f"_rebal{rebal_freq}")

    runner.plot_cumulative(
        strategy_name=strategy_name,
        include_factors=True,
    )

    runner.plot_turnover()

    runner.plot_outperformance(mkt_only=True)

    return result


if __name__ == "__main__":
    estimator = CovEstimators.DNK.value(shrinkage_type="linear", window_size=365 * 20)
    run_result = run_backtest(estimator=estimator)

    print(run_result)  # noqa: T201
