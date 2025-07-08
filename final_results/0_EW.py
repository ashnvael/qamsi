#%% md
# ## Equally-Weighted Portfolio Backtest.
#%%
from __future__ import annotations

import pandas as pd

#%%
from qamsi.dataset_builder_functions import build_jkp_dataset
from qamsi.runner import Runner
from qamsi.features.preprocessor import Preprocessor
from qamsi.config.trading_config import TradingConfig
from qamsi.strategies.heuristics.equally_weighted import EWStrategy
from run import Dataset
#%%
REBAL_FREQ = "ME"
dataset = Dataset.JKP

trading_config = TradingConfig(
    total_exposure=1,
    max_exposure=1,
    min_exposure=0,
    trading_lag_days=1,
)

experiment_config = dataset.value()

experiment_config.N_LOOKBEHIND_PERIODS = None
experiment_config.REBALANCE_FREQ = REBAL_FREQ

factors = pd.read_csv(experiment_config.PATH_OUTPUT / "factors.csv")
factors["date"] = pd.to_datetime(factors["date"])
factors = factors.set_index("date")
factor_names = tuple(factors.columns.astype(str).tolist())
experiment_config.FACTORS = factor_names

preprocessor = Preprocessor()

runner = Runner(
    dataset_builder_fn=lambda config: build_jkp_dataset(config, verbose=True),
    experiment_config=experiment_config,
    trading_config=trading_config,
    verbose=True,
)
#%%
strategy = EWStrategy()

strategy_name = strategy.__class__.__name__

result = runner(
    feature_processor=preprocessor,
    strategy=strategy,
    hedger=None,
)
print(result)
#%%
result.std_xs_r, result.sharpe
#%%
runner.plot_cumulative(
    strategy_name=strategy_name,
    include_factors=True,
)
#%%
runner.plot_turnover()
#%%
runner.plot_outperformance(mkt_only=True)
