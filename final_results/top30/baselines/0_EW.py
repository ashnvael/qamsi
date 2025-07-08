#%% md
# ## Equally-Weighted Portfolio Backtest.
#%%
from __future__ import annotations

#%%
from qamsi.config.trading_config import TradingConfig
from qamsi.strategies.heuristics.equally_weighted import EWStrategy
from run import Dataset, initialize
#%%
REBAL_FREQ = "ME"
DATASET = Dataset.TOPN_US
TOP_N = 30

trading_config = TradingConfig(
    total_exposure=1,
    max_exposure=1,
    min_exposure=0,
    trading_lag_days=1,
)

preprocessor, runner = initialize(
    dataset=DATASET,
    with_causal_window=False,
    trading_config=trading_config,
    topn=TOP_N,
    rebal_freq=REBAL_FREQ,
)
#%%
strategy = EWStrategy()

strategy_name = strategy.__class__.__name__

result = runner(
    feature_processor=preprocessor,
    strategy=strategy,
    hedger=None,
)
result
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
#%%
runner.save(DATASET.name + strategy_name + f"_rebal{REBAL_FREQ}")