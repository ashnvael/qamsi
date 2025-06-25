# Semester Project // Quantitative Asset Management & Systematic Investments, Spring 2025
_Viacheslav Buchkov_, _Nikolai Kurlovich_\
University of Zürich

**Abstract**: De Nard and Kostovic (2025) present a novel approach to estimating the risk-optimized portfolio by learning the optimal shrinkage estimation, created by an "expert Oracle" in the supervised learning manner. Our paper shows that this Behavioral Cloning approach might be significantly improved by introducing Inverse Reinforcement Learning, where a policy is optimized to copy this "expert Oracle". Such an approach solves the issues of distributional shifts and cascading policy errors that are present in Behavior Cloning, resulting in suboptimal risk-optimized portfolios. Empirical experiments show that in fully out-of-sample construction our model improves on the existing results for linear shrinkage, providing a new data-driven approach to construct risk-optimized portfolios.

**Keywords**: Imitation learning; Inverse reinforcement learning; portfolio management reinforcement learning; risk optimization

## Install

```
git clone https://github.com/ashnvael/qamsi.git
```

## How To Create A New Covariance Estimator

```
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from qamsi.strategies.optimization_data import PredictionData, TrainingData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class NewCovEstimator(BaseCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._fitted_cov = None

    def _fit(self, training_data: TrainingData) -> None:
        # Specify here your fitting logic (e.g., store a historical covmat)
        ...

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        # Specify here your prediction logic (e.g., get a stored estimate)
        ...
```

## How To Create A New Data-Driven Covariance Estimator

```
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class NewRLCovEstimator(BaseRLCovEstimator):
    def __init__(self, shrinkage_type: str, window_size: int | None = None) -> None:
        super().__init__(shrinkage_type=shrinkage_type, window_size=window_size)

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        # Specify here your model (might be classical ML / RL / etc.)
        self.model = ...
        self.model.fit(X=features, y=shrinkage_target)

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        pred = self.model.predict(features).item()

        return pred
```

## How To Run A Backtest

```
from qamsi.config.trading_config import TradingConfig
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from run import Dataset, initialize

# Specify the model hyperparameters
REBAL_FREQ = "ME" # Rebalance frequency (Month End)
DATASET = Dataset.TOPN_US # Dataset (Top N largest stocks by Market Cap)
TOP_N = 30
ESTIMATION_WINDOW = 365 # Covariance matrix wstimation window in days
TRAINING_WINDOW = 365 * 20 # Model training window

# Initialize the RL Estimator from De Nard & Kostovic (2025)
ESTIMATOR = CovEstimators.DNK.value(shrinkage_type="linear", window_size=)

# Specify the trading configurations
trading_config = TradingConfig(
    broker_fee=0, # Broker commission in decimals
    bid_ask_spread=0, # Bid-ask spread in decimals
    total_exposure=1, # Budget constraint
    max_exposure=None, # Maximum weight constraint
    min_exposure=None, # Minimum weight constraint
    trading_lag_days=1, # Trading Lag
)

preprocessor, runner = initialize(
    dataset=DATASET,
    trading_config=trading_config,
    topn=TOP_N,
    rebal_freq=REBAL_FREQ,
)

# Create a Minimum Variance strategy with optimization
strategy = MinVariance(
    cov_estimator=ESTIMATOR,
    trading_config=trading_config,
    window_size=ESTIMATION_WINDOW,
)

strategy_name = ESTIMATOR.__class__.__name__

result = runner(
    feature_processor=preprocessor,
    strategy=strategy,
    hedger=None,
)
# Get the backtesting metrics
print(result)
```
