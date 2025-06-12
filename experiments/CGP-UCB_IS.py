#%%
from __future__ import annotations

#%%
import pandas as pd
from qamsi.config.trading_config import TradingConfig
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from qamsi.features.preprocessor import Preprocessor
from run import Dataset
#%%
REBAL_FREQ = "ME"
DATASET = Dataset.SPX_US
ESTIMATION_WINDOW = 365 * 1
#%%
experiment_config = DATASET.value()

stocks = tuple(
    pd.read_csv(experiment_config.PATH_OUTPUT / experiment_config.STOCKS_LIST_FILENAME)
    .iloc[:, 0]
    .astype(str)
    .tolist(),
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
    exclude_names=[
        *list(stocks),
        experiment_config.RF_NAME,
        *experiment_config.HEDGING_ASSETS,
        *factor_names,
        *prices,
    ],
)

trading_config = TradingConfig(
    broker_fee=0.05 / 100,
    bid_ask_spread=0.03 / 100,
    total_exposure=1,
    max_exposure=1,
    min_exposure=0,
    trading_lag_days=1,
)

runner = Runner(
    experiment_config=experiment_config,
    trading_config=trading_config,
    verbose=False,
)
#%%
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import fmin_l_bfgs_b, minimize

class ShrinkageGP:
    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        max_shrinkage: float = 1,
        max_iterations: int = 10,
        acq_beta: float = 1.96,
    ) -> None:
        self.start = start
        self.end = end
        self.acq_beta = acq_beta
        self._domain_ra = (0, max_shrinkage)
        self.max_iterations = max_iterations

        self.previous_points = []
        self.previous_targets = []

        self.optimal_shrinkage = None

        self.gpr = GaussianProcessRegressor(
            kernel=RBF(),
            n_restarts_optimizer=12,
            random_state=12,
        )

    def optimize(self, ra: float) -> float:
        estimator = CovEstimators.RISKFOLIO.value(
            alpha=ra,
            estimator_type="shrunk",
        )

        strategy = MinVariance(
            cov_estimator=estimator,
            trading_config=trading_config,
            window_size=ESTIMATION_WINDOW,
        )

        fitted_r = runner.run_one_step(
            start_date=self.start,
            end_date=self.end,
            feature_processor=preprocessor,
            strategy=strategy,
        )
        return -fitted_r.std().item()

    def next_recommendation(self) -> np.ndarray:
        return self.optimize_acquisition_function().item()

    def optimize_acquisition_function(self) -> np.ndarray:
        def objective(x: np.array):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        np.random.seed(12)
        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x_init = self._domain_ra[0] + (
                self._domain_ra[1] - self._domain_ra[0]
            ) * np.random.rand(1)
            result = fmin_l_bfgs_b(
                objective,
                x0=np.array([x_init]),
                bounds=[self._domain_ra],
                approx_grad=True,
            )
            x_values.append(result[0])
            f_values.append(result[1])

        ind = np.argmin(f_values)
        return x_values[ind]

    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        mu_f, sigma_f = self.gpr.predict(x.reshape(-1, 1), return_std=True)

        return (mu_f + sigma_f * self.acq_beta)[0]

    def add_data_point(self, x: float, acq_fn_val: float) -> None:
        self.previous_points.append(x)
        self.previous_targets.append(acq_fn_val)

        self.gpr.fit(
            np.array(self.previous_points).reshape(-1, 1),
            np.array(self.previous_targets),
        )

    def get_solution(self) -> tuple[float, np.ndarray]:
        idx = np.argmax(np.array(self.previous_targets))
        ra_init = self.previous_points[idx]

        def objective(x: np.array):
            x = np.array(x).reshape(-1, 1)
            mean, _ = self.gpr.predict(x, return_std=True)
            return -mean.item()

        np.random.seed(12)
        result = minimize(objective, x0=ra_init, bounds=[self._domain_ra])
        ra_opt = result.x.item()

        sharpe = self.optimize(ra_opt)
        self.optimal_sharpe = sharpe

        return ra_opt

    def solve(self) -> float:
        return self._run_search(self.max_iterations)

    def __call__(self):
        return self.solve()

    def _run_search(self, max_iter: int) -> float:
        for _ in range(max_iter):
            ra = self.next_recommendation()

            sharpe = self.optimize(ra)

            self.add_data_point(ra, sharpe)

        ra_opt = self.get_solution()

        return ra_opt

import gc

available_dates = runner.returns.simple_returns.loc["2014-11-03":].index

optimal = []
for i, date in enumerate(available_dates):
    start_date = date
    end_date = runner.returns.simple_returns.loc[start_date:].iloc[20:].index[0]

    opt = ShrinkageGP(
        start=start_date,
        end=end_date,
    )
    opt_ra = opt()
    print(f"Date: {start_date}, Volatility: {-opt.optimal_sharpe * np.sqrt(252):.6f}, Naive Volatility: {-opt.optimize(0.1) * np.sqrt(252):.6f}, Shrinkage: {opt_ra:.2f}")

    optimal.append([start_date, -opt.optimal_sharpe, -opt.optimize(0.1), opt_ra])

    if i % 20 == 0:
        optimal_df = pd.DataFrame(optimal, columns=["date", "vol", "naive_vol", "shrinkage"]).set_index("date")
        optimal_df.to_csv("targets.csv", index=True, header=True)

    gc.collect()
    del opt
