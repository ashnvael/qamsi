from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import fmin_l_bfgs_b, minimize

from qamsi.config.trading_config import TradingConfig
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from run import Dataset, initialize

REBAL_FREQ = "ME"
DATASET = Dataset.TOPN_US
TOP_N = 30
ESTIMATION_WINDOW = 365

trading_config = TradingConfig(
    total_exposure=1,
    max_exposure=None,
    min_exposure=None,
    trading_lag_days=1,
)

preprocessor, runner = initialize(
    dataset=DATASET,
    trading_config=trading_config,
    topn=TOP_N,
    rebal_freq=REBAL_FREQ,
)


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
        estimator = CovEstimators.QIS.value(shrinkage=ra)

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
from tqdm import tqdm

available_dates = runner.returns.simple_returns.loc["1982-01-01":].index

last_date = available_dates[-1]

optimal = []
i = 0
for date in available_dates:
    start_date = date
    end_date = date + pd.DateOffset(months=1)

    if end_date > last_date:
        break

    opt = ShrinkageGP(
        start=start_date,
        end=end_date,
    )
    opt_ra = opt()
    print(
        f"Start Date: {start_date}, End Date: {end_date}, Volatility: {-opt.optimal_sharpe * np.sqrt(252):.6f}, Naive Volatility: {-opt.optimize(0.1) * np.sqrt(252):.6f}, Shrinkage: {opt_ra:.2f}"
    )

    optimal.append(
        [start_date, end_date, -opt.optimal_sharpe, -opt.optimize(0.1), opt_ra]
    )

    if i % 20 == 0:
        optimal_df = pd.DataFrame(
            optimal, columns=["start_date", "end_date", "vol", "naive_vol", "shrinkage"]
        ).set_index("start_date")
        optimal_df.to_csv("targets.csv", index=True, header=True)

    gc.collect()
    del opt

    i += 1

optimal_df = pd.DataFrame(
    optimal, columns=["start_date", "end_date", "vol", "naive_vol", "shrinkage"]
).set_index("start_date")
optimal_df.to_csv("targets.csv", index=True, header=True)
