from __future__ import annotations

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import fmin_l_bfgs_b, minimize

from qamsi.config.trading_config import TradingConfig
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from qamsi.features.preprocessor import Preprocessor
from run import Dataset

REBAL_FREQ = 21
DATASET = Dataset.SPX_US
ESTIMATION_WINDOW = 365 * 1

experiment_config = DATASET.value()

stocks = tuple(
    pd.read_csv(experiment_config.PATH_OUTPUT / experiment_config.STOCKS_LIST_FILENAME)
    .iloc[:, 0]
    .astype(str)
    .tolist(),
)
experiment_config.ASSET_UNIVERSE = stocks  # type: ignore  # noqa: PGH003

experiment_config.START_DATE = "1980-01-01"
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


class ShrinkageGPOOS:
    def __init__(
        self,
        max_shrinkage: float = 1,
        max_iterations: int = 10,
        acq_beta: float = 1.96,
    ) -> None:
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

    def optimize(self, ra: float, start: pd.Timestamp, end: pd.Timestamp,) -> float:
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
            start_date=start,
            end_date=end,
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

        return ra_opt

available_dates = runner.returns.simple_returns.iloc[252:].index

last_date = available_dates[-1]

optimal = []
i = 0
opt = ShrinkageGPOOS()
for date in tqdm(available_dates):
    start_date = date
    end_date = runner.returns.simple_returns.loc[start_date:].iloc[20:].index[0]

    if end_date > last_date:
        break

    ra = opt.next_recommendation()

    sharpe = opt.optimize(ra, start_date, end_date)

    opt.add_data_point(ra, sharpe)

    ra_opt = opt.get_solution()

    optimal.append([end_date, ra_opt])

    if i % 20 == 0:
        optimal_df = pd.DataFrame(optimal, columns=["end_date", "shrinkage"]).set_index("end_date")
        optimal_df.to_csv("gp_ucb.csv", index=True, header=True)

    i += 1

optimal_df = pd.DataFrame(optimal, columns=["end_date", "shrinkage"]).set_index(
    "end_date"
)
optimal_df.to_csv("gp_ucb.csv", index=True, header=True)
