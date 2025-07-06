from __future__ import annotations

from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import fmin_l_bfgs_b, minimize

from qamsi.config.trading_config import TradingConfig
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from run import Dataset


DATASET = Dataset.SPX_US


def get_runner() -> Runner:
    experiment_config = DATASET.value()

    stocks = tuple(
        pd.read_csv(
            experiment_config.PATH_OUTPUT / experiment_config.STOCKS_LIST_FILENAME
        )
        .iloc[:, 0]
        .astype(str)
        .tolist(),
    )
    experiment_config.ASSET_UNIVERSE = stocks  # type: ignore  # noqa: PGH003

    experiment_config.START_DATE = "1981-01-02"
    experiment_config.MIN_ROLLING_PERIODS = 252
    experiment_config.REBALANCE_FREQ = "ME"

    factors = pd.read_csv(experiment_config.PATH_OUTPUT / "factors.csv")
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.set_index("date")
    factor_names = tuple(factors.columns.astype(str).tolist())
    experiment_config.FACTORS = factor_names

    trading_config = TradingConfig(
        total_exposure=1,
        max_exposure=1,
        min_exposure=0,
        trading_lag_days=1,
    )

    return Runner(
        experiment_config=experiment_config,
        trading_config=trading_config,
        verbose=False,
    )


class BanditEnvironment:
    def __init__(self, experiment_runner: Runner, start_date: pd.Timestamp) -> None:
        self.experiment_runner = experiment_runner

        self.features = self.experiment_runner.features
        # self.features = self.experiment_runner.factors.shift(1).fillna(0)

        self.start_dates = self.experiment_runner.returns.simple_returns.loc[
            start_date:
        ].index
        self.end_dates = [
            date + pd.tseries.offsets.BDay(n=21) for date in self.start_dates
        ]

        self.current_id = 0
        self.current_start = self.start_dates[self.current_id]
        self.current_end = self.end_dates[self.current_id]

        self._action_hist = []

    def __len__(self):
        return len(self.start_dates)

    def get_current_id(self):
        return self.current_id

    def get_current_date(self):
        return self.current_start

    def get_current_end_date(self):
        return self.current_end

    def reset(self):
        self.current_id = 0
        return self.get_context()

    def get_context(self) -> np.ndarray:
        return self.features.loc[self.current_start].fillna(0).to_numpy()

    def get_prediction_context(self, date: pd.Timestamp) -> np.ndarray:
        closest_date_index = self.features.index.asof(date)
        return self.features.loc[closest_date_index].fillna(0).to_numpy()

    def step(self, action: float) -> tuple[np.ndarray, float]:
        self._action_hist.append([self.current_end, action])

        estimator = CovEstimators.RISKFOLIO.value(
            alpha=action,
            estimator_type="shrunk",
        )

        strategy = MinVariance(
            cov_estimator=estimator,
            trading_config=self.experiment_runner.trading_config,
            window_size=365,
        )

        fitted_r = self.experiment_runner.run_one_step(
            start_date=self.current_start,
            end_date=self.current_end,
            feature_processor=None,
            strategy=strategy,
        )

        self.current_id += 1
        self.current_start = self.start_dates[self.current_id]
        self.current_end = self.end_dates[self.current_id]

        return self.get_context(), -fitted_r.std().item()

    @property
    def action_hist(self):
        return pd.DataFrame(self._action_hist, columns=["date", "cgp_ucb"]).set_index(
            "date"
        )


class CGPUCBAgent:
    N_RECENT_POINTS: int = 252

    def __init__(
        self,
        n_contexts: int = 1,
        n_actions: int = 100,
        exploration_mode: bool = False,
        max_shrinkage: float = 1,
        max_iterations: int = 10,
    ) -> None:
        self._domain_ra = (0, max_shrinkage)
        self.max_iterations = max_iterations

        self.previous_actions = []
        self.previous_contexts = []
        self.previous_targets = []

        self.optimal_shrinkage = None

        self.feat_scaler = StandardScaler()
        self.targets_scaler = StandardScaler(with_std=False)

        self.kernel1 = RBF()
        self.kernel2 = RBF()
        self.kernel = self.kernel1 * self.kernel2
        self.gpr = self._init_gpr()

        self._exploration_mode = exploration_mode

        self.current_round = 1
        self.X_cardinality = n_actions * n_contexts
        self.acq_beta = self.optimal_beta_selection(
            self.current_round, self.X_cardinality
        )

    @property
    def exploration_mode(self):
        return self._exploration_mode

    @exploration_mode.setter
    def exploration_mode(self, value: bool):
        self._exploration_mode = value

    def _init_gpr(self) -> GaussianProcessRegressor | BayesianRidge | None:
        return GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=0,
            optimizer=None,
            random_state=12,
        )

    @staticmethod
    def optimal_beta_selection(t: int, input_space_size: int, delta: float = 0.8):
        return 2 * np.log(input_space_size * (t**2) * (np.pi**2) / (6 * delta))

    def next_recommendation(self, context: np.ndarray) -> float:
        return self.optimize_acquisition_function(context).item()

    def get_action(self, context: np.ndarray) -> float:
        self.acq_beta = self.optimal_beta_selection(
            self.current_round, self.X_cardinality
        )
        # self.acq_beta = 1.96
        self.current_round += 1
        if self._exploration_mode:
            return self.next_recommendation(context)
        else:
            return self.get_solution(context)

    def optimize_acquisition_function(self, context: np.ndarray) -> np.ndarray:
        def objective(x: np.array):
            return -self.acquisition_function(x, context)

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

    def acquisition_function(self, a: np.ndarray, context: np.ndarray) -> np.ndarray:
        X = np.concatenate([a.reshape(1, 1), context.reshape(1, -1)], axis=1)

        X = self.feat_scaler.transform(X)

        mu_f, sigma_f = self.gpr.predict(X, return_std=True)
        mu_f = self.targets_scaler.inverse_transform(mu_f.reshape(-1, 1))[:, 0]

        return (mu_f + sigma_f * self.acq_beta)[0]

    def add_data_point(
        self, action: float, context: np.ndarray, acq_fn_val: float
    ) -> None:
        self.previous_actions.append(action)
        self.previous_contexts.append(context)
        self.previous_targets.append(acq_fn_val)

        # Create a new GP model if we have too many points to avoid slowdown
        if (
            len(self.previous_actions) >= self.N_RECENT_POINTS
        ):  # Threshold can be adjusted based on performance needs
            # Create a new GP model with the same parameters but trained on a subset of recent points
            # This prevents the model from becoming slower with each iteration
            recent_actions = self.previous_actions[
                -self.N_RECENT_POINTS :
            ]  # Keep the most recent 50 points
            recent_contexts = self.previous_contexts[-self.N_RECENT_POINTS :]
            recent_targets = self.previous_targets[-self.N_RECENT_POINTS :]
        else:
            recent_actions = self.previous_actions
            recent_contexts = self.previous_contexts
            recent_targets = self.previous_targets

        # recent_actions = self.previous_actions
        # recent_contexts = self.previous_contexts
        # recent_targets = self.previous_targets

        recent_actions = np.array(recent_actions).reshape(-1, 1)
        recent_contexts = np.array(recent_contexts)
        X = np.concatenate([recent_actions, recent_contexts], axis=1)
        X = self.feat_scaler.fit_transform(X)

        y = np.array(recent_targets)
        y = self.targets_scaler.fit_transform(y.reshape(-1, 1))[:, 0]

        self.gpr.fit(
            X,
            y,
        )

    def get_solution(self, context: np.ndarray) -> float:
        idx = np.argmax(np.array(self.previous_targets))
        alpha_init = self.previous_actions[idx]

        def objective(a: np.array):
            a = np.array(a)
            x = np.concatenate([a.reshape(1, 1), context.reshape(1, -1)], axis=1)
            x = self.feat_scaler.transform(x)
            mean, _ = self.gpr.predict(x, return_std=True)
            mean = self.targets_scaler.inverse_transform(mean.reshape(-1, 1))[:, 0]
            return -mean.item()

        np.random.seed(12)
        result = minimize(objective, x0=alpha_init, bounds=[self._domain_ra])
        alpha_opt = result.x.item()

        return alpha_opt


def train(start_date: str | None = "1982-01-01"):
    start_date = pd.Timestamp(start_date) if start_date else None

    runner = get_runner()
    rebal_dates = runner.init_backtester().rebal_schedule
    # lagged_rebal_dates = [date - pd.Timedelta(days=runner.trading_config.trading_lag_days) for date in rebal_dates]

    true_optimal = pd.read_csv("targets.csv")
    true_optimal["start_date"] = pd.to_datetime(true_optimal["start_date"])
    true_optimal = true_optimal.set_index("start_date")

    env = BanditEnvironment(runner, start_date=start_date)

    # initialize CGP-UCB
    agent = CGPUCBAgent(n_contexts=len(env), exploration_mode=True)
    rounds = len(env)
    context = env.reset()
    agent.add_data_point(0.0, context, 0.0)
    optim = []
    try:
        for i in (pbar := tqdm(range(rounds))):
            action = agent.get_action(context)
            context, reward = env.step(action)
            agent.add_data_point(action, context, reward)

            current_date = env.get_current_date()
            true_optimal_action = (
                true_optimal.loc[current_date, "shrinkage"]
                if current_date in true_optimal.index
                else 0.0
            )
            true_optimal_value = (
                true_optimal.loc[current_date, "vol"]
                if current_date in true_optimal.index
                else 0.0
            )

            pbar.set_description(
                f"Date: {current_date}, Action: {action:.6f}, Optimal Action: {true_optimal_action:.6f}, Reward: {reward:.6f}, Optimal Reward: {-true_optimal_value:.6f}"
            )

            if env.get_current_end_date() in rebal_dates:
                agent.exploration_mode = False
                rebal = env.get_current_end_date()
                pred_context = env.get_prediction_context(rebal)
                solution = agent.get_solution(pred_context)
                optim_solution = (
                    true_optimal.loc[current_date, "shrinkage"]
                    if current_date in true_optimal.index
                    else 0.0
                )
                # print(f"\nRebalance: {rebal}, Solution: {solution}, Optimal Solution: {optim_solution}")
                optim.append([rebal, solution])
            else:
                agent.exploration_mode = True

            if i % 20 == 0:
                env.action_hist.to_csv("cgp_ucb.csv", index=True, header=True)

        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_gmv.csv", index=True, header=True)
    except:
        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_gmv.csv", index=True, header=True)


if __name__ == "__main__":
    train()
