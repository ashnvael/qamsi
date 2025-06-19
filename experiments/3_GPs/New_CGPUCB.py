from __future__ import annotations

from tqdm import tqdm
from IPython.display import clear_output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import BayesianRidge
from scipy.optimize import minimize, fmin_l_bfgs_b

from qamsi.config.trading_config import TradingConfig
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from run import Dataset, initialize


REBAL_FREQ = "ME"
DATASET = Dataset.TOPN_US
TOP_N = 30
ESTIMATION_WINDOW = 365

TRADING_CONFIG = TradingConfig(
    total_exposure=1,
    max_exposure=None,
    min_exposure=None,
    trading_lag_days=1,
)


class BanditEnvironment:
    def __init__(
        self,
        experiment_runner: Runner,
        start_date: pd.Timestamp,
        train_end: pd.Timestamp,
        min_reward: float | None = None,
        max_reward: float | None = None,
    ) -> None:
        self.experiment_runner = experiment_runner
        self.min_reward = min_reward
        self.max_reward = max_reward

        scaler = StandardScaler()

        self.features = self.experiment_runner.features.drop(
            columns=["irl"], errors="ignore"
        )
        scaler.fit(self.features.loc[self.features.index < train_end])
        self.features = scaler.transform(self.features)
        self.features = pd.DataFrame(
            self.features,
            columns=self.experiment_runner.features.columns,
            index=self.experiment_runner.features.index,
        )

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

    def __len__(self) -> int:
        return len(self.start_dates)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    def get_current_id(self) -> int:
        return self.current_id

    def get_current_date(self) -> pd.Timestamp:
        return self.current_start

    def get_current_end_date(self) -> pd.Timestamp:
        return self.current_end

    def reset(self) -> np.ndarray:
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

        reward = -fitted_r.std().item()
        if self.min_reward is not None and self.max_reward is not None:
            if reward < self.min_reward:
                self.min_reward = reward

            if reward > self.max_reward:
                self.max_reward = reward

            reward = reward - self.min_reward
            reward /= self.max_reward - self.min_reward
            reward = np.clip(reward, 0.0, 1.0)

        return self.get_context(), reward

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

        self.first_iteration = True

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

        if self.first_iteration:
            X = self.feat_scaler.fit_transform(X)
            self.first_iteration = False
        else:
            X = self.feat_scaler.transform(X)

        mu_f, sigma_f = self.gpr.predict(X, return_std=True)

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
            return -mean.item()

        np.random.seed(12)
        result = minimize(objective, x0=alpha_init, bounds=[self._domain_ra])
        alpha_opt = result.x.item()

        return alpha_opt


def plot_regret(
    train_regrets: list[float],
    val_regrets: list[float],
):
    clear_output()
    n_cols = 1
    fig, axs = plt.subplots(1, n_cols, figsize=(12, 4))

    if n_cols == 1:
        axs = [axs]

    axs[0].plot(range(1, len(train_regrets) + 1), train_regrets, label="CGP-UCB")
    axs[0].plot(range(1, len(val_regrets) + 1), val_regrets, label="Optimal")
    axs[0].set_ylabel("Regret")

    for ax in axs:
        ax.set_xlabel("epoch")
        ax.legend()

    plt.show()


def train(start_date: str | None = "1982-01-01"):
    start_date = pd.Timestamp(start_date) if start_date else None
    train_end = pd.Timestamp("1999-12-31")

    _, runner = initialize(
        dataset=DATASET,
        trading_config=TRADING_CONFIG,
        topn=TOP_N,
        rebal_freq=REBAL_FREQ,
    )
    rebal_dates = runner.init_backtester().rebal_schedule

    true_optimal = pd.read_csv("targets.csv")
    true_optimal["start_date"] = pd.to_datetime(true_optimal["start_date"])
    true_optimal = true_optimal.set_index("start_date")
    true_optimal["reward"] = -true_optimal["vol"]

    true_optimal_train = true_optimal.loc[true_optimal.index < train_end]
    min_reward = true_optimal_train["reward"].min()
    max_reward = true_optimal_train["reward"].max()
    if min_reward is not None and max_reward is not None:
        true_optimal["reward"] = (true_optimal["reward"] - min_reward) / (
            max_reward - min_reward
        )

    env = BanditEnvironment(
        runner,
        min_reward=min_reward,
        max_reward=max_reward,
        start_date=start_date,
        train_end=train_end,
    )

    optimal_grid = np.linspace(0.0, 1.0, 100, endpoint=False)
    optimal_grid = np.append(optimal_grid, np.array([1.0]))

    agent = CGPUCBAgent(
        n_contexts=env.n_features,
        n_actions=optimal_grid.shape[0],
        exploration_mode=True,
    )

    # initialize CGP-UCB
    rounds = len(env)
    optim = []
    context = env.reset()
    rewards = []
    optimal_values = []
    try:
        for t in (pbar := tqdm(range(rounds))):
            current_date = env.get_current_date()

            if current_date in rebal_dates:
                # TODO(@V): Causal window here!
                agent.exploration_mode = False
            else:
                agent.exploration_mode = True

            action = agent.get_action(context)
            context, reward = env.step(action)

            agent.add_data_point(action, context, reward)

            rewards.append(reward)
            # avg_reward = total_reward / (t + 1)

            true_optimal_action = (
                true_optimal.loc[current_date, "shrinkage"]
                if current_date in true_optimal.index
                else 0.0
            )
            true_optimal_value = (
                true_optimal.loc[current_date, "reward"]
                if current_date in true_optimal.index
                else 0.0
            )

            pbar.set_description(
                f"Date: {current_date}, Action: {action:.6f}, Optimal Action: {true_optimal_action:.6f}, Reward: {reward:.6f}, Optimal Reward: {true_optimal_value:.6f}"
            )

            optimal_values.append(true_optimal_value)
            if t % 300 == 0 and t > 0:
                plot_regret(rewards, optimal_values)

            if current_date in rebal_dates:
                optim.append([current_date, action])

            # optim.append([current_date, action])

            if t % 20 == 0:
                env.action_hist.to_csv("cgp_ucb.csv", index=True, header=True)

        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_gmv.csv", index=True, header=True)
    except:
        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_gmv.csv", index=True, header=True)


if __name__ == "__main__":
    train()
