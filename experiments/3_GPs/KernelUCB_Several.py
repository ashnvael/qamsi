from __future__ import annotations

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import numpy as np
import pandas as pd
from contextual_bandits import KernelUCB
import matplotlib.pyplot as plt
from IPython.display import clear_output

from qamsi.config.trading_config import TradingConfig
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from run import Dataset, initialize


REBAL_FREQ = "ME"
DATASET = Dataset.TOPN_US
TOP_N = 30
ESTIMATION_WINDOW = 365

N_RESAMPLES = 10

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

    def sample_reward(self, action: float) -> float:
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

        reward = -fitted_r.std().item()
        if self.min_reward is not None and self.max_reward is not None:
            if reward < self.min_reward:
                self.min_reward = reward

            if reward > self.max_reward:
                self.max_reward = reward

            reward = reward - self.min_reward
            reward /= self.max_reward - self.min_reward
            reward = np.clip(reward, 0.0, 1.0)

        return reward

    def step(self, action: float) -> tuple[np.ndarray, float]:
        self._action_hist.append([self.current_end, action])

        reward = self.sample_reward(action)

        self.current_id += 1
        self.current_start = self.start_dates[self.current_id]
        self.current_end = self.end_dates[self.current_id]

        return self.get_context(), reward

    @property
    def action_hist(self):
        return pd.DataFrame(self._action_hist, columns=["date", "cgp_ucb"]).set_index(
            "date"
        )


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

    optimal_grid = np.linspace(0.0, 1.0, 10, endpoint=False)
    optimal_grid = np.append(optimal_grid, np.array([1.0]))

    agent = KernelUCB(
        len(optimal_grid), env.n_features, alpha=2.0, kernel="rbf", gamma=0.5
    )

    # initialize CGP-UCB
    rounds = len(env)
    optim = []
    context = env.reset()
    rewards = []
    prod_chosen_arm = None
    try:
        for t in tqdm(range(rounds)):
            current_date = env.get_current_date()

            if current_date in rebal_dates:
                # TODO(@V): Causal window here!
                agent.alpha = 0.0

                chosen_arm = agent.select_arm(context)
                prod_chosen_arm = chosen_arm

                action = optimal_grid[chosen_arm].item()
                reward = env.sample_reward(action)

                agent.update(chosen_arm, context, reward)
                optim.append([current_date, action])
            else:
                agent.alpha = 2.0
                prod_chosen_arm = None

            for chosen_arm in range(len(optimal_grid)):
                if prod_chosen_arm is not None and chosen_arm == prod_chosen_arm:
                    continue

                action = optimal_grid[chosen_arm].item()
                reward = env.sample_reward(action)

                agent.update(chosen_arm, context, reward)

                if chosen_arm == len(optimal_grid) - 1:
                    context, reward = env.step(action)

            rewards.append(reward)
            # avg_reward = total_reward / (t + 1)

            if t % 20 == 0:
                env.action_hist.to_csv("cgp_ucb_all.csv", index=True, header=True)

        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_all.csv", index=True, header=True)
    except:
        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_all.csv", index=True, header=True)


if __name__ == "__main__":
    train()
