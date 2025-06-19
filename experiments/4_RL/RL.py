from __future__ import annotations

from tqdm import tqdm

import numpy as np
import pandas as pd
import gym
from gym import spaces

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC, PPO

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


class BanditEnvironment(gym.Env):
    def __init__(self, experiment_runner: Runner, start_date: pd.Timestamp) -> None:
        super().__init__()

        self.experiment_runner = experiment_runner
        self.features = self.experiment_runner.features.astype(np.float32)

        self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.features.shape[1]),
            dtype=np.float32,
        )

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

    def reset(self, *args, **kwargs):
        self.current_id = 0
        return self.get_context()

    def get_context(self):
        return self.features.loc[self.current_start].fillna(0).to_numpy().reshape(1, -1)

    def step(self, action: np.array) -> tuple[np.ndarray, float, float, dict]:
        action = action[0]

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

        obs = self.get_context()
        reward = -fitted_r.std().item()
        done = 1

        return obs, reward, done, {}

    @property
    def action_hist(self):
        return pd.DataFrame(self._action_hist, columns=["date", "cgp_ucb"]).set_index(
            "date"
        )

    def render(self, mode: str = "human", close: bool = False):
        self.action_hist.plot()


def train(start_date: str | None = "1982-01-01"):
    start_date = pd.Timestamp(start_date) if start_date else None

    runner = get_runner()

    true_optimal = pd.read_csv("targets.csv")
    true_optimal["start_date"] = pd.to_datetime(true_optimal["start_date"])
    true_optimal = true_optimal.set_index("start_date")

    env = BanditEnvironment(runner, start_date=start_date)

    # initialize CGP-UCB
    agent = SAC("MlpPolicy", env, verbose=1, device="mps")
    agent.learn(total_timesteps=1_000)
    # agent.save("rl_model")

    obs = env.reset()
    for _ in (pbar := tqdm(range(100))):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        current_date = env.get_current_date()
        true_optimal_action = true_optimal.loc[current_date, "shrinkage"]
        true_optimal_value = true_optimal.loc[current_date, "vol"]

        pbar.set_description(
            f"Date: {current_date}, Action: {action.item():.6f}, Optimal Action: {true_optimal_action:.6f}, Reward: {reward:.6f}, Optimal Reward: {-true_optimal_value:.6f}"
        )

        env.action_hist.to_csv("rl.csv", index=True, header=True)

    # rounds = len(env)
    # context = env.reset()
    # agent.add_data_point(0.0, context, 0.0)
    # for i in (pbar := tqdm(range(rounds))):
    #     action = agent.get_action(context)
    #     context, reward = env.step(action)
    #     agent.add_data_point(action, context, reward)
    #
    #     current_date = env.get_current_date()
    #     true_optimal_action = true_optimal.loc[current_date, "shrinkage"]
    #     true_optimal_value = true_optimal.loc[current_date, "vol"]
    #
    #     pbar.set_description(f"Date: {current_date}, Action: {action:.6f}, Optimal Action: {true_optimal_action:.6f}, Reward: {reward:.6f}, Optimal Reward: {-true_optimal_value:.6f}")
    #
    #     agent.exploration_mode = False
    #
    #     if i % 20 == 0:
    #         env.action_hist.to_csv("cgp_ucb.csv", index=True, header=True)


if __name__ == "__main__":
    train()
