from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from imitation.data.wrappers import RolloutInfoWrapper

import gymnasium
from gymnasium import spaces


class BanditEnvironment(gymnasium.Env):
    def __init__(
        self,
        experiment_runner: Runner,
        start_date: pd.Timestamp,
        train_end: pd.Timestamp,
        min_reward: float,
        max_reward: float,
    ) -> None:
        super().__init__()

        self.min_reward = min_reward
        self.max_reward = max_reward

        self.experiment_runner = experiment_runner

        scaler = StandardScaler()

        self.features = self.experiment_runner.features.astype(np.float32)
        scaler.fit(self.features.loc[self.features.index < train_end])
        self.features = scaler.transform(self.features)
        self.features = pd.DataFrame(
            self.features,
            columns=self.experiment_runner.features.columns,
            index=self.experiment_runner.features.index,
        )

        self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.features.shape[1]),
            dtype=np.float32,
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

    def __len__(self):
        return len(self.start_dates)

    def get_current_id(self):
        return self.current_id

    def get_current_date(self):
        return self.current_start

    def get_current_end_date(self):
        return self.current_end

    def reset(self, *args, **kwargs):
        return self.get_context(), {}

    def get_context(self):
        return self.features.loc[self.current_start].fillna(0).to_numpy().reshape(1, -1)

    def step(
        self, action: np.array
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
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
        vol = fitted_r.std().item()

        reward = vol_to_reward(vol)
        if self.min_reward is not None and self.max_reward is not None:
            if reward < self.min_reward:
                self.min_reward = reward
            if reward > self.max_reward:
                self.max_reward = reward
            reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)

        done = 1

        return obs, np.array([reward]), np.array([done]), np.array([0.0]), {}

    @property
    def action_hist(self):
        return pd.DataFrame(self._action_hist, columns=["date", "cgp_ucb"]).set_index(
            "date"
        )

    def render(self, mode: str = "human", close: bool = False):
        self.action_hist.plot()


class OptimalEnvironment(gymnasium.Env):
    def __init__(
        self,
        experiment_runner: Runner,
        true_optimal: pd.DataFrame,
        start_date: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> None:
        super().__init__()

        self.experiment_runner = experiment_runner
        self.true_optimal = true_optimal

        scaler = StandardScaler()

        self.features = self.experiment_runner.features.astype(np.float32)
        scaler.fit(self.features.loc[self.features.index < train_end])
        self.features = scaler.transform(self.features)
        self.features = pd.DataFrame(
            self.features,
            columns=self.experiment_runner.features.columns,
            index=self.experiment_runner.features.index,
        )

        self.optimal_rewards = vol_to_reward(self.true_optimal["vol"]).astype(
            np.float32
        )

        self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.features.shape[1]),
            dtype=np.float32,
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

    def __len__(self):
        return len(self.start_dates)

    def get_current_id(self):
        return self.current_id

    def get_current_date(self):
        return self.current_start

    def get_current_end_date(self):
        return self.current_end

    def reset(self, *args, **kwargs):
        return self.get_context(), {}

    def get_context(self):
        return self.features.loc[self.current_start].fillna(0).to_numpy().reshape(1, -1)

    def step(self, action: np.array) -> tuple[np.ndarray, float, float, float, dict]:
        action = action[0]

        self._action_hist.append([self.current_end, action])

        obs = self.get_context()
        reward = self.optimal_rewards[self.current_start]
        done = 1

        self.current_id += 1
        self.current_start = self.start_dates[self.current_id]
        self.current_end = self.end_dates[self.current_id]

        return obs, reward, done, 0.0, {}

    @property
    def action_hist(self):
        return pd.DataFrame(self._action_hist, columns=["date", "cgp_ucb"]).set_index(
            "date"
        )

    def render(self, mode: str = "human", close: bool = False):
        self.action_hist.plot()


# Function to initialize an environment instance
def make_env(
    experiment_runner: Runner,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
    min_reward: float,
    max_reward: float,
) -> Callable:
    """
    Returns a callable that creates an instance of BanditEnvironment.

    Args:
        experiment_runner: The Runner instance required by the environment.
        start_date: Start date of the environment.
        train_end: Training end date.

    Returns:
        A callable function that creates an environment.
    """

    def _init():
        return BanditEnvironment(
            experiment_runner, start_date, train_end, min_reward, max_reward
        )

    return _init


# Function to initialize an environment instance
def make_optimal_env(
    experiment_runner: Runner,
    true_optimal: pd.DataFrame,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Callable:
    """
    Returns a callable that creates an instance of BanditEnvironment.

    Args:
        experiment_runner: The Runner instance required by the environment.
        start_date: Start date of the environment.
        train_end: Training end date.

    Returns:
        A callable function that creates an environment.
    """

    def _init():
        return RolloutInfoWrapper(
            OptimalEnvironment(experiment_runner, true_optimal, start_date, train_end)
        )

    return _init
