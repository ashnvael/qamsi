from __future__ import annotations

from typing import Callable

from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output

import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.policies.base import NonTrainablePolicy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.data.types import TransitionsMinimal
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

from qamsi.config.trading_config import TradingConfig
from qamsi.runner import Runner
from qamsi.strategies.estimated.min_var import MinVariance
from qamsi.cov_estimators.cov_estimators import CovEstimators
from run import Dataset


DATASET = Dataset.SPX_US

def get_runner() -> Runner:
    experiment_config = DATASET.value()

    stocks = tuple(
        pd.read_csv(experiment_config.PATH_OUTPUT / experiment_config.STOCKS_LIST_FILENAME)
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
    def __init__(self, experiment_runner: Runner, start_date: pd.Timestamp, train_end: pd.Timestamp) -> None:
        super().__init__()

        self.experiment_runner = experiment_runner

        scaler = StandardScaler()

        self.features = self.experiment_runner.features
        scaler.fit(self.features.loc[self.features.index < train_end])
        self.features = scaler.transform(self.features)
        self.features = pd.DataFrame(
            self.features,
            columns=self.experiment_runner.features.columns,
            index=self.experiment_runner.features.index,
        )

        self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, self.features.shape[1]), dtype=np.float32
        )

        self.start_dates = self.experiment_runner.returns.simple_returns.loc[start_date:].index
        self.end_dates = [date + pd.tseries.offsets.BDay(n=21) for date in self.start_dates]

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
        return pd.DataFrame(self._action_hist, columns=["date", "cgp_ucb"]).set_index("date")

    def render(self, mode: str = "human", close: bool = False):
        self.action_hist.plot()


class ExpertPolicy(NonTrainablePolicy):
    def __init__(self, env: gym.Env, optimal_policy: pd.Series) -> None:
        super().__init__(
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        self.policy = optimal_policy.to_numpy()
        self.current_id = 0

    def _choose_action(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        action = self.policy[self.current_id].round(1)
        self.current_id += 1
        return np.array([action])


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

# Function to initialize an environment instance
def make_env(experiment_runner: Runner, start_date: pd.Timestamp, train_end: pd.Timestamp) -> Callable:
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
        return BanditEnvironment(experiment_runner, start_date, train_end)
    return _init

def train(start_date: str | None = "1982-01-01"):
    start_date = pd.Timestamp(start_date) if start_date else None
    train_end = pd.Timestamp("1999-12-31")

    runner = get_runner()
    rebal_dates = runner.init_backtester().rebal_schedule

    true_optimal = pd.read_csv("targets.csv")
    true_optimal["start_date"] = pd.to_datetime(true_optimal["start_date"])
    true_optimal = true_optimal.set_index("start_date")
    true_optimal["reward"] = 1 / true_optimal["vol"]

    # Number of environments to vectorize (e.g., 4 parallel environments)
    n_envs = 1

    # Create the vectorized environment using DummyVecEnv
    env = DummyVecEnv([make_env(runner, start_date, train_end) for _ in range(n_envs)])

    expert = ExpertPolicy(env, true_optimal["shrinkage"])
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=len(true_optimal), min_episodes=None),
        rng=np.random.default_rng(12),
        verbose=True,
    )

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.99,
        n_epochs=5,
        seed=12,
    )
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    train_data = true_optimal.loc[true_optimal.index < train_end]
    gail_trainer.train(len(train_data))

    try:
        obs = env.reset()
        rewards = []
        optimal_values = []
        optim = []
        for t in (pbar := tqdm(range(100))):
            action, _states = gail_trainer.policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            current_date = env.get_current_date()
            true_optimal_action = true_optimal.loc[current_date, "shrinkage"]
            true_optimal_value = true_optimal.loc[current_date, "vol"]

            pbar.set_description(f"Date: {current_date}, Action: {action.item():.6f}, Optimal Action: {true_optimal_action:.6f}, Reward: {reward:.6f}, Optimal Reward: {-true_optimal_value:.6f}")

            env.action_hist.to_csv("rl.csv", index=True, header=True)

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
            if t % 20 == 0 and t > 0:
                plot_regret(rewards, optimal_values)

            optim.append([current_date, action])

            if t % 20 == 0:
                env.action_hist.to_csv("cgp_ucb.csv", index=True, header=True)

        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_gmv.csv", index=True, header=True)
    except:
        optim = pd.DataFrame(optim, columns=["date", "cgp_ucb"]).set_index("date")
        optim.to_csv("cgp_ucb_gmv.csv", index=True, header=True)


if __name__ == "__main__":
    train()
