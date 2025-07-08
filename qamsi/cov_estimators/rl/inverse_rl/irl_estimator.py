from __future__ import annotations

from typing import Callable, Type, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.policies.base import NonTrainablePolicy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.policies import serialize
from imitation.data import rollout, types

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator
from qamsi.cov_estimators.rl.env import make_env, make_optimal_env


class ExpertPolicy(NonTrainablePolicy):
    def __init__(self, env: gym.Env, optimal_policy: pd.Series) -> None:
        super().__init__(
            observation_space=env.observation_space, action_space=env.action_space
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


class IRLCovEstimator(BaseRLCovEstimator):
    def __init__(
        self,
        shrinkage_type: str,
        imitation_trainer_cls: Type[DemonstrationAlgorithm],
        policy_builder: Callable[[DummyVecEnv], BaseAlgorithm],
        save_path: Path | None = None,
        window_size: int | None = None,
        use_saved_policy: bool = True,
        random_seed: int = 12,
    ) -> None:
        super().__init__(shrinkage_type=shrinkage_type, window_size=window_size)

        self.save_path = save_path / "gen_policy" if save_path is not None else None
        self.use_saved_policy = use_saved_policy
        self.random_seed = random_seed

        self.imitation_trainer_cls = imitation_trainer_cls
        self.policy_builder = policy_builder

    def collect_rollouts(
        self, env: DummyVecEnv, optimal_env: DummyVecEnv, optimal_actions: pd.Series
    ) -> Sequence[types.TrajectoryWithRew]:
        expert = ExpertPolicy(env, optimal_actions)

        return rollout.rollout(
            expert,
            optimal_env,
            rollout.make_sample_until(
                min_timesteps=len(optimal_actions),
                min_episodes=None,
            ),
            rng=np.random.default_rng(self.random_seed),
            verbose=True,
        )

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        if ...:
            self.policy = serialize.load_stable_baselines_model(
                self.policy.__class__, path=str(self.save_path), venv=self.env
            )
        else:
            n_envs = 1

            rewards = ...

            env = DummyVecEnv([make_env(self.shrinkage) for _ in range(n_envs)])
            optimal_env = DummyVecEnv(
                [
                    make_optimal_env(runner, true_optimal, start_date, train_end)
                    for _ in range(n_envs)
                ]
            )

            self.policy = self.policy_builder(env)

            self.reward_net = BasicShapedRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                normalize_input_layer=RunningNorm,
            )

            rollouts = self.collect_rollouts(
                env=env,
                optimal_env=optimal_env,
                optimal_actions=shrinkage_target,
            )
            self.trainer = self.imitation_trainer_cls(
                demonstrations=rollouts,
                demo_batch_size=1024,
                gen_replay_buffer_capacity=512,
                n_disc_updates_per_round=8,
                venv=self.env,
                gen_algo=self.policy,
                reward_net=self.reward_net,
            )
            self.trainer.train(len(self.env))
            self.policy = self.trainer.policy

        if self.save_path is not None:
            self._save()

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        action, _ = self.policy.predict(features.to_numpy(), deterministic=True)
        return action.item()

    def _save(self) -> None:
        serialize.save_stable_model(
            self.save_path,
            self.policy,
            self.trainer.__class__.__name__ + f"_{self.policy.__class__.__name__}_",
        )
