from __future__ import annotations

from tqdm import tqdm

import numpy as np
import pandas as pd
import GPy
from CGP_UCB import CGPUCB

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

    experiment_config.START_DATE = "1980-01-01"
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


class BanditEnvironment:
    def __init__(self, experiment_runner: Runner, start_date: pd.Timestamp | None = None) -> None:
        self.action_space = np.linspace(0, 1, 100)

        self.experiment_runner = experiment_runner

        # self.features = self.experiment_runner.features
        self.features = self.experiment_runner.factors.shift(1).fillna(0)

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

    def reset(self):
        self.current_id = 0
        return self.get_context()

    def get_context(self):
        return self.features.loc[self.current_start]

    def contexts(self):
        return [self.features.loc[date].to_numpy() for date in self.start_dates]

    def step(self, action: np.array) -> np.array:
        action = action[0]

        self._action_hist.append([self.current_start, action])

        estimator = CovEstimators.RISKFOLIO.value(
            alpha=action,
            estimator_type="shrunk",
        )

        strategy = MinVariance(
            cov_estimator=estimator,
            trading_config=self.experiment_runner.trading_config,
            window_size=252,
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

        return np.array([1 / fitted_r.std().item() * 100])

    @property
    def action_hist(self):
        return pd.DataFrame(self._action_hist, columns=["date", "cgp_ucb"]).set_index("date")


def train(start: str | None = None):
    start = pd.Timestamp(start) if start else None
    runner = get_runner()

    env = BanditEnvironment(runner, start)
    actions = env.action_space
    contexts = np.array(env.contexts())

    print(len(actions), len(contexts), len(actions) * len(contexts))

    kernel1 = GPy.kern.RBF(input_dim=1, active_dims=[0])
    kernel2 = GPy.kern.RBF(input_dim=1, active_dims=[1])
    kernel = kernel1 * kernel2

    # initialize CGP-UCB
    agent = CGPUCB(
        actions=actions,
        contexts=contexts,
        sample_from_environment=env.step,
        kernel=kernel,
    )
    rounds = len(env)
    for i in (pbar := tqdm(range(rounds))):
        context_index = env.get_current_id()
        agent.learn(context_index)

        if i % 20 == 0:
            env.action_hist.to_csv("cgp_ucb.csv", index=True, header=True)

if __name__ == "__main__":
    train("1981-01-01")
