from __future__ import annotations


if __name__ == "__main__":
    from imitation.algorithms.adversarial.airl import AIRL
    from stable_baselines3 import SAC

    from qamsi.cov_estimators.rl.inverse_rl.irl_estimator import IRLCovEstimator
    from qamsi.config.trading_config import TradingConfig

    from run import Dataset, run_backtest

    rebal_freq = "ME"
    top_n = 30
    dataset = Dataset.TOPN_US

    trading_config = TradingConfig(
        broker_fee=0.05 / 100,
        bid_ask_spread=0.03 / 100,
        total_exposure=1,
        max_exposure=None,
        min_exposure=None,
        trading_lag_days=1,
    )

    estimator = IRLCovEstimator(
        shrinkage_type="linear",
        imitation_trainer_cls=AIRL,
        policy_builder=lambda env: SAC("MlpPolicy", env, verbose=0, device="mps"),
        dataset=dataset,
        trading_config=trading_config,
        rebal_freq=rebal_freq,
        topn=top_n,
        save_path=None,
        window_size=None,
        use_saved_policy=False,
        random_seed=12,
    )

    run_result = run_backtest(
        estimator=estimator,
        dataset=dataset,
        rebal_freq=rebal_freq,
        trading_config=trading_config,
        topn=top_n,
    )

    print(run_result)  # noqa: T201
