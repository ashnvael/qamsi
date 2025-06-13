from __future__ import annotations

import math
import numpy as np
import pandas as pd

from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.strategies.optimization_data import PredictionData, TrainingData


def _QIS(Y, h: float | None = None, k: float | None = None):
    # Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    # Post-Condition: Sigmahat dataframe is returned

    # Set df dimensions
    N = Y.shape[0]  # num of columns
    p = Y.shape[1]  # num of rows

    # default setting
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)  # demean
        k = 1

    # vars
    n = N - k  # adjust effective sample size
    c = p / n  # concentration ratio

    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample = (sample + sample.T) / 2  # make symmetrical

    # Spectral decomp
    lambda1, u = np.linalg.eig(sample)  # use LAPACK routines
    lambda1 = lambda1.real  # clip imaginary part due to rounding error
    u = u.real  # clip imaginary part for eigenvectors

    lambda1 = lambda1.real.clip(min=0)  # reset negative values to 0
    dfu = pd.DataFrame(u, columns=lambda1)  # create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1, inplace=True)  # sort df by column index
    lambda1 = dfu.columns  # recapture sorted lambda

    # COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    if h is None:
        h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35  # smoothing parameter
    invlambda = (
        1 / lambda1[max(1, p - n + 1) - 1 : p]
    )  # inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl["lambda"] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]  # like  1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())  # Reset column names
    Lj_i = Lj.subtract(Lj.T)  # like (1/lambda_j)-(1/lambda_i)

    theta = (
        Lj.multiply(Lj_i)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )  # smoothed Stein shrinker
    Htheta = (
        Lj.multiply(Lj * h)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )  # its conjugate
    Atheta2 = theta**2 + Htheta**2  # its squared amplitude

    if p <= n:  # case where sample covariance matrix is not singular
        delta = 1 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )  # optimally shrunk eigenvalues
        delta = delta.to_numpy()
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda.to_numpy()))  # shrinkage of null
        #                                                 eigenvalues
        delta = np.repeat(delta0, p - n)
        delta = np.concatenate((delta, 1 / (invlambda * Atheta2)), axis=None)

    deltaQIS = delta * (sum(lambda1) / sum(delta))  # preserve trace

    temp1 = dfu.to_numpy()
    temp2 = np.diag(deltaQIS)
    temp3 = dfu.T.to_numpy().conjugate()
    # reconstruct covariance matrix
    sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
    return sigmahat


class QISCovEstimator(BaseCovEstimator):
    def __init__(self, h: int | None = None, k: int | None = None) -> None:
        super().__init__()
        self.h = h
        self.k = k

        self._fitted_vols = None
        self._fitted_corr = None
        self._fitted_cov = None

        self._obs_cov = None

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns

        self._obs_cov = ret.cov()
        cov = _QIS(self._obs_cov, self.h, self.k)

        self._fitted_cov = pd.DataFrame(
            cov, index=self._available_assets, columns=self._available_assets
        )

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self._fitted_cov
