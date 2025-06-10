import numpy as np


def corr_matrix_from_cov(var_covar: np.ndarray) -> np.ndarray:
    diag_inv = np.diag(1 / np.sqrt(np.diag(var_covar)))
    return diag_inv @ var_covar @ diag_inv


def var_covar_from_corr_array(
    corr_array: np.ndarray, volatilities: np.ndarray
) -> np.ndarray:
    return volatilities @ corr_array @ volatilities


def var_covar_from_corr_array_mac(
    corr_array: np.ndarray, volatilities: np.ndarray
) -> np.ndarray:
    # Use np.dot() due to Apple Silicon chip issues in numpy
    return np.dot(np.dot(volatilities, corr_array), volatilities)
