import numpy as np


def avg_non_diagonal_elements(corr_matrix):
    """Function to compute the average of non-diagonal elements in each correlation matrix."""
    # Select the non-diagonal elements using numpy
    non_diag = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    return np.nanmean(non_diag)


def avg_corr(rolling_window):
    # Compute the correlation matrix for the rolling window
    corr_matrix = rolling_window.corr()

    # Compute the average of non-diagonal elements
    return avg_non_diagonal_elements(corr_matrix)
