import numpy as np
import numpy.typing as npt
import math

def make_tasks(low, high, n, seed=None) -> npt.ArrayLike:
    """Generate task parameters"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(np.linspace(low, high, n).round(0))

def make_cov(k:int, no_cor:bool=True, seed=None) -> npt.ArrayLike:
    """Generate covariance matrix"""
    if seed is not None:
        np.random.seed(seed)
    cov = np.random.randn(k, k)
    if no_cor:
        cov = np.diag(np.diag(cov))
        return cov
    else:
        return cov @ cov.T

def nearest_square_dims(n:int) -> int | int:
    """Reshape vector to nearest square that minimizes the difference between dimensions n x m"""
    rows = math.floor(math.sqrt(n))
    cols = math.ceil(math.sqrt(n))
    while rows * cols < n:
        if (cols - rows) <= 1:
            cols += 1
        else:
            rows += 1
    return rows, cols