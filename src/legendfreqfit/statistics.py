from typing import Tuple

import numpy as np

def emp_cdf(
    data: np.array,  # the data to make a cdf out of
    bins=100,  # either number of bins or list of bin edges
) -> Tuple[np.array, np.array]:
    """
    Create a binned empirical CDF given a dataset
    """

    if not (isinstance(bins, int) or isinstance(bins, np.ndarray) or isinstance(bins, list)):
        raise TypeError(f"bins must be array-like or int, instead is {type(bins)}")
    
    h, b = np.histogram(data, bins)

    return np.cumsum(h)/np.sum(h), b


def dkw_band(
    cdf: np.array,  # binned CDF
    nevts: int,  # number of events the CDF is based off of
    CL: float = 0.68,  # confidence level for band
) -> Tuple[np.array, np.array]:
    """
    Returns the confidence band for a given CDF following the DKW inequality
    https://lamastex.github.io/scalable-data-science/as/2019/jp/11.html
    """
    alpha = 1 - CL
    eps = np.sqrt(np.log(2 / alpha) / (2 * nevts))
    lo_band = np.maximum(cdf - eps, np.zeros_like(cdf))
    hi_band = np.minimum(cdf + eps, np.ones_like(cdf))
    return lo_band, hi_band