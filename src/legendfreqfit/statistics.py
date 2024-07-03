import logging
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)


def emp_cdf(
    data: np.array,  # the data to make a cdf out of
    bins=100,  # either number of bins or list of bin edges
) -> Tuple[np.array, np.array]:
    """
    Create a binned empirical CDF given a dataset
    """

    if isinstance(bins, int):
        x = np.linspace(np.nanmin(data), np.nanmax(data), bins)
    elif isinstance(bins, np.ndarray) or isinstance(bins, list):
        x = np.array(bins)
    else:
        raise TypeError(f"bins must be array-like or int, instead is {type(bins)}")

    return np.array([len(np.where(data <= b)[0]) / len(data) for b in x[1:]]), x


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


def test_statistic_asymptotic_limit(
    t_mus: np.array, mu: float, mu_0: float, sigma: float
) -> np.array:
    """
    In the asymptotic limit, the test statiistics become distribute according to a noncentral chi-squared distribution

    Parameters
    ----------
    t_mus
        Array of test statistic values at which to evaluate the distribution
    mu
        Strength parameter under test
    mu_0
        Strength parameter the data are distributed according to
    sigma
        Standard deviation obtained from covariance matrix of estimators for all parameters
    """

    non_centrality = (mu - mu_0) ** 2 / sigma**2

    if non_centrality == 0:
        return 1 / np.sqrt(t_mus * 2 * np.pi)
    else:
        return (1 / np.sqrt(t_mus * 2 * np.pi)) * (
            np.exp(-0.5 * (np.sqrt(t_mus) + np.sqrt(non_centrality)) ** 2)
            + (np.exp(-0.5 * (np.sqrt(t_mus) - np.sqrt(non_centrality)) ** 2))
        )
