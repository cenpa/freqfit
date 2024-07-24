import logging
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import binom

log = logging.getLogger(__name__)


def emp_cdf(
    data: np.array,  # the data to make a cdf out of
    bins = 100,  # either number of bins or list of bin edges
) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    data
        unbinned data
    bins
        number of bins or array-like of bin edges
        
    Create a binned empirical CDF given a dataset. Empirical CDF is evaluated at right bin edge; the value corresponds to
    the PDF integrated up to the right bin edge.
    """

    if isinstance(bins, int):
        binedges = np.linspace(np.nanmin(data), np.nanmax(data), bins)
    elif isinstance(bins, np.ndarray) or isinstance(bins, list):
        binedges = np.array(bins)
    else:
        raise TypeError(f"bins must be array-like or int, instead is {type(bins)}")
    
    h, b = np.histogram(data, bins)

    return np.cumsum(h)/np.sum(h), b

def binomial_unc_band(
    cdf: np.array, # binned CDF
    nevts: int,  # number of events the CDF is based off of
    CL: float = 0.68,  # confidence level for band
):
    """
    Returns the confidence band for a given CDF by taking the confidence interval of a 
    binomial distribution with N=nevts and P=the value of the CDF at each point
    """
    interval = binom.interval(CL, nevts, cdf)
    lo_binom_band = interval[0]/nevts
    hi_binom_band = interval[1]/nevts

    return lo_binom_band, hi_binom_band

def dkw_band(
    cdf: np.array,  # binned CDF
    nevts: int,  # number of events the CDF is based off of
    CL: float = 0.68,  # confidence level for band
) -> Tuple[np.array, np.array]:
    """
    Returns the confidence band for a given CDF following the DKW inequality
    https://lamastex.github.io/scalable-data-science/as/2019/jp/11.html
    """
    alpha = 1.0 - CL
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

def toy_ts_critical(
    ts: np.array,  # list of test statistics (output of Experiment.toy_ts)
    bins = 100,  # int or array, number of bins or list of bin edges for CDF
    step: float = 0.01, # specify the (approximate) step size for the bins if list of bins is not passed
    threshold: float = 0.9,  # critical threshold for test statistic
    confidence: float = 0.68,  # width of confidence interval on the CDF
    plot: bool = False,  # if True, save plots of CDF and PDF with critical bands
    plot_dir: str = "", # directory where to save plots
    plot_title: str = "",
):
    """
    Returns the critical value of the test statistic for the specified threshold and the confidence interval on this 
    critical value.
    """

    if isinstance(bins, int):
        bins = np.linspace(0, np.nanmax(ts), int((np.nanmax(ts))/step))

    cdf, binedges = emp_cdf(ts, bins) # note that the CDF is evaluated at the right bin edge

    lo_band, hi_band = binomial_unc_band(cdf, nevts=len(ts), CL=confidence)

    # TODO: would like to use interpolation so that result is independent of number of bins, etc.
    # but this is a little tricky because need inverse of function and this is not necessarily strictly
    # increasing. So instead allow the user to specify the bin step so this is respected regardless of 
    # number of events (which can dictate bin size if fixed number of bins is requested b/c more likely to get
    # very large test statistic with more events).

    idx_crit = np.where(cdf >= threshold)[0][0]
    critical = binedges[idx_crit]

    lo = lo_band[idx_crit]
    hi = hi_band[idx_crit]

    lo_idx = np.where(cdf >= lo)[0][0]
    hi_idx = np.where(cdf >= hi)[0][0]

    lo_ts = binedges[lo_idx]
    hi_ts = binedges[hi_idx]

    if plot:
        int_thresh = int(100 * threshold)
        int_conf = int(100 * confidence)

        fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(11, 5))

        axs[0].plot(binedges[1::], cdf, c='C0', label="empirical CDF")
        axs[0].plot(
            binedges[1::], lo_band, c='C0', ls='--', label=f"CDF {int_conf}% CL interval"
        )
        axs[0].plot(binedges[1::], hi_band, c='C0', ls='--',)
        axs[0].plot(
            binedges, chi2.cdf(binedges, df=1), color="black", label=r"$\chi^2_{dof=1}$ CDF"
        )

        axs[0].axvline(critical, color="g", alpha=0.75, label=f"{int_thresh}% CL, "+r"$t_C = {{{:0.2f}}}_{{-{:0.2f}}}^{{+{:0.2f}}}$".format(critical, critical-lo_ts, hi_ts-critical))
        axs[0].axvspan(lo_ts, hi_ts, alpha=0.25, color="g")
        axs[0].axhline(threshold, color="orange", alpha=0.75, label=r"actual CL: ${:0.1f} \pm {:0.1f}$%".format(100*threshold, 100*(hi-lo)/2.0))
        axs[0].axhspan(lo, hi, color="orange", alpha=0.25)
        axs[0].set_xlabel(r"$t$")
        axs[0].set_ylabel(r"CDF$(t)$")
        axs[0].legend()
        axs[0].set_ylim([0,1])
        axs[0].set_xlim([0,None])
        axs[0].grid()

        # bin the PDF with slightly larger bins if the step size is too small (just for viewing)
        bincenters = (binedges[1:] + binedges[:-1]) / 2.0
        binsize = bincenters[1] - bincenters[0] # assume uniformly spaced bins
        factor = 1
        while ((len(ts) / (len(binedges)/factor)) < 10):
            factor *= 2
        
        pdf = np.histogram(ts, bins=binedges[::factor])[0] / len(ts)
        
        axs[1].stairs(pdf, binedges[::factor], fill=False, zorder=3)
        axs[1].plot(bincenters, chi2.pdf(bincenters, df=1)*binsize*factor, color="k", label=r"$\chi^2_{dof=1}$ PDF")
        axs[1].axvline(
            chi2.ppf(threshold, df=1),
            color="black",
            linestyle="dashed",
            label=f"{int_thresh}% CL, "+r"$\chi^2_C = $"+f'{chi2.ppf(threshold, df=1):0.2f}',
        )

        axs[1].axvline(critical, color="g", alpha=0.75, label=f"{int_thresh}% CL, "+r"$t_C = {{{:0.2f}}}_{{-{:0.2f}}}^{{+{:0.2f}}}$".format(critical, critical-lo_ts, hi_ts-critical))
        axs[1].axvspan(lo_ts, hi_ts, color="g", alpha=0.25)


        axs[1].set_xlabel(r"t")
        axs[1].set_ylabel(r"$P(t)$")
        axs[1].legend()
        axs[1].set_yscale('log')
        axs[1].set_ylim([1/(10*len(ts)),1.0])
        axs[1].set_xlim([0,None])
        axs[1].grid(zorder=0)
        
        plt.suptitle(plot_title)
        plt.savefig(plot_dir + f"ts_critical_{int_thresh}.pdf", dpi=300)

    return (critical, lo_ts, hi_ts), (threshold, lo, hi)