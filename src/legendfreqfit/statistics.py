import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, chi2

log = logging.getLogger(__name__)


def emp_cdf(
    data: np.array,  # the data to make a cdf out of
    bins=100,  # either number of bins or list of bin edges
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

    return np.cumsum(h) / np.sum(h), b

def percentile(
    data: np.array, # the data to make a cdf out of
    percentiles: np.array  # which percentiles to find; should be in [0, 1]
):
    """
    Returns the test statistic, linearly interpolated, that defines the percentiles in `percentiles`
    for the given data
    """
    nevts = len(data)
    data = sorted(data)
    if isinstance(percentiles, float):
        percentiles = np.array([percentiles])
    results = np.zeros_like(percentiles)
    for i, p in enumerate(percentiles):
        p_idx = int(p*nevts)
        p_rem = p*nevts - p_idx
        results[i] = data[p_idx] + p_rem*(data[p_idx+1] - data[p_idx])
    return results


def binomial_unc_band(
    cdf: np.array,  # binned CDF
    nevts: int,  # number of events the CDF is based off of
    CL: float = 0.68,  # confidence level for band
):
    """
    Returns the confidence band for a given CDF by taking the confidence interval of a
    binomial distribution with N=nevts and P=the value of the CDF at each point
    """
    interval = binom.interval(CL, nevts, cdf)
    lo_binom_band = interval[0] / nevts
    hi_binom_band = interval[1] / nevts

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
    bins=None,  # int or array, number of bins or list of bin edges for CDF
    step: float = 0.01,  # specify the (approximate) step size for the bins if list of bins is not passed
    threshold: float = 0.9,  # critical threshold for test statistic
    confidence: float = 0.68,  # width of confidence interval on the CDF 
    plot: bool = False,  # if True, save plots of CDF and PDF with critical bands
    plot_dir: str = "",  # directory where to save plots
    plot_title: str = "",
):
    """
    Returns the critical value of the test statistic for the specified threshold and the confidence interval on this
    critical value.
    Bins are only used for plotting purposes
    """
    nevts = len(ts)
    binom_interval = binom.interval(confidence, nevts, threshold)
    lo_binom_percentile = binom_interval[0] / nevts
    hi_binom_percentile = binom_interval[1] / nevts

    ts_crit, ts_lo, ts_hi = percentile(data=ts, percentiles=[threshold, lo_binom_percentile, hi_binom_percentile])


    if plot:
        if isinstance(bins, int):
            bins = np.linspace(0, np.nanmax(ts), bins)
        elif not isinstance(bins, np.ndarray):
            bins = np.linspace(0, np.nanmax(ts), int(np.nanmax(ts) / step) )

        cdf, binedges = emp_cdf(
            ts, bins
        )  # note that the CDF is evaluated at the right bin edge

        lo_band, hi_band = binomial_unc_band(cdf, nevts=len(ts), CL=confidence)

        # TODO: would like to use interpolation so that result is independent of number of bins, etc.
        # but this is a little tricky because need inverse of function and this is not necessarily strictly
        # increasing. So instead allow the user to specify the bin step so this is respected regardless of
        # number of events (which can dictate bin size if fixed number of bins is requested b/c more likely to get
        # very large test statistic with more events).

        int_thresh = int(100 * threshold)
        int_conf = int(100 * confidence)

        fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(11, 5))

        axs[0].plot(binedges[1::], cdf, c="C0", label="empirical CDF")
        axs[0].plot(
            binedges[1::],
            lo_band,
            c="C0",
            ls="--",
            label=f"CDF {int_conf}% CL interval",
        )
        axs[0].plot(
            binedges[1::],
            hi_band,
            c="C0",
            ls="--",
        )
        axs[0].plot(
            binedges,
            chi2.cdf(binedges, df=1),
            color="black",
            label=r"$\chi^2_{dof=1}$ CDF",
        )

        axs[0].axvline(
            ts_crit,
            color="g",
            alpha=0.75,
            label=f"{int_thresh}% CL, "
            + rf"$t_C = {{{ts_crit:0.2f}}}_{{-{ts_crit-ts_lo:0.2f}}}^{{+{ts_hi-ts_crit:0.2f}}}$",
        )
        axs[0].axvspan(ts_lo, ts_hi, alpha=0.25, color="g")
        axs[0].axhline(
            threshold,
            color="orange",
            alpha=0.75,
            label=rf"actual CL: ${100*threshold:0.1f} \pm {100*(hi_binom_percentile-lo_binom_percentile)/2.0:0.1f}$%",
        )
        axs[0].axhspan(lo_binom_percentile, hi_binom_percentile, color="orange", alpha=0.25)
        axs[0].set_xlabel(r"$t$")
        axs[0].set_ylabel(r"CDF$(t)$")
        axs[0].legend()
        axs[0].set_ylim([0, 1])
        axs[0].set_xlim([0, None])
        axs[0].grid()

        # bin the PDF with slightly larger bins if the step size is too small (just for viewing)
        bincenters = (binedges[1:] + binedges[:-1]) / 2.0
        binsize = bincenters[1] - bincenters[0]  # assume uniformly spaced bins
        factor = 1
        while (len(ts) / (len(binedges) / factor)) < 10:
            factor *= 2

        pdf = np.histogram(ts, bins=binedges[::factor])[0] / len(ts)

        axs[1].stairs(pdf, binedges[::factor], fill=False, zorder=3)
        axs[1].plot(
            bincenters,
            chi2.pdf(bincenters, df=1) * binsize * factor,
            color="k",
            label=r"$\chi^2_{dof=1}$ PDF",
        )
        axs[1].axvline(
            chi2.ppf(threshold, df=1),
            color="black",
            linestyle="dashed",
            label=f"{int_thresh}% CL, "
            + r"$\chi^2_C = $"
            + f"{chi2.ppf(threshold, df=1):0.2f}",
        )

        axs[1].axvline(
            ts_crit,
            color="g",
            alpha=0.75,
            label=f"{int_thresh}% CL, "
            + rf"$t_C = {{{ts_crit:0.2f}}}_{{-{ts_crit-ts_lo:0.2f}}}^{{+{ts_hi-ts_crit:0.2f}}}$",
        )
        axs[1].axvspan(ts_lo, ts_hi, color="g", alpha=0.25)

        axs[1].set_xlabel(r"t")
        axs[1].set_ylabel(r"$P(t)$")
        axs[1].legend()
        axs[1].set_yscale("log")
        axs[1].set_ylim([1 / (10 * len(ts)), 1.0])
        axs[1].set_xlim([0, None])
        axs[1].grid(zorder=0)

        plt.suptitle(plot_title)
        plt.savefig(plot_dir + f"ts_critical_{int_thresh}.pdf", dpi=300)

        return (ts_crit, ts_lo, ts_hi), (threshold, lo_binom_percentile, hi_binom_percentile), fig

    return (ts_crit, ts_lo, ts_hi), (threshold, lo_binom_percentile, hi_binom_percentile)


def toy_ts_critical_p_value(
    ts: np.array,  # list of test statistics (output of Experiment.toy_ts)
    ts_exp: float,  # value of the test-statistic from experiment data at this s value
    bins=100,  # int or array, number of bins or list of bin edges for CDF
    step: float = 0.01,  # specify the (approximate) step size for the bins if list of bins is not passed
    plot: bool = False,  # if True, save plots of CDF and PDF with critical bands
    plot_dir: str = "",  # directory where to save plots
    plot_title: str = "",
):
    """
    Returns the p-value associated with the test statistic from an experiment by comparing with the PDF generated from toys
    """

    # Step one is to get the critical p-values. We can do this by using the empirical CDF and seeing where the experiment
    # test statistic crosses it vertically, and then find the probability by looking horitzontally

    if isinstance(bins, int):
        bins = np.linspace(0, np.nanmax(ts), int((np.nanmax(ts)) / step))

    cdf, binedges = emp_cdf(
        ts, bins
    )  # note that the CDF is evaluated at the right bin edge

    bin_crossing = (np.abs(binedges - ts_exp)).argmin()
    p_value = 1 - cdf[bin_crossing]

    if plot:
        fig = plt.figure(figsize=(11, 6))

        plt.step(binedges[1::], cdf, c="C0", label="empirical CDF")

        plt.axvline(ts_exp, color="g", alpha=0.75, label="Observed Test Statistic")

        plt.axhline(
            cdf[bin_crossing],
            color="orange",
            alpha=0.75,
            label=rf"actual CL: ${p_value*100:0.1f}$%",
        )
        plt.xlabel(r"$t$")
        plt.ylabel(r"CDF$(t)$")
        plt.legend()
        plt.ylim([0, 1])
        plt.xlim([0, None])
        plt.grid()

        plt.suptitle(plot_title)
        plt.savefig(plot_dir + "ts_critical_p_value.pdf", dpi=300)

        return p_value, fig

    return p_value


def brazil_data(toy_ts: np.array, ts_observed: np.array):
    """
    Parameters
    ----------
    toy_ts
        List of lists. Each list is a list of test statistics from the toys generated at that value
    ts_observed
        List. These are the observed values of the test statistic from the experiment for a given value

    Returns
    -------
    p_values
        A list of the p-values associated with the observed data
    """

    if len(toy_ts) != len(ts_observed):
        raise ValueError(
            "The number of scanned points for the toys is not equal to the number of observed test statistics"
        )

    p_values = []
    for i, ts_exp in enumerate(ts_observed):
        p_values.append(toy_ts_critical_p_value(toy_ts[i], ts_exp, plot=False))

    return p_values
