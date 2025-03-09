import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize
from scipy.stats import binom, chi2

from .models.constants import M76, NA

# plotting colors
NICE_BLUE = "#668DA5"
NICE_RED = "#B4584D"
NICE_GREEN = "#ABB1A2"
NICE_PINK = "#CCACAD"


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
    data: np.array,  # the data to make a cdf out of
    percentiles: np.array,  # which percentiles to find; should be in [0, 1]
):
    """
    Returns the test statistic that defines the percentiles in `percentiles`
    for the given data
    """
    nevts = len(data)
    data = sorted(data)
    if isinstance(percentiles, float):
        percentiles = np.array([percentiles])
    results = np.zeros_like(percentiles)
    for i, p in enumerate(percentiles):
        p_idx = int(p * nevts)
        p_rem = p * nevts - p_idx
        results[i] = data[p_idx]
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


def ts_critical(
    ts: np.array,  # list of test statistics (output of Experiment.toy_ts)
    threshold: float = 0.9,  # critical threshold for test statistic
    confidence: float = 0.68,  # width of confidence interval on the CDF
    plot: bool = False,  # if True, save plots of CDF and PDF with critical bands
    bins=None,  # int or array, number of bins or list of bin edges for CDF
    step: float = 0.01,  # specify the (approximate) step size for the bins if list of bins is not passed
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

    ts_crit, ts_lo, ts_hi = percentile(
        data=ts, percentiles=[threshold, lo_binom_percentile, hi_binom_percentile]
    )

    if plot:
        if isinstance(bins, int):
            bins = np.linspace(np.floor(np.nanmin(ts)), np.nanmax(ts), bins)
        elif not isinstance(bins, np.ndarray):
            bins = np.linspace(
                np.floor(np.nanmin(ts)), np.nanmax(ts), int(np.nanmax(ts) / step)
            )

        cdf, binedges = emp_cdf(
            ts, bins
        )  # note that the CDF is evaluated at the right bin edge

        lo_band, hi_band = binomial_unc_band(cdf, nevts=len(ts), CL=confidence)

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
        axs[0].axhspan(
            lo_binom_percentile, hi_binom_percentile, color="orange", alpha=0.25
        )
        axs[0].set_xlabel(r"$t$")
        axs[0].set_ylabel(r"CDF$(t)$")
        axs[0].legend()
        axs[0].set_ylim([0, 1])
        axs[0].set_xlim([bins[0], None])
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
        axs[1].set_xlim([bins[0], None])
        axs[1].grid(zorder=0)

        plt.suptitle(plot_title)
        plt.savefig(plot_dir + f"ts_critical_{int_thresh}.pdf", dpi=300)

        return (
            (ts_crit, ts_lo, ts_hi),
            (threshold, lo_binom_percentile, hi_binom_percentile),
            fig,
        )

    return (ts_crit, ts_lo, ts_hi), (
        threshold,
        lo_binom_percentile,
        hi_binom_percentile,
    )


def p_value(ts: np.array, ts_exp: float):
    """
    Returns p-value and uncertainty of an experimental test statistic and distribution of toy test statistics.
    """
    tot = 1.0 * len(ts)
    hi = np.sum(ts > ts_exp)

    pval = hi / tot

    # binomial uncertainty at 1 sigma
    unc = 1.0 / np.sqrt(tot) * np.sqrt(pval * (1.0 - pval))

    return pval, unc


def toy_ts_critical_p_value(
    ts: np.array,  # list of test statistics (output of Experiment.toy_ts)
    ts_exp: float,  # value of the test-statistic from experiment data at this s value
    bins=100,  # int or array, number of bins or list of bin edges for plotting
    step: float = 0.01,  # specify the (approximate) step size for the bins if list of bins is not passed, for plotting
    plot: bool = False,  # if True, save plots of CDF and PDF with critical bands
    plot_dir: str = "",  # directory where to save plots
    plot_title: str = "",
):
    """
    Returns the p-value associated with the observed test statistic from an experiment by comparing with the PDF generated from toys
    """

    ts_sorted = np.sort(ts)
    p_value = len(ts_sorted[ts_sorted >= ts_exp]) / len(ts_sorted)

    if plot:
        fig = plt.figure(figsize=(11, 6))

        pdf, binedges = np.histogram(ts, bins=bins)

        plt.stairs(pdf, binedges, color="C0", label="PDF")

        plt.axvline(
            ts_exp, color="g", alpha=0.75, label=rf"p-value: ${p_value*100:0.1f}$%"
        )

        plt.xlabel(r"$t$")
        plt.ylabel(r"PDF$(t)$")
        plt.legend()
        plt.grid()

        plt.suptitle(plot_title)
        plt.savefig(plot_dir + "ts_critical_p_value.pdf", dpi=300)

        return p_value, fig

    return p_value


def get_p_values(toy_ts: np.array, ts_observed: np.array):
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
        log.warning(
            "The number of scanned points for the toys is not equal to the number of observed test statistics"
        )

    p_values = []
    for i, ts_exp in enumerate(ts_observed):
        p_values.append(toy_ts_critical_p_value(toy_ts[i], ts_exp, plot=False))

    return p_values


def sensitivity(
    toy_ts_zero_signal: list,
    toy_ts: list,
    s_list: list,
    CL: float = 0.9,
    bins=1000,  # int or array, number of bins or list of bin edges for CDF
    step: float = 0.01,  # specify the (approximate) step size for the bins if list of bins is not passed
    plot: bool = False,  # if True, save plots of CDF and PDF with critical bands
    plot_dir: str = "",  # directory where to save plots
    plot_title: str = "",
    save: bool = False,
):
    """
    Parameters
    ----------
    toy_ts
        List of lists. Each list is a a list of test statistics for toys generated with non-zero signal tested against that non-zero s-value
    toy_ts_zero_signal
        List of lists. Each list test statistics generated with zero signal and tested against a non-zero signal hypothesis
    s_list
        the list of the s_values being tested against

    Returns
    -------
    p_values
        A list of the p-values associated with the observed data


    Notes
    -----
    This function loops over the S-grid tested. At each value of S, the median of the PDF of toy_ts_zero_signal (generated with 0 signal and tested against S)
    is projected onto the PDF toy_ts (generated at S and tested against S) and the p-value of the median is calculated.
    """

    if len(toy_ts) != len(toy_ts_zero_signal):
        raise ValueError("input arrays must match in length!")

    p_values_median = []
    p_values_hi = []
    p_values_lo = []

    gammas = (
        M76 * s_list / (np.log(2) * NA)
    )  # convert from reduced s-value to half-rate

    for i, ts in enumerate(toy_ts_zero_signal):
        ts = np.array(ts)

        median_ts = np.median(ts)
        upper_ts = np.quantile(ts, 0.5 + CL / 2, method="linear")
        lower_ts = np.quantile(ts, 0.5 - CL / 2, method="linear")

        sorted_toy_ts = np.sort(toy_ts[i])

        p_values_median.append(
            len(sorted_toy_ts[sorted_toy_ts >= median_ts]) / len(sorted_toy_ts)
        )
        p_values_hi.append(
            len(sorted_toy_ts[sorted_toy_ts >= upper_ts]) / len(sorted_toy_ts)
        )
        p_values_lo.append(
            len(sorted_toy_ts[sorted_toy_ts >= lower_ts]) / len(sorted_toy_ts)
        )

        if plot:
            fig = plt.figure(figsize=(18, 10))

            pdf_zero_sig, binedges_zero_sig = np.histogram(
                ts, bins=bins, range=(-2, 25)
            )
            plt.stairs(
                pdf_zero_sig,
                binedges_zero_sig,
                color=NICE_RED,
                label=rf"PDF for $\Gamma$'={s_list[i]:.2e}|$\Gamma$=0",
            )

            pdf, binedges = np.histogram(toy_ts[i], bins=bins, range=(-2, 25))

            plt.stairs(
                pdf,
                binedges,
                color=NICE_BLUE,
                label=rf"PDF for $\Gamma$'={s_list[i]:.2e}|$\Gamma$={s_list[i]:.2e}",
            )
            plt.axvline(
                median_ts,
                color=NICE_GREEN,
                alpha=0.95,
                label="Median 0-Signal Test-Statistic",
            )
            plt.axvline(
                upper_ts,
                color=NICE_GREEN,
                ls="--",
                alpha=0.95,
                label=f"Upper {CL*100:.2f}%",
            )
            plt.axvline(
                lower_ts,
                color=NICE_PINK,
                ls="--",
                alpha=0.95,
                label=f"Lower {CL*100:.2f}%",
            )

            plt.yscale("log")
            plt.xlabel(r"$t$")
            plt.ylabel(r"PDF$(t)$")
            plt.legend()
            plt.ylim([0.9, 1e5])
            plt.xlim([-1, 25])
            plt.grid()

            plt.suptitle(plot_title)
            if save:
                fig.savefig(
                    plot_dir + f"ts_critical_p_value_{s_list[i]:.4f}.jpg", dpi=300
                )
            else:
                plt.show()

    return p_values_median, p_values_hi, p_values_lo


def find_crossing(
    scanned_var: np.array,
    ts: np.array,
    t_crits: np.array,
    method: str = "l",
):
    """
    Parameters
    ----------
    scanned_var
        Values of the variable that is being scanned
    ts
        Values of the test statistic at the scanned values
    t_crits
        The critical value or values of the test statistic
    interpolation_mode
        The mode in which to interpolate the crossing between ts and t_crits. "i" for index before crossing, "l" for linear, and "q" for quadratic are supported

    Notes
    -----
    It is important that ts and t_crits are on the same grid!
    This can handle two crossings in case of discovery!
    """

    # First check if t_crits is a scalar, then turn it into an array of the length of ts
    if isinstance(t_crits, float) or isinstance(t_crits, int):
        t_crits = np.full(len(ts), t_crits)

    diff = t_crits - ts - 0  # now we can find some zeros!
    crossing_idxs = []
    for i in range(0, len(diff) - 1, 1):
        if diff[i] <= 0 < diff[i + 1]:
            crossing_idxs.append(i)
        if diff[i] > 0 >= diff[i + 1]:
            crossing_idxs.append(i)

    # If index before crossing interpolation do the following:
    if method == "i":
        crossing_points = scanned_var[crossing_idxs]

    # If linear mode, treat the t_crits as linear interpolation, as well as the ts
    if (method == "l") or (method == "complex"):
        crossing_points = []
        for i in crossing_idxs:
            alpha = (ts[i + 1] - ts[i]) / (scanned_var[i + 1] - scanned_var[i])
            beta = (t_crits[i + 1] - t_crits[i]) / (scanned_var[i + 1] - scanned_var[i])
            intersection = (ts[i] - t_crits[i] + (beta - alpha) * scanned_var[i]) / (
                beta - alpha
            )
            crossing_points.append(intersection)

    if method == "q":
        raise NotImplementedError("Not yet implemented! Sorry!")

    if method == "complex":  # use the first pass guesses and get crazy with it
        f = interpolate.interp1d(scanned_var, diff)

        if len(crossing_points) >= 1:
            new_crosses = []
            for cross in crossing_points:
                # walk  forwards to find a change in sign
                new_sign = -1 * np.sign(f(np.amax([cross - 0.03, scanned_var[0]])))
                upper_limit = None
                for s in scanned_var[
                    scanned_var > np.amax([cross - 0.03, scanned_var[0]])
                ]:
                    if np.sign(f(s)) == new_sign:
                        upper_limit = s
                if upper_limit is None:
                    new_crosses.append(np.nan)

                sol = optimize.root_scalar(
                    f,
                    bracket=[np.amax([cross - 0.03, scanned_var[0]]), upper_limit],
                    method="brentq",
                )
                new_crosses.append(sol.root)

            crossing_points = new_crosses

    return crossing_points
