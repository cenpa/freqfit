"""
This model follows the MJD analysis for the peak shape. The peak shape is modeled as the sum of a full-energy Gaussian
component and an exponentially modified Gaussian tail to approximate the peak shape distortion due to incomplete charge
collection. See S. I. Alvis et al., Phys. Rev. C 100, 025501 (2019) for details. In addition, a slightly different
analysis window is used, again following MJD.

Also see J.M. Lopez-Castano and Ping-Han Chu. Energy systematic of majorana demonstrator. Technical report, 2022.
Internal Report, Feb. 9, 2022. for an explanation of the peak shape used.
"""

import sys
from math import erfc

import numba as nb
import numpy as np

import freqfit.models.constants as constants
from freqfit.utils import inspectparameters

nb_kwd = {
    "nopython": True,
    "parallel": False,
    "nogil": True,
    "cache": True,
    "fastmath": True,
    "inline": "always",
}

QBB = constants.QBB
N_A = constants.NA
M_A = constants.MDET

# default analysis window and width
# window
#     uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
#     where edges of window are monotonically increasing (this is not checked), in keV.
#     Default is typical analysis window.
WINDOW = np.array(constants.MJD_WINDOW)

WINDOWSIZE = 0.0
for i in range(len(WINDOW)):
    WINDOWSIZE += WINDOW[i][1] - WINDOW[i][0]

SEED = 42  # set the default random seed

limit = np.log(sys.float_info.max) / 10


# yoinked from https://github.com/legend-exp/pygama/blob/main/src/pygama/math/functions/gauss.py#L72
@nb.jit(**nb_kwd)
def nb_gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    Normalised Gaussian PDF, w/ args: mu, sigma. The support is :math:`(-\infty, \infty)`

    .. math::
        pdf(x, \mu, \sigma) = \frac{1}{\sqrt{2\pi}}e^{(\frac{x-\mu}{\sigma}^2)/2}

    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    """

    if sigma == 0:
        invs = np.inf
    else:
        invs = 1.0 / sigma
    z = (x - mu) * invs
    invnorm = invs / np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z**2) * invnorm


# yoinked from https://github.com/legend-exp/pygama/blob/main/src/pygama/math/functions/exgauss.py
@nb.jit(**nb_kwd)
def nb_gauss_tail_exact(
    x: float, mu: float, sigma: float, tau: float, tmp: float
) -> float:
    r"""
    Exact form of a normalized exponentially modified Gaussian PDF.
    It computes the following PDF:


    .. math::
        pdf(x, \tau,\mu,\sigma) = \frac{1}{2|\tau|}e^{\frac{x-\mu}{\tau}+\frac{\sigma^2}{2\tau^2}}\text{erfc}\left(\frac{\tau(\frac{x-\mu}{\sigma})+\sigma}{|\tau|\sqrt{2}}\right)


    Where :math:`tmp = \frac{x-\mu}{\tau}+\frac{\sigma^2}{2\tau^2}` is precomputed in :func:`nb_exgauss_pdf` to save computational time.


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    tau
        The characteristic scale of the Gaussian tail
    tmp
        The scaled version of the exponential argument


    See Also
    --------
    :func:`nb_exgauss_pdf`
    """

    abstau = np.absolute(tau)
    if tmp < limit:
        tmp = tmp
    else:
        tmp = limit
    if sigma == 0 or abstau == 0:
        return x * 0
    z = (x - mu) / sigma
    tail_f = (
        (1 / (2 * abstau))
        * np.exp(tmp)
        * erfc((tau * z + sigma) / (np.sqrt(2) * abstau))
    )
    return tail_f


# yoinked from https://github.com/legend-exp/pygama/blob/main/src/pygama/math/functions/exgauss.py
@nb.jit(**nb_kwd)
def nb_gauss_tail_approx(
    x: np.ndarray, mu: float, sigma: float, tau: float
) -> np.ndarray:
    r"""
    Approximate form of a normalized exponentially modified Gaussian PDF
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    tau
        The characteristic scale of the Gaussian tail


    See Also
    --------
    :func:`nb_exgauss_pdf`
    """
    if sigma == 0:
        return x * 0
    elif (sigma + tau * (x - mu) / sigma) == 0:
        return x * 0
    den = 1 / (sigma + tau * (x - mu) / sigma)
    tail_f = sigma * nb_gauss_pdf(x, mu, sigma) * den * (1.0 - tau * tau * den * den)
    return tail_f


# yoinked from https://github.com/legend-exp/pygama/blob/main/src/pygama/math/functions/exgauss.py
@nb.jit(**nb_kwd)
def nb_exgauss_pdf(x: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
    r"""
    Normalized PDF of an exponentially modified Gaussian distribution. Its range of support is :math:`x\in(-\infty,\infty)`, :math:`\tau\in(-\infty,\infty)`
    Calls either :func:`nb_gauss_tail_exact` or :func:`nb_gauss_tail_approx` depending on which is computationally cheaper


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    tau
        The characteristic scale of the Gaussian tail


    See Also
    --------
    :func:`nb_gauss_tail_exact`, :func:`nb_gauss_tail_approx`
    """

    x = np.asarray(x)
    tail_f = np.empty_like(x, dtype=np.float64)

    for i in nb.prange(x.shape[0]):
        if tau == 0:
            tail_f[i] = np.nan
        else:
            tmp = ((x[i] - mu) / tau) + ((sigma**2) / (2 * tau**2))
            if tmp < limit:
                tail_f[i] = nb_gauss_tail_exact(x[i], mu, sigma, tau, tmp)
            else:
                tail_f[i] = nb_gauss_tail_approx(x[i], mu, sigma, tau)
    return tail_f


@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    S: float,
    BI: float,
    frac: float,
    delta: float,
    sigma: float,
    tau: float,
    gamma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    check_window: bool = False,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    S
        The signal rate, in units of counts/(kg*yr)
    BI
        The background index rate, in counts/(kg*yr*keV)
    frac
        portion of the peak in the tail
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    tau
        scale parameter of the tail in keV
    gamma
        scaling parameter for tau and sigma
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + mu, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp_gauss = mu_S / (np.sqrt(2 * np.pi) * gamma * sigma) * (1 - frac)

    exgaus = mu_S * frac * nb_exgauss_pdf(Es, QBB + delta, gamma * sigma, gamma * tau)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            (
                exgaus[i]
                + S_amp_gauss
                * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * (gamma * sigma) ** 2))
            )
            + B_amp
        )

    if check_window:
        for i in nb.prange(Es.shape[0]):
            inwindow = False
            for j in range(len(WINDOW)):
                if WINDOW[j][0] <= Es[i] <= WINDOW[j][1]:
                    inwindow = True
            if not inwindow:
                y[i] = 0.0

    return y


@nb.jit(**nb_kwd)
def nb_density(
    Es: np.array,
    S: float,
    BI: float,
    frac: float,
    delta: float,
    sigma: float,
    tau: float,
    gamma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    check_window: bool = False,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    S
        The signal rate, in units of counts/(kg*yr)
    BI
        The background index rate, in counts/(kg*yr*keV)
    frac
        portion of the peak in the tail
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    tau
        scale parameter of the tail in keV
    gamma
        scaling parameter for tau and sigma
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB + mu, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp_gauss = mu_S / (np.sqrt(2 * np.pi) * gamma * sigma) * (1.0 - frac)

    exgaus = mu_S * frac * nb_exgauss_pdf(Es, QBB + delta, gamma * sigma, gamma * tau)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (
            exgaus[i]
            + S_amp_gauss
            * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * (gamma * sigma) ** 2))
        ) + B_amp

    if check_window:
        for i in nb.prange(Es.shape[0]):
            inwindow = False
            for j in range(len(WINDOW)):
                if WINDOW[j][0] <= Es[i] <= WINDOW[j][1]:
                    inwindow = True
            if not inwindow:
                y[i] = 0.0

    return mu_S + mu_B, y


@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_rvs(
    n_sig: int,
    n_bkg: int,
    frac: float,
    delta: float,
    sigma: float,
    tau: float,
    gamma: float,
    seed: int = SEED,
) -> np.array:
    """
    Parameters
    ----------
    n_sig
        Number of signal events to pull from
    n_bkg
        Number of background events to pull from
    frac
        portion of the peak in the tail
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    tau
        scale parameter of the tail in keV
    gamma
        scaling parameter for tau and sigma
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """

    np.random.seed(seed)

    # preallocate for draws
    Es = np.append(np.zeros(n_sig), np.zeros(n_bkg))

    # get signal events from Gaussian with tail
    # first, draw to determine whether from Gaussian or exgaus
    which = np.random.uniform(0, 1, n_sig)
    # draw Gaussian smearing
    smear = np.random.normal(0, gamma * sigma, size=n_sig)
    # draw exponential
    exp = np.random.exponential(scale=gamma * tau, size=n_sig)

    # depending on whether the event should fall in the tail, subtract some energy from Qbb before Gaussian smearing
    for i in range(n_sig):
        if which[i] < frac:
            Es[i] = QBB + delta - exp[i] + smear[i]
        else:
            Es[i] = QBB + delta + smear[i]

    # Get background events from a uniform distribution
    bkg = np.random.uniform(0, 1, n_bkg)

    breaks = np.zeros(shape=(len(WINDOW), 2))
    for i in range(len(WINDOW)):
        thiswidth = WINDOW[i][1] - WINDOW[i][0]

        if i > 0:
            breaks[i][0] = breaks[i - 1][1]

        if i < len(WINDOW):
            breaks[i][1] = breaks[i][0] + thiswidth / WINDOWSIZE

        for j in range(len(bkg)):
            if breaks[i][0] <= bkg[j] <= breaks[i][1]:
                Es[n_sig + j] = (bkg[j] - breaks[i][0]) * thiswidth / (
                    breaks[i][1] - breaks[i][0]
                ) + WINDOW[i][0]

    return Es


@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_extendedrvs(
    S: float,
    BI: float,
    frac: float,
    delta: float,
    sigma: float,
    tau: float,
    gamma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    seed: int = SEED,
) -> np.array:
    """
    Parameters
    ----------
    S
        expected rate of signal events in events/(kg*yr)
    BI
        rate of background events in events/(kev*kg*yr)
    frac
        portion of the peak in the tail
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    tau
        scale parameter of the tail in keV
    gamma
        scaling parameter for tau and sigma
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """

    n_sig = np.random.poisson(S * (eff + effuncscale * effunc) * exp)
    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    return nb_rvs(n_sig, n_bkg, frac, delta, sigma, tau, gamma, seed=seed), (n_sig, n_bkg)


class mjd_0vbb_gen:
    def __init__(self):
        self.parameters = inspectparameters(self.density)
        del self.parameters["check_window"]  # this should not be seen by iminuit

    def pdf(
        self,
        Es: np.array,
        S: float,
        BI: float,
        frac: float,
        delta: float,
        sigma: float,
        tau: float,
        gamma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_pdf(
            Es,
            S,
            BI,
            frac,
            delta,
            sigma,
            tau,
            gamma,
            eff,
            effunc,
            effuncscale,
            exp,
            check_window,
        )

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        S: float,
        BI: float,
        frac: float,
        delta: float,
        sigma: float,
        tau: float,
        gamma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_density(
            Es,
            S,
            BI,
            frac,
            delta,
            sigma,
            tau,
            gamma,
            eff,
            effunc,
            effuncscale,
            exp,
            check_window,
        )

    # should we have an rvs method for drawing a random number of events?
    # `extendedrvs`
    # needs to use same parameters as the rest of the functions...
    def rvs(
        self,
        n_sig: int,
        n_bkg: int,
        frac: float,
        delta: float,
        sigma: float,
        tau: float,
        gamma: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_rvs(n_sig, n_bkg, frac, delta, sigma, tau, gamma, seed=seed)

    def extendedrvs(
        self,
        S: float,
        BI: float,
        frac: float,
        delta: float,
        sigma: float,
        tau: float,
        gamma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(
            S, BI, frac, delta, sigma, tau, gamma, eff, effunc, effuncscale, exp, seed=seed
        )

    # function call needs to take the same parameters as the other function calls, in the same order repeated twice
    # order is Es, S, BI, frac, delta, sigma, tau, gamma, eff, effunc, effuncscale, exp, check_window
    # this is intended only for empty datasets
    # returns `None` if we couldn't combine the datasets (a dataset was not empty)
    def combine(
        self,
        a_Es: np.array,
        a_S: float,
        a_BI: float,
        a_frac: float,
        a_delta: float,
        a_sigma: float,
        a_tau: float,
        a_gamma: float,
        a_eff: float,
        a_effunc: float,
        a_effuncscale: float,
        a_exp: float,
        b_Es: np.array,
        b_S: float,
        b_BI: float,
        b_frac: float,
        b_delta: float,
        b_sigma: float,
        b_tau: float,
        b_gamma: float,
        b_eff: float,
        b_effunc: float,
        b_effuncscale: float,
        b_exp: float,
    ) -> list | None:
        # datasets must be empty to be combined
        if len(a_Es) != 0 or len(b_Es) != 0:
            return None

        Es = np.array([])  # both of these datasets are empty
        S = 0.0  # this should be overwritten in the fit later
        BI = 0.0  # this should be overwritten in the fit later

        exp = a_exp + b_exp  # total exposure

        # exposure weighted fixed parameters (important to calculate correctly)
        sigma = (a_exp * a_sigma + b_exp * b_sigma) / exp
        eff = (a_exp * a_eff + b_exp * b_eff) / exp
        delta = (a_exp * a_delta + b_exp * b_delta) / exp
        tau = (a_exp * a_tau + b_exp * b_tau) / exp
        gamma = (a_exp * a_gamma + b_exp * b_gamma) / exp
        frac = (a_exp * a_frac + b_exp * b_frac) / exp

        # these are fully correlated in this model so the direct sum is appropriate
        # (maybe still appropriate even if not fully correlated?)
        effunc = (a_exp * a_effunc + b_exp * b_effunc) / exp

        effuncscale = 0.0  # this should be overwritten in the fit later

        return [Es, S, BI, frac, delta, sigma, tau, gamma, eff, effunc, effuncscale, exp]

    def can_combine(
        self,
        a_Es: np.array,
        a_S: float,
        a_BI: float,
        a_frac: float,
        a_delta: float,
        a_sigma: float,
        a_tau: float,
        a_gamma: float,
        a_eff: float,
        a_effunc: float,
        a_effuncscale: float,
        a_exp: float,
    ) -> bool:
        """
        This sets an arbitrary rule if this dataset can be combined with other datasets.
        In this case, if the dataset contains no data, then it can be combined, but more complex rules can be imposed.
        """
        if len(a_Es) == 0:
            return True
        else:
            return False


mjd_0vbb = mjd_0vbb_gen()
