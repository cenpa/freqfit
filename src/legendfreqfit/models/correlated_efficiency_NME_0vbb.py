"""
Define a class whose input is the effective Majorana mass and incorporates uncertainty on the NME
"""
import numba as nb
import numpy as np

import legendfreqfit.models.constants as constants
from legendfreqfit.utils import inspectparameters

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
M_A = constants.M76

# default analysis window and width
# window
#     uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
#     where edges of window are monotonically increasing (this is not checked), in keV.
#     Default is typical analysis window.
WINDOW = np.array(constants.WINDOW)

WINDOWSIZE = 0.0
for i in range(len(WINDOW)):
    WINDOWSIZE += WINDOW[i][1] - WINDOW[i][0]

SEED = 42  # set the default random seed


@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    m_bb: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    NME: float,
    check_window: bool = False,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    m_bb
        The effective neutrino mass
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    NME
        The nuclear matrix element
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * m_bb^2 * NME^2
    mu_B = exp * BI * windowsize
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    mu_S = m_bb**2 * NME**2 * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2)) + B_amp
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
    m_bb: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    NME: float,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    m_bb
        The effective Majorana mass
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    NME
        The nuclear matrix element

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * m_bb^2 * NME^2
    mu_B = exp * BI * windowsize
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB - delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts

    mu_S = m_bb**2 * NME**2 * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    if sigma == 0:
        return np.inf, np.full_like(Es, np.inf, dtype=np.float64)

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2)) + B_amp

    return mu_S + mu_B, y


@nb.jit(**nb_kwd)
def nb_log_density(
    Es: np.array,
    m_bb: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    NME: float,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    m_bb
        The effective Majorana mass
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    NME
        The nuclear matrix element

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * m_bb^2 * NME
    mu_B = exp * BI * windowsize
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB - delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    mu_S = m_bb**2 * NME**2 * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    if sigma == 0:
        return np.inf, np.full_like(Es, np.inf, dtype=np.float64)

    if mu_S + mu_B < 0:
        return 0, np.full_like(Es, -np.inf)

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        pdf = S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2)) + B_amp

        if pdf <= 0:
            y[i] = -np.inf

        # Make an approximation based on machine precision, following J. Detwiler's suggestion
        u = (S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2))) / B_amp

        if u <= 1e-8:
            y[i] = np.log(B_amp) + u
        else:
            y[i] = np.log(pdf)

    return mu_S + mu_B, y


@nb.jit(**nb_kwd)
def nb_density_gradient(
    Es: np.array,
    m_bb: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    NME: float,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    m_bb
        The effective Majorana mass
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    NME
        The nuclear matrix element

    Notes
    -----
    This function computes the gradient of the density function and returns a tuple where the first element is the gradient of the CDF, and the second element is the gradient of the PDF.
    The first element has shape (K,) where K is the number of parameters, and the second element has shape (K,N) where N is the length of Es.
    mu_S = S * exp * (eff + effuncscale * effunc)
    mu_B = exp * BI * windowsize
    pdf(E) = [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize]
    cdf(E) = mu_S + mu_B
    """
    raise NotImplementedError("Not yet implemented!")


@nb.jit(**nb_kwd)
def nb_logpdf(
    Es: np.array,
    m_bb: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    NME: float,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    m_bb
        The effective Majorana mass
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    NME
        The nuclear matrix element

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * m_bb**2 * NME
    mu_B = exp * BI * windowsize
    logpdf(E) = log(1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta, sigma) + mu_B/windowsize])
    """

    # Precompute the signal and background counts
    mu_S = m_bb**2 * NME**2 * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    if sigma == 0:  # need this check for fitting
        return np.full_like(
            Es, np.log(exp * BI / (mu_S + mu_B))
        )  # TODO: make sure this simplification makes sense in the limit

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        pdf = (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2)) + B_amp
        )

        if pdf <= 0:
            y[i] = -np.inf
        else:
            y[i] = np.log(pdf)

    return y


@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_rvs(
    n_sig: int,
    n_bkg: int,
    delta: float,
    sigma: float,
    seed: int = SEED,
) -> np.array:
    """
    Parameters
    ----------
    n_sig
        Number of signal events to pull from
    n_bkg
        Number of background events to pull from
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """

    np.random.seed(seed)

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(np.random.normal(QBB - delta, sigma, size=n_sig), np.zeros(n_bkg))

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
    m_bb: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    NME: float,
    seed: int = SEED,
) -> np.array:
    """
    Parameters
    ----------
    m_bb
        expected effective Majorana mass
    BI
        rate of background events in events/(kev*kg*yr)
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    NME
        The nuclear matrix element
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """

    np.random.seed(seed)

    n_sig = np.random.poisson(m_bb**2 * NME**2 * (eff + effuncscale * effunc) * exp)
    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(np.random.normal(QBB - delta, sigma, size=n_sig), np.zeros(n_bkg))

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

    return Es, (n_sig, n_bkg)


class correlated_efficiency_NME_0vbb_gen:  # noqa: N816
    def __init__(self):
        self.parameters = inspectparameters(self.density)
        pass

    def pdf(
        self,
        Es: np.array,
        m_bb: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        NME: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_pdf(
            Es, m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME, check_window
        )

    def logpdf(
        self,
        Es: np.array,
        m_bb: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        NME: float,
    ) -> np.array:
        return nb_logpdf(Es, m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME)

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        m_bb: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        NME: float,
    ) -> np.array:
        return nb_density(
            Es, m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME
        )

    # for iminuit ExtendedUnbinnedNLL
    def density_gradient(
        self,
        Es: np.array,
        m_bb: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        NME: float,
    ) -> np.array:
        return nb_density_gradient(
            Es, m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME
        )

    # for iminuit ExtendedUnbinnedNLL
    def log_density(
        self,
        Es: np.array,
        m_bb: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        NME: float,
    ) -> np.array:
        return nb_log_density(
            Es, m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME
        )

    # should we have an rvs method for drawing a random number of events?
    # `extendedrvs`
    # needs to use same parameters as the rest of the functions...
    def rvs(
        self,
        n_sig: int,
        n_bkg: int,
        delta: float,
        sigma: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_rvs(n_sig, n_bkg, delta, sigma, seed=seed)

    def extendedrvs(
        self,
        m_bb: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        NME: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(
            m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME, seed=seed
        )

    def plot(
        self,
        Es: np.array,
        m_bb: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
        NME: float,
    ) -> None:
        y = nb_pdf(Es, m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME)

        import matplotlib.pyplot as plt

        plt.step(Es, y)
        plt.show()

    # function call needs to take the same parameters as the other function calls, in the same order repeated twice
    # this is intended only for empty datasets
    # returns `None` if we couldn't combine the datasets (a dataset was not empty)
    def combine(
        self,
        a_Es: np.array,
        a_m_bb: float,
        a_BI: float,
        a_delta: float,
        a_sigma: float,
        a_eff: float,
        a_effunc: float,
        a_effuncscale: float,
        a_exp: float,
        a_NME: float,
        b_Es: np.array,
        b_m_bb: float,
        b_BI: float,
        b_delta: float,
        b_sigma: float,
        b_eff: float,
        b_effunc: float,
        b_effuncscale: float,
        b_exp: float,
        b_NME: float,
    ) -> list | None:
        # datasets must be empty to be combined
        if len(a_Es) != 0 or len(b_Es) != 0:
            return None

        Es = np.array([])  # both of these datasets are empty
        m_bb = 0.0  # this should be overwritten in the fit later
        BI = 0.0  # this should be overwritten in the fit later

        exp = a_exp + b_exp  # total exposure

        # exposure weighted fixed parameters (important to calculate correctly)
        sigma = (a_exp * a_sigma + b_exp * b_sigma) / exp
        eff = (a_exp * a_eff + b_exp * b_eff) / exp
        delta = (a_exp * a_delta + b_exp * b_delta) / exp

        # these are fully correlated in this model so the direct sum is appropriate
        # (maybe still appropriate even if not fully correlated?)
        effunc = (a_exp * a_effunc + b_exp * b_effunc) / exp

        effuncscale = 0.0  # this should be overwritten in the fit later
        NME = 0.0  # this should be overwritten in the fit later

        return [Es, m_bb, BI, delta, sigma, eff, effunc, effuncscale, exp, NME]

    def can_combine(
        self,
        a_Es: np.array,
        a_m_bb: float,
        a_BI: float,
        a_delta: float,
        a_sigma: float,
        a_eff: float,
        a_effunc: float,
        a_effuncscale: float,
        a_exp: float,
        a_NME: float,
    ) -> bool:
        """
        This sets an arbitrary rule if this dataset can be combined with other datasets.
        In this case, if the dataset contains no data, then it can be combined, but more complex rules can be imposed.
        """
        if len(a_Es) == 0:
            return True
        else:
            return False


correlated_efficiency_NME_0vbb = correlated_efficiency_NME_0vbb_gen()  # noqa: N816
