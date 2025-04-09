import matplotlib.pyplot as plt
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
WINDOW = np.array(constants.WINDOW)

WINDOWSIZE = 0.0
for i in range(len(WINDOW)):
    WINDOWSIZE += WINDOW[i][1] - WINDOW[i][0]

SEED = 42  # set the default random seed


@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    exp: float,
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
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    exp
        The exposure, in kg*yr
    window
        uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
        where edges of window are monotonically increasing (this is not checked), in keV.
        Default is typical analysis window.

    Notes
    -----
    This function computes the following:
    mu_S = eff * exp * S
    mu_B = exp * BI * windowsize
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp
    mu_B = exp * BI * WINDOWSIZE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * sigma**2)) + B_amp
        )

    return y


@nb.jit(**nb_kwd)
def nb_density(
    Es: np.array,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    exp: float,
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
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    exp
        The exposure, in kg*yr
    window
        uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
        where edges of window are monotonically increasing (this is not checked), in keV.
        Default is typical analysis window.

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = eff * exp * S
    mu_B = exp * BI * windowsize
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp
    mu_B = exp * BI * WINDOWSIZE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = S_amp * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * sigma**2)) + B_amp

    return mu_S + mu_B, y


@nb.jit(**nb_kwd)
def nb_logpdf(
    Es: np.array,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    exp: float,
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
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    exp
        The exposure, in kg*yr
    window
        uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
        where edges of window are monotonically increasing (this is not checked), in keV.
        Default is typical analysis window.

    Notes
    -----
    This function computes the following:
    mu_S = eff * exp * S
    mu_B = exp * BI * windowsize
    logpdf(E) = log(1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize])
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp
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
            S_amp * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * sigma**2)) + B_amp
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
    window
        uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
        where edges of window are monotonically increasing (this is not checked), in keV.
        Default is typical analysis window.

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """

    np.random.seed(seed)

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(np.random.normal(QBB + delta, sigma, size=n_sig), np.zeros(n_bkg))

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
    delta: float,
    sigma: float,
    eff: float,
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
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    seed
        specify a seed, otherwise uses default seed
    window
        uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
        where edges of window are monotonically increasing (this is not checked), in keV.
        Default is typical analysis window.

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.

    Returns
    -------
    rvs
        The random variables drawn
    (n_sig, n_bkg)
        Number of signal counts drawn and number of background counts
    """

    np.random.seed(seed)

    n_sig = np.random.poisson(S * eff * exp)
    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(np.random.normal(QBB + delta, sigma, size=n_sig), np.zeros(n_bkg))

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


@nb.jit(**nb_kwd)
def nb_density_gradient(
    Es: np.array,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    exp: float,
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
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    eff
        The global signal efficiency, unitless
    exp
        The exposure, in kg*yr
    window
        uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
        where edges of window are monotonically increasing (this is not checked), in keV.
        Default is typical analysis window.

    Notes
    -----
    This function computes the gradient of the density function and returns a tuple where the first element is the gradient of the CDF, and the second element is the gradient of the PDF.
    The first element has shape (K,) where K is the number of parameters, and the second element has shape (K,N) where N is the length of Es.
    mu_S = eff * exp * S
    mu_B = exp * BI * windowsize
    pdf(E) = [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize]
    cdf(E) = mu_S + mu_B
    """

    # mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp

    grad_CDF = np.array(
        [eff * exp, exp * WINDOWSIZE, 0, 0, S * exp, S * eff + BI * WINDOWSIZE]
    )

    grad_PDF = np.zeros(shape=(6, len(Es)))
    for i in nb.prange(Es.shape[0]):
        # For readability, don't precompute anything and see how performance is impacted
        grad_PDF[0][i] = (
            eff
            * exp
            * (1 / (np.sqrt(2 * np.pi) * sigma))
            * np.exp(-1 * (Es[i] - QBB - delta) ** 2 / (2 * sigma**2))
        )
        grad_PDF[1][i] = exp
        grad_PDF[2][i] = (
            ((Es[i] - QBB - delta) / sigma**2)
            * mu_S
            * (1 / (np.sqrt(2 * np.pi) * sigma))
            * np.exp(-1 * (Es[i] - QBB - delta) ** 2 / (2 * sigma**2))
        )
        grad_PDF[3][i] = (
            (
                ((Es[i] - QBB - delta) ** 2 - sigma**2)
                / (np.sqrt(2 * np.pi) * sigma**4)
            )
            * mu_S
            * np.exp(-1 * (Es[i] - QBB - delta) ** 2 / (2 * sigma**2))
        )
        grad_PDF[4][i] = (
            S
            * exp
            * (1 / (np.sqrt(2 * np.pi) * sigma))
            * np.exp(-1 * (Es[i] - QBB - delta) ** 2 / (2 * sigma**2))
        )
        grad_PDF[5][i] = (
            S
            * eff
            * (1 / (np.sqrt(2 * np.pi) * sigma))
            * np.exp(-1 * (Es[i] - QBB - delta) ** 2 / (2 * sigma**2))
            + BI
        )

    return grad_CDF, grad_PDF


class gaussian_on_uniform_gen:
    def __init__(self):
        self.parameters = inspectparameters(self.density)
        pass

    def pdf(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        return nb_pdf(Es, S, BI, delta, sigma, eff, exp)

    def logpdf(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        return nb_logpdf(Es, S, BI, delta, sigma, eff, exp)

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        return nb_density(Es, S, BI, delta, sigma, eff, exp)

    def density_gradient(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> tuple:
        return nb_density_gradient(Es, S, BI, delta, sigma, eff, exp)

    # for iminuit ExtendedUnbinnedNLL
    def log_density(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        mu_S = S * eff * exp
        mu_B = exp * BI * WINDOWSIZE

        # Do a quick check and return -inf if log args are negative
        if (mu_S + mu_B <= 0) or np.isnan(np.array([mu_S, mu_B])).any():
            return mu_S + mu_B, np.full(Es.shape[0], -np.inf)
        else:
            return (
                mu_S + mu_B,
                np.log(mu_S + mu_B) + nb_logpdf(Es, S, BI, delta, sigma, eff, exp),
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
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(S, BI, delta, sigma, eff, exp, seed=seed)

    def plot(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> None:
        y = nb_pdf(Es, S, BI, delta, sigma, eff, exp)

        plt.step(Es, y)
        plt.show()


gaussian_on_uniform = gaussian_on_uniform_gen()
