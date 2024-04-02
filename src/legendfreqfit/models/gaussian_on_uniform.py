import matplotlib.pyplot as plt
import numba as nb
import numpy as np

import legendfreqfit.constants as constants

nb_kwd = {
    "nopython": True,
    "parallel": True,
    "nogil": True,
    "cache": True,
    "fastmath": True,
}

QBB = constants.QBB
N_A = constants.NA
M_A = constants.MA
SEED = 42  # set the default random seed


@nb.jit(**nb_kwd)
def nb_likelihood(
    Es: np.array,
    window: float,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    exp: float,
) -> float:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    window
        width of the fit window in keV
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

    Notes
    -----
    This function computes the following:
    mu_S = ln(2) * (N_A/M_A) * eff * exp * S
    mu_B = exp * BI * window
    pdf = prod_j {1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/window]}
    """
    # Precompute the signal and background counts
    #mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp
    mu_B = exp * BI * window

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S /(np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    likelihood = 1
    for i in nb.prange(Es.shape[0]):
        likelihood *= (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * sigma**2)) + B_amp
        )

    return likelihood


@nb.jit(**nb_kwd)
def nb_loglikelihood(
    Es: np.array,
    window: float,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    exp: float,
) -> float:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    window
        width of the fit window in keV
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

    Notes
    -----
    This function computes the following:
    mu_S = ln(2) * (N_A/M_A) * eff * exp * S
    mu_B = exp * BI * window
    pdf = prod_j {1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/window]}
    """
    # Precompute the signal and background counts
    #mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp
    mu_B = exp * BI * window

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S /(np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    loglikelihood = 0
    for i in nb.prange(Es.shape[0]):
        loglikelihood += -1.0 * np.log(mu_S + mu_B) + np.log(
            S_amp * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * sigma**2)) + B_amp
        )

    return loglikelihood


@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    window: float,
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
    window
        width of the fit window in keV
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

    Notes
    -----
    This function computes the following:
    mu_S = ln(2) * (N_A/M_A) * eff * exp * S
    mu_B = exp * BI * window
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/window]
    """
    # Precompute the signal and background counts
    #mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp
    mu_B = exp * BI * window

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S /(np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * sigma**2)) + B_amp
        )

    return y


@nb.jit(**nb_kwd)
def nb_logpdf(
    Es: np.array,
    window: float,
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
    window
        width of the fit window in keV
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

    Notes
    -----
    This function computes the following:
    mu_S = ln(2) * (N_A/M_A) * eff * exp * S
    mu_B = exp * BI * window
    logpdf(E) = log(1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/window])
    """
    # Precompute the signal and background counts
    #mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * eff * exp
    mu_B = exp * BI * window

    if sigma == 0:  # need this check for fitting
        return np.full_like(
            Es, np.log(exp * BI / (mu_S + mu_B))
        )  # TODO: make sure this simplification makes sense in the limit

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S /(np.sqrt(2 * np.pi) * sigma)

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

# need to allow for removing parts of the range for BI events
@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_rvs(
    n_sig: int,
    n_bkg: int,
    E_lo: float,
    E_hi: float,
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
    E_lo
        The lower energy bound of the BI window
    E_hi
        The high energy bound of the BI window
    delta
        Systematic energy offset from QBB, in keV
    sigma
        The energy resolution at QBB, in keV
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    """
    
    np.random.seed(seed)
    
    # Get energy of signal events from a Gaussian distribution
    sig = np.random.normal(QBB + delta, sigma, size=n_sig)

    # Get energy of background events from a uniform distribution
    Es = np.append(sig, np.random.uniform(E_lo, E_hi, size=n_bkg))

    return Es


class gaussian_on_uniform_gen:
    def __init__(self):
        pass

    def pdf(
        self,
        Es: np.array,
        window: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        return nb_pdf(Es, window, S, BI, delta, sigma, eff, exp)

    def logpdf(
        self,
        Es: np.array,
        window: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        return nb_logpdf(Es, window, S, BI, delta, sigma, eff, exp)
    
    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        window: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        mu_S = S * eff * exp
        mu_B = exp * BI * window
        return (mu_S+mu_B, (mu_S+mu_B)*nb_pdf(Es, window, S, BI, delta, sigma, eff, exp))

    def likelihood(
        self,
        Es: np.array,
        window: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> float:
        return nb_likelihood(Es, window, S, BI, delta, sigma, eff, exp)

    def loglikelihood(
        self,
        Es: np.array,
        window: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> float:
        return nb_loglikelihood(Es, window, S, BI, delta, sigma, eff, exp)
    
    # need to allow for removing parts of the range for BI events
    def rvs(
        self,
        n_sig: int,
        n_bkg: int,
        E_lo: float,
        E_hi: float,
        delta: float,
        sigma: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_rvs(n_sig, n_bkg, E_lo, E_hi, delta, sigma, seed)

    def plot(
        self,
        Es: np.array,
        window: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> None:
        y = nb_pdf(Es, window, S, BI, delta, sigma, eff, exp)

        plt.step(Es, y)
        plt.show()


gaussian_on_uniform = gaussian_on_uniform_gen()
