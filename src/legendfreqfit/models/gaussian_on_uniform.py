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
Qbb = constants.QBB
N_A = constants.NA
m_a = constants.MA


@nb.jit(**nb_kwd)
def nb_likelihood(
    Es: np.array,
    deltaE: float,
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
    deltaE
        The energy of the fit window
    S
        The signal rate, in units of counts/yr
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from Qbb, in keV
    sigma
        The energy resolution at Qbb, in keV
    eff
        The global signal efficiency
    exp
        The exposure, in kg*yr

    Notes
    -----
    This function computes the following:
    mu_S = ln(2) * (N_A/m_a) * eff * exp * S
    mu_B = exp * BI * deltaE
    pdf = prod_j {1/(mu_S+mu_B) * [mu_S * norm(E_j, Qbb + delta, sigma) + mu_B/deltaE]}
    """
    # Precompute the signal and background counts
    mu_S = np.log(2) * (N_A * S) * eff * exp / m_a
    mu_B = exp * BI * deltaE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = (np.log(2) * (N_A * S) * eff * exp) / (m_a * np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    L = 1
    for i in nb.prange(Es.shape[0]):
        L *= (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - Qbb - delta) ** 2) / (2 * sigma**2)) + B_amp
        )

    return L


@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    deltaE: float,
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
    deltaE
        The energy of the fit window
    S
        The signal rate, in units of counts/yr
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from Qbb, in keV
    sigma
        The energy resolution at Qbb, in keV
    eff
        The global signal efficiency
    exp
        The exposure, in kg*yr

    Notes
    -----
    This function computes the following:
    mu_S = ln(2) * (N_A/m_a) * eff * exp * S
    mu_B = exp * BI * deltaE
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, Qbb + delta, sigma) + mu_B/deltaE]
    """
    # Precompute the signal and background counts
    mu_S = np.log(2) * (N_A * S) * eff * exp / m_a
    mu_B = exp * BI * deltaE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = (np.log(2) * (N_A * S) * eff * exp) / (m_a * np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - Qbb - delta) ** 2) / (2 * sigma**2)) + B_amp
        )

    return y


@nb.jit(**nb_kwd)
def nb_logpdf(
    Es: np.array,
    deltaE: float,
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
    deltaE
        The energy of the fit window
    S
        The signal rate, in units of counts/yr
    BI
        The background index rate, in counts/(kg*yr*keV)
    delta
        Systematic energy offset from Qbb, in keV
    sigma
        The energy resolution at Qbb, in keV
    eff
        The global signal efficiency
    exp
        The exposure, in kg*yr

    Notes
    -----
    This function computes the following:
    mu_S = ln(2) * (N_A/m_a) * eff * exp * S
    mu_B = exp * BI * deltaE
    logpdf(E) = log(1/(mu_S+mu_B) * [mu_S * norm(E_j, Qbb + delta, sigma) + mu_B/deltaE])
    """
    # Precompute the signal and background counts
    mu_S = np.log(2) * (N_A * S) * eff * exp / m_a
    mu_B = exp * BI * deltaE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = (np.log(2) * (N_A * S) * eff * exp) / (m_a * np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = np.log(
            (1 / (mu_S + mu_B))
            * (S_amp * np.exp(-((Es[i] - Qbb - delta) ** 2) / (2 * sigma**2)) + B_amp)
        )

    return y


class gaussian_on_uniform_gen:
    def __init__(self):
        pass

    def pdf(
        self,
        Es: np.array,
        deltaE: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        return nb_pdf(Es, deltaE, S, BI, delta, sigma, eff, exp)

    def logpdf(
        self,
        Es: np.array,
        deltaE: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> np.array:
        return nb_logpdf(Es, deltaE, S, BI, delta, sigma, eff, exp)

    def likelihood(
        self,
        Es: np.array,
        deltaE: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> float:
        return nb_likelihood(Es, deltaE, S, BI, delta, sigma, eff, exp)

    def plot(
        self,
        Es: np.array,
        deltaE: float,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        exp: float,
    ) -> None:
        y = nb_pdf(Es, deltaE, S, BI, delta, sigma, eff, exp)

        plt.step(Es, y)
        plt.show()


gaussian_on_uniform = gaussian_on_uniform_gen()
