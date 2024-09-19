import matplotlib.pyplot as plt
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
}

QBB = constants.QBB
N_A = constants.NA
M_A = constants.MA

# window
#     must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
#     where edges of window are monotonically increasing (this is not checked), in keV.
#     Default is typical analysis window.

# default analysis window and width
WINDOW = np.array(constants.WINDOW)

WINDOWSIZE = 0.0
for i in range(len(WINDOW)):
    WINDOWSIZE += WINDOW[i][1] - WINDOW[i][0]

SEED = 42  # set the default random seed


@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV

    Notes
    -----
    This function makes an approximation to the model `gaussian_on_uniform` and assumes that the peak has zero
    contribution to the pdf to simplify computation. Because of this, we assume that there are no signal events in the
    window and therefore the normalization should be only to the expected number of background events in the window. Note that
    whether the events are properly in the window is not checked!

    From `gaussian_on_uniform`:
    mu_S = eff * exp * S
    mu_B = exp * BI * windowsize
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize]

    But instead in this pdf we should have
    # mu_S = eff * exp * S <-- not needed, so do not need these parameters
    mu_B = exp * BI * windowsize
    # pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize] <-- we consider a different window so we should have instead
    pdf(E) = 1/(mu_B) * [mu_B / windowsize] --> 1 / windowsize

    So in fact we need no parameters other than the windowsize. We take this as a global constant however to speed up computation further!
    """

    # windowsize = 0.0
    # for i in range(len(window)):
    #     windowsize += window[i][1] - window[i][0]

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    # mu_S = S * eff * exp
    # mu_B = exp * BI * windowsize

    # Precompute the prefactors so that way we save multiplications in the for loop
    # B_amp = exp * BI
    # S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    # y = np.empty_like(Es, dtype=np.float64)
    # for i in nb.prange(Es.shape[0]):
    #     y[i] = (1 / (mu_S + mu_B)) * (
    #         S_amp * np.exp(-((Es[i] - QBB - delta) ** 2) / (2 * sigma**2)) + B_amp
    #     )

    y = np.full_like(Es, fill_value=1.0 / WINDOWSIZE, dtype=np.float64)

    return y


@nb.jit(**nb_kwd)
def nb_density(
    Es: np.array,
    BI: float,
    exp: float,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    BI
        The background index rate, in counts/(kg*yr*keV)
    exp
        The exposure, in kg*yr
    """

    # expected number of background events
    mu_B = exp * BI * WINDOWSIZE

    y = np.full_like(Es, fill_value=exp * BI, dtype=np.float64)

    return mu_B, y


@nb.jit(**nb_kwd)
def nb_logpdf(
    Es: np.array,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    """

    y = np.full_like(Es, fill_value=np.log(1.0 / WINDOWSIZE), dtype=np.float64)

    return y


@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_extendedrvs(
    BI: float,
    exp: float,
    seed: int = SEED,
) -> np.array:
    """
    Parameters
    ----------
    BI
        rate of background events in events/(kev*kg*yr)
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """

    np.random.seed(seed)

    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # preallocate for background draws
    Es = np.zeros(n_bkg)

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
                Es[j] = (bkg[j] - breaks[i][0]) * thiswidth / (
                    breaks[i][1] - breaks[i][0]
                ) + WINDOW[i][0]

    return Es, (0, n_bkg)


class bkg_region_0vbb_gen:
    def __init__(self):
        self.parameters = inspectparameters(self.density)
        pass

    def pdf(
        self,
        Es: np.array,
    ) -> np.array:
        return nb_pdf(Es)

    def logpdf(
        self,
        Es: np.array,
    ) -> np.array:
        return nb_logpdf(Es)

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        BI: float,
        exp: float,
    ) -> np.array:
        return nb_density(Es, BI, exp)

    def extendedrvs(
        self,
        BI: float,
        exp: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(BI, exp, seed=seed)

    def plot(
        self,
        Es: np.array,
    ) -> None:
        y = nb_pdf(Es)

        plt.step(Es, y)
        plt.show()


bkg_region_0vbb = bkg_region_0vbb_gen()
