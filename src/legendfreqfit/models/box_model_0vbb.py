"""
0vbb ROI is treated as a flat rectangle on a flat background. Built in order to understand the tailing effects of the Gaussian signal model.
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
ROI_WIDTH = 5  # width of the ROI in sigma

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
    S: float,
    BI: float,
    delta: float,
    sigma: float,
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
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    ROI = ROI_WIDTH * sigma
    mu_S = S * (eff + effuncscale * effunc) * exp * ROI
    mu_B = exp * BI * WINDOWSIZE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = S * (eff + effuncscale * effunc) * exp

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        if (Es[i] - QBB + delta > -1 * (ROI_WIDTH * sigma / 2)) & (
            Es[i] - QBB + delta < (ROI_WIDTH * sigma / 2)
        ):
            y[i] = (1 / (mu_S + mu_B)) * (S_amp + B_amp)
        else:
            y[i] = (1 / (mu_S + mu_B)) * (B_amp)

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
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
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
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB - delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    # S *= 0.01
    # BI *= 0.0001
    ROI = ROI_WIDTH * sigma
    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    if sigma == 0:
        return np.inf, np.full_like(Es, np.inf, dtype=np.float64)

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = S * (eff + effuncscale * effunc) * exp

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        if (Es[i] - QBB + delta > -1 * (ROI / 2)) & (Es[i] - QBB + delta < (ROI / 2)):
            y[i] = S_amp / ROI + B_amp
        else:
            y[i] = B_amp

    return mu_S + mu_B, y


@nb.jit(**nb_kwd)
def nb_density_gradient(
    Es: np.array,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
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
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr

    Notes
    -----
    This function computes the gradient of the density function and returns a tuple where the first element is the gradient of the CDF, and the second element is the gradient of the PDF.
    The first element has shape (K,) where K is the number of parameters, and the second element has shape (K,N) where N is the length of Es.
    mu_S = S * exp * (eff + effuncscale * effunc)
    mu_B = exp * BI * windowsize
    pdf(E) = [mu_S * norm(E_j, QBB + delta, sigma) + mu_B/windowsize]
    cdf(E) = mu_S + mu_B
    """

    # mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    raise NotImplementedError("Not implemented yet")


@nb.jit(**nb_kwd)
def nb_logpdf(
    Es: np.array,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
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
    effunc
        uncertainty on the efficiency
    effuncscale
        scaling parameter of the efficiency
    exp
        The exposure, in kg*yr

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    logpdf(E) = log(1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta, sigma) + mu_B/windowsize])
    """
    raise NotImplementedError("Not yet implemented! sorry!")


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
    raise NotImplementedError("Not implemented! Sorry!")


@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_extendedrvs(
    S: float,
    BI: float,
    delta: float,
    sigma: float,
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
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """
    # S *= 0.01
    # BI *= 0.0001
    ROI = ROI_WIDTH * sigma

    np.random.seed(seed)

    n_sig = np.random.poisson(S * (eff + effuncscale * effunc) * exp)
    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(
        np.random.uniform(QBB - delta - ROI / 2, QBB - delta + ROI / 2, n_sig),
        np.zeros(n_bkg),
    )

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


class box_model_0vbb_gen:
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
        effunc: float,
        effuncscale: float,
        exp: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_pdf(
            Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, check_window
        )

    def logpdf(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
    ) -> np.array:
        return nb_logpdf(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp)

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
    ) -> np.array:
        return nb_density(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp)

    # for iminuit ExtendedUnbinnedNLL
    def density_gradient(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
    ) -> np.array:
        return nb_density_gradient(
            Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp
        )

    # for iminuit ExtendedUnbinnedNLL
    def log_density(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
    ) -> np.array:
        mu_S = S * (eff + effuncscale * effunc) * exp
        mu_B = exp * BI * WINDOWSIZE

        # Do a quick check and return -inf if log args are negative
        if (mu_S + mu_B <= 0) or np.isnan(np.array([mu_S, mu_B])).any():
            return mu_S + mu_B, np.full(Es.shape[0], -np.inf)
        else:
            return (
                mu_S + mu_B,
                np.log(mu_S + mu_B)
                + nb_logpdf(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp),
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
        effunc: float,
        effuncscale: float,
        exp: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(
            S, BI, delta, sigma, eff, effunc, effuncscale, exp, seed=seed
        )

    def plot(
        self,
        Es: np.array,
        S: float,
        BI: float,
        delta: float,
        sigma: float,
        eff: float,
        effunc: float,
        effuncscale: float,
        exp: float,
    ) -> None:
        y = nb_pdf(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp)

        import matplotlib.pyplot as plt

        plt.step(Es, y)
        plt.show()

    # function call needs to take the same parameters as the other function calls, in the same order repeated twice
    # this is intended only for empty datasets
    # returns `None` if we couldn't combine the datasets (a dataset was not empty)
    def combine(
        self,
        a_Es: np.array,
        a_S: float,
        a_BI: float,
        a_delta: float,
        a_sigma: float,
        a_eff: float,
        a_effunc: float,
        a_effuncscale: float,
        a_exp: float,
        b_Es: np.array,
        b_S: float,
        b_BI: float,
        b_delta: float,
        b_sigma: float,
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

        # these are fully correlated in this model so the direct sum is appropriate
        # (maybe still appropriate even if not fully correlated?)
        effunc = (a_exp * a_effunc + b_exp * b_effunc) / exp

        effuncscale = 0.0  # this should be overwritten in the fit later

        return [Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp]

    def can_combine(
        self,
        a_Es: np.array,
        a_S: float,
        a_BI: float,
        a_delta: float,
        a_sigma: float,
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

    def intial_guess(self, Es: np.array, exp_tot: float, eff_tot: float) -> tuple:
        """
        Give a better initial guess for the signal and background rate given an array of data
        The signal rate is estimated in a +/-5 keV window around Qbb, the BI is estimated from everything outside that window

        Parameters
        ----------
        Es
            A numpy array of observed energy data
        exp_tot
            The total exposure of the experiment
        eff_tot
            The total efficiency of the experiment
        """
        QBB_ROI_SIZE = [
            5,
            5,
        ]  # how many keV away from QBB in - and + directions we are defining the ROI
        BKG_WINDOW_SIZE = WINDOWSIZE - np.sum(
            QBB_ROI_SIZE
        )  # subtract off the keV we are counting as the signal region
        n_sig = 0
        n_bkg = 0
        for E in Es:
            if QBB - QBB_ROI_SIZE[0] <= E <= QBB + QBB_ROI_SIZE[1]:
                n_sig += 1
            else:
                n_bkg += 1

        # find the expected BI
        BI_guess = n_bkg / (BKG_WINDOW_SIZE * exp_tot)

        # Now find the expected signal rate
        n_sig -= (
            n_bkg * np.sum(QBB_ROI_SIZE) / BKG_WINDOW_SIZE
        )  # subtract off the expected number of BI counts in ROI

        s_guess = n_sig / (exp_tot * eff_tot)

        return s_guess, BI_guess


box_model_0vbb = box_model_0vbb_gen()
