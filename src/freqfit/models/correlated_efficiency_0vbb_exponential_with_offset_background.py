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
WINDOW = np.array(constants.WINDOW)

WINDOWSIZE = 0.0
for i in range(len(WINDOW)):
    WINDOWSIZE += WINDOW[i][1] - WINDOW[i][0]

FULLWINDOWSIZE = WINDOW[-1][1] - WINDOW[0][0]


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
    a: float,
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
    a
        Controls the decay constant of the exponential background
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = BI * W * exp
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta, sigma) + N*BI*W*exp*(np.exp(-a*(E_j-x_lo)/a + 1))]
    """
    x_lo = WINDOW[0][0]
    x1 = WINDOW[0][1]
    x2 = WINDOW[1][0]
    x3 = WINDOW[1][1]
    x4 = WINDOW[-1][0]
    x_hi = WINDOW[-1][1]

    # Compute the normalization for the CDF
    invnorm = (x_hi - x4 + x3 - x2 + x1 - x_lo) - (
        np.exp(-a * (x_hi - x_lo))
        - np.exp(-a * (x4 - x_lo))
        + np.exp(-a * (x3 - x_lo))
        - np.exp(-a * (x2 - x_lo))
        + np.exp(-a * (x1 - x_lo))
        - 1
    )

    if invnorm == 0:
        norm = np.inf
    else:
        norm = 1 / invnorm

    if a == 0:
        a_inv = np.inf
    else:
        a_inv = 1 / a

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = WINDOWSIZE * BI * exp

    # Precompute the prefactors so that way we save multiplications in the for loop
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)
    B_amp = norm * mu_B

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2))
            + B_amp * (np.exp(-a * (Es[i] - x_lo)) * a_inv + 1)
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
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    a: float,
    check_window: bool = True,
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
    a
        Controls the decay constant of the exponential background

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = BI * W * exp
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB + delta, sigma) + N * BI * W * exp * (np.exp(-a*(E_j-x_lo))/a + 1)]
    """

    x_lo = WINDOW[0][0]
    x1 = WINDOW[0][1]
    x2 = WINDOW[1][0]
    x3 = WINDOW[1][1]
    x4 = WINDOW[-1][0]
    x_hi = WINDOW[-1][1]

    # Compute the normalization for the CDF
    invnorm = (x_hi - x4 + x3 - x2 + x1 - x_lo) - (
        np.exp(-a * (x_hi - x_lo))
        - np.exp(-a * (x4 - x_lo))
        + np.exp(-a * (x3 - x_lo))
        - np.exp(-a * (x2 - x_lo))
        + np.exp(-a * (x1 - x_lo))
        - 1
    )

    if invnorm == 0:
        norm = np.inf
    else:
        norm = 1 / invnorm

    if a == 0:
        a_inv = np.inf
    else:
        a_inv = 1 / a

    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = WINDOWSIZE * BI * exp

    if sigma == 0:
        return np.inf, np.full_like(Es, np.inf, dtype=np.float64)

    # Precompute the prefactors so that way we save multiplications in the for loop
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)
    B_amp = norm * mu_B

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = S_amp * np.exp(
            -((Es[i] - QBB + delta) ** 2) / (2 * sigma**2)
        ) + B_amp * (np.exp(-a * (Es[i] - x_lo)) * a_inv + 1)

    if check_window:
        for i in nb.prange(Es.shape[0]):
            inwindow = False
            for j in range(len(WINDOW)):
                if WINDOW[j][0] <= Es[i] <= WINDOW[j][1]:
                    inwindow = True
            if not inwindow:
                y[i] = 0.0

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
    m: float,
) -> np.array:
    raise NotImplementedError
    return


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
    m: float,
) -> np.array:
    raise NotImplementedError
    return


@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_rvs(
    n_sig: int,
    n_bkg: int,
    delta: float,
    sigma: float,
    seed: int = SEED,
) -> np.array:
    raise NotImplementedError
    return


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
    a: float,
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
    a
        Controls the decay of the exponential background
    seed
        specify a seed, otherwise uses default seed

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """
    raise NotImplementedError
    # S *= 0.01
    # BI *= 0.0001

    # np.random.seed(seed)

    # n_sig = np.random.poisson(S * (eff + effuncscale * effunc) * exp)
    # n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # x_lo = WINDOW[0][0]
    # x1 = WINDOW[0][1]
    # x2 = WINDOW[1][0]
    # x3 = WINDOW[1][1]
    # x4 = WINDOW[-1][0]
    # x_hi = WINDOW[-1][1]

    # # Compute the normalization for the CDF
    # invnorm = (x_hi - x4 + x3 - x2 + x1 -x_lo) - (np.exp(-a*(x_hi-x_lo)) - np.exp(-a*(x4-x_lo)) + np.exp(-a*(x3-x_lo)) - np.exp(-a*(x2-x_lo))+
    # np.exp(-a*(x1-x_lo)) - 1)

    # if invnorm == 0:
    #     norm = np.inf
    # else:
    #     norm = 1/invnorm

    # Es = np.random.exponential(a, n_bkg) + x_lo
    # # Make sure we drew in the correct window
    # for i in nb.prange(len(Es)):
    #     inwindow = False
    #     for j in range(len(WINDOW)):
    #         if WINDOW[j][0] <= Es[i] <= WINDOW[j][1]:
    #             inwindow = True
    #     if inwindow:
    #         # loop until we do get a count inside a window
    #         new_inwindow = True
    #         while new_inwindow:
    #             newdraw = np.random.exponential(a, 1)[0] + x_lo
    #             new_inwindowcheck = False
    #             for j in range(len(WINDOW)-1):
    #                 if WINDOW[j][1] <= newdraw <= WINDOW[j+1][0]:
    #                     new_inwindowcheck = True
    #             if not new_inwindowcheck:
    #                 new_inwindow = False
    #     Es[i] = newdraw

    #     Es = np.append(Es, np.random.normal(QBB - delta, sigma, size=n_sig))

    # return Es, (n_bkg, n_sig)


class correlated_efficiency_0vbb_exponential_background_gen:
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
        a: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_pdf(
            Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, a, check_window
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
        a: float,
    ) -> np.array:
        return nb_logpdf(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, a)

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
        a: float,
    ) -> np.array:
        return nb_density(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, a)

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
        a: float,
    ) -> np.array:
        return nb_density_gradient(
            Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, a
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
        a: float,
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
                + nb_logpdf(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, a),
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
        a: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_rvs(n_sig, n_bkg, delta, sigma, a, seed=seed)

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
        a: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(
            S, BI, delta, sigma, eff, effunc, effuncscale, exp, a, seed=seed
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
        a: float,
    ) -> None:
        y = nb_pdf(Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, a)

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
        a_m: float,
        b_Es: np.array,
        b_S: float,
        b_BI: float,
        b_delta: float,
        b_sigma: float,
        b_eff: float,
        b_effunc: float,
        b_effuncscale: float,
        b_exp: float,
        b_m: float,
    ) -> list | None:
        # datasets must be empty to be combined
        if len(a_Es) != 0 or len(b_Es) != 0:
            return None

        Es = np.array([])  # both of these datasets are empty
        S = 0.0  # this should be overwritten in the fit later
        BI = 0.0  # this should be overwritten in the fit later
        m = 0.0  # this should be overwritten in the fit later

        exp = a_exp + b_exp  # total exposure

        # exposure weighted fixed parameters (important to calculate correctly)
        sigma = (a_exp * a_sigma + b_exp * b_sigma) / exp
        eff = (a_exp * a_eff + b_exp * b_eff) / exp
        delta = (a_exp * a_delta + b_exp * b_delta) / exp

        # these are fully correlated in this model so the direct sum is appropriate
        # (maybe still appropriate even if not fully correlated?)
        effunc = (a_exp * a_effunc + b_exp * b_effunc) / exp

        effuncscale = 0.0  # this should be overwritten in the fit later

        return [Es, S, BI, delta, sigma, eff, effunc, effuncscale, exp, m]

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
        a_m: float,
    ) -> bool:
        """
        This sets an arbitrary rule if this dataset can be combined with other datasets.
        In this case, if the dataset contains no data, then it can be combined, but more complex rules can be imposed.
        """
        if len(a_Es) == 0:
            return True
        else:
            return False


correlated_efficiency_0vbb_exponential_background = (
    correlated_efficiency_0vbb_exponential_background_gen()
)
