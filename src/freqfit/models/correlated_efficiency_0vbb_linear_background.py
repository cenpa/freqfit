"""Like all of the functions in `correlated_efficiency_0vbb`, but with a lienar background instead of a uniform one."""
import numba as nb
import numpy as np

import freqfit.models.constants as constants
from freqfit.model import Model

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
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = m/2(E_hi^2-E_lo^2) + BI*exp*(E_hi-E_lo)
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta, sigma) + a*E + BI*exp]
    """
    x0 = WINDOW[0][0]
    x1 = WINDOW[-1][1]

    mid = (x1 + x0) / 2.0

    # take normalized slope and convert to actual slope
    slope = a * (2.0 / (FULLWINDOWSIZE * FULLWINDOWSIZE))

    b = 1.0 / FULLWINDOWSIZE

    includedarea = 0.0
    for i in nb.prange(WINDOW.shape[0]):
        includedarea += (
            2.0
            * (WINDOW[i][1] - WINDOW[i][0])
            * (slope * (-2.0 * mid + WINDOW[i][0] + WINDOW[i][1]) + 2 * b)
        )

    totarea = (
        2.0
        * (WINDOW[-1][1] - WINDOW[0][0])
        * (slope * (-2.0 * mid + WINDOW[0][0] + WINDOW[-1][1]) + 2 * b)
    )

    amp = totarea / includedarea

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = WINDOWSIZE * BI * exp

    # Precompute the prefactors so that way we save multiplications in the for loop
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2))
            + (slope * (Es[i] - mid) + b) * amp
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

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = m/2(E_hi^2-E_lo^2) + BI*exp*(E_hi-E_lo)
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB - delta, sigma) + m*E + b]
    """

    x0 = WINDOW[0][0]
    x1 = WINDOW[-1][1]

    mid = (x1 + x0) / 2.0

    # take normalized slope and convert to actual slope
    slope = a * (2.0 / (FULLWINDOWSIZE * FULLWINDOWSIZE))

    b = 1.0 / FULLWINDOWSIZE

    includedarea = 0.0
    for i in nb.prange(WINDOW.shape[0]):
        includedarea += (
            2.0
            * (WINDOW[i][1] - WINDOW[i][0])
            * (slope * (-2.0 * mid + WINDOW[i][0] + WINDOW[i][1]) + 2 * b)
        )

    totarea = (
        2.0
        * (WINDOW[-1][1] - WINDOW[0][0])
        * (slope * (-2.0 * mid + WINDOW[0][0] + WINDOW[-1][1]) + 2 * b)
    )

    if includedarea == 0:
        return np.inf, np.full_like(Es, np.inf, dtype=np.float64)

    amp = totarea / includedarea * BI * exp * WINDOWSIZE

    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = WINDOWSIZE * BI * exp

    if sigma == 0:
        return np.inf, np.full_like(Es, np.inf, dtype=np.float64)

    # Precompute the prefactors so that way we save multiplications in the for loop
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (
            S_amp * np.exp(-((Es[i] - QBB + delta) ** 2) / (2 * sigma**2))
            + (slope * (Es[i] - mid) + b) * amp
        )

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

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """
    # S *= 0.01
    # BI *= 0.0001

    n_sig = np.random.poisson(S * (eff + effuncscale * effunc) * exp)
    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # preallocate for background draws
    Es = np.zeros(n_bkg, dtype=np.float64)

    x0 = WINDOW[0][0]
    x1 = WINDOW[-1][1]

    mid = (x1 + x0) / 2.0

    # take normalized slope and convert to actual slope
    slope = a * (2.0 / (FULLWINDOWSIZE * FULLWINDOWSIZE))

    # there's a precision issue with how I am drawing rvs when a is very small
    if abs(a) < 1e-12:
        slope = 0.0

    b = 1.0 / FULLWINDOWSIZE

    # find area of each section and percent of cdf
    cumareas = np.zeros(len(WINDOW))
    percentages = np.zeros(len(WINDOW))
    areas = np.zeros(len(WINDOW))

    totarea = 0.0
    for i in nb.prange(WINDOW.shape[0]):
        area = (
            2.0
            * (WINDOW[i][1] - WINDOW[i][0])
            * (slope * (-2.0 * mid + WINDOW[i][0] + WINDOW[i][1]) + 2 * b)
        )
        totarea += area
        cumareas[i] = totarea
        areas[i] = area

    for i in nb.prange(WINDOW.shape[0]):
        percentages[i] = cumareas[i] / totarea

    # figure out which window each count belongs to
    whichwindow = np.random.uniform(0.0, 1.0, n_bkg)
    numwindow = np.zeros(WINDOW.shape[0], dtype="i")
    for i in nb.prange(n_bkg):
        for j in nb.prange(WINDOW.shape[0]):
            if whichwindow[i] < percentages[j]:
                numwindow[j] += 1
                break

    m = slope
    # now draw the events for each window
    Es = np.zeros(n_bkg)
    k = 0
    for i in nb.prange(WINDOW.shape[0]):
        x0 = WINDOW[i][0]
        rvs = 0.5 * areas[i] * np.random.uniform(0.0, 1.0, numwindow[i])
        for j in nb.prange(numwindow[i]):
            if m == 0.0:
                Es[k] = WINDOW[i][0] + rvs[j] / (2.0 * b)
            else:
                Es[k] = (
                    (
                        b**2
                        + 2 * b * m * (x0 - mid)
                        + m * (mid**2 * m - 2 * mid * m * x0 + m * x0**2 + rvs[j])
                    )
                    ** 0.5
                    - b
                    + mid * m
                ) / m

            k += 1

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(Es, np.random.normal(QBB + delta, sigma, size=n_sig))

    return Es, (n_sig, n_bkg)


class correlated_efficiency_0vbb_linear_background_gen(Model):
    def __init__(self):
        self.parameters = self.inspectparameters(self.density)
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
    def graddensity(
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
    def logdensity(
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
    ) -> np.array:
        return nb_rvs(n_sig, n_bkg, delta, sigma, a)

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
    ) -> np.array:
        return nb_extendedrvs(
            S, BI, delta, sigma, eff, effunc, effuncscale, exp, a
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

    def combine(
        self,
        datasets: list,
    ) -> list :

        Es = np.array([])  # both of these datasets are empty
        S = 0.0  # this should be overwritten in the fit later
        BI = 0.0  # this should be overwritten in the fit later
        effuncscale = 0.0  # this should be overwritten in the fit later
        a = 0.0 # this should be overwritten in the fit later

        num = len(datasets)

        deltas = np.zeros(num)
        sigmas = np.zeros(num)
        effs = np.zeros(num)
        effuncs = np.zeros(num)
        effuncscales = np.zeros(num)
        exps = np.zeros(num)
        for i, dataset in enumerate(datasets):
            # first few elements not needed (we know data is empty)
            deltas[i]       = dataset[3]
            sigmas[i]       = dataset[4]
            effs[i]         = dataset[5]
            effuncs[i]      = dataset[6]
            effuncscales[i] = dataset[7]
            exps[i]         = dataset[8]

        totexp = np.sum(exps)  # total exposure
        eff = np.sum(exps * effs) / totexp # exposure weighted efficiency
        sigma = np.sum(sigmas * exps * effs) / (totexp * eff) # sensitive exposure weighted resolution
        delta = np.sum(deltas * exps * effs) / (totexp * eff) # sensitive exposure weighted bias correction

        # these are fully correlated in this model so the direct sum is appropriate
        # (maybe still appropriate even if not fully correlated?)
        effunc = np.sum(exps * effuncs) / totexp

        return [Es, S, BI, delta, sigma, eff, effunc, effuncscale, totexp, a]

    def can_combine(
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
    ) -> bool:
        """
        This sets an arbitrary rule if this dataset can be combined with other datasets.
        In this case, if the dataset contains no data, then it can be combined, but more complex rules can be imposed.
        """
        if len(a_Es) == 0:
            return True
        else:
            return False


correlated_efficiency_0vbb_linear_background = (
    correlated_efficiency_0vbb_linear_background_gen()
)
