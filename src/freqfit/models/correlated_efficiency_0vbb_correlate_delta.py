"""
PDF that correlates all energy biases with one global parameter alpha_delta
"""
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
    alpha_delta: float,
    delta_unc: float,
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
        Uncertainty on the efficiency
    effuncscale
        Scaling parameter of the efficiency
    exp
        The exposure, in kg*yr
    alpha_delta
        Global correlation between all energy biases
    delta_unc
        Uncertainty on delta
    check_window
        Whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    pdf(E) = 1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta * alpha_delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (1 / (mu_S + mu_B)) * (
            S_amp
            * np.exp(
                -((Es[i] - QBB + delta + alpha_delta * delta_unc) ** 2)
                / (2 * sigma**2)
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
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    alpha_delta: float,
    delta_unc: float,
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
    alpha_delta
        The global scaling of energy biases
    delta_unc
        Uncertainty on delta

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB - delta * alpha_delta, sigma) + mu_B/windowsize]
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * (eff + effuncscale * effunc) * exp / M_A
    # S *= 0.01
    # BI *= 0.0001
    mu_S = S * (eff + effuncscale * effunc) * exp
    mu_B = exp * BI * WINDOWSIZE
    # mu_B = exp * BI * WINDOWSIZE/(np.sqrt(sigma**2 + delta_unc**2) * 2*np.sqrt(2* np.log(2)))

    if sigma == 0:
        return np.inf, np.full_like(Es, np.inf, dtype=np.float64)

    # Precompute the prefactors so that way we save multiplications in the for loop
    B_amp = exp * BI
    # B_amp = exp * BI / (np.sqrt(sigma**2 + delta_unc**2) * 2*np.sqrt(2* np.log(2)))
    S_amp = mu_S / (np.sqrt(2 * np.pi) * sigma)

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (
            S_amp
            * np.exp(
                -((Es[i] - QBB + delta + alpha_delta * delta_unc) ** 2)
                / (2 * sigma**2)
            )
            + B_amp
        )

    return mu_S + mu_B, y


@nb.jit(**nb_kwd)
def nb_log_density(
    Es: np.array,
    S: float,
    BI: float,
    delta: float,
    sigma: float,
    eff: float,
    effunc: float,
    effuncscale: float,
    exp: float,
    alpha_delta: float,
    delta_unc: float,
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
    alpha_delta
        Global energy bias uncertainty
    delta_unc
        Uncertainty on delta

    Notes
    -----
    This function computes the following, faster than without a numba wrapper:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    CDF(E) = mu_S + mu_B
    pdf(E) =[mu_S * norm(E_j, QBB - delta * alpha_delta, sigma) + mu_B/windowsize]
    """
    raise NotImplementedError("This is not yet implemented.")


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
    alpha_delta: float,
    delta_unc: float,
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
    alpha_delta
        The global energy bias scaling
    delta_unc
        Uncertainty on delta

    Notes
    -----
    This function computes the gradient of the density function and returns a tuple where the first element is the gradient of the CDF, and the second element is the gradient of the PDF.
    The first element has shape (K,) where K is the number of parameters, and the second element has shape (K,N) where N is the length of Es.
    mu_S = S * exp * (eff + effuncscale * effunc)
    mu_B = exp * BI * windowsize
    pdf(E) = [mu_S * norm(E_j, QBB + delta * alpha_delta, sigma) + mu_B/windowsize]
    cdf(E) = mu_S + mu_B
    """
    raise NotImplementedError("This is not yet implemented, sorry!")


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
    alpha_delta: float,
    delta_unc: float,
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
    alpha_delta
        The global energy bias scaling
    delta_unc
        Uncertainty on delta

    Notes
    -----
    This function computes the following:
    mu_S = (eff + effuncscale * effunc) * exp * S
    mu_B = exp * BI * windowsize
    logpdf(E) = log(1/(mu_S+mu_B) * [mu_S * norm(E_j, QBB - delta * alpha_delta, sigma) + mu_B/windowsize])
    """

    # Precompute the signal and background counts
    # mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_S = S * (eff + effuncscale * effunc) * exp
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
            S_amp
            * np.exp(
                -((Es[i] - QBB + delta + alpha_delta * delta_unc) ** 2)
                / (2 * sigma**2)
            )
            + B_amp
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
    alpha_delta: float,
    delta_unc: float,
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
    alpha_delta
        The global energy bias scaling
    delta_unc
        The uncertainty on delta

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(
        np.random.normal(QBB - delta - alpha_delta * delta_unc, sigma, size=n_sig),
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

    return Es


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
    alpha_delta: float,
    delta_unc: float,
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
    alpha_delta
        The global energy bias scaling
    delta_unc
        The uncertainty on delta4

    Notes
    -----
    This function pulls from a Gaussian for signal events and from a uniform distribution for background events
    in the provided windows, which may be discontinuous.
    """
    # S *= 0.01
    # BI *= 0.0001

    n_sig = np.random.poisson(S * (eff + effuncscale * effunc) * exp)
    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # Get energy of signal events from a Gaussian distribution
    # preallocate for background draws
    Es = np.append(
        np.random.normal(QBB - delta - delta_unc * alpha_delta, sigma, size=n_sig),
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


class correlated_efficiency_0vbb_correlate_delta_gen(Model):
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
        alpha_delta: float,
        delta_unc: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_pdf(
            Es,
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            exp,
            alpha_delta,
            delta_unc,
            check_window,
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
        alpha_delta: float,
        delta_unc: float,
    ) -> np.array:
        return nb_logpdf(
            Es,
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            exp,
            alpha_delta,
            delta_unc,
        )

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
        alpha_delta: float,
        delta_unc: float,
    ) -> np.array:
        return nb_density(
            Es,
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            exp,
            alpha_delta,
            delta_unc,
        )

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
        alpha_delta: float,
        delta_unc: float,
    ) -> np.array:
        return nb_density_gradient(
            Es,
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            exp,
            alpha_delta,
            delta_unc,
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
        alpha_delta: float,
        delta_unc: float,
    ) -> np.array:
        return nb_log_density(
            Es,
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            exp,
            alpha_delta,
            delta_unc,
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
        alpha_delta: float,
        delta_unc: float,
    ) -> np.array:
        return nb_rvs(n_sig, n_bkg, delta, sigma, alpha_delta, delta_unc)

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
        alpha_delta: float,
        delta_unc: float,
    ) -> np.array:
        return nb_extendedrvs(
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            exp,
            alpha_delta,
            delta_unc,
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
        alpha_delta: float,
        delta_unc: float,
    ) -> None:
        y = nb_pdf(
            Es,
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            exp,
            alpha_delta,
            delta_unc,
        )

        import matplotlib.pyplot as plt

        plt.step(Es, y)
        plt.show()

    def combine(
        self,
        datasets: list,#List[Tuple[np.array,...],...],
    ) -> list:

        Es = np.array([])  # both of these datasets are empty
        S = 0.0  # this should be overwritten in the fit later
        BI = 0.0  # this should be overwritten in the fit later
        effuncscale = 0.0  # this should be overwritten in the fit later
        alpha_delta = 0.0  # this should be overwritten in the fit later

        num = len(datasets)

        deltas = np.zeros(num)
        sigmas = np.zeros(num)
        effs = np.zeros(num)
        effuncs = np.zeros(num)
        effuncscales = np.zeros(num)
        exps = np.zeros(num)
        delta_uncs = np.zeros(num)
        for i, dataset in enumerate(datasets):
            # first few elements not needed (we know data is empty)
            deltas[i]       = dataset[3]
            sigmas[i]       = dataset[4]
            effs[i]         = dataset[5]
            effuncs[i]      = dataset[6]
            effuncscales[i] = dataset[7]
            exps[i]         = dataset[8]    
            delta_uncs[i]   = dataset[9]     

        totexp = np.sum(exps)  # total exposure
        eff = np.sum(exps * effs) / totexp # exposure weighted efficiency
        sigma = np.sum(sigmas * exps * effs) / (totexp * eff) # sensitive exposure weighted resolution
        delta = np.sum(deltas * exps * effs) / (totexp * eff) # sensitive exposure weighted bias correction
        
        # TODO:IS THIS CORRECT?
        delta_unc = np.sum(delta_uncs * exps * effs) / (totexp * eff) # sensitive exposure weighted bias correction

        # these are fully correlated in this model so the direct sum is appropriate
        # (maybe still appropriate even if not fully correlated?)
        effunc = np.sum(exps * effuncs) / totexp

        return [
            Es,
            S,
            BI,
            delta,
            sigma,
            eff,
            effunc,
            effuncscale,
            totexp,
            alpha_delta,
            delta_unc,
        ]

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
        alpha_delta: float,
        delta_unc: float,
    ) -> bool:
        """
        This sets an arbitrary rule if this dataset can be combined with other datasets.
        In this case, if the dataset contains no data, then it can be combined, but more complex rules can be imposed.
        """
        if len(a_Es) == 0:
            return True
        else:
            return False


correlated_efficiency_0vbb_correlate_delta = (
    correlated_efficiency_0vbb_correlate_delta_gen()
)
