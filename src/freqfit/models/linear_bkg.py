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
}

# window
#     must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
#     where edges of window are monotonically increasing (this is not checked), in keV.
#     Default is typical analysis window.

# default analysis window and width
WINDOW = np.array(constants.WINDOW)

WINDOWSIZE = 0.0
for i in range(len(WINDOW)):
    WINDOWSIZE += WINDOW[i][1] - WINDOW[i][0]

FULLWINDOWSIZE = WINDOW[-1][1] - WINDOW[0][0]

SEED = 42  # set the default random seed


@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    a: float,
    BI: float,
    exp: float,
    check_window: bool = False,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    a   
        "normalized" slope between -1 and 1 inclusive
    BI
        rate of background in cts/exposure/energy (not used in pdf)
    exp
        exposure (not used in pdf)
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)
    """

    x0 = WINDOW[0][0]
    x1 = WINDOW[-1][1]

    mid = (x1 + x0) / 2.0

    # take normalized slope and convert to actual slope
    slope = a * (2.0 / (FULLWINDOWSIZE * FULLWINDOWSIZE))

    b = 1.0 / FULLWINDOWSIZE

    includedarea = 0.0
    for i in nb.prange(WINDOW.shape[0]):
        includedarea += (2.0 * (WINDOW[i][1] - WINDOW[i][0]) 
            * (slope * (-2.0 * mid + WINDOW[i][0] + WINDOW[i][1]) + 2 * b))

    totarea = (2.0 * (WINDOW[-1][1] - WINDOW[0][0]) 
        * (slope * (-2.0 * mid + WINDOW[0][0] + WINDOW[-1][1]) + 2 * b))

    amp = totarea / includedarea

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (slope * (Es[i] - mid) + b) * amp

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
    a: float,
    BI: float,
    exp: float,
    check_window: bool = False,
) -> np.array:
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    a   
        "normalized" slope between -1 and 1 inclusive
    BI
        rate of background in cts/exposure/energy
    exp
        exposure
    check_window
        whether to check if the passed Es fall inside of the window. Default is False and assumes that the passed Es
        all fall inside the window (for speed)
    """

    x0 = WINDOW[0][0]
    x1 = WINDOW[-1][1]

    mid = (x1 + x0) / 2.0

    # take normalized slope and convert to actual slope
    slope = a * (2.0 / (FULLWINDOWSIZE * FULLWINDOWSIZE))

    b = 1.0 / FULLWINDOWSIZE

    includedarea = 0.0
    for i in nb.prange(WINDOW.shape[0]):
        includedarea += (2.0 * (WINDOW[i][1] - WINDOW[i][0]) 
            * (slope * (-2.0 * mid + WINDOW[i][0] + WINDOW[i][1]) + 2 * b))

    totarea = (2.0 * (WINDOW[-1][1] - WINDOW[0][0]) 
        * (slope * (-2.0 * mid + WINDOW[0][0] + WINDOW[-1][1]) + 2 * b))

    amp = totarea / includedarea * BI * exp * WINDOWSIZE

    # Initialize and execute the for loop
    y = np.zeros_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        y[i] = (slope * (Es[i] - mid) + b) * amp

    if check_window:
        for i in nb.prange(Es.shape[0]):
            inwindow = False
            for j in range(len(WINDOW)):
                if WINDOW[j][0] <= Es[i] <= WINDOW[j][1]:
                    inwindow = True
            if not inwindow:
                y[i] = 0.0
    
    return WINDOWSIZE * BI * exp, y

@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_extendedrvs(
    a: float,
    BI: float,
    exp: float,
    seed: int = SEED,
) -> (np.array, int):
    """
    Parameters
    ----------
    Es
        Energies at which this function is evaluated, in keV
    a   
        "normalized" slope between -1 and 1 inclusive
    BI
        rate of background in cts/exposure/energy
    exp
        exposure
    """

    np.random.seed(seed)

    n_bkg = np.random.poisson(BI * exp * WINDOWSIZE)

    # preallocate for background draws
    Es = np.zeros(n_bkg, dtype=np.float64)

    x0 = WINDOW[0][0]
    x1 = WINDOW[-1][1]

    mid = (x1 + x0) / 2.0

    # take normalized slope and convert to actual slope
    slope = a * (2.0 / (FULLWINDOWSIZE * FULLWINDOWSIZE))
    
    # there's a precision issue with how I am drawing rvs when a is very small
    if abs(a) < 1E-12:
        slope = 0.0

    b = 1.0 / FULLWINDOWSIZE

    # find area of each section and percent of cdf 
    cumareas = np.zeros(len(WINDOW))
    percentages = np.zeros(len(WINDOW))
    areas = np.zeros(len(WINDOW))

    totarea = 0.0
    for i in nb.prange(WINDOW.shape[0]):
        area = 2.0 * (WINDOW[i][1] - WINDOW[i][0]) * (slope * (-2.0 * mid + WINDOW[i][0] + WINDOW[i][1]) + 2 * b)
        totarea += area
        cumareas[i] = totarea
        areas[i] = area

    for i in nb.prange(WINDOW.shape[0]):
        percentages[i] = cumareas[i] / totarea

    # figure out which window each count belongs to
    whichwindow = np.random.uniform(0.0, 1.0, n_bkg)
    numwindow = np.zeros(WINDOW.shape[0], dtype='i')
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
        rvs = 0.5*areas[i]*np.random.uniform(0.0, 1.0, numwindow[i])
        for j in nb.prange(numwindow[i]):
            if m == 0.0:
                Es[k] = WINDOW[i][0] + rvs[j] / (2.0*b)
            else:
                Es[k] = ((b**2 + 2*b*m*(x0-mid) + m*(mid**2*m - 2*mid*m*x0 + m*x0**2 + rvs[j]))**0.5 - b + mid*m) / m
            
            k += 1

    return (Es, (n_bkg, 0))

class linear_bkg_gen:
    def __init__(self):
        self.parameters = inspectparameters(self.density)
        del self.parameters["check_window"] # used for plotting and stuff, screws up rvs since not present there
        pass

    def pdf(
        self,
        Es: np.array,
        a: float,
        BI: float,
        exp: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_pdf(Es, a, BI, exp, check_window=check_window)

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        a: float,
        BI: float,
        exp: float,
        check_window: bool = False,
    ) -> np.array:
        return nb_density(Es, a, BI, exp, check_window=check_window)

    def extendedrvs(
        self,
        a: float,
        BI: float,
        exp: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(a, BI, exp, seed=seed)

linear_bkg = linear_bkg_gen()
