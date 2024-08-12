import numba as nb
import numpy as np

from legendfreqfit.utils import inspectparameters

nb_kwd = {
    "nopython": True,
    "parallel": False,
    "nogil": True,
    "cache": True,
    "fastmath": True,
    "inline": "always",
}

SEED = 42  # set the default random seed

@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    S: float,
) -> np.array:

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        if Es[i] == 1:
            y[i] = 1.0
        else:
            y[0] = 0.0
                
    return y

@nb.jit(**nb_kwd)
def nb_density(
    Es: np.array,
    S: float,
) -> np.array:

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        if Es[i] == 1:
            y[i] = S
        else:
            y[0] = 0.0
                
    return S, y

@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_extendedrvs(
    S: float,
    seed: int = SEED,
) -> np.array:

    np.random.seed(seed)

    n_sig = np.random.poisson(S)
    
    Es = np.ones(n_sig)

    return Es, (n_sig, 0)

class onebin_poisson_gen:
    def __init__(self):
        self.parameters = inspectparameters(self.density)
        pass

    def pdf(
        self,
        Es: np.array,
        S: float,
    ) -> np.array:
        return nb_pdf(Es, S)

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        S: float,
    ) -> np.array:
        return nb_density(Es, S)

    def extendedrvs(
        self,
        S: float,
        seed: int = SEED,
    ) -> np.array:
        return nb_extendedrvs(S, seed=seed)

onebin_poisson = onebin_poisson_gen()
