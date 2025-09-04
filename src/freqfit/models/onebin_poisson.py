import numba as nb
import numpy as np

from freqfit.model import Model

nb_kwd = {
    "nopython": True,
    "parallel": False,
    "nogil": True,
    "cache": True,
    "fastmath": True,
    "inline": "always",
}

@nb.jit(**nb_kwd)
def nb_pdf(
    Es: np.array,
    S: float,
    B: float,
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
    B: float,
) -> np.array:

    # Initialize and execute the for loop
    y = np.empty_like(Es, dtype=np.float64)
    for i in nb.prange(Es.shape[0]):
        if Es[i] == 1:
            y[i] = S + B
        else:
            y[0] = 0.0
                
    return S + B, y

@nb.jit(nopython=True, fastmath=True, cache=True, error_model="numpy")
def nb_extendedrvs(
    S: float,
    B: float,
) -> np.array:


    n_sig = np.random.poisson(S + B)
    
    Es = np.ones(n_sig)

    return Es, (n_sig, 0)

class onebin_poisson_gen(Model):
    def __init__(self):
        self.parameters = self.inspectparameters(self.density)
        pass

    def pdf(
        self,
        Es: np.array,
        S: float,
        B: float,
    ) -> np.array:
        return nb_pdf(Es, S, B)

    def logpdf(
        self,
        Es: np.array,
        S: float,
        B: float,
    ) -> np.array:
        return np.log(nb_pdf(Es, S, B))

    # for iminuit ExtendedUnbinnedNLL
    def density(
        self,
        Es: np.array,
        S: float,
        B: float,
    ) -> np.array:
        return nb_density(Es, S, B)

    def logdensity(
        self,
        Es: np.array,
        S: float,
        B: float,
    ) -> np.array:
        return np.log(nb_density(Es, S, B))

    def graddensity(
        self,
        Es: np.array,
        S: float,
        B: float,
    ) -> np.array:
        raise NotImplementedError
        return

    def rvs(
        self,
        S: float,
        B: float,
    ) -> np.array:
        raise NotImplementedError
        return 

    def extendedrvs(
        self,
        S: float,
        B: float,
    ) -> np.array:
        return nb_extendedrvs(S, B)

    # not supported
    def combine(
        self,
        datasets: list,#List[Tuple[np.array,...],...],
    ) -> list:
        raise NotImplementedError
        return

    # not supported
    def can_combine(
        self,
        Es: np.array,
        S: float,
        B: float,
    ) -> bool:
        return False

onebin_poisson = onebin_poisson_gen()
