"""
A class that holds a toy dataset and its associated model and cost function,
"""

import warnings
import numpy as np
from iminuit import cost
from legendfreqfit import Dataset

SEED = 42

class Toy:
    def __init__(
            self,
            dataset: Dataset,
            par: np.array,
            seed: int = SEED,
        ) -> None:
        """
        dataset
            `Dataset` to make a toy model from
        par
            1D `np.array` that contains the values of parameters of the model to pull from, based on
            format of `Superset.maketoy`. Don't love this and maybe it changes later to a dict.
        """

        self.model = dataset.model
        self._parlist = dataset._parlist
        self._parlist_indices = dataset._parlist_indices

        self.toydata = self.maketoy(*par, seed=seed)

        if (dataset.costfunction is cost.ExtendedUnbinnedNLL):
            self.costfunction = cost.ExtendedUnbinnedNLL(self.toydata, self.density)
        elif (dataset.costfunction is cost.UnbinnedNLL):
            self.costfunction = cost.UnbinnedNLL(self.toydata, self.density)
        else:
            msg = (
                f"`Toy`: only `ExtendedUnbinnedNLL` or `UnbinnedNLL` are currently supported as cost functions."
            )
            raise RuntimeError(msg)
        
        return
        
    def maketoy(
        self, 
        *par, 
        seed: int = SEED, # must be passed as keyword
        ) -> np.array:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]     

        self.toy = self.model.extendedrvs(*self._parlist, seed=seed)

        return self.toy

    def density(
        self, 
        data, 
        *par,
        ) -> np.array:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]
                    
        return self.model.density(data, *self._parlist)
        
    def toyll(
        self, 
        *par,
        ) -> float:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]     
        
        return self.model.loglikelihood(self.toy, *self._parlist)