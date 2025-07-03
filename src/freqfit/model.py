"""
Abstract base class for freqfit models
"""

from abc import ABC, abstractmethod
import inspect
import numpy as np

class Model(ABC):
    @abstractmethod
    def pdf(
        self, 
        data: np.array,
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def logpdf(
        self, 
        data: np.array,
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def density(
        self, 
        data: np.array,
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def logdensity(
        self, 
        data: np.array,
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def graddensity(
        self, 
        data: np.array,
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def rvs(
        self, 
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def extendedrvs(
        self, 
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def can_combine(
        self, 
        data: np.array,
        *parameters,
    ) -> bool:
        """
        Should take a set of data and parameters and decide whether it could be combined with another set of data 
        and parameters. Decision must be made on the basis of the single set alone.
        """
        pass

    @abstractmethod
    def combine(
        self, 
        x: list,#List[Tuple[np.array,...]],
    ) -> np.array:
        """
        Should take a list of N tuples of (data, parameters) and returned a combined single tuple of (data, parameters).
        Assume that all passed datasets can be combined - this is checked by CombinedDataset using self.can_combine().
        """
        pass

    # takes a model function and returns a dict of its parameters with their default value
    @staticmethod
    def inspectparameters(
        func,
    ) -> dict:
        """
        Returns a `dict` of parameters that methods of this model take as keys. Values are default values of the
        parameters. Assumes the first argument of the model is `data` and not a model parameter, so this key is not
        returned.
        """
        # pulled some of this from `iminuit.util`
        try:
            signature = inspect.signature(func)
        except ValueError:  # raised when used on built-in function
            return {}

        r = {}
        for i, item in enumerate(signature.parameters.items()):
            if i == 0:
                continue

            name, par = item

            if (default := par.default) is par.empty:
                r[name] = "nodefaultvalue"
            else:
                r[name] = default

        return r