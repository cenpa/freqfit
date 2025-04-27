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
    def combine(
        self, 
        *parameters,
    ) -> np.array:
        pass

    @abstractmethod
    def cancombine(
        self, 
        data: np.array,
        *parameters,
    ) -> bool:
        pass

    @abstractmethod
    def initialguess(
        self, 
        data: np.array,
        *parameters,
    ) -> np.array:
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