"""
Abstract base class for initial guesses. Needs to take an Experiment and return a guess
for each fit parameter as a dict.
"""

from abc import ABC, abstractmethod

from .experiment import Experiment


class Guess(ABC):
    """
    Abstract base class for initial-guess providers.

    A concrete implementation must implement :meth:`guess` and return a dictionary
    mapping each fit parameter name to an initial value suitable for initializing
    ``iminuit.Minuit``.
    """
    @abstractmethod
    def guess(
        self,
        experiment: Experiment,
    ) -> dict:
        """
        Compute an initial parameter guess for an experiment.

        Parameters
        ----------
        experiment
            The :class:`~freqfit.experiment.Experiment` instance being fit.

        Returns
        -------
        dict
            Mapping ``{parameter_name: initial_value}``.
        """
        pass
