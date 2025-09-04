"""
Abstract base class for initial guesses. Needs to take an Experiment and return a guess 
for each fit parameter as a dict.
"""

from abc import ABC, abstractmethod
from .experiment import Experiment

class Guess(ABC):
    @abstractmethod
    def guess(
        self, 
        experiment: Experiment,
    ) -> dict:
        pass

