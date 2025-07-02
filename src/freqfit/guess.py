"""
Abstract base class for initial guesses. Needs to take an Experiment and return a guess 
for each fit parameter.
"""

from abc import ABC, abstractmethod

class Guess(ABC):
    @abstractmethod
    def guess(
        self, 
        experiment:
    ) -> dict:
        pass

