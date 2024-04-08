"""
A class that controls a pseudoexperiment and calls the `Superset` class.
"""

from legendfreqfit.superset import Superset

class Pseudoexperiment(Superset):
    def __init__(
        self,
        datasets: dict,
        name: str = None,
        ) -> None:

        super().__init__(datasets=datasets, name=name)

        # get the fit parameters and set the parameter initial values
        
        # self.minuit = Minuit(self.costfunction)

