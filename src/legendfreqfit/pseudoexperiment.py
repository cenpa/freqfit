"""
A class that controls a pseudoexperiment and calls the `Superset` class.
"""

from legendfreqfit.superset import Superset
from legendfreqfit.utils import load_config

class Pseudoexperiment(Superset):
    def __init__(
        self,
        file: str,
        name: str = None,
        ) -> None:

        config = load_config(file=file)

        super().__init__(datasets=config["datasets"], parameters=config["parameters"], 
                         constraints=config["constraints"], name=name)

        # get the fit parameters and set the parameter initial values
        
        # self.minuit = Minuit(self.costfunction)

