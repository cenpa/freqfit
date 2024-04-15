"""
A class that controls a pseudoexperiment and calls the `Superset` class.
"""

from legendfreqfit.superset import Superset
from legendfreqfit.utils import load_config
from iminuit.minuit import Minuit
import copy

class Pseudoexperiment(Superset):
    def __init__(
        self,
        file: str,
        name: str = None,
        ) -> None:

        config = load_config(file=file)
        self.config = config

        constraints = self.config["constraints"] if "constraints" in self.config else None

        super().__init__(datasets=self.config["datasets"], parameters=self.config["parameters"], 
                         constraints=constraints, name=name)

        # get the fit parameters and set the parameter initial values
        self.guess = self.initialguess()
        self.minuit = Minuit(self.costfunction, **self.guess)

        # to set limits and fixed variables
        self.minuit_reset()


        self.minuit_global = None

    def initialguess(
        self,
        ) -> dict:

        guess = {fitpar: self.parameters[fitpar]["value"] if "value" in self.parameters[fitpar] 
                        else None for fitpar in self.fitparameters}
        
        # for fitpar, value in guess.items():
        #     if value is None:
        #         guess[fitpar] = 1e-9
        
        # could put other stuff here to get a better initial guess though probably that should be done
        # somewhere specific to the analysis. Or maybe this function could call something for the analysis.
        # eh, probably better to make the initial guess elsewhere and stick it in the config file.
        # for that reason, I'm commenting out the few lines above.

        return guess
        
    def minuit_reset(
        self,
        ) -> None:

        # resets the minimization and stuff
        # does not change limits but does remove "fixed" attribute of variables
        self.minuit.reset()

        # overwrite the limits
        # note that this information can also be contained in a Dataset when instantiated 
        # and is overwritten here

        # also set which parameters are fixed
        for parname, pardict in self.config["parameters"].items():
            if parname in self.minuit.parameters:
                if "limits" in pardict:
                    self.minuit.limits[parname] = pardict["limits"]
                if "fixed" in pardict:
                    self.minuit.fixed[parname] = pardict["fixed"]    

        return
        

