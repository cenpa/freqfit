"""
A class that controls a pseudoexperiment and calls the `Superset` class.
"""

from legendfreqfit.superset import Superset
from legendfreqfit.utils import load_config, grab_results
from iminuit.minuit import Minuit
import warnings

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
        
        # collect which parameters are included as nuisance parameters
        self.nuisance = []
        for parname, pardict in self.config["parameters"].items():
            if "includeinfit" in pardict and pardict["includeinfit"]:
                if "nuisance" in pardict and pardict["nuisance"]:
                    if "fixed" in pardict and pardict["fixed"]:
                        msg = (
                            f"{parname} has `fixed` as `True` and `nuisance` as `True`. {parname} will be treated as fixed."
                        )
                        warnings.warn(msg)
                    else:
                        self.nuisance.append(parname)

        # get the fit parameters and set the parameter initial values
        self.guess = self.initialguess()
        self.minuit = Minuit(self.costfunction, **self.guess)

        # raise a RunTime error if function evaluates to NaN
        self.minuit.throw_nan = True

        # to set limits and fixed variables
        self.minuit_reset()
        
        # to store the best fit result
        self.best = None


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

    def bestfit(
        self,
        force=False,
        ) -> dict:

        # don't run this more than once if we don't have to
        if self.best is not None and not force:
            return self.best

        # remove any previous minimizations
        self.minuit_reset()

        self.minuit.migrad()

        self.best = grab_results(self.minuit)

        return self.best
    
    def profile(
        self,
        parameters: dict,
        ) -> dict:
        """
        parameters
            `dict` where keys are names of parameters to fix and values are the value that the parameter should be 
            fixed to
        """

        self.minuit_reset()

        for parname, parvalue in parameters.items():
            self.minuit.fixed[parname] = True
            self.minuit.values[parname] = parvalue

        self.minuit.migrad()

        return grab_results(self.minuit)