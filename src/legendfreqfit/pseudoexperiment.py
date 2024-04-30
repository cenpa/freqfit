"""
A class that controls a pseudoexperiment and calls the `Superset` class.
"""

import warnings

import numpy as np
from iminuit import Minuit

from legendfreqfit.superset import Superset
from legendfreqfit.toy import Toy
from legendfreqfit.utils import grab_results, load_config, emp_cdf, dkw_band

SEED = 42


class Pseudoexperiment(Superset):
    def __init__(
        self,
        config: dict,
        name: str = None,
    ) -> None:
        self.config = config

        constraints = (
            self.config["constraints"] if "constraints" in self.config else None
        )

        super().__init__(
            datasets=self.config["datasets"],
            parameters=self.config["parameters"],
            constraints=constraints,
            name=name,
        )

        # collect which parameters are included as nuisance parameters
        self.nuisance = []
        for parname, pardict in self.config["parameters"].items():
            if "includeinfit" in pardict and pardict["includeinfit"]:
                if "nuisance" in pardict and pardict["nuisance"]:
                    if "fixed" in pardict and pardict["fixed"]:
                        msg = f"{parname} has `fixed` as `True` and `nuisance` as `True`. {parname} will be treated as fixed."
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

    @classmethod
    def file(
        cls,
        file: str,
        name: str = None,
    ):
        config = load_config(file=file)
        return cls(config=config, name=name)

    def initialguess(
        self,
    ) -> dict:
        guess = {
            fitpar: self.parameters[fitpar]["value"]
            if "value" in self.parameters[fitpar]
            else None
            for fitpar in self.fitparameters
        }

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
        force: bool = False,
    ) -> dict:
        """
        force
            By default (`False`), if `self.best` has a result, the minimization will be skipped and that
            result will be returned instead. If `True`, the minimization will be run and the result returned and
            stored as `self.best`

        Performs a global minimization and returns a `dict` with the results. These results are also stored in
        `self.best`.
        """
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

    # this corresponds to t_mu or t_mu^tilde depending on whether there is a limit on the parameters
    def ts(
        self,
        profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
        force: bool = False,
    ) -> float:
        """
        force
            See `Pseudoexperiment.bestfit()` for description. Default is `False`.
        """
        denom = self.bestfit(force=force)["fval"]

        num = self.profile(parameters=profile_parameters)["fval"]

        # because these are already -2*ln(L) from iminuit
        return num - denom

    def maketoy(
        self,
        parameters: dict,
        seed: int = SEED,
    ) -> Toy:
        toy = Toy(pseuodexperiment=self, parameters=parameters, seed=seed)

        return toy

    def toy_ts(
        self,
        parameters: dict,  # parameters and values needed to generate the toys
        profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
        num: int = 1,
        seed: int = SEED,
    ):
        """
        Makes a number of toys and returns their test statistics.
        """

        ts = np.zeros(num)

        for i in range(num):
            thistoy = self.maketoy(parameters=parameters, seed=seed + i)
            ts[i] = thistoy.ts(profile_parameters=profile_parameters)

        return ts

    def toy_ts_critical(
            self,
            ts_dist: np.array, # output of toy_ts 
            threshold: float = 0.9, # critical threshold for test statistic
            confidence: float = 0.68, # width of confidence interval 
    ):
        """
        Returns the critical value of the test statistic and confidence interval 
        """
        cdf, bins = emp_cdf(ts_dist)

        lo_band, hi_band = dkw_band(cdf, nevts=len(ts_dist), confidence=confidence)

        idx_crit = np.where(cdf >= threshold)[0][0]
        critical = bins[idx_crit]

        lo = lo_band[idx_crit] 
        hi = hi_band[idx_crit] 

        lo_idx = np.where(cdf >= lo)[0][0]
        hi_idx = np.where(cdf >= hi)[0][0]
        
        return critical, bins[lo_idx], bins[hi_idx] 