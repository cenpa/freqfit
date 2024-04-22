"""
A class that holds a collection of fake datasets and associated hardware
"""

from iminuit import Minuit

from legendfreqfit.utils import grab_results

SEED = 42


class Toy:
    def __init__(
        self,
        pseuodexperiment,
        parameters: dict,
        seed: int = SEED,
    ) -> None:
        """
        pseuodexperiment
            `Pseuodexperiment` to base this `Toy` on
        parameters
            `dict` of parameters and their values to model with
        """

        self.pseuodexperiment = pseuodexperiment

        self.costfunction = None
        for i, (datasetname, dataset) in enumerate(pseuodexperiment.datasets.items()):
            # worried that this is not totally deterministic (depends on number of Datasets),
            # but more worried that random draws between datasets would be correlated otherwise.
            thisseed = seed + i

            # allocate length in case `parameters` is passed out of order
            par = [None for j in range(len(dataset.fitparameters))]

            for j, (fitpar, fitparindex) in enumerate(dataset.fitparameters.items()):
                if fitpar not in parameters:
                    msg = f"`Toy`: for `Dataset` {datasetname}, parameter `{fitpar}` not found in passed `parameters`"
                    raise KeyError(msg)

                par[j] = parameters[fitpar]

            # make the fake data for this particular dataset
            toydata = dataset.rvs(*par, seed=thisseed)

            # make the cost function for this particular dataset
            thiscostfunction = dataset._costfunctioncall(toydata, dataset.density)

            # tell the cost function which parameters to use
            thiscostfunction._parameters = dataset.costfunction._parameters

            if i == 0:
                self.costfunction = thiscostfunction
            else:
                self.costfunction += thiscostfunction

        # fitparameters of Toy are a little different than fitparameters of Dataset
        self.fitparameters = self.costfunction._parameters

        for constraintname, constraint in pseuodexperiment.constraints.items():
            self.costfunction += constraint

        guess = {}
        for par in self.fitparameters:
            guess |= {par: parameters[par]}

        self.minuit = Minuit(self.costfunction, **guess)
        self.best = None

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
        for parname, pardict in self.pseuodexperiment.config["parameters"].items():
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

    # this corresponds to t_mu or t_mu^tilde depending on whether there is a limit on the parameters
    def ts(
        self,
        profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
        force: bool = False,
    ) -> float:
        denom = self.bestfit(force=force)["fval"]

        num = self.profile(parameters=profile_parameters)["fval"]

        # because these are already -2*ln(L) from iminuit
        return num - denom
