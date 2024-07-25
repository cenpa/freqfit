"""
A class that holds a collection of fake datasets and associated hardware
"""
import logging

import numpy as np
from iminuit import Minuit, cost

from legendfreqfit.utils import grab_results

SEED = 42

log = logging.getLogger(__name__)


class Toy:
    def __init__(
        self,
        experiment,
        parameters: dict,
        seed: int = SEED,
    ) -> None:
        """
        experiment
            `experiment` to base this `Toy` on
        parameters
            `dict` of parameters and their values to model with
        vary_nuisance
            whether to vary the nuisance parameters by their constraints. If True, draws a new value of the nuisance
            parameter and sets the constraint to be centered at this new value.
        """

        self.experiment = experiment
        self.toydata_to_save = (
            []
        )  # list to store the data drawn for this toy. A flat array, consists of all data from all datasets
        self.varied_nuisance_to_save = (
            []
        )  # 2d array of randomly varied nuisance parameters, per dataset

        # draw random nuisance parameters according to their constraints
        if self.experiment._nuisance_to_vary_values is not None:
            np.random.seed(seed=seed)
            randnuisance = np.random.multivariate_normal(
                self.experiment._nuisance_to_vary_values,
                self.experiment._nuisance_to_vary_covar,
            )

            # now assign the random values to the passed parameters (or to not passed parameters?)
            for i, nuipar in enumerate(self.experiment.nuisance_to_vary):
                parameters[nuipar] = randnuisance[i]

            # Save the values of these randomized nuisance parameters
            self.varied_nuisance_to_save.append(randnuisance)

        # find which parameters are part of Datasets that have data
        parstofitthathavedata = set()

        self.costfunction = None
        for i, (datasetname, dataset) in enumerate(experiment.datasets.items()):
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
            self.toydata_to_save.extend(toydata)  # store the data as list of lists

            # make the cost function for this particular dataset
            thiscostfunction = dataset._costfunctioncall(toydata, dataset.density)

            # tell the cost function which parameters to use
            thiscostfunction._parameters = dataset.costfunction._parameters

            # find which parameters are part of Datasets that have data
            if toydata.size > 0:
                # add the fit parameters of this Dataset if there is some data
                for fitpar in thiscostfunction._parameters:
                    parstofitthathavedata.add(fitpar)

            if i == 0:
                self.costfunction = thiscostfunction
            else:
                self.costfunction += thiscostfunction

        # fitparameters of Toy are a little different than fitparameters of Dataset
        self.fitparameters = self.costfunction._parameters

        # we need to adjust the constraints for nuisance parameters if we varied them
        # just the central values though, not the uncertainties
        # so probably the easiest way to do this is to make new objects rather than reference the NormalConstraint
        # from the parent Experiment
        for constraintname, constraint in experiment.constraints.items():
            conval = constraint.value
            for i, par in enumerate(constraint._parameters):
                if par in self.experiment.nuisance_to_vary:
                    conval[i] = parameters[par]
            self.costfunction += cost.NormalConstraint(
                constraint._parameters, conval, error=constraint.covariance
            )

        guess = {}
        for par in self.fitparameters:
            guess |= {par: parameters[par]}

        self.minuit = Minuit(self.costfunction, **guess)
        self.best = None

        # now check which nuisance parameters can be fixed if no data and are not part of a Dataset that has data
        self.fixed_bc_no_data = {}
        for parname in self.experiment.nuisance:
            if (
                "fix_if_no_data" in self.experiment.parameters[parname]
                and self.experiment.parameters[parname]["fix_if_no_data"]
            ):
                # check if this parameter is part of a Dataset that has data
                if parname not in parstofitthathavedata:
                    self.fixed_bc_no_data[parname] = self.experiment.parameters[
                        parname
                    ]["value"]

        # to set limits and fixed variables
        # this function will also fix those nuisance parameters which can be fixed because they are not part of a
        # Dataset that has data
        self.minuit_reset()

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
        for parname, pardict in self.experiment.parameters.items():
            if parname in self.minuit.parameters:
                if "limits" in pardict:
                    self.minuit.limits[parname] = pardict["limits"]
                if "fixed" in pardict:
                    self.minuit.fixed[parname] = pardict["fixed"]
                # fix those nuisance parameters which can be fixed because they are not part of a
                # Dataset that has data
                if parname in self.fixed_bc_no_data:
                    self.minuit.fixed[parname] = True
                    self.minuit.values[parname] = self.fixed_bc_no_data[parname]
        return

    def bestfit(
        self,
        force: bool = False,
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

    # mostly pulled directly from iminuit, with some modifications to ignore empty Datasets and also to format
    # plots slightly differently
    def visualize(
        self,
        parameters: dict,
        component_kwargs=None,
    ) -> None:
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current figure.
        Subplots are created to visualize each part of the cost function, the figure
        height is increased accordingly. Parts without a visualize method are silently
        ignored.

        Does not draw Datasets that are empty (have no events).

        Parameters
        ----------
        args : array-like
            Parameter values.
        component_kwargs : dict of dicts, optional
            Dict that maps an index to dict of keyword arguments. This can be
            used to pass keyword arguments to a visualize method of a component with
            that index.
        **kwargs :
            Other keyword arguments are forwarded to all components.
        """
        from matplotlib import pyplot as plt

        args = []
        for par in self.fitparameters:
            if par not in parameters:
                msg = f"parameter {par} was not provided"
                raise KeyError(msg)
            args.append(parameters[par])

        n = 0
        for comp in self.costfunction:
            if hasattr(comp, "visualize"):
                if hasattr(comp, "data"):
                    if len(comp.data) > 0:
                        n += 1
                else:
                    n += 1

        fig = plt.gcf()
        fig.set_figwidth(n * fig.get_figwidth() / 1.5)
        _, ax = plt.subplots(1, n, num=fig.number)

        if component_kwargs is None:
            component_kwargs = {}

        i = 0
        for k, (comp, cargs) in enumerate(self.costfunction._split(args)):
            if not hasattr(comp, "visualize"):
                continue
            if hasattr(comp, "data") and len(comp.data) == 0:
                continue
            kwargs = component_kwargs.get(k, {})
            plt.sca(ax[i])
            comp.visualize(cargs, **kwargs)
            i += 1
