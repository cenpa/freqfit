"""
A class that holds a collection of fake datasets and associated hardware
"""
import logging

import numpy as np
from iminuit import Minuit, cost

from legendfreqfit.utils import grab_results
from legendfreqfit.dataset import combine_datasets

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

        self.experiment = experiment # the experiment this Toy is based on
        self.toy_data_to_save = ([])  # list to store the data drawn for this toy. A flat array, consists of all data from all datasets
        self.varied_nuisance_to_save = ([])  # 2d array of randomly varied nuisance parameters, per dataset
        self.costfunction = None # costfunction for this Toy
        self.fitparameters = None # fit parameters from the costfunction, reference to self.costfunction._parameters
        self.minuit = None # Minuit object
        self.best = None # best fit
        self.fixed_bc_no_data = {} # parameters that can be fixed because no data in their Datasets
        self.combined_datasets = {} # holds combined_datasets
        self.included_in_combined_datasets = {}

        # vary the toy parameters as indicated
        if len(self.experiment._toypars_to_vary) > 0:
            np.random.seed(seed=seed)

            pars, vals, covar = self.experiment.get_constraints(self.experiment._toypars_to_vary)

            varied_toypars = np.random.multivariate_normal(vals, covar)

            # now assign the random values to the passed parameters (or to not passed parameters?)
            for i, par in enumerate(pars):
                parameters[par] = varied_toypars[i]
                self.experiment._toy_parameters[par]["value"] = varied_toypars[i]

            # Save the values of these randomized nuisance parameters
            self.varied_nuisance_to_save.append(varied_toypars)

        # draw the toy data
        for i, (datasetname, dataset) in enumerate(experiment.datasets.items()):
            # worried that this is not totally deterministic (depends on number of Datasets),
            # but more worried that random draws between datasets would be correlated otherwise.
            thisseed = seed + i

            # allocate length in case `parameters` is passed out of order
            pars = [None for j in range(len(dataset.fitparameters))]

            for j, (fitpar, fitparindex) in enumerate(dataset.fitparameters.items()):
                if fitpar not in parameters:
                    msg = f"`Toy`: for `Dataset` {datasetname}, parameter `{fitpar}` not found in passed `parameters`"
                    raise KeyError(msg)

                pars[j] = parameters[fitpar]

            # make the toy for this particular dataset
            dataset.toy(par=pars, seed=thisseed) # saved in dataset._toy_data
            self.toy_data_to_save.extend(dataset._toy_data)  # store the data as list of lists

        # combine datasets
        if self.experiment.options["try_to_combine_datasets"]:
            # maybe there's more than one combined_dataset group
            for cdsname in self.experiment._combined_datasets_config:
                # find the Datasets to try to combine
                ds_tocombine = []
                for dsname in self.experiment.datasets.keys():
                    if self.experiment.datasets[dsname].try_combine and self.experiment.datasets[dsname].combined_dataset == cdsname:
                        ds_tocombine.append(self.experiment.datasets[dsname])

                combined_dataset, included_datasets = combine_datasets(
                    datasets=ds_tocombine, 
                    model=self.experiment._combined_datasets_config[cdsname]["model"],
                    model_parameters=self.experiment._combined_datasets_config[cdsname]["model_parameters"],
                    parameters=self.experiment._toy_parameters,
                    costfunction=self.experiment._combined_datasets_config[cdsname]["costfunction"],
                    name=self.experiment._combined_datasets_config[cdsname]["name"] if "name" in self.experiment._combined_datasets_config[cdsname] else "",
                    use_toy_data=True,
                )

                # now to see what datasets actually got included in the combined_dataset
                for dsname in included_datasets:
                    self.experiment.datasets[dsname]._toy_is_combined = True

                if len(included_datasets) > 0:
                    self.combined_datasets[cdsname] = combined_dataset
                    self.included_in_combined_datasets[cdsname] = included_datasets

        # to find which parameters are part of Datasets that have data
        parstofitthathavedata = set()

        # add the costfunctions together
        first = True
        for dsname, ds in self.experiment.datasets.items():
            # skip this dataset if it has been combined
            if ds._toy_is_combined:
                continue
            # make the cost function for this particular dataset
            thiscostfunction = ds._costfunctioncall(ds._toy_data, ds.density)
            # tell the cost function which parameters to use
            thiscostfunction._parameters = ds.costfunction._parameters
            if first:
                self.costfunction = thiscostfunction
                first = False
            else:
                self.costfunction += thiscostfunction
            
            # find which parameters are part of Datasets that have data
            if ds._toy_data.size > 0:
                # add the fit parameters of this Dataset if there is some data
                for fitpar in thiscostfunction._parameters:
                    parstofitthathavedata.add(fitpar)  

        for cdsname, cds in self.combined_datasets.items():
            # make the cost function for this particular combined_dataset
            thiscostfunction = cds._costfunctioncall(cds.data, cds.density)
            # tell the cost function which parameters to use
            thiscostfunction._parameters = cds.costfunction._parameters
            if first:
                self.costfunction = thiscostfunction
                first = False
            else:
                self.costfunction += thiscostfunction

            # find which parameters are part of combined_datasets that have data
            if cds.data.size > 0:
                # add the fit parameters of this combined_dataset if there is some data
                for fitpar in thiscostfunction._parameters:
                    parstofitthathavedata.add(fitpar)  

        # fitparameters of Toy are a little different than fitparameters of Dataset
        self.fitparameters = self.costfunction._parameters

        # get the cosntraints needed from the fitparameters
        # but note that we need to adjust the central values for the varied pars
        # check if they are in the varied pars list and pull the value from parameters if so

        pars, values, covariance = self.experiment.get_constraints(self.fitparameters)

        if pars is not None:
            for i, par in enumerate(pars):
                if par in self.experiment._toypars_to_vary:
                    values[i] = parameters[par]
        
            self.constraints = cost.NormalConstraint(pars, values, error=covariance)

            self.costfunction = self.costfunction + self.constraints

        guess = {}
        for par in self.fitparameters:
            guess |= {par: parameters[par]}

        self.minuit = Minuit(self.costfunction, **guess)

        # now check which nuisance parameters can be fixed if no data and are not part of a Dataset that has data
        for parname in self.fitparameters:
            if (
                "fix_if_no_data" in self.experiment._toy_parameters[parname]
                and self.experiment._toy_parameters[parname]["fix_if_no_data"]
            ):
                # check if this parameter is part of a Dataset that has data
                if parname not in parstofitthathavedata:
                    self.fixed_bc_no_data[parname] = self.experiment._toy_parameters[
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
        for parname, pardict in self.experiment._toy_parameters.items():
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
