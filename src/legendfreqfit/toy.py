"""
A class that holds a collection of fake datasets and associated hardware
"""
import logging
from copy import deepcopy

import numpy as np
from iminuit import Minuit, cost

from legendfreqfit import initial_guesses
from legendfreqfit.dataset import combine_datasets
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
        """

        self.experiment = experiment  # the experiment this Toy is based on
        self.toy_data_to_save = (
            []
        )  # list to store the data drawn for this toy. A flat array, consists of all data from all datasets
        self.toy_num_drawn_to_save = (
            []
        )  # list to store tuples of the number of signal and background counts drawn per dataset
        self.parameters_to_save = np.array(
            [np.full(len(self.experiment._toy_parameters), np.nan)]
        )  # the parameters of the toy
        self.costfunction = None  # costfunction for this Toy
        self.fitparameters = None  # fit parameters from the costfunction, reference to self.costfunction._parameters
        self.minuit = None  # Minuit object
        self.guess = None  # initial guess for minuit
        self.best = None  # best fit
        self.fixed_bc_no_data = (
            {}
        )  # parameters that can be fixed because no data in their Datasets
        self.combined_datasets = {}  # holds combined_datasets
        self.included_in_combined_datasets = {}
        self.seed = seed
        self.user_gradient = experiment.user_gradient
        self.data = (
            []
        )  # A flat array of all the data. The data may be split between datasets, this is just an aggregate

        # reset toy_parameters to "default" values
        # deepcopy so we can mutate this without fear
        self.experiment._toy_parameters = deepcopy(self.experiment.parameters)

        # overwrite the toy parameters with the passed parameters
        for par in parameters.keys():
            self.experiment._toy_parameters[par]["value"] = parameters[par]

        # If datasets have been combined, re-assign those combined parameter values to the de-combined toy_parameters
        for combined_ds in experiment.included_in_combined_datasets.keys():
            # Datasets may have been combined into more than one new datasets, that's why we loop over the keys
            # Now, loop over datasets and add back in their parameter values
            for ds in experiment.included_in_combined_datasets[combined_ds]:
                # Check to make sure that the user hasn't overridden any of the parameters in datasets that were combined
                # This needs to happen so we don't overwrite them when we de-combine datasets next
                if any(
                    x in list(parameters.keys())
                    for x in list(experiment.datasets[ds].model_parameters.keys())
                ):
                    raise NotImplementedError(
                        "Overriding a parameter that is in a combined dataset is not supported."
                    )

                # Loop through the model parameters in the dataset and find their values from the combined dataset
                for model_par in experiment.datasets[ds].model_parameters.keys():
                    ds_par_name = experiment.datasets[ds].model_parameters[model_par]
                    # find the corresponding parameter name in the combined dataset
                    combined_ds_par_name = experiment.combined_datasets[
                        combined_ds
                    ].model_parameters[model_par]

                    # nuisance parameters should not be passed in parameters, so we need to skip the nuisance parameters from the dataset
                    if combined_ds_par_name in parameters.keys():
                        parameters[ds_par_name] = parameters[combined_ds_par_name]

        # vary the toy parameters as indicated
        if len(self.experiment._toy_pars_to_vary) > 0:
            np.random.seed(seed=self.seed)

            pars, vals, covar = self.experiment.get_constraints(
                self.experiment._toy_pars_to_vary
            )

            varied_toy_pars = []
            # check if parameters are all independent, draw from simpler distribution if so
            if np.all(covar == np.diag(np.diagonal(covar))):
                varied_toy_pars = np.random.normal(vals, np.sqrt(np.diagonal(covar)))
            else:
                varied_toy_pars = np.random.multivariate_normal(
                    vals, covar
                )  # sooooooooo SLOW

            # now assign the random values to the passed parameters (or to not passed parameters?)
            for i, par in enumerate(pars):
                parameters[par] = varied_toy_pars[i]
                self.experiment._toy_parameters[par]["value"] = varied_toy_pars[i]

        # draw the toy data
        for i, (datasetname, dataset) in enumerate(experiment.datasets.items()):
            # worried that this is not totally deterministic (depends on number of Datasets),
            # but more worried that random draws between datasets would be correlated otherwise.
            thisseed = self.seed + i

            # allocate length in case `parameters` is passed out of order
            pars = [None for j in range(len(dataset.fitparameters))]

            for j, (fitpar, fitparindex) in enumerate(dataset.fitparameters.items()):
                if fitpar not in parameters:
                    msg = f"`Toy`: for `Dataset` {datasetname}, parameter `{fitpar}` not found in passed `parameters`"
                    raise KeyError(msg)

                pars[j] = parameters[fitpar]

            # make the toy for this particular dataset
            dataset.toy(par=pars, seed=thisseed)  # saved in dataset._toy_data
            self.toy_data_to_save.extend(
                dataset._toy_data
            )  # store the data as list of lists
            self.toy_num_drawn_to_save.extend(
                dataset._toy_num_drawn
            )  # also save the number of signal and background counts drawn

        # combine datasets
        if self.experiment.options["try_to_combine_datasets"]:
            # maybe there's more than one combined_dataset group
            for cdsname in self.experiment._combined_datasets_config:
                # find the Datasets to try to combine
                ds_tocombine = []
                for dsname in self.experiment.datasets.keys():
                    if (
                        self.experiment.datasets[dsname].try_combine
                        and self.experiment.datasets[dsname].combined_dataset == cdsname
                    ):
                        ds_tocombine.append(self.experiment.datasets[dsname])

                combined_dataset, included_datasets = combine_datasets(
                    datasets=ds_tocombine,
                    model=self.experiment._combined_datasets_config[cdsname]["model"],
                    model_parameters=self.experiment._combined_datasets_config[cdsname][
                        "model_parameters"
                    ],
                    parameters=self.experiment._toy_parameters,
                    costfunction=self.experiment._combined_datasets_config[cdsname][
                        "costfunction"
                    ],
                    name=cdsname,
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
                # Add the data to self.data
                self.data.extend(ds._toy_data)
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

        # get the constraints needed from the fitparameters
        # but note that we need to adjust the central values for the varied pars
        # check if they are in the varied pars list and pull the value from parameters if so

        pars, values, covariance = self.experiment.get_constraints(self.fitparameters)

        if pars is not None:
            for i, par in enumerate(pars):
                if par in self.experiment._toy_pars_to_vary:
                    values[i] = parameters[par]

            self.constraints = cost.NormalConstraint(pars, values, error=covariance)

            self.costfunction = self.costfunction + self.constraints

        self.guess = self.initialguess()

        self.minuit = Minuit(self.costfunction, **self.guess)
        self.minuit.tol = 0.00001  # set the tolerance
        self.minuit.strategy = 2

        # raise a RunTime error if function evaluates to NaN
        self.minuit.throw_nan = True

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

        # save the values of the toy parameters
        for i, (parname, pardict) in enumerate(self.experiment._toy_parameters.items()):
            self.parameters_to_save[0, i] = pardict["value"]

        # to set limits and fixed variables
        # this function will also fix those nuisance parameters which can be fixed because they are not part of a
        # Dataset that has data
        self.minuit_reset()

    def initialguess(
        self,
    ) -> dict:
        if self.experiment.initial_guess_function is None:
            guess = {}
            for par in self.fitparameters:
                guess |= {par: self.experiment._toy_parameters[par]["value"]}

        else:
            func = getattr(initial_guesses, self.experiment.initial_guess_function)
            guess = func(self)

        return guess

    def minuit_reset(
        self,
        use_physical_limits: bool = True,  # for numerators of test statistics, want this to be False
    ) -> None:
        # resets the minimization and stuff
        # does not change limits but does remove "fixed" attribute of variables
        # 2024/08/09: This no longer seems to be true? Not sure if something changed in iminuit or if I was wrong?
        self.minuit.reset()

        # overwrite the limits
        # note that this information can also be contained in a Dataset when instantiated
        # and is overwritten here

        # also set which parameters are fixed
        for parname, pardict in self.experiment._toy_parameters.items():
            if parname in self.minuit.parameters:
                self.minuit.fixed[parname] = False
                self.minuit.limits[parname] = (-1.0 * np.inf, np.inf)

                if "limits" in pardict:
                    self.minuit.limits[parname] = pardict["limits"]
                if use_physical_limits and "physical_limits" in pardict:
                    self.minuit.limits[parname] = pardict["physical_limits"]
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
        use_physical_limits: bool = True,
    ) -> dict:
        # don't run this more than once if we don't have to
        if self.best is not None and not force:
            return self.best

        # remove any previous minimizations
        self.minuit_reset(use_physical_limits=use_physical_limits)

        try:
            if self.experiment.backend == "minuit":
                self.minuit.migrad()
            elif self.experiment.backend == "scipy":
                self.minuit.scipy(method=self.experiment.scipy_minimizer)
            else:
                raise NotImplementedError(
                    "Iminuit backend is not set to `minuit` or `scipy`"
                )

        except RuntimeError:
            msg = f"`Toy` with seed {self.seed} has best fit throwing NaN"
            logging.warning(msg)

        if not self.minuit.valid:
            msg = f"`Toy` with seed {self.seed} has invalid best fit"
            logging.warning(msg)

        self.best = grab_results(self.minuit)

        if self.guess == self.best["values"]:
            msg = f"`Toy` with seed {self.seed} has best fit values very close to initial guess"
            logging.warning(msg)

        return self.best

    def profile(
        self,
        parameters: dict,
        use_physical_limits: bool = True,
    ) -> dict:
        """
        parameters
            `dict` where keys are names of parameters to fix and values are the value that the parameter should be
            fixed to
        """

        self.minuit_reset(use_physical_limits=use_physical_limits)

        for parname, parvalue in parameters.items():
            self.minuit.fixed[parname] = True
            self.minuit.values[parname] = parvalue

        try:
            if self.experiment.backend == "minuit":
                self.minuit.migrad()
            elif self.experiment.backend == "scipy":
                self.minuit.scipy(method=self.experiment.scipy_minimizer)
            else:
                raise NotImplementedError(
                    "Iminuit backend is not set to `minuit` or `scipy`"
                )
        except RuntimeError:
            msg = f"`Toy` with seed {self.seed} has profile raising NaN with parameters {parameters}"
            logging.warning(msg)

        if not self.minuit.valid:
            msg = f"`Toy` with seed {self.seed} has invalid profile"
            logging.warning(msg)

        results = grab_results(self.minuit)

        if self.guess == results["values"]:
            msg = f"`Toy` with seed {self.seed} has profile fit values very close to initial guess"
            logging.warning(msg)

        # also include the fixed parameters
        for parname, parvalue in parameters.items():
            results[parname] = parvalue

        return results

    # this corresponds to t_mu or t_mu^tilde depending on whether there is a physical limit on the parameters
    def ts(
        self,
        profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
        force: bool = False,
    ) -> float:
        """
        force
            See `experiment.bestfit()` for description. Default is `False`.
        """

        use_physical_limits = False  # for t_mu and q_mu
        if (
            self.experiment.test_statistic == "t_mu_tilde"
            or self.experiment.test_statistic == "q_mu_tilde"
        ):
            use_physical_limits = True

        denom = self.bestfit(force=force, use_physical_limits=use_physical_limits)[
            "fval"
        ]

        # see Cowan (2011) Eq. 14 and Eq. 16
        if (
            self.experiment.test_statistic == "q_mu"
            or self.experiment.test_statistic == "q_mu_tilde"
        ):
            for parname, parvalue in profile_parameters.items():
                if self.best["values"][parname] > parvalue:
                    return 0.0

        num = self.profile(
            parameters=profile_parameters, use_physical_limits=use_physical_limits
        )["fval"]

        ts = num - denom

        if ts < 0:
            msg = f"`Toy` with seed {self.seed} gave test statistic below zero: {ts}"
            logging.warning(msg)

        # because these are already -2*ln(L) from iminuit
        return ts

    # mostly pulled directly from iminuit, with some modifications to ignore empty Datasets and also to format
    # plots slightly differently
    def visualize(
        self,
        component_kwargs=None,
    ) -> None:
        """
        Visualize the last toy.

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

        parameters = {}
        for par, pardict in self.experiment._toy_parameters.items():
            parameters[par] = pardict["value"]

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
