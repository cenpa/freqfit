"""
A class that controls an experiment and calls the `Superset` class.
"""
import logging

import numpy as np
from iminuit import Minuit

from legendfreqfit.superset import Superset
from legendfreqfit.toy import Toy
from legendfreqfit.utils import grab_results, load_config

SEED = 42

log = logging.getLogger(__name__)


class Experiment(Superset):
    def __init__(
        self,
        config: dict,
        name: str = None,
    ) -> None:
        self.options = {}
        self.options["try_to_combine_datasets"] = False
        self.test_statistic = "t_mu"
        self.toy = None  # the last Toy from this experiment
        self.best = None  # to store the best fit result
        self.guess = None  # store the initial guess
        self.minuit = None  # Minuit object

        self.fixed_bc_no_data = (
            {}
        )  # check which nuisance parameters can be fixed in the fit due to no data

        combined_datasets = None
        if "options" in config:
            if "try_to_combine_datasets" in config["options"]:
                self.options["try_to_combine_datasets"] = config["options"][
                    "try_to_combine_datasets"
                ]
                if "combined_datasets" in config:
                    combined_datasets = config["combined_datasets"]
                    msg = "found 'combined_datasets' in config"
                    logging.info(msg)

            if "name" in config["options"] and (name is None):
                name = config["options"]["name"]

            if "test_statistic" in config["options"]:
                if config["options"]["test_statistic"] in [
                    "t_mu",
                    "t_mu_tilde",
                    "q_mu",
                    "q_mu_tilde",
                ]:
                    self.test_statistic = config["options"]["test_statistic"]
                    msg = f"setting test statistic: {self.test_statistic}"
                    logging.info(msg)
        else:
            msg = "setting test statistic to default: t_mu"
            logging.info(msg)

        constraints = None
        if "constraints" in config:
            constraints = config["constraints"]
            msg = "found 'constraints' in config"
            logging.info(msg)
        else:
            msg = "did not find 'constraints' in config"
            logging.info(msg)

        super().__init__(
            datasets=config["datasets"],
            parameters=config["parameters"],
            constraints=constraints,
            combined_datasets=combined_datasets,
            name=name,
            try_to_combine_datasets=self.options["try_to_combine_datasets"],
        )

        # get the fit parameters and set the parameter initial values
        self.guess = self.initialguess()
        self.minuit = Minuit(self.costfunction, **self.guess)
        self.minuit.tol = 0.00001  # set the tolerance
        self.minuit.strategy = 2

        # raise a RunTime error if function evaluates to NaN
        self.minuit.throw_nan = True

        # check which nuisance parameters can be fixed in the fit due to no data

        # find which parameters are part of Datasets that have data
        parstofitthathavedata = set()
        for datasetname in self.datasets:
            # check if there is some data
            if self.datasets[datasetname].data.size > 0:
                # add the fit parameters of this Dataset if there is some data
                for fitpar in self.datasets[datasetname].fitparameters:
                    parstofitthathavedata.add(fitpar)

        # now check which fit parameters can be fixed if no data and are not part of a Dataset that has data
        for parname in self.fitparameters:
            if (
                "fix_if_no_data" in self.parameters[parname]
                and self.parameters[parname]["fix_if_no_data"]
            ):
                # check if this parameter is part of a Dataset that has data
                if parname not in parstofitthathavedata:
                    self.fixed_bc_no_data[parname] = self.parameters[parname]["value"]

        # to set limits and fixed variables
        # this function will also fix those fit parameters which can be fixed because they are not part of a
        # Dataset that has data
        self.minuit_reset()

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
        for parname, pardict in self.parameters.items():
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
        self.minuit_reset(use_physical_limits=use_physical_limits)

        self.minuit.migrad()

        if not self.minuit.valid:
            msg = (f"`Toy` with seed {self.seed} has invalid best fit")
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

        # set fixed parameters
        for parname, parvalue in parameters.items():
            self.minuit.fixed[parname] = True
            self.minuit.values[parname] = parvalue

        self.minuit.migrad()

        if not self.minuit.valid:
            msg = (f"`Toy` with seed {self.seed} has invalid profile")
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
        if self.test_statistic == "t_mu_tilde" or self.test_statistic == "q_mu_tilde":
            use_physical_limits = True

        denom = self.bestfit(force=force, use_physical_limits=use_physical_limits)[
            "fval"
        ]

        # see Cowan (2011) Eq. 14 and Eq. 16
        if self.test_statistic == "q_mu" or self.test_statistic == "q_mu_tilde":
            for parname, parvalue in profile_parameters.items():
                if self.best["values"][parname] > parvalue:
                    return 0.0

        num = self.profile(
            parameters=profile_parameters, use_physical_limits=use_physical_limits
        )["fval"]

        ts = num - denom

        if ts < 0:
            msg = (f"`Toy` with seed {self.seed} gave test statistic below zero: {ts}")
            logging.warning(msg)

        # because these are already -2*ln(L) from iminuit
        return ts

    def maketoy(
        self,
        parameters: dict,
        seed: int = SEED,
    ) -> Toy:
        self.toy = Toy(experiment=self, parameters=parameters, seed=seed)

        return self.toy

    def toy_ts(
        self,
        parameters: dict,  # parameters and values needed to generate the toys
        profile_parameters: dict
        | list,  # which parameters to fix and their value (rest are profiled)
        num: int = 1,
        seed: np.array = None,
    ):
        """
        Makes a number of toys and returns their test statistics.
        Having the seed be an array allows for different jobs producing toys on the same s-value to have different seed numbers
        """
        np.random.seed(SEED)
        if seed is None:
            seed = np.random.randint(1000000, size=num)
        else:
            if len(seed) != num:
                raise ValueError("Seeds must have same length as the number of toys!")
        if isinstance(profile_parameters, dict):
            ts = np.zeros(num)
            data_to_return = []
            paramvalues_to_return = []
            num_drawn = []
            for i in range(num):
                thistoy = self.maketoy(parameters=parameters, seed=seed[i])
                ts[i] = thistoy.ts(profile_parameters=profile_parameters)
                data_to_return.append(thistoy.toy_data_to_save)
                paramvalues_to_return.append(thistoy.parameters_to_save)
                num_drawn.append(thistoy.toy_num_drawn_to_save)

        # otherwise profile parameters is a list of dicts
        else:
            ts = np.zeros((len(profile_parameters), num))
            data_to_return = []
            paramvalues_to_return = []
            num_drawn = []
            for i in range(num):
                thistoy = self.maketoy(parameters=parameters, seed=seed[i])
                for j in range(len(profile_parameters)):
                    ts[j][i] = thistoy.ts(profile_parameters=profile_parameters[j])
                data_to_return.append(thistoy.toy_data_to_save)
                paramvalues_to_return.append(thistoy.parameters_to_save)
                num_drawn.append(thistoy.toy_num_drawn_to_save)

        # Need to flatten the data_to_return in order to save it in h5py
        data_to_return_flat = (
            np.ones(
                (len(data_to_return), np.nanmax([len(arr) for arr in data_to_return]))
            )
            * np.nan
        )
        for i, arr in enumerate(data_to_return):
            data_to_return_flat[i, : len(arr)] = arr

        num_drawn_to_return_flat = (
            np.ones((len(num_drawn), np.nanmax([len(arr) for arr in num_drawn])))
            * np.nan
        )
        for i, arr in enumerate(num_drawn):
            num_drawn_to_return_flat[i, : len(arr)] = arr

        return ts, data_to_return_flat, paramvalues_to_return, num_drawn_to_return_flat

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
