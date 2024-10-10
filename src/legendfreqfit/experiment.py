"""
A class that controls an experiment and calls the `Superset` class.
"""
import itertools
import logging
from copy import deepcopy

import numpy as np
from iminuit import Minuit

from legendfreqfit import initial_guesses
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
        self.backend = "minuit"
        self.scipy_minimizer = None
        self.toy = None  # the last Toy from this experiment
        self.best = None  # to store the best fit result
        self.guess = None  # store the initial guess
        self.minuit = None  # Minuit object
        self.tolerance = 0.00001  # tolerance for iminuit or other minimizer
        self.scan = False
        self.scan_bestfit = False
        self.scan_grid = None  # this is a dictionary, each key is a fit parameter and its values are the parameter values to scan along in one dimension
        self.user_gradient = (
            False  # option to use a user-specified density gradient for a model
        )
        self.data = (
            []
        )  # A flat array of all the data. The data may be split between datasets, this is just an aggregate
        self.initial_guess_function = None

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

            if "backend" in config["options"]:
                self.backend = config["options"]["backend"]

            if "tolerance" in config["options"]:
                self.tolerance = config["options"]["tolerance"]

            if "scipy_minimizer" in config["options"]:
                if config["options"]["scipy_minimizer"] in ["None", "none"]:
                    self.scipy_minimizer = None
                else:
                    self.scipy_minimizer = config["options"]["scipy_minimizer"]
                    if self.scipy_minimizer not in [
                        "Nelder-Mead",
                        "Powell",
                        "CG",
                        "BFGS",
                        "Newton-CG",
                        "L-BFGS-B",
                        "TNC",
                        "COBYLA",
                        "COBYQA",
                        "SLSQP",
                        "trust-constr",
                        "dogleg",
                        "trust-ncg",
                        "trust-exact",
                        "trust-krylov",
                    ]:
                        raise NotImplementedError(
                            f"{self.scipy_minimizer} is not a valid minimizer"
                        )

            if "user_gradient" in config["options"]:
                self.user_gradient = config["options"]["user_gradient"]

            if "scan" in config["options"]:
                self.scan = config["options"]["scan"]

                if "scan_grid" in config["options"]:
                    self.scan_grid = config["options"]["scan_grid"]

            if "initial_guess_function" in config["options"]:
                if config["options"]["initial_guess_function"] in ["None", "none"]:
                    self.initial_guess_function = None
                else:
                    self.initial_guess_function = config["options"][
                        "initial_guess_function"
                    ]

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
            user_gradient=self.user_gradient,
        )

        # get the fit parameters and set the parameter initial values
        self.guess = self.initialguess()
        self.minuit = Minuit(self.costfunction, **self.guess)
        self.minuit.tol = self.tolerance  # set the tolerance
        self.minuit.strategy = 2

        # raise a RunTime error if function evaluates to NaN
        self.minuit.throw_nan = True

        # check which nuisance parameters can be fixed in the fit due to no data

        # find which parameters are part of Datasets that have data
        parstofitthathavedata = set()
        for datasetname in self.datasets:
            # check if there is some data
            if self.datasets[datasetname].data.size > 0:
                # Add the data to self.data
                self.data.extend(self.datasets[datasetname].data)
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

        # If scan is not none, initialize the hypercube grid
        if (self.scan) and (self.scan_grid is not None):
            self.hypercube_grid = self.create_hypercube(self.scan_grid)

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
        if self.initial_guess_function is None:
            guess = {
                fitpar: self.parameters[fitpar]["value"]
                if "value" in self.parameters[fitpar]
                else None
                for fitpar in self.fitparameters
            }

        else:
            func = getattr(initial_guesses, self.initial_guess_function)
            guess = func(self)

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

        if self.scan_bestfit:
            grid = np.linspace(1.0e-9, 0.2, 200)
            args = [[{"global_S": float(xx)}] for xx in grid]

            ts = []
            for arg in args:
                ts.append(self.profile(arg[0]))
            best = ts[np.argmin([t["fval"] for t in ts])]
            self.best = best
            if not best["valid"]:
                msg = "`Experiment` has invalid best fit"
                logging.warning(msg)

        elif self.scan:
            y = np.empty(len(self.hypercube_grid))
            for i in range(len(self.hypercube_grid)):
                y[i] = self.minuit._fcn(self.hypercube_grid[i])

            best = np.min(y)
            ibest = np.argmin(y)

            self.best = {
                "fval": best,
                "values": {},
                "valid": True,
            }  # TODO: add the rest of return values of interest, like parameter names at minimum
            for par in self.minuit.parameters:
                ipar, vpar = self.minuit._normalize_key(par)
                self.best["values"][par] = self.hypercube_grid[ibest][ipar]

        else:
            try:
                if self.backend == "minuit":
                    self.minuit.migrad()
                elif self.backend == "scipy":
                    self.minuit.scipy(method=self.scipy_minimizer)
                elif self.backend == "minimum_minimizer":
                    # Run through 3 minimizers and pick the best of them
                    self.minuit.simplex()
                    result1 = grab_results(self.minuit)
                    self.minuit.migrad()
                    result2 = grab_results(self.minuit)
                    self.minuit.scipy(method="Powell")
                    result3 = grab_results(self.minuit)
                    min_fval = np.argmin(
                        [result1["fval"], result2["fval"], result3["fval"]]
                    )
                    if min_fval == 0:
                        self.minuit.simplex()
                    elif min_fval == 1:
                        self.minuit.migrad()
                    else:
                        self.minuit.scipy(method="Powell")
                else:
                    raise NotImplementedError(
                        "Iminuit backend is not set to `minuit` or `scipy`"
                    )
            except RuntimeError:
                msg = "`Experiment` has invalid best fit"
                logging.warning(msg)

            if not self.minuit.valid:
                msg = "`Experiment` has invalid best fit"
                logging.warning(msg)

            self.best = grab_results(self.minuit)

        if self.guess == self.best["values"]:
            msg = "`Experiment` has best fit values very close to initial guess"
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

        if self.scan:
            # need to remake the hypercube because there are now fewer parameters
            # pop the fixed parameters from the supplied scan_grid
            new_grid_dict = deepcopy(self.scan_grid)
            for parname in parameters.keys():
                if parname in new_grid_dict.keys():
                    new_grid_dict.pop(parname)
            hypercube_grid = self.create_hypercube(new_grid_dict)

            y = np.empty(len(hypercube_grid))
            for i in range(len(hypercube_grid)):
                y[i] = self.minuit._fcn(hypercube_grid[i])

            best = np.min(y)
            ibest = np.argmin(y)
            results = {
                "fval": best,
                "values": {},
                "valid": True,
            }  # TODO: add the rest of return values of interest
            for par in self.minuit.parameters:
                ipar, vpar = self.minuit._normalize_key(par)
                results["values"][par] = hypercube_grid[ibest][ipar]

        else:
            try:
                if self.backend == "minuit":
                    self.minuit.migrad()
                elif self.backend == "scipy":
                    self.minuit.scipy(method=self.scipy_minimizer)
                elif self.backend == "minimum_minimizer":
                    self.minuit.simplex()
                    result1 = grab_results(self.minuit)
                    self.minuit.migrad()
                    result2 = grab_results(self.minuit)
                    self.minuit.scipy(method="Powell")
                    result3 = grab_results(self.minuit)
                    min_fval = np.argmin(
                        [result1["fval"], result2["fval"], result3["fval"]]
                    )
                    if min_fval == 0:
                        self.minuit.simplex()
                    elif min_fval == 1:
                        self.minuit.migrad()
                    else:
                        self.minuit.scipy(method="Powell")
                else:
                    raise NotImplementedError(
                        "Iminuit backend is not set to `minuit` or `scipy`"
                    )
            except RuntimeError:
                msg = f"`Experiment` throwing NaN has invalid profile at {parameters}"
                logging.warning(msg)

            if not self.minuit.valid:
                msg = "`Experiment` has invalid profile"
                logging.warning(msg)

            results = grab_results(self.minuit)

        if self.guess == results["values"]:
            msg = "`Experiment` has profile fit values very close to initial guess"
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
            msg = f"`Experiment` gave test statistic below zero: {ts}"
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

    def create_hypercube(self, scan_grid) -> list:
        """
        Parameters
        ----------
        m
            Minuit object that holds that parameters and parameter names
        scan_grid
            The dictionary object that holds the points in each dimension to scan
        """
        scan_pars = list(scan_grid.keys())
        scan_list_of_lists = [list(scan_grid[scan_par]) for scan_par in scan_pars]
        hypercube_missing_other_values = itertools.product(
            *scan_list_of_lists
        )  # This list is missing all of the other function values that the minuit object needs, let's add them back in!

        # get the indices within the minuit object of all of parameters on the grid
        ipar_list = []
        for scan_par in scan_pars:
            ipar_list.append(self.minuit._normalize_key(scan_par)[0])
        ipar_list = np.array(ipar_list)
        values = deepcopy(
            np.array(self.minuit.values)
        )  # Get all the values, don't mutate them

        hypercube = []
        for par in hypercube_missing_other_values:
            values[ipar_list] = par
            hypercube.append(deepcopy(values))

        return hypercube
