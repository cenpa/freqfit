"""
A class that controls an experiment and calls the `Superset` class.
"""
from collections.abc import Callable
import itertools
import logging
from copy import deepcopy

import numpy as np
from iminuit import Minuit

from .superset import Superset
from .toy import Toy
# from .workspace import Workspace
from .constraints import Constraints
from .parameters import Parameters

SEED = 42

log = logging.getLogger(__name__)

class Experiment:
    def __init__(
        self,
        datasets: dict,
        parameters: type[Parameters],
        constraints: type[Constraints],
        options: dict,
    ) -> None:
        
        self.datasets = datasets 
        self.parameters = parameters
        self.options = options
        self.costfunction = None
        self.best = None

        # set initial guess function
        self.guessfcn = options["initial_guess_fcn"]
        if self.guessfcn is None:
            self.guessfcn = initial_guess

        # check which nuisance parameters can be fixed in the fit due to no data
        self.fixed_bc_no_data = {}  

        # get the parameters of the fit
        self.fitparameters = parameters.get_fitparameters(self.datasets)

        # add the costfunctions together
        first = True
        for dsname in self.datasets.keys():
            if first:
                self.costfunction = self.datasets[dsname].costfunction
                first = False
            else:
                self.costfunction += self.datasets[dsname].costfunction
        
        if constraints:
            self.costfunction += constraints.get_cost(self.fitparameters)

        # evaluate the initial guess and store it for later
        self.guess = self.guessfcn(self.datasets, self.parameters, constraints)
        
        # create the Minuit object
        self.minuit = Minuit(self.costfunction, **self.guess)

        # set Minuit options
        self.minuit.tol = options["iminuit_tolerance"]  # set the tolerance
        self.minuit.precision = options["iminuit_precision"]
        self.minuit.strategy = options["iminuit_strategy"]
        self.minuit.throw_nan = True # raise a RunTime error if function evaluates to NaN

        # get fit parameters that are only part of Datasets with no data
        parsnodata = self.parameters.get_fitparameters(self.datasets, nodata=True)

        # now check which fit parameters can be fixed if no data and are not part of a Dataset that has data
        for parname in parsnodata:
            if parameters(parname)["fix_if_no_data"]:
                self.fixed_bc_no_data[parname] = self.parameters(parname)["value"]

        # to set limits and fixed variables
        # this function will also fix those fit parameters which can be fixed because they are not part of a
        # Dataset that has data
        self.minuit_reset()

    def minuit_reset(
        self,
        use_physical_limits: bool = True,  # for numerators of test statistics, want this to be False
    ) -> None:
        # resets the minimization and stuff
        # does not change limits but does remove "fixed" attribute of variables
        # 2024/08/09: This no longer seems to be true? Not sure if something changed in iminuit or if I was wrong?
        self.minuit.reset()

        # Restore the first initial guess
        for key in self.guess.keys():
            self.minuit.values[key] = self.guess[key]

        # overwrite the limits
        # also set which parameters are fixed
        for parname, pardict in self.fitparameters.items():
            if parname in self.minuit.parameters:
                self.minuit.fixed[parname] = pardict["fixed"]
                self.minuit.limits[parname] = pardict["limits"]
                if use_physical_limits:
                    self.minuit.limits[parname] = pardict["physical_limits"]
                
                # fix those nuisance parameters which can be fixed because they are not part of a
                # Dataset that has data
                if parname in self.fixed_bc_no_data:
                    self.minuit.fixed[parname] = True
                    self.minuit.values[parname] = self.fixed_bc_no_data[parname]

        return
    
    def grab_results(
        self,
    ) -> dict:
        # I checked whether we need to shallow/deep copy these and it seems like we do not

        toreturn = {}
        toreturn["errors"] = self.minuit.errors.to_dict()  # returns dict
        toreturn["fixed"] = self.minuit.fixed.to_dict()  # returns dict
        toreturn["fval"] = self.minuit.fval  # returns float
        toreturn["nfit"] = self.minuit.nfit  # returns int
        toreturn["npar"] = self.minuit.npar  # returns int
        toreturn["parameters"] = self.minuit.parameters  # returns tuple of str
        toreturn["tol"] = self.minuit.tol  # returns float
        toreturn["valid"] = self.minuit.valid  # returns bool
        toreturn["values"] = self.minuit.values.to_dict()  # returns dict

        if self.options["use_grid_rounding"]:
            toreturn["values"] = {
                key: np.round(value, self.parameters(key)["grid_rounding_num_decimals"])
                for key, value in self.minuit.values.to_dict().items()
            }  # returns dict

            # overwrite the fval at the truncated params point in parameter space
            toreturn["fval"] = self.minuit._fcn(
                toreturn["values"].values()
            )  

        return toreturn

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

        try:
            if self.options["backend"] == "minuit":
                self.minuit.migrad(**self.options["minimizer_options"])
            else: # scipy
                self.minuit.scipy(
                    method=self.scipy_minimizer, 
                    options=self.options["minimizer_options"],
                )
        except RuntimeError:
            msg = "Experiment has invalid best fit"
            logging.debug(msg)

        if not self.minuit.valid:
            msg = "Experiment has invalid best fit"
            logging.debug(msg)

        self.best = self.grab_results()

        if self.guess == self.best["values"]:
            msg = "Experiment has best fit values very close to initial guess"
            logging.debug(msg)

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

        try:
            if self.options["backend"] == "minuit":
                self.minuit.migrad(**self.options["minimizer_options"])
            else: # scipy
                self.minuit.scipy(
                    method=self.scipy_minimizer, 
                    options=self.options["minimizer_options"],
                )

        except RuntimeError:
            msg = f"Experiment throwing NaN has invalid profile at {parameters}"
            logging.debug(msg)

        if not self.minuit.valid:
            msg = "Experiment has invalid profile"
            logging.debug(msg)

        results = self.grab_results()

        if self.guess == results["values"]:
            msg = "Experiment has profile fit values very close to initial guess"
            logging.debug(msg)

        # also include the fixed parameters
        for parname, parvalue in parameters.items():
            results["values"][parname] = parvalue

        return results

    def ts(
        self,
        profile_parameters: dict,
        force: bool = False,
    ) -> float:
        """
        Compute the profiled test statistic at some point in the parameter space of interest, profiling out the other 
        parameters.

        To use the t_mu_tilde test statistic, you must specify a physical limit on the parameters or interest. 
        Otherwise, the test statistic computed is t_mu (which is correct under no physical limit on the parameter). 
        It is up to the user to specify this, and it is not checked.

        Parameters
        ----------
        profile_parameters : dict
            which parameters to fix and at what value (rest are profiled out)
        force : bool, optional
            Whether to use the stored best fit (default: `False`) or to recompute it (`True`). 
            See `experiment.bestfit()` for description.
        """

        use_physical_limits = False  # for t_mu and q_mu
        if self.options["test_statistic"] == "t_mu_tilde" or self.options["test_statistic"] == "q_mu_tilde":
            use_physical_limits = True

        denom = self.bestfit(force=force, use_physical_limits=use_physical_limits)[
            "fval"
        ]

        # see G. Cowan, K. Cranmer, E. Gross, and O. Vitells,  Eur. Phys. J. C 71, 1554 (2011) - eqns 14 and 16
        if self.options["test_statistic"] == "q_mu" or self.options["test_statistic"] == "q_mu_tilde":
            for parname, parvalue in profile_parameters.items():
                if self.best["values"][parname] > parvalue:
                    return 0.0, 0.0, 0.0

        num = self.profile(
            parameters=profile_parameters, use_physical_limits=use_physical_limits
        )["fval"]

        ts = num - denom

        if ts < 0:
            msg = f"Experiment gave test statistic below zero: {ts}"
            logging.debug(msg)

        # because these are already -2*ln(L) from iminuit
        return ts, denom, num

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
            # relies on ordering, but this should be the same
            if k < len(self.datasets):
                ax[i].set_title(self.datasets[list(self.datasets.keys())[k]].name)
            i += 1
        
        return fig

    # mostly pulled directly from iminuit, with some modifications to ignore empty Datasets and also to format
    # plots slightly differently
    def pull_plot(
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
        totnum = 0
        for comp in self.costfunction:
            totnum += 1
            if hasattr(comp, "visualize"):
                if hasattr(comp, "data"):
                    if len(comp.data) > 0:
                        n += 1
                else:
                    n += 1

        fig = plt.gcf()
        # fig.set_figwidth(n * fig.get_figwidth() / 1.5)
        _, ax = plt.subplots(1, 1, num=fig.number)

        if component_kwargs is None:
            component_kwargs = {}

        i = 0
        for k, (comp, cargs) in enumerate(self.costfunction._split(args)):
            # assumes nuisance constraints are the last cost function...
            if k != totnum - 1:
                continue
            if not hasattr(comp, "visualize"):
                continue
            if hasattr(comp, "data") and len(comp.data) == 0:
                continue
            kwargs = component_kwargs.get(k, {})
            plt.sca(ax)
            comp.visualize(cargs, **kwargs)
            i += 1
        
        return fig

# default initial guess function - user should probably provide their own
def initial_guess(
    datasets: dict,
    parameters: type[Parameters],
    constraints: type[Constraints] = None,
    ) -> dict:

    # get fit parameters of these datasets
    pars = parameters.get_fitparameters(datasets)

    return {p:pars[p]["value"] for p in list(pars)}

# class Toy(Experiment):
#     def __init__(
#         self,
#         datasets: dict,
#         parameters: type[Parameters],
#         constraints: type[Constraints],
#         options: dict,
#     ) -> None:

#         # stores these for later when needed for rvs
#              # doesn't provide access to Experiment class methods until toy data has been generated

#         self.toy_datasets = datasets
#         self.options = options
#         self.parameters = parameters
#         self.toy_constraints = constraints

#     def reset(
#         self,
#     ) -> None:
#         self.validtoy = False
    
#     def rvs(
#         self,
#         toy_parameters: dict,
#     ) -> None:


#         self.reset()


#         super.init(
#             datasets=rvs_datasets,
#             parameters=self.parameters,
#             constraints=rvs_constraints,
#             options=self.options
#         )

#         self.validtoy = True


class OLDExperiment(Superset):
    def __init__(
        self,
        config: dict,
        name: str = "",
    ) -> None:
        self.toy = None  # the last Toy from this experiment
        self.best = None  # to store the best fit result
        self.guess = None  # store the initial guess
        self.minuit = None  # Minuit object

        self.test_statistic = "t_mu"

        # minimization options
        self.backend = "minuit"  # "minuit" or "scipy"
        self.scipy_minimizer = None
        self.use_log = False  # Option in the config to use the logdensity instead of the density in the cost function
        self.iminuit_tolerance = 1e-5  # tolerance for iminuit or other minimizer
        self.iminuit_precision = 1e-10  # precision for the Iminuit minimizer, especially important for scipy minimizers
        self.iminuit_strategy = 0
        self.minimizer_options = {}  # dict of options to pass to the iminuit minimizer
        self.use_grid_rounding = False  # whether to stick parameters on a grid when evaluating the test statistic after minimizing
        self.grid_rounding_num_decimals = (
            {}
        )  # dict telling us how many decimals to round the parameters to when grid rounding

        self.scan = False
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
            if self.backend == "scipy":
                if "scipy_minimizer" in config["options"]:
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
                        None,
                    ]:
                        raise NotImplementedError(
                            f"{self.scipy_minimizer} is not a valid minimizer"
                        )


            if "scan" in config["options"]:
                self.scan = config["options"]["scan"]

                if "scan_grid" in config["options"]:
                    self.scan_grid = config["options"]["scan_grid"]


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


        super().__init__(
            datasets=config["datasets"],
            parameters=config["parameters"],
            constraints=constraints,
            combined_datasets=combined_datasets,
            name=name,
            try_to_combine_datasets=self.try_to_combine_datasets,
            user_gradient=self.user_gradient,
            use_log=self.use_log,
        )

        # get the fit parameters and set the parameter initial values
        self.guess = self.initialguess()
        self.minuit = Minuit(self.costfunction, **self.guess)
        self.minuit.tol = self.iminuit_tolerance  # set the tolerance
        self.minuit.precision = self.iminuit_precision
        self.minuit.strategy = self.iminuit_strategy

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

        # set the grid rounding
        if self.use_grid_rounding:
            for parname in list(config["parameters"].keys()):
                if "grid_rounding_num_decimals" in self.parameters[parname]:
                    self.grid_rounding_num_decimals[parname] = self.parameters[parname][
                        "grid_rounding_num_decimals"
                    ]
                else:
                    self.grid_rounding_num_decimals[parname] = 128  # something big

        # to set limits and fixed variables
        # this function will also fix those fit parameters which can be fixed because they are not part of a
        # Dataset that has data
        self.minuit_reset()

        # If scan is not none, initialize the hypercube grid
        if (self.scan) and (self.scan_grid is not None):
            self.hypercube_grid = self.create_hypercube(self.scan_grid)

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

        if self.scan:
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
                    self.minuit.migrad(**self.minimizer_options)
                elif self.backend == "scipy":
                    self.minuit.scipy(
                        method=self.scipy_minimizer, options=self.minimizer_options
                    )
                else:
                    raise NotImplementedError(
                        "Iminuit backend is not set to `minuit` or `scipy`"
                    )
            except RuntimeError:
                msg = "`Experiment` has invalid best fit"
                logging.debug(msg)

            if not self.minuit.valid:
                msg = "`Experiment` has invalid best fit"
                logging.debug(msg)

            self.best = grab_results(
                self.minuit,
                use_grid_rounding=self.use_grid_rounding,
                grid_rounding_num_decimals=self.grid_rounding_num_decimals,
            )

        if "global_S" in self.best["values"]:
            if self.best["values"]["global_S"] < 1e-20:
                self.best = self.profile({"global_S": 0.0})

        if self.guess == self.best["values"]:
            msg = "`Experiment` has best fit values very close to initial guess"
            logging.debug(msg)

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
                    self.minuit.migrad(**self.minimizer_options)
                elif self.backend == "scipy":
                    self.minuit.scipy(
                        method=self.scipy_minimizer, options=self.minimizer_options
                    )
                else:
                    raise NotImplementedError(
                        "Iminuit backend is not set to `minuit` or `scipy`"
                    )
            except RuntimeError:
                msg = f"`Experiment` throwing NaN has invalid profile at {parameters}"
                logging.debug(msg)

            if not self.minuit.valid:
                msg = "`Experiment` has invalid profile"
                logging.debug(msg)

            results = grab_results(
                self.minuit,
                use_grid_rounding=self.use_grid_rounding,
                grid_rounding_num_decimals=self.grid_rounding_num_decimals,
            )

        if self.guess == results["values"]:
            msg = "`Experiment` has profile fit values very close to initial guess"
            logging.debug(msg)

        # also include the fixed parameters
        for parname, parvalue in parameters.items():
            results["values"][parname] = parvalue

        return results


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
            numerators = np.zeros(num)
            denominators = np.zeros(num)
            data_to_return = []
            paramvalues_to_return = []
            num_drawn = []
            for i in range(num):
                thistoy = self.maketoy(parameters=parameters, seed=seed[i])
                ts[i], denominators[i], numerators[i] = thistoy.ts(
                    profile_parameters=profile_parameters
                )
                data_to_return.append(thistoy.toy_data_to_save)
                paramvalues_to_return.append(thistoy.parameters_to_save)
                num_drawn.append(thistoy.toy_num_drawn_to_save)

        # otherwise profile parameters is a list of dicts
        else:
            ts = np.zeros((len(profile_parameters), num))
            numerators = np.zeros((len(profile_parameters), num))
            denominators = np.zeros((len(profile_parameters), num))
            data_to_return = []
            paramvalues_to_return = []
            num_drawn = []
            for i in range(num):
                thistoy = self.maketoy(parameters=parameters, seed=seed[i])
                for j in range(len(profile_parameters)):
                    ts[j][i], denominators[j][i], numerators[j][i] = thistoy.ts(
                        profile_parameters=profile_parameters[j]
                    )
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

        return (
            ts,
            data_to_return_flat,
            paramvalues_to_return,
            num_drawn_to_return_flat,
            denominators,
            numerators,
        )



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
