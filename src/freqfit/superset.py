"""
A class that holds a combination of `Dataset` and `NormalConstraint`.
"""
import logging
from copy import deepcopy

import numpy as np
from iminuit import cost

from freqfit.dataset import Dataset, combine_datasets

SEED = 42

log = logging.getLogger(__name__)


class Superset:
    def __init__(
        self,
        datasets: dict,
        parameters: dict,
        constraints: dict = None,
        combined_datasets: dict = None,
        name: str = "",
        try_to_combine_datasets: bool = False,
        user_gradient: bool = False,
        use_log: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        datasets
            `dict`
        parameters
            `dict`
        constraints
            `dict`
        user_gradient
            If true, use the user's gradient in dataset costfunction
        use_log
            Use the logpdf in the cost function if true
        """

        self.name = name  # name of the Superset

        # parameter dictionary that contains all parameters used in any Datasets and combined_datasets. Not necessarily fit parameters.
        self.parameters = parameters

        self.datasets = {}  # dataset name: Dataset
        self.combined_datasets = {}  # combined_dataset name : Dataset
        self._combined_datasets_config = (
            combined_datasets  # config for combined_datasets for use in Toy
        )
        self.included_in_combined_datasets = (
            {}
        )  # combined_dataset name : list of dataset names contained

        self.costfunction = None  # iminuit costfunction object that will contain the Dataset costfunctions and NormalConstraint
        self.fitparameters = (
            None  # reference to self.costfunction._parameters, parameters of the fit
        )

        self.constraints = None  # the constraint costfunction
        self.constraints_parameters = {}  # parameter : index in values and covariance
        self.constraints_values = None
        self.constraints_covariance = None

        self._toy_pars_to_vary = []  # list of parameters to vary
        self._toy_parameters = (
            {}
        )  # a copy of self.parameters that we can mutate without care

        if try_to_combine_datasets:
            msg = "option 'try_to_combine_datasets' set to True - will attempt to combine `Dataset` where indicated"
            logging.info(msg)

            if combined_datasets is None or len(combined_datasets) == 0:
                msg = "option 'try_to_combine_datasets' set to True but `combined_datasets` is missing or empty!"
                logging.error(msg)

        alldspars = set()
        # create the Datasets
        for dsname in datasets.keys():
            try_combine = False
            combined_dataset = None
            if try_to_combine_datasets:
                try_combine = (
                    datasets[dsname]["try_to_combine"]
                    if "try_to_combine" in datasets[dsname]
                    else False
                )
                combined_dataset = (
                    datasets[dsname]["combined_dataset"]
                    if "combined_dataset" in datasets[dsname]
                    else None
                )
                # error checking in Dataset

            self.datasets[dsname] = Dataset(
                data=datasets[dsname]["data"],
                model=datasets[dsname]["model"],
                model_parameters=datasets[dsname]["model_parameters"],
                parameters=parameters,
                costfunction=datasets[dsname]["costfunction"],
                name=dsname,
                try_combine=try_combine,
                combined_dataset=combined_dataset,
                user_gradient=user_gradient,
                use_log=use_log,
            )

            for par in datasets[dsname]["model_parameters"].values():
                alldspars.add(par)

        # here is where we should try to combine datasets (or maybe just above?)
        # I *think* we want to keep the datasets as is and maybe add a new holder for
        # the combined datasets and add only the relevant costfunctions together. going to have to look
        # at how constraints are added and covariance matrix made.

        if try_to_combine_datasets:
            # maybe there's more than one combined_dataset group
            for cdsname in combined_datasets:
                # find the Datasets to try to combine
                ds_tocombine = []
                for dsname in self.datasets.keys():
                    if (
                        self.datasets[dsname].try_combine
                        and self.datasets[dsname].combined_dataset == cdsname
                    ):
                        ds_tocombine.append(self.datasets[dsname])

                combined_dataset, included_datasets = combine_datasets(
                    datasets=ds_tocombine,
                    model=combined_datasets[cdsname]["model"],
                    model_parameters=combined_datasets[cdsname]["model_parameters"],
                    parameters=self.parameters,
                    costfunction=combined_datasets[cdsname]["costfunction"],
                    name=cdsname,
                    user_gradient=user_gradient,
                    use_log=use_log,
                )

                if len(included_datasets) > 0:
                    self.combined_datasets[cdsname] = combined_dataset
                    self.included_in_combined_datasets[cdsname] = included_datasets

                for par in combined_datasets[cdsname]["model_parameters"].values():
                    alldspars.add(par)

        # add the costfunctions together
        first = True
        for dsname in self.datasets.keys():
            if self.datasets[dsname].is_combined:
                continue
            if first:
                self.costfunction = self.datasets[dsname].costfunction
                first = False
            else:
                self.costfunction += self.datasets[dsname].costfunction

        for cdsname in self.combined_datasets.keys():
            if first:
                self.costfunction = self.combined_datasets[cdsname].costfunction
                first = False
            else:
                self.costfunction += self.combined_datasets[cdsname].costfunction

        # fitparameters of Superset are a little different than fitparameters of Dataset
        self.fitparameters = self.costfunction._parameters

        # check that parameters are actually used in the Datasets or combined_datasets and remove them if not
        for parameter in list(self.parameters.keys()):
            if parameter not in alldspars:
                msg = f"'{parameter}' included as a parameter but not used in a `Dataset` - removing '{parameter}' as a parameter"
                logging.warning(msg)
                del self.parameters[parameter]

        # collect which parameters are included as parameters to vary for the toys
        for dsname, ds in self.datasets.items():
            for parname in ds._toy_pars_to_vary:
                if parname not in self._toy_pars_to_vary:
                    self._toy_pars_to_vary.append(parname)
                    msg = f"`Superset` '{self.name}' added parameter '{parname}' as a parameter to vary for toys"
                    logging.info(msg)

        if len(self._toy_pars_to_vary) > 0 and constraints is None:
            msg = "have parameters to vary but no constraints found!"
            logging.error(msg)
            raise ValueError(msg)

        if constraints is None:
            msg = "no constraints were provided"
            logging.info(msg)

        if constraints is not None:
            msg = "all constraints will be combined into a single `NormalConstraint`"
            logging.info(msg)

            # shove all the constraints in one big matrix
            for constraintname, constraint in constraints.items():
                # would love to move this somewhere else, maybe sanitize the config before doing anything
                if len(constraint["parameters"]) != len(constraint["values"]):
                    if len(constraint["values"]) == 1:
                        constraint["values"] = np.full(
                            len(constraint["parameters"]), constraint["values"]
                        )
                        msg = f"in constraint '{constraintname}', assigning 1 provided value to all {len(constraint['parameters'])} 'parameters'"
                        logging.warning(msg)
                    else:
                        msg = f"constraint '{constraintname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['values'])} 'values'"
                        logging.error(msg)
                        raise ValueError(msg)

                if "covariance" in constraint and "uncertainty" in constraint:
                    msg = f"constraint '{constraintname}' has both 'covariance' and 'uncertainty'; this is ambiguous - use only one!"
                    logging.error(msg)
                    raise KeyError(msg)

                if "covariance" not in constraint and "uncertainty" not in constraint:
                    msg = f"constraint '{constraintname}' has neither 'covariance' nor 'uncertainty' - one (and only one) must be provided!"
                    logging.error(msg)
                    raise KeyError(msg)

                # do some cleaning up of the config here
                if "uncertainty" in constraint:
                    if len(constraint["uncertainty"]) > 1:
                        constraint["uncertainty"] = np.full(
                            len(constraint["parameters"]), constraint["uncertainty"]
                        )
                        msg = f"constraint '{constraintname}' has {len(constraint['parameters'])} parameters but only 1 uncertainty - assuming this is constant uncertainty for each parameter"
                        logging.warning(msg)

                    if len(constraint["uncertainty"]) != len(constraint["parameters"]):
                        msg = f"constraint '{constraintname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['uncertainty'])} 'uncertainty' - should be same length or single uncertainty"
                        logging.error(msg)
                        raise ValueError(msg)

                    # convert to covariance matrix so that we're always working with the same type of object
                    constraint["covariance"] = np.diag(constraint["uncertainty"]) ** 2
                    del constraint["uncertainty"]

                    msg = f"constraint '{constraintname}': converting provided 'uncertainty' to 'covariance'"
                    logging.info(msg)

                else:  # we have the covariance matrix for this constraint
                    if len(constraint["parameters"]) == 1:
                        msg = f"constraint '{constraintname}' has one parameter but uses 'covariance' - taking this at face value"
                        logging.info(msg)

                    if np.shape(constraint["covariance"]) != (
                        len(constraint["parameters"]),
                        len(constraint["parameters"]),
                    ):
                        msg = f"constraint '{constraintname}' has 'covariance' of shape {np.shape(constraint['covariance'])} but it should be shape {(len(constraint['parameters']), len(constraint['parameters']))}"
                        logging.error(msg)
                        raise ValueError(msg)

                    if not np.allclose(
                        constraint["covariance"], np.asarray(constraint["covariance"]).T
                    ):
                        msg = f"constraint '{constraintname}' has non-symmetric 'covariance' matrix - this is not allowed."
                        logging.error(msg)
                        raise ValueError(msg)

                    sigmas = np.sqrt(np.diag(np.asarray(constraint["covariance"])))
                    cov = np.outer(sigmas, sigmas)
                    corr = constraint["covariance"] / cov
                    if not np.all(np.logical_or(np.abs(corr) < 1, np.isclose(corr, 1))):
                        msg = f"constraint '{constraintname}' 'covariance' matrix does not seem to contain proper correlation matrix"
                        logging.error(msg)
                        raise ValueError(msg)

                for par in constraint["parameters"]:
                    if par in self.constraints_parameters:
                        msg = f"parameter {par} is used in multiple constraints - not allowed"
                        logging.error(msg)
                        raise KeyError(msg)

                    self.constraints_parameters[par] = len(self.constraints_parameters)

            # initialize now that we know how large to make them
            self.constraints_values = np.full(len(self.constraints_parameters), np.nan)
            self.constraints_covariance = np.identity(len(self.constraints_parameters))

            for constraintname, constraint in constraints.items():
                # now put the values in
                for par, value in zip(constraint["parameters"], constraint["values"]):
                    self.constraints_values[self.constraints_parameters[par]] = value

                # now put the covariance matrix in
                for i in range(len(constraint["parameters"])):
                    for j in range(len(constraint["parameters"])):
                        self.constraints_covariance[
                            self.constraints_parameters[constraint["parameters"][i]],
                            self.constraints_parameters[constraint["parameters"][j]],
                        ] = constraint["covariance"][i, j]

            self._add_constraints_to_costfunction()  # stored in self.constraints

        # so we can mutate self._toy_parameters later without it being a problem?
        self._toy_parameters = deepcopy(self.parameters)

        # that's all, folks!

    def _add_constraints_to_costfunction(
        self,
    ) -> None:
        if self.constraints is not None:
            msg = "already added constraints to costfunction! not adding again"
            logging.warning(msg)
            return self.constraints

        pars, values, covariance = self.get_constraints(self.fitparameters)

        self.constraints = cost.NormalConstraint(pars, values, error=covariance)

        self.costfunction = self.costfunction + self.constraints

        return

    # gets the appropriate values and covariance submatrix for the requested parameters, if they exist
    # returns a tuple that contains a list of parameters found constraints for, their values, covariance matrix
    def get_constraints(
        self,
        parameters: list,
    ) -> tuple:
        if len(self.constraints_parameters) == 0:
            return (None, None, None)

        pars = []
        inds = []
        for par in parameters:
            if par in self.constraints_parameters:
                pars.append(par)
                inds.append(self.constraints_parameters[par])

        values = self.constraints_values[inds]
        covar = self.constraints_covariance[np.ix_(inds, inds)]

        return (pars, values, covar)
