"""
A class that holds constraints, which can represent auxialiary measurements.
"""
import logging

import numpy as np
from iminuit import cost

SEED = 42

log = logging.getLogger(__name__)

class Constraints:

    def __init__(
        self,
        constraints: dict,
    ) -> None:

        self.constraints_parameters = {}

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


        return None

    def get_cost(
        self,
        parameters: list,
    ) -> None:

        pars, values, covariance = self.get_constraints(parameters)

        return cost.NormalConstraint(pars, values, error=covariance)

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

