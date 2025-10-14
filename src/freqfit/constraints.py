"""
A class that holds constraints, which can represent auxiliary measurements.
"""
import logging
from copy import deepcopy

import numpy as np
from iminuit import cost

from .parameters import Parameters

log = logging.getLogger(__name__)


class Constraints:
    def __init__(
        self,
        constraints: dict,
        parameters: Parameters,
    ) -> None:
        self.parameters = parameters  # Parameters object
        self._constraint_groups = {}

        # make groups of constraints based on what parameters are in them - to reduce overall # of constraints
        for ctname, ct in constraints.items():
            foundplace = False
            for grpname, grp in self._constraint_groups.items():
                # has to match whether the constraint should be varied
                if ct["vary"] != grp["vary"]:
                    continue

                noparsin = True
                for par in ct["parameters"]:
                    if par in grp["parameters"]:
                        noparsin = False

                if noparsin:
                    foundplace = True
                    grp["constraint_names"].append(ctname)
                    for par in ct["parameters"]:
                        grp["parameters"][par] = len(grp["parameters"])

                    break

            if not foundplace:
                newgrpname = f"constraint_group_{len(self._constraint_groups)}"
                newgrp = {
                    "constraint_names": [],
                    "parameters": {},  # parameter name : index in values
                    "values": None,
                    "covariance": None,
                }

                # match this first constraint
                newgrp["vary"] = ct["vary"]

                newgrp["constraint_names"].append(ctname)
                for par in ct["parameters"]:
                    newgrp["parameters"][par] = len(newgrp["parameters"])

                self._constraint_groups[newgrpname] = newgrp

        # now combine the constraints in each group into a single constraint
        # initialize matrices now that we know how large to make them and put the values in
        for grpname, grp in self._constraint_groups.items():
            grp["values"] = np.full(len(grp["parameters"]), np.nan)
            grp["covariance"] = np.identity(len(grp["parameters"]))

            i = 0
            for ctname in grp["constraint_names"]:
                l = 0  # noqa: E741
                for j in range(len(constraints[ctname]["parameters"])):
                    l += 1
                    grp["values"][i + j] = constraints[ctname]["values"][j]

                    for k in range(len(constraints[ctname]["parameters"])):
                        grp["covariance"][i + j, i + k] = constraints[ctname][
                            "covariance"
                        ][j, k]

                i += l

    def get_cost(
        self,
        parameters: list,
    ) -> type[cost.Cost]:
        constraint_groups = self.get_constraints(parameters)

        # add the costfunctions together
        toreturn = None
        first = True
        for grpname, grp in constraint_groups.items():
            if first:
                toreturn = cost.NormalConstraint(
                    list(grp["parameters"].keys()),
                    grp["values"],
                    error=grp["covariance"],
                )

                first = False
            else:
                toreturn += cost.NormalConstraint(
                    list(grp["parameters"].keys()),
                    grp["values"],
                    error=grp["covariance"],
                )

        return toreturn

    # gets the appropriate values and covariance submatrix for the requested parameters, if they exist
    # returns a tuple that contains a list of parameters found constraints for, their values, covariance matrix
    def get_constraints(
        self,
        parameters: list,
    ) -> dict:
        toreturn = {}

        if len(self._constraint_groups) == 0:
            return toreturn

        for grpname, grp in self._constraint_groups.items():
            foundsomepars = False
            for par in parameters:
                if par in grp["parameters"]:
                    if grpname not in toreturn:
                        toreturn[grpname] = {
                            "parameters": {},
                        }

                    toreturn[grpname]["parameters"][par] = grp["parameters"][par]
                    foundsomepars = True

            if foundsomepars:
                inds = list(toreturn[grpname]["parameters"].values())
                toreturn[grpname]["values"] = grp["values"][inds]
                toreturn[grpname]["covariance"] = grp["covariance"][np.ix_(inds, inds)]
                toreturn[grpname]["vary"] = grp["vary"]

        return toreturn


class ToyConstraints(Constraints):
    def __init__(
        self,
        constraints: dict,
        parameters: Parameters,
    ) -> None:
        super().__init__(constraints, parameters)

        # check if parameters are all independent, draw from simpler distribution if so
        for grpname, grp in self._constraint_groups.items():
            grp["all_independent"] = False
            if np.all(grp["covariance"] == np.diag(np.diagonal(grp["covariance"]))):
                grp["all_independent"] = True

        self._base_constraint_groups = deepcopy(self._constraint_groups)

    def rvs(
        self,
        parameters: dict = {},  # noqa: B006
    ) -> None:
        """
        Using the provided central values of the parameters,
        varies the provided parameters which are expected to be varied. Then `get_cost` can be called to get an updated cost function

        Parameters
        ----------
        parameters
            dict of parameter names and their new central values from which a random variate will be drawn.
            Additional parameters will be ignored. If a parameter is not provided, the default central value will
            be used.
        """

        # TODO: extremely slow! (added lines at **1**)
        # self._constraint_groups = deepcopy(self._base_constraint_groups)

        for grpname, grp in self._constraint_groups.items():
            if grp["vary"]:
                for i, par in enumerate(grp["parameters"].keys()):
                    # set central values to the provided ones, keep the rest as is
                    if par in parameters:
                        grp["values"][i] = parameters[par]
                    # added lines **1**
                    else:
                        grp["values"][i] = self._base_constraint_groups[grpname][
                            "values"
                        ][i]

                gooddraws = False
                while not gooddraws:
                    # check if parameters are all independent, draw from simpler distribution if so
                    if grp["all_independent"]:
                        grp["values"] = np.random.normal(
                            grp["values"], np.sqrt(np.diagonal(grp["covariance"]))
                        )
                    else:
                        grp["values"] = np.random.multivariate_normal(
                            grp["values"], grp["covariance"]
                        )  # sooooooooo SLOW

                    gooddraws = True
                    # check whether drawn values are within the supported domain
                    for i, par in enumerate(grp["parameters"].keys()):
                        lims = self.parameters(par)["domain"]

                        if not (
                            (grp["values"][i] >= lims[0])
                            and (grp["values"][i] <= lims[1])
                        ):
                            gooddraws = False
                            break

        return
