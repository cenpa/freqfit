"""
A class that holds constraints, which can represent auxialiary measurements.
"""
import logging

import numpy as np
from iminuit import cost

SEED = 42

log = logging.getLogger(__name__)

# this class is going to need to re-worked at some point. Not really capable of
# handling multiple constraints on the same parameter right now.

class Constraints:

    def __init__(
        self,
        constraints: dict,
    ) -> None:

        self._constraint_groups = {
            "constraint_group_0":{
                "constraint_names":[],
                "parameters":[], 
                "values":None,
                "covariance":None}}

        # make groups of constraints based on what parameters are in them - to reduce overall # of constraints
        for ctname, ct in constraints.items():
            for grpname, grp in self._constraint_groups.items():
                noparsin = True
                for par in ct["parameters"]:
                    if par in grp["parameters"]:
                        noparsin = False
                
                if noparsin:
                    grp["constraint_names"].append(ctname)
                    for par in ct["parameters"]:
                        grp["parameters"].append(par)

        # now combine the constraints in each group into a single constraint
        for grpname, grp in self._constraint_groups.items():
            grp["values"] = np.full(len(grp["parameters"], np.nan))
            grp["covariance"] = np.identity(len(grp["parameters"]))


        self.constraints_parameters = {}
        self.constraints_parameters[par] = len(self.constraints_parameters)

        # initialize now that we know how large to make them
        self.constraints_values = np.full(len(self.constraints_parameters), np.nan)
        self.constraints_covariance = np.identity(len(self.constraints_parameters))
        self.pars_to_vary = []
        for constraintname, constraint in constraints.items():
            # now put the values in
            for par, value in zip(constraint["parameters"], constraint["values"]):
                self.constraints_values[self.constraints_parameters[par]] = value
                
                if constraint["vary"]:
                    self.pars_to_vary.append(par)

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

    def rvs(
        self,
        parameters:dict,
    ) -> dict:
        """
        varies the provided parameters which are expected to be varied
        """

        pars, values, covar = self.get_constraints(self.pars_to_vary)

        # set central values to the provided ones
        for i, par in enumerate(pars):
            if par in parameters:
                values[i] = parameters[par]

        rvs = []
        # check if parameters are all independent, draw from simpler distribution if so
        if np.all(covar == np.diag(np.diagonal(covar))):
            rvs = np.random.normal(values, np.sqrt(np.diagonal(covar)))
        else:
            rvs = np.random.multivariate_normal(
                values, covar
            )  # sooooooooo SLOW        
        

