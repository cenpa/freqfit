"""
A class that holds a collection of fake datasets
"""

import warnings
import numpy as np
from iminuit import cost

SEED = 42

class Toy:
    def __init__(
            self,
            superset,
            parameters: dict,
            seed: int = SEED,
        ) -> None:
        """
        superset
            `Superset` to base this `Toy` on
        parameters
            `dict` of parameters and their values to model with
        """

        self.superset = superset

        self.costfunction = None
        for i, (datasetname, dataset) in enumerate(superset.datasets.items()):
            # worried that this is not totally deterministic (depends on number of Datasets),
            # but more worried that random draws between datasets would be correlated otherwise.
            thisseed = seed + i 

            # allocate length in case `parameters` is passed out of order
            par = [None for j in range(len(dataset.fitparameters))]

            for j, (fitpar, fitparindex) in enumerate(dataset.fitparameters.items()):
                if fitpar not in parameters:
                    msg = (
                        f"`Toy`: for `Dataset` {datasetname}, parameter `{fitpar}` not found in passed `parameters`"
                    )
                    raise KeyError(msg)
                
                par[j] = parameters[fitpar]

            # make the fake data for this particular dataset
            toydata = dataset.rvs(*par, seed=thisseed)

            # make the cost function for this particular dataset
            thiscostfunction = dataset._costfunctioncall(toydata, dataset.density)

            # tell the cost function which parameters to use
            thiscostfunction._parameters = dataset.costfunction._parameters
            
            if (i==0):
                self.costfunction = thiscostfunction
            else: 
                self.costfunction += thiscostfunction

       # fitparameters of Toy are a little different than fitparameters of Dataset
        self.fitparameters = self.costfunction._parameters

        for constraintname, constraint in superset.constraints.items():
            self.costfunction += constraint
        
        return
 
    def ll(
        self,
        parameters: dict,
        ) -> float:

        ll = 0.0

        for i, (datasetname, dataset) in enumerate(self.datasets.items()):
            # these are the parameters and their order needed to call the dataset's model functions
            fitparameters = dataset.fitparameters.items()

            # allocate length in case `parameters` is passed out of order
            par = [None for j in range(len(fitparameters))]

            for j, (fitpar, fitparindex) in enumerate(fitparameters):
                if fitpar not in parameters:
                    msg = (
                        f"parameter `{fitpar}` not found in passed `parameters`"
                    )
                    raise KeyError(msg)
                
                par[j] = parameters[fitpar]

            ll += dataset.toyll(*par)
        
        return ll