"""
A class that holds a combination of datasets.
"""

import warnings
from legendfreqfit.dataset import Dataset

SEED = 42

class Superset:
    def __init__(
        self,
        datasets: dict,
        name: str = None,
        ) -> None:
        """
        Parameters
        ----------
        datasets
            `dict`
        """

        self.name = name
        self.datasets = {}

        # create the Datasets
        for datasetname in datasets:
            self.datasets[datasetname] = Dataset(
                data=datasets[datasetname]["data"],
                model=datasets[datasetname]["model"],
                parameters=datasets[datasetname]["parameters"],
                costfunction=datasets[datasetname]["costfunction"],
                name=datasetname)
    
        # add the costfunctions together
        self.costfunction = None
        for i, datasetname in enumerate(self.datasets):
            if (i==0):
                self.costfunction = self.datasets[datasetname].costfunction
            else:
                self.costfunction += self.datasets[datasetname].costfunction
        
        # fitparameters of Superset are a little different than fitparameters of Dataset
        self.fitparameters = self.costfunction._parameters
    
    
    def maketoy(
        self,
        parameters: dict,
        seed:int = SEED,
        ) -> None:

        for i, (datasetname, dataset) in enumerate(self.datasets.items()):
            # worried that this is not totally deterministic (depends on number of Datasets),
            # but more worried that random draws between datasets would be correlated otherwise.
            thisseed = seed + i 

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

            dataset.maketoy(*par, seed=thisseed)
        
        return None
    
    def toyll(
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
                


