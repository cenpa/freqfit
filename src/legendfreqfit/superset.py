"""
A class that holds a combination of datasets.
"""

import warnings
from legendfreqfit.dataset import Dataset

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
        
        # parameters of Superset are different parameters of Dataset
        # maybe they should be the same? Well I changed the name to fitparameters instead
        self.fitparameters = self.costfunction._parameters