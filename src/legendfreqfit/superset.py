"""
A class that holds a combination of datasets and NormalConstraints.
"""


from iminuit import cost

from legendfreqfit.dataset import Dataset

SEED = 42


class Superset:
    def __init__(
        self,
        datasets: dict,
        parameters: dict,
        constraints: dict = None,
        name: str = None,
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
        """

        self.name = name
        self.parameters = parameters
        self.datasets = {}
        self.constraints = {}
        self.toy = None

        # create the Datasets
        for datasetname in datasets:
            self.datasets[datasetname] = Dataset(
                data=datasets[datasetname]["data"],
                model=datasets[datasetname]["model"],
                model_parameters=datasets[datasetname]["model_parameters"],
                parameters=parameters,
                costfunction=datasets[datasetname]["costfunction"],
                name=datasetname,
            )

        # add the costfunctions together
        self.costfunction = None
        for i, datasetname in enumerate(self.datasets):
            if i == 0:
                self.costfunction = self.datasets[datasetname].costfunction
            else:
                self.costfunction += self.datasets[datasetname].costfunction

        # fitparameters of Superset are a little different than fitparameters of Dataset
        self.fitparameters = self.costfunction._parameters

        if constraints is not None:
            for constraintname, constraint in constraints.items():
                self.constraints |= {
                    constraintname: self.add_normalconstraint(
                        parameters=constraint["parameters"],
                        values=constraint["values"],
                        covariance=constraint["covariance"],
                    )
                }

    def add_normalconstraint(
        self,
        parameters: list[str],
        values: list[float],
        covariance,
    ) -> cost.NormalConstraint:
        thiscost = cost.NormalConstraint(parameters, values, covariance)

        self.costfunction = self.costfunction + thiscost

        return thiscost
