"""
A class that holds a combination of `Dataset` and `NormalConstraint`.
"""
import logging

from iminuit import cost

from legendfreqfit.dataset import Dataset

SEED = 42

log = logging.getLogger(__name__)


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

        # check that parameters are actually used in the Datasets and remove them if not
        for parameter in list(self.parameters.keys()):
            is_used = False
            for datasetname in datasets:
                if parameter in self.datasets[datasetname].model_parameters.values():
                    is_used = True
                    break
            if not is_used:
                msg = f"'{parameter}' included as a parameter but not used in a `Dataset` - removing '{parameter}' as a parameter"
                logging.warning(msg)
                del self.parameters[parameter]

        # add in the NormalConstraint        
        if constraints is not None:
            for constraintname, constraint in constraints.items():
                is_used = True
                # check that every parameter in this constraint is used in a Dataset
                for par in constraint["parameters"]:
                    if par not in self.parameters:
                        is_used = False
                        msg = (
                            f"constraint '{constraintname}' includes parameter '{par}', which is not used in any `Dataset` - '{constraintname}' not added as a constraint."
                        )
                        logging.warning(msg)       

                if is_used:
                    self.constraints |= {
                        constraintname: self.add_normalconstraint(
                            parameters=constraint["parameters"],
                            values=constraint["values"],
                            covariance=constraint["covariance"],
                        )
                    }
                    msg = (
                        f"added '{constraintname}' as `NormalConstraint`"
                    )
                    log.debug(msg=msg)
                 

    def add_normalconstraint(
        self,
        parameters: list[str],
        values: list[float],
        covariance,
    ) -> cost.NormalConstraint:
        thiscost = cost.NormalConstraint(parameters, values, covariance)

        self.costfunction = self.costfunction + thiscost

        return thiscost
