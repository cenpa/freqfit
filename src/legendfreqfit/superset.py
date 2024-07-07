"""
A class that holds a combination of `Dataset` and `NormalConstraint`.
"""
import logging

from iminuit import cost
import numpy as np

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
        combine_constraints: bool = False,
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

        # default option
        if combine_constraints == None:
            combine_constraints = False
        
        if combine_constraints:
            msg = f"option 'combine_constraints' set to True - all constraints will be combined into a single `NormalConstraint`"
            logging.debug(msg)
        
        self.name = name
        self.parameters = parameters
        self.datasets = {}
        self.constraints = {}
        self.constraint_parameters = {} # parameter: index
        self.constraint_values = np.array([])
        self.constraint_covariance = None

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
                if constraintname in self.constraints:
                    msg = (
                        f"multiple constraints have name '{constraintname}'"
                    )
                    logging.error(msg)      

                is_used = True
                # check that every parameter in this constraint is used in a Dataset
                for par in constraint["parameters"]:
                    if par not in self.parameters:
                        is_used = False
                        msg = (
                            f"constraint '{constraintname}' includes parameter '{par}', which is not used in any `Dataset` - '{constraintname}' not added as a constraint."
                        )
                        logging.warning(msg) 
                    # more importantly, check that the constraint is on a fit parameter (as opposed to a fixed parameter)
                    # if the constraint is on a fixed parameter, don't add the constraint   
                    elif par not in self.fitparameters:
                        is_used = False
                        msg = (
                            f"constraint '{constraintname}' includes parameter '{par}', which is not a parameter to be fit (probably used a fixed parameter) - '{constraintname}' not added as a constraint."
                        )
                        logging.warning(msg)    

                if is_used:
                    if len(constraint["parameters"]) != len(constraint["values"]):
                        msg = (
                            f"constraint '{constraintname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['values'])} 'values'"
                        )
                        logging.error(msg)   
                    
                    if 'covariance' in constraint and 'uncertainty' in constraint:
                        msg = (
                            f"constraint '{constraintname}' has both 'covariance' and 'uncertainty'; this is ambiguous - use only one!"
                        )
                        logging.error(msg)   
                    
                    if 'covariance' not in constraint and 'uncertainty' not in constraint:
                        msg = (
                            f"constraint '{constraintname}' has neither 'covariance' nor 'uncertainty' - one (and only one) must be provided!"
                        )
                        logging.error(msg)                          
                                            
                    # add the parameters to the set but checks whether they already exist in a constraint
                    for par, value in zip(constraint["parameters"], constraint["values"]):
                        if par not in list(self.constraint_parameters.keys()):
                            self.constraint_parameters[par] = len(self.constraint_parameters)
                            self.constraint_values = np.append(self.constraint_values, value)
                            if self.constraint_covariance is None:
                                self.constraint_covariance = np.identity(1)
                            elif len(self.constraint_covariance) < len(self.constraint_parameters):
                                self.constraint_covariance = np.pad(self.constraint_covariance, ((0,1),(0,1)))
                        else:
                            msg = f"parameter {par} is used in multiple constraints - not currently implemented"
                            raise NotImplementedError(msg)  
                    
                    # now we need to check whether the constraint includes the covariance matrix or the uncertainty
                    if len(constraint["parameters"]) == 1:
                        if "covariance" in constraint:
                            constraint["uncertainty"] = np.sqrt(constraint["covariance"])
                            del constraint["covariance"]
                            msg = (
                                f"constraint '{constraintname}' has one parameter but uses 'covariance' - converting this to 'uncertainty' by taking square root"
                            )
                            logging.warning(msg)                         

                        i = self.constraint_parameters[constraint["parameters"][0]]
                        self.constraint_covariance[i,i] = np.square(constraint["uncertainty"])

                        if combine_constraints:
                            msg = (
                                f"including '{constraintname}' in the combined constraint"
                            )
                            log.debug(msg=msg)                        
                        else:
                            self.constraints |= {
                                constraintname: self._add_normalconstraint(
                                    parameters=constraint["parameters"],
                                    values=constraint["values"],
                                    error=constraint["uncertainty"],
                                )
                            }
                            msg = (
                                f"added '{constraintname}' as `NormalConstraint`"
                            )
                            log.debug(msg=msg)
                    
                    else: # more than 1 parameter in the constraint
                        if "uncertainty" in constraint:
                            if len(constraint["uncertainty"]) == 1:
                                constraint["uncertainty"] = np.full_like(constraint["parameters"], constraint["uncertainty"])  
                                msg = (
                                    f"constraint '{constraintname}' has {len(constraint['parameters'])} parameters but only 1 uncertainty - assuming this is constant uncertainty for each parameter"
                                )
                                logging.warning(msg)    
                            
                            if len(constraint["uncertainty"]) != constraint["parameters"]:
                                msg = (
                                    f"constraint '{constraintname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['uncertainty'])} 'uncertainty' - should be same length or single uncertainty"
                                )
                                logging.error(msg) 

                            for par, uncertainty in zip(constraint["parameters"], constraint["uncertainty"]):
                                i = self.constraint_parameters[par]
                                self.constraint_covariance[i,i] = np.square(uncertainty)                               

                            if combine_constraints:
                                msg = (
                                    f"including '{constraintname}' in the combined constraint"
                                )
                                log.debug(msg=msg)                              
                            else:
                                self.constraints |= {
                                    constraintname: self._add_normalconstraint(
                                        parameters=constraint["parameters"],
                                        values=constraint["values"],
                                        error=constraint["uncertainty"],
                                    )
                                }
                                msg = (
                                    f"added '{constraintname}' as `NormalConstraint`"
                                )
                                log.debug(msg=msg)
                        
                        else: # we have the covariance matrix for this constraint
                            if np.shape(constraint["covariance"]) != (len(constraint["parameters"]), len(constraint["parameters"])):
                                msg = (
                                    f"constraint '{constraintname}' has 'covariance' of shape {np.shape(constraint['covariance'])} but it should be shape {(len(constraint['parameters']), len(constraint['parameters']))}"
                                )
                                logging.error(msg)  

                            if np.allclose(constraint["covariance"], np.asarray(constraint["covariance"]).T):
                                msg = (
                                    f"constraint '{constraintname}' has non-symmetric 'covariance' matrix - this is not allowed."
                                )
                                logging.error(msg)

                            sigmas = np.sqrt(np.diag(np.asarray(constraint["covariance"])))     
                            cov = np.outer(sigmas, sigmas)
                            corr = constraint["covariance"] / cov
                            if not np.all(np.logical_or(np.abs(corr) < 1, np.isclose(corr, 1))):
                                msg = (
                                    f"constraint '{constraintname}' 'covariance' matrix does not seem to contain proper correlation matrix"
                                )
                                logging.error(msg)                                

                            for i in range(len(constraint["parameters"])):
                                for j in range(len(constraint["parameters"])):
                                    self.constraint_covariance[self.constraint_parameters[constraint["parameters"][i]], self.constraint_parameters[constraint["parameters"][j]]] = (
                                        constraint["covariance"][i,j]
                                    )                                            

                            if combine_constraints:
                                msg = (
                                    f"including '{constraintname}' in the combined constraint"
                                )
                                log.debug(msg=msg)                              
                            else:
                                self.constraints |= {
                                    constraintname: self._add_normalconstraint(
                                        parameters=constraint["parameters"],
                                        values=constraint["values"],
                                        error=constraint["covariance"],
                                    )
                                }
                                msg = (
                                    f"added '{constraintname}' as `NormalConstraint`"
                                )
                                log.debug(msg=msg)
            
            # we've looped through all the individual constraints - now to combine them if desired
            if combine_constraints:
                self.constraints["combined_constraints"] = self._add_normalconstraint(
                                        parameters=list(self.constraint_parameters.keys()),
                                        values=self.constraint_values,
                                        error=self.constraint_covariance,
                                    )
                msg = (
                    f"added 'combined_constraints' as `NormalConstraint`"
                )
                log.debug(msg=msg)
                    
    def _add_normalconstraint(
        self,
        parameters: list[str],
        values: list[float],
        error,
    ) -> cost.NormalConstraint:
        thiscost = cost.NormalConstraint(parameters, values, error=error)

        self.costfunction = self.costfunction + thiscost

        return thiscost
