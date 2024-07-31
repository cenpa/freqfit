"""
A class that holds a combination of `Dataset` and `NormalConstraint`.
"""
import logging

from iminuit import cost
import numpy as np

from legendfreqfit.dataset import Dataset, combine_datasets

SEED = 42

log = logging.getLogger(__name__)


class Superset:
    def __init__(
        self,
        datasets: dict,
        parameters: dict,
        constraints: dict = None,
        combined_datasets: dict = None,
        name: str = None,
        try_to_combine_datasets: bool = False,
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

        self.name = name # name of the Superset

        # parameter dictionary that contains all parameters used in any Datasets and combined_datasets. Not necessarily fit parameters.
        self.parameters = parameters 

        self.datasets = {} # dataset name: Dataset
        self.combined_datasets = {} # combined_dataset name : Dataset
        self.included_in_combined_datasets = {} # combined_dataset name : list of dataset names contained

        self.costfunction = None # iminuit costfunction object that will contain the Dataset costfunctions and NormalConstraint
        self.fitparameters = None # reference to self.costfunction._parameters, parameters of the fit

        self.constraints = {}
        self.constraint_parameters = {} # parameter : index in constraint_values and constraint_covariance
        self.constraint_values = np.array([]) # central values of constraints
        self.constraint_covariance = None # covariance matrix of constraints

        self.toypars_to_vary = {} # parameter : index in toypars_to_vary_values and toypars_to_vary_covariance
        self.toypars_to_vary_values = np.array([]) # central values of toypars_to_vary parameters
        self.toypars_to_vary_covariance = None # covariance matrix of toypars_to_vary parameters

        msg = f"all constraints will be combined into a single `NormalConstraint`"
        logging.info(msg)

        if try_to_combine_datasets:
            msg = f"option 'try_to_combine_datasets' set to True - will attempt to combine `Dataset` where indicated"
            logging.info(msg)       

            if combined_datasets is None or len(combined_datasets)==0:
                msg = f"option 'try_to_combine_datasets' set to True but `combined_datasets` is missing or empty!"
                logging.error(msg)
        
        # create the Datasets
        for dsname in datasets.keys():
            try_combine = False
            combined_dataset = None
            if try_to_combine_datasets:
                try_combine = datasets[dsname]["try_to_combine"] if "try_to_combine" in datasets[dsname] else False
                combined_dataset = datasets[dsname]["combined_dataset"] if "combined_dataset" in datasets[dsname] else None
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
            )

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
                    if self.datasets[dsname].try_combine and self.datasets[dsname].combined_dataset == cdsname:
                        ds_tocombine.append(self.datasets[dsname])

                combined_dataset, included_datasets = combine_datasets(
                    datasets=ds_tocombine, 
                    model=combined_datasets[cdsname]["model"],
                    model_parameters=combined_datasets[cdsname]["model_parameters"],
                    parameters=self.parameters,
                    costfunction=combined_datasets[cdsname]["costfunction"],
                    name=combined_datasets[cdsname]["name"] if "name" in combined_datasets[cdsname] else "",
                )

                # now to see what datasets actually got included in the combined_dataset
                for dsname in included_datasets:
                    self.datasets[dsname].is_combined = True

                if len(included_datasets) > 0:
                    self.combined_datasets[cdsname] = combined_dataset
                    self.included_in_combined_datasets[cdsname] = included_datasets

        # add the costfunctions together
        first = True
        for dsname in self.datasets.keys():
            if not self.datasets[dsname].is_combined:
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
            used = False

            # probably fewer of these so they go first
            for cdsname, cds in self.combined_datasets.items():
                if parameter in cds.model_parameters.values():
                    used = True
                    break
                
            if not used:
                for dsname in datasets:
                    if parameter in self.datasets[dsname].model_parameters.values():
                        used = True
                        break

            if not used:
                msg = f"'{parameter}' included as a parameter but not used in a `Dataset` - removing '{parameter}' as a parameter"
                logging.warning(msg)
                del self.parameters[parameter]
        
        # collect which parameters are included as parameters to vary for the toys
        for parname, pardict in self.parameters.items():
            if "vary_by_constraint" in pardict and pardict["vary_by_constraint"]:
                self.toypars_to_vary[parname] = len(self.toypars_to_vary) 
                msg = f"added parameter '{parname}' as a parameter to vary for toys"
                logging.info(msg)
        
        # if no constraints and nothing needs constraints, we're done
        if len(self.toypars_to_vary) == 0 and constraints is None:
            return
        
        if len(self.toypars_to_vary) > 0 and constraints is None:
            msg = f"have parameters to vary but no constraints found!"
            logging.error(msg)
            raise ValueError(msg)

        # we want to run through the constraints to 1) get the constraints for the fit paramterts and 
        # 2) get the values and covariance for the parameters to vary for the toys

        # let's get lists of the constraints for (1) and (2)
        # and also track which parameters are in the constraints to look for duplicates
        constraints_for_constraints = []
        constraints_for_toypars_to_vary = []
        pars_in_constraints = set()
        for constraintname, constraint in constraints.items():  
            
            # would love to move this somewhere else, maybe sanitize the config before doing anything
            if len(constraint["parameters"]) != len(constraint["values"]):
                if len(constraint["values"]) == 1:
                    constraint["values"] = np.full(len(constraint["parameters"]), constraint["values"])
                    msg = (
                        f"in constraint '{constraintname}', assigning 1 provided value to all {len(constraint['parameters'])} 'parameters'"
                    )
                    logging.warning(msg)                    
                else:
                    msg = (
                        f"constraint '{constraintname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['values'])} 'values'"
                    )
                    logging.error(msg) 
                    raise ValueError(msg)  
            
            if 'covariance' in constraint and 'uncertainty' in constraint:
                msg = (
                    f"constraint '{constraintname}' has both 'covariance' and 'uncertainty'; this is ambiguous - use only one!"
                )
                logging.error(msg)  
                raise KeyError(msg)  
            
            if 'covariance' not in constraint and 'uncertainty' not in constraint:
                msg = (
                    f"constraint '{constraintname}' has neither 'covariance' nor 'uncertainty' - one (and only one) must be provided!"
                )
                logging.error(msg)  
                raise KeyError(msg) 

            # do some cleaning up of the config here
            if len(constraint["parameters"]) == 1:
                if "covariance" in constraint:
                    constraint["uncertainty"] = np.sqrt(constraint["covariance"])
                    del constraint["covariance"]
                    msg = (
                        f"constraint '{constraintname}' has one parameter but uses 'covariance' - converting this to 'uncertainty' by taking square root"
                    )
                    logging.warning(msg)  
            else:
                if "uncertainty" in constraint:
                    if len(constraint["uncertainty"]) == 1:
                        constraint["uncertainty"] = np.full(len(constraint["parameters"]), constraint["uncertainty"])
                        msg = (
                            f"constraint '{constraintname}' has {len(constraint['parameters'])} parameters but only 1 uncertainty - assuming this is constant uncertainty for each parameter"
                        )
                        logging.warning(msg)    
                    
                    if len(constraint["uncertainty"]) != len(constraint["parameters"]):
                        msg = (
                            f"constraint '{constraintname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['uncertainty'])} 'uncertainty' - should be same length or single uncertainty"
                        )
                        logging.error(msg) 
                
                else: # we have the covariance matrix for this constraint
                    if np.shape(constraint["covariance"]) != (len(constraint["parameters"]), len(constraint["parameters"])):
                        msg = (
                            f"constraint '{constraintname}' has 'covariance' of shape {np.shape(constraint['covariance'])} but it should be shape {(len(constraint['parameters']), len(constraint['parameters']))}"
                        )
                        logging.error(msg)  

                    if not np.allclose(constraint["covariance"], np.asarray(constraint["covariance"]).T):
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

            all_used_for_constraints = True
            all_used_for_toypars_to_vary = True
            # check that every parameter in this constraint is used in a Dataset
            for par in constraint["parameters"]:
                if par not in pars_in_constraints:
                    pars_in_constraints.add(par)
                else:
                    msg = f"parameter {par} is used in multiple constraints - not allowed"
                    logging.error(msg)

                inpars = True
                if par not in self.parameters:
                    inpars = False
                    all_used_for_constraints = False
                    all_used_for_toypars_to_vary = False
                    msg = (
                        f"constraint '{constraintname}' includes parameter '{par}', which is not used in any `Dataset` - constraint '{constraintname}' not included."
                    )
                    logging.warning(msg) 

                # constraints added to the costfunction have to be fit parameters 
                if inpars and par not in self.fitparameters:
                    all_used_for_constraints = False
                    msg = (
                        f"constraint '{constraintname}' includes parameter '{par}', which is not a parameter to be fit"
                        + f"(probably used a fixed parameter or part of a combined_dataset) - '{constraintname}' not added as a constraint."
                    )
                    logging.warning(msg)    
                
                if inpars and par not in self.toypars_to_vary:
                    all_used_for_toypars_to_vary = False
            
            if all_used_for_constraints:
                constraints_for_constraints.append(constraintname)

            if all_used_for_toypars_to_vary:
                constraints_for_toypars_to_vary.append(constraintname)
        
        for constraintname in constraints_for_constraints:
            constraint = constraints[constraintname]
                                    
            # add the parameters to the set
            for par, value in zip(constraint["parameters"], constraint["values"]):
                self.constraint_parameters[par] = len(self.constraint_parameters)
                self.constraint_values = np.append(self.constraint_values, value)
                
                if self.constraint_covariance is None:
                    self.constraint_covariance = np.identity(1)
                elif len(self.constraint_covariance) < len(self.constraint_parameters):
                    self.constraint_covariance = np.pad(self.constraint_covariance, ((0,1),(0,1)))

            # now we need to check whether the constraint includes the covariance matrix or the uncertainty
            if "uncertainty" in constraint:
                for par, uncertainty in zip(constraint["parameters"], constraint["uncertainty"]):
                    i = self.constraint_parameters[par]
                    self.constraint_covariance[i,i] = np.square(uncertainty)                                                           
                
            else: # we have the covariance matrix for this constraint                            
                for i in range(len(constraint["parameters"])):
                    for j in range(len(constraint["parameters"])):
                        self.constraint_covariance[self.constraint_parameters[constraint["parameters"][i]], self.constraint_parameters[constraint["parameters"][j]]] = (
                            constraint["covariance"][i,j]
                        )                                              
                
            msg = f"including '{constraintname}' in the combined constraint"
            log.info(msg=msg)                               
    
        # we've looped through all the individual constraints - now to add the combined constraint to the costfunction
        self.constraints["combined_constraints"] = self._add_normalconstraint(
                                parameters=list(self.constraint_parameters.keys()),
                                values=self.constraint_values,
                                error=self.constraint_covariance,
                            )
        msg = f"added 'combined_constraints' as `NormalConstraint`"
        log.info(msg=msg)

        # now to do similar for the constraints_for_toypars_to_vary

        # preallocate if we have some parameters to vary for the toys
        if len(self.toypars_to_vary) > 0:
            self.toypars_to_vary_values = np.full(len(self.toypars_to_vary), np.nan)
            self.toypars_to_vary_covariance = np.identity(len(self.toypars_to_vary))

        for constraintname in constraints_for_toypars_to_vary:
            constraint = constraints[constraintname]
                                    
            # add the parameters to the set
            for par, value in zip(constraint["parameters"], constraint["values"]):
                self.toypars_to_vary_values[self.toypars_to_vary[par]] = value

            # now we need to check whether the constraint includes the covariance matrix or the uncertainty
            if "uncertainty" in constraint:
                for par, uncertainty in zip(constraint["parameters"], constraint["uncertainty"]):
                    i = self.toypars_to_vary[par]
                    self.toypars_to_vary_covariance[i,i] = np.square(uncertainty)                                                           
                
            else: # we have the covariance matrix for this constraint                            
                for i, par_i in enumerate(constraint["parameters"]):
                    for j, par_j in enumerate(constraint["parameters"]):
                        self.toypars_to_vary_covariance[
                            self.toypars_to_vary[par_i], 
                            self.toypars_to_vary[par_j]] = (
                                constraint["covariance"][i,j]
                        )                                              
                
            msg = f"including constraint '{constraintname}' parameters as parameters to vary for the toys"
            log.info(msg=msg)   

        # check that all parameters that need to be varied for toys have a constraint                          
        for parname, parind in self.toypars_to_vary.items():
            if parind is None:
                msg = f"no constraint was found for parameter '{parname}' which has 'vary_by_constraint' True"

        # that's all, folks!
                    
    def _add_normalconstraint(
        self,
        parameters: list[str],
        values: list[float],
        error,
    ) -> cost.NormalConstraint:
        thiscost = cost.NormalConstraint(parameters, values, error=error)

        self.costfunction = self.costfunction + thiscost

        return thiscost
