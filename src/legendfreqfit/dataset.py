"""
A class that holds a dataset and an associated model,
"""

import warnings
import numpy as np
from iminuit import cost, util
import inspect
import typing

class dataset:
    def __init__(
        self, 
        data:np.array,
        model,
        parameters: dict,
        costfunction,
        ) -> None:
        """
        Parameters
        ----------
        model
            a model class that has some particular functions
        kwmap
            a dictionary for this dataset that maps the aribitrary variable names to
            the variable names that the model is expecting.
        """

        self.data = data

        self.modelname = model
        modelparameters = inspectmodel(self.modelname)
        
        # check that all passed parameters are valid
        for parameter in parameters:
            if parameter not in modelparameters:
                msg = (
                    f"parameter `{parameter}` not found in model `{self.modelname}`."
                )
                raise KeyError(msg)
        
        # check that all required parameters were passed
        for i,modelparameter in enumerate(modelparameters):
            if i==0: # because model should take data as first argument
                continue
            if ((modelparameters[modelparameter] == "nodefaultvalue") 
                and (modelparameter not in parameters)):
                msg = (
                    f"required model parameter `{modelparameter}` not found in parameters"
                )
                raise KeyError(msg)

        self.parameters = parameters
        self._parlist = []
        self._partofitindices = [] # indices in self._parlist of the the parameters to be fit

        if ((costfunction is cost.ExtendedUnbinnedNLL) or (costfunction is cost.UnbinnedNLL)):
            self.costfunction = costfunction(self.data, self.model)
        else:
            msg = (
                f"only `cost.ExtendedUnbinnedNLL` or `cost.UnbinnedNLL` are supported as cost functions"
            )
            raise RuntimeError(msg)

        # now we make the parameters of the cost function
        # need to go in order of the model
        for i, modelparameter in enumerate(modelparameters):   
            if i==0: # because model should take data as first argument
                continue
            # if not passed, use default value (already checked that required parameters passed)
            if modelparameter not in parameters:
                self._parlist.append(modelparameters[modelparameter])
            # parameter was passed and should be included in the fit
            elif "includeinfit" in parameters[modelparameter] and parameters[modelparameter]["includeinfit"]:       
                self.costfunction._parameters |= {
                    parameters[modelparameter]["name"] if "name" in parameters[modelparameter] else modelparameter: 
                    parameters[modelparameter]["limits"] if "limits" in parameters[modelparameter] else None}
                self._parlist.append(None)
                self._partofitindices.append(i-1) # because model should take data as first argument
            else: # parameter was passed but should not be included in the fit
                if (("value" not in parameters[modelparameter]) and (modelparameters[parameter] == "nodefaultvalue")):
                    msg = (
                        f"value for parameter `{parameter}` is required for model `{self.model}` parameter `{parameter}`"
                    )
                    raise KeyError(msg)
                self._parlist.append(parameters[modelparameter]["value"])

        return 

    def rvs(self) -> np.array:
        if self.data is not None:
            msg = (
                f"data already exists in dataset"
            )
            warnings.warn(msg)
        
        self.data = self.model.rvs()

        return self.data

    def model(self, data, *par):
        #par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._partofitindices[i]] = par[i]
            
        return self.modelname(data, *self._parlist)

# takes a model and returns a dict of its parameters with their default value
def inspectmodel(model):
    # pulled some of this from `iminuit.util`
    try:
        signature = inspect.signature(model)
    except ValueError:  # raised when used on built-in function
        return {}

    r = {}
    for name, par in signature.parameters.items():
        if (default:=par.default) is par.empty:
            r[name] = "nodefaultvalue"
        else:
            r[name] = default
        
    return r