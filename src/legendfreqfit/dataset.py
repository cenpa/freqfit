"""
A class that holds a dataset and its associated model and cost function,
"""

import warnings
import numpy as np
from iminuit import cost, util
import inspect
import typing

class Dataset:
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
        data
            D `ndarray` of the data, which should be a list of energies
        model
            callable model to be passed to the cost function (e.g. `gaussian_on_uniform.density`).`model`
            must have a callable form `model(data, a, b, c, ...)` where `data` takes a 1D `ndarray` corresponding
            to unbinned events, and `a, b, c,...` are the parameters of the model, which may take any type and may include
            default values. `model` must return the form expected by `costfunction`. Those parameters which are to be 
            fit by `iminuit` must take a single number.
        parameters
            `dict` for the parameters of this particular model that allow for control over how a fit is performed across
            multiple datasets. The `dict` should have keys corresponding to each required parameter of `model` (each
            parameter without a default value), where the key is a string of the name of each parameter, e.g. for
            parameters `a, b, c, ...`, the keys are `"a", "b", "c", ...`.
            
            The value for each key should be a `dict` that can contain the following keys and values: 
            - "name": a string containing a different name for the parameter that will be passed to `iminuit` in place
                of the original name. This can be used to fit the same parameter across multiple datasets since
                `iminuit` references variables by name. This key is optional and the default parameter name from `model`
                will be used if "name" is not found.

            - "includeinfit": True or False, corresponding to whether this parameter should be included in the `iminuit`
                fit or if its value should be fixed. This allows for passing variables of any type to the underlying
                `model`. If True, no "value" will be passed to `model` for this parameter and will instead be taken
                from `iminuit` during the minimization. If "False", then "value" is required if the parameter does not
                have a default value and will be passed to `model`. Note that it is still possible to include 
                parameters in the `iminuit` minimization and set them to `fixed`, which can be desireable in some cases.
            
            - "value": If "includeinfit" is `False` and no default value for the parameter is specified in `model`, then
                this is required. This may be any type and will be passed to `model`. If "includeinfit" is `True`, this
                is not used. If "includeinfit" is `False` and a default value for the parameter exists, this will 
                overwrite the default.
            
            - "limits": This allows for setting limits on the parameters for `iminuit`. It is still possible to set
                limits through `iminuit` for parameters, but this key can be convenient for parameters not shared 
                between `Dataset`. Note that for parameters shared across several `Dataset`, limits specified for 
                such a parameter may be overwritten if a different limit is specified in a different `Dataset` whose
                cost functions are then added. 

        costfunction
            an `iminuit` cost function. Currently, only `cost.ExtendedUnbinnedNLL` or `cost.UnbinnedNLL` are supported 
            as cost functions.

        Notes
        -----
        Currently, only `cost.ExtendedUnbinnedNLL` or `cost.UnbinnedNLL` are supported as cost functions.
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
        # par should be 1D array like
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