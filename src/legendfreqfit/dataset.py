"""
A class that holds a dataset and its associated model and cost function,
"""


import numpy as np
from iminuit import cost

SEED = 42


class Dataset:
    def __init__(
        self,
        data: np.array,
        model,
        model_parameters: dict[str, str],
        parameters: dict,
        costfunction: cost.Cost,
        name: str = "",
    ) -> None:
        """
        Parameters
        ----------
        data
            `list` or `np.array` of the unbinned data
        model
            model `class` to be passed to the cost function. The `model`
            class must have a method `model.density(data, a, b, c, ...)` where `data` takes a 1D `ndarray` corresponding
            to unbinned events, and `a, b, c, ...` are the parameters of the model, which may take any type and may
            include default values. `model` must return the form expected by `costfunction`. Those parameters which
            are to be fit by `iminuit` must take a single number. `model` must have a `parameters` attribute that is a
            `dict` with keys of the parameter names `a, b, c, ...` and values of the default values of the parameters
            or "nodefaultvalue" if no default is provided. (e.g. the return of the `utils.inspectparameters()` method).
        model_parameters
            `dict` that contains the mapping between the named model parameters `a, b, c, ...` and a string containing
            a different name for the parameter that will be passed to `iminuit` in place of the original name.
            This can be used to fit the same parameter across multiple datasets since `iminuit` references variables
            by name.
        parameters
            `dict` for the parameters of the fit that control how the fit is performed.

            - "includeinfit": True or False, corresponding to whether this parameter should be included in the `iminuit`
                fit or if its value should be fixed. This allows for passing variables of any type to the underlying
                `model`. If True, no "value" will be passed to `model` for this parameter and will instead be taken
                from `iminuit` during the minimization. If "False", then "value" is required if the parameter does not
                have a default value and will be passed to `model`. Note that it is still possible to include
                parameters in the `iminuit` minimization and set them to `fixed`, which can be desirable in some cases.

            - "value": If "includeinfit" is `False` and no default value for the parameter is specified in `model`, then
                this is required. This may be any type and will be passed to `model`. If "includeinfit" is `True`, this
                is not used. If "includeinfit" is `False` and a default value for the parameter exists, this will
                overwrite the default.

            - "limits": This allows for setting limits on the parameters for `iminuit`.
        costfunction
            an `iminuit` cost function. Currently, only `cost.ExtendedUnbinnedNLL` or `cost.UnbinnedNLL` are supported
            as cost functions.
        name
            `str` name for the `Dataset`

        Notes
        -----
        Currently, only `cost.ExtendedUnbinnedNLL` or `cost.UnbinnedNLL` are supported as cost functions.
        """

        self.data = np.asarray(data)
        self.name = name
        self.model = model

        # check that all passed parameters are valid
        for parameter in model_parameters:
            if parameter not in self.model.parameters:
                msg = f"`Dataset` `{self.name}`: model_parameter `{parameter}` not found in model `{self.model}`."
                raise KeyError(msg)

        # check that all required parameters were passed
        for parameter, defaultvalue in self.model.parameters.items():
            if (defaultvalue is str and defaultvalue == "nodefaultvalue") and (
                parameter not in model_parameters
            ):
                msg = f"`Dataset` `{self.name}`: required model parameter `{parameter}` not found in model_parameters"
                raise KeyError(msg)

        # make the cost function
        if (costfunction is cost.ExtendedUnbinnedNLL) or (
            costfunction is cost.UnbinnedNLL
        ):
            self._costfunctioncall = costfunction
            self.costfunction = costfunction(self.data, self.density)
        else:
            msg = f"`Dataset` `{self.name}`: only `cost.ExtendedUnbinnedNLL` or `cost.UnbinnedNLL` are supported as \
                cost functions"
            raise RuntimeError(msg)

        self.model_parameters = model_parameters
        self._parlist = []
        self._parlist_indices = (
            []
        )  # indices in self._parlist of the the parameters to be fit
        # dict of those parameters for fit, keys are parameter names, values are indices in self._parlist (same as in
        # self._parlist_indices)
        self.fitparameters = {}

        # now we make the parameters of the cost function
        # need to go in order of the model
        for i, (par, defaultvalue) in enumerate(self.model.parameters.items()):
            # if not passed, use default value (already checked that required parameters passed)
            if par not in model_parameters:
                self._parlist.append(defaultvalue)

            # parameter was passed and should be included in the fit
            elif ("includeinfit" in parameters[model_parameters[par]]) and (
                parameters[model_parameters[par]]["includeinfit"]
            ):
                self.costfunction._parameters |= {
                    model_parameters[par]: parameters[model_parameters[par]]["limits"]
                    if "limits" in parameters[model_parameters[par]]
                    else None
                }
                self._parlist.append(None)
                self._parlist_indices.append(i)
                self.fitparameters |= {model_parameters[par]: i}

            else:  # parameter was passed but should not be included in the fit
                if ("value" not in parameters[model_parameters[par]]) and (
                    defaultvalue == "nodefaultvalue"
                ):
                    msg = f"`Dataset` `{self.name}`: value for parameter `{par}` is required for \
                        model `{self.model}` parameter `{par}`"
                    raise KeyError(msg)
                self._parlist.append(parameters[model_parameters[par]]["value"])

        return

    def density(
        self,
        data,
        *par,
    ) -> np.array:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]

        return self.model.density(data, *self._parlist)

    def rvs(
        self,
        *par,
        seed: int = SEED,  # must be passed as keyword
    ) -> np.array:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]

        return self.model.extendedrvs(*self._parlist, seed=seed)
