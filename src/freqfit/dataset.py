"""
A class that holds a dataset and its associated model and cost function,
"""
import logging

import numpy as np
from iminuit import cost

SEED = 42

log = logging.getLogger(__name__)


class Dataset:
    def __init__(
        self,
        data: np.array,
        model,
        model_parameters: dict[str, str],
        parameters: dict,
        costfunction: cost.Cost,
        name: str = "",
        try_to_combine: bool = False,
        combined_dataset: str = None,
        use_user_gradient: bool = False,
        use_log: bool = False,
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
        use_user_gradient
            If true use the user's gradient in the costfunction
        use_log
            If true, use the log of the PDF instead of the PDF within the Iminuit cost function

        Notes
        -----
        Currently, only `cost.ExtendedUnbinnedNLL` or `cost.UnbinnedNLL` are supported as cost functions.
        """

        self.data = np.asarray(data)  # the data of this Dataset
        self.name = name  # name of this Dataset
        self.model = model  # model object for this Dataset
        self.model_parameters = (
            model_parameters  # model parameter name : parameter name
        )

        # self._costfunctioncall = None  # function call for the costfunction
        self.costfunction = None  # iminuit cost function object

        self._parlist = (
            []
        )  # list that contains all of the parameters of the model for this Dataset in the correct order
        self._parlist_indices = (
            []
        )  # indices in self._parlist of the the parameters to be fit

        self.fitparameters = (
            {}
        )  # fit parameter names : index in self._parlist (same as in self._parlist_indices)
        self._toy_pars_to_vary = (
            {}
        )  # parameter to vary for toys : index in self._parlist


        self.use_user_gradient = use_user_gradient
        self.use_log = use_log

        # self._toy_data = None  # place to hold Toy data for this Dataset
        # self._toy_num_drawn = None  # place to hold the number of signal and number of background events drawn for a toy
        # self._toy_is_combined = (
        #     False  # if this Dataset is combined into a combined_dataset in the Toy
        # )


        # do something with these
        self.try_to_combine = try_to_combine  # whether to attempt to combine this Dataset into a combined_dataset
        self.is_combined = (
            False  # whether this Dataset is combined into a combined_dataset
        )
        self.combined_dataset = (
            None  # name of the combined_dataset that this Dataset is part of
        )
        if self.try_to_combine:
            self.combined_dataset = combined_dataset

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
        # self._costfunctioncall = costfunction
        if self.use_user_gradient:
            self.costfunction = costfunction(
                self.data, self.density, grad=self.density_gradient
            )
        elif use_log:
            self.costfunction = costfunction(
                self.data,
                self.log_density,
                log=True,
            )
        else:
            self.costfunction = costfunction(self.data, self.density)

        # now we make the parameters of the cost function
        # need to go in order of the model
        for i, (par, defaultvalue) in enumerate(self.model.parameters.items()):
            # if not passed, use default value (already checked that required parameters passed)
            if par not in model_parameters:
                self._parlist.append(defaultvalue)
                break  # no constraints on non-passed parameters

            # parameter was passed and should be included in the fit
            elif ("includeinfit" in parameters[model_parameters[par]]) and (
                parameters[model_parameters[par]]["includeinfit"]
            ):
                self.costfunction._parameters |= {
                    model_parameters[par]: parameters[model_parameters[par]]["limits"]
                    if "limits" in parameters[model_parameters[par]]
                    else None
                }

                if "value" not in parameters[model_parameters[par]]:
                    msg = (
                        f"`Dataset` `{self.name}`: value for parameter `{par}` is required for"
                        + f" model `{model}` parameter `{par}`"
                    )
                    raise KeyError(msg)
                self._parlist.append(parameters[model_parameters[par]]["value"])
                self._parlist_indices.append(i)
                self.fitparameters |= {model_parameters[par]: i}

            else:  # parameter was passed but should not be included in the fit
                if ("value" not in parameters[model_parameters[par]]) and (
                    defaultvalue == "nodefaultvalue"
                ):
                    msg = (
                        f"`Dataset` '{self.name}': value for parameter '{par}' is required for"
                        + f"model '{model}' parameter '{par}'"
                    )
                    raise KeyError(msg)
                self._parlist.append(parameters[model_parameters[par]]["value"])

            if ("vary_by_constraint" in parameters[model_parameters[par]]) and (
                parameters[model_parameters[par]]["vary_by_constraint"]
            ):
                self._toy_pars_to_vary[model_parameters[par]] = i
                msg = f"`Dataset` '{self.name}': adding parameter '{model_parameters[par]}' as parameter to vary for toys"
                logging.info(msg)

        return

    def density(
        self,
        data,
        *par,  # DO NOT DELETE THE * - NEEDED FOR IMINUIT
    ) -> np.array:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]

        return self.model.density(data, *self._parlist)

    def log_density(
        self,
        data,
        *par,  # DO NOT DELETE THE * - NEEDED FOR IMINUIT
    ) -> np.array:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]

        return self.model.log_density(data, *self._parlist)

    def density_gradient(
        self,
        data,
        *par,  # DO NOT DELETE THE * - NEEDED FOR IMINUIT
    ) -> np.array:
        """
        Parameters
        ----------
        data
            Unbinned data
        par
            Potentially a subset of the actual model density_gradient parameters, depending on the config
        """
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]

        grad_cdf, grad_pdf = self.model.density_gradient(data, *self._parlist)

        # Mask the return values according to what the actual cost function expects
        return grad_cdf[self._parlist_indices], *grad_pdf[[self._parlist_indices], :]


class ToyDataset(Dataset):
    def __init__(        
        self,
        *args,
    ) -> None:

        super().__init__(*args)

        self.num_drawn = None

    # resets some toy attributes
    def reset(
        self,
    ) -> None:
        self.data = None
        self.num_drawn = None
        self.is_combined = False
        return

    def _rvs(
        self,
        par,
        seed: int = SEED,
    ) -> np.array:
        # par should be 1D array like
        # assign the positional parameters to the correct places in the model parameter list
        for i in range(len(par)):
            self._parlist[self._parlist_indices[i]] = par[i]

        # TODO: extendedrvs here? make it more generic or require this in Model?
        rvs, num_drawn = self.model.extendedrvs(*self._parlist, seed=seed)
        return rvs, num_drawn

    # generates toy data and sets some attributes
    def toy(
        self,
        par,
        seed: int = SEED,
    ) -> None:
        self.reset()
        self.data, self.num_drawn = self.rvs(par, seed=seed)
        return


class CombinedDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset, ...],
        model,
        model_parameters: dict[str, str],
        parameters: dict,
        costfunction: cost.Cost,
        name: str = "",
        use_user_gradient: bool = False,
        use_log: bool = False,
    ) -> None:
        """
        Does not check whether provided datasets can be combined.
        """

        self.datasets = datasets
        topass = []
        for ds in datasets:
            topass.append((ds.data, *ds._parlist))
        combination = model.combine(topass)
        combined_data = combination[0]

        # now to set the parameters based on the combined results
        combined_parameters = {}
        # model.parameters contains all parameters, including default valued ones
        for i, par in enumerate(model.parameters.keys()):
            # let Dataset handle errors if parameters are missing
            if par in model_parameters:
                parname = model_parameters[par]
                # this is a reference not a copy!
                combined_parameters[parname] = parameters[parname]
                # hence why it is important to check whether we should overwrite the value
                if (
                    "value_from_combine" in parameters[parname]
                    and parameters[parname]["value_from_combine"]
                ):
                    # i+1 because data occupies index 0
                    combined_parameters[parname]["value"] = combination[i + 1]
        
        data = combination[0]

        super().__init__(
            data,
            model,
            model_parameters,
            combined_parameters,
            costfunction,
            name,
            use_user_gradient=use_user_gradient,
            use_log=use_log,
        )

        return

# temporary while refactoring
def combine_datasets():
    return