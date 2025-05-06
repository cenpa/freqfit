import inspect
import logging

import numpy as np
import yaml

log = logging.getLogger(__name__)

#-----------------------------
# moved to Model
#-----------------------------
# takes a model function and returns a dict of its parameters with their default value
def inspectparameters(
    func,
) -> dict:
    """
    Returns a `dict` of parameters that methods of this model take as keys. Values are default values of the
    parameters. Assumes the first argument of the model is `data` and not a model parameter, so this key is not
    returned.
    """
    # pulled some of this from `iminuit.util`
    try:
        signature = inspect.signature(func)
    except ValueError:  # raised when used on built-in function
        return {}

    r = {}
    for i, item in enumerate(signature.parameters.items()):
        if i == 0:
            continue

        name, par = item

        if (default := par.default) is par.empty:
            r[name] = "nodefaultvalue"
        else:
            r[name] = default

    return r



def grab_results(
    minuit,
    use_grid_rounding: bool = False,
    grid_rounding_num_decimals: dict = {},  # noqa: B006
) -> dict:
    # I checked whether we need to shallow/deep copy these and it seems like we do not

    toreturn = {}
    toreturn["errors"] = minuit.errors.to_dict()  # returns dict
    toreturn["fixed"] = minuit.fixed.to_dict()  # returns dict
    toreturn["fval"] = minuit.fval  # returns float
    toreturn["nfit"] = minuit.nfit  # returns int
    toreturn["npar"] = minuit.npar  # returns int
    toreturn["parameters"] = minuit.parameters  # returns tuple of str
    toreturn["tol"] = minuit.tol  # returns float
    toreturn["valid"] = minuit.valid  # returns bool
    toreturn["values"] = minuit.values.to_dict()  # returns dict

    if use_grid_rounding:
        toreturn["values"] = {
            key: np.around(value, grid_rounding_num_decimals[key])
            for key, value in minuit.values.to_dict().items()
        }  # returns dict
        toreturn["fval"] = minuit._fcn(
            toreturn["values"].values()
        )  # overwrite the fval with the truncated params

    return toreturn





# negative of the exponent of scientific notation of a number
def negexpscinot(number):
    base10 = np.log10(abs(number))
    return int(-1 * np.floor(base10))
