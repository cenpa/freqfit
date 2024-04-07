import inspect

# takes a model function and returns a dict of its parameters with their default value
def inspectparameters(
        func
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
        if (i==0):
            continue

        name, par = item

        if (default:=par.default) is par.empty:
            r[name] = "nodefaultvalue"
        else:
            r[name] = default
        
    return r