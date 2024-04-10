import inspect
import yaml
import importlib

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
        if (i==0):
            continue

        name, par = item

        if (default:=par.default) is par.empty:
            r[name] = "nodefaultvalue"
        else:
            r[name] = default
        
    return r

def load_config(
        file: str,
        ) -> dict:
    """
    Loads a config file and converts `str` for some fields to the appropriate objects.
    """
    with open(file) as stream:
        config = yaml.safe_load(stream)

    # get list of models and cost functions to import
    models = set()
    costfunctions = set()
    if "datasets" not in config:
        msg = (
            f"`datasets` not found in `{file}`"
        )
        raise KeyError(msg)

    if "parameters" not in config:
        msg = (
            f"`parameters` not found in `{file}`"
        )
        raise KeyError(msg)
    
    for datasetname, dataset in config["datasets"].items():
        if "model" in dataset:
            models.add(dataset["model"])
        else:
            msg = (
                f"`{datasetname}` has no `model`"
            )
            raise KeyError(msg)
        
        if "costfunction" in dataset:
            costfunctions.add(dataset["costfunction"])
        else:
            msg = (
                f"`{datasetname}` has no `costfunction`"
            )
            raise KeyError(msg)

    # this is specific to set up of 0vbb model
    for model in models:
        modelclassname = model.split('.')[-1]
        modelclass = getattr(importlib.import_module(model), modelclassname)

        for datasetname, dataset in config["datasets"].items():
            if dataset["model"] == model:
                dataset["model"] = modelclass

    # specific to iminuit
    for costfunctionname in costfunctions:
        costfunction = getattr(importlib.import_module("iminuit.cost"), costfunctionname)

        for datasetname, dataset in config["datasets"].items():
            if dataset["costfunction"] == costfunctionname:
                dataset["costfunction"] = costfunction

    # convert any limits from string to python object
    for par, pardict in config["parameters"].items():
        if "limits" in pardict and type(pardict["limits"]) is str:
            pardict["limits"] = eval(pardict["limits"])

    return config
    