import importlib
import inspect
import warnings

import yaml


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


def load_config(
    file: str,
) -> dict:
    """
    Loads a config file and converts `str` for some fields to the appropriate objects.
    """
    with open(file) as stream:
        # switch from safe_load to load in order to check for duplicate keys
        config = yaml.load(stream, Loader=UniqueKeyLoader)

    # get list of models and cost functions to import
    models = set()
    costfunctions = set()
    if "datasets" not in config:
        msg = f"`datasets` not found in `{file}`"
        raise KeyError(msg)

    if "parameters" not in config:
        msg = f"`parameters` not found in `{file}`"
        raise KeyError(msg)

    for datasetname, dataset in config["datasets"].items():
        if "model" in dataset:
            models.add(dataset["model"])
        else:
            msg = f"`{datasetname}` has no `model`"
            raise KeyError(msg)

        if "costfunction" in dataset:
            costfunctions.add(dataset["costfunction"])
        else:
            msg = f"`{datasetname}` has no `costfunction`"
            raise KeyError(msg)

    # this is specific to set up of 0vbb model
    for model in models:
        modelclassname = model.split(".")[-1]
        modelclass = getattr(importlib.import_module(model), modelclassname)

        for datasetname, dataset in config["datasets"].items():
            if dataset["model"] == model:
                dataset["model"] = modelclass

    # specific to iminuit
    for costfunctionname in costfunctions:
        costfunction = getattr(
            importlib.import_module("iminuit.cost"), costfunctionname
        )

        for datasetname, dataset in config["datasets"].items():
            if dataset["costfunction"] == costfunctionname:
                dataset["costfunction"] = costfunction

    # convert any limits from string to python object
    for par, pardict in config["parameters"].items():
        if "limits" in pardict and type(pardict["limits"]) is str:
            pardict["limits"] = eval(pardict["limits"])
        if "includeinfit" in pardict and not pardict["includeinfit"]:
            if "nuisance" in pardict and pardict["nuisance"]:
                msg = f"{par} has `includeinfit` as `False` but `nuisance` as `True`. {par} will not be included in fit."
                warnings.warn(msg)

    return config


def grab_results(
    minuit,
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

    return toreturn

# use this YAML loader to detect duplicate keys in a config file
# https://stackoverflow.com/a/76090386
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            each_key = self.construct_object(key_node, deep=deep)
            if each_key in mapping:
                raise ValueError(f"Duplicate Key: {each_key!r} is found in YAML File.\n"
                                 f"Error File location: {key_node.end_mark}")
            mapping.add(each_key)
        return super().construct_mapping(node, deep)