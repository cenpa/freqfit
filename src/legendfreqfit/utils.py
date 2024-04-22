import importlib
import inspect
import json
import re
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
        config = yaml.safe_load(stream)

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


def convert_gerda_results_to_config(
    metadata_path: str, data_path: str, output_path: str
) -> None:
    """
    Parameters
    ----------
    metadata_path
        Absolute path to `ovbb-analysis-parameters.txt` as provided by the GERDA Collaboration
    data_path
        Absolute path to `0vbb-analysis-event-list.txt`  as provided by the GERDA Collaboration
    output_path
        Absolute path to dump the `gerda_config.yaml` file
    """

    # Load the metadata file
    f_metadata = open(metadata_path)
    file_metadata = f_metadata.readlines()
    f_metadata.close()

    # Convert the file to one large string, then modify the string so that it can be loaded as a json dict
    file_as_string_metadata = "".join(file_metadata[22:])
    d_metadata = re.sub(r"\}\n\{", "},{", file_as_string_metadata)
    d_metadata = re.sub(r'("eff_tot": *.*),', r"\1", d_metadata)
    metadata = json.loads(d_metadata)

    # Load the analysis data
    f_data = open(data_path)
    file_data = f_data.readlines()
    f_data.close()

    # Get lists of the event times, energies, and which detectors
    times = []
    energies = []
    detectors = []
    for row in file_data[10:]:
        time = row[22:32]
        energy = row[-8:-2]
        detector = row[43:48] if row[43] != " " else row[44:48]
        detector = detector.strip()

        times.append(time)
        energies.append(energy)
        detectors.append(detector)

    output_data = dict({"datasets": dict({})})
    datasets = list(metadata.keys())
    # Create empty datasets in partitions that need it
    for dset in datasets:
        for det in list(metadata[dset].keys()):
            output_data["datasets"][dset + "_" + det] = dict({})
            output_data["datasets"][dset + "_" + det]["data"] = []

    # Place the event energies in the correct datasets
    for j in range(len(energies)):
        for i in range(len(datasets) - 1):
            if datasets[i] <= times[j] < datasets[i + 1]:
                output_data["datasets"][datasets[i] + "_" + detectors[j]][
                    "data"
                ].append(float(energies[j]))
        if times[j] > datasets[-1]:
            output_data["datasets"][datasets[-1] + "_" + detectors[j]]["data"].append(
                float(energies[j])
            )

    # Make sure we are outputting a list for yaml, this might not be necessary depending on how we read it in
    for dset in datasets:
        for det in list(metadata[dset].keys()):
            output_data["datasets"][dset + "_" + det]["data"] = list(
                output_data["datasets"][dset + "_" + det]["data"]
            )

    # add all the stuff we need to each dataset to identify which parameters to fit
    # NOTE: these are the lines that link the S and B parameters together for ALL datasets

    for key in list(output_data["datasets"].keys()):
        output_data["datasets"][key]["costfunction"] = "ExtendedUnbinnedNLL"
        output_data["datasets"][key][
            "model"
        ] = "legendfreqfit.models.gaussian_on_uniform"
        output_data["datasets"][key]["model_parameters"] = dict({})
        output_data["datasets"][key]["model_parameters"]["S"] = "global_S"
        output_data["datasets"][key]["model_parameters"]["BI"] = "global_BI"
        output_data["datasets"][key]["model_parameters"]["delta"] = f"delta_{key}"
        output_data["datasets"][key]["model_parameters"]["sigma"] = f"sigma_{key}"
        output_data["datasets"][key]["model_parameters"]["eff"] = f"eff_{key}"
        output_data["datasets"][key]["model_parameters"]["exp"] = f"exp_{key}"

    # Now set the parameters and initial values from the metadata
    output_data["parameters"] = dict({})

    output_data["parameters"]["global_S"] = dict({})

    output_data["parameters"]["global_S"]["includeinfit"] = True
    output_data["parameters"]["global_S"]["nuisance"] = False
    output_data["parameters"]["global_S"]["fixed"] = False
    output_data["parameters"]["global_S"]["limits"] = str([0, None])
    output_data["parameters"]["global_S"]["value"] = 1e-9

    output_data["parameters"]["global_BI"] = dict({})
    output_data["parameters"]["global_BI"]["includeinfit"] = True
    output_data["parameters"]["global_BI"]["nuisance"] = False
    output_data["parameters"]["global_BI"]["fixed"] = False
    output_data["parameters"]["global_BI"]["limits"] = str([0, None])
    output_data["parameters"]["global_BI"]["value"] = 1e-3

    # now handle the loop over the individual dataset nuisance parameters
    datasets = list(metadata.keys())
    # Create empty datasets in partitions that need it
    for dset in datasets:
        for det in list(metadata[dset].keys()):
            output_data["parameters"]["delta_" + dset + "_" + det] = dict({})
            output_data["parameters"]["delta_" + dset + "_" + det][
                "includeinfit"
            ] = False
            output_data["parameters"]["delta_" + dset + "_" + det]["nuisance"] = True
            output_data["parameters"]["delta_" + dset + "_" + det]["fixed"] = False
            output_data["parameters"]["delta_" + dset + "_" + det]["limits"] = None
            output_data["parameters"]["delta_" + dset + "_" + det]["value"] = 0.0

            output_data["parameters"]["sigma_" + dset + "_" + det] = dict({})
            output_data["parameters"]["sigma_" + dset + "_" + det][
                "includeinfit"
            ] = True
            output_data["parameters"]["sigma_" + dset + "_" + det]["nuisance"] = True
            output_data["parameters"]["sigma_" + dset + "_" + det]["fixed"] = True
            output_data["parameters"]["sigma_" + dset + "_" + det]["limits"] = None
            output_data["parameters"]["sigma_" + dset + "_" + det]["value"] = metadata[
                dset
            ][det]["fwhm"]

            output_data["parameters"]["eff_" + dset + "_" + det] = dict({})
            output_data["parameters"]["eff_" + dset + "_" + det]["includeinfit"] = False
            output_data["parameters"]["eff_" + dset + "_" + det]["limits"] = None
            output_data["parameters"]["eff_" + dset + "_" + det]["value"] = metadata[
                dset
            ][det]["eff_tot"]

            output_data["parameters"]["exp_" + dset + "_" + det] = dict({})
            output_data["parameters"]["exp_" + dset + "_" + det]["includeinfit"] = False
            output_data["parameters"]["exp_" + dset + "_" + det]["limits"] = None
            output_data["parameters"]["exp_" + dset + "_" + det]["value"] = metadata[
                dset
            ][det]["exposure"]

    # Now, enjoy the fruits of our labors. Write to a file
    f = open(output_path + "/gerda_config.yaml", "w+")
    yaml.safe_dump(output_data, f)
    f.close()
