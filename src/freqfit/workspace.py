"""
Workspace class for freqfit that controls aspects of the statistical analysis.
"""
import importlib
import numpy as np
import yaml

import logging

log = logging.getLogger(__name__)

SEED = 42

class Workspace:
    def __init__(
        self,
        config: dict,
        name: str = "",
    ) -> None:

        return

    @classmethod
    def from_file(
        cls,
        file: str,
        name: str = "",
    ):
        config = cls.load_config(file=file)
        return cls(config=config, name=name)

    @classmethod
    def from_dict(
        cls,
        input: dict,
        name: str = "",
    ):
        config = cls.load_config(file=input)
        return cls(config=config, name=name)

    @staticmethod
    def load_config(
        file: str | dict,
    ) -> dict:
        """
        Loads a config file or dict and converts `str` for some fields to the appropriate objects.
        """

        # if it's not a dict, it might be a path to a file
        if not isinstance(file, dict):
            with open(file) as stream:
                # switch from safe_load to load in order to check for duplicate keys
                config = yaml.load(stream, Loader=UniqueKeyLoader)

        else:
            config = file

        # get list of models and cost functions to import
        models = set()
        costfunctions = set()
        if "datasets" not in config:
            msg = f"`datasets` not found in `{file if file is not dict else 'provided `dict`'}`"
            raise KeyError(msg)

        if "parameters" not in config:
            msg = f"`parameters` not found in `{file if file is not dict else 'provided `dict`'}`"
            raise KeyError(msg)

        for datasetname, dataset in config["datasets"].items():
            if "model" in dataset:
                models.add(dataset["model"])
            else:
                msg = f"dataset `{datasetname}` has no `model`"
                raise KeyError(msg)

            if "costfunction" in dataset:
                costfunctions.add(dataset["costfunction"])
            else:
                msg = f"dataset `{datasetname}` has no `costfunction`"
                raise KeyError(msg)

        if "constraints" in config:
            for constraintname, constraint in config["constraints"].items():
                if "parameters" not in constraint:
                    msg = f"constraint `{constraintname}` has no `parameters`"
                    raise KeyError(msg)
                else:
                    # these need to be lists for other stuff
                    if not isinstance(constraint["parameters"], list):
                        constraint["parameters"] = [constraint["parameters"]]
                    if not isinstance(constraint["values"], list):
                        constraint["values"] = [constraint["values"]]
                    if "uncertainty" in constraint and not isinstance(
                        constraint["uncertainty"], list
                    ):
                        constraint["uncertainty"] = [constraint["uncertainty"]]
                    if "covariance" in constraint and not isinstance(
                        constraint["covariance"], np.ndarray
                    ):
                        constraint["covariance"] = np.asarray(constraint["covariance"])

        if "combined_datasets" in config:
            for groupname, group in config["combined_datasets"].items():
                if "model" in group:
                    models.add(group["model"])
                else:
                    msg = f"combined_datasets `{groupname}` has no `model`"
                    raise KeyError(msg)

            if "costfunction" in group:
                costfunctions.add(group["costfunction"])
            else:
                msg = f"combined_datasets `{groupname}` has no `costfunction`"
                raise KeyError(msg)

        # this is specific to set up of 0vbb model
        for model in models:
            modelclassname = model.split(".")[-1]
            modelclass = getattr(importlib.import_module(model), modelclassname)

            for datasetname, dataset in config["datasets"].items():
                if dataset["model"] == model:
                    dataset["model"] = modelclass

            if "combined_datasets" in config:
                for groupname, group in config["combined_datasets"].items():
                    if group["model"] == model:
                        group["model"] = modelclass

        # specific to iminuit
        for costfunctionname in costfunctions:
            costfunction = getattr(
                importlib.import_module("iminuit.cost"), costfunctionname
            )

            for datasetname, dataset in config["datasets"].items():
                if dataset["costfunction"] == costfunctionname:
                    dataset["costfunction"] = costfunction

            if "combined_datasets" in config:
                for groupname, group in config["combined_datasets"].items():
                    if group["costfunction"] == costfunctionname:
                        group["costfunction"] = costfunction

        # convert any limits from string to python object
        for par, pardict in config["parameters"].items():
            if "limits" in pardict and type(pardict["limits"]) is str:
                pardict["limits"] = eval(pardict["limits"])

            if "physical_limits" in pardict and type(pardict["physical_limits"]) is str:
                pardict["physical_limits"] = eval(pardict["physical_limits"])
            
            if "value_from_combine" in pardict:
                pardict["value"] = np.nan

        if "options" in config:
            for option, optionval in config["options"].items():
                if optionval in ["none", "None"]:
                    config["options"][option] = None

        return config


# use this YAML loader to detect duplicate keys in a config file
# https://stackoverflow.com/a/76090386 
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            each_key = self.construct_object(key_node, deep=deep)
            if each_key in mapping:
                raise ValueError(
                    f"Duplicate Key: {each_key!r} is found in YAML File.\n"
                    f"Error File location: {key_node.end_mark}"
                )
            mapping.add(each_key)
        return super().construct_mapping(node, deep)