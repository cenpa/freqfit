"""
Workspace class for freqfit that controls aspects of the statistical analysis.
"""
import importlib
import numpy as np
import yaml

from .dataset import Dataset, ToyDataset, CombinedDataset
from .parameters import Parameters
from .constraints import Constraints
from .experiment import Experiment

import logging

log = logging.getLogger(__name__)

SEED = 42

class Workspace:
    def __init__(
        self,
        config: dict,
    ) -> None:

        # load the config - done (move error checking to when loading the config and out of the other classes)

        # option defaults
        self.options = {}
        self.options["backend"] = "minuit" # "minuit" or "scipy"
        self.options["initial_guess_fcn"] = None
        self.options["iminuit_precision"] = 1e-10
        self.options["iminuit_strategy"] = 0
        self.options["iminuit_tolerance"] = 1e-5
        self.options["minimizer_options"] = {} # dict of options to pass to the iminuit minimizer
        self.options["scipy_minimizer"] = None
        self.options["try_to_combine_datasets"] = False
        self.options["test_statistic"] = "t_mu" # "t_mu", "q_mu", "t_mu_tilde", or "q_mu_tilde"
        self.options["use_grid_rounding"] = False # whether to stick parameters on a grid when evaluating the test statistic after minimizing
        self.options["use_log"] = False
        self.options["use_user_gradient"] = False

        # load in the global options
        if "options" in config:
            for opt in config["options"]:
                self.options[opt] = config["options"][opt]

        msg = f"setting backend to {config['options']['backend']}"
        logging.info(msg)

        # create the Parameters
        self.parameters = Parameters(config['parameters'])

        # create the Datasets
        self.datasets = {}

        for dsname, ds in config['datasets'].items():
            dsobj = Dataset(
                data=ds["data"],
                model=ds["model"],
                model_parameters=ds["model_parameters"],
                parameters=self.parameters,
                costfunction=ds["costfunction"],
                name=dsname,
                try_to_combine=ds['try_to_combine'] if 'try_to_combine' in ds else False,
                combined_dataset=ds['combined_dataset'] if 'combined_dataset' in ds else None,
                use_user_gradient=self.options["use_user_gradient"],
                use_log=self.options["use_log"],
            )

            self.datasets[dsname] = dsobj

        # create the CombinedDatasets
        # maybe there's more than one combined_dataset group
        for cdsname, cds in config['combined_datasets'].items():
            # find the Datasets to try to combine
            ds_tocombine = []
            dsname_tocombine = []
            for dsname, ds in self.datasets.items():
                if (
                    ds.try_to_combine
                    and ds.combined_dataset == cdsname
                    and ds.model.can_combine(ds.data, *ds._parlist)
                ):
                    ds_tocombine.append(ds)
                    dsname_tocombine.append(dsname)

            if len(ds_tocombine) > 1:
                combined_dataset = CombinedDataset(
                    datasets=ds_tocombine,
                    model=cds["model"],
                    model_parameters=cds["model_parameters"],
                    parameters=self.parameters,
                    costfunction=cds["costfunction"],
                    name=cdsname,
                    use_user_gradient=self.options["use_user_gradient"],
                    use_log=self.options["use_log"],
                )

                self.datasets[cdsname] = combined_dataset

                msg = f"created CombinedDataset '{cdsname}'"
                logging.info(msg)

                # delete the combined datasets
                for dsname in dsname_tocombine:
                    self.datasets.pop(dsname)
                    
                    msg = f"combined Dataset '{dsname}' into CombinedDataset '{cdsname}'"
                    logging.info(msg)

        # create the Constraints 
        self.constraints = None
        if not config["constraints"]:
            msg = "no constraints were provided"
            logging.info(msg)
        else:
            msg = "all constraints will be combined into a single NormalConstraint"
            logging.info(msg)

            self.constraints = Constraints(config["constraints"])

        # create the Experiment
        self.experiment = Experiment(
            datasets=self.datasets, 
            parameters=self.parameters, 
            constraints=self.constraints, 
            options=self.options,
            )

        return
        

        # create the ToyDatasets
        # create the Toy

        return

    @classmethod
    def from_file(
        cls,
        file: str,
    ):
        config = cls.load_config(file=file)
        return cls(config=config)

    @classmethod
    def from_dict(
        cls,
        input: dict,
    ):
        config = cls.load_config(file=input)
        return cls(config=config)

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

        for item in ["datasets", "parameters"]:
            if item not in config:
                msg = f"{item} not found in `{file if file is not dict else 'provided `dict`'}`"
                raise KeyError(msg)
        
        if "options" not in config:
            config["options"] = {}
        
        if "constraints" not in config:
            config["constraints"] = {}

        # get list of models and cost functions to import
        models = set()
        costfunctions = set()
        for datasetname, dataset in config["datasets"].items():
            if "model" in dataset:
                models.add(dataset["model"])
            else:
                msg = f"dataset `{datasetname}` has no Model"
                raise KeyError(msg)

            if "costfunction" in dataset:
                if dataset["costfunction"] in ["ExtendedUnbinnedNLL", "UnbinnedNLL"]:
                    costfunctions.add(dataset["costfunction"])
                else:
                    msg = f"Dataset `{datasetname}`: only 'ExtendedUnbinnedNLL' or 'UnbinnedNLL' are \
                        supported as cost functions"
                    raise NotImplementedError(msg)                    
            else:
                msg = f"dataset `{datasetname}` has no `costfunction`"
                raise KeyError(msg)
            
            if "try_to_combine" in dataset and dataset["try_to_combine"]:
                if "combined_dataset" not in dataset or not dataset["combined_dataset"]:
                        msg = (f"Dataset `{datasetname}` has `try_combine` `{dataset['try_to_combine']}` but "
                            + f"`combined_dataset` missing or empty")
                        raise KeyError(msg)     
                elif (("combined_datasets" not in config)
                    or (dataset["combined_dataset"] not in config["combined_datasets"])):
                        msg = (f"Dataset `{datasetname}` has `combined_dataset` `{dataset['combined_dataset']}` but " 
                            + f"`combined_datasets` missing or does not contain `{dataset['combined_dataset']}`")
                        raise KeyError(msg)   
                elif (config["combined_datasets"][dataset["combined_dataset"]]["model"] != dataset["model"]):
                        msg = (f" Dataset `{datasetname}` Model `{dataset['model']}` not the same as CombinedDataset "
                            + f"`{dataset['combined_dataset']}` Model "
                            + f"`{config['combined_datasets'][dataset['combined_dataset']]['model']}`")
                        raise ValueError(msg)

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
                    msg = f"combined_datasets `{groupname}` has no Model"
                    raise KeyError(msg)

            if "costfunction" in group:
                costfunctions.add(group["costfunction"])
            else:
                msg = f"combined_datasets `{groupname}` has no `costfunction`"
                raise KeyError(msg)
        else:
            config["combined_datasets"] = {}

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
        # set defaults if options missing
        for par, pardict in config["parameters"].items():

            for item in ["limits", "value"]:
                if item not in pardict:
                    pardict[item] = None

            for item in ["includeinfit", "fixed", "fix_if_no_data", "vary_by_constraint", "value_from_combine"]:
                if item not in pardict:
                    pardict[item] = False
            
            if "grid_rounding_num_decimals" not in pardict:
                pardict["grid_rounding_num_decimals"] = 128

            if "limits" in pardict and type(pardict["limits"]) is str:
                pardict["limits"] = eval(pardict["limits"])

            if "physical_limits" not in pardict:
                pardict["physical_limits"] = None
            elif type(pardict["physical_limits"]) is str:
                pardict["physical_limits"] = eval(pardict["physical_limits"])
            
        # options
        for option, optionval in config["options"].items():
            if optionval in ["none", "None"]:
                config["options"][option] = None
        
        for item in ["user_gradient", "use_log", "scan", "use_grid_rounding"]:
            if item not in config["options"]:
                config["options"][item] = False

        for item in ["initial_guess_function"]:
            if item not in config["options"]:
                config["options"][item] = None
        
        if "backend" not in config["options"]:
            config["options"]["backend"] = "minuit"
        
        if config["options"]["backend"] not in ["minuit", "scipy"]:
            raise NotImplementedError(
              "backend is not set to 'minuit' or 'scipy'"  
            )

        if "minimizer_options" not in config["options"]:
            config["options"]["minimizer_options"] = {}
        
        if not isinstance(config["options"]["minimizer_options"], dict):
            raise ValueError("options: minimizer_options must be a dict")

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