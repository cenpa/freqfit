"""
Workspace class for freqfit that controls aspects of the statistical analysis.
"""
import importlib
import numpy as np
import yaml

from .dataset import Dataset, ToyDataset, CombinedDataset
from .parameters import Parameters
from .constraints import Constraints, ToyConstraints
from .experiment import Experiment
from .model import Model
from .guess import Guess

import logging

log = logging.getLogger(__name__)

SEED = 42

class Workspace:
    def __init__(
        self,
        config: dict,
    ) -> None:

        # load the config

        # load in the global options - defaults and error checking in load_config
        self.options = config["options"]

        msg = f"setting backend to {config['options']['backend']}"
        logging.info(msg)

        # create the Parameters
        self.parameters = Parameters(config['parameters'])

        # create the Datasets
        datasets = {}
        for dsname, ds in config['datasets'].items():
            datasets[dsname] = Dataset(
                data=ds["data"],
                model=ds["model"],
                model_parameters=ds["model_parameters"],
                parameters=self.parameters,
                costfunction=ds["costfunction"],
                name=dsname,
                try_to_combine=self.options["try_to_combine_datasets"],
                combined_dataset=ds['combined_dataset'],
                use_user_gradient=self.options["use_user_gradient"],
                use_log=self.options["use_log"],
            )

        # create the ToyDatasets
        self._toy_datasets = {}
        for dsname, ds in config['datasets'].items():
            self._toy_datasets[dsname] = ToyDataset(
                toy_model=ds["toy_model"],
                toy_model_parameters=ds["toy_model_parameters"] if "toy_model_parameters" in ds else ds["model_parameters"],
                model=ds["model"],
                model_parameters=ds["model_parameters"],
                parameters=self.parameters,
                costfunction=ds["costfunction"],
                name=dsname,
                try_to_combine=self.options["try_to_combine_datasets"],
                combined_dataset=ds['combined_dataset'],
                use_user_gradient=self.options["use_user_gradient"],
                use_log=self.options["use_log"],                
            )

        self._combined_datasets = config['combined_datasets']

        # create the CombinedDatasets
        # maybe there's more than one combined_dataset group
        for cdsname, cds in self._combined_datasets.items():
            # find the Datasets to try to combine
            ds_tocombine = []
            dsname_tocombine = []
            for dsname, ds in datasets.items():
                if (
                    self.options["try_to_combine_datasets"]
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

                datasets[cdsname] = combined_dataset

                msg = f"created CombinedDataset '{cdsname}'"
                logging.info(msg)

                # delete the combined datasets
                for dsname in dsname_tocombine:
                    datasets.pop(dsname)
                    
                    msg = f"combined Dataset '{dsname}' into CombinedDataset '{cdsname}'"
                    logging.info(msg)

        # create the Constraints and ToyConstraints
        self.constraints = None
        if not config["constraints"]:
            msg = "no constraints were provided"
            logging.info(msg)
        else:
            msg = "constraints were provided"
            logging.info(msg)
            self.constraints = Constraints(config["constraints"], self.parameters)
            self.toy_constraints = ToyConstraints(config["constraints"], self.parameters)

        # create the Experiment
        self.experiment = Experiment(
            datasets=datasets, 
            parameters=self.parameters, 
            constraints=self.constraints, 
            options=self.options,
            )       

        return

    def make_toy(
        self,
        toy_parameters: dict,
        seed: int = SEED,
    ) -> Experiment:
        """
        returns an Experiment with toy data that has been varied according to the provided parameters

        Parameters
        ----------
        toy_parameters: dict
            Dictionary containing values of the parameters at which the toy data should be generated.
            Format is parameter name : parameter value.
        seed: int
            seed for random number generation
        """

        # seed here
        np.random.seed(seed)

        # vary the datasets
        rvs_datasets = {}
        for dsname, ds in self._toy_datasets.items():
            ds.rvs(toy_parameters)
            rvs_datasets["toy_" + dsname] = ds

        # combine the datasets
        for cdsname, cds in self._combined_datasets.items():
            # find the Datasets to try to combine
            ds_tocombine = []
            dsname_tocombine = []
            for dsname, ds in rvs_datasets.items():
                if (
                    self.options["try_to_combine_datasets"]
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
                    name="toy_" + cdsname,
                    use_user_gradient=self.options["use_user_gradient"],
                    use_log=self.options["use_log"],
                )

                rvs_datasets[cdsname] = combined_dataset

                # delete the combined datasets
                for dsname in dsname_tocombine:
                    rvs_datasets.pop(dsname)

        # vary the toy constraints
        self.toy_constraints.rvs(toy_parameters)

        # create the toy Experiment
        self.toy = Experiment(
            datasets=rvs_datasets, 
            parameters=self.parameters, 
            constraints=self.toy_constraints, 
            options=self.options,
            )  

        return self.toy    

    # has to allow for parallelization
    def toy_ts(
        self,
        toy_parameters: dict, # parameters and values needed to generate the toys
        profile_parameters: dict, # which parameters to fix and their value (rest are profiled)
        num: int = 1,
        seeds: np.array = None,   
        info: bool = False,     
    ) -> tuple[np.array, dict]:
        """
        Makes a number of toys and returns their test statistics.
        Having the seed be an array allows for different jobs producing toys on the same s-value to have different seed numbers

        parameters
            `dict` where keys are names of parameters to fix and values are the value that the parameter should be
            fixed to during profiling   
        info
            whether to return additional information about the toys (default: False)     
        """

        # if seeds aren't provided, we need to generate them ourselves so all toys aren't the same
        if seeds is None:
            seeds = np.random.randint(1e9, size=num)
        
        if len(seeds) != num:
            raise ValueError("Seeds must have same length as the number of toys!")
        
        if isinstance(profile_parameters, dict):
            profile_parameters = [profile_parameters]

        ts = np.zeros((len(profile_parameters), num))
        numerators = np.zeros((len(profile_parameters), num))
        denominators = np.zeros((len(profile_parameters), num))
        data_to_return = []
        paramvalues_to_return = []
        num_drawn = []
        for i in range(num):
            thistoy = self.make_toy(toy_parameters=toy_parameters, seed=seeds[i])
            for j in range(len(profile_parameters)):
                ts[j][i], denominators[j][i], numerators[j][i] = thistoy.ts(
                    profile_parameters=profile_parameters[j]
                )

            # TODO: add this info back in make_toy()
            
            # data_to_return.append(thistoy.toy_data_to_save)
            # paramvalues_to_return.append(thistoy.parameters_to_save)
            # num_drawn.append(thistoy.toy_num_drawn_to_save)

        if info:
            # Need to flatten the data_to_return in order to save it in h5py
            data_to_return_flat = (
                np.ones(
                    (len(data_to_return), np.nanmax([len(arr) for arr in data_to_return]))
                )
                * np.nan
            )
            for i, arr in enumerate(data_to_return):
                data_to_return_flat[i, : len(arr)] = arr

            num_drawn_to_return_flat = (
                np.ones((len(num_drawn), np.nanmax([len(arr) for arr in num_drawn])))
                * np.nan
            )
            for i, arr in enumerate(num_drawn):
                num_drawn_to_return_flat[i, : len(arr)] = arr

            info_to_return = {
                "data": data_to_return_flat,
                "paramvalues": paramvalues_to_return,
                "num_drawn": num_drawn_to_return_flat,
                "denominators": denominators,
                "numerators": numerators,
                }

            return (
                ts,
                info_to_return,
            )

        return (ts, {})

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
        Loads a config file or dict and converts `str` for some fields to the appropriate objects. Performs some
        error checking and sets defaults for missing fields where possible.

        Parameters
        ----------
        file : str | dict
            path to a config file or a config dictionary
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

        for item in ["options", "constraints", "combined_datasets"]:
            if item not in config:
                config[item] = {}

        # options

        # defaults
        options_defaults = {
            "backend"                   : "minuit"  ,   # "minuit" or "scipy"
            "iminuit_precision"         : 1e-10     ,
            "iminuit_strategy"          : 0         , 
            "iminuit_tolerance"         : 1e-5      ,
            "initial_guess"             : {"fcn": None, "module": None},  
            "minimizer_options"         : {}        ,   # dict of options to pass to the iminuit minimizer
            "num_cores"                 : 1         ,
            "num_toys"                  : 1000      ,
            "scan"                      : False     ,
            "scipy_minimizer"           : None      ,
            "seed_start"                : 0         ,
            "try_to_combine_datasets"   : False     ,
            "test_statistic"            : "t_mu"    ,   # "t_mu", "q_mu", "t_mu_tilde", or "q_mu_tilde"
            "use_grid_rounding"         : False     ,   # evaluate the test statistic on a parameter space grid after minimizing
            "use_log"                   : False     ,
            "use_user_gradient"         : False     ,
        }

        for key, val in options_defaults.items():
            if key not in config["options"]:
                config["options"][key] = val

        for option, optionval in config["options"].items():
            if optionval in ["none", "None"]:
                config["options"][option] = None
        
        if config["options"]["backend"] not in ["minuit", "scipy"]:
            raise NotImplementedError(
              "backend is not set to 'minuit' or 'scipy'"  
            )
        
        if not isinstance(config["options"]["minimizer_options"], dict):
            raise ValueError("options: minimizer_options must be a dict")

        if config["options"]["initial_guess"]["fcn"] is not None:
            config["options"]["initial_guess"] = Workspace.load_class(config["options"]["initial_guess"])

            if not issubclass(config["options"]["initial_guess"], Guess):
                raise TypeError(f"initial guess must inherit from 'Guess'")

            # instantiate guess class and set guess function
            config["options"]["initial_guess"] = config["options"]["initial_guess"]().guess

        if config["options"]["try_to_combine_datasets"]:
            if "combined_datasets" not in config:
                msg = (f"option 'try_to_combine_datasets' is True but `combined_datasets` is missing")
                raise KeyError(msg) 

        # get list of models and cost functions to import
        models = []
        costfunctions = set()
        for dsname, ds in config["datasets"].items():

            for item in ["model", "costfunction"]:
                if item not in ds:
                    msg = f"Dataset '{dsname} has no '{item}'"
                    raise KeyError(msg)
            
            if ds["model"] not in models:
                models.append(ds["model"])
            
            if "toy_model" not in ds:
                ds["toy_model"] = ds["model"]

            if ds["toy_model"] not in models:
                models.append(ds["toy_model"])
            
            if ds["costfunction"] not in ["ExtendedUnbinnedNLL", "UnbinnedNLL"]:
                msg = f"Dataset '{dsname}': only 'ExtendedUnbinnedNLL' or 'UnbinnedNLL' are \
                    supported as cost functions"
                raise NotImplementedError(msg)   

            costfunctions.add(ds["costfunction"])                 
            
            if config["options"]["try_to_combine_datasets"]:
                if (ds["combined_dataset"] not in config["combined_datasets"]):
                        msg = (f"Dataset `{dsname}` has `combined_dataset` `{ds['combined_dataset']}` but " 
                            + f"`combined_datasets` does not contain `{ds['combined_dataset']}`")
                        raise KeyError(msg)   
                elif (config["combined_datasets"][ds["combined_dataset"]]["model"] != ds["model"]):
                        msg = (f" Dataset `{dsname}` Model `{ds['model']['fcn']}` not the same as CombinedDataset "
                            + f"`{ds['combined_dataset']}` Model "
                            + f"`{config['combined_datasets'][ds['combined_dataset']]['model']['fcn']}`")
                        raise ValueError(msg)

            # set default after checking previous
            for dsname, ds in config["datasets"].items():
                if "combined_dataset" not in ds:
                    ds["combined_dataset"] = None

        # constraints
        for ctname, constraint in config["constraints"].items():
            if "parameters" not in constraint:
                msg = f"constraint `{ctname}` has no `parameters`"
                raise KeyError(msg)
            
            if "vary" not in constraint:
                constraint["vary"] = False

            if "covariance" in constraint and "uncertainty" in constraint:
                msg = f"constraint '{ctname}' has both 'covariance' and 'uncertainty'; this is ambiguous - use only one!"
                logging.error(msg)
                raise KeyError(msg)

            if "covariance" not in constraint and "uncertainty" not in constraint:
                msg = f"constraint '{ctname}' has neither 'covariance' nor 'uncertainty' - one (and only one) must be provided!"
                logging.error(msg)
                raise KeyError(msg)
                
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

            if len(constraint["parameters"]) != len(constraint["values"]):
                if len(constraint["values"]) == 1:
                    constraint["values"] = np.full(
                        len(constraint["parameters"]), constraint["values"]
                    )
                    msg = f"in constraint '{ctname}', assigning 1 provided value to all {len(constraint['parameters'])} 'parameters'"
                    logging.warning(msg)
                else:
                    msg = f"constraint '{ctname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['values'])} 'values'"
                    logging.error(msg)
                    raise ValueError(msg)

            # do some cleaning up of the config here
            if "uncertainty" in constraint:
                if len(constraint["uncertainty"]) > 1:
                    constraint["uncertainty"] = np.full(
                        len(constraint["parameters"]), constraint["uncertainty"]
                    )
                    msg = f"constraint '{ctname}' has {len(constraint['parameters'])} parameters but only 1 uncertainty - assuming this is constant uncertainty for each parameter"
                    logging.warning(msg)

                if len(constraint["uncertainty"]) != len(constraint["parameters"]):
                    msg = f"constraint '{ctname}' has {len(constraint['parameters'])} 'parameters' but {len(constraint['uncertainty'])} 'uncertainty' - should be same length or single uncertainty"
                    logging.error(msg)
                    raise ValueError(msg)

                # convert to covariance matrix so that we're always working with the same type of object
                constraint["covariance"] = np.diag(constraint["uncertainty"]) ** 2
                del constraint["uncertainty"]

                msg = f"constraint '{ctname}': converting provided 'uncertainty' to 'covariance'"
                logging.info(msg)

            else:  # we have the covariance matrix for this constraint
                if len(constraint["parameters"]) == 1:
                    msg = f"constraint '{ctname}' has one parameter but uses 'covariance' - taking this at face value"
                    logging.info(msg)

                if np.shape(constraint["covariance"]) != (
                    len(constraint["parameters"]),
                    len(constraint["parameters"]),
                ):
                    msg = f"constraint '{ctname}' has 'covariance' of shape {np.shape(constraint['covariance'])} but it should be shape {(len(constraint['parameters']), len(constraint['parameters']))}"
                    logging.error(msg)
                    raise ValueError(msg)

                if not np.allclose(
                    constraint["covariance"], np.asarray(constraint["covariance"]).T
                ):
                    msg = f"constraint '{ctname}' has non-symmetric 'covariance' matrix - this is not allowed."
                    logging.error(msg)
                    raise ValueError(msg)

                sigmas = np.sqrt(np.diag(np.asarray(constraint["covariance"])))
                cov = np.outer(sigmas, sigmas)
                corr = constraint["covariance"] / cov
                if not np.all(np.logical_or(np.abs(corr) < 1, np.isclose(corr, 1))):
                    msg = f"constraint '{ctname}' 'covariance' matrix does not seem to contain proper correlation matrix"
                    logging.error(msg)
                    raise ValueError(msg)
                    
        # combined datasets
        for cdsname, cds in config["combined_datasets"].items():
            for item in ["model", "costfunction"]:
                if item not in cds:
                    msg = f"combined_datasets '{cdsname}' has no '{item}"
                    raise KeyError(msg)

            if cds["model"] not in models:
                models.append(cds["model"])
            costfunctions.add(cds["costfunction"])

        # load models
        for model in models:
            modelclass = Workspace.load_class(model)

            if not issubclass(modelclass.__class__, Model):
                raise TypeError(f"model '{modelclass}' must inherit from 'Model'")

            for dsname, ds in config["datasets"].items():
                if ds["model"] == model:
                    ds["model"] = modelclass
                
                if ds["toy_model"] == model:
                    ds["toy_model"] = modelclass

            for cdsname, cds in config["combined_datasets"].items():
                if cds["model"] == model:
                    cds["model"] = modelclass

        # specific to iminuit
        for costfunctionname in costfunctions:
            costfunction = getattr(
                importlib.import_module("iminuit.cost"), costfunctionname
            )

            for dsname, ds in config["datasets"].items():
                if ds["costfunction"] == costfunctionname:
                    ds["costfunction"] = costfunction

            for cdsname, cds in config["combined_datasets"].items():
                if cds["costfunction"] == costfunctionname:
                    cds["costfunction"] = costfunction

        # parameters
        
        # convert any limits from string to fpython object
        # set defaults if options missing
        for par, pardict in config["parameters"].items():

            for item in ["limits", "physical_limits"]:
                if item not in pardict:
                    pardict[item] = [None, None]
            
            if "value" not in pardict:
                pardict["value"] = None

            for item in ["includeinfit", "fixed", "fix_if_no_data", "value_from_combine"]:
                if item not in pardict:
                    pardict[item] = False
            
            if "grid_rounding_num_decimals" not in pardict:
                pardict["grid_rounding_num_decimals"] = 128

            for item in ["limits", "physical_limits"]:
                if type(pardict[item]) is str:
                    pardict[item] = eval(pardict[item])
            
            # TODO: change this so that "domain" is passed by user and controls everything?
            pardict["domain"] = pardict["limits"]
            if pardict["domain"][0] == None:
                pardict["domain"][0] = -1*np.inf
            if pardict["domain"][1] == None:
                pardict["domain"][1] == np.inf

        return config

    @staticmethod
    def load_class(
        info: dict,
    ):
        if "fcn" not in info or "module" not in info:
            msg = f"missing 'module' or 'path' key when attempting to load {info}"
            logging.info(msg)

            raise KeyError(msg)
        
        try: 
            lib = importlib.import_module(info["module"])
            thisclass = getattr(lib, info["fcn"])

            msg = f"loaded '{info['fcn']}' from module '{info['module']}'"
            logging.info(msg)

            return thisclass
        except Exception as e:
            msg = f"tried to load '{info['fcn']}' from '{info['module']}' but not a module, try as a path"
            logging.info(msg)
            logging.info(e)

            try:
                spec = importlib.util.spec_from_file_location("fakemodule", info["module"])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                thisclass = getattr(module, info["fcn"])

                msg = f"loaded '{info['fcn']}' from path '{info['module']}'"
                logging.info(msg)

                return thisclass
            
            except Exception as e:
                msg = f"tried to load '{info['fcn']}' from '{info['module']}' but not a module or path - aborting"
                logging.info(msg)
                logging.info(e)
                pass

        raise KeyError(msg)
        
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