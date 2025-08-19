"""
Workspace class for freqfit that controls aspects of the statistical analysis.
"""
import importlib
import numpy as np
import yaml
from numba import njit
import multiprocessing as mp
import h5py
import os
from copy import deepcopy
from collections import defaultdict

from .dataset import Dataset, ToyDataset, CombinedDataset
from .parameters import Parameters
from .constraints import Constraints, ToyConstraints
from .experiment import Experiment
from .model import Model
from .guess import Guess

import logging

log = logging.getLogger(__name__)

SEED = 42
NUM_CORES = int(os.cpu_count() / 2)

class Workspace:
    def __init__(
        self,
        config: dict,
        jobid: int = 0,
    ) -> None:

        # set internal variables relating to toy generation
        self.jobid = jobid

        # load the config

        # load in the global options - defaults and error checking in load_config
        self.options = config["options"]
        self.out_path = self.options["out_path"]
        self.numcores = self.options["numcores"]
        self.overwrite_files = self.options["overwrite_files"]
        self.name = self.options["name"]
        self.numtoy = self.options["numtoy"]
        
        msg = f"setting backend to {config['options']['backend']}"
        logging.info(msg)

        if self.overwrite_files:
            msg = f"overwrite files set to {self.overwrite_files}"
            logging.warn(msg)

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
        self._datasets = deepcopy(datasets)

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
        set_numba_random_seed(seed) # numba holds RNG seeds in thread local storage, so set it up here

        # check that the user hasn't accidentally passed the full output of `profile`
        if "fixed" in toy_parameters.keys():
            raise KeyError("toy_parameters should be a dictionary parameter name : parameter value. Perhaps you meant to access the 'results' of a profile?")

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
            seed=seed,
            )  

        return self.toy    

    # has to allow for parallelization
    def toy_ts(
        self,
        toy_parameters: dict, # parameters and values needed to generate the toys
        profile_parameters: dict|list, # which parameters to fix and their value (rest are profiled)
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

            # TODO: add this info back in make_toy() or do not???
            if info:
                data = []
                nd = [0,0]
                paramvalues_to_return = []
                for ds in thistoy.datasets:
                    data.extend(thistoy.datasets[ds].data[:])
                    if hasattr(thistoy.datasets[ds], "num_drawn"):
                        nd[0] += thistoy.datasets[ds].num_drawn[0]
                        nd[1] += thistoy.datasets[ds].num_drawn[1]
                data_to_return.append(data)
                paramvalues_to_return.append(0)
                num_drawn.append(nd)

        # TODO: should this be removed ??? record only seeds and ts?
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
    
    def scan_ts(
        self, var_to_profile: str, var_values: np.array, profile_dict: dict = {}  # noqa:B006
    ) -> np.array:
        """
        Parameters
        ----------
        var_values
            The values of the variable to scan over
        profile_dict
            Other values of variables the user wants fixed during a scan

        Returns
        -------
        ts
            Value of the specified test statistic at the scanned values
        """
        # Create the arguments to multiprocess over
        args = [
            [{f"{var_to_profile}": float(xx), **profile_dict}] for xx in var_values
        ]

        with mp.Pool(self.numcores) as pool:
            ts = pool.starmap(self.experiment.ts, args)
        return ts
        
    def toy_ts_mp(
            self,
            parameters: dict,  # parameters and values needed to generate the toys
            profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
            num: int = 0,
            info: bool = False, 
        ):
            """
            Makes a number of toys and returns their test statistics. Multiprocessed
            """
            self.numtoy = num if (num!=0) else self.numtoy
            toys_per_core = np.full(self.numcores, self.numtoy // self.numcores)
            toys_per_core = np.insert(
                toys_per_core, len(toys_per_core), self.numtoy % self.numcores
            )

            # remove any cores with 0 toys
            index = np.argwhere(toys_per_core == 0)
            toys_per_core = np.delete(toys_per_core, index)

            # In order to ensure toys aren't correlated between experiments, use the experiment name to set the seed

            experiment_seed = 0

            for c in self.name:
                experiment_seed += ord(c)

            if experiment_seed > 2**31:
                raise ValueError(
                    "Experiment seed cannot be too large, try naming the experiment a smaller string."
                )

            # Pick the random seeds that we will pass to toys
            seeds = np.arange(
                experiment_seed + self.jobid * self.numtoy,
                experiment_seed + (self.jobid + 1) * self.numtoy,
            )
            # seeds *= 5000  # need to multiply this by a large number because if seed numbers differ by fewer than num of datasets, then adjacent toys will have the same energies pulled but in different datasets
            # # See line 115 in toys.py, thisseed = self.seed + i
            # # If you have more than 5000 datasets, I am sorry
            # if len(self.experiment.datasets.items()) > 5000:
            #     raise ValueError(
            #         "You need to change the spacing between seeds for completely uncorrelated toys."
            #     )

            if (seeds > 2**32).any():
                raise ValueError(
                    "Experiment seed cannot be too large, try multiplying the seeds by a smaller number."
                )
            seeds_per_toy = []

            j = 0
            for i, num in enumerate(toys_per_core):
                seeds_per_toy.append(seeds[j : j + num])
                j = j + num

            args = [
                [parameters, profile_parameters, num_toy, seeds_per_toy[i], info]
                for i, num_toy in enumerate(toys_per_core)
            ]  # give each core multiple MCs

            with mp.Pool(self.numcores) as pool:
                return_args = pool.starmap(self.toy_ts, args)


            # TODO: fix this up to get correct printout, esp. in 2D grid case
            if info:  
                ts = np.array([item[0][:] for item in return_args])
                data_to_return = [item for _, val in return_args for item in val["data"]]
                # data_to_return is a jagged list, each element is a 2d-array filled it nans
                # First, find the maximum length of array we will need to pad to
                maxlen = np.amax([len(arr) for arr in data_to_return])
                data_flattened = [e for arr in data_to_return for e in arr]

                # # Need to flatten the data_to_return in order to save it in h5py
                # data_to_return_flat = np.ones((len(data_flattened), maxlen)) * np.nan
                # for i, arr in enumerate(data_flattened):
                #     data_to_return_flat[i, : len(arr)] = arr
                data_to_return_flat = data_flattened
                return (
                    np.hstack(ts), {"data": data_to_return_flat}
                )
                # ts = [arr[0] for arr in return_args]
                # data_to_return = [arr[1] for arr in return_args]
                # nuisance_to_return = [arr[2] for arr in return_args]
                # num_drawn_to_return = [arr[3] for arr in return_args]
                # ts_denom = [arr[4] for arr in return_args]
                # ts_num = [arr[5] for arr in return_args]
                # # data_to_return is a jagged list, each element is a 2d-array filled it nans
                # # First, find the maximum length of array we will need to pad to
                # maxlen = np.amax([len(arr[0]) for arr in data_to_return])
                # data_flattened = [e for arr in data_to_return for e in arr]

                # # Need to flatten the data_to_return in order to save it in h5py
                # data_to_return_flat = np.ones((len(data_flattened), maxlen)) * np.nan
                # for i, arr in enumerate(data_flattened):
                #     data_to_return_flat[i, : len(arr)] = arr

                # maxlen = np.amax([len(arr[0]) for arr in num_drawn_to_return])
                # num_drawn_flattened = [e for arr in num_drawn_to_return for e in arr]

                # # Need to flatten the data_to_return in order to save it in h5py
                # num_drawn_to_return_flat = np.ones((len(num_drawn_flattened), maxlen)) * np.nan
                # for i, arr in enumerate(num_drawn_flattened):
                #     num_drawn_to_return_flat[i, : len(arr)] = arr
            else:
                ts = np.array([item[0][:] for item in return_args])
                return (
                    np.hstack(ts), {}
                )

    def run_and_save_toys(
        self,
        toy_generation_profile_dict: dict, 
        toy_test_profile_dict: dict|list,
        toy_generation_profile_dict_override: dict = {}, # noqa:B006
        toy_pars_override: dict = {}, # noqa:B006
        overwrite_files: bool = None,
        info: bool = False 
    ):
        """
        Generate toys at a hypothesis and test against a (potentially different) hypothesis.

        Parameters
        ----------
        toy_generation_profile_dict
            A dictionary of values we want to fix during all of the profiles that generate the parameters to seed the toys
        
        toy_test_profile_dict
            A dicitonary of values we want to fix during the profile when computing the test statistic for a toy
        
        toy_generation_profile_dict_override
            If a dict is passed, generate toys at these values instead of the ones profiled out during toy_generation_profile_dict
        
        toy_pars_override
            If provided, use these parameters for toy generation, skipping any profiling
        
        overwrite_files
            whether to overwrite result files if found, uses global option of SetLimit as default
        
        info
            If false, save only the test statistics. If true, save lots of information
        """
        # handle the output file
        if overwrite_files is None:
            overwrite_files = self.overwrite_files

        filename = (
            self.out_path
            + f"/{list(toy_generation_profile_dict.values())}_{self.jobid}.h5"
        )

        if os.path.exists(filename) and not overwrite_files:
            msg = f"file {filename} exists - use option `overwrite_files` to overwrite"
            raise RuntimeError(msg)
        
        # First we need to pass the parameter values to generate the toys at: either profile out the variable we are scanning, or user supplied
        if toy_pars_override:
            toypars = toy_pars_override
        else:
            toypars = self.experiment.profile(
                toy_generation_profile_dict
            )["values"]
        
        # override any of the toypars if we need to 
        if toy_generation_profile_dict_override:
            for key in toy_generation_profile_dict_override:
                if key in toypars.keys:
                    toypars[key] = toy_generation_profile_dict_override[key]

        # run the toys now
        (
            toyts,
            info_dict
        ) = self.toy_ts_mp(
            toypars,
            toy_test_profile_dict,
            num=self.numtoy,
            info=info
        )

       # Now, save the toys to a file

        if overwrite_files and os.path.exists(filename):
            msg = f"overwriting existing file {filename}"
            logging.warning(msg)
            os.remove(filename)

        f = h5py.File(filename, "a")

        if isinstance(toy_test_profile_dict, list):
            new_dict = defaultdict(list)
            for d in toy_test_profile_dict:
                for k, v in d.items():
                    new_dict[k].append(v)
            toy_test_profile_dict  = new_dict

        dset = f.create_dataset(
            "test_parameters_names", data=list(toy_test_profile_dict.keys())
        )
        dset = f.create_dataset(
            "test_parameters_values", data=list(toy_test_profile_dict.values())
        )

        dset = f.create_dataset("ts", data=toyts)
        dset.attrs.update(toy_generation_profile_dict)

        if info:
            # raise NotImplementedError("not implemented, check back soon.")
            # dset = f.create_dataset("ts_denom", data=toyts_denom)
            # dset = f.create_dataset("ts_num", data=toyts_num)
            # dset = f.create_dataset("s", data=scan_point)
            dset = f.create_dataset("Es", data=info_dict["data"])
            # # dset = f.create_dataset("nuisance", data=nuisance)
            # dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)
            # dset = f.create_dataset("seed", data=seeds_to_save)

        f.close()

        return None
    
    def run_hypothesis_test(
        self,
        poi_name: str,
        poi_value: float,        
        overwrite_files: bool = None,
        info: bool = False 
    ):
        """
        Perform a hypothesis test for one parameter of interest. Toys are generated using the poi_value and tested against the poi_value
        Parameters
        ----------
        poi_name
            Name of the parameter of interest
        poi_value
            Value at which to generate and test toys

        """
        self.run_and_save_toys({poi_name: poi_value}, {poi_name: poi_value}, overwrite_files=overwrite_files, info=info)
        return None
    
    def run_exclusion(
        self,
        poi_name: str,
        poi_values: np.array,
        overwrite_files: bool = None,
        info: bool = False 
    ):
        """
        Perform an exclusion test for one parameter of interest. Toys are generated at 0 for the poi and tested against the poi_value
        Parameters
        ----------
        poi_name
            Name of the parameter of interest
        poi_value
            Value at which to test toys

        """
        # Add 0 to the scan points if it is not there
        if 0.0 not in poi_values:
            poi_values = np.insert(poi_values, 0, 0.0)
        self.run_and_save_toys({poi_name: 0.0}, [{poi_name: scan_point} for scan_point in poi_values], overwrite_files=overwrite_files, info=info)
        return None
    
    def run_discovery(
        self,
        poi_name: str,
        poi_value: np.array,
        overwrite_files: bool = None,
        info: bool = False 
    ):
        """
        Perform a discovery test for one parameter of interest. Toys are generated at poi_value for the poi and tested against 0.0
        Parameters
        ----------
        poi_name
            Name of the parameter of interest
        poi_value
            Value at which to generate toys

        """
        self.run_and_save_toys({poi_name: poi_value}, {poi_name: 0.0}, overwrite_files=overwrite_files, info=info)
        return None
    
    def run_coverage(
        self,
        poi_name: str,
        poi_value: float,
        coverage_values: np.array,
        overwrite_files: bool = None,
        info: bool = False 
    ):
        """
        Check the coverage. Toys are generated at poi_value for the poi and tested against all the coverage_values
        Parameters
        ----------
        poi_name
            Name of the parameter of interest
        poi_value
            Value at which to generate toys
        coverage_values
            Values at which to test toys
        """
        # check if the profile grid has the scan point in it, otherwise fail gracefully
        if poi_value not in coverage_values:
            raise ValueError(
                f"{poi_value} not in user specified grid {coverage_values}!"
            )
        self.run_and_save_toys({poi_name: poi_value}, [{poi_name: scan_point} for scan_point in coverage_values], overwrite_files=overwrite_files, info=info)
        return None
    
    def run_mismodel(
        self,
        poi_name: str,
        poi_value: float,
        mismodel_pars: dict,
        coverage_values: np.array,
        overwrite_files: bool = None,
        info: bool = False 
    ):
        """
        Compute test statistics under mismodeling. Toys are generated at poi_value and with mismodel_pars, but mismodel_pars are allowed to float during testing at poi_value
        Parameters
        ----------
        poi_name
            Name of the parameter of interest
        poi_value
            Value at which to generate and test toys
        mismodel_pars
            Additional parameters to fix during profiling for toy generation, but floated during testing
        """
        if poi_value not in coverage_values:
            raise ValueError(
                f"{poi_value} not in user specified grid {coverage_values}!"
            )
        self.run_and_save_toys({poi_name: poi_value, **mismodel_pars}, [{poi_name: scan_point} for scan_point in coverage_values], overwrite_files=overwrite_files, info=info)
        return None

    def run_joint_profile(
        self,
        poi_name: str,
        poi_value: float,
        joint_profile_pars: dict,
        overwrite_files: bool = None,
        info: bool = False 
    ):
        """
        Compute test statistics for multiple poi.  Toys are generated at poi_value and with joint_profile_pars, and are also tested agains poi_value and joint_profile_pars
        Parameters
        ----------
        poi_name
            Name of the parameter of interest
        poi_value
            Value at which to generate and test toys
        mismodel_pars
            Additional parameters to fix during profiling for toy generation and testing
        """
        self.run_and_save_toys({poi_name: poi_value, **joint_profile_pars}, {poi_name: poi_value, **joint_profile_pars}, overwrite_files=overwrite_files, info=info)
        return None

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
            "out_path"                  : "."       ,
            "numcores"                  : NUM_CORES ,
            "overwrite_files"           : False     ,
            "name"                      : ""        ,
            "numtoy"                    : 0         ,
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
                log.debug("`toy_model` not provided, using dataset model instead")
                ds["toy_model"] = ds["model"]

            if ds["toy_model"] not in models:
                models.append(ds["toy_model"])
            
            if ds["costfunction"] not in ["ExtendedUnbinnedNLL", "UnbinnedNLL"]:
                msg = f"Dataset '{dsname}': only 'ExtendedUnbinnedNLL' or 'UnbinnedNLL' are \
                    supported as cost functions"
                raise NotImplementedError(msg)   

            costfunctions.add(ds["costfunction"])                 
                    
            if config["options"]["try_to_combine_datasets"] and ds["combined_dataset"] is not None:
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

            if "limits" not in pardict:
                pardict["limits"] = [None, None]

            if "physical_limits" not in pardict:
                pardict["physical_limits"] = None
            
            if "value" not in pardict:
                pardict["value"] = None

            for item in ["includeinfit", "fixed", "fix_if_no_data", "value_from_combine", "poi"]:
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
                pardict["domain"][1] = np.inf
                
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
    
@njit
def set_numba_random_seed(seed):
    np.random.seed(seed)