"""
This class inherits from `Experiment`, and also holds the name of the variable to profile
"""

import logging
import multiprocessing as mp
import os

import h5py
import numpy as np
from scipy.special import erfcinv

from freqfit.experiment import Experiment

NUM_CORES = 20  # TODO: change this to an environment variable, or something that detects available cores
SEED = 42

log = logging.getLogger(__name__)


class SetLimit(Experiment):
    def __init__(
        self,
        config: dict,
        jobid: int = 0,
        numtoy: int = 0,
        out_path: str = ".",
        name: str = "",
        overwrite_files: bool = False,
        numcores: int = NUM_CORES,
        profile_grid: np.array = None,
    ) -> None:
        """
        This class inherits from `Experiment`, and also holds the name of the variable to profile
        """
        super().__init__(config, name)

        self.test_statistics = (
            None  # maybe we want to store the test statistics internally?
        )
        self.var_to_profile = None
        self.jobid = jobid
        self.numtoy = numtoy
        self.out_path = out_path
        self.numcores = numcores
        self.overwrite_files = overwrite_files
        self.profile_grid = profile_grid

    def set_var_to_profile(self, var_to_profile: str):
        """
        Parameters
        ----------
        var_to_profile
            string -- the variable we are going to scan over to compute test-statistics at
        """

        if var_to_profile not in self.fitparameters:
            msg = f"variable '{var_to_profile}' not found in fit parameters"
            logging.error(msg)
            raise ValueError(msg)
        self.var_to_profile = var_to_profile

    def set_profile_grid(self, profile_grid: np.array):
        """
        Parameters
        ----------
        scan_grid
            values of the variable we are going to scan over to compute test-statistics at
        """

        self.profile_grid = profile_grid

    def wilks_ts_crit(self, CL: float) -> float:
        """
        Parameters
        ----------
        CL
            The confidence level at which to compute the critical value, i.e. 0.9 for 90%

        Returns
        -------
        t_crit
            The critical value of the test statistic

        Notes
        -----
        Using Wilks' approximation, we assume the test statistic has a chi-square PDF with 1 degree of freedom.
        We compute the critical value of the test statistic for the given confidence level.
        This is independent of the parameter we are scanning over.
        """
        alpha = 1 - CL  # convert to a significance level, an alpha, one-sided

        return 2 * erfcinv(alpha) ** 2

    def scan_ts(
        self, var_values: np.array, profile_dict: dict = {}  # noqa:B006
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
            [{f"{self.var_to_profile}": float(xx), **profile_dict}] for xx in var_values
        ]

        with mp.Pool(self.numcores) as pool:
            ts = pool.starmap(self.ts, args)
        return ts

    def toy_ts_mp(
        self,
        parameters: dict,  # parameters and values needed to generate the toys
        profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
        num: int = 1,
    ):
        """
        Makes a number of toys and returns their test statistics. Multiprocessed
        """
        x = np.arange(0, num)
        toys_per_core = np.full(self.numcores, num // self.numcores)
        toys_per_core = np.insert(
            toys_per_core, len(toys_per_core), num % self.numcores
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
        seeds *= 5000  # need to multiply this by a large number because if seed numbers differ by fewer than num of datasets, then adjacent toys will have the same energies pulled but in different datasets
        # See line 115 in toys.py, thisseed = self.seed + i
        # If you have more than 5000 datasets, I am sorry
        if len(self.datasets.items()) > 5000:
            raise ValueError(
                "You need to change the spacing between seeds for completely uncorrelated toys."
            )

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
            [parameters, profile_parameters, num_toy, seeds_per_toy[i]]
            for i, num_toy in enumerate(toys_per_core)
        ]  # give each core multiple MCs

        with mp.Pool(self.numcores) as pool:
            return_args = pool.starmap(self.toy_ts, args)

        ts = [arr[0] for arr in return_args]
        data_to_return = [arr[1] for arr in return_args]
        nuisance_to_return = [arr[2] for arr in return_args]
        num_drawn_to_return = [arr[3] for arr in return_args]
        ts_denom = [arr[4] for arr in return_args]
        ts_num = [arr[5] for arr in return_args]

        # data_to_return is a jagged list, each element is a 2d-array filled it nans
        # First, find the maximum length of array we will need to pad to
        maxlen = np.amax([len(arr[0]) for arr in data_to_return])
        data_flattened = [e for arr in data_to_return for e in arr]

        # Need to flatten the data_to_return in order to save it in h5py
        data_to_return_flat = np.ones((len(data_flattened), maxlen)) * np.nan
        for i, arr in enumerate(data_flattened):
            data_to_return_flat[i, : len(arr)] = arr

        maxlen = np.amax([len(arr[0]) for arr in num_drawn_to_return])
        num_drawn_flattened = [e for arr in num_drawn_to_return for e in arr]

        # Need to flatten the data_to_return in order to save it in h5py
        num_drawn_to_return_flat = np.ones((len(num_drawn_flattened), maxlen)) * np.nan
        for i, arr in enumerate(num_drawn_flattened):
            num_drawn_to_return_flat[i, : len(arr)] = arr

        return (
            np.hstack(ts),
            data_to_return_flat,
            np.vstack(nuisance_to_return),
            num_drawn_to_return_flat,
            seeds,
            np.hstack(ts_denom),
            np.hstack(ts_num),
        )

    def run_and_save_toys(
        self,
        scan_point,
        profile_dict: dict = {},  # noqa:B006
        scan_point_override=None,
        overwrite_files: bool = None,
        compute_conditional: bool = False,
        save_only_ts: bool = True,
        compute_coverage: bool = False,
        toy_pars_override: dict = None,  # noqa:B006
    ):
        """
        Runs toys at specified scan point.
        This can be used to scan on a hypercube if `profile_dict` is passed and `compute_conditional` is false,
        as `profile_dict` is used to fix parameters when generating the toys as well as to fix parameters during the profile.

        Parameters
        ----------
        profile_dict
            An optional dictionary of values we want to fix during all of the profiles

        overwrite_files
            whether to overwrite result files if found, uses global option of SetLimit as default

        compute_conditional
            If true, `profile_dict` is passed, then toys are generated at those values, but allowed to float during the profile fit

        save_only_ts
            Save only the test statistics

        compute_coverage
            If true, use `self.profile_grid` to scan all points on the profile grid, not just scan point. Useful for determining coverage
            by computing limits toy-by-toy and comparing limit to `scan_point`

        toy_pars_override
            If a dict is passed, generate toys at these values instead of the ones profiled out at `scan_point`
        """

        if overwrite_files is None:
            overwrite_files = self.overwrite_files

        filename = ""
        if not profile_dict:
            filename = self.out_path + f"/{scan_point}_{self.jobid}.h5"
        else:
            filename = (
                self.out_path
                + f"/{scan_point}_{list(profile_dict.values())}_{self.jobid}.h5"
            )

        if os.path.exists(filename) and not overwrite_files:
            msg = f"file {filename} exists - use option `overwrite_files` to overwrite"
            raise RuntimeError(msg)

        # First we need to pass the parameter values to generate the toys at: either profile out the variable we are scanning, or user supplied
        if toy_pars_override:
            toypars = toy_pars_override
        else:
            toypars = self.profile(
                {f"{self.var_to_profile}": scan_point, **profile_dict}
            )["values"]

        if scan_point_override is not None:
            toypars[f"{self.var_to_profile}"] = scan_point_override
        else:
            toypars[
                f"{self.var_to_profile}"
            ] = scan_point  # override here if we want to compare the power of the toy ts to another scan_point

        # Now we can run the toys
        if compute_conditional and profile_dict:
            # don't fix the profile_dict points during the fits
            # compute the test statistic at just the given point by default

            (
                toyts,
                data,
                nuisance,
                num_drawn,
                seeds_to_save,
                toyts_denom,
                toyts_num,
            ) = self.toy_ts_mp(
                toypars,
                {f"{self.var_to_profile}": scan_point},
                num=self.numtoy,
            )
        else:
            # Scan a toy over all points if asked
            if compute_coverage and self.profile_grid is not None:
                # check if the profile grid has the scan point in it, otherwise fail gracefully
                if scan_point not in self.profile_grid:
                    raise ValueError(
                        f"{scan_point} not in user specified grid {self.profile_grid}!"
                    )
                (
                    toyts,
                    data,
                    nuisance,
                    num_drawn,
                    seeds_to_save,
                    toyts_denom,
                    toyts_num,
                ) = self.toy_ts_mp(
                    toypars,
                    [
                        {f"{self.var_to_profile}": one_scan_point}
                        for one_scan_point in self.profile_grid
                    ],
                    num=self.numtoy,
                )
            # Default to scanning over just the one point specified
            else:
                (
                    toyts,
                    data,
                    nuisance,
                    num_drawn,
                    seeds_to_save,
                    toyts_denom,
                    toyts_num,
                ) = self.toy_ts_mp(
                    toypars,
                    {f"{self.var_to_profile}": scan_point, **profile_dict},
                    num=self.numtoy,
                )

        # Now, save the toys to a file

        if overwrite_files and os.path.exists(filename):
            msg = f"overwriting existing file {filename}"
            logging.warning(msg)
            os.remove(filename)

        f = h5py.File(filename, "a")

        if profile_dict:
            dset = f.create_dataset(
                "profile_parameters_names", data=list(profile_dict.keys())
            )
            dset = f.create_dataset(
                "profile_parameters_values", data=list(profile_dict.values())
            )

        dset = f.create_dataset("ts", data=toyts)
        dset = f.create_dataset("s", data=scan_point)

        if not save_only_ts:
            dset = f.create_dataset("ts_denom", data=toyts_denom)
            dset = f.create_dataset("ts_num", data=toyts_num)
            dset = f.create_dataset("s", data=scan_point)
            dset = f.create_dataset("Es", data=data)
            # dset = f.create_dataset("nuisance", data=nuisance)
            dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)
            dset = f.create_dataset("seed", data=seeds_to_save)

        f.close()

        return None

    def run_and_save_brazil(
        self,
        scan_points: list,
        profile_dict: dict = {},  # noqa:B006
        overwrite_files: bool = None,
        compute_conditional: bool = False,
        save_only_ts: bool = True,
        toy_pars_override: dict = None,
    ) -> None:
        """
        Runs toys at 0 signal rate and computes the test statistic for different signal hypotheses.
        If compute_conditional is True and profile_dict is passed, then toys are generated at `profile_dict`,
        but are allowed to float in the fits.

        If compute_conditional is passed as False and a `profile_dict` is passed in order to scan a hypercube, it will work.
        However, this will be slow, and the recommended job submission differs. Use `run_and_save_brazil_with_profile_parameters` instead.

        save_only_ts
            Save only the test statistics

        toy_pars_override
            If a dict is passed, generate toys at these values instead of the ones profiled out at `scan_point`
        """

        if profile_dict and not compute_conditional:
            log.warning(
                "Job submission for hypercube scan using `profile_dict` and `compute_conditional` as False is not recommended.\
                 Use `run_and_save_brazil_with_profile_parameters` instead "
            )

        if overwrite_files is None:
            overwrite_files = self.overwrite_files

        filename = ""
        if not profile_dict:
            filename = self.out_path + f"/0_{self.jobid}.h5"
        else:
            filename = (
                self.out_path + f"/0_{list(profile_dict.values())}_{self.jobid}.h5"
            )

        if os.path.exists(filename) and not overwrite_files:
            msg = f"file {filename} exists - use option `overwrite_files` to overwrite"
            raise RuntimeError(msg)

        # First we need to pass the parameter values to generate the toys at:  either profile out the variable we are scanning at 0 signal rate, or user supplied
        if toy_pars_override:
            toypars = toy_pars_override
        else:
            toypars = self.profile({f"{self.var_to_profile}": 0.0, **profile_dict})[
                "values"
            ]

        # Add 0 to the scan points if it is not there
        if 0.0 not in scan_points:
            scan_points = np.insert(scan_points, 0, 0.0)

        # Now we can run the toys
        if compute_conditional and profile_dict:
            (
                toyts,
                data,
                nuisance,
                num_drawn,
                seeds_to_save,
                toyts_denom,
                toyts_num,
            ) = self.toy_ts_mp(
                toypars,
                [{f"{self.var_to_profile}": scan_point} for scan_point in scan_points],
                num=self.numtoy,
            )
        else:
            (
                toyts,
                data,
                nuisance,
                num_drawn,
                seeds_to_save,
                toyts_denom,
                toyts_num,
            ) = self.toy_ts_mp(
                toypars,
                [
                    {f"{self.var_to_profile}": scan_point, **profile_dict}
                    for scan_point in scan_points
                ],
                num=self.numtoy,
            )

        # Now, save the toys to a file
        if overwrite_files and os.path.exists(filename):
            msg = f"overwriting existing file {filename}"
            logging.warning(msg)
            os.remove(filename)

        f = h5py.File(filename, "a")
        dset = f.create_dataset("ts", data=toyts)
        dset = f.create_dataset("s", data=scan_points)

        if not save_only_ts:
            dset = f.create_dataset("ts_num", data=toyts_num)
            dset = f.create_dataset("ts_denom", data=toyts_denom)
            dset = f.create_dataset("Es", data=data)
            dset = f.create_dataset("nuisance", data=nuisance)
            dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)
            dset = f.create_dataset("seed", data=seeds_to_save)

        f.close()

        return None

    def run_and_save_brazil_with_profile_parameters(
        self,
        scan_point: float,
        profile_dict: dict = {},  # noqa:B006
        overwrite_files: bool = None,
        compute_conditional: bool = False,
        save_only_ts: bool = True,
    ) -> None:
        """
        Runs toys at 0 signal rate and computes the test statistic for different signal hypotheses
        If we are running with profile_dict parameters in order to scan a hypercube,
        the optimal job submission differs from the above and is more similar to run_and_save_toys

        TODO: refactor this with `run_and_save_brazil`
        """

        if overwrite_files is None:
            overwrite_files = self.overwrite_files

        filename = ""
        if not profile_dict:
            filename = self.out_path + f"/0_{self.jobid}.h5"
        else:
            filename = (
                self.out_path + f"/0_{list(profile_dict.values())}_{self.jobid}.h5"
            )

        if os.path.exists(filename) and not overwrite_files:
            msg = f"file {filename} exists - use option `overwrite_files` to overwrite"
            raise RuntimeError(msg)

        # First we need to profile out the variable we are scanning at 0 signal rate
        toypars = self.profile({f"{self.var_to_profile}": 0.0, **profile_dict})[
            "values"
        ]

        # Now we can run the toys
        if compute_conditional and profile_dict:
            # don't fix the profile_dict points during the fits
            (
                toyts,
                data,
                nuisance,
                num_drawn,
                seeds_to_save,
                toyts_denom,
                toyts_num,
            ) = self.toy_ts_mp(
                toypars,
                {f"{self.var_to_profile}": scan_point},
                num=self.numtoy,
            )
        else:
            (
                toyts,
                data,
                nuisance,
                num_drawn,
                seeds_to_save,
                toyts_denom,
                toyts_num,
            ) = self.toy_ts_mp(
                toypars,
                {f"{self.var_to_profile}": scan_point, **profile_dict},
                num=self.numtoy,
            )

        # Now, save the toys to a file
        if overwrite_files and os.path.exists(filename):
            msg = f"overwriting existing file {filename}"
            logging.warning(msg)
            os.remove(filename)

        f = h5py.File(filename, "a")
        dset = f.create_dataset("ts", data=toyts)
        dset = f.create_dataset("seed", data=seeds_to_save)
        dset = f.create_dataset(
            "profile_parameters_names", data=list(profile_dict.keys())
        )
        dset = f.create_dataset(
            "profile_parameters_values", data=list(profile_dict.values())
        )
        dset = f.create_dataset("s", data=scan_point)

        if not save_only_ts:
            dset = f.create_dataset("ts_num", data=toyts_num)
            dset = f.create_dataset("ts_denom", data=toyts_denom)
            dset = f.create_dataset("Es", data=data)
            dset = f.create_dataset("nuisance", data=nuisance)
            dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)

        f.close()

        return None

    def run_and_save_joint_profile_toys(
        self,
        scan_points: np.array,
        profile_dict: dict = {},  # noqa:B006
        scan_point_override=None,
        overwrite_files: bool = None,
        compute_conditional: bool = False,
        save_only_ts: bool = True,
    ):
        """
        Runs toys at list of scan points.
        This can be used to scan on a hypercube if `profile_dict` is passed and `compute_conditional` is false,
        as `profile_dict` is used to fix parameters when generating the toys as well as to fix parameters during the profile.
        This differs from `run_and_save_toys` because `scan_points` is now an array.

        TODO: combine this and `run_and_save_toys`

        Parameters
        ----------
        profile_dict
            An optional dictionary of values we want to fix during all of the profiles

        overwrite_files
            whether to overwrite result files if found, uses global option of SetLimit as default

        compute_conditional
            If true, `profile_dict` is passed, then toys are generated at those values, but allowed to float during the profile fit

        save_only_ts
            Save only the test statistics
        """

        if overwrite_files is None:
            overwrite_files = self.overwrite_files

        for scan_point in scan_points:
            filename = ""
            if not profile_dict:
                filename = self.out_path + f"/{scan_point}_{self.jobid}.h5"
            else:
                filename = (
                    self.out_path
                    + f"/{scan_point}_{list(profile_dict.values())}_{self.jobid}.h5"
                )

            if os.path.exists(filename) and not overwrite_files:
                msg = f"file {filename} exists - use option `overwrite_files` to overwrite"
                raise RuntimeError(msg)

            # First we need to profile out the variable we are scanning
            toypars = self.profile(
                {f"{self.var_to_profile}": scan_point, **profile_dict}
            )["values"]

            if scan_point_override is not None:
                toypars[f"{self.var_to_profile}"] = scan_point_override
            else:
                toypars[
                    f"{self.var_to_profile}"
                ] = scan_point  # override here if we want to compare the power of the toy ts to another scan_point

            # Now we can run the toys
            if compute_conditional and profile_dict:
                # don't fix the profile_dict points during the fits
                (
                    toyts,
                    data,
                    nuisance,
                    num_drawn,
                    seeds_to_save,
                    toyts_denom,
                    toyts_num,
                ) = self.toy_ts_mp(
                    toypars,
                    {f"{self.var_to_profile}": scan_point},
                    num=self.numtoy,
                )
            else:
                (
                    toyts,
                    data,
                    nuisance,
                    num_drawn,
                    seeds_to_save,
                    toyts_denom,
                    toyts_num,
                ) = self.toy_ts_mp(
                    toypars,
                    {f"{self.var_to_profile}": scan_point, **profile_dict},
                    num=self.numtoy,
                )

            # Now, save the toys to a file

            if overwrite_files and os.path.exists(filename):
                msg = f"overwriting existing file {filename}"
                logging.warning(msg)
                os.remove(filename)

            f = h5py.File(filename, "a")

            if profile_dict:
                dset = f.create_dataset(
                    "profile_parameters_names", data=list(profile_dict.keys())
                )
                dset = f.create_dataset(
                    "profile_parameters_values", data=list(profile_dict.values())
                )

            dset = f.create_dataset("ts", data=toyts)
            dset = f.create_dataset("s", data=scan_point)

            if not save_only_ts:
                dset = f.create_dataset("ts_denom", data=toyts_denom)
                dset = f.create_dataset("ts_num", data=toyts_num)
                dset = f.create_dataset("Es", data=data)
                # dset = f.create_dataset("nuisance", data=nuisance)
                dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)
                dset = f.create_dataset("seed", data=seeds_to_save)

            f.close()

        return None
