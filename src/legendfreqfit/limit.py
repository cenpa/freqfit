"""
This class inherits from `Experiment`, and also holds the name of the variable to profile
"""

import logging
import multiprocessing as mp

import h5py
import numpy as np
import os
from scipy.special import erfcinv

from legendfreqfit.experiment import Experiment
from legendfreqfit.models import constants

NUM_CORES = 30  # TODO: change this to an environment variable, or something that detects available cores
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
        self.numcores = NUM_CORES  # default

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

    def find_crossing(
        self,
        scanned_var: np.array,
        ts: np.array,
        t_crits: np.array,
        interpolation_mode: str = "l",
    ):
        """
        Parameters
        ----------
        scanned_var
            Values of the variable that is being scanned
        ts
            Values of the test statistic at the scanned values
        t_crits
            The critical value or values of the test statistic
        interpolation_mode
            The mode in which to interpolate the crossing between ts and t_crits. "i" for index before crossing, "l" for linear, and "q" for quadratic are supported

        Notes
        -----
        It is important that ts and t_crits are on the same grid!
        This can handle two crossings in case of discovery!
        """
        # First check if t_crits is a scalar, then turn it into an array of the length of ts
        if isinstance(t_crits, float) or isinstance(t_crits, int):
            t_crits = np.full(len(ts), t_crits)

        diff = t_crits - ts  # now we can find some zeros!
        crossing_idxs = []
        for i in range(0, len(diff) - 1, 1):
            if diff[i] <= 0 < diff[i + 1]:
                crossing_idxs.append(i)
            if diff[i] > 0 >= diff[i + 1]:
                crossing_idxs.append(i)

        # If index before crossing interpolation do the following:
        if interpolation_mode == "i":
            crossing_points = scanned_var[crossing_idxs]

        # If linear mode, treat the t_crits as linear interpolation, as well as the ts
        if interpolation_mode == "l":
            crossing_points = []
            for i in crossing_idxs:
                alpha = (ts[i + 1] - ts[i]) / (scanned_var[i + 1] - scanned_var[i])
                beta = (t_crits[i + 1] - t_crits[i]) / (
                    scanned_var[i + 1] - scanned_var[i]
                )
                intersection = (
                    ts[i] - t_crits[i] + (beta - alpha) * scanned_var[i]
                ) / (beta - alpha)
                crossing_points.append(intersection)

        if interpolation_mode == "q":
            raise NotImplementedError("Not yet implemented! Sorry!")

        return crossing_points

    def create_fine_scan(self, s_crit, num_points):
        """
        Parameters
        ----------
        s_crit
            The s-value at the critical value of the test statistic found by Wilks' approximation
        num_points
            The number of points to symmetrically create a grid from
        """
        M76 = constants.M76
        NA = constants.NA

        # Scan a whole order of magnitude quickly to see if we need to rescan in case Wilks' isn't good
        T_est = 1 / (M76 * s_crit / (np.log(2) * NA))
        T_hi = 2 * T_est
        T_lo = T_est / 2

        S_lo = 1 / (M76 * T_hi / (np.log(2) * NA))
        S_hi = 1 / (M76 * T_lo / (np.log(2) * NA))

        #         lo_range = np.linspace(S_lo, s_crit, num_points//2 + num_points%2, endpoint=False)
        #         hi_range = np.linspace(s_crit, S_hi, num_points//2)

        #         fine = np.append(lo_range, hi_range)

        # Actually, assume we are within 20% of the correct limit and rescan based on this. Linear in S-space, 1/T...
        # This is good for setting a limit but not for making a plot with
        lo_range = np.linspace(
            s_crit * 0.8, s_crit, num_points // 2 + num_points % 2, endpoint=False
        )
        hi_range = np.linspace(s_crit, s_crit * 1.2, num_points // 2)

        fine = np.append(lo_range, hi_range)

        return fine

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
        overwrite_files: bool = False,
    ):
        """
        Runs toys at specified scan point and returns the critical value of the test statistic and its uncertainty

        Parameters
        ----------
        profile_dict
            An optional dictionary of values we want to fix during all of the profiles
        
        overwrite_files: bool, optional (default: False)
            whether to overwrite result files if found

        """
        # First we need to profile out the variable we are scanning
        toypars = self.profile({f"{self.var_to_profile}": scan_point, **profile_dict})[
            "values"
        ]
        if scan_point_override is not None:
            toypars[f"{self.var_to_profile}"] = scan_point_override
        else:
            toypars[
                f"{self.var_to_profile}"
            ] = scan_point  # override here if we want to compare the power of the toy ts to another scan_point

        # Now we can run the toys
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
        if not profile_dict:
            file_name = self.out_path + f"/{scan_point}_{self.jobid}.h5"

            if overwrite_files and os.path.exists(file_name):
                msg = f"overwriting existing file {file_name}"
                logging.warn(msg)
                os.remove(file_name)

            f = h5py.File(file_name, "a")
        else:
            file_name = (
                self.out_path
                + f"/{scan_point}_{list(profile_dict.values())}_{self.jobid}.h5"
            )

            if overwrite_files and os.path.exists(file_name):
                msg = f"overwriting existing file {file_name}"
                logging.warn(msg)
                os.remove(file_name)

            f = h5py.File(file_name, "a")
            dset = f.create_dataset(
                "profile_parameters_names", data=list(profile_dict.keys())
            )
            dset = f.create_dataset(
                "profile_parameters_values", data=list(profile_dict.values())
            )

        dset = f.create_dataset("ts", data=toyts)
        dset = f.create_dataset("ts_denom", data=toyts_denom)
        dset = f.create_dataset("ts_num", data=toyts_num)
        dset = f.create_dataset("s", data=scan_point)
        # dset = f.create_dataset("Es", data=data)
        # dset = f.create_dataset("nuisance", data=nuisance)
        # dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)
        dset = f.create_dataset("seed", data=seeds_to_save)

        f.close()

        return None

    def run_and_save_brazil(
        self,
        scan_points,
    ) -> None:
        """
        Runs toys at 0 signal rate and computes the test statistic for different signal hypotheses
        """
        # First we need to profile out the variable we are scanning at 0 signal rate
        toypars = self.profile({f"{self.var_to_profile}": 1.0e-9})["values"]

        # Add 0 to the scan points if it is not there
        if 1.0e-9 not in scan_points:
            scan_points = np.insert(scan_points, 0, 1.0e-9)

        # Now we can run the toys
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

        # Now, save the toys to a file
        file_name = self.out_path + f"/1E-9_{self.jobid}.h5"
        f = h5py.File(file_name, "a")
        dset = f.create_dataset("ts", data=toyts)
        dset = f.create_dataset("ts_num", data=toyts_num)
        dset = f.create_dataset("ts_denom", data=toyts_denom)
        dset = f.create_dataset("s", data=scan_points)
        dset = f.create_dataset("Es", data=data)
        dset = f.create_dataset("nuisance", data=nuisance)
        dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)
        dset = f.create_dataset("seed", data=seeds_to_save)

        f.close()

        return None

    def run_and_save_brazil_with_profile_parameters(
        self,
        scan_point,
        profile_dict: dict = {},  # noqa:B006
    ) -> None:
        """
        Runs toys at 0 signal rate and computes the test statistic for different signal hypotheses
        If we are running with profile_dict parameters, the optimal job submission differs from the above and is more similar to run_and_save_toys
        """
        # First we need to profile out the variable we are scanning at 0 signal rate
        toypars = self.profile({f"{self.var_to_profile}": 1.0e-9, **profile_dict})[
            "values"
        ]

        # Now we can run the toys
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
        file_name = (
            self.out_path
            + f"/1E-9_{scan_point}_{list(profile_dict.values())}_{self.jobid}.h5"
        )
        f = h5py.File(file_name, "a")
        dset = f.create_dataset("ts", data=toyts)
        dset = f.create_dataset("ts_num", data=toyts_num)
        dset = f.create_dataset("ts_denom", data=toyts_denom)
        dset = f.create_dataset("s", data=scan_point)
        dset = f.create_dataset(
            "profile_parameters_names", data=list(profile_dict.keys())
        )
        dset = f.create_dataset(
            "profile_parameters_values", data=list(profile_dict.values())
        )
        # dset = f.create_dataset("Es", data=data)
        # dset = f.create_dataset("nuisance", data=nuisance)
        # dset = f.create_dataset("num_sig_num_bkg_drawn", data=num_drawn)
        dset = f.create_dataset("seed", data=seeds_to_save)

        f.close()

        return None
