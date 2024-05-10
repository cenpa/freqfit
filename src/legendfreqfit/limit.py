"""
Class that holds a pseudoexperiment and has methods that help set a frequentist limit
"""


import multiprocessing as mp

import numpy as np
from scipy.special import erfcinv

from legendfreqfit.pseudoexperiment import Pseudoexperiment

NUM_CORES = 2  # TODO: change this to an environment variable, or something that detects available cores
SEED = 42


class SetLimit(Pseudoexperiment):
    def __init__(
        self,
        config: dict,
        name: str = None,
    ) -> None:
        """
        This class inherits from the pseudoexperiment, and also holds the name of the variable to profile
        """
        super().__init__(config, name)

        self.test_statistics = (
            None  # maybe we want to store the test statistics internally?
        )
        self.var_to_profile = None

    def set_var_to_profile(self, var_to_profile: str):
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

    def scan_ts(self, var_values: np.array) -> np.array:
        """
        Parameters
        ----------
        var_values
            The values of the variable to scan over

        Returns
        -------
        ts
            Value of the specified test statistic at the scanned values
        """
        # Create the arguments to multiprocess over
        args = [[{f"{self.var_to_profile}": float(xx)}] for xx in var_values]

        with mp.Pool(NUM_CORES) as pool:
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

    def create_fine_scan(self, var):
        # TODO: write this function
        return None

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
        toys_per_core = np.full(NUM_CORES, num // NUM_CORES)
        toys_per_core = np.insert(toys_per_core, len(toys_per_core), num % NUM_CORES)
        args = [
            [parameters, profile_parameters, num_toy] for num_toy in toys_per_core
        ]  # give each core multiple MCs

        with mp.Pool(NUM_CORES) as pool:
            ts = pool.starmap(self.toy_ts, args)

        return np.hstack(ts)

    def run_toys(
        self,
        scan_point,
        num_toys,
        threshold: float = 0.9,
        confidence: float = 0.68,
        scan_point_override=None,
    ) -> list[np.array, np.array]:
        """
        Runs toys at specified scan point and returns the critical value of the test statistic and its uncertainty
        """
        # First we need to profile out the variable we are scanning
        toypars = self.profile({f"{self.var_to_profile}": scan_point})["values"]
        if scan_point_override is not None:
            toypars[f"{self.var_to_profile}"] = scan_point_override
        else:
            toypars[
                f"{self.var_to_profile}"
            ] = scan_point  # override here if we want to compare the power of the toy ts to another scan_point

        # Now we can run the toys
        toyts = self.toy_ts_mp(
            toypars, {f"{self.var_to_profile}": scan_point}, num=num_toys
        )

        # Now grab the critical test statistic
        t_crit, t_crit_low, t_crit_high = self.toy_ts_critical(
            toyts, threshold=threshold, confidence=confidence
        )
        return toyts, t_crit, t_crit_low, t_crit_high
