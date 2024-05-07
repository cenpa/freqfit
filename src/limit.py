"""
Class that holds a pseudoexperiment and has methods that help set a frequentist limit
"""


import numpy as np
from scipy.special import erfcinv

from legendfreqfit.pseudoexperiment import Pseudoexperiment

SEED = 42


class SetLimit(Pseudoexperiment):
    def __init__(
        self,
        config: dict,
        name: str = None,
    ) -> None:
        super().__init__(config, name)

        self.test_statistics = None

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

    def scan_ts(self, var_to_scan: dict) -> np.array:
        """
        Parameters
        ----------
        var_to_scan
            A dictionary containing the variable to scan and its values

        Returns
        -------
        ts
            Value of the specified test statistic at the scanned values
        """
        return None

    def find_crossing(self, var, ts, t_crit, interpolation_mode="i"):
        # Need to make sure this can handle two crossings in case of discovery!
        return None

    def create_fine_scan(self, var):
        return None

    def run_toys(self, scan_points, num_toys, CL):
        """
        Runs toys at specified scan points and returns the critical value of the test statistic and its uncertainty
        """
        return None
