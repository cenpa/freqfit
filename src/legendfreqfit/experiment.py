"""
A class that controls an experiment and calls the `Superset` class.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from scipy.stats import chi2

from legendfreqfit.superset import Superset
from legendfreqfit.toy import Toy
from legendfreqfit.utils import grab_results, load_config
from legendfreqfit.statistics import dkw_band, emp_cdf

SEED = 42


class Experiment(Superset):
    def __init__(
        self,
        config: dict,
        name: str = None,
    ) -> None:
        self.config = config

        constraints = (
            self.config["constraints"] if "constraints" in self.config else None
        )

        super().__init__(
            datasets=self.config["datasets"],
            parameters=self.config["parameters"],
            constraints=constraints,
            name=name,
        )

        # collect which parameters are included as nuisance parameters
        self.nuisance = []
        for parname, pardict in self.config["parameters"].items():
            if "includeinfit" in pardict and pardict["includeinfit"]:
                if "nuisance" in pardict and pardict["nuisance"]:
                    if "fixed" in pardict and pardict["fixed"]:
                        msg = f"{parname} has `fixed` as `True` and `nuisance` as `True`. {parname} will be treated as fixed."
                        warnings.warn(msg)
                    else:
                        self.nuisance.append(parname)

        # get the fit parameters and set the parameter initial values
        self.guess = self.initialguess()
        self.minuit = Minuit(self.costfunction, **self.guess)

        # raise a RunTime error if function evaluates to NaN
        self.minuit.throw_nan = True

        # to set limits and fixed variables
        self.minuit_reset()

        # to store the best fit result
        self.best = None

    @classmethod
    def file(
        cls,
        file: str,
        name: str = None,
    ):
        config = load_config(file=file)
        return cls(config=config, name=name)

    def initialguess(
        self,
    ) -> dict:
        guess = {
            fitpar: self.parameters[fitpar]["value"]
            if "value" in self.parameters[fitpar]
            else None
            for fitpar in self.fitparameters
        }

        # for fitpar, value in guess.items():
        #     if value is None:
        #         guess[fitpar] = 1e-9

        # could put other stuff here to get a better initial guess though probably that should be done
        # somewhere specific to the analysis. Or maybe this function could call something for the analysis.
        # eh, probably better to make the initial guess elsewhere and stick it in the config file.
        # for that reason, I'm commenting out the few lines above.

        return guess

    def minuit_reset(
        self,
    ) -> None:
        # resets the minimization and stuff
        # does not change limits but does remove "fixed" attribute of variables
        self.minuit.reset()

        # overwrite the limits
        # note that this information can also be contained in a Dataset when instantiated
        # and is overwritten here

        # also set which parameters are fixed
        for parname, pardict in self.config["parameters"].items():
            if parname in self.minuit.parameters:
                if "limits" in pardict:
                    self.minuit.limits[parname] = pardict["limits"]
                if "fixed" in pardict:
                    self.minuit.fixed[parname] = pardict["fixed"]

        return

    def bestfit(
        self,
        force: bool = False,
    ) -> dict:
        """
        force
            By default (`False`), if `self.best` has a result, the minimization will be skipped and that
            result will be returned instead. If `True`, the minimization will be run and the result returned and
            stored as `self.best`

        Performs a global minimization and returns a `dict` with the results. These results are also stored in
        `self.best`.
        """
        # don't run this more than once if we don't have to
        if self.best is not None and not force:
            return self.best

        # remove any previous minimizations
        self.minuit_reset()

        self.minuit.migrad()

        self.best = grab_results(self.minuit)

        return self.best

    def profile(
        self,
        parameters: dict,
    ) -> dict:
        """
        parameters
            `dict` where keys are names of parameters to fix and values are the value that the parameter should be
            fixed to
        """

        self.minuit_reset()

        for parname, parvalue in parameters.items():
            self.minuit.fixed[parname] = True
            self.minuit.values[parname] = parvalue

        self.minuit.migrad()

        return grab_results(self.minuit)

    # this corresponds to t_mu or t_mu^tilde depending on whether there is a limit on the parameters
    def ts(
        self,
        profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
        force: bool = False,
    ) -> float:
        """
        force
            See `experiment.bestfit()` for description. Default is `False`.
        """
        denom = self.bestfit(force=force)["fval"]

        num = self.profile(parameters=profile_parameters)["fval"]

        # because these are already -2*ln(L) from iminuit
        return num - denom

    def maketoy(
        self,
        parameters: dict,
        seed: int = SEED,
    ) -> Toy:
        toy = Toy(experiment=self, parameters=parameters, seed=seed)

        return toy

    def toy_ts(
        self,
        parameters: dict,  # parameters and values needed to generate the toys
        profile_parameters: dict,  # which parameters to fix and their value (rest are profiled)
        num: int = 1,
        seed: int = SEED,
    ):
        """
        Makes a number of toys and returns their test statistics.
        """

        ts = np.zeros(num)
        np.random.seed(SEED)
        seed = np.random.randint(1000000, size=num)

        for i in range(num):
            thistoy = self.maketoy(parameters=parameters, seed=seed[i])
            ts[i] = thistoy.ts(profile_parameters=profile_parameters)

        return ts

    def toy_ts_critical(
        self,
        ts_dist: np.array,  # output of toy_ts
        bins=100,  # int or array, numbins or list of bin edges for CDF
        threshold: float = 0.9,  # critical threshold for test statistic
        confidence: float = 0.68,  # width of confidence interval
        plot: bool = False,  # if True, save plots of CDF and PDF with critical bands
    ):
        """
        Returns the critical value of the test statistic and confidence interval
        """
        cdf, bins = emp_cdf(ts_dist, bins)

        lo_band, hi_band = dkw_band(cdf, nevts=len(ts_dist), confidence=confidence)

        idx_crit = np.where(cdf >= threshold)[0][0]
        critical = bins[idx_crit]

        lo = lo_band[idx_crit]
        hi = hi_band[idx_crit]

        lo_idx = np.where(cdf >= lo)[0][0]
        hi_idx = np.where(cdf >= hi)[0][0]

        lo_ts = bins[lo_idx]
        hi_ts = bins[hi_idx]

        if plot:
            int_thresh = int(100 * threshold)
            int_conf = int(100 * confidence)
            fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(11, 5))
            axs[0].plot(bins, cdf, color="red", label="data cdf")
            axs[0].plot(
                bins, lo_band, color="orange", label=f"cdf {int_conf}% interval"
            )
            axs[0].plot(bins, hi_band, color="orange")
            axs[0].plot(
                bins, chi2.cdf(bins, df=1), color="black", label="chi2 df=1 cdf"
            )

            axs[0].axvline(critical, color="blue", label=f"{int_thresh}% ts")
            axs[0].axvspan(lo_ts, hi_ts, alpha=0.5, color="blue")
            axs[0].axhline(threshold, color="green")
            axs[0].axhspan(lo, hi, color="green", alpha=0.5)
            axs[0].set_xlabel(r"$\tilde{t}_S$")
            axs[0].set_ylabel(r"$cdf(\tilde{t}_S)$")
            axs[0].legend()

            bincenters = (bins[1:] + bins[:-1]) / 2
            axs[1].hist(ts_dist, bins=bins, density=True)
            axs[1].plot(bincenters, chi2.pdf(bincenters, df=1), "k", label="chi2 df=1")

            axs[1].axvline(critical, color="red", label="ts_crit")
            axs[1].axvspan(lo_ts, hi_ts, color="blue", alpha=0.5)
            axs[1].axvline(
                chi2.ppf(threshold, df=1),
                color="black",
                linestyle="dashed",
                label="chi2 critical",
            )

            axs[1].set_xlabel(r"$\tilde{t}_S$")
            axs[1].set_ylabel(r"$pdf(\tilde{t}_S)$")
            axs[1].legend()

            plt.savefig(f"ts_critical_{int_thresh}.jpg")

        return critical, lo_ts, hi_ts
