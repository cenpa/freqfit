import numpy as np
# from iminuit import Minuit

# from freqfit.experiment import Experiment
from freqfit.guess import Guess

from freqfit.models import constants
from freqfit.models.correlated_efficiency_0vbb import (
    correlated_efficiency_0vbb_gen
    )
from freqfit.models.correlated_efficiency_0vbb_correlate_delta import (
    correlated_efficiency_0vbb_correlate_delta_gen
    )
from freqfit.models.correlated_efficiency_0vbb_linear_background import (
    correlated_efficiency_0vbb_linear_background_gen
    )
from freqfit.models.correlated_efficiency_0vbb_exponential_background import (
    correlated_efficiency_0vbb_exponential_background_gen
    )
from freqfit.models.mjd_0vbb import mjd_0vbb_gen
    
# default analysis window and width
# window
#     uniform background regions to pull from, must be a 2D array of form e.g. `np.array([[0,1],[2,3]])`
#     where edges of window are monotonically increasing (this is not checked), in keV.
#     Default is typical analysis window.
WINDOW = np.array(constants.WINDOW)

WINDOWSIZE = 0.0
for i in range(len(WINDOW)):
    WINDOWSIZE += WINDOW[i][1] - WINDOW[i][0]

QBB = constants.QBB

class initial_guess_0vbb(Guess):
    def guess(
        self,
        experiment
        ):

        # Loop through the datasets and grab the exposures, efficiencies, and sigma from all datasets
        totexp = 0.0
        sigma_expweighted = 0.0
        eff_expweighted = 0.0
        effunc_expweighted = 0.0
        Es = []

        # Find which datasets share a background index
        BI_list = [par for par in experiment.fitparameters if "BI" in par]
        ds_list = []
        ds_names = []

        for BI in BI_list:
            ds_per_BI = []
            for ds in experiment.datasets.values():
                if (BI in ds.fitparameters):
                    ds_per_BI.append(ds)

            ds_list.append(ds_per_BI)

            ds_names.append([ds.name for ds in ds_per_BI])

        # Fix all the fit parameters in the minuit object, then loosen S, all the BI and the global_effuncscale
        guess = {
            fitpar: experiment.fitparameters[fitpar]["value"]
            for fitpar in experiment.fitparameters
        }

        # Then perform the loop over datasets that share a background index
        BI_guesses = []
        for ds_BI in ds_list:

            # Get estimates for these parameters based only on the datasets contributing to one BI
            BI_totexp = 0.0
            BI_sigma_expweighted = 0.0
            BI_eff_expweighted = 0.0
            BI_effunc_expweighted = 0.0
            Es_per_BI = []
            for ds in ds_BI:
                if (
                    isinstance(ds.model, correlated_efficiency_0vbb_gen)
                    or isinstance(ds.model, correlated_efficiency_0vbb_correlate_delta_gen)
                    or isinstance(
                        ds.model, correlated_efficiency_0vbb_linear_background_gen
                    )
                    or isinstance(
                        ds.model, correlated_efficiency_0vbb_exponential_background_gen
                    )
                ):
                    BI_totexp = BI_totexp + ds._parlist_values[7]
                    BI_sigma_expweighted = (
                        BI_sigma_expweighted + ds._parlist_values[3] * ds._parlist_values[7]
                    )
                    BI_eff_expweighted = (
                        BI_eff_expweighted + ds._parlist_values[4] * ds._parlist_values[7]
                    )
                    BI_effunc_expweighted = (
                        BI_effunc_expweighted + ds._parlist_values[5] * ds._parlist_values[7]
                    )
                    
                    Es_per_BI.extend(ds.data)
                    
                elif isinstance(ds.model, mjd_0vbb_gen):
                    BI_totexp = BI_totexp + ds._parlist_values[10]
                    BI_sigma_expweighted = (
                        BI_sigma_expweighted + ds._parlist_values[4] * ds._parlist_values[10]
                    )
                    BI_eff_expweighted = (
                        BI_eff_expweighted + ds._parlist_values[7] * ds._parlist_values[10]
                    )
                    BI_effunc_expweighted = (
                        BI_effunc_expweighted + ds._parlist_values[8] * ds._parlist_values[10]
                    )
                    Es_per_BI.extend(ds.data)
                else:
                    raise NotImplementedError(
                        f"Model of type {ds.model} not yet implemented here!"
                    )

            totexp += BI_totexp
            sigma_expweighted += BI_sigma_expweighted
            eff_expweighted += BI_eff_expweighted
            effunc_expweighted += BI_effunc_expweighted
            Es.extend(Es_per_BI)

            BI_sigma_expweighted = BI_sigma_expweighted / BI_totexp
            BI_eff_expweighted = BI_eff_expweighted / BI_totexp
            BI_effunc_expweighted = BI_effunc_expweighted / BI_totexp

            # Finally, we are ready to make our guess for this BI
            # If we get only one count in the signal window, then this guess will estimate too low a background
            # So, if BI is guessed as 0 and S is not 0, smear out the signal rate between them

            BI_guess, _ = self.guess_BI_S(
                Es_per_BI, BI_totexp, BI_eff_expweighted, BI_sigma_expweighted
            )

            BI_guesses.append(BI_guess)

        # Compute the total for the experiment, so that we can better guess an initial S value
        sigma_expweighted = sigma_expweighted / BI_totexp
        eff_expweighted = eff_expweighted / BI_totexp
        effunc_expweighted = effunc_expweighted / BI_totexp

        _, S_guess = self.guess_BI_S(Es, totexp, eff_expweighted, sigma_expweighted)

        # update the BI
        for i, BI in enumerate(BI_list):
            guess[BI] = BI_guesses[i]

        # Update the signal guess
        guess["global_S"] = S_guess

        return guess

    def guess_BI_S(
        self,
        Es, 
        totexp, 
        eff_expweighted, 
        sigma_expweighted
        ):  # noqa: N802
        """
        Give a better initial guess for the signal and background rate given an array of data
        The signal rate is estimated in a +/-5 keV window around Qbb, the BI is estimated from everything outside that window

        Parameters
        ----------
        Es
            A numpy array of observed energy data
        totexp
            The total exposure of the experiment
        eff_expweighted
            The total efficiency of the experiment
        sigma_expweighted
            The total sigma of the QBB peak
        """
        roi_size = [
            3 * sigma_expweighted,
            3 * sigma_expweighted,
        ]  # how many keV away from QBB in - and + directions we are defining the ROI
        bkg_window_size = WINDOWSIZE - np.sum(
            roi_size
        )  # subtract off the keV we are counting as the signal region
        n_sig = 0
        n_bkg = 0
        for E in Es:
            if QBB - roi_size[0] <= E <= QBB + roi_size[1]:
                n_sig += 1
            else:
                n_bkg += 1

        # find the expected BI
        BI_guess = n_bkg / (bkg_window_size * totexp)

        # # Now find the expected signal rate
        # n_sig -= (
        #     n_bkg * np.sum(roi_size) / bkg_window_size
        # )  # subtract off the expected number of BI counts in ROI

        s_guess = n_sig / (totexp * eff_expweighted)

        if s_guess < 0:
            s_guess = 0

        if BI_guess <= 0:
            BI_guess = 0
            
        return BI_guess, s_guess