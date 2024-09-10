"""
Script for users to write their own initial guesses and pass them into the `experiment` and `toy` classes
"""

import numpy as np

import legendfreqfit.models.constants as constants
from legendfreqfit.models.correlated_efficiency_0vbb import (
    correlated_efficiency_0vbb_gen,
)
from legendfreqfit.models.mjd_0vbb import mjd_0vbb_gen

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


def zero_nu_initial_guess(experiment):
    # Loop through the datasets and grab the exposures, efficiencies, and sigma
    totexp = 0.0
    sigma_expweighted = 0.0
    eff_expweighted = 0.0
    effunc_expweighted = 0.0
    Es = []

    # need to handle this differently depending on if this is a toy or not/what datasets have been combined
    # This is for a toy
    if hasattr(experiment, "experiment"):
        for ds in experiment.experiment.datasets.values():
            if isinstance(ds.model, correlated_efficiency_0vbb_gen):
                totexp = totexp + ds._parlist[7]
                sigma_expweighted = sigma_expweighted + ds._parlist[3] * ds._parlist[7]
                eff_expweighted = eff_expweighted + ds._parlist[4] * ds._parlist[7]
                effunc_expweighted = (
                    effunc_expweighted + ds._parlist[5] * ds._parlist[7]
                )
                Es.extend(ds._toy_data)
            elif isinstance(ds.model, mjd_0vbb_gen):
                totexp = totexp + ds._parlist[10]
                sigma_expweighted = sigma_expweighted + ds._parlist[4] * ds._parlist[10]
                eff_expweighted = eff_expweighted + ds._parlist[7] * ds._parlist[10]
                effunc_expweighted = (
                    effunc_expweighted + ds._parlist[8] * ds._parlist[10]
                )
                Es.extend(ds._toy_data)
            else:
                raise NotImplementedError(
                    f"Model of type {ds.model} not yet implemented here!"
                )

    # This is an experiment
    else:
        for ds in experiment.datasets.values():
            if isinstance(ds.model, correlated_efficiency_0vbb_gen):
                totexp = totexp + ds._parlist[7]
                sigma_expweighted = sigma_expweighted + ds._parlist[3] * ds._parlist[7]
                eff_expweighted = eff_expweighted + ds._parlist[4] * ds._parlist[7]
                effunc_expweighted = (
                    effunc_expweighted + ds._parlist[5] * ds._parlist[7]
                )
                Es.extend(ds.data)
            elif isinstance(ds.model, mjd_0vbb_gen):
                totexp = totexp + ds._parlist[10]
                sigma_expweighted = sigma_expweighted + ds._parlist[4] * ds._parlist[10]
                eff_expweighted = eff_expweighted + ds._parlist[7] * ds._parlist[10]
                effunc_expweighted = (
                    effunc_expweighted + ds._parlist[8] * ds._parlist[10]
                )
                Es.extend(ds.data)
            else:
                raise NotImplementedError(
                    f"Model of type {ds.model} not yet implemented here!"
                )

    sigma_expweighted = sigma_expweighted / totexp
    eff_expweighted = eff_expweighted / totexp
    effunc_expweighted = effunc_expweighted / totexp

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
    sigma_tot
        The total sigma of the QBB peak
    """
    QBB_ROI_SIZE = [
        5 * sigma_expweighted,
        5 * sigma_expweighted,
    ]  # how many keV away from QBB in - and + directions we are defining the ROI
    BKG_WINDOW_SIZE = WINDOWSIZE - np.sum(
        QBB_ROI_SIZE
    )  # subtract off the keV we are counting as the signal region
    n_sig = 0
    n_bkg = 0
    for E in Es:
        if QBB - QBB_ROI_SIZE[0] <= E <= QBB + QBB_ROI_SIZE[1]:
            n_sig += 1
        else:
            n_bkg += 1

    # find the expected BI
    BI_guess = n_bkg / (BKG_WINDOW_SIZE * totexp)

    # Now find the expected signal rate
    n_sig -= (
        n_bkg * np.sum(QBB_ROI_SIZE) / BKG_WINDOW_SIZE
    )  # subtract off the expected number of BI counts in ROI

    s_guess = n_sig / (totexp * eff_expweighted)

    # need to handle this differently depending on if this is a toy or not/what datasets have been combined
    if hasattr(experiment, "experiment"):
        guess = {}
        for par in experiment.fitparameters:
            guess |= {par: experiment.experiment._toy_parameters[par]["value"]}
    else:
        guess = {
            fitpar: experiment.parameters[fitpar]["value"]
            if "value" in experiment.parameters[fitpar]
            else None
            for fitpar in experiment.fitparameters
        }

    # If we get only one count in the signal window, then this guess will estimate too low a background
    # So, if BI is guessed as 0 and S is not 0, smear out the signal rate between them
    if s_guess < 0:
        s_guess = 0

    if (BI_guess < 1e-6) and (s_guess != 0):
        BI_guess = s_guess / 2
        s_guess /= 2

    # update only the S and BI
    for fitpar in guess.keys():
        if "BI" in fitpar:
            guess[fitpar] = BI_guess / 0.0001  # For scaled BI
        if "global_S" in fitpar:
            guess[fitpar] = s_guess / 0.01  # For scaled S in the model

    return guess
