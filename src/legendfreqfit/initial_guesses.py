"""
Script for users to write their own initial guesses and pass them into the `experiment` and `toy` classes
"""

import numpy as np
from iminuit import Minuit

import legendfreqfit.models.constants as constants

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
    # figure out if this is a toy or not:
    if hasattr(experiment, "experiment"):
        is_toy = True
        loop_exp = experiment.experiment
    else:
        is_toy = False
        loop_exp = experiment
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
        for ds in loop_exp.datasets.values():
            if is_toy:
                if (BI in ds.fitparameters) & (not ds._toy_is_combined):
                    ds_per_BI.append(ds)
            else:
                if (BI in ds.fitparameters) & (not ds.is_combined):
                    ds_per_BI.append(ds)

        if is_toy:
            for ds in experiment.combined_datasets.values():
                if BI in ds.fitparameters:
                    ds_per_BI.append(ds)
        else:
            for ds in loop_exp.combined_datasets.values():
                if BI in ds.fitparameters:
                    ds_per_BI.append(ds)

        ds_list.append(ds_per_BI)

        ds_names.append([ds.name for ds in ds_per_BI])

    # Fix all the fit parameters in the minuit object, then loosen S, all the BI and the global_effuncscale
    if is_toy:
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

    minuit = Minuit(experiment.costfunction, **guess)
    for par in minuit.parameters:
        minuit.fixed[par] = True
    minuit.fixed["global_S"] = False
    minuit.limits["global_S"] = (1e-9, None)

    # minuit.fixed["global_effuncscale"] = True
    # minuit.limits["global_effuncscale"] = (-100, 100)
    for BI in BI_list:
        minuit.fixed[f"{BI}"] = False
        minuit.limits[f"{BI}"] = (1e-9, None)

        if "empty" in BI:
            minuit.fixed[f"{BI}"] = True

    # minuit.simplex()
    minuit.migrad()
    guess = minuit.values.to_dict()

    return guess
