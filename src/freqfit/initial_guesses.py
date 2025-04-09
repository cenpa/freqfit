"""
Script for users to write their own initial guesses and pass them into the `experiment` and `toy` classes
"""

import numpy as np
from iminuit import Minuit

import freqfit.models.constants as constants
from freqfit.models.box_model_0vbb import box_model_0vbb_gen
from freqfit.models.correlated_efficiency_0vbb import (
    correlated_efficiency_0vbb_gen,
)
from freqfit.models.correlated_efficiency_0vbb_correlate_delta import (
    correlated_efficiency_0vbb_correlate_delta_gen,
)
from freqfit.models.correlated_efficiency_0vbb_exponential_background import (
    correlated_efficiency_0vbb_exponential_background_gen,
)
from freqfit.models.correlated_efficiency_0vbb_linear_background import (
    correlated_efficiency_0vbb_linear_background_gen,
)
from freqfit.models.correlated_efficiency_NME_0vbb import (
    correlated_efficiency_NME_0vbb_gen,
)
from freqfit.models.mjd_0vbb import mjd_0vbb_gen
from freqfit.models.mjd_NME_0vbb import mjd_NME_0vbb_gen
from freqfit.models.truncated_correlated_efficiency_0vbb import (
    truncated_correlated_efficiency_0vbb_gen,
)

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
            guess |= {
                par: experiment.experiment.toy_params_for_initial_guess[par]["value"]
            }
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

    if "global_S" in list(minuit.fixed):
        minuit.fixed["global_S"] = False
        minuit.limits["global_S"] = (0, None)

    if "global_m_bb" in list(minuit.fixed):
        minuit.fixed["global_m_bb"] = False
        minuit.limits["global_m_bb"] = (0, None)

    # minuit.fixed["global_effuncscale"] = True
    # minuit.limits["global_effuncscale"] = (-100, 100)
    for BI in BI_list:
        minuit.fixed[f"{BI}"] = False
        minuit.limits[f"{BI}"] = (0, None)

        if "empty" in BI:
            minuit.fixed[f"{BI}"] = True

    # minuit.simplex()
    minuit.migrad()
    guess = minuit.values.to_dict()

    return guess


def guess_BI_S(Es, totexp, eff_expweighted, sigma_expweighted):  # noqa: N802
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
    QBB_ROI_SIZE = [
        3 * sigma_expweighted,
        3 * sigma_expweighted,
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

    # # Now find the expected signal rate
    # n_sig -= (
    #     n_bkg * np.sum(QBB_ROI_SIZE) / BKG_WINDOW_SIZE
    # )  # subtract off the expected number of BI counts in ROI

    s_guess = n_sig / (totexp * eff_expweighted)

    return BI_guess, s_guess


def guess_BI_m_bb(Es, totexp, eff_expweighted, sigma_expweighted, NME):  # noqa: N802
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
    NME
        The central value of the NME used to generate this toy
    """
    QBB_ROI_SIZE = [
        3 * sigma_expweighted,
        3 * sigma_expweighted,
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

    m_guess = np.sqrt(n_sig / (totexp * eff_expweighted * NME**2))

    return BI_guess, m_guess


def poisson_initial_guess(experiment):
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
            guess |= {
                par: experiment.experiment.toy_params_for_initial_guess[par]["value"]
            }
    else:
        guess = {
            fitpar: experiment.parameters[fitpar]["value"]
            if "value" in experiment.parameters[fitpar]
            else None
            for fitpar in experiment.fitparameters
        }

    # Then perform the loop over datasets that share a background index
    BI_guesses = []
    is_mbb_scan = False
    for ds_BI in ds_list:
        # Get estimates for these parameters based only on the datasets contributing to one BI
        BI_totexp = 0.0
        BI_sigma_expweighted = 0.0
        BI_eff_expweighted = 0.0
        BI_effunc_expweighted = 0.0
        Es_per_BI = []

        # need to handle this differently depending on if this is a toy or not/what datasets have been combined
        for ds in ds_BI:
            if (
                isinstance(ds.model, correlated_efficiency_0vbb_gen)
                or isinstance(ds.model, box_model_0vbb_gen)
                or isinstance(ds.model, truncated_correlated_efficiency_0vbb_gen)
                or isinstance(ds.model, correlated_efficiency_NME_0vbb_gen)
                or isinstance(ds.model, correlated_efficiency_0vbb_correlate_delta_gen)
                or isinstance(
                    ds.model, correlated_efficiency_0vbb_linear_background_gen
                )
                or isinstance(
                    ds.model, correlated_efficiency_0vbb_exponential_background_gen
                )
            ):
                BI_totexp = BI_totexp + ds._parlist[7]
                BI_sigma_expweighted = (
                    BI_sigma_expweighted + ds._parlist[3] * ds._parlist[7]
                )
                BI_eff_expweighted = (
                    BI_eff_expweighted + ds._parlist[4] * ds._parlist[7]
                )
                BI_effunc_expweighted = (
                    BI_effunc_expweighted + ds._parlist[5] * ds._parlist[7]
                )
                if is_toy:
                    if ds._toy_data is not None:
                        Es_per_BI.extend(ds._toy_data)
                    else:
                        Es_per_BI.extend([])
                else:
                    Es_per_BI.extend(ds.data)
                # report the NME if this is an m_bb toy
                if isinstance(ds.model, correlated_efficiency_NME_0vbb_gen):
                    NME = ds._parlist[-1]
                    is_mbb_scan = True  # we cannot mix and match S and m_bb, if this is true we are using m_bb only
            elif isinstance(ds.model, mjd_0vbb_gen) or isinstance(
                ds.model, mjd_NME_0vbb_gen
            ):
                BI_totexp = BI_totexp + ds._parlist[10]
                BI_sigma_expweighted = (
                    BI_sigma_expweighted + ds._parlist[4] * ds._parlist[10]
                )
                BI_eff_expweighted = (
                    BI_eff_expweighted + ds._parlist[7] * ds._parlist[10]
                )
                BI_effunc_expweighted = (
                    BI_effunc_expweighted + ds._parlist[8] * ds._parlist[10]
                )
                if is_toy:
                    if ds._toy_data is not None:
                        Es_per_BI.extend(ds._toy_data)
                    else:
                        Es_per_BI.extend([])
                else:
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
        if is_mbb_scan:
            BI_guess, m_bb_guess = guess_BI_m_bb(
                Es_per_BI, BI_totexp, BI_eff_expweighted, BI_sigma_expweighted, NME
            )
            if m_bb_guess < 0:
                m_bb_guess = 0

            if BI_guess <= 0:
                BI_guess = 0
        else:
            BI_guess, s_guess = guess_BI_S(
                Es_per_BI, BI_totexp, BI_eff_expweighted, BI_sigma_expweighted
            )
            if s_guess < 0:
                s_guess = 0

            if BI_guess <= 0:
                BI_guess = 0

        BI_guesses.append(BI_guess)

    # Compute the total for the experiment, so that we can better guess an initial S value
    sigma_expweighted = sigma_expweighted / BI_totexp
    eff_expweighted = eff_expweighted / BI_totexp
    effunc_expweighted = effunc_expweighted / BI_totexp

    if is_mbb_scan:
        m_bb_guess = guess_BI_m_bb(Es, totexp, eff_expweighted, sigma_expweighted, NME)[
            1
        ]
        if m_bb_guess <= 0:
            m_bb_guess = 0
    else:
        S_guess = guess_BI_S(Es, totexp, eff_expweighted, sigma_expweighted)[1]
        if S_guess <= 0:
            S_guess = 0

    # need to handle this differently depending on if this is a toy or not/what datasets have been combined
    if is_toy:
        guess = {}
        for par in experiment.fitparameters:
            guess |= {
                par: experiment.experiment.toy_params_for_initial_guess[par]["value"]
            }
    else:
        guess = {
            fitpar: experiment.parameters[fitpar]["value"]
            if "value" in experiment.parameters[fitpar]
            else None
            for fitpar in experiment.fitparameters
        }

    # update the BI
    for i, BI in enumerate(BI_list):
        guess[BI] = BI_guesses[i]

    # Update the signal guess
    if is_mbb_scan:
        guess["global_m_bb"] = m_bb_guess
    else:
        guess["global_S"] = S_guess

    return guess
