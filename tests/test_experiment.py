import numpy as np

from freqfit import Workspace


def fixed_B_workspace():
    # one-bin Poisson with B in the fit but fixed, so it is a Minuit parameter that should never move
    config = Workspace.load_config("tests/FC_config.yaml")
    config["parameters"]["B"]["includeinfit"] = True
    config["parameters"]["B"]["fixed"] = True
    return Workspace(config), config["parameters"]["B"]["value"]


def test_minuit_reset_fixed_value():
    # an initial guess function can estimate a fixed parameter from data (e.g. initial_guess_0vbb does this for BI);
    # minuit_reset should restore the guess for the free parameters but keep fixed ones at their configured value
    ws, B = fixed_B_workspace()
    experiment = ws.experiment
    experiment.guess = {"S": 3.0, "B": 2 * B}

    experiment.minuit_reset()

    assert experiment.minuit.fixed["B"]
    assert experiment.minuit.values["B"] == B
    assert not experiment.minuit.fixed["S"]
    assert experiment.minuit.values["S"] == 3.0


def test_bestfit_keeps_fixed_value():
    ws, B = fixed_B_workspace()
    experiment = ws.experiment
    experiment.guess = {"S": 3.0, "B": 2 * B}

    best = experiment.bestfit(force=True)

    assert best["values"]["B"] == B
    # with one observed count and B fixed, S_hat is pushed to its physical limit
    assert np.isclose(best["values"]["S"], 0.0, atol=1e-3)
