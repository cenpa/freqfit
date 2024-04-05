import numpy as np
from iminuit import Minuit, cost

import legendfreqfit.constants as constants
from legendfreqfit.models import gaussian_on_uniform

QBB = constants.QBB
N_A = constants.NA
M_A = constants.MA


def test_gaussian_on_uniform_pdf():
    eff = 0.6
    exp = 100
    S = 1e-24
    BI = 2e-4
    window = 260
    delta = 0
    sigma = 1

    Es = np.linspace(1930, 2190, 100)

    # compute using our function
    model_values = gaussian_on_uniform.pdf(Es, S, BI, delta, sigma, eff, exp, windowsize=window)

    # Compute using scipy
    mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_B = exp * BI * window
    scipy_values = (1 / (mu_S + mu_B)) * (
        mu_S
        * np.exp(-((Es - QBB - delta) ** 2) / (2 * (sigma**2)))
        / (np.sqrt(2 * np.pi) * sigma)
        + mu_B / window
    )

    assert np.allclose(model_values, scipy_values, rtol=1e-3)


def test_gaussian_on_uniform_logpdf():
    eff = 0.6
    exp = 100
    S = 1e-24
    BI = 2e-4
    window = 240
    delta = 0
    sigma = 1

    Es = np.linspace(1930, 2190, 100)

    # compute using our function
    model_values = gaussian_on_uniform.logpdf(Es, S, BI, delta, sigma, eff, exp, windowsize=window)

    # Compute using scipy
    mu_S = np.log(2) * (N_A * S) * eff * exp / M_A
    mu_B = exp * BI * window
    scipy_values = np.log(
        (1 / (mu_S + mu_B))
        * (
            mu_S
            * np.exp(-((Es - QBB - delta) ** 2) / (2 * (sigma**2)))
            / (np.sqrt(2 * np.pi) * sigma)
            + mu_B / window
        )
    )

    assert np.allclose(model_values, scipy_values, rtol=1e-3)


def test_gaussian_on_uniform_rvs():
    n_sig = 1000
    n_bkg = 100
    delta = 0
    sigma = 1

    random_sample = gaussian_on_uniform.rvs(n_sig, n_bkg,delta, sigma)

    assert np.allclose(np.median(random_sample), QBB, rtol=1e-1)


def test_iminuit_integration():
    n_sig = 1000
    n_bkg = 100
    delta = 0
    sigma = 1

    random_sample = gaussian_on_uniform.rvs(n_sig, n_bkg, delta, sigma)

    c = cost.UnbinnedNLL(random_sample, gaussian_on_uniform.pdf)
    c._parameters.pop("windowsize")
    m = Minuit(c, S=1, BI=0.1, delta=-0.1, sigma=0.6, eff=0.9, exp=0.9)
    m.fixed["eff", "exp"] = True
    m.migrad()

    assert np.allclose(m.values["sigma"], 1, rtol=1e-1)
    assert np.allclose(m.values["delta"], 0.01, rtol=1e0)

    c = cost.UnbinnedNLL(random_sample, gaussian_on_uniform.logpdf, log=True)
    c._parameters.pop("windowsize")
    m = Minuit(c, S=1, BI=0.1, delta=-0.1, sigma=0.6, eff=0.9, exp=0.9)
    m.fixed["eff", "exp"] = True
    m.migrad()

    assert np.allclose(m.values["sigma"], 1, rtol=1e-1)
    assert np.allclose(m.values["delta"], 0.01, rtol=1e0)
