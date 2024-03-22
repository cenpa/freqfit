import numpy as np
from iminuit import Minuit, cost

import legendfreqfit.constants as constants
from legendfreqfit.models import gaussian_on_uniform

Qbb = constants.QBB
N_A = constants.NA
m_a = constants.MA


def test_gaussian_on_uniform_pdf():
    eff = 0.6
    exp = 100
    S = 1e-24
    BI = 2e-4
    deltaE = 260
    delta = 0
    sigma = 1

    Es = np.linspace(1930, 2190, 100)

    # compute using our function
    model_values = gaussian_on_uniform.pdf(Es, deltaE, S, BI, delta, sigma, eff, exp)

    # Compute using scipy
    mu_S = np.log(2) * (N_A * S) * eff * exp / m_a
    mu_B = exp * BI * deltaE
    scipy_values = (1 / (mu_S + mu_B)) * (
        mu_S
        * np.exp(-((Es - Qbb - delta) ** 2) / (2 * (sigma**2)))
        / (np.sqrt(2 * np.pi) * sigma)
        + mu_B / deltaE
    )

    assert np.allclose(model_values, scipy_values, rtol=1e-3)


def test_gaussian_on_uniform_logpdf():
    eff = 0.6
    exp = 100
    S = 1e-24
    BI = 2e-4
    deltaE = 260
    delta = 0
    sigma = 1

    Es = np.linspace(1930, 2190, 100)

    # compute using our function
    model_values = gaussian_on_uniform.logpdf(Es, deltaE, S, BI, delta, sigma, eff, exp)

    # Compute using scipy
    mu_S = np.log(2) * (N_A * S) * eff * exp / m_a
    mu_B = exp * BI * deltaE
    scipy_values = np.log(
        (1 / (mu_S + mu_B))
        * (
            mu_S
            * np.exp(-((Es - Qbb - delta) ** 2) / (2 * (sigma**2)))
            / (np.sqrt(2 * np.pi) * sigma)
            + mu_B / deltaE
        )
    )

    assert np.allclose(model_values, scipy_values, rtol=1e-3)


def test_gaussian_on_uniform_rvs():
    n_sig = 1000
    n_bkg = 100
    E_lo = 1930
    E_hi = 2190
    delta = 0
    sigma = 1

    random_sample = gaussian_on_uniform.rvs(n_sig, n_bkg, E_lo, E_hi, delta, sigma)

    assert np.allclose(np.median(random_sample), 2039, rtol=1e-1)


def test_iminuit_integration():
    n_sig = 1000
    n_bkg = 100
    E_lo = 1930
    E_hi = 2190
    delta = 0
    sigma = 1

    random_sample = gaussian_on_uniform.rvs(n_sig, n_bkg, E_lo, E_hi, delta, sigma)

    c = cost.UnbinnedNLL(random_sample, gaussian_on_uniform.pdf)
    m = Minuit(c, deltaE=260, S=1, BI=0.1, delta=-0.1, sigma=0.6, eff=0.9, exp=0.9)
    m.fixed["deltaE"] = True
    m.migrad()

    assert np.allclose(m.values["sigma"], 1, rtol=1e-1)
    assert np.allclose(m.values["delta"], 0.01, rtol=1e0)

    c = cost.UnbinnedNLL(random_sample, gaussian_on_uniform.logpdf, log=True)
    m = Minuit(c, deltaE=260, S=1, BI=0.1, delta=-0.1, sigma=0.6, eff=0.9, exp=0.9)
    m.fixed["deltaE"] = True
    m.limits["sigma", "S", "BI", "eff", "exp"] = (0.001, 2)
    m.migrad()

    assert np.allclose(m.values["sigma"], 1, rtol=1e0)
    assert np.allclose(m.values["delta"], 0.04, rtol=1e-1)
