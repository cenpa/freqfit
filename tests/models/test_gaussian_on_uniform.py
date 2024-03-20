import numpy as np

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
