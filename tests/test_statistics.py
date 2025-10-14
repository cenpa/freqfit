# Test the functions provided in the statistics module
import numpy as np
import pytest
from scipy.stats import chi2, ncx2, norm

import freqfit.statistics as fstat


def test_emp_cdf():
    np.random.seed(310)
    rvs = norm.rvs(loc=20, size=10000)
    bin_edges = np.linspace(0, 40, 1000)
    cdf = norm.cdf(bin_edges[1:], loc=20)
    cdf_computed, _ = fstat.emp_cdf(rvs, bin_edges)

    assert np.allclose(cdf, cdf_computed, atol=1e-2)


def test_emp_cdf_error_handling():
    # Check that we get an exception if parameters is not a Parameters instance
    with pytest.raises(TypeError):
        fstat.emp_cdf([1, 2, 3, 4], "bin_edges")


def test_quantile():
    data = [0, 1, 2, 3]
    # median for this dataset w/o interpolation should be 1 based on averaged inverted cdf
    assert fstat.quantile(data, 0.5) == [2]

    assert fstat.quantile(data, 0.2) == [0]
    assert fstat.quantile(data, 0.25) == [1]
    assert fstat.quantile(data, 0.75) == [3]


def test_binomial_unc_band():
    cdf = 0.8
    nevts = 1000
    err_lo, err_hi = fstat.binomial_unc_band(cdf=[cdf], nevts=nevts, CL=0.68)

    # the 1*sigma error for a binomial process can be estimated by sqrt((p*(1-p))/N)
    err_est = np.sqrt((cdf * (1 - cdf)) / nevts)
    assert np.abs(err_lo - (cdf - err_est)) < 1e-3
    assert np.abs(err_hi - (cdf + err_est)) < 1e-3


def test_dkw_band():
    # the dkw as computed by a random draw should contain the true cdf
    np.random.seed(312)
    bin_edges = np.linspace(0, 40, 1000)
    cdf_true = norm.cdf(bin_edges[1:], loc=20)
    for i in range(10):
        rvs = norm.rvs(loc=20, size=1000)
        cdf_computed, _ = fstat.emp_cdf(rvs, bin_edges)
        band_lo, band_hi = fstat.dkw_band(cdf_computed, nevts=1000, CL=0.90)
        assert np.all((cdf_true >= band_lo) & (cdf_true <= band_hi))


def test_test_statistic_asymptotic_limit():
    t_mu = 0.01
    mu = 1
    mu0 = mu
    sigma = 2

    # test that we get a chi2 with 1 dof
    pdf_true = chi2.pdf(t_mu, df=1, loc=0, scale=1)
    pdf_computed = fstat.test_statistic_asymptotic_limit(t_mu, mu, mu0, sigma)
    assert np.allclose(pdf_true, pdf_computed, rtol=1e-5)

    # test that we get a chi2 with 1 dof, non-central
    mu0 = 0
    nc = (mu - mu0) ** 2 / sigma**2
    pdf_true = ncx2.pdf(t_mu, df=1, nc=nc, loc=0, scale=1)
    pdf_computed = fstat.test_statistic_asymptotic_limit(t_mu, mu, mu0, sigma)
    assert np.allclose(pdf_true, pdf_computed, rtol=1e-5)


def test_ts_critical():
    data = [0, 1, 2, 3, 4]
    nevts = len(data)
    cdf = 0.9
    # the 1*sigma error for a binomial process can be estimated by sqrt((p*(1-p))/N)
    err_est = np.sqrt((cdf * (1 - cdf)) / nevts)
    (t_crit, t_crit_l0, t_crit_hi), _ = fstat.ts_critical(data)

    assert t_crit == [4]
    assert t_crit_hi == [4]
    assert t_crit_l0 == [4]


def test_p_value():
    data = np.array([0, 1, 2, 3, 4, 5])
    t_crit = 3
    p_exact = 2 / len(data)
    p, punc = fstat.p_value(data, t_crit)
    assert p == p_exact
    assert np.allclose(punc, np.sqrt((p_exact * (1 - p_exact)) / len(data)), rtol=1e-10)


def test_toy_ts_critical_p_value():
    data = np.array([0, 1, 2, 3, 4, 5])
    t_crit = 3
    p_exact = 3 / len(data)
    p = fstat.toy_ts_critical_p_value(data, t_crit)
    assert p == p_exact


def test_get_p_values():
    data1 = np.array([0, 1, 2, 3, 4, 5])
    t_crit = 3
    p_exact1 = 3 / len(data1)

    data2 = np.array([3, 4, 5, 6, 7, 8])
    p_exact2 = 1

    ps = fstat.get_p_values([data1, data2], [t_crit, t_crit])
    assert ps[0] == p_exact1
    assert ps[1] == p_exact2


def test_find_crossing():
    def line(x, m, b):
        return m * x + b

    def quad(x, a, b, c):
        return a * x**2 + b * x + c

    x = np.linspace(-1, 10, 100)
    li = line(x, 1, 1)
    t_crit = np.full_like(li, 2)
    cross_exact = 1
    cross = fstat.find_crossing(x, li, t_crit)

    assert np.allclose(cross_exact, cross, rtol=1e-6)

    q = quad(x, 1, -3, 2)
    cross_exact = [0, 3]
    cross = fstat.find_crossing(x, q, t_crit)
    assert np.allclose(cross_exact, cross, rtol=1e-6)
