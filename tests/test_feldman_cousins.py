# For a one-bin Poisson model, the test statistic distribution is analytically computable
import numpy as np
import pytest
from scipy.stats import poisson

import freqfit.statistics as fstat
from freqfit import Workspace


def test_compare_to_FC():  # noqa: N802
    # for pairs of true signal and true background rates, with a hypothesis test at the true signal rate
    # compare a generated test statistic distribution to the directly computed one

    def compute_FC_pdf(s_true, b_true):  # noqa:N802
        """
        Compute the test statistic PDF for a one-bin Poisson model with known background when the hypothesis being tested is s=s_true
        """
        ts_exact = []
        N = np.arange(0, 20)  # observed number of counts
        pdf_exact = poisson.pmf(N, s_true + b_true)
        s_hat = np.maximum(N - b_true, 0)
        ts_exact = 2 * (
            (s_true - s_hat) - N * np.log((s_true + b_true) / (s_hat + b_true))
        )
        ts_exact, pdf_exact = zip(*sorted(zip(ts_exact, pdf_exact)))  # sort the ts
        return np.array(ts_exact), np.array(pdf_exact)

    # Make a mesh grid of pairs
    X, Y = np.meshgrid(np.arange(1, 5), np.arange(1, 5))
    cartesian_product = np.stack([X.ravel(), Y.ravel()], axis=1)

    # now check that the uncertainty bands from the generated cdf contains the true cdf
    for test_pairs in cartesian_product:
        S = test_pairs[0]
        B = test_pairs[1]

        config = Workspace.load_config("tests/FC_config.yaml")
        config["parameters"]["S"]["value"] = S
        config["parameters"]["B"]["value"] = B
        ws = Workspace(config)

        ts_obs, _ = ws.toy_ts({"S": S}, {"S": S}, num=100, seeds=np.arange(100))

        ts_exact, pdf_exact = compute_FC_pdf(S, B)
        cdf_exact = np.cumsum(pdf_exact)

        cdf_computed, _ = fstat.emp_cdf(ts_obs[0], [*ts_exact - 1e-4, ts_exact[-1] + 1])
        cdf_lo, cdf_hi = fstat.dkw_band(cdf_computed, len(ts_obs[0]), CL=0.9)

        assert np.all((cdf_exact >= cdf_lo) & (cdf_exact <= cdf_hi))


def q_tilde_workspace(extra_poi=()):
    config = Workspace.load_config("tests/FC_config.yaml")
    config["options"]["test_statistic"] = "t_and_q_tilde"
    for parname in extra_poi:
        config["parameters"][parname]["poi"] = True
    return Workspace(config)


def test_q_tilde_uses_poi():
    # only the parameter of interest decides whether q is set to 0, wherever it is in the dict
    experiment = q_tilde_workspace().experiment
    experiment.best = {"values": {"S": 2.0, "B": 10.0}}

    assert experiment._poi_above_tested({"S": 1.0, "B": 6.0})
    assert experiment._poi_above_tested({"B": 6.0, "S": 1.0})
    assert not experiment._poi_above_tested({"S": 3.0, "B": 6.0})  # B is above its value, S is not
    assert not experiment._poi_above_tested({"B": 6.0})


def test_q_tilde_multiple_poi():
    # with several parameters of interest, q is set to 0 if any of the tested ones is above its tested value
    experiment = q_tilde_workspace(extra_poi=("B",)).experiment
    assert experiment.parameters.poi == ["B", "S"]
    experiment.best = {"values": {"S": 2.0, "B": 10.0}}

    assert experiment._poi_above_tested({"S": 3.0, "B": 6.0})
    assert experiment._poi_above_tested({"S": 1.0, "B": 12.0})
    assert not experiment._poi_above_tested({"S": 3.0, "B": 12.0})
    assert not experiment._poi_above_tested({"S": 3.0})  # B is a poi but is not being tested


def test_q_tilde_zero_at_S_zero():
    # q_tilde tested at S=0 must be 0 for every toy when S >= 0: either S_hat > 0 (so q is set to 0)
    # or S_hat = 0 (so t is 0 too)
    ws = q_tilde_workspace()
    ts_obs, _ = ws.toy_ts({"S": 4.0}, {"S": 0.0}, num=100, seeds=np.arange(100))
    t, q = ts_obs[0][:, 0], ts_obs[0][:, 1]

    assert np.any(t > 1)  # the toys have signal, so the check below is not trivial
    assert np.allclose(q, 0, atol=1e-6)


@pytest.mark.parametrize("test_statistic", ["t_mu", "t_mu_tilde", "q_mu", "q_mu_tilde", "t_and_q_tilde"])
@pytest.mark.parametrize("poi", [None, False])
def test_requires_poi(test_statistic, poi):
    # a parameter of interest is required whatever the test statistic, and `poi: false` does not count
    config = Workspace.load_config("tests/FC_config.yaml")
    config["options"]["test_statistic"] = test_statistic
    if poi is None:
        del config["parameters"]["S"]["poi"]
    else:
        config["parameters"]["S"]["poi"] = poi
    with pytest.raises(ValueError, match="parameter of interest"):
        Workspace(config)
