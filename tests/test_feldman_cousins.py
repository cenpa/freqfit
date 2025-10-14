# For a one-bin Poisson model, the test statistic distribution is analyitically computable
from freqfit import Workspace
from scipy.stats import poisson, norm
import freqfit.statistics as fstat
import numpy as np

def test_compare_to_FC():
    # for pairs of true signal and true background rates, with a hypothesis test at the true signal rate
    # compare a generated test statistic distribution to the directly computed one

    def compute_FC_pdf(s_true, b_true):
        """
        Compute the test statistic PDF for a one-bin Poisson model with known background when the hypothesis being tested is s=s_true
        """
        ts_exact = []
        N = np.arange(0,20) # observed number of counts
        pdf_exact = poisson.pmf(N, s_true+b_true)
        s_hat = np.maximum(N-b_true, 0)
        ts_exact = 2*((s_true-s_hat) - N*np.log((s_true+b_true)/(s_hat+b_true)))
        ts_exact, pdf_exact = zip(*sorted(zip(ts_exact, pdf_exact))) # sort the ts
        return np.array(ts_exact), np.array(pdf_exact)


    # Make a mesh grid of pairs
    X, Y = np.meshgrid(np.arange(1,5), np.arange(1,5))
    cartesian_product = np.stack([X.ravel(), Y.ravel()], axis=1)

    # now check that the uncertainty bands from the generated cdf contains the true cdf
    for test_pairs in cartesian_product:
        S = test_pairs[0]
        B = test_pairs[1]

        config = Workspace.load_config("FC_config.yaml")
        config["parameters"]["S"]["value"] = S
        config["parameters"]["B"]["value"] = B
        ws = Workspace(config)

        ts_obs,_ = ws.toy_ts({"S":S}, {"S":S}, num=100, seeds=np.arange(100))

        ts_exact, pdf_exact = compute_FC_pdf(S, B)
        cdf_exact = np.cumsum(pdf_exact)

        cdf_computed, _ = fstat.emp_cdf(ts_obs[0], [*ts_exact-1e-4,ts_exact[-1]+1])
        cdf_lo, cdf_hi = fstat.dkw_band(cdf_computed, len(ts_obs[0]), CL=0.9)

        assert np.all((cdf_exact>=cdf_lo) & (cdf_exact<=cdf_hi))
