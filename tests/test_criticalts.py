import numpy as np
from scipy.stats import chi2

from freqfit import Experiment
from freqfit.statistics import dkw_band, emp_cdf, ts_critical


def test_dkw():
    data = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]
    bins = [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
    ]

    cdf, _ = emp_cdf(data=data, bins=bins)

    dkw_lo, dkw_hi = dkw_band(cdf, len(data))

    hand_cdf = np.array(
        [
            0.2,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.82,
            0.84,
            0.86,
            0.88,
            0.9,
            0.92,
            0.94,
            0.96,
            0.98,
            1,
        ]
    )
    hand_dkw_hi = np.minimum(hand_cdf + 0.13537, 1)
    hand_dkw_lo = np.maximum(hand_cdf - 0.13537, 0)

    assert (hand_cdf == cdf).all()
    assert np.isclose(dkw_lo, hand_dkw_lo, atol=1e-5).all()
    assert np.isclose(dkw_hi, hand_dkw_hi, atol=1e-5).all()


def test_criticalts():
    true_S = 10.0
    p = Experiment.file("tests/config_test_highsignal.yaml", "experiment")

    # profile over the test statistic
    x = np.arange(0, 30, 0.1)
    y = np.zeros_like(x)
    for i, xx in enumerate(x):
        y[i], *_ = p.ts({"global_S": xx})

    toypars = p.profile({"global_S": true_S})["values"]
    toypars["global_S"] = true_S

    numtoys = 2000
    toyts = p.toy_ts(toypars, {"global_S": true_S}, num=numtoys)[0]
    nbins = 500

    (crit95, lo95, hi95), _ = ts_critical(
        toyts, bins=nbins, confidence=0.99, threshold=0.95
    )
    assert lo95 < chi2.ppf(0.95, df=1) and chi2.ppf(0.95, df=1) < hi95

    (crit90, lo90, hi90), _ = ts_critical(
        toyts, bins=nbins, confidence=0.99, threshold=0.9
    )
    assert lo90 < chi2.ppf(0.9, df=1) and chi2.ppf(0.9, df=1) < hi90

    (crit68, lo68, hi68), _ = ts_critical(
        toyts, bins=nbins, confidence=0.99, threshold=0.68
    )
    assert lo68 < chi2.ppf(0.68, df=1) and chi2.ppf(0.68, df=1) < hi68
