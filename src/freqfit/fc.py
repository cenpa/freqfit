"""A module for computing Feldman-Cousins Poisson confidence intervals.

Computes unified confidence intervals for Poisson counting experiments with a
known background level based on the Feldman-Cousins ordering principle:
Feldman and Cousins, Phys. Rev. D 57, 3873 (1998).

Uses analytical solutions and inversions to perform all computations on-the-fly,
so no memory allocation or setup is required: just call the desired function
with the appropriate parameters.

The results agree to 3 decimal places with nearly all entries in Table IV of the
Feldman-Cousins paper. Deviations were only observed for very high background
and very low observed counts. Investigations by the author revealed these
calculations to likely be more accurate than the published versions.

Similar / related projects:
- https://docs.gammapy.org/dev/stats/feldman_cousins.html
Requires lengthy precalculation of coverage maps based on user-supplied tables,
but handles other PDFs besides the Poisson distribution.
- https://github.com/mxmeier/feldman_cousins
Similar in functionality and usage to the gammapy package.

Example:

n = 3
bg = 3
cl = 0.90
mu_step = 0.001
print(fc.get_upper_limit(n, bg, cl, mu_step))

Output: 4.424
FC Table 4: 4.42
"""


import numpy as np
from scipy.stats import poisson
from scipy.special import erfinv


def get_lnR(n: int, mu: float, bg: float) -> float:
    """Computes the log of the FC ratio.

    Args:
        n: observed counts
        mu: expected signal counts
        bg: expected background counts

    Returns:
        lnR: log of L(n|mu+bg) / L(n|mu_best + bg)
    """
    if n == 0 and bg == 0: return -mu
    if mu + bg == 0:
        if n == 0: return 0
        return -np.inf
    mu_best = np.maximum(n-bg, 0)
    return n*(np.log(mu+bg) - np.log(mu_best+bg)) + mu_best - mu


def get_R(n, mu, bg):
    """Exponentiation of get_lnR"""
    return np.exp(get_lnR(n, mu, bg))


def get_equalR_mu(nLo: int, nHi: int, bg: float) -> float:
    """Finds the mu that solves R(nLo, mu, bg) = R(nHi, mu, bg).

    This seemingly unnecessary operation offers a huge speed up for computing FC
    limits. It can be computed analytically. The arguments nLo and nHi form sort
    of a "test" coverage region, and one searches for the mu returned by this
    function that has the "right" likelihood ratio value.

    Args:
        nLo: lower limit of "test" coverage region
        nHi: upper limit of "test" coverage region
        bg: expected background counts

    Returns:
        mu, the expected signal counts that solves
        R(nLo, mu, bg) = R(nHi, mu, bg)
    """
    if nLo >= nHi or nHi < bg: return 0
    if nLo == 0: return nHi / np.exp(1 - bg/nHi) - bg
    mu1 = np.maximum(nLo-bg, 0) + bg
    mu = np.exp((nHi*np.log(nHi)-nHi - nLo*np.log(mu1)+mu1)/(nHi-nLo)) - bg
    return mu


def get_edges(mu: float, bg: float, cl: float = 0.90) -> (int, int, float):
    """Finds the limits of the coverage patch (at cl) for mu+bg.

    Args:
        mu: expected signal counts
        bg: expected background counts
        cl: confidence level for the coverage patch, 90% by defaul

    Returns:
        The tuple (nLo, nHi, cov) containing the coverage and the lower and
        upper observed counts for the coverage patch of mu: solves Σ_nLo^nHi
        P(n|mu+bg) >= cl
    """
    if mu == 0: 
        nHi = poisson.ppf(cl, bg)
        return 0, nHi, poisson.cdf(nHi, bg)
    nLo = nHi = int(mu+bg)
    pLo = pHi = coverage = poisson.pmf(nLo, mu+bg)
    while coverage < cl:
        if nLo > 0 and get_lnR(nLo-1, mu, bg) > get_lnR(nHi+1, mu, bg):
            pLo *= nLo/(mu+bg)
            nLo -= 1
            coverage += pLo
        else:
            nHi += 1
            pHi *= (mu+bg)/nHi
            coverage += pHi
    return nLo, nHi, coverage


def get_lnRc(mu: float, bg: float, cl: float = 0.90) -> float:
    """Computes ln(Rc) for this mu.

    The critical ratio Rc is the minimum value of the likelihood ratio for n
    within the confidence interval for the given mu. The sum of P(n|mu+bg) for n
    with R >= Rc is as-close-as-possible but not lower than cl.

    Args:
        mu: expected signal counts
        bg: expected background counts
        cl: confidence level for the interval

    Returns:
        lnRc, the log of the critical ratio Rc
    """
    nLo, nHi, _ = get_edges(mu, bg, cl)
    return np.minimum(get_lnR(nLo, mu, bg), get_lnR(nHi, mu, bg))


def get_Rc(mu: float, bg: float, cl: float = 0.90) -> float:
    """Exponentiation of get_lnRc"""
    return np.exp(get_lnRc(mu, bg, cl))


def get_upper_limit_brute_force(n: int,
                                bg: float,
                                cl: float = 0.90,
                                mu_step: float = 0.001) -> float:
    """Gets FC upper limits, brute force method.

    Saved for posterity. This is a slow but straightforward computation of the
    Feldman-Cousins upper limit at confidence level cl when n counts are
    observed and bg background counts were expected.

    Args:
        n: observed counts
        bg: expected background counts
        cl: confidence level for the upper limit
        mu_step: precision for computing the upper limit

    Returns:
        mu, the upper limit on the signal strength, accurate to ±mu_step and
        rounded to the nearest interval of mu_step.
    """
    cl_sigma = erfinv(cl)*np.sqrt(2)
    mu = n + cl_sigma*np.sqrt(n)
    nLo, _, _ = get_edges(mu, bg, cl)
    while nLo > n and mu > 0:
        mu -= mu_step
        if mu < 0: mu = 0
        nLo, _, _ = get_edges(mu, bg, cl)
    while nLo <= n:
        mu += mu_step
        nLo, _, _ = get_edges(mu, bg, cl)
    return np.round(mu/mu_step)*mu_step


def get_upper_limit(n: int,
                    bg: float,
                    cl: float = 0.90,
                    mu_step: float = 0.001) -> float:
    """Gets FC upper limits, fast method.

    This is a fast but tricky computation of the Feldman-Cousins upper limit at
    confidence level cl when n counts are observed and bg background counts were
    expected.

    Args:
        n: observed counts
        bg: expected background counts
        cl: confidence level for the upper limit
        mu_step: precision for computing the upper limit

    Returns:
        mu, the upper limit on the signal strength, accurate to ±mu_step and
        rounded to the nearest interval of mu_step.
    """
    # FC is based on the ordering parameter R = P(n|mu+b) / P(n|mu_best+b)
    # where mu_best = max(0, n-b). One way to understand the confidence
    # intervals is to think of each mu as having a critical value R_c(mu|b) for
    # the ordering parameter, so that if n has R(n|mu,b) >= R_c(mu|b) then it is
    # inside the interval for that mu. When an experiment is done with known b
    # and gets result n, one then just needs to find the minimal / maximal mu
    # with R(mu|n,b) >= R_c(mu|b): these are the lower and upper FC limits.
    # However, computing R_c(mu|b) is time intensive, as it requires integrating
    # the Poisson distribution with variable limits.
    #
    # Consider the family of curves R(mu|n,b) as a function of mu:
    # R(mu|n,b) = (mu+b)^n exp(-(mu+b)) * const
    # These curves are monotonically decreasing for n < b, and for n > b rise to
    # a maximum at mu+b = n before falling monotonically beyond that.  The
    # R_c(mu|b) have to occur somewhere along these curves. R_c hops from curve
    # to curve for different n to achieve the coverage closest to CL without
    # undercovering. So the -last- R(mu|n,b) that is >= R_c(mu|b) (giving mu_UL)
    # will either be at or just above a point where R(mu|n,b) = R(mu|n',b) for
    # some n' > n. Moreover, these points require that R(mu|n,b) be falling and
    # R(mu|n',b) be rising, and the latter can only occur for n > b. A similar
    # argument can be made for lower limits.
    #
    # So: the upper limits can be more efficiently found by first stepping up in
    # n' from max(n+1, ceil(b)), and for each computing the mu' such that
    # R(mu|n,b) = R(mu|n',b), which is a one-step analytic calculation. One then
    # computes R_c(mu'), and stops stepping up in n' when R(mu'|n,b) < R_c(mu').
    # Having found a candidate for mu_UL, one can peek forward in mu to see if
    # higher mu have lower R_c. If so, a binary search can be performed between
    # there and the next higher mu' to find mu_UL.
    nprime = np.maximum(n+1, np.ceil(bg)) - 1
    lnR = lnRc = 0
    while lnR >= lnRc:
        nprime += 1
        mu = get_equalR_mu(n, nprime, bg)
        lnR = get_lnR(n, mu, bg)
        lnRc = get_lnRc(mu, bg, cl)
    nprime -= 1
    muLo = get_equalR_mu(n, nprime, bg) + mu_step
    lnR = get_lnR(n, muLo, bg)
    lnRc = get_lnRc(muLo, bg, cl)
    if lnR < lnRc: return muLo - mu_step
    # if we get here, do a binary search between mu(n') and mu(n'+1)
    muHi = get_equalR_mu(n, nprime+1, bg)
    while muHi-muLo > mu_step:
        mu = (muLo + muHi)/2
        lnR = get_lnR(n, mu, bg)
        lnRc = get_lnRc(mu, bg, cl)
        if lnR < lnRc: muHi = mu
        else: muLo = mu
    return np.round(muLo/mu_step)*mu_step


def get_lower_limit(n: int,
                    bg: float,
                    cl: float = 0.90,
                    mu_step: float = 0.001) -> float:
    """Gets FC lower limits, fast method.

    This is a fast but tricky computation of the Feldman-Cousins lower limit at
    confidence level cl when n counts are observed and bg background counts were
    expected. See code comments in get_upper_limit for the algorithm.

    Args:
        n: observed counts
        bg: expected background counts
        cl: confidence level for the lower limit
        mu_step: precision for computing the lower limit

    Returns:
        mu, the lower limit on the signal strength, accurate to ±mu_step and
        rounded to the nearest interval of mu_step.
    """
    # First check n = 0 or mu = 0, to see if we can quickly exit.
    if n == 0: return 0

    mu = 0
    lnR = get_lnR(n, mu, bg)
    lnRc = get_lnRc(mu, bg, cl)
    if lnR >= lnRc: return 0

    # Now start from n and walk backwards. We should not get to zero!
    nprime = n
    lnR = lnRc = 0
    while lnR >= lnRc and nprime > 0:
        nprime -= 1
        mu = get_equalR_mu(nprime, n, bg)
        lnR = get_lnR(n, mu, bg)
        lnRc = get_lnRc(mu, bg, cl)
    if nprime > 0 or lnR < lnRc: nprime += 1
    muHi = get_equalR_mu(nprime, n, bg) - mu_step
    lnR = get_lnR(n, muHi, bg)
    lnRc = get_lnRc(muHi, bg, cl)
    if lnR < lnRc: return muHi + mu_step
    # if we get here, do a binary search between mu(n'-1) and mu(n')
    if nprime > 0: muLo = get_equalR_mu(nprime-1, n, bg)
    else: muLo = 0
    while muHi-muLo > mu_step:
        mu = (muLo + muHi)/2
        lnR = get_lnR(n, mu, bg)
        lnRc = get_lnRc(mu, bg, cl)
        if lnR < lnRc: muLo = mu
        else: muHi = mu
    return np.round(muLo/mu_step)*mu_step


def median_limit_sensitivity(bg: float,
                             cl: float = 0.90,
                             mu_step: float = 0.001) -> float:
    """Computes the FC median limit sensitivity.

    The median limit sensitivity is the upper limit an experiment with
    background level bg would result in if the observed number of counts were
    equal to the median of the expected count distribution when the signal is
    too small to be measured (mu << bg)

    Args:
        bg: expected background counts
        cl: confidence level for the upper limit
        mu_step: precision for computing the lower limit

    Returns:
        mu, the median limit sensitivity, accurate to ±mu_step and rounded to
        the nearest interval of mu_step.
    """
    return get_upper_limit(poisson.median(bg), bg, cl, mu_step)


def mean_limit_sensitivity(bg: float,
                           cl: float = 0.90,
                           mu_step: float = 0.001) -> float:
    """Computes the FC mean limit sensitivity.

    The mean limit sensitivity is the average of the upper limits an ensemeble
    of identical experiment with background level bg would result in if the
    signal is too small to be measured (mu << bg) so that the observed counts
    would be Poisson distributed with mean = bg.

    Args:
        bg: expected background counts
        cl: confidence level for the upper limit
        mu_step: precision for computing the lower limit

    Returns:
        mu, the mean limit sensitivity, accurate to ±mu_step and rounded to the
        nearest interval of mu_step.
    """
    get_upper_limit_vec = np.vectorize(get_upper_limit)
    def f(n):
        return get_upper_limit_vec(n, bg, cl, mu_step)
    return poisson.expect(f, [bg])


def median_discovery_sensitivity(bg: float,
                                 cl: float = 0.9973,
                                 mu_step: float = 0.001) -> float:
    """Computes the discovery sensitivity.

    The discovery sensitivity is the smallest signal strength mu such that an
    ensemeble of identical experiment with background level bg would report a
    discovery (mu > 0) at cl at least 50% of the time.

    This concept does not make use of the FC ordering parameter. However, it can
    be quickly calculated using the FC construction, because the confidence band
    for mu=0 corresponds to the background-only hypothesis. So the first mu that
    gives a poisson median n_med that is not inide this interval is the
    discovery sensitivity.

    Args:
        bg: expected background counts
        cl: confidence level for the upper limit
        mu_step: precision for computing the lower limit

    Returns:
        mu, the discovery sensitivity, accurate to ±mu_step and rounded to the
        nearest interval of mu_step.
    """
    _, nHi, _ = get_edges(0, bg, cl)
    n_crit = nHi + 1
    mu = n_crit
    dmu = 1
    # now do a binary search for mu within n_crit ± dmu
    while dmu > mu_step:
        if poisson.median(mu) < n_crit:
            mu += dmu
        else:
            mu -= dmu
        if mu < 0: mu = 0
        dmu /= 2
    return np.round((mu - bg)/mu_step)*mu_step
