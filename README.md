# legendfreqfit
LEGEND 0v2b frequentist analysis

### inputs needed
per detector per partition
- runtime + unc.
- PSD efficiency + unc.
- mass
- active volume fraction + unc.
- LAr coincidence efficiency + unc.
- containment efficiency + unc.
- enrichment factor + unc.
- multiplicity efficiency + unc.
- data cleaning efficiency + unc.
- muon veto efficiency + unc.
- resolution + unc.
- energy offset + unc.

### TODO
- can we switch to `numba-stats`? (https://pypi.org/project/numba-stats/)
- right now, we have `numba` running parallel at the level of the model. Do we want that? It seems like maybe not... Especially since these will be very fast computations. Instead want the parallelization at a higher level.
- probably need to settle on some variable name conventions (Louis prefers more characters :) )
- maybe we want to compute `np.log(2)` etc. and use values directly? not sure how much faster (a lot in testing, but IDK how many times these actually need to be computed)
- models shouldn't multiply a bunch of little numbers together when returning a `logpdf` and instead sum up logs of pdfs
- models should have a function `loglikelihood` to compute the -2LL by summation instead of multiplying a bunch of little numbers together to get the likelihood
- models need a `density` to return form expected by `iminuit` ([https://scikit-hep.org/iminuit/notebooks/cost_functions.html)](https://scikit-hep.org/iminuit/notebooks/cost_functions.html#Extended-unbinned-fit))
