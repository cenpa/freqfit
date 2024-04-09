w# legendfreqfit
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
- think we are going to have to use `_parameters` dict from `iminuit` see (https://github.com/scikit-hep/iminuit/issues/941) and (https://scikit-hep.org/iminuit/reference.html#iminuit.util.describe)
- can we switch to `numba-stats`? (https://pypi.org/project/numba-stats/)
- right now, we have `numba` running parallel at the level of the model. Do we want that? It seems like maybe not... Especially since these will be very fast computations. Instead want the parallelization at a higher level.
- probably need to settle on some variable name conventions (Louis prefers more characters :) )
- maybe we want to compute `np.log(2)` etc. and use values directly? not sure how much faster (a lot in testing, but IDK how many times these actually need to be computed)
- models shouldn't multiply a bunch of little numbers together when returning a `logpdf` and instead sum up logs of pdfs
- models should have a function `loglikelihood` to compute the -2LL by summation instead of multiplying a bunch of little numbers together to get the likelihood
- [x] models need a `density` to return form expected by `iminuit` ([https://scikit-hep.org/iminuit/notebooks/cost_functions.html)](https://scikit-hep.org/iminuit/notebooks/cost_functions.html#Extended-unbinned-fit))

### "config" format

A `.yaml` file should be set up like the following. (Need to adjust how `Dataset` and `Superset` take their parameters.)

```yaml
- "datasets":
    - "datasetname1"
        - "data": [2039.0, 2145.1, 1956.7, 2012.9]
        - "model": gaussian_on_uniform
        - "model_parameters":
            - "S": "global_S"
            - "BI": "BI_datasetname1"
            - "delta": "delta_datasetname1"
            - "sigma": "sigma_datasetname1"
            - "eff": "eff_datasetname1"
            - "exp": "exp_datasetname1"

    - "datasetname2"
        - "data": [2045.1, 1966.5, 2112.9]
        - "model": gaussian_on_uniform
        - "model_parameters":
            - "S": "global_S"
            - "BI": "BI_datasetname2"
            - "delta": "delta_datasetname2"
            - "sigma": "sigma_datasetname2"
            - "eff": "eff_datasetname2"
            - "exp": "exp_datasetname2"

- "parameters":
    - "global_S":
        - "includeinfit": True
        - "limits": (0, None)
        - "value": ~
    - "BI_datasetname1":
        - "includeinfit": True
        - "limits": (0, None)
        - "value": ~
    - "delta_datasetname1"
        - "includeinfit": False
        - "limits": ~
        - "value": 0.0
    - "sigma_datasetname1"
        - "includeinfit": False
        - "limits": ~
        - "value": 1.0
    - "eff_datasetname1"
        - "includeinfit": False
        - "limits": ~
        - "value": 1.0
    - "exp_datasetname1"
        - "includeinfit": False
        - "limits": ~
        - "value": 1.0
    - "BI_datasetname2":
        - "includeinfit": True
        - "limits": (0, None)
        - "value": ~
    - "delta_datasetname2"
        - "includeinfit": False
        - "limits": ~
        - "value": 0.0
    - "sigma_datasetname2"
        - "includeinfit": False
        - "limits": ~
        - "value": 1.0
    - "eff_datasetname2"
        - "includeinfit": False
        - "limits": ~
        - "value": 1.0
    - "exp_datasetname2"
        - "includeinfit": False
        - "value": 1.0

# collection of `NormalConstraint`
- "constraints":
    - "constraint1"
        - "parameters": [] # list of the parameters in the order of `values` and `covariance`
        - "values": [] # list of the central value of the parameters
        - "covariance": [] # covariance matrix of the parameters
```

### development help
If you're using a Python virtual environment for development (`venv`), add something like these lines to `.venv/bin/activate` to add the `legendfreqfit` module to your `PYTHONPATH`.

```bash
PYTHONPATH="${PYTHONPATH}:/path/to/git/repo/legendfreqfit/src"
export PYTHONPATH
```
