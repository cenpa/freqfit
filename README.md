# legendfreqfit
LEGEND 0v2b frequentist analysis

---

### config format

Config files are `.yaml` files that contain several different dictionaries, described below. There are 4 primary dictionaries at the top level: `datasets`, `parameters`, `constraints`, and `options`. 

Note that the default value for options not explicitly provided is usually `False`.

```yaml
datasets: # the collection of datasets
  ds1: # name of dataset
    costfunction: ExtendedUnbinnedNLL # which cost function to use (only ExtendedUnbinnedNLL and UnbinnedNLL are supported)
    data: [...] # list of data
    model: legendfreqfit.models.mymodel # name/location of model to use
    model_parameters: # parameter of the model: name given to this parameter
      a: a_ds1
      b: global_b # shared parameter names indicate the same parameter across different datasets

  ds2:
    costfunction: ExtendedUnbinnedNLL
    data: [...] 
    model: legendfreqfit.models.mymodel
    model_parameters:
      a: a_ds2
      b: global_par2 # shared parameter names indicate the same parameter across different datasets
  

parameters: # the collection of parameters
  a_ds1: # parameter name
    value: 0.0 # parameter value (note it will be a fixed parameter by default)
  a_ds2:
    value: 0.0 # initial guess for the parameter (since it is a fitted parameter)
    includeinfit: true  # whether to include in the fit
    nuisance: true  # whether this is a nuisance parameter
    fix_if_no_data: true  # if the dataset has no data, then we will skip trying to fit this parameter
    vary_by_constraint: true  # if we run toys, we will vary the true value of this parameter by the provided constraint
    fixed: false # you can fix parameters that are included in the fit (the only time this option is used)
  global_b:
    value: 0.0 
    includeinfit: true 
    nuisance: false

constraints: # the collection of constraints (a Gaussian constraint term in the likelihood)
  constraint_a_ds2: # name of the constraint
    parameters: a_ds2 # which parameter(s) the constraint applies to
    uncertainty: 1.0 # 1 sigma uncertainty on the parameter(s)
    values: 0.0 # should be the same value(s) as in `parameters`

options: # a collection of global options (only one option so far)
  combine_constraints: true # whether to combine constraints into a single `NormalConstraint` to reduce computation time
```

For `constraints`, there are some additional options for how to specify these constraints. You can provide a list of parameters, values, and uncertainties:

```yaml
constraints:
  myconstraint:
    parameters: [a, b, c]
    uncertainty: [1, 5, 3]
    values: [0, 10, 5]
```

or you can apply a single uncertainty and value to several parameters

```yaml
constraints:
  myconstraint:
    parameters: [a, b, c]
    uncertainty: 1.0
    values: 0.0
```

or you can provide a covariance matrix (which must be positive semi-definite, i.e. a proper covariance matrix). Note that the syntax changes from `uncertainty` to `covariance` in this case. (Using both will generate an error.)

```yaml
constraints:
  myconstraint:
    parameters: [a, b]
    covariance: [[1, -0.5], [-0.5, 1]]
    values: [2, 3]
```

---

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

---

### TODO
- GERDA: figure out uncertainties
- do we need to label parameters as both nuisance and includeinfit?
- add way to combine experiments, can separately fit the test statistics for each experiment and then combine them through a convolution of their pdfs?
- need a way to run the analysis automatically for an experiment.
- implement PDG Chp 40 Eq 40.16 for Asimov dataset somewhere - check how to use with unbinned data.
- options for different test statistics (see Cowan)
- right now, we have `numba` running parallel at the level of the model. Do we want that? It seems like maybe not... Especially since these will be very fast computations. Instead want the parallelization at a higher level.
- probably need to settle on some variable name conventions (Louis prefers more characters :) )
- maybe we want to compute `np.log(2)` etc. and use values directly? not sure how much faster (a lot in testing, but IDK how many times these actually need to be computed)
- [x] I think probably we need to rename the models to something more specific to 0vbb
- [x] separate event energies into two types: 1. those close to Qbb and 2. those not close to Qbb. Can fit those close to Qbb using Gaussian + uniform w/ nuisance parameters and those not close with just a uniform. Should improve computation speed but requires two different models and then separate the data based on the two different models. The separation of the events into these two different categories should be done in the config file. Just need to write a model for the purely uniform case!

  Actually, this becomes rather tricky, because we need to draw rvs from both models but have them appropriately pulled from. So I guess the correct thing to do would be to specify different analysis windows for them that are not overlapping.

- [x] made models for this /\
- [x] models need a `density` to return form expected by `iminuit` ([https://scikit-hep.org/iminuit/notebooks/cost_functions.html)](https://scikit-hep.org/iminuit/notebooks/cost_functions.html#Extended-unbinned-fit))
- [x] think we are going to have to use `_parameters` dict from `iminuit` see (https://github.com/scikit-hep/iminuit/issues/941) and (https://scikit-hep.org/iminuit/reference.html#iminuit.util.describe)
- [x] add a "isnuisance" parameter key to `parameters` `dict`
- [x] ~~think we need a class to hold an `iminuit` result, because you can't access the values of the other parameters from `Minuit.mnprofile()`, which we will need for our toys. So instead, have a class to hold the result of `Minuit.migrad()` and do separate calls for values of the parameters to profile over. This is rather ugly... but not sure what else to do.~~ use `utils.grab_results()` and it returns a `dict`
- [x] for `gaussian_on_uniform`, check that `loglikelihood` and `likelihood` are implemented correctly - the analysis window needs to be removed if it is included. This will affect the calculation of the likelihood if the passed `Es` are treated as bins. the likelihood and loglikehood here should be zero. (For now, these are removed.)
- ~~models shouldn't multiply a bunch of little numbers together when returning a `logpdf` and instead sum up logs of pdfs~~
- ~~models should have a function `loglikelihood` to compute the -2LL by summation instead of multiplying a bunch of little numbers together to get the likelihood~~
- [x] add logging?
- [x] ~~can we switch to `numba-stats`? (https://pypi.org/project/numba-stats/)~~ NO
- [x] get warnings to go both to stdout and log
- [x] double check that normalconstraint wants covariance matrix / what to put if only for a single variable? - `NormalConstraint` has `error` parameter that takes a list of 1 sigma uncertainties or a 2D covariance matrix in the case of multiple parameters. For a single parameter, must use uncertainty.
- [x] compare parameter values and values in constraints to make sure they agree? put the value only in one place?
- [x] check constraints to make sure that parameters are not duplicated across constraints?
- [x] it's possible to pass in a config file where parameters are not labeled kind of correctly and iminuit will not use all of the parameters that the user might think it is using. This can be maybe problematic. The only way to be sure what parameters are being used right now is to look at the fitparameters of the Experiment/Superset or to go into each Dataset and look through their `model_parameters`. Some thinking needed...
- [x] MJD: combine exposure and efficiency uncertainties

---

### development help
You can install the repository using `pip` as an editable file. Just do `pip install -e` while inside of `legendfreqfit/`.

### running on cenpa-rocks
1. Make sure you have a user directory in the LEGEND location on `eliza1`: `mkdir /data/eliza1/LEGEND/users/$USER`
2. Add the PYTHONUSERBASE to your `~/.bashrc`: `export PYTHONUSERBASE=/data/eliza1/LEGEND/users/$USER/pythonuserbase`
3. The code is located at `/data/eliza1/LEGEND/sw/legendfreqfit`. In order to pip3 install it, run the following
4. Activate the singularity shell `singularity shell --bind /data/eliza1/LEGEND/:/data/eliza1/LEGEND/ /data/eliza1/LEGEND/sw/containers/python3-10.sif`
5. Pip3 install as an editable file. When located inside the `/data/eliza1/LEGEND/sw/legendfreqfit` directory, run `pip install -e .` (NOTE: you may need to run the command `python3 -m pip install --upgrade pip` in order for this pip install to succeed)
6. Exit the singularity shell and run the code


