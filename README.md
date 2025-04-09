# freqfit


[![DOI](https://zenodo.org/badge/962842518.svg)](https://doi.org/10.5281/zenodo.15185401)


Unbinned frequentist analysis

---

### config format

Config files are `.yaml` files that contain several different dictionaries, described below. There are 5 primary dictionaries at the top level: `datasets`, `combined_datasets`, `parameters`, `constraints`, and `options`.

Note that the default value for options not explicitly provided is (usually) `False`.

```yaml
datasets: # the collection of datasets
  ds1: # name of dataset
    costfunction: ExtendedUnbinnedNLL # which cost function to use (only ExtendedUnbinnedNLL and UnbinnedNLL are supported)
    data: [...] # list of data
    model: freqfit.models.mymodel # name/location of model to use
    model_parameters: # parameter of the model: name given to this parameter
      a: a_ds1
      b: global_b # shared parameter names indicate the same parameter across different datasets

  ds2:
    try_to_combine: True # whether to attempt to combine this dataset with others (see further details below)
    combined_dataset: empty_ds # name of the group to attempt to join
    costfunction: ExtendedUnbinnedNLL
    data: [] # this dataset has no data
    model: freqfit.models.mymodel
    model_parameters:
      a: a_ds2
      b: global_b

  ds3:
    try_to_combine: True
    combined_dataset: empty_ds
    costfunction: ExtendedUnbinnedNLL
    data: []
    model: freqfit.models.mymodel
    model_parameters:
      a: a_ds3
      b: global_b

combined_datasets: # details of how we should attempt to combine datasets
  empty_ds: # name of the combined datasets
    costfunction: ExtendedUnbinnedNLL
    model: freqfit.models.mymodel # model for the combined group (must match the datasets being added for now)
    model_parameters:
      a: a_empty_ds
      b: global_b

parameters: # the collection of parameters
  a_ds1: # parameter name
    value: 0.0 # parameter value (note it will be a fixed parameter by default)
  a_ds2:
    value: 0.0 # initial guess for the parameter (since it is a fitted parameter)
    includeinfit: true  # whether to include in the fit
    fix_if_no_data: true  # if the dataset has no data, then we will skip trying to fit this parameter
    vary_by_constraint: true  # if we run toys, we will vary the true value of this parameter by the provided constraint
    fixed: false # you can fix parameters that are included in the fit (the only time this option is used, default is false)
  a_ds3:
    value: 0.0
  a_empty_ds: # parameters from combined_datasets also need to appear (nuisance parameters are varied at the level of datasets, not combined datasets)
    value_from_combine: true # will use the value from the combined datasets here, which is particularly important for fixed parameters
    includeinfit: false # in this case, we want to fix the value of this parameter after combining the datasets
  global_b:
    value: 0.0
    includeinfit: true

constraints: # the collection of constraints (a Gaussian constraint term in the likelihood)
  constraint_a_ds2: # name of the constraint
    parameters: a_ds2 # which parameter(s) the constraint applies to
    uncertainty: 1.0 # 1 sigma uncertainty on the parameter(s)
    values: 0.0 # should be the same value(s) as in `parameters`

options: # a collection of global options (only one option so far)
  try_to_combine_datasets: true # whether to attempt to combine datasets
  name: experiment name
  test_statistic: t_mu # can be "t_mu", "t_mu_tilde", "q_mu", "q_mu_tilde"
```

For `constraints`, there are some additional options for how to specify these constraints. You can provide a list of parameters, values, and uncertainties:

```yaml
constraints:
  myconstraint:
    parameters: [a, b, c]
    uncertainty: [1, 5, 3]
    values: [0, 10, 5]
```

or you can apply a single uncertainty to several parameters (note that each needs a separate value currently)

```yaml
constraints:
  myconstraint:
    parameters: [a, b, c]
    uncertainty: 1.0
    values: [0, 10, 5]
```

or you can provide a covariance matrix (which must be positive semi-definite, i.e. a proper covariance matrix). Note that the syntax changes from `uncertainty` to `covariance` in this case. (Using both will generate an error.)

```yaml
constraints:
  myconstraint:
    parameters: [a, b]
    covariance: [[1, -0.5], [-0.5, 1]]
    values: [2, 3]
```

Constraints will be combined into a single `NormalConstraint` as this dramatically improves computation time. This takes the form of a multivariate Gaussian with central values and covariance matrix calculated from the supplied constraints. Only constraints that refer to fit parameters are used. (Parameters in a single provided constraint must all be fit parameters.)

Constraints are also used to specify how nuisance parameters should be varied for toys. All parameters in a single constraint must be included as a parameter of a dataset, but do not necessarily need to be parameters in the fit.

You can specify independent datasets that should later be combined `combined_datasets`. This is useful for LEGEND where we have many, independent datasets with their own nuisance parameters. For our fit, it is much faster to simply combine all datasets that have no events (are empty). However, in generating our toys, we would like to vary the nuisance parameters and draw events randomly for all datasets. We therefore would like to combine datasets during our toys on the fly. Since, for each toy, we do no a prior know which datasets are empty and can be combined, we have written the code in such a way as to attempt to combine datasets. This is a very niche use case and probably only relevant for the 0vbb fit.

Test statistic definitions come from [G. Cowan, K. Cranmer, E. Gross, and O. Vitells, Eur. Phys. J. C 71, 1554 (2011)](https://doi.org/10.1140/epjc/s10052-011-1554-0).

Once you have a config file made, you can load it by doing

```python
from freqfit import Experiment

myexperiment = Experiment.file("myconfig.yaml")
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

### development help
You can install the repository using `pip` as an editable file. Just do `pip install -e` while inside of `freqfit/`.

### running on cenpa-rocks
1. Make sure you have a user directory in the LEGEND location on `eliza1`: `mkdir /data/eliza1/LEGEND/users/$USER`
2. Add the PYTHONUSERBASE to your `~/.bashrc`: `export PYTHONUSERBASE=/data/eliza1/LEGEND/users/$USER/pythonuserbase`
3. The code is located at `/data/eliza1/LEGEND/sw/freqfit`. In order to pip3 install it, run the following
4. Activate the singularity shell `singularity shell --bind /data/eliza1/LEGEND/:/data/eliza1/LEGEND/ /data/eliza1/LEGEND/sw/containers/python3-10.sif`
5. Pip3 install as an editable file. When located inside the `/data/eliza1/LEGEND/sw/freqfit` directory, run `pip install -e .` (NOTE: you may need to run the command `python3 -m pip install --upgrade pip` in order for this pip install to succeed)
6. Exit the singularity shell and run the code


### logging help
Just add these lines to your script.

```python
import logging
logging.basicConfig(level=logging.INFO) # or some other level
```

---

### Job Submission on CENPA-rocks
Job submission scripts for L200 are located in the `submission` subfolder in this repo. `submit_l200.sh` is a script that creates the S-grid to scan over by calling `l200_s_grid.py` and then submits jobs to the cluster to generate toys at those s-points by calling `run_l200.sh` which uses `l200_toys.py` to run the toys. Toys at 0-signal are generated and tested against this s-grid of alternative hypotheses using the `l200_brazil.sh` submission script which calls `l200_brazil.py` to actually run these 0-signal toys.
