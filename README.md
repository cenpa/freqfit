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
- implement PDG Chp 40 Eq 40.16 for Asimov dataset somewhere - check how to use with unbinned data.
- options for different test statistics (see Cowan)
- can we switch to `numba-stats`? (https://pypi.org/project/numba-stats/)
- right now, we have `numba` running parallel at the level of the model. Do we want that? It seems like maybe not... Especially since these will be very fast computations. Instead want the parallelization at a higher level.
- probably need to settle on some variable name conventions (Louis prefers more characters :) )
- maybe we want to compute `np.log(2)` etc. and use values directly? not sure how much faster (a lot in testing, but IDK how many times these actually need to be computed)
- models shouldn't multiply a bunch of little numbers together when returning a `logpdf` and instead sum up logs of pdfs
- models should have a function `loglikelihood` to compute the -2LL by summation instead of multiplying a bunch of little numbers together to get the likelihood
- [x] models need a `density` to return form expected by `iminuit` ([https://scikit-hep.org/iminuit/notebooks/cost_functions.html)](https://scikit-hep.org/iminuit/notebooks/cost_functions.html#Extended-unbinned-fit))
- [x] think we are going to have to use `_parameters` dict from `iminuit` see (https://github.com/scikit-hep/iminuit/issues/941) and (https://scikit-hep.org/iminuit/reference.html#iminuit.util.describe)
- [x] add a "isnuisance" parameter key to `parameters` `dict`
- [x] ~~think we need a class to hold an `iminuit` result, because you can't access the values of the other parameters from `Minuit.mnprofile()`, which we will need for our toys. So instead, have a class to hold the result of `Minuit.migrad()` and do separate calls for values of the parameters to profile over. This is rather ugly... but not sure what else to do.~~ use `utils.grab_results()` and it returns a `dict`
- [x] for `gaussian_on_uniform`, check that `loglikelihood` and `likelihood` are implemented correctly - the analysis window needs to be removed if it is included. This will affect the calculation of the likelihood if the passed `Es` are treated as bins. the likelihood and loglikehood here should be zero. (For now, these are removed.)

### development help
If you're using a Python virtual environment for development (`venv`), add something like these lines to `.venv/bin/activate` to add the `legendfreqfit` module to your `PYTHONPATH`.

```bash
PYTHONPATH="${PYTHONPATH}:/path/to/git/repo/legendfreqfit/src"
export PYTHONPATH
```


### running on cenpa-rocks
1. Make sure you have a user directory in the LEGEND location on `eliza1`: `mkdir /data/eliza1/LEGEND/users/$USER`
2. Add the PYTHONUSERBASE to your `~/.bashrc`: `export PYTHONUSERBASE=/data/eliza1/LEGEND/users/$USER/pythonuserbase`
3. The code is located at `/data/eliza1/LEGEND/sw/legendfreqfit`. In order to pip3 install it, run the following
4. Activate the singularity shell `singularity shell --bind /data/eliza1/LEGEND/:/data/eliza1/LEGEND/ /data/eliza1/LEGEND/sw/containers/python3-10.sif`
5. Pip3 install as an editable file. When located inside the `/data/eliza1/LEGEND/sw/legendfreqfit` directory, run `pip install -e .` (NOTE: you may need to run the command `python3 -m pip install --upgrade pip` in order for this pip install to succeed)
6. Exit the singularity shell and run the code

### "config" format

A `.yaml` file should be set up like `tests/config_test.yaml`. There are a few things in there that will raise warnings, which I was trying to test.
