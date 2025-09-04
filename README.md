# freqfit


[![DOI](https://zenodo.org/badge/962842518.svg)](https://doi.org/10.5281/zenodo.15185401)


Unbinned frequentist analysis

---

READ ME IS A WORK IN PROGRESS

### config format

Config files are `.yaml` files that contain several different dictionaries, described below. There are 5 primary dictionaries at the top level: `datasets`, `combined_datasets`, `parameters`, `constraints`, and `options`.

Constraints will be combined into a single `NormalConstraint` as this dramatically improves computation time. This takes the form of a multivariate Gaussian with central values and covariance matrix calculated from the supplied constraints. Only constraints that refer to fit parameters are used. (Parameters in a single provided constraint must all be fit parameters.)

Constraints are also used to specify how nuisance parameters should be varied for toys. All parameters in a single constraint must be included as a parameter of a dataset, but do not necessarily need to be parameters in the fit.

You can specify independent datasets that should later be combined `combined_datasets`. This is useful for LEGEND or other quasi-background free experiments where we have many, independent datasets with their own nuisance parameters. For our fit, it is much faster to simply combine all datasets that have no events (are empty). However, in generating our toys, we would like to vary the nuisance parameters and draw events randomly for all datasets. We therefore would like to combine datasets during our toys on the fly. Since, for each toy, we do no a prior know which datasets are empty and can be combined, we have written the code in such a way as to attempt to combine datasets. This is a very niche use case and probably only relevant for the 0vbb fit.

Test statistic definitions come from [G. Cowan, K. Cranmer, E. Gross, and O. Vitells, Eur. Phys. J. C 71, 1554 (2011)](https://doi.org/10.1140/epjc/s10052-011-1554-0).

Once you have a config file made, you can load it by doing

```python
from freqfit import Workspace

ws = Workspace.from_file("myconfig.yaml")
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

---
# negative of the exponent of scientific notation of a number
def negexpscinot(number):
    base10 = np.log10(abs(number))
    return int(-1 * np.floor(base10))
