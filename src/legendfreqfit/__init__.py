"""
A package for setting frequenstist limits from unbinned data
"""

from legendfreqfit.dataset import Dataset
from legendfreqfit.experiment import Experiment
from legendfreqfit.limit import SetLimit
from legendfreqfit.superset import Superset
from legendfreqfit.toy import Toy

__all__ = [
    "Dataset",
    "Experiment",
    "SetLimit",
    "Superset",
    "Toy",
    "PlotLimit" "__version__",
]
