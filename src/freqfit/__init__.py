"""
A package for setting frequenstist limits from unbinned data
"""

from freqfit.dataset import Dataset
from freqfit.experiment import Experiment
from freqfit.limit import SetLimit
from freqfit.superset import Superset
from freqfit.toy import Toy

__all__ = [
    "Dataset",
    "Experiment",
    "SetLimit",
    "Superset",
    "Toy",
    "PlotLimit" "__version__",
]
