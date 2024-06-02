"""
A package for setting frequenstist limits from unbinned data
"""

from legendfreqfit._version import version as __version__
from legendfreqfit.dataset import Dataset
from legendfreqfit.limit import SetLimit
from legendfreqfit.experiment import Experiment
from legendfreqfit.superset import Superset
from legendfreqfit.toy import Toy

__all__ = [
    "Dataset",
    "Experiment",
    "SetLimit",
    "Superset",
    "Toy",
    "__version__",
]
