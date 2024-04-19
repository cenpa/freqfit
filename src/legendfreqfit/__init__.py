"""
A package for setting frequenstist limits from unbinned data
"""

from legendfreqfit._version import version as __version__
from legendfreqfit.pseudoexperiment import Pseudoexperiment
from legendfreqfit.toy import Toy
from legendfreqfit.dataset import Dataset
from legendfreqfit.superset import Superset

__all__ = ["__version__"]
