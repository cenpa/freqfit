"""
A package for setting frequenstist limits from unbinned data
"""

from freqfit.dataset import Dataset, ToyDataset, CombinedDataset
from freqfit.experiment import Experiment
from freqfit.workspace import Workspace
import freqfit.statistics
import freqfit.fc

__all__ = [
    "Dataset",
    "ToyDataset",
    "CombinedDataset",
    "Experiment",
    "Superset",
    "Workspace",
    "PlotLimit",
    "__version__",
]
