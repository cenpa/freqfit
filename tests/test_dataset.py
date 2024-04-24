import numpy as np
from iminuit import cost

from legendfreqfit.models import gaussian_on_uniform


def test_dataset():
    data = np.array([1, 2, 3, 4])

    model = gaussian_on_uniform

    model_parameters = {
        "S": "global_S",
        "BI": "global_BI",
        "delta": "delta_dset0",
        "sigma": "sigma_dset0",
        "eff": "eff_dset0",
        "exp": "exp_dset0",
    }

    parameters = {
        "global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None]},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None]},
        "delta_dset0": {"includeinfit": False, "value": 1e-1, "limits": None},
        "sigma_dset0": {"includeinfit": False, "value": 1e-2, "limits": None},
        "eff_dset0": {"includeinfit": False, "value": 1e-3, "limits": [0, 1]},
        "exp_dset0": {"includeinfit": False, "value": 1e-4, "limits": [0, None]},
    }

    costfunction = cost.ExtendedUnbinnedNLL
    name = "test_dset"
    assert True
