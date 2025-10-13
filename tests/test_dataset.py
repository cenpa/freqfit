import numpy as np
from iminuit import cost
import pytest

from freqfit.dataset import Dataset
from freqfit.models import gaussian_on_uniform
from freqfit.parameters import Parameters


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

    parameters = Parameters({
        "global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None]},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None]},
        "delta_dset0": {"includeinfit": False, "value": 1e-1, "limits": None},
        "sigma_dset0": {"includeinfit": False, "value": 1e-2, "limits": None},
        "eff_dset0": {"includeinfit": False, "value": 1e-3, "limits": [0, 1]},
        "exp_dset0": {"includeinfit": False, "value": 1e-4, "limits": [0, None]},
    })


    costfunction = cost.ExtendedUnbinnedNLL
    name = "test_dset"

    dset = Dataset(data, model, model_parameters, parameters, costfunction, name)

    assert np.array_equal(dset.data, data)
    assert dset.name == name
    assert dset.model == model
    assert dset._parlist_indices == [
        0,
        1,
    ]  # We are fitting both S and BI, which are the first 2 parameters of the gaussian_on_uniform density
    assert np.array_equal(
        dset._parlist[:], ['global_S', 'global_BI', 'delta_dset0', 'sigma_dset0', 'eff_dset0', 'exp_dset0']
    )  # Check that the initial values are set correctly, exclude the window
    assert np.array_equal(
        dset._parlist_values[:], [1e-9, 1e-3, 1e-1, 1e-2, 1e-3, 1e-4]
    )  # Check that the initial values are set correctly, exclude the window


    # Test the density function handling
    S = 1
    BI = 2
    reference_density = gaussian_on_uniform.density(data, S, BI, 1e-1, 1e-2, 1e-3, 1e-4)
    test_density = dset.density(data, S, BI)

    assert np.array_equal(test_density[0], reference_density[0])
    assert np.array_equal(test_density[1], reference_density[1])

    reference_density_gradient = gaussian_on_uniform.graddensity(
        data, S, BI, 1e-1, 1e-2, 1e-3, 1e-4
    )
    test_density_gradient = dset.graddensity(data, S, BI)

    # Test the density gradient function handling
    assert (
        len(test_density_gradient[0]) == 2
    )  # make sure we are masking things correctly
    mask_idxs = np.array([0, 1])
    assert np.array_equal(
        test_density_gradient[0], reference_density_gradient[0][mask_idxs]
    )  # only the first two parameters are being varied

    assert np.array_equal(
        test_density_gradient[1][0], reference_density_gradient[1][0, :]
    )  # only the first two parameters are being varied
    assert np.array_equal(
        test_density_gradient[1][1], reference_density_gradient[1][1, :]
    )  # only the first two parameters are being varied


def test_dataset_errors():
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

    # Check that we get an exception if parameters is not a Parameters instance
    with pytest.raises(ValueError):
        dset = Dataset(data, model, model_parameters, parameters, costfunction, name)