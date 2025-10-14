import numpy as np
from iminuit import cost
import pytest

from freqfit.dataset import Dataset, ToyDataset, CombinedDataset
from freqfit.models import gaussian_on_uniform
from freqfit.parameters import Parameters
from freqfit.workspace import set_numba_random_seed


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


    # Finally, check that info from parameters is being respected by the Iminuit object
    # check that only the requested fit values are keys in the iminuit object
    assert list(dset.costfunction._parameters.keys()) == ["global_S", "global_BI"]
    assert np.array_equal(dset.costfunction.data, data)


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
        dset = ToyDataset(model, model_parameters,  model, model_parameters, parameters, costfunction, name)


    parameters = Parameters(parameters)

    # check that we raise an error when a model parameter is missing 
    model_parameters_wrong = {
        "BI": "global_BI",
        "delta": "delta_dset0",
        "sigma": "sigma_dset0",
        "eff": "eff_dset0",
        "exp": "exp_dset0",
    }

    with pytest.raises(KeyError):
        dset = Dataset(data, model, model_parameters_wrong, parameters, costfunction, name)
        dset = ToyDataset(model, model_parameters_wrong, model, model_parameters_wrong, parameters, costfunction, name)


    # check that we raise an error when a model parameter is missnamed missing 
    model_parameters_wrong = {
        "signal": "global_S",
        "BI": "global_BI",
        "delta": "delta_dset0",
        "sigma": "sigma_dset0",
        "eff": "eff_dset0",
        "exp": "exp_dset0",
    }

    with pytest.raises(KeyError):
        dset = Dataset(data, model, model_parameters_wrong, parameters, costfunction, name)
        dset = ToyDataset(model, model_parameters_wrong, model, model_parameters_wrong, parameters, costfunction, name)


def test_toydataset():
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

    dset = ToyDataset(model, model_parameters,model, model_parameters, parameters, costfunction, name)

    assert np.array_equal(dset.data, []) # no data until rvs
    assert dset.name == name
    assert dset.toy_model == model
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

    # Test the RV handling
    np.random.seed(206)
    set_numba_random_seed(206) # numba holds RNG seeds in thread local storage, so set it up here
    dset.rvs({"global_S":0.0, "global_BI":0.0}) # should create no data
    assert len(dset.data) == 0
    dset.reset()
    assert dset.costfunction is None # should have reset

    dset.rvs({"global_S":100, "global_BI":100}) # should create some data
    assert np.allclose(dset.data, np.array([2069.60986637]), rtol=1e-6)
    assert dset.num_drawn == (0,1)

    # Test the density function handling
    S = 100
    BI = 100
    reference_density = gaussian_on_uniform.density(dset.data, S, BI, 1e-1, 1e-2, 1e-3, 1e-4)
    test_density = dset.density(dset.data, S, BI)

    assert np.array_equal(test_density[0], reference_density[0])
    assert np.array_equal(test_density[1], reference_density[1])

    reference_density_gradient = gaussian_on_uniform.graddensity(
        dset.data, S, BI, 1e-1, 1e-2, 1e-3, 1e-4
    )
    test_density_gradient = dset.graddensity(dset.data, S, BI)

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


def test_combined_dataset():
    data = np.array([2038, 2040, 2039])

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
        "delta_dset0": {"includeinfit": True, "value": 0.5, "limits": None},
        "sigma_dset0": {"includeinfit": True, "value": 0.1, "limits": None},
        "eff_dset0": {"includeinfit": False, "value": 0.5, "limits": [0, 1]},
        "exp_dset0": {"includeinfit": False, "value": 1, "limits": [0, None]},
    })

    model_parameters2 = {
        "S": "global_S",
        "BI": "global_BI",
        "delta": "delta_dset1",
        "sigma": "sigma_dset1",
        "eff": "eff_dset1",
        "exp": "exp_dset1",
    }

    parameters2 = Parameters({
        "global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None]},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None]},
        "delta_dset1": {"includeinfit": True, "value": -0.5, "limits": None},
        "sigma_dset1": {"includeinfit": True, "value": 0.9, "limits": None},
        "eff_dset1": {"includeinfit": False, "value": 0.5, "limits": [0, 1]},
        "exp_dset1": {"includeinfit": False, "value": 1, "limits": [0, None]},
    })


    costfunction = cost.ExtendedUnbinnedNLL
    name = "test_dset1"

    dset1 = Dataset(data, model, model_parameters, parameters, costfunction, name)

    name2 = "test_dset2"

    dset2 = Dataset(data, model, model_parameters2, parameters2, costfunction, name2)

    # make the combined dataset
    model_empty = {
        "S": "global_S",
        "BI": "global_BI",
        "delta": "delta_empty",
        "sigma": "sigma_empty",
        "eff": "eff_empty",
        "exp": "exp_empty",
    }

    empty_parameters = Parameters({
        "global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None], "value_from_combine": False},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None], "value_from_combine": False},
        "delta_empty": {"includeinfit": True, "value": -0.5, "limits": None, "value_from_combine": True},
        "sigma_empty": {"includeinfit": True, "value": 0.9, "limits": None, "value_from_combine": True},
        "eff_empty": {"includeinfit": False, "value": 0.5, "limits": [0, 1], "value_from_combine": True},
        "exp_empty": {"includeinfit": False, "value": 1, "limits": [0, None], "value_from_combine": True},
    })
    combined_dset = CombinedDataset([dset1,dset2], model, model_empty, empty_parameters, costfunction, name='empty')
    assert len(combined_dset.data) == 0 # should have no data
    
    # Finally, check that info from parameters is being respected by the Iminuit object
    # check that only the requested fit values are keys in the iminuit object
    assert list(combined_dset.costfunction._parameters.keys()) == ["global_S", "global_BI", "delta_empty", "sigma_empty"]

    assert combined_dset.name == "empty"
    assert combined_dset.model == model
    assert combined_dset._parlist_indices == [
        0,
        1,
        2,
        3
    ]  # We are fitting both S and BI, which are the first 2 parameters of the gaussian_on_uniform density
    assert np.array_equal(
        combined_dset._parlist[:], ['global_S', 'global_BI', 'delta_empty', 'sigma_empty', 'eff_empty', 'exp_empty']
    )  # Check that the initial values are set correctly, exclude the window
    assert np.array_equal(
        combined_dset._parlist_values[:], [1e-9, 1e-3, 0.0, 0.5, 0.5, 2]
    )  # Check that the initial values are set correctly, should just be a straight average of the two dataset values
