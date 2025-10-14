import numpy as np
from iminuit import cost

from freqfit.dataset import Dataset
from freqfit.models import gaussian_on_uniform
from freqfit.parameters import Parameters



def test_parameters():
    par_dict =  dict({"global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None], "poi":True},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None]},
        "delta_dset0": {"includeinfit": False, "value": 1e-1, "limits": None},
        "sigma_dset0": {"includeinfit": False, "value": 1e-2, "limits": None},
        "eff_dset0": {"includeinfit": False, "value": 1e-3, "limits": [0, 1]},
        "exp_dset0": {"includeinfit": False, "value": 1e-4, "limits": [0, None]}})

    parameters = Parameters(par_dict)
    
    # check the poi is set correctly
    assert parameters.poi[0] == "global_S"

    for key, value in par_dict.items():
        assert parameters(key) == value


    # check that the get_parameters method works correctly

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

    pardict1 = {
        "global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None]},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None]},
        "delta_dset0": {"includeinfit": False, "value": 0.5, "limits": None},
        "sigma_dset0": {"includeinfit": True, "value": 0.1, "limits": None},
        "eff_dset0": {"includeinfit": False, "value": 0.5, "limits": [0, 1]},
        "exp_dset0": {"includeinfit": False, "value": 1, "limits": [0, None]},
    }

    parameters = Parameters(pardict1)

    model_parameters2 = {
        "S": "global_S",
        "BI": "global_BI",
        "delta": "delta_dset1",
        "sigma": "sigma_dset1",
        "eff": "eff_dset1",
        "exp": "exp_dset1",
    }
    
    pardict2 = {
        "global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None]},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None]},
        "delta_dset1": {"includeinfit": True, "value": -0.5, "limits": None},
        "sigma_dset1": {"includeinfit": True, "value": 0.9, "limits": None},
        "eff_dset1": {"includeinfit": False, "value": 0.5, "limits": [0, 1]},
        "exp_dset1": {"includeinfit": False, "value": 1, "limits": [0, None]},
    }

    parameters2 = Parameters(pardict2)


    costfunction = cost.ExtendedUnbinnedNLL
    name = "test_dset1"

    dset1 = Dataset(data, model, model_parameters, parameters, costfunction, name)

    name2 = "test_dset2"

    dset2 = Dataset([], model, model_parameters2, parameters2, costfunction, name2)

    total_pars = Parameters({**pardict1, **pardict2})
    total_par_dict = total_pars.get_parameters({"dset1": dset1, "dset2": dset2})
    
    # make sure that all items are accounted for
    pop_overlap_keys = []
    for key, value in pardict1.items():
        assert total_par_dict[key] == value
        total_par_dict.pop(key)
        if key in ["global_S", "global_BI"]:
            pop_overlap_keys.append(key)

    for key, value in pardict2.items():
        # skip global_S and global_BI because we should have popped them out above
        if key in pop_overlap_keys:
            continue

        assert total_par_dict[key] == value
        total_par_dict.pop(key)
    
    assert len(total_par_dict) == 0


    # Check we can grab datasets with no data correctly
    total_par_dict = total_pars.get_parameters({"dset1": dset1, "dset2": dset2}, nodata=True) 
    # make sure that all items are accounted for

    for key, value in pardict2.items():
        if key in pop_overlap_keys:
            continue
        assert total_par_dict[key] == value
        total_par_dict.pop(key)


    assert len(total_par_dict) == 0


    # now check that fit parameters is constructed appropriately
    fit_par_dict = total_pars.get_fitparameters({"dset1": dset1, "dset2": dset2})
    fit_dict =  {
        "global_S": {"includeinfit": True, "value": 1e-9, "limits": [0, None]},
        "global_BI": {"includeinfit": True, "value": 1e-3, "limits": [0, None]},
        "sigma_dset0": {"includeinfit": True, "value": 0.1, "limits": None},
        "delta_dset1": {"includeinfit": True, "value": -0.5, "limits": None},
        "sigma_dset1": {"includeinfit": True, "value": 0.9, "limits": None},
    }

    for key, value in fit_dict.items():
        assert fit_par_dict[key] == value
        fit_par_dict.pop(key)

    assert len(fit_par_dict) == 0 

    # Check that fitparameters works for empty datasets
    fit_par_dict = total_pars.get_fitparameters({"dset1": dset1, "dset2": dset2}, nodata=True)
    fit_dict.pop("global_S")
    fit_dict.pop("global_BI")
    fit_dict.pop("sigma_dset0")

    for key, value in fit_dict.items():
        assert fit_par_dict[key] == value
        fit_par_dict.pop(key)

    assert len(fit_par_dict) == 0 







    


