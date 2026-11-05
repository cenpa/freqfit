import h5py
import numpy as np
import pytest

from freqfit import Workspace


def fixed_B_workspace():
    # one-bin Poisson with B in the fit but fixed, so it is a Minuit parameter that should never move
    config = Workspace.load_config("tests/FC_config.yaml")
    config["parameters"]["B"]["includeinfit"] = True
    config["parameters"]["B"]["fixed"] = True
    return Workspace(config), config["parameters"]["B"]["value"]


def test_minuit_reset_fixed_value():
    # an initial guess function can estimate a fixed parameter from data (e.g. initial_guess_0vbb does this for BI);
    # minuit_reset should restore the guess for the free parameters but keep fixed ones at their configured value
    ws, B = fixed_B_workspace()
    experiment = ws.experiment
    experiment.guess = {"S": 3.0, "B": 2 * B}

    experiment.minuit_reset()

    assert experiment.minuit.fixed["B"]
    assert experiment.minuit.values["B"] == B
    assert not experiment.minuit.fixed["S"]
    assert experiment.minuit.values["S"] == 3.0


def test_bestfit_keeps_fixed_value():
    ws, B = fixed_B_workspace()
    experiment = ws.experiment
    experiment.guess = {"S": 3.0, "B": 2 * B}

    best = experiment.bestfit(force=True)

    assert best["values"]["B"] == B
    # with one observed count and B fixed, S_hat is pushed to its physical limit
    assert np.isclose(best["values"]["S"], 0.0, atol=1e-3)


def test_toy_ts_mp_uneven_cores():
    # numtoy not a multiple of numcores leaves a last chunk of a different size; the per-core results must still stack
    config = Workspace.load_config("tests/FC_config.yaml")
    ws = Workspace(config)
    ws.numcores = 3
    pars = {"S": 2.0, "B": config["parameters"]["B"]["value"]}
    profile = [{"S": 0.0}, {"S": 2.0}]
    num = 7

    ts, _ = ws.toy_ts_mp(pars, profile, num=num)
    ts_info, info = ws.toy_ts_mp(pars, profile, num=num, info=True)

    assert ts.shape[:2] == (len(profile), num)
    assert len(info["seeds"]) == num
    np.testing.assert_allclose(ts, ts_info)


def free_B_workspace(tmp_path, toy_info_to_save):
    # one-bin Poisson with B fitted, so it has profiled values to save
    config = Workspace.load_config("tests/FC_config.yaml")
    config["parameters"]["B"]["includeinfit"] = True
    config["parameters"]["B"]["fixed"] = False
    config["options"]["profiled_params_to_save"] = ["B"]
    config["options"]["toy_info_to_save"] = toy_info_to_save
    config["options"]["out_path"] = str(tmp_path)
    config["options"]["numtoy"] = 7
    ws = Workspace(config, jobid=1)
    ws.numcores = 3
    return ws


def test_toy_info_to_save(tmp_path):
    # saving only the profiled values gives the same ts and profiled values as info=True, without the data or seeds
    toypars = {"S": 2.0, "B": 6.0}
    tested = [{"S": 0.0}, {"S": 2.0}]
    files = {}
    for name, toy_info, info in (("only", ["profiled_values"], False), ("info", [], True), ("none", [], False)):
        out = tmp_path / name
        out.mkdir()
        ws = free_B_workspace(out, toy_info)
        ws.run_and_save_toys(toypars, tested, toy_pars_override=toypars, info=info)
        files[name] = h5py.File(next(out.glob("*.h5")), "r")

    assert set(files["only"]) == {"ts", "test_parameters_names", "test_parameters_values", "profiled_values_to_return"}
    assert {"data", "seeds", "profiled_values_to_return"} <= set(files["info"])
    assert "profiled_values_to_return" not in files["none"]
    assert files["only"]["profiled_values_to_return"].shape == (len(tested), 7, 1)
    np.testing.assert_array_equal(files["only"]["ts"][:], files["info"]["ts"][:])
    np.testing.assert_array_equal(
        files["only"]["profiled_values_to_return"][:], files["info"]["profiled_values_to_return"][:]
    )
    for f in files.values():
        f.close()


def test_toy_info_to_save_rejects_unknown_items():
    config = Workspace.load_config("tests/FC_config.yaml")
    config["options"]["toy_info_to_save"] = ["profiled_values", "everything"]
    with pytest.raises(ValueError, match="toy_info_to_save"):
        Workspace.load_config(config)
