import numpy as np
from iminuit import cost

from freqfit.constraints import Constraints, ToyConstraints
from freqfit.parameters import Parameters
from freqfit.workspace import set_numba_random_seed

pardict1 = {
    "global_S": {
        "includeinfit": True,
        "value": 1e-9,
        "limits": [0, None],
        "domain": [0, np.inf],
    },
    "global_BI": {
        "includeinfit": True,
        "value": 1e-3,
        "limits": [0, None],
        "domain": [0, np.inf],
    },
    "delta_dset0": {
        "includeinfit": True,
        "value": 0.5,
        "limits": None,
        "domain": [-np.inf, np.inf],
    },
    "sigma_dset0": {
        "includeinfit": True,
        "value": 0.1,
        "limits": None,
        "domain": [-np.inf, np.inf],
    },
    "eff_dset0": {
        "includeinfit": False,
        "value": 0.5,
        "limits": [0, 1],
        "domain": [0, 1],
    },
    "exp_dset0": {
        "includeinfit": False,
        "value": 1,
        "limits": [0, None],
        "domain": [0, np.inf],
    },
}
# check constraints when there are covariances allowed between them
pardict2 = {
    "global_S": {
        "includeinfit": True,
        "value": 1e-9,
        "limits": [0, None],
        "domain": [0, np.inf],
    },
    "global_BI": {
        "includeinfit": True,
        "value": 1e-3,
        "limits": [0, None],
        "domain": [0, np.inf],
    },
    "delta_dset1": {
        "includeinfit": True,
        "value": 0.2,
        "limits": None,
        "domain": [-np.inf, np.inf],
    },
    "sigma_dset1": {
        "includeinfit": False,
        "value": 0.1,
        "limits": None,
        "domain": [-np.inf, np.inf],
    },
    "eff_dset1": {
        "includeinfit": False,
        "value": 0.5,
        "limits": [0, 1],
        "domain": [0, 1],
    },
    "exp_dset1": {
        "includeinfit": False,
        "value": 1,
        "limits": [0, None],
        "domain": [0, np.inf],
    },
}


def test_constraints():
    parameters = Parameters(pardict1)

    constr = {
        "constraint_delta_dset0": {
            "parameters": ["delta_dset0"],
            "values": [0.5],
            "covariance": np.diag([0.1]) ** 2,
            "vary": [True],
        },
        "constraint_sigma_dset0": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = Constraints(constr, parameters)

    assert constraints.parameters == parameters
    assert len(constraints._constraint_groups) == 2
    assert (
        constraints._constraint_groups["constraint_group_0"]["constraint_names"][0]
        == "constraint_delta_dset0"
    )
    assert (
        constraints._constraint_groups["constraint_group_1"]["constraint_names"][0]
        == "constraint_sigma_dset0"
    )

    assert (
        constraints._constraint_groups["constraint_group_0"]["parameters"][
            "delta_dset0"
        ]
        == 0
    )
    assert (
        constraints._constraint_groups["constraint_group_1"]["parameters"][
            "sigma_dset0"
        ]
        == 0
    )

    assert constraints._constraint_groups["constraint_group_0"]["vary"][0] is True
    assert constraints._constraint_groups["constraint_group_1"]["vary"][0] is False

    assert constraints._constraint_groups["constraint_group_0"]["values"][0] == 0.5
    assert constraints._constraint_groups["constraint_group_1"]["values"][0] == 0.1

    assert np.allclose(
        constraints._constraint_groups["constraint_group_0"]["covariance"][0][0], 1e-2
    )
    assert np.allclose(
        constraints._constraint_groups["constraint_group_1"]["covariance"][0][0], 1e-4
    )


def test_constraints_multiple_variables():
    parameters = Parameters({**pardict1, **pardict2})

    constr = {
        "constraint_delta": {
            "parameters": ["delta_dset0", "delta_dset1"],
            "values": [0.5, 0.2],
            "covariance": np.array([[0.1, 0.2], [0.3, 0.01]]),
            "vary": [True],
        },
        "constraint_sigma": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = Constraints(constr, parameters)

    assert len(constraints._constraint_groups) == 2
    assert (
        constraints._constraint_groups["constraint_group_0"]["constraint_names"][0]
        == "constraint_delta"
    )
    assert (
        constraints._constraint_groups["constraint_group_1"]["constraint_names"][0]
        == "constraint_sigma"
    )

    assert (
        constraints._constraint_groups["constraint_group_0"]["parameters"][
            "delta_dset0"
        ]
        == 0
    )
    assert (
        constraints._constraint_groups["constraint_group_0"]["parameters"][
            "delta_dset1"
        ]
        == 1
    )
    assert (
        constraints._constraint_groups["constraint_group_1"]["parameters"][
            "sigma_dset0"
        ]
        == 0
    )

    assert constraints._constraint_groups["constraint_group_0"]["vary"][0] is True
    assert constraints._constraint_groups["constraint_group_1"]["vary"][0] is False

    assert np.allclose(
        constraints._constraint_groups["constraint_group_0"]["values"], [0.5, 0.2]
    )
    assert constraints._constraint_groups["constraint_group_1"]["values"][0] == 0.1

    assert np.allclose(
        constraints._constraint_groups["constraint_group_0"]["covariance"][0],
        [0.1, 0.2],
        [0.3, 0.01],
    )
    assert np.allclose(
        constraints._constraint_groups["constraint_group_1"]["covariance"][0][0], 1e-4
    )


def test_get_constraints():
    # check the other methods in the constraints class

    parameters = Parameters({**pardict1, **pardict2})

    constr = {
        "constraint_delta": {
            "parameters": ["delta_dset0", "delta_dset1"],
            "values": [0.5, 0.2],
            "covariance": np.array([[0.1, 0.2], [0.3, 0.01]]),
            "vary": [True],
        },
        "constraint_sigma": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = Constraints(constr, parameters)

    partial_constraints = constraints.get_constraints(["delta_dset1"])
    assert partial_constraints["constraint_group_0"]["values"] == 0.2
    assert partial_constraints["constraint_group_0"]["vary"][0] is True
    assert np.allclose(partial_constraints["constraint_group_0"]["covariance"], [0.01])


def test_toy_constraints_inherits_correctly():
    parameters = Parameters(pardict1)

    constr = {
        "constraint_delta_dset0": {
            "parameters": ["delta_dset0"],
            "values": [0.5],
            "covariance": np.diag([0.1]) ** 2,
            "vary": [True],
        },
        "constraint_sigma_dset0": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = ToyConstraints(constr, parameters)

    # Check that it reproduces the base constraints class

    assert constraints.parameters == parameters
    assert len(constraints._base_constraint_groups) == 2
    assert (
        constraints._base_constraint_groups["constraint_group_0"]["constraint_names"][0]
        == "constraint_delta_dset0"
    )
    assert (
        constraints._base_constraint_groups["constraint_group_1"]["constraint_names"][0]
        == "constraint_sigma_dset0"
    )

    assert (
        constraints._base_constraint_groups["constraint_group_0"]["parameters"][
            "delta_dset0"
        ]
        == 0
    )
    assert (
        constraints._base_constraint_groups["constraint_group_1"]["parameters"][
            "sigma_dset0"
        ]
        == 0
    )

    assert constraints._base_constraint_groups["constraint_group_0"]["vary"][0] is True
    assert constraints._base_constraint_groups["constraint_group_1"]["vary"][0] is False

    assert constraints._base_constraint_groups["constraint_group_0"]["values"][0] == 0.5
    assert constraints._base_constraint_groups["constraint_group_1"]["values"][0] == 0.1

    assert np.allclose(
        constraints._base_constraint_groups["constraint_group_0"]["covariance"][0][0],
        1e-2,
    )
    assert np.allclose(
        constraints._base_constraint_groups["constraint_group_1"]["covariance"][0][0],
        1e-4,
    )


def test_toy_constraints_independent_parameters():
    parameters = Parameters(pardict1)

    constr = {
        "constraint_delta_dset0": {
            "parameters": ["delta_dset0"],
            "values": [0.5],
            "covariance": np.diag([0.1]) ** 2,
            "vary": [True],
        },
        "constraint_sigma_dset0": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = ToyConstraints(constr, parameters)

    # Check that the constraint groups were properly labeled as independent
    for key, value in constraints._base_constraint_groups.items():
        assert value["all_independent"]


def test_toy_constraints_keeps_base_constraints_unchanged():
    # check that drawing random variables does not update the base constraint groups
    parameters = Parameters(pardict1)

    constr = {
        "constraint_delta_dset0": {
            "parameters": ["delta_dset0"],
            "values": [0.5],
            "covariance": np.diag([0.1]) ** 2,
            "vary": [True],
        },
        "constraint_sigma_dset0": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = ToyConstraints(constr, parameters)
    constraints_no_rvs = Constraints(constr, parameters)

    set_numba_random_seed(502)
    constraints.rvs()

    # Check that the base constraint groups match the correct unaltered constraint groups
    for key, value in constraints._base_constraint_groups.items():
        value.pop("all_independent")
        assert constraints_no_rvs._constraint_groups[key] == value


def test_toy_constraints_rvs():
    # check that we can draw random variables
    parameters = Parameters(pardict1)

    constr = {
        "constraint_delta_dset0": {
            "parameters": ["delta_dset0"],
            "values": [0.5],
            "covariance": np.diag([0.1]) ** 2,
            "vary": [True],
        },
        "constraint_sigma_dset0": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = ToyConstraints(constr, parameters)

    set_numba_random_seed(203)
    constraints.rvs()

    # Check that random variables are drawn
    assert constraints._constraint_groups["constraint_group_0"]["values"][0] != 0.5
    assert constraints._constraint_groups["constraint_group_1"]["values"][0] != 0.1


def test_toy_constraints_rvs_override():
    # check that giving a new central value to a parameter works
    parameters = Parameters(pardict1)

    constr = {
        "constraint_delta_dset0": {
            "parameters": ["delta_dset0"],
            "values": [0.5],
            "covariance": np.diag([0.1]) ** 2,
            "vary": [True],
        },
        "constraint_sigma_dset0": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = ToyConstraints(constr, parameters)

    set_numba_random_seed(203)
    constraints.rvs({"delta_dset0": 1000})

    # Check that random variables are drawn
    assert (
        constraints._constraint_groups["constraint_group_0"]["values"][0] > 500
    )  # this value should be very large because we re-centered it to 1000
    assert (
        constraints._constraint_groups["constraint_group_1"]["values"][0] != 0.1
    ) & (
        constraints._constraint_groups["constraint_group_1"]["values"][0] < 1
    )  # this value should stay very small


def test_toy_constraints_rvs_correlated_parameters():
    # check that the constraints on correlated parameters are varied appropriately
    parameters = Parameters({**pardict1, **pardict2})

    constr = {
        "constraint_delta": {
            "parameters": ["delta_dset0", "delta_dset1"],
            "values": [0.5, 0.2],
            "covariance": np.array([[0.1, 0.05], [0.05, 0.1]]),
            "vary": [True],
        },
        "constraint_sigma": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = ToyConstraints(constr, parameters)
    # Check that the constraint groups were properly labeled
    assert (
        constraints._base_constraint_groups["constraint_group_0"]["all_independent"]
        is False
    )
    assert (
        constraints._base_constraint_groups["constraint_group_1"]["all_independent"]
        is True
    )

    set_numba_random_seed(812)
    constraints.rvs()

    # Check that random variables are drawn
    assert constraints._constraint_groups["constraint_group_0"]["values"][0] != 0.5
    assert constraints._constraint_groups["constraint_group_0"]["values"][1] != 0.2
    assert constraints._constraint_groups["constraint_group_1"]["values"][0] != 0.1


def test_constraint_get_cost():
    parameters = Parameters(pardict1)

    constr = {
        "constraint_delta_dset0": {
            "parameters": ["delta_dset0"],
            "values": [0.5],
            "covariance": np.diag([0.1]) ** 2,
            "vary": [True],
        },
        "constraint_sigma_dset0": {
            "parameters": ["sigma_dset0"],
            "values": [0.1],
            "covariance": np.diag([0.01]) ** 2,
            "vary": [False],
        },
    }

    constraints = Constraints(constr, parameters)

    c = constraints.get_cost(["delta_dset0"])
    assert isinstance(c, cost.NormalConstraint)
    assert list(c._parameters.keys()) == ["delta_dset0"]
    assert c.value == [0.5]
    assert c._cov == np.diag([0.1]) ** 2
