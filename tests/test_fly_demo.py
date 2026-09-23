from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco.fly_demo import FlyRoverController, fly_rover_xml


@pytest.fixture
def route():
    return {
        "approach_xyz_m": [0.0, 0.0, 0.0],
        "takeoff_xyz_m": [1.0, 0.0, 0.3],
        "landing_edge_xyz_m": [1.65, 0.0, 0.2],
        "landing_target_xyz_m": [2.15, 0.0, 0.2],
        "uphill_unit_xy": [1.0, 0.0],
        "heading_yaw_rad": 0.0,
    }


def data_at(x, *, y=0.0, time=0.4):
    return SimpleNamespace(
        qpos=np.array([x, y, 0.2, 1.0, 0.0, 0.0, 0.0]),
        ctrl=np.zeros(4),
        time=time,
    )


def test_example_rover_compiles_and_actuators_cover_speed_sweep():
    model = mujoco.MjModel.from_xml_string(fly_rover_xml())
    assert model.nu == 4
    assert model.actuator_ctrllimited.all()
    assert np.all(model.actuator_ctrlrange[:, 1] >= 2.5 / 0.08)
    assert np.all(model.actuator_ctrlrange[:, 0] <= -2.5 / 0.08)


@pytest.mark.parametrize("x", [0.5, 0.99, 1.01, 1.64, 1.8])
def test_policy_keeps_speed_at_lip_and_across_gap(route, x):
    controller = FlyRoverController(route, 2.5)
    data = data_at(x)
    controller(None, data, step=200, mode="policy")
    np.testing.assert_allclose(data.ctrl, [31.25] * 4)
    assert controller.phase == "approach"


def test_policy_brakes_after_landing_target_and_reset_restarts(route):
    controller = FlyRoverController(route, 2.2)
    data = data_at(2.2)
    controller(None, data, step=200, mode="policy")
    np.testing.assert_array_equal(data.ctrl, [0.0] * 4)
    assert controller.phase == "settle"
    controller.reset(None, data)
    data.qpos[0] = 0.5
    controller(None, data, step=201, mode="policy")
    np.testing.assert_allclose(data.ctrl, [27.5] * 4)


def test_policy_corrects_lateral_drift_and_warmup_stays_still(route):
    controller = FlyRoverController(route, 2.0)
    data = data_at(0.5, y=0.1, time=0.2)
    controller(None, data, step=100, mode="policy")
    np.testing.assert_array_equal(data.ctrl, [0.0] * 4)
    data.time = 0.4
    controller(None, data, step=200, mode="policy")
    assert data.ctrl[0] > data.ctrl[1]
    assert np.max(data.ctrl) < 40


def test_invalid_speed_rejected_instead_of_actuator_clipping(route):
    with pytest.raises(ValueError, match="speed"):
        FlyRoverController(route, 3.5)
