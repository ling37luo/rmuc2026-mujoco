from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco.query import HeightFieldData
from rmuc2026_mujoco.slope_routes import screen_slope_route, select_slope_route
from rmuc2026_mujoco.slope_runtime import reset_slope_spawn


def slope_field():
    x = y = np.linspace(-3, 3, 601)
    xx, _ = np.meshgrid(x, y)
    z = np.tan(np.radians(9)) * (np.clip(xx, -0.6, 0.6) + 0.6)
    return HeightFieldData(x, y, z)


def patch():
    return {
        "patch_id": "test",
        "center_xyz_m": [0, 0, 0.1],
        "uphill_unit_xy": [1, 0],
        "usable_width_m": 0.6,
        "horizontal_run_m": 0.8,
    }


def test_continuous_slope_has_flat_approach_and_exit():
    route = screen_slope_route(slope_field(), patch())
    assert route["low_xyz_m"][0] <= -0.75
    assert route["high_xyz_m"][0] >= 0.75
    assert route["height_gain_m"] == pytest.approx(1.2 * math.tan(math.radians(9)))
    assert route["dynamic_status"] == "NOT_RUN"


def test_narrow_obstacle_between_old_catalog_samples_is_rejected():
    field = slope_field()
    field.height_m[298:303, 303:305] += 0.12
    with pytest.raises(ValueError, match="lateral_obstacle"):
        screen_slope_route(field, patch())


def test_high_footprint_missing_at_a_drop_is_rejected():
    field = slope_field()
    field.height_m[:, field.x_m > 0.5] = -0.3
    with pytest.raises(ValueError, match="step_or_discontinuous|drop_or_gap"):
        screen_slope_route(field, patch())


def test_explicit_rejected_patch_does_not_fall_back():
    catalog = {
        "patches": [
            {"patch_id": "bad", "route": None, "route_rejection": "cliff"},
            {"patch_id": "ok", "route": {"route_id": "ok"}},
        ]
    }
    assert select_slope_route(catalog)["route_id"] == "ok"
    with pytest.raises(ValueError, match="cliff"):
        select_slope_route(catalog, "bad")


def test_reset_clears_dynamic_state_and_places_robot_above_contact():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="rmuc2026_field/rmuc2026_field_collision" type="plane" size="4 4 .1"/>
      <body pos="0 0 .5"><freejoint/><geom type="sphere" size=".12"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    route = {"low_xyz_m": [0, 0, 0], "high_xyz_m": [1, 0, 0], "heading_yaw_rad": 0.2}
    first = reset_slope_spawn(model, data, route)
    assert first[2] == pytest.approx(0.122, abs=1e-5)
    data.ctrl[:] = 1
    data.qfrc_applied[:] = 40
    data.qvel[:] = 2
    data.time = 5
    second = reset_slope_spawn(model, data, route)
    np.testing.assert_array_equal(first, second)
    assert data.time == 0 and not data.qvel.any() and not data.qfrc_applied.any()
    assert all(c.dist >= 0 for c in data.contact)


def test_robot_with_incompatible_contact_mask_has_clear_reset_error():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="rmuc2026_field/rmuc2026_field_collision" type="plane" size="4 4 .1"
            contype="2" conaffinity="1"/>
      <body><freejoint/><geom type="sphere" size=".12" contype="0" conaffinity="1"/></body>
      </worldbody></mujoco>""")
    route = {"low_xyz_m": [0, 0, 0], "heading_yaw_rad": 0}
    with pytest.raises(ValueError, match="collision masks"):
        reset_slope_spawn(model, mujoco.MjData(model), route)
