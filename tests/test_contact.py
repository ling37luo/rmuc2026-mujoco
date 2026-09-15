from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco import MujocoModelError, apply_friction_preset
from rmuc2026_mujoco.contact import refine_planar_ramps


@pytest.mark.parametrize("explicit", [False, True])
def test_low_friction_reaches_actual_contact_despite_high_friction_robot(explicit: bool) -> None:
    pair = (
        (
            '<contact><pair geom1="rmuc2026_field_collision" geom2="wheel" '
            'friction="2 2 .01 .001 .001"/></contact>'
        )
        if explicit
        else ""
    )
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <geom name="rmuc2026_field_collision" type="plane" size="3 3 .1"/>
      <body pos="0 0 .09"><freejoint/><geom name="wheel" type="sphere"
      size=".1" friction="2 .01 .001" priority="4"/></body>
      </worldbody>{pair}</mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.contact[0].friction[0] == pytest.approx(2.0)
    robot_before = model.geom_friction[1].copy()

    report = apply_friction_preset(model, "low")
    mujoco.mj_forward(model, data)

    assert data.ncon > 0
    assert data.contact[0].friction == pytest.approx((0.35, 0.35, 0.002, 0.00005, 0.00005))
    np.testing.assert_array_equal(model.geom_friction[1], robot_before)
    assert report["contact_priority"] == 5
    assert apply_friction_preset(model, "low")["contact_priority"] == 5


def test_global_contact_override_rejected_without_mutation() -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco><option><flag override="enable"/></option>
      <worldbody><geom name="rmuc2026_field_collision" type="plane" size="1 1 .1"/>
      </worldbody></mujoco>""")
    before = model.geom_friction.copy()
    with pytest.raises(MujocoModelError, match="global contact override"):
        apply_friction_preset(model, "low")
    np.testing.assert_array_equal(model.geom_friction, before)


def ramp_example():
    x = np.linspace(-0.2, 1.4, 81)
    y = np.linspace(-0.7, 0.7, 71)
    plane = np.broadcast_to(0.2 + 0.3 * x, (len(y), len(x))).copy()
    height = plane.copy()
    height[35, 35] += 0.02
    normal = np.array([-0.3, 0.0, 1.0])
    geometry = dict(
        low_edge_center_xyz_m=[0, 0, 0.2],
        normal_xyz=normal / np.linalg.norm(normal),
        uphill_unit_xy=[1, 0],
        horizontal_run_m=1.2,
        surface_width_m=1.0,
    )
    return x, y, plane, height, geometry


def test_planar_refinement_removes_spike_without_adding_or_moving_other_surfaces() -> None:
    x, y, plane, height, geometry = ramp_example()
    before = height.copy()
    refined, report = refine_planar_ramps(x, y, height, [geometry])
    assert refined[35, 35] == pytest.approx(plane[35, 35], abs=1e-12)
    np.testing.assert_array_equal(height, before)
    np.testing.assert_array_equal(refined[:5], before[:5])
    assert report["outside_footprints_bitwise_equal"]
    assert report["records"][0]["before_core_error_max_m"] == pytest.approx(0.02)


def test_refinement_rejects_large_disagreement_and_overlapping_patches() -> None:
    x, y, _, height, geometry = ramp_example()
    with pytest.raises(ValueError, match="overlapping"):
        refine_planar_ramps(x, y, height, [geometry, geometry])
    height[35, 35] += 0.1
    before = height.copy()
    with pytest.raises(ValueError, match="disagreement"):
        refine_planar_ramps(x, y, height, [geometry])
    np.testing.assert_array_equal(height, before)
