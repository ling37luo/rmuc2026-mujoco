"""Physical smoke for the optional rulebook-sized movable energy-unit proxy."""

from __future__ import annotations

import math

import mujoco
import pytest

from rmuc2026_mujoco.energy_unit import add_energy_unit


def _flat_scene() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.worldbody.add_geom(
        name="test_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.1],
        friction=[1.0, 0.005, 0.0001],
    )
    return spec


def test_energy_unit_is_dynamic_rulebook_mass_and_survives_floor_drop() -> None:
    spec = _flat_scene()
    record = add_energy_unit(spec, name="test_unit", center_xyz_m=(0.0, 0.0, 0.2))
    model = spec.compile()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_unit")
    top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_unit_solid_end")
    assert record["physical_geoms"] == 19
    assert record["friction_is_official"] is False
    assert model.nq == 7
    assert model.body_mass[body] == pytest.approx(0.4)
    assert model.geom_size[top, 0] == pytest.approx(0.0475)
    assert model.geom_size[top, 1] == pytest.approx(0.015)
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert 0.070 < data.qpos[2] < 0.085
    assert data.ncon > 0
    assert all(warning.number == 0 for warning in data.warning)


@pytest.mark.parametrize(
    ("name", "position", "mass", "friction"),
    [
        ("bad name", (0.0, 0.0, 0.2), 0.4, (1.0, 0.005, 0.0001)),
        ("unit", (math.nan, 0.0, 0.2), 0.4, (1.0, 0.005, 0.0001)),
        ("unit", (0.0, 0.0, 0.2), 0.5, (1.0, 0.005, 0.0001)),
        ("unit", (0.0, 0.0, 0.2), 0.4, (0.0, 0.005, 0.0001)),
    ],
)
def test_invalid_energy_unit_does_not_mutate_scene(name, position, mass, friction) -> None:
    spec = _flat_scene()
    before = spec.compile().nbody
    with pytest.raises(ValueError):
        add_energy_unit(
            spec,
            name=name,
            center_xyz_m=position,
            mass_kg=mass,
            friction=friction,
        )
    assert spec.compile().nbody == before
