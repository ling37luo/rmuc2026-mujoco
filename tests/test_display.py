from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco import FieldAsset, FieldDisplayController, ManifestError, MujocoModelError


def _asset(
    *,
    lighting: bool = True,
    livery: bool = True,
    livery_kind: str = "official_rulebook_overhead_render_surface_guide",
) -> FieldAsset:
    manifest: dict[str, object] = {}
    if lighting:
        manifest["visual_display"] = {
            "lighting": {
                "key_light_name": "rmuc2026_key_light",
                "fill_light_name": "rmuc2026_fill_light",
                "default_mode": "flat",
                "modes": ["flat", "shadow"],
                "toggle_key": "L",
                "physics_changed": False,
            }
        }
    if livery:
        manifest["visual_layers"] = {
            "livery": {
                "kind": livery_kind,
                "geom_name": "rmuc2026_surface_guide",
                "geom_group": 4,
                "default_visible": False,
                "toggle_key": "G",
                "physics": False,
                "mesh_file": "visual/surface_guide.obj",
                "mesh_sha256": "0" * 64,
                "texture_file": "visual/surface_guide.png",
                "texture_sha256": "1" * 64,
                "contains_baked_scene_content": ["robots", "obstacles", "shadows"],
            }
        }
    return FieldAsset(
        root=Path("."),
        manifest=manifest,
        manifest_sha256="",
        verified_files=(),
        hashes_verified=True,
        _files_by_relative={},
    )


def _model(
    *,
    prefix: str = "",
    duplicate_key: bool = False,
    include_livery: bool = True,
    robot_group: int = 1,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    duplicate = (
        '<light name="rmuc2026_key_light" castshadow="false" active="false"/>'
        if duplicate_key
        else ""
    )
    livery = (
        f'''<geom name="{prefix}rmuc2026_surface_guide" type="box" size="1 1 .01"
          pos="0 0 -.02" group="4" contype="0" conaffinity="0"/>'''
        if include_livery
        else ""
    )
    xml = f"""
<mujoco>
  <visual>
    <headlight active="1" ambient=".11 .12 .13" diffuse=".31 .32 .33"
               specular=".01 .02 .03"/>
  </visual>
  <worldbody>
    <light name="robot_light" castshadow="true" active="false"/>
    {duplicate}
    <light name="{prefix}rmuc2026_key_light" castshadow="true" active="true"/>
    <light name="{prefix}rmuc2026_fill_light" castshadow="false" active="true"/>
    {livery}
    <body name="robot" pos="0 0 1">
      <freejoint/>
      <geom name="robot_geom" type="sphere" size=".1" friction=".8 .01 .001"
            group="{robot_group}"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = np.arange(model.nq) * 0.01
    data.qvel[:] = np.arange(model.nv) * -0.02
    return model, data


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, object_type, name))


def test_apply_and_key_toggles_only_change_declared_display_state() -> None:
    controller = FieldDisplayController(_asset())
    model, data = _model()
    geomgroup = np.ones(6, dtype=np.uint8)
    key_id = _id(model, mujoco.mjtObj.mjOBJ_LIGHT, "rmuc2026_key_light")
    fill_id = _id(model, mujoco.mjtObj.mjOBJ_LIGHT, "rmuc2026_fill_light")
    guide_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc2026_surface_guide")

    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    light_active = model.light_active.copy()
    headlight = (
        int(model.vis.headlight.active),
        model.vis.headlight.ambient.copy(),
        model.vis.headlight.diffuse.copy(),
        model.vis.headlight.specular.copy(),
    )
    geom_contype = model.geom_contype.copy()
    geom_conaffinity = model.geom_conaffinity.copy()
    geom_friction = model.geom_friction.copy()

    status = controller.apply(model, geomgroup)
    assert (
        status
        == controller.status
        == {
            "lighting_available": True,
            "lighting_mode": "flat",
            "livery_available": True,
            "livery_visible": False,
        }
    )
    assert not model.light_castshadow[key_id]
    assert not model.light_castshadow[fill_id]
    assert geomgroup[4] == 0

    assert controller.press_name("l")
    assert controller.press_name("G")
    assert not controller.press_name("x")
    controller.apply(model, geomgroup)
    assert model.light_castshadow[key_id]
    assert not model.light_castshadow[fill_id]
    assert geomgroup[4] == 1
    assert controller.lighting_mode == "shadow"
    assert controller.livery_visible

    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(model.light_active, light_active)
    assert int(model.vis.headlight.active) == headlight[0]
    np.testing.assert_array_equal(model.vis.headlight.ambient, headlight[1])
    np.testing.assert_array_equal(model.vis.headlight.diffuse, headlight[2])
    np.testing.assert_array_equal(model.vis.headlight.specular, headlight[3])
    np.testing.assert_array_equal(model.geom_contype, geom_contype)
    np.testing.assert_array_equal(model.geom_conaffinity, geom_conaffinity)
    np.testing.assert_array_equal(model.geom_friction, geom_friction)
    assert model.geom_group[guide_id] == 4


def test_constructor_modes_and_attached_names() -> None:
    controller = FieldDisplayController(_asset(), lighting="shadow", livery="on")
    model, _data = _model(prefix="attached/")
    geomgroup = np.zeros(6, dtype=np.uint8)

    controller.apply(model, geomgroup)

    key_id = _id(model, mujoco.mjtObj.mjOBJ_LIGHT, "attached/rmuc2026_key_light")
    assert model.light_castshadow[key_id]
    assert geomgroup[4] == 1
    assert controller.toggle_lighting() == "flat"
    assert controller.toggle_livery() is False
    controller.apply(model, geomgroup)
    assert not model.light_castshadow[key_id]
    assert geomgroup[4] == 0


def test_controller_accepts_ground_marking_overlay_and_legacy_livery() -> None:
    new_controller = FieldDisplayController(
        _asset(livery_kind="rulebook_derived_ground_marking_overlay")
    )
    legacy_controller = FieldDisplayController(_asset())

    assert new_controller.livery_available
    assert legacy_controller.livery_available
    assert new_controller.press_name("G")
    assert new_controller.livery_visible


def test_ambiguous_attached_name_fails_before_any_mutation() -> None:
    controller = FieldDisplayController(_asset(), lighting="shadow", livery="on")
    model, _data = _model(prefix="attached/", duplicate_key=True)
    geomgroup = np.zeros(6, dtype=np.uint8)
    castshadow = model.light_castshadow.copy()

    with pytest.raises(MujocoModelError, match="ambiguous field key light"):
        controller.apply(model, geomgroup)

    np.testing.assert_array_equal(model.light_castshadow, castshadow)
    assert geomgroup[4] == 0


def test_livery_group_mismatch_fails_before_lighting_mutation() -> None:
    controller = FieldDisplayController(_asset(), lighting="flat", livery="on")
    model, _data = _model()
    geomgroup = np.zeros(6, dtype=np.uint8)
    key_id = _id(model, mujoco.mjtObj.mjOBJ_LIGHT, "rmuc2026_key_light")
    model.light_castshadow[key_id] = True
    guide_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc2026_surface_guide")
    model.geom_group[guide_id] = 3

    with pytest.raises(MujocoModelError, match="geom group disagrees"):
        controller.apply(model, geomgroup)

    assert model.light_castshadow[key_id]
    assert geomgroup[4] == 0


def test_livery_group_conflict_with_robot_fails_before_any_mutation() -> None:
    controller = FieldDisplayController(_asset(), lighting="flat", livery="on")
    model, _data = _model(robot_group=4)
    geomgroup = np.zeros(6, dtype=np.uint8)
    key_id = _id(model, mujoco.mjtObj.mjOBJ_LIGHT, "rmuc2026_key_light")
    model.light_castshadow[key_id] = True

    with pytest.raises(MujocoModelError, match="also used by other geoms: robot_geom"):
        controller.apply(model, geomgroup)

    assert model.light_castshadow[key_id]
    assert geomgroup[4] == 0


def test_legacy_pack_controls_are_unavailable_and_noop() -> None:
    controller = FieldDisplayController(_asset(lighting=False, livery=False))
    model, _data = _model()
    geomgroup = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)
    castshadow = model.light_castshadow.copy()
    original_groups = geomgroup.copy()

    assert not controller.lighting_available
    assert not controller.livery_available
    assert not controller.press_name("L")
    assert not controller.press_name("G")
    assert controller.toggle_lighting() == "flat"
    assert controller.toggle_livery() is False
    controller.apply(model, geomgroup)

    np.testing.assert_array_equal(model.light_castshadow, castshadow)
    np.testing.assert_array_equal(geomgroup, original_groups)
    with pytest.raises(ValueError, match="shadow lighting is unavailable"):
        FieldDisplayController(_asset(lighting=False), lighting="shadow")
    with pytest.raises(ValueError, match="livery is unavailable"):
        FieldDisplayController(_asset(livery=False), livery="on")


def test_collision_only_profile_may_omit_disabled_livery_layer() -> None:
    controller = FieldDisplayController(_asset(), livery="off")
    model, _data = _model(include_livery=False)
    geomgroup = np.ones(6, dtype=np.uint8)

    controller.apply(model, geomgroup)

    assert geomgroup[4] == 1
    assert controller.press_name("G")
    with pytest.raises(MujocoModelError, match="runtime profile does not contain"):
        controller.apply(model, geomgroup)


def test_malformed_display_contract_is_rejected() -> None:
    asset = _asset()
    asset.manifest["visual_display"]["lighting"]["physics_changed"] = True

    with pytest.raises(ManifestError, match="must not claim a physics change"):
        FieldDisplayController(asset)
