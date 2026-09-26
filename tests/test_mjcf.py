from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

import pytest

from rmuc2026_mujoco import (
    FIELD_COLLISION_GEOM_NAME,
    UNOFFICIAL_FRICTION_PRESETS,
    FieldAsset,
    MujocoModelError,
    compose_with_robot,
    inject_exact_heightfield,
    load_model,
    surface_at,
)


def _expected_heightfield() -> np.ndarray:
    return np.asarray(
        [[0.25, 0.375, 0.5, 0.625], [0.375, 0.5, 0.625, 0.75], [0.5, 0.625, 0.75, 1.0]]
    )


def test_load_model_returns_pair_and_injects_exact_samples(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)
    model, data = load_model(asset)
    assert isinstance(model, mujoco.MjModel)
    assert isinstance(data, mujoco.MjData)
    assert model.nhfield == 1
    assert model.nmesh == 1
    actual = np.asarray(model.hfield_data).reshape(3, 4)
    assert np.max(np.abs(actual - _expected_heightfield())) <= 2.0e-7


def test_collision_only_profile_loads_no_visual_meshes(field_asset_dir: Path) -> None:
    model, _data = load_model(FieldAsset.open(field_asset_dir), profile="collision_only")
    assert model.nmesh == 0
    assert model.nhfield == 1
    actual = np.asarray(model.hfield_data).reshape(3, 4)
    assert np.max(np.abs(actual - _expected_heightfield())) <= 2.0e-7


def test_schema4_negative_heightfield_injection_keeps_below_zero_samples(tmp_path: Path) -> None:
    samples = tmp_path / "heightfield.npz"
    height = np.asarray([[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [1.0, 1.5, 2.0]], dtype=float)
    np.savez_compressed(
        samples,
        x_m=np.asarray([-1.0, 0.0, 1.0]),
        y_m=np.asarray([-1.0, 0.0, 1.0]),
        height_m=height,
    )
    asset = SimpleNamespace(
        manifest={"schema_version": 4},
        recommended_spawn={
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
        collision={
            "samples_file": "heightfield.npz",
            "rows_y": 3,
            "columns_x": 3,
            "minimum_height_m": -1.0,
            "maximum_height_m": 2.0,
            "half_size_xy_m": [1.0, 1.0],
            "geom_center_after_translation_m": [0.0, 0.0, 0.0],
        },
        file=lambda _relative, **_kwargs: samples,
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><hfield name="rmuc2026_collision" nrow="3" ncol="3" '
        'size="1 1 3 .05"/></asset><worldbody><geom type="hfield" '
        'hfield="rmuc2026_collision" pos="0 0 -1"/></worldbody></mujoco>'
    )

    inject_exact_heightfield(model, asset)

    np.testing.assert_allclose(
        np.asarray(model.hfield_data).reshape(3, 3), (height + 1.0) / 3.0, atol=2e-7
    )


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (-0.5, -0.25),  # interior of one triangle
        (1.5, 0.5),  # exactly on a cell diagonal
        (1.0, 0.0),  # internal grid node
    ],
)
def test_surface_query_matches_mujoco_hfield_raycast(
    field_asset_dir: Path,
    x: float,
    y: float,
) -> None:
    asset = FieldAsset.open(field_asset_dir)
    model, data = load_model(asset, profile="collision_only")
    origin = np.asarray([x, y, 3.0], dtype=np.float64)
    direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    geom_id = np.zeros(1, dtype=np.int32)
    ray_normal = np.zeros(3, dtype=np.float64)

    distance = mujoco.mj_ray(
        model,
        data,
        origin,
        direction,
        None,
        True,
        -1,
        geom_id,
        ray_normal,
    )
    sample = surface_at(asset, origin[0], origin[1], window_radius_m=0.0)

    assert distance > 0.0
    assert origin[2] - distance == pytest.approx(sample.height_m, abs=1.0e-7)
    assert tuple(ray_normal) == pytest.approx(sample.normal_xyz, abs=1.0e-7)


@pytest.mark.parametrize("preset", ["dry", "low", "high"])
def test_unofficial_friction_presets_change_only_field_geom(
    field_asset_dir: Path,
    preset: str,
) -> None:
    model, _data = load_model(
        FieldAsset.open(field_asset_dir),
        profile="collision_only",
        friction_preset=preset,
    )
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FIELD_COLLISION_GEOM_NAME)
    np.testing.assert_allclose(
        model.geom_friction[geom_id],
        UNOFFICIAL_FRICTION_PRESETS[preset],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_unknown_friction_preset_fails_closed(field_asset_dir: Path) -> None:
    with pytest.raises(MujocoModelError, match="unknown unofficial friction preset"):
        load_model(FieldAsset.open(field_asset_dir), friction_preset="official")


def test_compose_with_robot_uses_mjspec_and_preserves_robot(field_asset_dir: Path) -> None:
    robot = field_asset_dir.parent / "robot.xml"
    robot.write_text(
        """<mujoco model="robot"><worldbody><body name="robot" pos="0 0 1">
<freejoint/><geom name="robot_geom" type="sphere" size=".1"/>
</body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    model, data = compose_with_robot(FieldAsset.open(field_asset_dir), robot)
    assert isinstance(data, mujoco.MjData)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot") >= 0
    hfield_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_HFIELD, "rmuc2026_field/rmuc2026_collision"
    )
    assert hfield_id >= 0
    address = int(model.hfield_adr[hfield_id])
    actual = np.asarray(model.hfield_data[address : address + 12]).reshape(3, 4)
    assert np.max(np.abs(actual - _expected_heightfield())) <= 2.0e-7


def test_compose_strips_recognizable_embedded_field_copy(field_asset_dir: Path) -> None:
    robot = field_asset_dir.parent / "complete_viewer_scene.xml"
    robot.write_text(
        """<mujoco model="complete_scene"><worldbody>
<geom name="rmuc2026_field_collision" type="box" size="1 1 .01"/>
<light name="rmuc2026_key_light"/>
<body name="robot" pos="0 0 1"><freejoint/>
<geom name="robot_geom" type="sphere" size=".1"/>
</body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    model, _data = compose_with_robot(FieldAsset.open(field_asset_dir), robot, profile="full")
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc2026_field_collision") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "rmuc2026_key_light") == -1
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc2026_field/rmuc2026_field_collision"
        )
        >= 0
    )


def test_compose_strips_only_old_field_assets_and_preserves_robot_world(
    field_asset_dir: Path,
) -> None:
    robot = field_asset_dir.parent / "scene_with_old_field_assets.xml"
    mesh = (field_asset_dir / "visual/tetra.obj").as_posix()
    robot.write_text(
        f"""<mujoco model="scene_with_old_field_assets">
<asset>
  <mesh name="robot_mesh" file="{mesh}"/>
  <mesh name="rmuc2026_visual_mesh_0" file="{mesh}"/>
  <hfield name="rmuc2026_collision" nrow="3" ncol="4" size="1.5 1 2 .05"/>
  <texture name="rmuc2026_sky" type="skybox" builtin="gradient" width="16" height="96"/>
  <material name="rmuc2026_material_0" rgba=".2 .2 .2 1"/>
</asset>
<worldbody>
  <geom name="robot_floor" type="plane" size="2 2 .1"/>
  <geom name="rmuc2026_visual_0" type="mesh" mesh="rmuc2026_visual_mesh_0"
        material="rmuc2026_material_0" contype="0" conaffinity="0"/>
  <geom name="rmuc2026_field_collision" type="hfield" hfield="rmuc2026_collision"/>
  <light name="robot_light"/>
  <light name="rmuc2026_key_light"/>
  <camera name="robot_camera"/>
  <camera name="rmuc2026_overview"/>
  <body name="robot" pos="0 0 1"><freejoint/>
    <geom name="robot_geom" type="mesh" mesh="robot_mesh"/>
  </body>
</worldbody></mujoco>""",
        encoding="utf-8",
    )

    model, _ = compose_with_robot(FieldAsset.open(field_asset_dir), robot, profile="collision_only")

    assert model.nhfield == 1
    assert model.nmesh == 1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "robot_mesh") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "rmuc2026_visual_mesh_0") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_floor") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "robot_light") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_camera") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "rmuc2026_key_light") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rmuc2026_overview") == -1


def test_compose_collision_only_profile_and_unofficial_friction(field_asset_dir: Path) -> None:
    robot = field_asset_dir.parent / "robot-collision-only.xml"
    robot.write_text(
        """<mujoco model="robot"><worldbody><body name="robot" pos="0 0 1">
<freejoint/><geom name="robot_geom" type="sphere" size=".1" friction=".8 .01 .001"/>
</body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    model, _data = compose_with_robot(
        FieldAsset.open(field_asset_dir),
        robot,
        profile="collision_only",
        friction_preset="low",
    )
    assert model.nmesh == 0
    field_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "rmuc2026_field/rmuc2026_field_collision",
    )
    robot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_geom")
    np.testing.assert_allclose(
        model.geom_friction[field_geom_id],
        UNOFFICIAL_FRICTION_PRESETS["low"],
    )
    np.testing.assert_allclose(model.geom_friction[robot_geom_id], (0.8, 0.01, 0.001))
