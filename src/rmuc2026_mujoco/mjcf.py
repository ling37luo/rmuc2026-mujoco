"""MuJoCo loading and MjSpec composition for a validated runtime pack."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .errors import MujocoModelError
from .manifest import DEFAULT_RUNTIME_PROFILE, FieldAsset
from .query import load_heightfield


HFIELD_NAME = "rmuc2026_collision"
FIELD_ATTACH_PREFIX = "rmuc2026_field/"
FIELD_COLLISION_GEOM_NAME = "rmuc2026_field_collision"

# These are project-provided sensitivity presets, not RoboMaster specifications.
UNOFFICIAL_FRICTION_PRESETS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "dry": (1.0, 0.005, 0.0001),
        "low": (0.35, 0.002, 0.00005),
        "high": (1.5, 0.01, 0.0002),
    }
)


def load_model(
    asset: FieldAsset,
    *,
    profile: str = DEFAULT_RUNTIME_PROFILE,
    friction_preset: str | None = None,
) -> tuple[Any, Any]:
    """Compile the pack's field-only MJCF and install exact NPZ height samples."""

    mujoco = _mujoco()
    entrypoint = asset.entrypoint_for(profile)
    spec = _load_spec(mujoco, entrypoint, label=f"field profile {profile!r}")
    model = _compile_spec(spec, label=f"field profile {profile!r}")
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    if friction_preset is not None:
        apply_friction_preset(
            model,
            friction_preset,
            geom_name=FIELD_COLLISION_GEOM_NAME,
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def compose_with_robot(
    asset: FieldAsset,
    robot_xml: Path,
    *,
    profile: str = DEFAULT_RUNTIME_PROFILE,
    friction_preset: str | None = None,
) -> tuple[Any, Any]:
    """Attach the field to a robot MJCF using MjSpec and return a compiled pair.

    The caller normally provides a robot-only MJCF. If an exported viewer XML
    contains the recognizable root-level RMUC field copy, that old field copy
    is removed before attachment so the verified pack remains the only field
    source; ordinary robot floors, lights and world elements are preserved.
    """

    mujoco = _mujoco()
    robot_path = Path(robot_xml).expanduser().resolve()
    if robot_path.is_symlink() or not robot_path.is_file():
        raise MujocoModelError(f"robot_xml is missing or is a symlink: {robot_path}")
    robot_spec = _load_spec(mujoco, robot_path, label="robot XML")
    _strip_embedded_field_scene(robot_spec)
    field_spec = _load_spec(
        mujoco,
        asset.entrypoint_for(profile),
        label=f"field profile {profile!r}",
    )
    if not hasattr(robot_spec, "attach"):
        raise MujocoModelError("this operation requires a MuJoCo release with MjSpec.attach")
    try:
        robot_spec.copy_during_attach = True
        frame = robot_spec.worldbody.add_frame(name="rmuc2026_field_frame")
        robot_spec.attach(field_spec, frame=frame, prefix=FIELD_ATTACH_PREFIX)
    except (RuntimeError, ValueError) as exc:
        raise MujocoModelError(f"MjSpec could not attach the field to the robot: {exc}") from exc
    model = _compile_spec(robot_spec, label="composed robot and field")
    inject_exact_heightfield(
        model,
        asset,
        hfield_name=f"{FIELD_ATTACH_PREFIX}{HFIELD_NAME}",
    )
    if friction_preset is not None:
        apply_friction_preset(
            model,
            friction_preset,
            geom_name=f"{FIELD_ATTACH_PREFIX}{FIELD_COLLISION_GEOM_NAME}",
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def apply_friction_preset(
    model: Any,
    preset: str,
    *,
    geom_name: str = FIELD_COLLISION_GEOM_NAME,
) -> dict[str, object]:
    """Apply an explicitly unofficial field-friction sensitivity preset.

    Give the field precedence over robot geoms so low-friction trials really
    lower contact friction. This also selects the field's solver parameters;
    the returned audit reports that choice. Explicit pairs involving the field
    receive the same friction. Robot geom parameters remain unchanged.
    """

    try:
        values = UNOFFICIAL_FRICTION_PRESETS[preset]
    except KeyError as exc:
        choices = ", ".join(UNOFFICIAL_FRICTION_PRESETS)
        raise MujocoModelError(
            f"unknown unofficial friction preset {preset!r}; choose one of: {choices}"
        ) from exc
    mujoco = _mujoco()
    geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    if geom_id < 0:
        raise MujocoModelError(f"MuJoCo model does not contain field geom {geom_name!r}")
    if model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
        raise MujocoModelError("global contact override would mask the field friction preset")
    other_ids = np.arange(model.ngeom) != geom_id
    priority = max(0, int(np.max(model.geom_priority[other_ids], initial=0))) + 1
    # Preserve all authored contact dimensions instead of losing wheel rolling
    # or torsional friction when the higher-priority field owns the contact.
    condim = max(3, int(np.max(model.geom_condim, initial=3)))
    model.geom_priority[geom_id] = priority
    model.geom_condim[geom_id] = condim
    model.geom_friction[geom_id, :] = values
    pair_ids = np.flatnonzero((model.pair_geom1 == geom_id) | (model.pair_geom2 == geom_id))
    pair_values = (values[0], values[0], values[1], values[2], values[2])
    model.pair_friction[pair_ids, :] = pair_values
    applied = tuple(float(value) for value in model.geom_friction[geom_id, :])
    if not np.allclose(applied, values, rtol=0.0, atol=1.0e-12):
        raise MujocoModelError(f"friction preset {preset!r} was not applied exactly")
    return {
        "status": "PASS",
        "official": False,
        "preset": preset,
        "geom_name": geom_name,
        "friction": list(applied),
        "contact_priority": priority,
        "contact_dimension": condim,
        "explicit_pair_ids": pair_ids.tolist(),
        "solver_parameter_source": "field geom for dynamic pairs; authored explicit pairs retained",
        "claim_boundary": "unofficial sensitivity setting, not measured competition material",
    }


def inject_exact_heightfield(
    model: Any,
    asset: FieldAsset,
    *,
    hfield_name: str = HFIELD_NAME,
) -> dict[str, object]:
    """Overwrite PNG bootstrap values with hashed NPZ floats before simulation."""

    mujoco = _mujoco()
    hfield_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name))
    if hfield_id < 0:
        raise MujocoModelError(f"MuJoCo model does not contain hfield {hfield_name!r}")
    rows = int(model.hfield_nrow[hfield_id])
    columns = int(model.hfield_ncol[hfield_id])
    address = int(model.hfield_adr[hfield_id])
    world = load_heightfield(asset)
    terrain_offset = float(asset.recommended_spawn["terrain_height_m"])
    source_height = world.height_m + terrain_offset
    maximum = float(asset.collision["maximum_height_m"])
    minimum = float(asset.collision["minimum_height_m"])
    if source_height.shape != (rows, columns):
        raise MujocoModelError(
            f"heightfield shape mismatch: model={(rows, columns)}, asset={source_height.shape}"
        )
    schema = int(asset.manifest["schema_version"])
    if schema == 3 or (schema == 4 and minimum < 0.0):
        if (
            minimum >= 0.0
            or maximum <= minimum
            or float(np.min(source_height)) < minimum - 1e-8
            or float(np.max(source_height)) > maximum + 1e-8
        ):
            raise MujocoModelError("negative heightfield range is inconsistent")
        normalized = (source_height - minimum) / (maximum - minimum)
    else:
        normalized = np.clip(source_height / maximum, 0.0, 1.0)
    target = model.hfield_data[address : address + rows * columns]
    target[:] = normalized.reshape(-1).astype(target.dtype, copy=False)
    error = float(np.max(np.abs(np.asarray(target).reshape(rows, columns) - normalized)))
    if error > 2.0e-7:
        raise MujocoModelError(f"exact heightfield injection exceeded tolerance: {error}")
    return {
        "status": "PASS",
        "hfield_name": hfield_name,
        "shape": [rows, columns],
        "maximum_normalized_error": error,
        "source": str(asset.file(str(asset.collision["samples_file"]))),
    }


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise MujocoModelError("MuJoCo is not installed") from exc
    return mujoco


def _load_spec(mujoco: Any, path: Path, *, label: str) -> Any:
    try:
        return mujoco.MjSpec.from_file(str(path))
    except (RuntimeError, ValueError) as exc:
        raise MujocoModelError(f"MuJoCo could not parse {label}: {exc}") from exc


def _compile_spec(spec: Any, *, label: str) -> Any:
    try:
        return spec.compile()
    except (RuntimeError, ValueError) as exc:
        raise MujocoModelError(f"MuJoCo could not compile {label}: {exc}") from exc


def _strip_embedded_field_scene(robot_spec: Any) -> None:
    """Remove an old root-level RMUC field from a complete robot scene.

    Some externally supplied robot XML files are exported from a viewer and
    contain both the robot and an older RMUC field copy. Keeping even unused
    old meshes and hfields would compile a second field asset into every
    profile. Only run this narrow cleanup when a known root-level field marker
    is present; robot floors, lights, cameras and assets are left untouched.
    """

    worldbody = robot_spec.worldbody
    root_geoms = list(worldbody.geoms)
    marker_names = {
        str(geom.name) for geom in root_geoms if getattr(geom, "name", None) is not None
    }
    embedded_field = (
        "rmuc2026_field_collision" in marker_names
        or "rmuc2026_surface_guide" in marker_names
        or any(name.startswith("rmuc2026_visual_") for name in marker_names)
    )
    if not embedded_field:
        return

    def field_geom(name: str) -> bool:
        return (
            name in {"rmuc2026_field_collision", "rmuc2026_surface_guide"}
            or name.startswith(("rmuc2026_visual_", "rmuc2026_perimeter_"))
            or name.startswith("rmuc2026_official_wall_")
        )

    for geom in root_geoms:
        if field_geom(str(geom.name)):
            robot_spec.delete(geom)
    for light in list(worldbody.lights):
        if str(light.name) in {"rmuc2026_key_light", "rmuc2026_fill_light"}:
            robot_spec.delete(light)
    for camera in list(worldbody.cameras):
        if str(camera.name) == "rmuc2026_overview":
            robot_spec.delete(camera)

    for material in list(robot_spec.materials):
        name = str(material.name)
        if name.startswith("rmuc2026_material_") or name == "rmuc2026_surface_guide_material":
            robot_spec.delete(material)
    for mesh in list(robot_spec.meshes):
        name = str(mesh.name)
        if name.startswith("rmuc2026_visual_mesh_") or name in {
            "rmuc2026_surface_guide_mesh",
            "rmuc2026_official_wall_402",
            "rmuc2026_official_wall_403",
        }:
            robot_spec.delete(mesh)
    for hfield in list(robot_spec.hfields):
        if str(hfield.name) == "rmuc2026_collision":
            robot_spec.delete(hfield)
    for texture in list(robot_spec.textures):
        if str(texture.name) in {"rmuc2026_sky", "rmuc2026_surface_guide_texture"}:
            robot_spec.delete(texture)
