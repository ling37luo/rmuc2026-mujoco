"""Isaac Sim 6.1 PhysX terrain scale/contact probe (UNVERIFIED_RUNTIME).

Run this with the supported host's Isaac Sim ``python.sh``. This repository's
CI can exercise ``--check-only`` but does not have Isaac Sim/PhysX installed.
The probe uses source-grid triangles, cropped static fence boxes and 120 mm
diameter dynamic spheres; it does not validate an articulated robot or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import resource
import time
from typing import Callable

import numpy as np

from heightfield_mesh import environment_offsets, grid_to_triangle_mesh, triangle_height_at


_PROBE_KEYS = ("approach_xyz_m", "low_seam_xyz_m", "takeoff_xyz_m", "landing_target_xyz_m")
_RADIUS_M = 0.06
_DT_S = 0.001
_RAY_TOLERANCE_M = 0.001
_WALL_RAY_START_M = 0.2
_WALL_SPHERE_START_M = 0.15
_WALL_SPHERE_SPEED_M_S = 1.5


def _load_export(root: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    descriptor = json.loads((root / "descriptor.json").read_text(encoding="utf-8"))
    if (
        descriptor.get("artifact_type") != "rmuc2026_isaac_heightfield_input"
        or descriptor.get("schema_version") not in (1, 2, 3)
        or descriptor.get("grid_file") != "heightfield.npz"
    ):
        raise ValueError("unsupported offline Isaac export descriptor")
    grid_path = root / "heightfield.npz"
    if grid_path.is_symlink() or not grid_path.is_file():
        raise ValueError("missing or linked heightfield.npz")
    if hashlib.sha256(grid_path.read_bytes()).hexdigest() != descriptor["grid_file_sha256"]:
        raise ValueError("heightfield.npz SHA-256 mismatch")
    with np.load(grid_path, allow_pickle=False) as grid:
        if set(grid.files) != {"x_m", "y_m", "height_m"}:
            raise ValueError("heightfield.npz arrays differ from export contract")
        x = np.asarray(grid["x_m"])
        y = np.asarray(grid["y_m"])
        height = np.asarray(grid["height_m"])
    record = descriptor["grid"]
    if (
        x.dtype != np.float64
        or y.dtype != np.float64
        or height.dtype != np.float64
        or height.shape != (record["rows_y"], record["columns_x"])
        or x.shape != (height.shape[1],)
        or y.shape != (height.shape[0],)
        or not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(height).all())
        or not (np.diff(x) > 0).all()
        or not (np.diff(y) > 0).all()
        or not np.allclose(np.diff(x), record["resolution_xy_m"][0], rtol=0, atol=1e-8)
        or not np.allclose(np.diff(y), record["resolution_xy_m"][1], rtol=0, atol=1e-8)
        or not np.array_equal(np.asarray(record["bounds_xy_m"]), [[x[0], y[0]], [x[-1], y[-1]]])
        or record["height_array_order"] != "height_m[y_index, x_index]"
        or record["height_values"] != "absolute_world_z_m_no_extra_scale_or_offset"
        or record["world_frame"] != "xyz_m_z_up"
        or record["interpolation"] != "mujoco_hfield_triangle"
    ):
        raise ValueError("exported grid geometry or units disagree with descriptor")
    return descriptor, x, y, height


def _static_boxes(descriptor: dict, x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Return source-bound, crop-contained static boxes in world meters."""
    if descriptor["schema_version"] == 1:
        return []
    collision = descriptor.get("collision")
    if not isinstance(collision, dict) or not isinstance(collision.get("static_boxes"), list):
        raise ValueError("export lacks collision.static_boxes")
    boxes = collision["static_boxes"]
    if descriptor["scenario_id"] in ("fly_ramp_north", "fly_ramp_south") and not boxes:
        raise ValueError("fly-ramp export has no cropped static perimeter box")
    names = set()
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("invalid static box record")
        name = box.get("name")
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name in names
            or not isinstance(box.get("contype"), int)
            or not isinstance(box.get("conaffinity"), int)
        ):
            raise ValueError("invalid static box identity or collision bits")
        names.add(name)
        try:
            center = np.asarray(box["pos_m"], dtype=np.float64)
            half = np.asarray(box["size_m"], dtype=np.float64)
            friction = np.asarray(box["friction"], dtype=np.float64)
            solref = np.asarray(box["solref"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid static box parameters: {name}") from exc
        if (
            center.shape != (3,)
            or half.shape != (3,)
            or friction.shape != (3,)
            or solref.shape != (2,)
            or not all(np.isfinite(arr).all() for arr in (center, half, friction, solref))
            or np.any(half <= 0)
            or np.any(friction < 0)
            or np.any(center[:2] - half[:2] < [x[0] - 1e-8, y[0] - 1e-8])
            or np.any(center[:2] + half[:2] > [x[-1] + 1e-8, y[-1] + 1e-8])
        ):
            raise ValueError(f"static box outside crop or invalid dimensions: {name}")
    return boxes


def _heightfield_contact(descriptor: dict) -> dict | None:
    """Read source contact metadata; older exports have no terrain contract."""
    if descriptor["schema_version"] < 3:
        return None
    collision = descriptor.get("collision")
    contact = collision.get("heightfield") if isinstance(collision, dict) else None
    if not isinstance(contact, dict) or set(contact) != {
        "name",
        "contype",
        "conaffinity",
        "friction",
        "solref",
    }:
        raise ValueError("schema 3 export lacks collision.heightfield")
    try:
        friction = np.asarray(contact["friction"], dtype=np.float64)
        solref = np.asarray(contact["solref"], dtype=np.float64)
        if (
            contact["name"] != "rmuc2026_field_collision"
            or type(contact["contype"]) is not int
            or type(contact["conaffinity"]) is not int
            or contact["contype"] < 0
            or contact["conaffinity"] < 0
            or friction.shape != (3,)
            or solref.shape != (2,)
            or not np.isfinite(friction).all()
            or not np.isfinite(solref).all()
            or np.any(friction < 0)
        ):
            raise ValueError("invalid terrain contact metadata")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid collision.heightfield contact metadata") from exc
    return contact


def _wall_probe(box: dict, approach_xy: list[float]) -> dict:
    """Place a ray and a moving sphere on the approach side of one fence."""
    center = np.asarray(box["pos_m"], dtype=np.float64)
    half = np.asarray(box["size_m"], dtype=np.float64)
    axis = int(np.argmin(half[:2]))
    tangential_axis = 1 - axis
    side = -1.0 if approach_xy[axis] < center[axis] else 1.0
    face = center[axis] + side * half[axis]
    ray = center.copy()
    ray[tangential_axis] += min(0.4, half[tangential_axis] * 0.5)
    ray[axis] = face + side * _WALL_RAY_START_M
    sphere = center.copy()
    sphere[axis] = face + side * _WALL_SPHERE_START_M
    direction = np.zeros(3)
    direction[axis] = -side
    return {
        "box_name": box["name"],
        "ray_origin_xyz_m": ray.tolist(),
        "sphere_initial_xyz_m": sphere.tolist(),
        "direction_xyz": direction.tolist(),
    }


def _check_environment_box_separation(boxes: list[dict], offsets: np.ndarray) -> None:
    """A cropped fence from one environment must not reach another."""
    for env_a, offset_a in enumerate(offsets):
        for offset_b in offsets[env_a + 1 :]:
            for box_a in boxes:
                center_a = np.asarray(box_a["pos_m"][:2]) + offset_a
                half_a = np.asarray(box_a["size_m"][:2])
                for box_b in boxes:
                    center_b = np.asarray(box_b["pos_m"][:2]) + offset_b
                    half_b = np.asarray(box_b["size_m"][:2])
                    if np.all(np.abs(center_a - center_b) < half_a + half_b):
                        raise ValueError("static boxes overlap across environments")


def _route_probes(descriptor: dict, x: np.ndarray, y: np.ndarray, height: np.ndarray) -> dict:
    route = descriptor["route"]
    probes = {}
    for key in _PROBE_KEYS:
        px, py, _route_height = map(float, route[key])
        probes[key.removesuffix("_xyz_m")] = {
            "xy_m": [px, py],
            "grid_surface_z_m": triangle_height_at(x, y, height, px, py),
        }
    return probes


def _non_node_ray_points(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    probes: dict,
    static_boxes: list[dict],
    wall_probes: list[dict],
    count: int = 8,
) -> list[tuple[float, float]]:
    saddle = np.abs(height[:-1, :-1] + height[1:, 1:] - height[:-1, 1:] - height[1:, :-1])
    points = []
    probe_xy = np.asarray([record["xy_m"] for record in probes.values()])
    for flat in np.argsort(saddle.ravel())[::-1]:
        row, col = np.unravel_index(flat, saddle.shape)
        px = float(x[col] + 0.25 * (x[col + 1] - x[col]))
        py = float(y[row] + 0.75 * (y[row + 1] - y[row]))
        if np.min(np.linalg.norm(probe_xy - [px, py], axis=1)) < 0.25:
            continue
        if any(
            abs(px - box["pos_m"][0]) <= box["size_m"][0]
            and abs(py - box["pos_m"][1]) <= box["size_m"][1]
            for box in static_boxes
        ):
            continue
        if any(
            np.linalg.norm(np.asarray(wall["sphere_initial_xyz_m"][:2]) - [px, py])
            < _RADIUS_M + 0.02
            for wall in wall_probes
        ):
            continue
        points.append((px, py))
        if len(points) == count:
            break
    if len(points) != count:
        raise ValueError("could not select enough non-node collider ray sites")
    return points


def _preflight(root: Path, envs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    descriptor, x, y, height = _load_export(root)
    vertices, faces = grid_to_triangle_mesh(x, y, height)
    probes = _route_probes(descriptor, x, y, height)
    offsets = environment_offsets(envs, x, y)
    boxes = _static_boxes(descriptor, x, y)
    terrain_contact = _heightfield_contact(descriptor)
    _check_environment_box_separation(boxes, offsets)
    wall_probes = [_wall_probe(box, descriptor["route"]["approach_xyz_m"][:2]) for box in boxes]
    source = descriptor["identity"]
    record = {
        "authoring_status": "UNVERIFIED_RUNTIME",
        "export_schema_version": descriptor["schema_version"],
        "scenario_id": descriptor["scenario_id"],
        "validation_status": descriptor["validation_status"],
        "grid_file_sha256": descriptor["grid_file_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_profile_hash": source["source_profile_hash"],
        "source_field_xml_sha256": source.get("source_field_xml_sha256"),
        "region_manifest_sha256": source["region_manifest_sha256"],
        "region_profile_hash": source["region_profile_hash"],
        "source_provenance": "descriptor_claim_check_against_source_region_before_transfer",
        "shape_yx": list(height.shape),
        "resolution_xy_m": descriptor["grid"]["resolution_xy_m"],
        "bounds_xy_m": descriptor["grid"]["bounds_xy_m"],
        "mesh_vertex_count_per_env": len(vertices),
        "mesh_triangle_count_per_env": len(faces),
        "mesh_topology": "mujoco_hfield_00_to_11_diagonal",
        "usd_point_float32_max_abs_quantization_m": float(
            np.max(np.abs(vertices - vertices.astype(np.float32).astype(np.float64)))
        ),
        "environment_offsets_xy_m": offsets.tolist(),
        "probe_sites": probes,
        "static_boxes": boxes,
        "static_box_count_per_env": len(boxes),
        "heightfield_contact": terrain_contact,
        "material_mapping": {
            "status": "source_sliding_friction_proxy_unverified_runtime"
            if terrain_contact is not None
            else "not_available_in_legacy_export",
            "physx_static_dynamic_from_mujoco_sliding": terrain_contact is not None,
            "friction_combine_mode": "max" if terrain_contact else None,
            "restitution": 0.0 if terrain_contact else None,
            "not_mapped": [
                "mujoco_torsional_friction",
                "mujoco_rolling_friction",
                "mujoco_solref",
                "mujoco_contype_conaffinity",
            ]
            if terrain_contact is not None
            else [],
        },
        "wall_probe_sites": wall_probes,
    }
    return x, y, height, record


def _raw_contact_body_path(value: object, decode_body_id: Callable[[int], object]) -> str | None:
    """Resolve an Isaac raw contact body token to its absolute USD prim path."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = decode_body_id(int(value))
    if hasattr(value, "pathString"):
        value = value.pathString
    return value if isinstance(value, str) and value.startswith("/") else None


def _raw_contact_observation(
    raw_contacts: object,
    sphere_path: str,
    expected_collider: str,
    decode_body_id: Callable[[int], object],
) -> dict:
    """Accept only raw pairs between this sphere and its own expected collider."""
    if not isinstance(raw_contacts, list):
        return {
            "raw_contact_count": 0,
            "observed_collider": None,
            "observed_colliders": [],
            "valid": False,
        }

    observed = []
    valid = bool(raw_contacts)
    for contact in raw_contacts:
        if not isinstance(contact, dict):
            valid = False
            continue
        body0 = _raw_contact_body_path(contact.get("body0"), decode_body_id)
        body1 = _raw_contact_body_path(contact.get("body1"), decode_body_id)
        if body0 == sphere_path and body1 is not None:
            collider = body1
        elif body1 == sphere_path and body0 is not None:
            collider = body0
        else:
            valid = False
            continue
        observed.append(collider)
        if collider != expected_collider:
            valid = False

    observed_colliders = sorted(set(observed))
    return {
        "raw_contact_count": len(raw_contacts),
        "observed_collider": observed_colliders[0] if len(observed_colliders) == 1 else None,
        "observed_colliders": observed_colliders,
        "valid": valid,
    }


def _run_physx(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    record: dict,
    *,
    envs: int,
    steps: int,
    device: str,
) -> dict:
    # Isaac imports must follow SimulationApp construction in standalone mode.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import carb
        import omni.usd
        from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
        from omni.physics.core import get_physics_scene_query_interface
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        started = time.perf_counter()
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        if SimulationManager.get_active_physics_engine() != "physx":
            SimulationManager.switch_physics_engine("physx")
        if SimulationManager.get_active_physics_engine() != "physx":
            raise RuntimeError("could not select PhysX backend")
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene = PhysxScene("/World/PhysicsScene")
        physics_scene.set_gravity((0.0, 0.0, -9.81))
        physics_scene.set_dt(_DT_S)
        physics_scene.set_enabled_gpu_dynamics(device.startswith("cuda"))

        def physics_material(path: str, sliding_friction: float):
            material = UsdShade.Material.Define(stage, path)
            prim = material.GetPrim()
            api = UsdPhysics.MaterialAPI.Apply(prim)
            api.CreateStaticFrictionAttr(sliding_friction)
            api.CreateDynamicFrictionAttr(sliding_friction)
            api.CreateRestitutionAttr(0.0)
            PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr().Set(
                PhysxSchema.Tokens.max
            )
            return material

        def bind_material(prim, material) -> None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        terrain_contact = record["heightfield_contact"]
        if terrain_contact is None:
            raise ValueError("PhysX runtime probe requires schema 3 terrain contact metadata")
        terrain_material = physics_material(
            "/World/PhysicsMaterials/terrain", float(terrain_contact["friction"][0])
        )
        box_materials = {
            box["name"]: physics_material(
                f"/World/PhysicsMaterials/{box['name']}", float(box["friction"][0])
            )
            for box in record["static_boxes"]
        }

        vertices, faces = grid_to_triangle_mesh(x, y, height)
        usd_points = [Gf.Vec3f(*map(float, vertex)) for vertex in vertices.astype(np.float32)]
        usd_face_counts = [3] * len(faces)
        usd_face_indices = faces.ravel().tolist()
        offsets = environment_offsets(envs, x, y)
        probes = record["probe_sites"]
        static_boxes = record["static_boxes"]
        wall_probes = record["wall_probe_sites"]
        sensors = {}
        sensor_bodies = {}
        expected_colliders = {}
        for env_index, (offset_x, offset_y) in enumerate(offsets):
            env_path = f"/World/env_{env_index}"
            env = UsdGeom.Xform.Define(stage, env_path)
            env.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(float(x[0] + offset_x), float(y[0] + offset_y), 0.0)
            )
            mesh = UsdGeom.Mesh.Define(stage, f"{env_path}/terrain")
            mesh.CreatePointsAttr(usd_points)
            mesh.CreateFaceVertexCountsAttr(usd_face_counts)
            mesh.CreateFaceVertexIndicesAttr(usd_face_indices)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(False)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                UsdPhysics.Tokens.none
            )
            bind_material(mesh.GetPrim(), terrain_material)
            for box in static_boxes:
                cube = UsdGeom.Cube.Define(stage, f"{env_path}/{box['name']}")
                cube.CreateSizeAttr(2.0)
                cube.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Vec3d(
                        float(box["pos_m"][0] - x[0]),
                        float(box["pos_m"][1] - y[0]),
                        float(box["pos_m"][2]),
                    )
                )
                cube.AddScaleOp().Set(Gf.Vec3f(*map(float, box["size_m"])))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                bind_material(cube.GetPrim(), box_materials[box["name"]])
            for label, probe in probes.items():
                px, py = probe["xy_m"]
                sphere_path = f"{env_path}/probe_{label}"
                sphere = UsdGeom.Sphere.Define(stage, sphere_path)
                sphere.CreateRadiusAttr(_RADIUS_M)
                sphere.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Vec3d(
                        float(px - x[0]),
                        float(py - y[0]),
                        float(probe["grid_surface_z_m"] + _RADIUS_M + 0.025),
                    )
                )
                UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                bind_material(sphere.GetPrim(), terrain_material)
                UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
                UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(0.4)
                sensors[(env_index, label)] = ContactSensor(
                    Contact.create(
                        f"{sphere_path}/contact",
                        min_threshold=0.0,
                        max_threshold=1e9,
                        translations=np.array([[0.0, 0.0, 0.0]]),
                    )
                )
                sensor_bodies[(env_index, label)] = sphere_path
                expected_colliders[(env_index, label)] = mesh.GetPath().pathString
            for wall in wall_probes:
                label = f"wall:{wall['box_name']}"
                sphere_path = f"{env_path}/probe_wall_{wall['box_name']}"
                sphere = UsdGeom.Sphere.Define(stage, sphere_path)
                sphere.CreateRadiusAttr(_RADIUS_M)
                initial = wall["sphere_initial_xyz_m"]
                sphere.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Vec3d(
                        float(initial[0] - x[0]),
                        float(initial[1] - y[0]),
                        float(initial[2]),
                    )
                )
                UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                bind_material(sphere.GetPrim(), box_materials[wall["box_name"]])
                body = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
                velocity = np.asarray(wall["direction_xyz"]) * _WALL_SPHERE_SPEED_M_S
                body.CreateVelocityAttr(Gf.Vec3f(*map(float, velocity)))
                UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(0.4)
                sensors[(env_index, label)] = ContactSensor(
                    Contact.create(
                        f"{sphere_path}/contact",
                        min_threshold=0.0,
                        max_threshold=1e9,
                        translations=np.array([[0.0, 0.0, 0.0]]),
                    )
                )
                sensor_bodies[(env_index, label)] = sphere_path
                expected_colliders[(env_index, label)] = f"{env_path}/{wall['box_name']}"

        SimulationManager.setup_simulation(dt=_DT_S, device=device)
        SimulationManager.step(steps=1)  # Start physics before scene queries or sensor reads.
        authoring_seconds = time.perf_counter() - started
        query = get_physics_scene_query_interface()
        ray_sites = _non_node_ray_points(x, y, height, probes, static_boxes, wall_probes)
        ray_records = []
        for env_index, (offset_x, offset_y) in enumerate(offsets):
            mesh_path = f"/World/env_{env_index}/terrain"
            for px, py in ray_sites:
                world_x = float(px + offset_x)
                world_y = float(py + offset_y)
                hit_found, hit = query.raycast_closest(
                    carb.Float3(world_x, world_y, 5.0),
                    carb.Float3(0.0, 0.0, -1.0),
                    10.0,
                    both_sides=False,
                )
                actual_z = float(hit.position.z) if hit_found else None
                expected_z = triangle_height_at(x, y, height, px, py)
                ray_records.append(
                    {
                        "env": env_index,
                        "xy_m": [world_x, world_y],
                        "hit": bool(hit_found),
                        "collider": str(hit.collision) if hit_found else None,
                        "expected_z_m": expected_z,
                        "observed_z_m": actual_z,
                        "abs_error_m": abs(actual_z - expected_z) if hit_found else None,
                        "valid": bool(
                            hit_found
                            and str(hit.collision) == mesh_path
                            and math.isfinite(actual_z)
                            and abs(actual_z - expected_z) <= _RAY_TOLERANCE_M
                        ),
                    }
                )
        wall_ray_records = []
        for env_index, (offset_x, offset_y) in enumerate(offsets):
            for wall in wall_probes:
                origin = np.asarray(wall["ray_origin_xyz_m"]) + [offset_x, offset_y, 0.0]
                hit_found, hit = query.raycast_closest(
                    carb.Float3(*map(float, origin)),
                    carb.Float3(*map(float, wall["direction_xyz"])),
                    0.5,
                    both_sides=False,
                )
                expected_path = f"/World/env_{env_index}/{wall['box_name']}"
                observed_distance = float(hit.distance) if hit_found else None
                wall_ray_records.append(
                    {
                        "env": env_index,
                        "box": wall["box_name"],
                        "hit": bool(hit_found),
                        "collider": str(hit.collision) if hit_found else None,
                        "expected_distance_m": _WALL_RAY_START_M,
                        "observed_distance_m": observed_distance,
                        "valid": bool(
                            hit_found
                            and str(hit.collision) == expected_path
                            and math.isfinite(observed_distance)
                            and abs(observed_distance - _WALL_RAY_START_M) <= _RAY_TOLERANCE_M
                        ),
                    }
                )

        first_contacts = {}
        begin_steps = time.perf_counter()
        for step in range(1, steps + 1):
            SimulationManager.step(steps=1)
            for identity, sensor in sensors.items():
                if identity in first_contacts:
                    continue
                frame = sensor.get_data()
                if bool(frame["in_contact"]):
                    force = float(np.asarray(frame["force"]).item())
                    contact = _raw_contact_observation(
                        sensor.get_raw_data(),
                        sensor_bodies[identity],
                        expected_colliders[identity],
                        PhysicsSchemaTools.intToSdfPath,
                    )
                    first_contacts[identity] = {
                        "env": identity[0],
                        "site": identity[1],
                        "step": step,
                        "time_s": step * _DT_S,
                        "force_n": force,
                        "number_of_contacts": int(frame["number_of_contacts"]),
                        "expected_collider": expected_colliders[identity],
                        **contact,
                        "valid": math.isfinite(force) and force > 0.0 and contact["valid"],
                    }
        step_seconds = time.perf_counter() - begin_steps
        report = {
            **record,
            "runtime_status": "PHYSX_CONTACT_PROBE_PASS"
            if len(first_contacts) == len(sensors)
            and all(row["valid"] for row in first_contacts.values())
            and all(row["valid"] for row in ray_records)
            and all(row["valid"] for row in wall_ray_records)
            else "PHYSX_CONTACT_PROBE_FAIL",
            "backend": "isaac_sim_6_1_physx",
            "device": device,
            "physics_dt_s": _DT_S,
            "environment_count": envs,
            "steps_per_environment": steps + 1,
            "sphere_radius_m": _RADIUS_M,
            "sphere_mass_kg": 0.4,
            "ray_max_abs_error_m": max(
                (row["abs_error_m"] for row in ray_records if row["abs_error_m"] is not None),
                default=None,
            ),
            "ray_samples": ray_records,
            "wall_ray_samples": wall_ray_records,
            "first_contacts": [first_contacts[key] for key in sorted(first_contacts)],
            "contacted_probe_count": len(first_contacts),
            "expected_probe_count": len(sensors),
            "authoring_and_cooking_wall_seconds": authoring_seconds,
            "stepping_wall_seconds": step_seconds,
            "physics_steps_per_second": steps / step_seconds,
            "environment_steps_per_second": envs * steps / step_seconds,
            "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "solver_warning_scan": "inspect_isaac_console_log_separately",
            "scope": "static_mesh_and_cropped_fence_120mm_spheres_no_articulated_robot",
        }
        return report
    finally:
        app.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "export", type=Path, help="offline export with descriptor.json and heightfield.npz"
    )
    parser.add_argument("--envs", type=int, choices=(1, 4, 16), default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--check-only", action="store_true", help="offline transform check; no Isaac import"
    )
    parser.add_argument(
        "--output", type=Path, help="new JSON report path; existing output is refused"
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    x, y, height, record = _preflight(args.export, args.envs)
    if args.check_only:
        report = {
            **record,
            "runtime_status": "UNVERIFIED_RUNTIME",
            "scope": (
                "terrain_only_legacy_export"
                if record["export_schema_version"] == 1
                else "terrain_and_cropped_fence_without_terrain_contact"
                if record["export_schema_version"] == 2
                else "terrain_and_cropped_fence_with_source_contact_metadata"
            ),
        }
    else:
        if record["export_schema_version"] < 3:
            parser.error("schema 1/2 lacks terrain contact metadata; export schema 3 for PhysX")
        report = _run_physx(
            x, y, height, record, envs=args.envs, steps=args.steps, device=args.device
        )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0 if report["runtime_status"] != "PHYSX_CONTACT_PROBE_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
