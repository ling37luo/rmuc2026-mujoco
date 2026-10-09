"""Source-bound step routes and paired contact checks; no field geometry edits.

The catalog selects isolated risers from reviewed CAD parts, then checks the
whole route against the highest source surface. A supported top surface is not
assumed to be reachable by every robot. The paired wheel check uses the same
probe against the runtime field and a local two-plane source reference.
"""

from __future__ import annotations

from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from .collision_candidate import SOURCE_GLB_SHA256
from .manifest import FieldAsset
from .query import load_heightfield
from .ramp_source_audit import _verified_glb
from .scenarios import scenario_descriptor
from .wheel_probe import _sample_mujoco_heightfield

# Part identities refer to the pinned unsimplified GLB, not simplified visuals.
REVIEW_PARTS = (235, 236, 382, 383, 476, 554, 556, 557, 558)
SAMPLE_SPACING_M = 0.0025
INTERIOR_ERROR_M = 0.001
SEAM_ERROR_M = 0.015
HALF_ROUTE_M = 0.4
ROUTE_WIDTH_M = 0.6


def _source_top(meshes, points, translation):
    origins = np.column_stack((points - translation[:2], np.full(len(points), 5.0)))
    directions = np.tile((0, 0, -1), (len(points), 1))
    top = np.full(len(points), -np.inf)
    lower, upper = origins[:, :2].min(0), origins[:, :2].max(0)
    for mesh in meshes:
        bounds = mesh.bounds
        if np.any(bounds[1, :2] < lower) or np.any(bounds[0, :2] > upper):
            continue
        locations, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
        if len(rays):
            np.maximum.at(top, rays, locations[:, 2] + translation[2])
    return top


def screen_step(along, across, source, runtime, *, source_seam_tolerance_m=0.005):
    """Measure two supported treads and their shared riser, including all lanes.

    Plane fits exclude the seam, but seam position is tested separately. This
    rejects roofs/obstacles intersecting a route rather than fitting them away.
    """
    if not np.isfinite(source).all() or not np.isfinite(runtime).all():
        raise ValueError("missing_source_support")
    s, t = np.meshgrid(along, across, indexing="ij")
    design = np.stack((s, t, np.ones_like(s)), axis=-1)
    planes = []
    for mask in (s < -0.025, s > 0.025):
        plane, *_ = np.linalg.lstsq(design[mask], source[mask], rcond=None)
        residual = float(np.max(np.abs(design[mask] @ plane - source[mask])))
        if residual > INTERIOR_ERROR_M or np.linalg.norm(plane[:2]) > math.tan(math.radians(3)):
            raise ValueError("tread_not_clear_or_planar")
        planes.append(plane)
    gap = design @ (planes[1] - planes[0])
    if not (np.min(gap) >= 0.1 and np.max(gap) <= 0.25):
        raise ValueError("not_an_isolated_10_to_25cm_step")
    interior = np.abs(s) > 0.02
    error = float(np.max(np.abs(source[interior] - runtime[interior])))
    if error > INTERIOR_ERROR_M:
        raise ValueError("runtime_source_tread_mismatch")
    seam_positions = []
    for samples in (source, runtime):
        fraction = (samples - design @ planes[0]) / gap
        positions = []
        for lane in fraction.T:
            crossings = np.flatnonzero((lane[:-1] < 0.5) & (lane[1:] >= 0.5))
            if len(crossings) != 1:
                raise ValueError("missing_or_multiple_risers")
            i = crossings[0]
            positions.append(
                along[i] + (along[i + 1] - along[i]) * (0.5 - lane[i]) / (lane[i + 1] - lane[i])
            )
        seam_positions.append(np.asarray(positions))
    seam_error = float(np.max(np.abs(seam_positions[0] - seam_positions[1])))
    if (
        abs(seam_positions[0][len(across) // 2]) > source_seam_tolerance_m
        or np.max(np.abs(seam_positions[0])) > SEAM_ERROR_M
    ):
        raise ValueError("source_riser_does_not_match_selected_face")
    if seam_error > SEAM_ERROR_M:
        raise ValueError("runtime_source_seam_mismatch")
    return {
        "status": "PASS_STATIC",
        "step_height_m": float(planes[1][2] - planes[0][2]),
        "max_tread_error_m": error,
        "max_seam_location_error_m": seam_error,
        "source_planes_stz": [p.tolist() for p in planes],
    }


def build_stair_catalog(asset, source_manifest):
    """Locate isolated source risers with clear 0.6 m wide approach/exit support."""
    import trimesh

    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    glb, ground_shift, source_hash = _verified_glb(Path(source_manifest))
    meshes = list(trimesh.load(glb, force="scene", process=False).dump(concatenate=False))
    if len(meshes) != 559:
        raise ValueError("source GLB part order changed")
    spawn = field.recommended_spawn
    translation = np.array(
        [
            -spawn["x_before_translation_m"],
            -spawn["y_before_translation_m"],
            ground_shift - spawn["terrain_height_m"],
        ]
    )
    grid = load_heightfield(field)
    along = np.linspace(-0.7, 0.7, 561)
    across = np.linspace(-ROUTE_WIDTH_M / 2, ROUTE_WIDTH_M / 2, 7)
    patches, rejected = [], []
    for part in REVIEW_PARTS:
        mesh = meshes[part]
        faces = np.flatnonzero((np.abs(mesh.face_normals[:, 2]) < 0.05) & (mesh.area_faces > 0.04))
        for face in faces:
            vertices = mesh.triangles[face] + translation
            center = vertices.mean(0)
            if not 0.1 <= np.ptp(vertices[:, 2]) <= 0.25:
                continue
            uphill = -mesh.face_normals[face, :2].copy()
            uphill /= np.linalg.norm(uphill)
            lateral = np.array([-uphill[1], uphill[0]])
            points = center[:2] + along[:, None, None] * uphill + across[None, :, None] * lateral
            name = f"stairs_{part}_{face}"
            try:
                if np.any(points < grid.bounds_xy_m[0]) or np.any(points > grid.bounds_xy_m[1]):
                    raise ValueError("route_outside_heightfield")
                runtime = _sample_mujoco_heightfield(grid, points[:, :, 0], points[:, :, 1])
                # Cheap runtime screening precedes expensive source-ray work.
                screen_step(along, across, runtime, runtime, source_seam_tolerance_m=SEAM_ERROR_M)
                source = _source_top(meshes, points.reshape(-1, 2), translation).reshape(
                    runtime.shape
                )
                audit = screen_step(along, across, source, runtime)
            except ValueError as exc:
                rejected.append({"id": name, "reason": str(exc)})
                continue
            low, high = center[:2] - HALF_ROUTE_M * uphill, center[:2] + HALF_ROUTE_M * uphill

            def xyz(xy):
                z = _sample_mujoco_heightfield(grid, np.array([xy[0]]), np.array([xy[1]]))[0]
                return [*xy.tolist(), float(z)]

            route = {
                "route_id": name,
                "source_part_index": part,
                "source_face_index": int(face),
                "low_xyz_m": xyz(low),
                "high_xyz_m": xyz(high),
                "uphill_unit_xy": uphill.tolist(),
                "heading_yaw_rad": math.atan2(uphill[1], uphill[0]),
                "length_m": 2 * HALF_ROUTE_M,
                "width_m": ROUTE_WIDTH_M,
                "riser_xy_m": center[:2].tolist(),
                "source_audit": audit,
                "waypoints_xyz_m": [
                    xyz(center[:2] + s * uphill)
                    for s in np.linspace(-HALF_ROUTE_M, HALF_ROUTE_M, 17)
                ],
            }
            patches.append({"patch_id": name, "route": route})
            break  # One isolated representative per source component.
    descriptor = scenario_descriptor(field, "stairs_basic")
    return {
        "schema_version": 1,
        "scenario_id": "stairs_basic",
        "status": "PASS_STATIC" if patches else "NO_ISOLATED_SOURCE_ROUTES",
        "source_manifest_sha256": field.manifest_sha256,
        "profile_hash": descriptor["profile_hash"],
        "heightfield_samples_sha256": field.collision["samples_sha256"],
        "cad_manifest_sha256": source_hash,
        "source_glb_sha256": SOURCE_GLB_SHA256,
        "patches": patches,
        "rejected": rejected,
        "scope": "isolated risers in reviewed source parts; not every stair, wall or multilevel area",
        "validation_status": field.manifest.get("validation_status"),
    }


def load_stair_catalog(asset, path_or_dict):
    """Read a catalog (or a stairs-check report) for this exact field pack."""
    if path_or_dict is None:
        raise ValueError("stairs_basic requires --route-catalog from stairs-check")
    doc = (
        path_or_dict
        if isinstance(path_or_dict, dict)
        else json.loads(Path(path_or_dict).read_text())
    )
    catalog = doc.get("catalog", doc)
    if (
        catalog.get("scenario_id") != "stairs_basic"
        or catalog.get("source_manifest_sha256") != asset.manifest_sha256
        or catalog.get("heightfield_samples_sha256") != asset.collision["samples_sha256"]
    ):
        raise ValueError("stair catalog does not match the selected field pack")
    if not catalog.get("patches"):
        raise ValueError("stair catalog has no accepted routes")
    return catalog


def _reference_step(spec, route):
    """Create only a probe reference, never an exported field or runtime patch.

    Each tread is one convex wedge, fitted to source ray heights. Adjacent
    wedges share the source riser XY and preserve the source contact settings.
    """
    import mujoco
    from .mjcf import FIELD_COLLISION_GEOM_NAME

    terrain = spec.geom(FIELD_COLLISION_GEOM_NAME)
    settings = {
        k: getattr(terrain, k).copy()
        if isinstance(getattr(terrain, k), np.ndarray)
        else getattr(terrain, k)
        for k in ("friction", "solref", "solimp", "condim", "priority", "contype", "conaffinity")
    }
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    c = np.asarray(route["riser_xy_m"])
    u = np.asarray(route["uphill_unit_xy"])
    lateral = np.array([-u[1], u[0]])
    planes = route["source_audit"]["source_planes_stz"]
    bottom = min(p[2] for p in planes) - 0.5
    for index, (limits, plane) in enumerate(zip(((-0.9, 0), (0, 0.9)), planes)):
        st = np.array([[limits[0], -0.5], [limits[1], -0.5], [limits[1], 0.5], [limits[0], 0.5]])
        xy = c + st[:, :1] * u + st[:, 1:] * lateral
        z = st @ np.asarray(plane[:2]) + plane[2]
        vertices = np.vstack((np.column_stack((xy, z)), np.column_stack((xy, np.full(4, bottom)))))
        name = f"stairs_reference_{index}"
        spec.add_mesh(name=name, uservert=vertices.ravel().tolist())
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, **settings
        )


def run_stair_checks(asset, source_manifest, *, speeds=(0.3, 0.5, 1.0)):
    """Pair the unchanged field with source-riser contact at each speed/direction."""
    from .acceptance import _run_route

    if not speeds or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("speeds must be positive and finite")
    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    catalog = build_stair_catalog(field, source_manifest)
    comparisons = []
    for patch, direction, speed in product(catalog["patches"], ("forward", "reverse"), speeds):
        route = patch["route"]
        probe_route = {
            "id": route["route_id"],
            "start_xy_m": route["low_xyz_m"][:2],
            "end_xy_m": route["high_xyz_m"][:2],
            "expect": "traverse",
        }
        kwargs = dict(direction=direction, speed_m_s=speed, max_penetration_m=0.025)
        runtime = _run_route(field, "collision_only", probe_route, **kwargs)
        reference = _run_route(
            field,
            "collision_only",
            probe_route,
            prepare_spec=lambda spec: _reference_step(spec, route),
            **kwargs,
        )
        # Failure to climb alone is not a defect: compare to the source riser.
        healthy = all(
            r["finite"] and not r["warnings"] and r["max_penetration_m"] <= 0.025
            for r in (runtime, reference)
        )
        matching = runtime["reached_finish"] == reference["reached_finish"]
        height_difference = max(
            abs(runtime[key] - reference[key])
            for key in ("max_center_height_m", "final_center_height_m")
        )
        speed_difference = abs(
            runtime["max_vertical_speed_m_s"] - reference["max_vertical_speed_m_s"]
        )
        progress_difference = abs(runtime["maximum_progress_m"] - reference["maximum_progress_m"])
        motion_matches = (
            height_difference <= 0.025 and speed_difference <= 0.5 and progress_difference <= 0.025
        )
        comparisons.append(
            {
                "route_id": route["route_id"],
                "direction": "uphill" if direction == "forward" else "downhill",
                "speed_m_s": speed,
                "status": "PASS" if healthy and matching and motion_matches else "FAIL",
                "max_height_difference_m": height_difference,
                "peak_vertical_speed_difference_m_s": speed_difference,
                "motion_matches": motion_matches,
                "progress_difference_m": progress_difference,
                "outcome": "TRAVERSED" if runtime["reached_finish"] else "BLOCKED",
                "source_outcome_matches": matching,
                "runtime": runtime,
                "source_reference": reference,
            }
        )
    return {
        "schema_version": 1,
        "scenario_id": "stairs_basic",
        "catalog": catalog,
        "status": "PASS"
        if comparisons and all(c["status"] == "PASS" for c in comparisons)
        else "FAIL",
        "wheel_comparisons": comparisons,
        "summary": {
            "cases": len(comparisons),
            "passed": sum(c["status"] == "PASS" for c in comparisons),
        },
        "scope": "static source fit and paired 120mm wheel contacts; robot traversal is a separate result",
        "geometry_modified": False,
    }
