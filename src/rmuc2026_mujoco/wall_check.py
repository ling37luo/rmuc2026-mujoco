"""Source wall ownership, repeated contact, end-cap and local corner checks."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .acceptance import _compare_trace, _contact_sample, _warning_counts
from .manifest import FieldAsset, sha256_file
from .perimeter_check import (
    _route_frame,
    _wheel_model,
    fence_corners,
    run_wheel_route,
    screen_support,
)
from .query import load_heightfield
from .ramp_source_audit import _verified_glb
from .stairs import _source_top
from .wall_tip_repair import _projected_top
from .wheel_probe import _sample_mujoco_heightfield


def _height(grid, xy):
    return _sample_mujoco_heightfield(grid, xy[..., 0], xy[..., 1])


def _surface_triangles(mesh, reference_vertices):
    # Match shared vertices within the geometric tolerance before comparing
    # faces: OBJ decimal rounding can straddle a rounding-bin boundary.
    distance = np.linalg.norm(mesh.vertices[:, None, :] - reference_vertices[None, :, :], axis=2)
    indices = np.argmin(distance, axis=1)
    return sorted(tuple(sorted(indices[face])) for face in mesh.faces)


def audit_wall_ownership(grid, source_mesh, runtime_mesh, region):
    """Compare actual mesh triangles and the transferred roof samples to source."""
    source = np.unique(np.round(source_mesh.vertices, 7), axis=0)
    runtime = np.unique(np.round(runtime_mesh.vertices, 7), axis=0)
    mesh_matches = source.shape == runtime.shape and bool(
        np.allclose(source, runtime, atol=2e-7, rtol=0)
    )
    mesh_matches = mesh_matches and _surface_triangles(source_mesh, source) == _surface_triangles(
        runtime_mesh, source
    )
    r0, r1, c0, c1 = region["changed_grid_rectangle_yx"]
    lower_flank = np.minimum(grid.height_m[r0 : r1 + 1, c0 - 2], grid.height_m[r0 : r1 + 1, c1 + 2])
    actual = grid.height_m[r0 : r1 + 1, c0 : c1 + 1]
    roof = _projected_top(source_mesh, grid.x_m[c0 : c1 + 1], grid.y_m[r0 : r1 + 1])
    floor_error = float(np.max(np.abs(actual - lower_flank[:, None])))
    roof_clearance = float(np.min(roof - actual))
    passed = (
        mesh_matches and np.isfinite(roof).all() and floor_error < 1e-6 and roof_clearance > 0.1
    )
    return {
        "source_part_index": region["source_part_index"],
        "status": "PASS" if passed else "FAIL",
        "mesh_matches_source": mesh_matches,
        "transferred_samples": int(actual.size),
        "max_floor_error_m": floor_error,
        "min_roof_to_heightfield_separation_m": roof_clearance,
    }


def build_wall_catalog(asset, source_manifest):
    import trimesh

    layer = asset.collision.get("source_contact_layer")
    if layer is None:
        raise ValueError("wall-check requires a pack with source wall collision")
    glb, shift, source_hash = _verified_glb(Path(source_manifest))
    meshes = list(trimesh.load(glb, force="scene", process=False).dump(concatenate=False))
    spawn = asset.recommended_spawn
    translation = np.array(
        [
            -spawn["x_before_translation_m"],
            -spawn["y_before_translation_m"],
            shift - spawn["terrain_height_m"],
        ]
    )
    grid = load_heightfield(asset)
    records = {r["source_part_index"]: r for r in layer["mesh_geoms"]}
    routes, ownership = [], []
    for region in layer["ownership_regions"]:
        part = region["source_part_index"]
        source = meshes[part].copy()
        source.apply_translation(translation)
        runtime = trimesh.load(asset.root / records[part]["file"], process=False)
        ownership.append(audit_wall_ownership(grid, source, runtime, region))
        low, high = source.bounds
        for side, face, sign in (("left", low[0], -1), ("right", high[0], 1)):
            for fraction in (0.25, 0.5, 0.75):
                routes.append(
                    dict(
                        id=f"wall_{part}_{side}_{int(100 * fraction)}",
                        axis=0,
                        inward_sign=sign,
                        face_m=float(face),
                        transverse_center_m=float(low[1] + fraction * (high[1] - low[1])),
                        side=side,
                        kind="side",
                        fence_geom=records[part]["name"],
                        slide=True,
                    )
                )
        # One end of each wall is embedded in the perimeter. Its low/high side
        # junctions are exercised from inside; only the free end has an end-cap approach.
        end, sign = (high[1], 1) if part == 402 else (low[1], -1)
        for offset in (-0.045, 0, 0.045):
            routes.append(
                dict(
                    id=f"wall_{part}_end_{offset:+.3f}",
                    axis=1,
                    inward_sign=sign,
                    face_m=float(end),
                    transverse_center_m=float((low[0] + high[0]) / 2 + offset),
                    side="free_end",
                    kind="end",
                    fence_geom=records[part]["name"],
                    slide=False,
                )
            )
        joined_y = low[1] + 0.45 if part == 402 else high[1] - 0.45
        for side, face, sign in (("left", low[0], -1), ("right", high[0], 1)):
            routes.append(
                dict(
                    id=f"wall_{part}_fence_{side}",
                    axis=0,
                    inward_sign=sign,
                    face_m=float(face),
                    transverse_center_m=float(joined_y),
                    side=side,
                    kind="fence_junction",
                    fence_geom=records[part]["name"],
                    slide=False,
                )
            )
    for route in routes:
        origin, normal, tangent = _route_frame(route)
        width = 0.48 if route["slide"] else 0.075
        d, t = np.meshgrid(np.linspace(0.065, 0.75, 70), np.linspace(-width, width, 97))
        xy = origin + d[..., None] * normal + t[..., None] * tangent
        z = _height(grid, xy)
        route["support"] = screen_support(d, t, z)
        source = _source_top(meshes, xy.reshape(-1, 2), translation).reshape(z.shape)
        if not np.isfinite(source).all():
            raise ValueError(f"source support missing on {route['id']}")
        route["max_source_error_m"] = float(np.max(np.abs(source - z)))
        # Existing zero-plane fill is disclosed, not mistaken for exact CAD.
        route["support_source_status"] = (
            "MATCH" if route["max_source_error_m"] <= 0.001 else "PROXY_FLOOR"
        )

    corners = fence_corners(asset.manifest["perimeter_fence"])
    offsets = np.linspace(-0.075, 0.075, 7)
    dx, dy = np.meshgrid(offsets, offsets)
    for corner in corners:
        xs, ys = corner["id"].split("_")
        signs = np.array([1 if xs == "left" else -1, 1 if ys == "bottom" else -1])
        origin = np.array(corner["xy_m"])
        candidates = []
        for x in np.arange(0.15, 1.31, 0.05):
            for y in np.arange(0.15, 1.31, 0.05):
                center = origin + signs * [x, y]
                xy = np.stack((center[0] + dx, center[1] + dy), -1)
                z = _height(grid, xy)
                if np.ptp(z) < 0.015 and -0.011 <= z.min() and z.max() < 0.6:
                    candidates.append((float(np.linalg.norm([x, y])), center, xy, z))
        if not candidates:
            raise ValueError(f"no supported local wheel start at corner {corner['id']}")
        _, center, xy, z = min(candidates, key=lambda row: row[0])
        source = _source_top(meshes, xy.reshape(-1, 2), translation).reshape(z.shape)
        source_error = float(np.max(np.abs(source - z))) if np.isfinite(source).all() else None
        corner.update(
            start_xy_m=center.tolist(),
            target_xy_m=(origin - 0.12 * signs).tolist(),
            initial_height_m=float(z[3, 3]),
            support_source_error_m=source_error,
            support_source_status="MATCH"
            if source_error is not None and source_error <= 0.001
            else "PROXY_FLOOR",
            axis=0,
            inward_sign=int(signs[0]),
            face_m=float(origin[0]),
            transverse_center_m=float(origin[1]),
        )
    return {
        "source_manifest_sha256": asset.manifest_sha256,
        "cad_manifest_sha256": source_hash,
        "source_glb_sha256": sha256_file(glb),
        "ownership": ownership,
        "routes": routes,
        "corners": corners,
        "scope": "Wall low/high sides, free ends and fence junctions. Corner trials are local 120mm probes; they do not establish a full robot route from the field interior. Existing proxy ground is explicitly reported.",
    }


def run_corner_route(model, corner, speed):
    """Contact the first actual blocking surface, retreat, then repeat without reset."""
    start, target = np.array(corner["start_xy_m"]), np.array(corner["target_xy_m"])
    unit = (target - start) / np.linalg.norm(target - start)
    height = corner["initial_height_m"]
    data = mujoco.MjData(model)
    data.qpos[:3] = [*start, height + 0.062]
    mujoco.mj_forward(model, data)
    stages = [("settle", start, 0.3)]
    duration = 2 * np.linalg.norm(target - start) / speed + 0.5
    stages += [
        ("approach", target, duration),
        ("retreat", start, duration),
        ("hold", start, 0.3),
    ] * 2
    hits, phases = [], []
    trace = {"qpos": [], "qvel": [], "contacts": []}
    max_pen = max_qacc = max_vz = 0.0
    max_contacts = 0
    finite = True
    for phase, goal, duration in stages:
        hit = None
        for _ in range(math.ceil(duration / model.opt.timestep)):
            velocity = 3 * (goal - data.qpos[:2])
            velocity *= min(1, speed / max(np.linalg.norm(velocity), 1e-12))
            data.qfrc_applied[:2] = np.clip(160 * (velocity - data.qvel[:2]), -200, 200)
            mujoco.mj_step(model, data)
            pairs, penetration, contact_finite = _contact_sample(model, data)
            max_pen = max(max_pen, penetration)
            max_qacc = max(max_qacc, float(np.max(np.abs(data.qacc))))
            max_vz = max(max_vz, abs(float(data.qvel[2])))
            max_contacts = max(max_contacts, int(data.ncon))
            finite = contact_finite and all(
                np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc)
            )
            trace["qpos"].append(data.qpos.copy())
            trace["qvel"].append(data.qvel.copy())
            trace["contacts"].append(dict(pairs))
            if not finite or _warning_counts(data):
                break
            if phase == "approach":
                for i, contact in enumerate(data.contact):
                    names = [model.geom(int(g)).name for g in contact.geom]
                    sign = 1 if names[1] == "perimeter_wheel" else -1
                    opposition = -sign * float(contact.frame[:2] @ unit)
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    if opposition > 0.6 and force[0] * opposition > 1 and hit is None:
                        hit = {
                            "time_s": float(data.time),
                            "geoms": names,
                            "position_m": contact.pos.tolist(),
                            "opposing_force_n": float(force[0] * opposition),
                        }
                if hit is not None and data.time - hit["time_s"] >= 0.03:
                    break
        reached = (
            hit is not None
            if phase == "approach"
            else (
                np.linalg.norm(data.qpos[:2] - start) < 0.06
                and abs(data.qpos[2] - height - 0.06) < 0.025
            )
        )
        phases.append(
            {"phase": phase, "reached": bool(reached), "final_xyz_m": data.qpos[:3].tolist()}
        )
        if hit is not None:
            hits.append(hit)
        if not finite or _warning_counts(data):
            break
    passed = (
        finite
        and not _warning_counts(data)
        and len(hits) == 2
        and max_pen <= 0.025
        and all(p["reached"] for p in phases)
    )
    return {
        "route_id": corner["id"],
        "speed_m_s": speed,
        "status": "PASS" if passed else "FAIL",
        "phases": phases,
        "blocking_contacts": hits,
        "finite": finite,
        "warnings": _warning_counts(data),
        "max_penetration_m": max_pen,
        "max_abs_qacc": max_qacc,
        "max_vertical_speed_m_s": max_vz,
        "max_contacts": max_contacts,
        "steps": len(trace["qpos"]),
    }, trace


def run_wall_checks(pack, source_manifest, *, profiles=None, speeds=(0.3, 0.5, 1.0)):
    asset = pack if isinstance(pack, FieldAsset) else FieldAsset.open(pack, verify=True)
    selected = tuple(profiles) if profiles is not None else asset.available_runtime_profiles
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profiles must be nonempty and unique")
    if not speeds or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("speeds must be finite and positive")
    for profile in selected:
        asset.entrypoint_for(profile)
    catalog = build_wall_catalog(asset, source_manifest)
    grid = load_heightfield(asset)
    trials, parity, solvers = [], [], {}
    for route in catalog["routes"] + catalog["corners"]:
        corner = "start_xy_m" in route
        if not corner:
            o, n, t = _route_frame(route)
            start = o + 0.6 * n + (-0.15 if route["slide"] else 0) * t
            height = float(_height(grid, start[None, :])[0])
        references = {}
        for profile in selected:
            model = _wheel_model(asset, profile, route)
            solvers[profile] = {
                "timestep_s": float(model.opt.timestep),
                "solver": mujoco.mjtSolver(model.opt.solver).name,
                "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
                "iterations": int(model.opt.iterations),
                "tolerance": float(model.opt.tolerance),
            }
            for speed in speeds:
                report, trace = (
                    run_corner_route(model, route, speed)
                    if corner
                    else run_wheel_route(
                        model, route, speed, height, cycles=2, slide=route["slide"]
                    )
                )
                trials.append(
                    {**report, "profile": profile, "kind": "corner" if corner else route["kind"]}
                )
                if profile == selected[0]:
                    references[speed] = trace
                else:
                    parity.append(
                        {
                            "route_id": route["id"],
                            "speed_m_s": speed,
                            "profile": profile,
                            **_compare_trace(references[speed], trace),
                        }
                    )
    passed = all(row["status"] == "PASS" for row in trials + parity + catalog["ownership"]) and all(
        corner["panel_join_status"] == "PASS_STATIC" for corner in catalog["corners"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "catalog": catalog,
        "trials": trials,
        "profile_parity": parity,
        "mujoco_version": mujoco.__version__,
        "profile_reference": selected[0],
        "solver_config": solvers,
        "summary": {
            "routes": len(catalog["routes"]),
            "corners": len(catalog["corners"]),
            "trials": len(trials),
            "passed": sum(t["status"] == "PASS" for t in trials),
            "failed": sum(t["status"] != "PASS" for t in trials),
        },
        "geometry_modified": False,
        "validation_status": asset.manifest.get("validation_status"),
    }
