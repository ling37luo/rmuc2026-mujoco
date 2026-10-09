"""Source-screened deck/fence contact routes, including sliding and retreat.

The wheel has free XYZ translation and axle rotation. Horizontal servo forces
are bounded; height is never commanded. A fence hit alone cannot pass a route:
both sliding directions and retreat must complete with continuous support.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import mujoco
import numpy as np

from .acceptance import _compare_trace, _contact_sample, _warning_counts
from .manifest import FieldAsset, sha256_file
from .mjcf import HFIELD_NAME, inject_exact_heightfield
from .query import load_heightfield
from .ramp_source_audit import _verified_glb
from .stairs import _source_top
from .wheel_probe import _sample_mujoco_heightfield

SIDES = (("left", 0, 1), ("right", 0, -1), ("bottom", 1, 1), ("top", 1, -1))
WHEEL_RADIUS_M = 0.06
SOURCE_ERROR_M = 0.001


def screen_support(depth, transverse, height):
    """Reject gaps, steps and steep/obstructed footprints, rather than filling them."""
    if not np.isfinite(height).all():
        raise ValueError("missing_support")
    design = np.stack((depth, transverse, np.ones_like(depth)), axis=-1)
    plane = np.linalg.lstsq(design.reshape(-1, 3), height.ravel(), rcond=None)[0]
    residual = float(np.max(np.abs(design @ plane - height)))
    slope = math.degrees(math.atan(float(np.linalg.norm(plane[:2]))))
    if residual > 0.005 or slope > 3:
        raise ValueError("support_not_continuous_and_nearly_flat")
    return {"plane_dtz": plane.tolist(), "max_residual_m": residual, "slope_deg": slope}


def fence_corners(contract):
    """Measure box joins without confusing them with drivable ground at a corner."""
    panels = {p["name"].rsplit("_", 1)[-1]: p for p in contract["panels"]}
    rows = []
    for xs, sx in (("left", 1), ("right", -1)):
        for ys, sy in (("bottom", 1), ("top", -1)):
            x, y = panels[xs], panels[ys]
            low = np.maximum(np.array(x["pos"]) - x["size"], np.array(y["pos"]) - y["size"])
            high = np.minimum(np.array(x["pos"]) + x["size"], np.array(y["pos"]) + y["size"])
            gap = max(0.0, float(np.max(low - high)))
            rows.append(
                {
                    "id": f"{xs}_{ys}",
                    "xy_m": [x["pos"][0] + sx * x["size"][0], y["pos"][1] + sy * y["size"][1]],
                    "panel_gap_m": gap,
                    "panel_join_status": "PASS_STATIC" if gap < 1e-9 else "FAIL_GAP",
                    "driving_status": "UNVERIFIED",
                }
            )
    return rows


def build_perimeter_catalog(asset, source_manifest):
    """Choose up to three separated source-supported strips on each fence side."""
    import trimesh

    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    fence = field.manifest.get("perimeter_fence")
    if fence is None:
        raise ValueError("perimeter-check requires a pack with a perimeter fence")
    glb, shift, source_hash = _verified_glb(Path(source_manifest))
    meshes = list(trimesh.load(glb, force="scene", process=False).dump(concatenate=False))
    spawn = field.recommended_spawn
    translation = np.array(
        [
            -spawn["x_before_translation_m"],
            -spawn["y_before_translation_m"],
            shift - spawn["terrain_height_m"],
        ]
    )
    grid = load_heightfield(field)
    panels = {p["name"].rsplit("_", 1)[-1]: p for p in fence["panels"]}
    depth, transverse = np.meshgrid(np.linspace(0.005, 0.8, 81), np.linspace(-0.65, 0.65, 131))
    routes, screening = [], {}
    for side, axis, sign in SIDES:
        panel = panels[side]
        face = panel["pos"][axis] + sign * panel["size"][axis]
        center, half = panel["pos"][1 - axis], panel["size"][1 - axis]
        points = np.zeros((*depth.shape, 2))
        points[..., axis] = face + sign * depth
        candidates, rejected = [], Counter()
        for t in np.arange(center - half + 0.7, center + half - 0.7, 0.2):
            points[..., 1 - axis] = t + transverse
            if np.any(points < grid.bounds_xy_m[0]) or np.any(points > grid.bounds_xy_m[1]):
                rejected["outside_heightfield"] += 1
                continue
            z = _sample_mujoco_heightfield(grid, points[..., 0], points[..., 1])
            try:
                support = screen_support(depth, transverse, z)
                # Avoid treating elevated facility roofs as driving areas.
                if np.max(z) > 0.6:
                    raise ValueError("elevated_structure")
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            candidates.append((float(t), support))
        selected, source_rejected = [], set()
        for target in (center - half, center, center + half):
            for t, support in sorted(candidates, key=lambda row: abs(row[0] - target)):
                if t in source_rejected or any(
                    abs(t - r["transverse_center_m"]) < 1.4 for r in selected
                ):
                    continue
                points[..., 1 - axis] = t + transverse
                z = _sample_mujoco_heightfield(grid, points[..., 0], points[..., 1])
                source = _source_top(meshes, points.reshape(-1, 2), translation).reshape(z.shape)
                if not np.isfinite(source).all() or np.max(np.abs(source - z)) > SOURCE_ERROR_M:
                    source_rejected.add(t)
                    rejected["missing_or_different_source_support"] += 1
                    continue
                selected.append(
                    {
                        "id": f"{side}_{len(selected)}",
                        "side": side,
                        "axis": axis,
                        "inward_sign": sign,
                        "face_m": face,
                        "transverse_center_m": t,
                        "fence_geom": panel["name"],
                        "support": support,
                        "max_source_error_m": float(np.max(np.abs(source - z))),
                    }
                )
                break
        routes.extend(selected)
        screening[side] = {"selected": len(selected), "rejected": dict(rejected)}
    corners = fence_corners(fence)
    for corner in corners:
        xs, ys = corner["id"].split("_")
        sx, sy = (1 if xs == "left" else -1), (1 if ys == "bottom" else -1)
        dx, dy = np.meshgrid(np.linspace(0.005, 0.8, 81), np.linspace(0.005, 0.8, 81))
        xy = np.stack((corner["xy_m"][0] + sx * dx, corner["xy_m"][1] + sy * dy), -1)
        source = _source_top(meshes, xy.reshape(-1, 2), translation).reshape(dx.shape)
        if np.isfinite(source).all():
            corner["source_relief_m"] = float(np.ptp(source))
        try:
            screen_support(dx, dy, source)
        except ValueError as exc:
            corner["driving_status"] = "SOURCE_OBSTRUCTED"
            corner["reason"] = str(exc)
    return {
        "schema_version": 1,
        "source_manifest_sha256": field.manifest_sha256,
        "heightfield_samples_sha256": field.collision["samples_sha256"],
        "cad_manifest_sha256": source_hash,
        "source_glb_sha256": sha256_file(glb),
        "routes": routes,
        "screening": screening,
        "corners": corners,
        "sampling_spacing_m": 0.01,
        "scope": "Selected 0.8 m deep, 1.3 m wide supported deck strips; corner box joins are static-only, obstructed corner ground is not certified.",
    }


def _route_frame(route):
    axis, sign = route["axis"], route["inward_sign"]
    normal, tangent, origin = np.zeros(2), np.zeros(2), np.zeros(2)
    normal[axis], tangent[1 - axis] = sign, 1
    origin[axis], origin[1 - axis] = route["face_m"], route["transverse_center_m"]
    return origin, normal, tangent


def _wheel_model(asset, profile, route):
    _, _, tangent = _route_frame(route)
    spec = mujoco.MjSpec.from_file(str(asset.entrypoint_for(profile)))
    body = spec.worldbody.add_body(name="perimeter_probe")
    for i, axis in enumerate(np.eye(3)):
        body.add_joint(name=f"perimeter_xyz{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
    body.add_joint(
        name="perimeter_spin", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[*tangent, 0], damping=0.001
    )
    body.add_geom(
        name="perimeter_wheel",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[*(-0.04 * tangent), 0, *(0.04 * tangent), 0],
        size=[WHEEL_RADIUS_M],
        mass=2,
        friction=[1, 0.005, 0.0001],
        condim=6,
    )
    model = spec.compile()
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    return model


def run_wheel_route(model, route, speed, initial_height):
    """Run the same bounded forces, with no vertical/lateral path constraint."""
    origin, normal, tangent = _route_frame(route)
    data = mujoco.MjData(model)
    data.qpos[:3] = [*(origin + 0.6 * normal - 0.15 * tangent), initial_height + 0.062]
    mujoco.mj_forward(model, data)
    stages = (
        ("settle", 0.3, None),
        ("approach", 0.8 / speed + 0.4, (-0.1, 0)),
        ("slide_positive", 1.2 / speed + 0.5, (0.06, 0.4)),
        ("slide_negative", 2.0 / speed + 0.5, (0.06, -0.4)),
        ("retreat", 1.4 / speed + 0.5, (0.6, -0.4)),
    )
    rows, pairs = [], Counter()
    max_penetration = max_qacc = max_z_error = 0.0
    max_contacts = cap_steps = 0
    finite, contained = True, True
    trace = {"qpos": [], "qvel": [], "contacts": []}
    plane = np.asarray(route["support"]["plane_dtz"])
    for name, duration, goal in stages:
        fence_steps, support_steps = 0, 0
        count = math.ceil(duration / model.opt.timestep)
        for _ in range(count):
            xy = data.qpos[:2] - origin
            local = np.array([xy @ normal, xy @ tangent])
            desired = np.zeros(2)
            if goal is not None:
                velocity = np.clip(3 * (np.asarray(goal) - local), -speed, speed)
                if name.startswith("slide"):
                    velocity[0] = -0.02  # Light wall preload; do not pin the wheel to the wall.
                desired = velocity[0] * normal + velocity[1] * tangent
            data.qfrc_applied[:2] = np.clip(160 * (desired - data.qvel[:2]), -200, 200)
            mujoco.mj_step(model, data)
            sample, penetration, contacts_finite = _contact_sample(model, data)
            finite = contacts_finite and all(
                np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc)
            )
            pairs.update(sample)
            max_penetration = max(max_penetration, penetration)
            max_qacc = max(max_qacc, float(np.max(np.abs(data.qacc))))
            max_contacts = max(max_contacts, int(data.ncon))
            fence_steps += int(any(route["fence_geom"] in p.split(" | ") for p in sample))
            ground_count = sum(
                v for p, v in sample.items() if "rmuc2026_field_collision" in p.split(" | ")
            )
            support_steps += int(ground_count > 0)
            cap_steps += int(ground_count >= 50)
            xy = data.qpos[:2] - origin
            local = np.array([xy @ normal, xy @ tangent])
            expected_z = float(np.array([*local, 1]) @ plane) + WHEEL_RADIUS_M
            max_z_error = max(max_z_error, abs(float(data.qpos[2]) - expected_z))
            contained = contained and local[0] >= 0 and abs(local[1]) <= 0.59 and local[0] <= 0.74
            trace["qpos"].append(data.qpos.copy())
            trace["qvel"].append(data.qvel.copy())
            trace["contacts"].append(dict(sample))
            if not finite or _warning_counts(data):
                break
        if name == "approach":
            reached = fence_steps > 0
        elif name.startswith("slide"):
            reached = fence_steps > 0 and abs(local[1] - goal[1]) <= 0.06
        elif name == "retreat":
            reached = np.max(np.abs(local - goal)) <= 0.06
        else:
            reached = True
        rows.append(
            {
                "phase": name,
                "reached": bool(reached),
                "final_depth_transverse_m": local.tolist(),
                "fence_contact_steps": fence_steps,
                "ground_contact_steps": support_steps,
            }
        )
        if not finite or _warning_counts(data):
            break
    warnings = _warning_counts(data)
    failures = [r["phase"] + "_incomplete" for r in rows if not r["reached"]]
    for bad, reason in (
        (not finite, "nonfinite_state_or_contact"),
        (bool(warnings), "solver_warning"),
        (not contained, "left_screened_strip_or_crossed_wall"),
        (max_penetration > 0.025, "excessive_penetration"),
        (max_z_error > 0.025, "lost_or_abnormal_support"),
        (len(rows) != len(stages), "incomplete_run"),
    ):
        if bad:
            failures.append(reason)
    return {
        "route_id": route["id"],
        "speed_m_s": speed,
        "status": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "finite": finite,
        "warnings": warnings,
        "contained_in_screened_strip": bool(contained),
        "phases": rows,
        "max_penetration_m": max_penetration,
        "max_support_height_error_m": max_z_error,
        "max_abs_qacc": max_qacc,
        "max_contacts": max_contacts,
        "heightfield_contact_cap_steps": cap_steps,
        "steps": len(trace["qpos"]),
        "contact_pairs": dict(pairs),
    }, trace


def run_perimeter_checks(pack, source_manifest, *, profiles=None, speeds=(0.3, 0.5, 1.0)):
    asset = pack if isinstance(pack, FieldAsset) else FieldAsset.open(pack, verify=True)
    selected = tuple(profiles) if profiles is not None else asset.available_runtime_profiles
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profiles must be nonempty and unique")
    if not speeds or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("speeds must be finite and positive")
    for profile in selected:
        asset.entrypoint_for(profile)
    catalog = build_perimeter_catalog(asset, source_manifest)
    grid = load_heightfield(asset)
    trials, parity, solvers = [], [], {}
    for route in catalog["routes"]:
        origin, normal, tangent = _route_frame(route)
        start = origin + 0.6 * normal - 0.15 * tangent
        height = float(
            _sample_mujoco_heightfield(grid, np.array([start[0]]), np.array([start[1]]))[0]
        )
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
                report, trace = run_wheel_route(model, route, speed, height)
                trials.append({**report, "profile": profile})
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
    complete_sides = {r["side"] for r in catalog["routes"]} == {s[0] for s in SIDES}
    passed = (
        complete_sides
        and all(t["status"] == "PASS" for t in trials + parity)
        and all(c["panel_join_status"] == "PASS_STATIC" for c in catalog["corners"])
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "catalog": catalog,
        "probe": "120mm wheel; free XYZ and axle spin; bounded horizontal force only",
        "mujoco_version": mujoco.__version__,
        "solver_config": solvers,
        "profile_reference": selected[0],
        "trials": trials,
        "profile_parity": parity,
        "summary": {
            "routes": len(catalog["routes"]),
            "trials": len(trials),
            "passed": sum(t["status"] == "PASS" for t in trials),
            "failed": sum(t["status"] != "PASS" for t in trials),
        },
        "geometry_modified": False,
        "validation_status": asset.manifest.get("validation_status"),
    }
