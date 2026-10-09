"""Whole-rover approaches to source-obstructed corners, followed by retreat."""

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from .acceptance import _contact_sample, _warning_counts
from .manifest import FieldAsset
from .mjcf import FIELD_ATTACH_PREFIX
from .perimeter_check import build_perimeter_catalog, fence_corners
from .query import load_heightfield
from .ramp_source_audit import _verified_glb
from .slope_demo import rover_xml
from .stairs import _source_top
from .training import MuJoCoScenario
from .wheel_probe import _sample_mujoco_heightfield

ROVER_RADIUS_M = 0.35


def solver_config(model):
    return dict(
        timestep_s=float(model.opt.timestep),
        solver=mujoco.mjtSolver(model.opt.solver).name,
        integrator=mujoco.mjtIntegrator(model.opt.integrator).name,
        iterations=int(model.opt.iterations),
        tolerance=float(model.opt.tolerance),
    )


class TraceDigest:
    """Bounded-memory fingerprint of every state and named contact pair."""

    def __init__(self):
        self.digest = hashlib.sha256()
        self.steps = 0

    def record(self, data, pairs):
        self.digest.update(np.asarray(data.qpos, dtype="<f8").tobytes())
        self.digest.update(np.asarray(data.qvel, dtype="<f8").tobytes())
        self.digest.update(json.dumps(sorted(pairs.items()), separators=(",", ":")).encode())
        self.steps += 1

    def report(self):
        return {"steps": self.steps, "state_contact_sha256": self.digest.hexdigest()}


def plan_corner_route(grid, corner, entries, wall_regions=()):
    """Find the nearest supported point connected to an accepted perimeter entry.

    A square enclosing the example rover is sampled at 5 cm spacing. The search
    uses 10 cm cardinal moves, so diagonals cannot cut across obstacle corners.
    This is a route screen; the dynamic test still has to contact and return.
    """
    xs, ys = corner["id"].split("_")
    signs = np.array([1 if xs == "left" else -1, 1 if ys == "bottom" else -1])
    origin = np.array(corner["xy_m"])
    axis = np.arange(0.45, 7.51, 0.1)
    x, y = np.meshgrid(axis, axis)
    xy = origin + np.stack((x, y), -1) * signs
    offsets = np.linspace(-ROVER_RADIUS_M, ROVER_RADIUS_M, 15)
    dx, dy = np.meshgrid(offsets, offsets)
    px, py = xy[..., 0, None, None] + dx, xy[..., 1, None, None] + dy
    inside = (
        (px.min(axis=(-2, -1)) >= grid.bounds_xy_m[0][0])
        & (px.max(axis=(-2, -1)) <= grid.bounds_xy_m[1][0])
        & (py.min(axis=(-2, -1)) >= grid.bounds_xy_m[0][1])
        & (py.max(axis=(-2, -1)) <= grid.bounds_xy_m[1][1])
    )
    z = _sample_mujoco_heightfield(
        grid,
        np.clip(px, grid.bounds_xy_m[0][0], grid.bounds_xy_m[1][0]),
        np.clip(py, grid.bounds_xy_m[0][1], grid.bounds_xy_m[1][1]),
    )
    valid = inside & (np.ptp(z, axis=(-2, -1)) < 0.05)
    valid &= (z.min(axis=(-2, -1)) >= -0.011) & (z.max(axis=(-2, -1)) < 0.6)
    for region in wall_regions:
        low, high = np.asarray(region["source_bounds_world_m"])
        valid &= ~(
            (xy[..., 0] + ROVER_RADIUS_M >= low[0])
            & (xy[..., 0] - ROVER_RADIUS_M <= high[0])
            & (xy[..., 1] + ROVER_RADIUS_M >= low[1])
            & (xy[..., 1] - ROVER_RADIUS_M <= high[1])
        )
    parents = {}
    for entry in entries:
        d = (np.asarray(entry) - origin) * signs
        node = tuple(np.rint((d[::-1] - axis[0]) / 0.1).astype(int))
        if min(node) >= 0 and max(node) < len(axis) and valid[node]:
            parents[node] = None
    queue = deque(parents)
    if not queue:
        raise ValueError(f"no supported perimeter entry for {corner['id']}")
    while queue:
        node = queue.popleft()
        for delta in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            adjacent = (node[0] + delta[0], node[1] + delta[1])
            if min(adjacent) < 0 or max(adjacent) >= len(axis) or adjacent in parents:
                continue
            if not valid[adjacent] or abs(z[adjacent][7, 7] - z[node][7, 7]) > 0.02:
                continue
            parents[adjacent] = node
            queue.append(adjacent)
    goal = min(parents, key=lambda node: np.hypot(x[node], y[node]))
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    if len(path) < 2:
        raise ValueError(f"no approach length for {corner['id']}")
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        if not np.array_equal(np.subtract(path[i], path[i - 1]), np.subtract(path[i + 1], path[i])):
            simplified.append(path[i])
    simplified.append(path[-1])
    return {
        "id": corner["id"],
        "corner_xy_m": origin.tolist(),
        "waypoints_xy_m": [xy[node].tolist() for node in simplified],
        "initial_height_m": float(z[path[0]][7, 7]),
        "nearest_screened_distance_m": float(np.hypot(x[goal], y[goal])),
        "footprint_radius_m": ROVER_RADIUS_M,
        "grid_spacing_m": 0.1,
        "max_screened_relief_m": 0.05,
        "connected_centres": len(parents),
    }, np.array([xy[node] for node in path])


def build_corner_catalog(asset, source_manifest, perimeter_catalog=None):
    import trimesh

    perimeter = perimeter_catalog or build_perimeter_catalog(asset, source_manifest)
    grid = load_heightfield(asset)
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
    walls = asset.collision.get("source_contact_layer", {}).get("ownership_regions", [])
    routes = []
    for corner in fence_corners(asset.manifest["perimeter_fence"]):
        entries = []
        for strip in perimeter["routes"]:
            if strip["side"] not in corner["id"].split("_"):
                continue
            point = np.zeros(2)
            axis = strip["axis"]
            point[axis] = strip["face_m"] + 0.45 * strip["inward_sign"]
            point[1 - axis] = strip["transverse_center_m"]
            entries.append(point)
        route, path = plan_corner_route(grid, corner, entries, walls)
        offsets = np.linspace(-ROVER_RADIUS_M, ROVER_RADIUS_M, 7)
        dx, dy = np.meshgrid(offsets, offsets)
        footprint = path[:, None, None, :] + np.stack((dx, dy), -1)
        source = _source_top(meshes, footprint.reshape(-1, 2), translation)
        runtime = _sample_mujoco_heightfield(grid, footprint[..., 0], footprint[..., 1]).ravel()
        if not np.isfinite(source).all():
            raise ValueError(f"source support missing on {route['id']}")
        source_error = float(np.max(np.abs(source - runtime)))
        signs = np.sign(path[0] - corner["xy_m"])
        centre = np.asarray(corner["xy_m"]) + signs * 0.36
        literal = centre + np.stack((dx, dy), -1)
        corner_source = _source_top(meshes, literal.reshape(-1, 2), translation)
        obstructed = np.isfinite(corner_source).all() and (
            np.ptp(corner_source) > 0.05 or np.min(corner_source) > 0.6
        )
        route.update(
            max_source_support_error_m=source_error,
            support_source_status="MATCH" if source_error <= 0.001 else "PROXY_FLOOR",
            corner_access="SOURCE_OBSTRUCTED" if obstructed else "NO_SCREENED_CONNECTION",
            corner_source_relief_m=float(np.ptp(corner_source))
            if np.isfinite(corner_source).all()
            else None,
        )
        routes.append(route)
    return {
        "source_manifest_sha256": asset.manifest_sha256,
        "cad_manifest_sha256": source_hash,
        "routes": routes,
        "scope": "Public rover routes from accepted perimeter entries to the nearest screened corner approach, contact the next obstacle, then reverse along the same route. No source obstacles are removed.",
    }


def drive_corner(env, route, speed):
    """Drive wheels only; pose is assigned once at reset, never during motion."""
    model, data = env.model, env.data
    path = np.asarray(route["waypoints_xy_m"])
    direction = path[1] - path[0]
    yaw = math.atan2(direction[1], direction[0])
    qpos = model.qpos0.copy()
    qpos[:3] = [*path[0], route["initial_height_m"] + 0.113]
    qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    env.reset(qpos=qpos)
    toward = np.asarray(route["corner_xy_m"]) - path[-1]
    toward /= np.linalg.norm(toward)
    targets = [("settle", path[0], False)]
    targets += [("approach", target, False) for target in path[1:]]
    targets += [("contact", path[-1] + 0.45 * toward, False)]
    targets += [("retreat", target, True) for target in path[::-1]]
    targets += [("hold", path[0], True)]
    phases, hits = [], []
    trace = TraceDigest()
    max_pen = max_tilt = max_acc = 0.0
    max_contacts = unsupported_steps = unsupported_run = longest_unsupported = 0
    finite = True
    for phase, target, reverse in targets:
        duration = (
            1
            if phase in {"settle", "hold"}
            # Turning time does not shrink when the requested straight speed rises.
            else max(12, 3 * np.linalg.norm(target - data.qpos[:2]) / speed + 16)
        )
        reached, hit = False, None
        for _ in range(math.ceil(duration / model.opt.timestep)):
            delta = target - data.qpos[:2]
            distance = np.linalg.norm(delta)
            q = data.qpos[3:7]
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            heading = math.atan2(delta[1], delta[0]) + (math.pi if reverse else 0)
            error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
            vx = (-1 if reverse else 1) * min(speed, 2 * distance) if abs(error) < 0.25 else 0
            wz = float(np.clip(3 * error, -0.7, 0.7))
            if phase in {"settle", "hold"}:
                vx = wz = 0
            env.step([(vx - wz * 0.21) / 0.08, (vx + wz * 0.21) / 0.08] * 2)
            pairs, penetration, contact_finite = _contact_sample(model, data)
            finite = contact_finite and all(
                np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc)
            )
            trace.record(data, pairs)
            max_pen, max_contacts = max(max_pen, penetration), max(max_contacts, int(data.ncon))
            max_acc = max(max_acc, float(np.max(np.abs(data.qacc))))
            q = data.qpos[3:7]
            tilt = math.acos(float(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))
            max_tilt = max(max_tilt, tilt)
            unsupported_steps += int(data.time > 0.3 and not pairs)
            unsupported_run = unsupported_run + 1 if data.time > 0.3 and not pairs else 0
            longest_unsupported = max(longest_unsupported, unsupported_run)
            if phase == "contact":
                heading_xy = np.array([math.cos(yaw), math.sin(yaw)])
                for index, contact in enumerate(data.contact):
                    names = [model.geom(int(g)).name or "" for g in contact.geom]
                    field = [name.startswith(FIELD_ATTACH_PREFIX) for name in names]
                    if sum(field) != 1:
                        continue
                    sign = 1 if field[0] else -1
                    opposing = -sign * float(contact.frame[:2] @ heading_xy)
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, force)
                    if opposing > 0.6 and force[0] * opposing > 1 and hit is None:
                        hit = dict(
                            time_s=float(data.time),
                            geoms=names,
                            position_m=contact.pos.tolist(),
                            opposing_force_n=float(force[0] * opposing),
                        )
                if hit is not None and data.time - hit["time_s"] >= 0.03:
                    reached = True
                    break
            elif phase not in {"settle", "hold"} and distance < 0.05:
                reached = True
                break
            if not finite or _warning_counts(data) or tilt > 0.5:
                break
        if phase in {"settle", "hold"}:
            reached = np.linalg.norm(data.qpos[:2] - target) < 0.08
        phases.append(
            dict(
                phase=phase,
                reached=bool(reached),
                time_s=float(data.time),
                position_xyz_m=data.qpos[:3].tolist(),
            )
        )
        if hit is not None:
            hits.append(hit)
        if not reached or not finite or _warning_counts(data) or max_tilt > 0.5:
            break
    passed = finite and not _warning_counts(data) and max_pen <= 0.025 and max_tilt <= 0.5
    passed = (
        passed
        and len(phases) == len(targets)
        and all(p["reached"] for p in phases)
        and len(hits) == 1
    )
    return dict(
        route_id=route["id"],
        speed_m_s=speed,
        status="PASS" if passed else "FAIL",
        finite=finite,
        warnings=_warning_counts(data),
        phases=phases,
        blocking_contacts=hits,
        max_penetration_m=max_pen,
        max_tilt_rad=max_tilt,
        max_abs_qacc=max_acc,
        max_contacts=max_contacts,
        unsupported_steps=unsupported_steps,
        max_unsupported_interval_s=longest_unsupported * float(model.opt.timestep),
        solver_config=solver_config(model),
        **trace.report(),
    )


def run_corner_checks(
    pack, source_manifest, *, profiles=None, speeds=(0.3, 0.5, 1.0), perimeter_catalog=None
):
    asset = pack if isinstance(pack, FieldAsset) else FieldAsset.open(pack, verify=True)
    selected = tuple(profiles) if profiles is not None else asset.available_runtime_profiles
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profiles must be nonempty and unique")
    if not speeds or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("speeds must be positive and finite")
    catalog = build_corner_catalog(asset, source_manifest, perimeter_catalog)
    trials, parity, reference = [], [], {}
    with tempfile.TemporaryDirectory(prefix="rmuc-corner-check-") as directory:
        robot = Path(directory) / "rover.xml"
        robot.write_text(rover_xml())
        for profile in selected:
            env = MuJoCoScenario(asset, robot, profile=profile)
            for route in catalog["routes"]:
                for speed in speeds:
                    result = drive_corner(env, route, speed)
                    trials.append({**result, "profile": profile})
                    key = (route["id"], speed)
                    value = (result["steps"], result["state_contact_sha256"])
                    if profile == selected[0]:
                        reference[key] = value
                    else:
                        parity.append(
                            dict(
                                route_id=key[0],
                                speed_m_s=speed,
                                profile=profile,
                                status="PASS" if value == reference[key] else "FAIL",
                            )
                        )
            del env
    passed = len(catalog["routes"]) == 4 and all(r["status"] == "PASS" for r in trials + parity)
    return dict(
        status="PASS" if passed else "FAIL",
        catalog=catalog,
        trials=trials,
        profile_parity=parity,
        robot_mjcf_sha256=hashlib.sha256(rover_xml().encode()).hexdigest(),
        controller="bounded waypoint wheel control; contact for 30ms then reverse",
        summary=dict(
            trials=len(trials),
            passed=sum(r["status"] == "PASS" for r in trials),
            failed=sum(r["status"] != "PASS" for r in trials),
        ),
    )
