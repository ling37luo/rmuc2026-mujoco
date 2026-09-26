"""Repeatable, robot-independent field physics and profile benchmarks.

Run with ``python -m rmuc2026_mujoco.acceptance PACK --output runs/field.json``.
An optional route file supplies local, world-space approaches to seams and
walls; no unreleased CAD coordinates or private robot assets are embedded here.
The 120 mm route wheel is constrained to its centreline, so its results are
contact evidence rather than an articulated robot or policy acceptance test.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
import traceback
from typing import Any, Sequence

import mujoco
import numpy as np

from .manifest import FieldAsset
from .mjcf import HFIELD_NAME, compose_with_robot, inject_exact_heightfield
from .query import height_at


PROFILE_ORDER = ("full", "interactive_lite", "collision_only")
DEFAULT_ENV_COUNTS = (1, 4, 16)
DEFAULT_SPEEDS_M_S = (0.3, 0.5, 1.0)
WHEEL_RADIUS_M = 0.06
WHEEL_WIDTH_M = 0.08
WHEEL_MASS_KG = 2.0
DRIVE_GAIN_S_INV = 80.0
MAX_DRIVE_FORCE_N = 200.0
PROBE_BODY = "rmuc2026_acceptance_probe_body"
PROBE_GEOM = "rmuc2026_acceptance_probe_wheel"
PATH_JOINT = "rmuc2026_acceptance_probe_path"
VERTICAL_JOINT = "rmuc2026_acceptance_probe_vertical"


def _rss_bytes() -> int | None:
    """Current resident memory, not the lifetime process high-water mark."""

    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _warning_counts(data: Any) -> dict[str, int]:
    return {
        name: int(data.warning[int(kind)].number)
        for name, kind in mujoco.mjtWarning.__members__.items()
        if name != "mjNWARNING" and int(data.warning[int(kind)].number) > 0
    }


def _pair_name(model: Any, geom1: int, geom2: int) -> str:
    names = []
    for geom in (geom1, geom2):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name is None:
            body = int(model.geom_bodyid[geom])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_name is None:
                body_name = f"body#{body}"
            local_index = geom - int(model.body_geomadr[body])
            name = f"{body_name}/unnamed_geom[{local_index}]"
        names.append(name)
    return " | ".join(sorted(names))


def _contact_sample(model: Any, data: Any) -> tuple[Counter[str], float, bool]:
    pairs: Counter[str] = Counter()
    maximum_penetration = 0.0
    finite = True
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        distance = float(contact.dist)
        finite = finite and math.isfinite(distance)
        if finite:
            maximum_penetration = max(maximum_penetration, -distance)
        pairs[_pair_name(model, int(contact.geom1), int(contact.geom2))] += 1
    return pairs, maximum_penetration, finite


def _probe_model(
    asset: FieldAsset,
    profile: str,
    *,
    start_xy_m: tuple[float, float],
    direction_xy: tuple[float, float],
) -> tuple[Any, Any]:
    """Add the same public 120 mm wheel to any field profile without editing it."""

    spec = mujoco.MjSpec.from_file(str(asset.entrypoint_for(profile)))
    ux, uy = direction_xy
    lateral = (-uy, ux)
    body = spec.worldbody.add_body(name=PROBE_BODY, pos=[*start_xy_m, 0.0])
    body.add_joint(name=PATH_JOINT, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[ux, uy, 0])
    body.add_joint(
        name=VERTICAL_JOINT,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        damping=0.02,
    )
    body.add_joint(
        name="rmuc2026_acceptance_probe_spin",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[*lateral, 0],
        damping=0.001,
    )
    body.add_geom(
        name=PROBE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[
            -WHEEL_WIDTH_M / 2 * lateral[0],
            -WHEEL_WIDTH_M / 2 * lateral[1],
            0,
            WHEEL_WIDTH_M / 2 * lateral[0],
            WHEEL_WIDTH_M / 2 * lateral[1],
            0,
        ],
        size=[WHEEL_RADIUS_M],
        mass=WHEEL_MASS_KG,
        friction=[1, 0.005, 0.0001],
        condim=6,
    )
    model = spec.compile()
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    data = mujoco.MjData(model)
    vertical_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, VERTICAL_JOINT)
    data.qpos[int(model.jnt_qposadr[vertical_id])] = (
        height_at(asset, *start_xy_m, interpolation="mujoco") + WHEEL_RADIUS_M + 0.002
    )
    mujoco.mj_forward(model, data)
    return model, data


def _initial_model(
    asset: FieldAsset,
    profile: str,
    robot_xml: Path | None,
) -> tuple[Any, Any]:
    if robot_xml is not None:
        return compose_with_robot(asset, robot_xml, profile=profile)
    return _probe_model(asset, profile, start_xy_m=(0.0, 0.0), direction_xy=(1.0, 0.0))


def _run_envs(
    model: Any,
    initial: Any,
    *,
    env_count: int,
    steps: int,
    capture_trace: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started_allocation = time.perf_counter()
    data_items = []
    for _ in range(env_count):
        data = mujoco.MjData(model)
        data.qpos[:] = initial.qpos
        data.qvel[:] = initial.qvel
        data.ctrl[:] = initial.ctrl
        mujoco.mj_forward(model, data)
        data_items.append(data)
    allocation_s = time.perf_counter() - started_allocation
    rss_after_allocation = _rss_bytes()

    pairs: Counter[str] = Counter()
    trace_qpos: list[np.ndarray] = []
    trace_qvel: list[np.ndarray] = []
    trace_contacts: list[dict[str, int]] = []
    max_penetration = 0.0
    max_abs_qacc = 0.0
    max_contacts = 0
    finite = True
    warnings: Counter[str] = Counter()
    first_failure_step: int | None = None
    completed = 0
    started = time.perf_counter()
    for step in range(steps):
        for env_index, data in enumerate(data_items):
            mujoco.mj_step(model, data)
            complete_state = (data.qpos, data.qvel, data.qacc, data.ctrl)
            if not all(np.isfinite(array).all() for array in complete_state):
                finite = False
            if np.isfinite(data.qacc).all() and data.qacc.size:
                max_abs_qacc = max(max_abs_qacc, float(np.max(np.abs(data.qacc))))
            max_contacts = max(max_contacts, int(data.ncon))
            if capture_trace:
                sample_pairs, penetration, contacts_finite = _contact_sample(model, data)
                pairs.update(sample_pairs)
                max_penetration = max(max_penetration, penetration)
                finite = finite and contacts_finite
                if len(trace_qpos) == step and data is data_items[0] and finite:
                    trace_qpos.append(np.array(data.qpos, copy=True))
                    trace_qvel.append(np.array(data.qvel, copy=True))
                    trace_contacts.append(dict(sample_pairs))
            current_warnings = _warning_counts(data)
            if current_warnings:
                warnings.update(current_warnings)
            if not finite or current_warnings:
                first_failure_step = step + 1
                break
        completed += env_count if first_failure_step is None else env_index + 1
        if first_failure_step is not None:
            break
    elapsed_s = time.perf_counter() - started
    result = {
        "status": "PASS" if finite and not warnings and completed == steps * env_count else "FAIL",
        "env_count": env_count,
        "steps_per_env_requested": steps,
        "total_steps_completed": completed,
        "steps_per_second": completed / elapsed_s if elapsed_s > 0 else None,
        "wall_time_s": elapsed_s,
        "data_allocation_s": allocation_s,
        "rss_after_allocation_bytes": rss_after_allocation,
        "rss_after_run_bytes": _rss_bytes(),
        "finite": finite,
        "warnings": dict(warnings),
        "first_failure_step": first_failure_step,
        "max_abs_qacc": max_abs_qacc,
        "max_contacts_per_env": max_contacts,
        "max_penetration_m": max_penetration if capture_trace else None,
        "contact_pairs": dict(sorted(pairs.items())) if capture_trace else None,
    }
    trace = (
        {"qpos": trace_qpos, "qvel": trace_qvel, "contacts": trace_contacts}
        if capture_trace
        else None
    )
    return result, trace


def _compare_trace(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    if not reference["qpos"] or not other["qpos"]:
        return {"status": "FAIL", "reason": "no complete finite trace"}
    if len(reference["qpos"]) != len(other["qpos"]):
        return {"status": "FAIL", "reason": "different completed step counts"}
    max_qpos = 0.0
    max_qvel = 0.0
    pairs_match = True
    for reference_qpos, other_qpos, reference_qvel, other_qvel, ref_pairs, other_pairs in zip(
        reference["qpos"],
        other["qpos"],
        reference["qvel"],
        other["qvel"],
        reference["contacts"],
        other["contacts"],
    ):
        if reference_qpos.shape != other_qpos.shape or reference_qvel.shape != other_qvel.shape:
            return {"status": "FAIL", "reason": "different state dimensions"}
        if reference_qpos.size:
            max_qpos = max(max_qpos, float(np.max(np.abs(reference_qpos - other_qpos))))
        if reference_qvel.size:
            max_qvel = max(max_qvel, float(np.max(np.abs(reference_qvel - other_qvel))))
        pairs_match = pairs_match and ref_pairs == other_pairs
    passed = max_qpos <= 1e-8 and max_qvel <= 1e-8 and pairs_match
    return {
        "status": "PASS" if passed else "FAIL",
        "max_abs_qpos_difference": max_qpos,
        "max_abs_qvel_difference": max_qvel,
        "contact_pairs_match_each_step": pairs_match,
        "tolerance": 1e-8,
    }


def _validate_routes(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    routes = document.get("routes") if isinstance(document, dict) else None
    if not isinstance(routes, list) or not routes:
        raise ValueError("route JSON needs a nonempty 'routes' list")
    seen: set[str] = set()
    selected = []
    for route in routes:
        if not isinstance(route, dict):
            raise ValueError("each route must be an object")
        name = route.get("id")
        outcome = route.get("expect")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("route id must be unique and nonempty")
        if outcome not in ("traverse", "block"):
            raise ValueError(f"route {name!r} expect must be 'traverse' or 'block'")
        blocking = route.get("blocking_geom")
        blocking_names = [blocking] if isinstance(blocking, str) else blocking
        if outcome == "block" and (
            not isinstance(blocking_names, list)
            or not blocking_names
            or any(not isinstance(item, str) or not item for item in blocking_names)
            or len(set(blocking_names)) != len(blocking_names)
        ):
            raise ValueError(f"route {name!r} needs blocking_geom name(s) for a block expectation")
        directions = route.get("directions", ["forward", "reverse"])
        if (
            not isinstance(directions, list)
            or not directions
            or any(not isinstance(item, str) for item in directions)
            or len(set(directions)) != len(directions)
            or any(item not in ("forward", "reverse") for item in directions)
        ):
            raise ValueError(
                f"route {name!r} directions must be forward/reverse without duplicates"
            )
        endpoints = []
        for key in ("start_xy_m", "end_xy_m"):
            try:
                point = np.asarray(route.get(key), dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"route {name!r} {key} must be two finite coordinates") from exc
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(f"route {name!r} {key} must be two finite coordinates")
            endpoints.append(tuple(map(float, point)))
        if math.dist(*endpoints) <= 0.01:
            raise ValueError(f"route {name!r} must be longer than 1 cm")
        selected.append(
            {
                **route,
                "start_xy_m": endpoints[0],
                "end_xy_m": endpoints[1],
                "directions": directions,
            }
        )
        seen.add(name)
    return selected, hashlib.sha256(raw).hexdigest()


def _run_route(
    asset: FieldAsset,
    profile: str,
    route: dict[str, Any],
    *,
    direction: str,
    speed_m_s: float,
    max_penetration_m: float,
) -> dict[str, Any]:
    start = route["start_xy_m"] if direction == "forward" else route["end_xy_m"]
    end = route["end_xy_m"] if direction == "forward" else route["start_xy_m"]
    length = math.dist(start, end)
    unit = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
    model, data = _probe_model(asset, profile, start_xy_m=start, direction_xy=unit)
    path_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PATH_JOINT)
    path_qpos = int(model.jnt_qposadr[path_id])
    path_dof = int(model.jnt_dofadr[path_id])
    budget = min(20000, math.ceil((2 * length / speed_m_s + 0.5) / model.opt.timestep))
    pairs: Counter[str] = Counter()
    maximum_progress = 0.0
    maximum_penetration = 0.0
    maximum_abs_qacc = 0.0
    max_contacts = 0
    finite = True
    warnings: dict[str, int] = {}
    completed = 0
    for step in range(budget):
        force = WHEEL_MASS_KG * DRIVE_GAIN_S_INV * (speed_m_s - float(data.qvel[path_dof]))
        data.qfrc_applied[path_dof] = float(np.clip(force, -MAX_DRIVE_FORCE_N, MAX_DRIVE_FORCE_N))
        mujoco.mj_step(model, data)
        completed = step + 1
        maximum_progress = max(maximum_progress, float(data.qpos[path_qpos]))
        if data.qacc.size and np.isfinite(data.qacc).all():
            maximum_abs_qacc = max(maximum_abs_qacc, float(np.max(np.abs(data.qacc))))
        max_contacts = max(max_contacts, int(data.ncon))
        sample_pairs, penetration, contact_finite = _contact_sample(model, data)
        pairs.update(sample_pairs)
        maximum_penetration = max(maximum_penetration, penetration)
        finite = (
            finite
            and contact_finite
            and all(np.isfinite(values).all() for values in (data.qpos, data.qvel, data.qacc))
        )
        warnings = _warning_counts(data)
        if not finite or warnings or maximum_progress >= length:
            break
    blocking_geom = route.get("blocking_geom")
    expected_geoms = {blocking_geom} if isinstance(blocking_geom, str) else set(blocking_geom or ())
    blocked_by_expected_geom = bool(
        expected_geoms and any(expected_geoms.intersection(pair.split(" | ")) for pair in pairs)
    )
    reached = maximum_progress >= length
    outcome_ok = (
        reached if route["expect"] == "traverse" else not reached and blocked_by_expected_geom
    )
    passed = finite and not warnings and maximum_penetration <= max_penetration_m and outcome_ok
    return {
        "route_id": route["id"],
        "profile": profile,
        "direction": direction,
        "speed_m_s": speed_m_s,
        "expected": route["expect"],
        "status": "PASS" if passed else "FAIL",
        "steps": completed,
        "route_length_m": length,
        "maximum_progress_m": maximum_progress,
        "reached_finish": reached,
        "blocked_by_expected_geom": blocked_by_expected_geom,
        "finite": finite,
        "warnings": warnings,
        "max_contacts": max_contacts,
        "max_penetration_m": maximum_penetration,
        "max_abs_qacc": maximum_abs_qacc,
        "contact_pairs": dict(sorted(pairs.items())),
    }


def _benchmark_one_profile(
    pack: str,
    profile: str,
    robot_xml: str | None,
    steps: int,
    env_counts: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Measure one profile in an otherwise fresh Python process."""

    asset = FieldAsset.open(pack, verify=True)
    robot = Path(robot_xml) if robot_xml is not None else None
    rss_before_load = _rss_bytes()
    started = time.perf_counter()
    model, initial = _initial_model(asset, profile, robot)
    load_s = time.perf_counter() - started
    rss_after_load = _rss_bytes()
    results = []
    trace = None
    for count in env_counts:
        result, trial_trace = _run_envs(
            model,
            initial,
            env_count=count,
            steps=steps,
            capture_trace=count == 1,
        )
        results.append(result)
        if trial_trace is not None:
            trace = trial_trace
    return (
        {
            "model_load_s": load_s,
            "rss_before_load_bytes": rss_before_load,
            "rss_after_load_bytes": rss_after_load,
            "rss_model_delta_bytes": (
                rss_after_load - rss_before_load
                if rss_after_load is not None and rss_before_load is not None
                else None
            ),
            "model_ngeom": int(model.ngeom),
            "model_nmesh": int(model.nmesh),
            "model_nv": int(model.nv),
            "timestep_s": float(model.opt.timestep),
            "env_counts": results,
        },
        trace,
    )


def _profile_process(
    sender: Any,
    pack: str,
    profile: str,
    robot_xml: str | None,
    steps: int,
    env_counts: tuple[int, ...],
) -> None:
    try:
        result, trace = _benchmark_one_profile(pack, profile, robot_xml, steps, env_counts)
        sender.send({"result": result, "trace": trace})
    except BaseException:
        sender.send({"error": traceback.format_exc()})
    finally:
        sender.close()


def run_acceptance(
    pack: str | Path,
    *,
    profiles: Sequence[str] | None = None,
    robot_xml: str | Path | None = None,
    steps: int = 1000,
    env_counts: Sequence[int] = DEFAULT_ENV_COUNTS,
    routes_json: str | Path | None = None,
    route_ids: Sequence[str] | None = None,
    max_route_penetration_m: float = 0.025,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure identical models and wheel approaches across declared profiles."""

    if steps < 1 or any(count < 1 for count in env_counts) or 1 not in env_counts:
        raise ValueError("steps must be positive and env counts must include 1")
    if not math.isfinite(max_route_penetration_m) or max_route_penetration_m <= 0:
        raise ValueError("max_route_penetration_m must be positive and finite")
    asset = FieldAsset.open(pack, verify=True)
    selected = (
        tuple(profiles)
        if profiles is not None
        else tuple(
            profile for profile in PROFILE_ORDER if profile in asset.available_runtime_profiles
        )
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profiles must be a nonempty list without duplicates")
    for profile in selected:
        asset.entrypoint_for(profile)
    robot = None if robot_xml is None else Path(robot_xml).expanduser().resolve()
    if robot is not None and not robot.is_file():
        raise ValueError(f"robot XML does not exist: {robot}")
    routes, route_sha256 = (
        ([], None) if routes_json is None else _validate_routes(Path(routes_json))
    )
    if routes_json is not None:
        route_document = json.loads(Path(routes_json).read_text(encoding="utf-8"))
        declared_manifest = route_document.get("source_manifest_sha256")
        if declared_manifest is not None and declared_manifest != asset.manifest_sha256:
            raise ValueError("route preset source manifest does not match the selected pack")
    if route_ids is not None:
        requested = set(route_ids)
        available = {route["id"] for route in routes}
        if not requested or requested - available:
            raise ValueError(f"unknown or empty route selection: {sorted(requested - available)}")
        routes = [route for route in routes if route["id"] in requested]

    profile_results: dict[str, Any] = {}
    traces: dict[str, dict[str, Any]] = {}
    context = multiprocessing.get_context("spawn")
    for profile in selected:
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_profile_process,
            args=(
                sender,
                str(asset.root),
                profile,
                str(robot) if robot else None,
                steps,
                tuple(env_counts),
            ),
        )
        process.start()
        sender.close()
        try:
            payload = receiver.recv()
        except EOFError as exc:
            process.join()
            raise RuntimeError(
                f"isolated profile {profile!r} exited without a benchmark result "
                f"(exit code {process.exitcode})"
            ) from exc
        finally:
            receiver.close()
        process.join()
        if "error" in payload:
            raise RuntimeError(f"isolated profile {profile!r} failed:\n{payload['error']}")
        if process.exitcode != 0:
            raise RuntimeError(f"isolated profile {profile!r} exit code {process.exitcode}")
        profile_results[profile] = payload["result"]
        if payload["trace"] is not None:
            traces[profile] = payload["trace"]

    reference_name = "full" if "full" in traces else next(iter(traces), None)
    parity = {}
    if reference_name is not None:
        parity = {
            profile: _compare_trace(traces[reference_name], trace)
            for profile, trace in traces.items()
            if profile != reference_name
        }

    route_results = [
        _run_route(
            asset,
            profile,
            route,
            direction=direction,
            speed_m_s=speed,
            max_penetration_m=max_route_penetration_m,
        )
        for profile in selected
        for route in routes
        for direction in route["directions"]
        for speed in DEFAULT_SPEEDS_M_S
    ]
    passed = (
        all(
            result["status"] == "PASS"
            for profile in profile_results.values()
            for result in profile["env_counts"]
        )
        and all(result["status"] == "PASS" for result in parity.values())
        and all(result["status"] == "PASS" for result in route_results)
    )
    output = {
        "schema_version": 1,
        "artifact_type": "rmuc2026_field_acceptance_benchmark",
        "status": "PASS" if passed else "FAIL",
        "pack": str(asset.root),
        "manifest_sha256": asset.manifest_sha256,
        "validation_status": asset.manifest.get("validation_status"),
        "robot_mjcf_sha256": hashlib.sha256(robot.read_bytes()).hexdigest() if robot else None,
        "probe": "external_robot_zero_control" if robot else "constrained_120mm_wheel",
        "steps_per_env": steps,
        "profile_reference": reference_name,
        "profile_measurement_isolation": "fresh_spawned_process_per_profile",
        "profiles": profile_results,
        "profile_parity": parity,
        "route_file_sha256": route_sha256,
        "selected_route_ids": [route["id"] for route in routes],
        "route_max_penetration_m": max_route_penetration_m if routes else None,
        "route_trials": route_results,
        "viewer_fps": None,
        "claim_boundary": (
            "Headless MuJoCo dynamics and constrained wheel contact only. Viewer FPS requires a "
            "separate visible-window measurement; this does not certify articulated robot policies, "
            "official material coefficients, or whole-field multilevel topology."
        ),
    }
    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(output, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--robot", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--env-counts", nargs="+", type=int, default=DEFAULT_ENV_COUNTS)
    parser.add_argument("--routes", type=Path, default=None)
    parser.add_argument("--route-ids", nargs="+", default=None)
    parser.add_argument("--max-route-penetration-m", type=float, default=0.025)
    args = parser.parse_args(argv)
    result = run_acceptance(
        args.pack,
        profiles=args.profiles,
        robot_xml=args.robot,
        steps=args.steps,
        env_counts=args.env_counts,
        routes_json=args.routes,
        route_ids=args.route_ids,
        max_route_penetration_m=args.max_route_penetration_m,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
