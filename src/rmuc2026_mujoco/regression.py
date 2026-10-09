"""One portable static-field regression, using public probes and example robots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import mujoco
import numpy as np

from .acceptance import _contact_sample, _run_route, _warning_counts
from .corner_check import TraceDigest, run_corner_checks, solver_config
from .fly_routes import fly_route_descriptor
from .fly_runtime import FlyRampSession
from .manifest import FieldAsset
from .perimeter_check import run_perimeter_checks
from .query import find_spawn_candidates
from .scenarios import scenario_descriptor
from .slope_catalog import slope_catalog
from .slope_demo import RoverController, rover_xml
from .slope_runtime import SlopeSession
from .stairs import run_stair_checks
from .training import MuJoCoScenario
from .wall_check import run_wall_checks

SUITE_ID = "rmuc2026_static_interaction_v1"


def _parity(rows):
    references, comparisons = {}, []
    for row in rows:
        key = row["case_id"]
        value = (row["steps"], row["state_contact_sha256"])
        if key not in references:
            references[key] = value
        else:
            comparisons.append(
                {
                    "case_id": key,
                    "profile": row["profile"],
                    "status": "PASS" if references[key] == value else "FAIL",
                }
            )
    return comparisons


def _surface_routes(asset):
    catalog = slope_catalog(asset)
    slopes = []
    for patch in catalog["patches"]:
        if patch.get("route") and patch["band_id"] not in {p["band_id"] for p in slopes}:
            slopes.append(patch)
    routes = [
        dict(
            id=p["route"]["route_id"],
            start_xy_m=p["route"]["low_xyz_m"][:2],
            end_xy_m=p["route"]["high_xyz_m"][:2],
            expect="traverse",
        )
        for p in slopes
    ]
    for name in ("fly_ramp_north", "fly_ramp_south"):
        route = fly_route_descriptor(asset, name)
        routes.append(
            dict(
                id=name,
                start_xy_m=route["approach_xyz_m"][:2],
                end_xy_m=route["takeoff_xyz_m"][:2],
                expect="traverse",
            )
        )
    return slopes, routes


def check_surface_routes(asset, profiles, speeds):
    slopes, routes = _surface_routes(asset)
    trials = []
    for profile in profiles:
        for route in routes:
            for direction in ("forward", "reverse"):
                for speed in speeds:
                    trace = TraceDigest()
                    result = _run_route(
                        asset,
                        profile,
                        route,
                        direction=direction,
                        speed_m_s=speed,
                        max_penetration_m=0.025,
                        trace_callback=trace.record,
                    )
                    trials.append(
                        {
                            **result,
                            **trace.report(),
                            "case_id": f"{route['id']}/{direction}/{speed}",
                        }
                    )
    parity = _parity(trials)
    passed = bool(slopes) and all(r["status"] == "PASS" for r in trials + parity)
    return dict(
        status="PASS" if passed else "FAIL",
        routes=routes,
        ordinary_patches=slopes,
        trials=trials,
        profile_parity=parity,
        summary=dict(trials=len(trials), passed=sum(r["status"] == "PASS" for r in trials)),
    )


def check_public_interfaces(asset, profiles):
    """Exercise public reset/step and programmatic manual/policy callbacks."""
    spawn = find_spawn_candidates(
        asset,
        count=1,
        footprint_radius_m=0.35,
        boundary_margin_m=0.5,
        max_slope_deg=3,
        max_relief_m=0.02,
    )
    if not spawn:
        raise ValueError("no screened flat spawn for public interaction check")
    spawn = spawn[0]
    route = dict(
        low_xyz_m=[spawn.x_m, spawn.y_m, spawn.terrain_height_m],
        uphill_unit_xy=[1, 0],
        length_m=0.3,
        heading_yaw_rad=0,
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="rmuc-regression-rover-") as directory:
        robot = Path(directory) / "rover.xml"
        robot.write_text(rover_xml())
        for profile in profiles:
            env = MuJoCoScenario(asset, robot, profile=profile, seed=0)
            initial = env.model.qpos0.copy()
            initial[:3] = [spawn.x_m, spawn.y_m, spawn.terrain_height_m + 0.113]
            initial[3:7] = [1, 0, 0, 0]
            for mode in ("human", "policy"):
                digests = []
                for repeat in range(2):
                    env.reset(qpos=initial)
                    controller = RoverController(route, "uphill", 0.3)
                    trace, warnings, finite = TraceDigest(), {}, True
                    pen = 0.0
                    controls = []
                    for step in range(1000):
                        if mode == "human" and step in (0, 300, 600, 900):
                            controller.press_name("X")
                            controller.press_name({0: "W", 300: "A", 600: "D", 900: "X"}[step])
                        controller(env.model, env.data, step=step, mode=mode)
                        env.step()
                        pairs, depth, contact_finite = _contact_sample(env.model, env.data)
                        trace.record(env.data, pairs)
                        controls.append(env.data.ctrl.copy())
                        pen = max(pen, depth)
                        finite &= contact_finite and all(
                            np.isfinite(v).all()
                            for v in (env.data.qpos, env.data.qvel, env.data.qacc)
                        )
                        warnings.update(_warning_counts(env.data))
                        if not finite or warnings:
                            break
                    digests.append(trace.report())
                control = np.asarray(controls)
                active = bool(np.max(np.abs(control)) > 0)
                if mode == "human":
                    difference = control[:, 1] - control[:, 0]
                    active &= bool(difference.max() > 0 and difference.min() < 0)
                repeatable = digests[0] == digests[1]
                passed = (
                    finite
                    and not warnings
                    and pen <= 0.025
                    and active
                    and repeatable
                    and trace.steps == 1000
                )
                rows.append(
                    dict(
                        case_id=f"interface/{mode}",
                        profile=profile,
                        status="PASS" if passed else "FAIL",
                        mode=mode,
                        finite=finite,
                        warnings=warnings,
                        reset_repeatable=repeatable,
                        control_active=active,
                        max_penetration_m=pen,
                        solver_config=solver_config(env.model),
                        **trace.report(),
                    )
                )
    parity = _parity(rows)
    return dict(
        status="PASS" if all(r["status"] == "PASS" for r in rows + parity) else "FAIL",
        trials=rows,
        profile_parity=parity,
        robot_mjcf_sha256=hashlib.sha256(rover_xml().encode()).hexdigest(),
        scope="Public composition, reset, 1000 steps repeated, manual key callbacks and policy callbacks; no GUI event/rendering claim.",
    )


def _session_episode(session, *, case_id, timeout_s):
    trace = TraceDigest()
    for _ in range(math.ceil(timeout_s / session.model.opt.timestep)):
        healthy = session.step()
        pairs, _, _ = _contact_sample(session.model, session.data)
        trace.record(session.data, pairs)
        if not healthy or session.failure or session.completed:
            break
    report = session.report()
    outcome = "COMPLETE" if session.completed else "FAILED" if session.failure else "TIMEOUT"
    # A controller failing to climb/jump is retained independently of field health.
    return {
        **report,
        **trace.report(),
        "solver_config": solver_config(session.model),
        "case_id": case_id,
        "task_outcome": outcome,
        "status": "PASS" if report["physics_status"] == "PASS" and report["finite"] else "FAIL",
    }


def check_example_episodes(asset, profiles, slopes, stair_catalog):
    rows = []
    for profile in profiles:
        if slopes:
            session = SlopeSession(asset, mode="policy", profile=profile, record_trajectory=False)
            for patch in slopes:
                for direction in ("uphill", "downhill", "roundtrip"):
                    session.reset(patch=patch["patch_id"], direction=direction, speed=0.5, seed=0)
                    rows.append(
                        _session_episode(
                            session,
                            case_id=f"slope/{patch['patch_id']}/{direction}",
                            timeout_s=4 * patch["route"]["length_m"] / 0.5 + 5,
                        )
                    )
            del session
        if stair_catalog and stair_catalog["patches"]:
            session = SlopeSession(
                asset,
                mode="policy",
                profile=profile,
                record_trajectory=False,
                route_catalog=stair_catalog,
            )
            for patch in stair_catalog["patches"]:
                for direction in ("uphill", "downhill"):
                    session.reset(patch=patch["patch_id"], direction=direction, speed=0.3, seed=0)
                    rows.append(
                        _session_episode(
                            session, case_id=f"stairs/{patch['patch_id']}/{direction}", timeout_s=10
                        )
                    )
            del session
        session = FlyRampSession(asset, profile=profile, speed_mps=2.0, record_trajectory=False)
        for scenario in ("fly_ramp_north", "fly_ramp_south"):
            session.reset(scenario_id=scenario, speed_mps=2.0, seed=0)
            rows.append(_session_episode(session, case_id=f"fly/{scenario}/2.0", timeout_s=8))
        del session
    parity = _parity(rows)
    return dict(
        status="PASS" if rows and all(r["status"] == "PASS" for r in rows + parity) else "FAIL",
        trials=rows,
        profile_parity=parity,
        summary=dict(
            episodes=len(rows),
            completed=sum(r["task_outcome"] == "COMPLETE" for r in rows),
            failed=sum(r["task_outcome"] == "FAILED" for r in rows),
            timed_out=sum(r["task_outcome"] == "TIMEOUT" for r in rows),
        ),
        scope="Numerical health and profile parity of included examples. Task outcomes are separate; field contact checks do not certify robot capabilities.",
    )


def summarize_sections(sections):
    if not sections:
        raise ValueError("no regression sections were run")
    failed = [name for name, result in sections.items() if result["status"] in {"FAIL", "ERROR"}]
    missing = [name for name, result in sections.items() if result["status"] == "NOT_AVAILABLE"]
    return dict(
        status="FAIL" if failed else "PARTIAL" if missing else "PASS",
        failed_sections=failed,
        unavailable_sections=missing,
        passed_sections=[name for name, result in sections.items() if result["status"] == "PASS"],
    )


def run_regression(
    pack, source_manifest, *, output, profiles=None, speeds=(0.3, 0.5, 1.0), progress=None
):
    """Generate routes and run checks; persist each section before continuing."""
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"output already exists: {target}")
    asset = FieldAsset.open(pack, verify=True)
    selected = tuple(profiles) if profiles is not None else asset.available_runtime_profiles
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profiles must be nonempty and unique")
    if not speeds or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("speeds must be positive and finite")
    for profile in selected:
        asset.entrypoint_for(profile)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema_version=1,
        suite_id=SUITE_ID,
        status="RUNNING",
        source_manifest_sha256=asset.manifest_sha256,
        profiles={
            p: scenario_descriptor(asset, "full_eval", profile=p)["profile_hash"] for p in selected
        },
        heightfield_samples_sha256=asset.collision["samples_sha256"],
        mujoco_version=mujoco.__version__,
        seed=0,
        speeds_m_s=list(speeds),
        validation_status=asset.manifest.get("validation_status"),
        geometry_modified=False,
        sections={},
    )

    def stage(name, run):
        if progress:
            progress(name, "RUNNING")
        start = time.perf_counter()
        try:
            value = run()
        except Exception as exc:
            value = dict(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        report["sections"][name] = {**value, "elapsed_s": time.perf_counter() - start}
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if progress:
            progress(name, value["status"])
        return value

    stage("interfaces", lambda: check_public_interfaces(asset, selected))
    surfaces = stage("slope_fly_contacts", lambda: check_surface_routes(asset, selected, speeds))
    stairs = stage("stairs", lambda: run_stair_checks(asset, source_manifest, speeds=speeds))

    def absent(reason):
        return dict(status="NOT_AVAILABLE", reason=reason)

    perimeter = stage(
        "perimeter",
        lambda: run_perimeter_checks(asset, source_manifest, profiles=selected, speeds=speeds)
        if asset.manifest.get("perimeter_fence")
        else absent("pack has no perimeter fence"),
    )
    stage(
        "corners",
        lambda: run_corner_checks(
            asset,
            source_manifest,
            profiles=selected,
            speeds=speeds,
            perimeter_catalog=perimeter.get("catalog"),
        )
        if asset.manifest.get("perimeter_fence")
        else absent("pack has no perimeter fence"),
    )
    stage(
        "walls",
        lambda: run_wall_checks(asset, source_manifest, profiles=selected, speeds=speeds)
        if asset.collision.get("source_contact_layer")
        else absent("pack has no source wall layer"),
    )
    stage(
        "example_robots",
        lambda: check_example_episodes(
            asset, selected, surfaces.get("ordinary_patches", []), stairs.get("catalog")
        ),
    )
    report.update(summarize_sections(report["sections"]))
    report["robot_task_outcomes"] = report["sections"]["example_robots"].get("summary")
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
