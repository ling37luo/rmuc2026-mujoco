"""Automatic ordinary-slope evaluation, with one reusable session per process."""

from __future__ import annotations

from collections import Counter
from itertools import product
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

from .manifest import FieldAsset
from .scenarios import scenario_descriptor
from .slope_catalog import slope_catalog
from .slope_routes import select_slope_route
from .slope_runtime import SlopeSession
from .training import _cpu_rss_bytes


DIRECTIONS = ("uphill", "downhill", "roundtrip")


def slope_cases(
    catalog,
    *,
    patches=None,
    directions=DIRECTIONS,
    speeds=(0.3, 0.5),
    repeats=1,
    seed=20260923,
    duration_s=None,
):
    """Build a stable case/seed order independent of worker count."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not directions or any(d not in DIRECTIONS for d in directions):
        raise ValueError("directions must contain uphill, downhill or roundtrip")
    if not speeds or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("speeds must be positive and finite")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration must be positive and finite")
    ids = (
        patches
        if patches is not None
        else [p["patch_id"] for p in catalog["patches"] if p.get("route") is not None]
    )
    routes = [select_slope_route(catalog, patch) for patch in dict.fromkeys(ids)]
    if not routes:
        raise ValueError("no screened slope routes in this pack")
    cases = []
    for route, direction, speed, repeat in product(
        routes, dict.fromkeys(directions), dict.fromkeys(speeds), range(repeats)
    ):
        distance = route["length_m"] * (2 if direction == "roundtrip" else 1)
        cases.append(
            {
                "case_index": len(cases),
                "case_id": f"slope_{len(cases):05d}",
                "patch": route["route_id"],
                "direction": direction,
                "speed_mps": float(speed),
                "repeat": repeat,
                "seed": int(seed) + len(cases),
                "timeout_s": float(duration_s)
                if duration_s is not None
                else max(8.0, 5 + 2 * distance / speed),
            }
        )
    return cases


def run_slope_episode(session, case):
    """Reset, run to completion/failure/timeout, and keep an independent result.

    This is an evaluation episode boundary, not a physical motion intervention.
    The caller may immediately reset and run another case after any outcome.
    """
    started = time.perf_counter()
    session.mode = "policy"
    session.reset(
        patch=case["patch"], direction=case["direction"], speed=case["speed_mps"], seed=case["seed"]
    )
    outcome, reason, error = "TIMEOUT", "timeout", None
    for _ in range(max(1, math.ceil(case["timeout_s"] / session.model.opt.timestep))):
        try:
            healthy = session.step()
        except Exception as exc:
            outcome, reason, error = (
                "ERROR",
                "controller_or_step_exception",
                f"{type(exc).__name__}: {exc}",
            )
            break
        if not healthy or session.failure:
            outcome = "FAIL"
            reason = (session.failure or {}).get("reason", "numerical_instability")
            break
        if session.completed:
            outcome, reason = "PASS", None
            break
    report = session.report()
    return {
        **report,
        **case,
        "outcome": outcome,
        "failure_reason": reason,
        "error": error,
        "terminated": outcome != "TIMEOUT",
        "truncated": outcome == "TIMEOUT",
        "elapsed_wall_s": time.perf_counter() - started,
    }


def _slope_worker(payload):
    import mujoco

    started = time.perf_counter()
    field = FieldAsset.open(payload["asset_root"], verify=True)
    session = SlopeSession(
        field,
        robot=payload["robot"],
        controller=payload["controller"],
        patch=payload["cases"][0]["patch"],
        mode="policy",
        profile=payload["profile"],
        record_trajectory=payload["trajectory_dir"] is not None,
    )
    solver = {
        "timestep_s": float(session.model.opt.timestep),
        "integrator": mujoco.mjtIntegrator(int(session.model.opt.integrator)).name,
        "solver": mujoco.mjtSolver(int(session.model.opt.solver)).name,
        "iterations": int(session.model.opt.iterations),
        "tolerance": float(session.model.opt.tolerance),
    }
    episodes = []
    for case in payload["cases"]:
        try:
            report = run_slope_episode(session, case)
        except Exception as exc:
            # A reset/controller problem in one case must not discard earlier
            # results or prevent the next independent case from being tried.
            report = {
                **case,
                "outcome": "ERROR",
                "failure_reason": "reset_or_report_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "steps": 0,
                "terminated": True,
                "truncated": False,
            }
        report["worker_index"] = payload["worker_index"]
        if payload["trajectory_dir"] is not None:
            path = Path(payload["trajectory_dir"]) / f"{case['case_id']}.json"
            # Never attach stale state from an episode whose reset failed.
            rows = session.rows if "leg_results" in report else []
            path.write_text(json.dumps({"summary": report, "rows": rows}, indent=2) + "\n")
            report["trajectory"] = str(path)
        episodes.append(report)
        if payload["progress"]:
            print(
                f"SLOPE_CASE {case['case_id']} patch={case['patch']} "
                f"direction={case['direction']} speed={case['speed_mps']:.2f} "
                f"outcome={report['outcome']} reason={report['failure_reason']}",
                file=sys.stderr,
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return {
        "worker_index": payload["worker_index"],
        "episodes": episodes,
        "steps_per_second": sum(e["steps"] for e in episodes) / elapsed,
        "elapsed_wall_s": elapsed,
        "cpu_max_rss_bytes": _cpu_rss_bytes(),
        "robot_mjcf_sha256": session.robot_hash,
        "solver_config": solver,
        "mujoco_version": mujoco.__version__,
    }


def _counts(episodes):
    outcomes = Counter(e["outcome"] for e in episodes)
    return {
        "episodes": len(episodes),
        "passed": outcomes["PASS"],
        "failed": outcomes["FAIL"],
        "timed_out": outcomes["TIMEOUT"],
        "errors": outcomes["ERROR"],
        "success_rate": outcomes["PASS"] / len(episodes),
    }


def run_slope_batch(
    asset,
    *,
    robot=None,
    controller=None,
    patches=None,
    directions=DIRECTIONS,
    speeds=(0.3, 0.5),
    repeats=1,
    workers=1,
    duration_s=None,
    seed=20260923,
    profile="collision_only",
    trajectory_dir=None,
    progress=False,
):
    """Run all selected slopes/directions automatically with spawn multiprocessing.

    Each process owns one model/data/controller and reuses it across episodes.
    No renderer, RL framework, private policy or shared MuJoCo state is needed.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if profile != "collision_only":
        raise ValueError("automatic slope batches use profile=collision_only; use view for visuals")
    if (robot is None) != (controller is None):
        raise ValueError("provide both --robot and --controller, or neither for the example rover")
    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    catalog = slope_catalog(field)
    cases = slope_cases(
        catalog,
        patches=patches,
        directions=directions,
        speeds=speeds,
        repeats=repeats,
        seed=seed,
        duration_s=duration_s,
    )
    workers_used = min(workers, len(cases))
    trajectory_path = (
        None if trajectory_dir is None else Path(trajectory_dir).expanduser().resolve()
    )
    if trajectory_path is not None:
        trajectory_path.mkdir(parents=True, exist_ok=True)
        if any(trajectory_path.glob("slope_*.json")):
            raise ValueError(
                "trajectory directory already contains slope episodes; choose a new directory"
            )
    common = {
        "asset_root": str(field.root),
        "robot": None if robot is None else str(Path(robot).expanduser().resolve()),
        "controller": controller,
        "profile": profile,
        "progress": progress,
        "trajectory_dir": None if trajectory_path is None else str(trajectory_path),
    }
    payloads = [
        {**common, "worker_index": i, "cases": cases[i::workers_used]} for i in range(workers_used)
    ]
    started = time.perf_counter()
    if workers_used == 1:
        results = [_slope_worker(payloads[0])]
    else:
        with mp.get_context("spawn").Pool(workers_used) as pool:
            results = pool.map(_slope_worker, payloads)
    elapsed = time.perf_counter() - started
    episodes = sorted((e for w in results for e in w["episodes"]), key=lambda e: e["case_index"])
    counts = _counts(episodes)
    warnings = Counter()
    for e in episodes:
        warnings.update(e.get("warnings", {}))
    steps = sum(e["steps"] for e in episodes)
    return {
        "schema_version": 1,
        "backend": "mujoco",
        "scenario_id": "slope_basic",
        "status": "PASS" if counts["passed"] == len(cases) else "FAIL",
        "profile": profile,
        "source_manifest_sha256": field.manifest_sha256,
        "profile_hash": scenario_descriptor(field, "slope_basic", profile=profile)["profile_hash"],
        "heightfield_samples_sha256": field.collision["samples_sha256"],
        "robot_mjcf_sha256": results[0]["robot_mjcf_sha256"],
        "controller": controller or "builtin_example_rover",
        "solver_config": results[0]["solver_config"],
        "mujoco_version": results[0]["mujoco_version"],
        "seed": seed,
        "workers": workers_used,
        "environment_count": workers_used,
        "episodes_per_worker": [len(w["episodes"]) for w in results],
        "summary": counts,
        "by_direction": {
            d: _counts([e for e in episodes if e["direction"] == d])
            for d in DIRECTIONS
            if any(e["direction"] == d for e in episodes)
        },
        "failure_reasons": dict(
            Counter(e["failure_reason"] for e in episodes if e["failure_reason"])
        ),
        "warnings": dict(warnings),
        "nonfinite_episodes": sum(e.get("finite") is False for e in episodes),
        "steps": steps,
        "steps_per_second": steps / elapsed,
        "elapsed_wall_s": elapsed,
        "worker_steps_per_second": [w["steps_per_second"] for w in results],
        "worker_cpu_max_rss_bytes": [w["cpu_max_rss_bytes"] for w in results],
        "gpu_memory_bytes": None,
        "episodes": episodes,
        "validation_status": field.manifest.get("validation_status"),
    }
