"""Deterministic, parallel fly-ramp robot evaluation on the unchanged field."""

from __future__ import annotations

from collections import Counter
from itertools import product
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

from .fly_runtime import FLY_SCENARIOS, FlyRampSession
from .manifest import FieldAsset
from .scenarios import scenario_descriptor
from .training import _cpu_rss_bytes


DEFAULT_SPEEDS_MPS = (1.5, 1.8, 2.0, 2.2, 2.5)


def fly_cases(
    *,
    scenarios=FLY_SCENARIOS,
    speeds=DEFAULT_SPEEDS_MPS,
    lateral_offsets=(0.0,),
    heading_offsets_deg=(0.0,),
    repeats=1,
    seed=20260923,
    duration_s=5.0,
):
    """Fix trial identity and seed before distributing cases to workers."""

    if not scenarios or any(name not in FLY_SCENARIOS for name in scenarios):
        raise ValueError(f"scenarios must contain only {FLY_SCENARIOS}")
    if not speeds or any(not math.isfinite(value) or value <= 0 for value in speeds):
        raise ValueError("fly-ramp speeds must be positive and finite")
    if not lateral_offsets or any(not math.isfinite(value) for value in lateral_offsets):
        raise ValueError("lateral offsets must be finite")
    if not heading_offsets_deg or any(not math.isfinite(value) for value in heading_offsets_deg):
        raise ValueError("heading offsets must be finite")
    if repeats < 1 or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("repeats and duration must be positive")
    cases = []
    for scenario, speed, lateral, heading, repeat in product(
        dict.fromkeys(scenarios),
        dict.fromkeys(speeds),
        dict.fromkeys(lateral_offsets),
        dict.fromkeys(heading_offsets_deg),
        range(repeats),
    ):
        index = len(cases)
        cases.append(
            {
                "case_index": index,
                "case_id": f"fly_{index:05d}",
                "scenario_id": scenario,
                "commanded_speed_mps": float(speed),
                "lateral_offset_m": float(lateral),
                "heading_offset_deg": float(heading),
                "repeat": repeat,
                "seed": int(seed) + index,
                "timeout_s": float(duration_s),
            }
        )
    return cases


def run_fly_episode(session, case):
    """Run one complete physical attempt without altering motion mid-flight."""

    started = time.perf_counter()
    session.mode = "policy"
    session.reset(
        scenario_id=case["scenario_id"],
        speed_mps=case["commanded_speed_mps"],
        lateral_offset_m=case["lateral_offset_m"],
        heading_offset_deg=case["heading_offset_deg"],
        seed=case["seed"],
    )
    outcome, reason, error = "TIMEOUT", None, None
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
            outcome = "PASS"
            break
    if outcome == "TIMEOUT":
        reason = session.progress.timeout_reason()
    report = session.report()
    return {
        **report,
        **case,
        "physics_status": "ERROR" if outcome == "ERROR" else report["physics_status"],
        "outcome": outcome,
        "failure_reason": reason,
        "error": error,
        "terminated": outcome != "TIMEOUT",
        "truncated": outcome == "TIMEOUT",
        "elapsed_wall_s": time.perf_counter() - started,
    }


def _fly_worker(payload):
    import mujoco

    started = time.perf_counter()
    field = FieldAsset.open(payload["asset_root"], verify=True)
    session = FlyRampSession(
        field,
        robot=payload["robot"],
        controller=payload["controller"],
        scenario_id=payload["cases"][0]["scenario_id"],
        speed_mps=payload["cases"][0]["commanded_speed_mps"],
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
            report = run_fly_episode(session, case)
        except Exception as exc:
            report = {
                **case,
                "outcome": "ERROR",
                "physics_status": "ERROR",
                "failure_reason": "reset_or_report_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "steps": 0,
                "terminated": True,
                "truncated": False,
            }
        report["worker_index"] = payload["worker_index"]
        if payload["trajectory_dir"] is not None:
            path = Path(payload["trajectory_dir"]) / f"{case['case_id']}.json"
            rows = session.rows if "flight_status" in report else []
            path.write_text(json.dumps({"summary": report, "rows": rows}, indent=2) + "\n")
            report["trajectory"] = str(path)
        episodes.append(report)
        if payload["progress"]:
            print(
                f"FLY_CASE {case['case_id']} ramp={case['scenario_id']} "
                f"speed={case['commanded_speed_mps']:.2f} "
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


def run_fly_batch(
    asset,
    *,
    robot=None,
    controller=None,
    scenarios=FLY_SCENARIOS,
    speeds=DEFAULT_SPEEDS_MPS,
    lateral_offsets=(0.0,),
    heading_offsets_deg=(0.0,),
    repeats=1,
    workers=1,
    duration_s=5.0,
    seed=20260923,
    profile="collision_only",
    trajectory_dir=None,
    progress=False,
):
    """Evaluate both ramps with one reusable model per spawned process."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if profile != "collision_only":
        raise ValueError("automatic fly-ramp batches use profile=collision_only")
    if (robot is None) != (controller is None):
        raise ValueError("provide both --robot and --controller, or neither for the example rover")
    if robot is None:
        from .fly_demo import MAX_FORWARD_SPEED_M_S

        if any(speed > MAX_FORWARD_SPEED_M_S for speed in speeds):
            raise ValueError(
                f"example fly rover supports up to {MAX_FORWARD_SPEED_M_S} m/s; "
                "provide --robot and --controller for a different range"
            )
    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    cases = fly_cases(
        scenarios=scenarios,
        speeds=speeds,
        lateral_offsets=lateral_offsets,
        heading_offsets_deg=heading_offsets_deg,
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
        if any(trajectory_path.glob("fly_*.json")):
            raise ValueError("trajectory directory already contains fly episodes; choose a new one")
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
        results = [_fly_worker(payloads[0])]
    else:
        with mp.get_context("spawn").Pool(workers_used) as pool:
            results = pool.map(_fly_worker, payloads)
    elapsed = time.perf_counter() - started
    episodes = sorted((e for w in results for e in w["episodes"]), key=lambda e: e["case_index"])
    outcomes = Counter(e["outcome"] for e in episodes)
    warnings = Counter()
    for episode in episodes:
        warnings.update(episode.get("warnings", {}))
    steps = sum(e["steps"] for e in episodes)
    physics_healthy = all(e.get("physics_status") == "PASS" for e in episodes)
    summary = {
        "episodes": len(episodes),
        "landed_stably": outcomes["PASS"],
        "failed": outcomes["FAIL"],
        "timed_out": outcomes["TIMEOUT"],
        "errors": outcomes["ERROR"],
        "success_rate": outcomes["PASS"] / len(episodes),
    }
    return {
        "schema_version": 1,
        "backend": "mujoco",
        "scenario_ids": list(dict.fromkeys(scenarios)),
        "status": "PASS" if physics_healthy else "FAIL",
        "task_status": "PASS" if outcomes["PASS"] == len(episodes) else "INCOMPLETE",
        "profile": profile,
        "source_manifest_sha256": field.manifest_sha256,
        "profile_hashes": {
            scenario: scenario_descriptor(field, scenario, profile=profile)["profile_hash"]
            for scenario in dict.fromkeys(scenarios)
        },
        "heightfield_samples_sha256": field.collision["samples_sha256"],
        "robot_mjcf_sha256": results[0]["robot_mjcf_sha256"],
        "controller": controller or "builtin_example_fly_rover",
        "solver_config": results[0]["solver_config"],
        "mujoco_version": results[0]["mujoco_version"],
        "seed": seed,
        "workers": workers_used,
        "environment_count": workers_used,
        "episodes_per_worker": [len(w["episodes"]) for w in results],
        "summary": summary,
        "by_scenario": {
            scenario: {
                "episodes": len(rows := [e for e in episodes if e["scenario_id"] == scenario]),
                "landed_stably": sum(e["outcome"] == "PASS" for e in rows),
            }
            for scenario in dict.fromkeys(scenarios)
        },
        "by_commanded_speed_mps": {
            str(speed): {
                "episodes": len(rows := [e for e in episodes if e["commanded_speed_mps"] == speed]),
                "landed_stably": sum(e["outcome"] == "PASS" for e in rows),
            }
            for speed in dict.fromkeys(speeds)
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


__all__ = ["DEFAULT_SPEEDS_MPS", "fly_cases", "run_fly_batch", "run_fly_episode"]
