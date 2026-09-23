"""Robot-agnostic turning benchmark and MuJoCo batch runner.

The field package owns the scenario, reset locations and stepping contract.  A
robot adapter owns its joint order, observation layout and action mapping.  In
particular, this module intentionally does not contain a Fudan controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import multiprocessing as mp
from pathlib import Path
import resource
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .control import _load_module
from .manifest import FieldAsset
from .mjcf import compose_with_robot
from .query import find_spawn_candidates
from .scenarios import get_scenario, scenario_descriptor


PHYSICS_HZ = 500
POLICY_HZ = 100
PHYSICS_TIMESTEP_S = 1.0 / PHYSICS_HZ
POLICY_INTERVAL_STEPS = PHYSICS_HZ // POLICY_HZ


@dataclass(frozen=True)
class TurnCommand:
    """A short command held by an external controller."""

    vx_mps: float
    yaw_rad_s: float
    height_m: float
    duration_s: float = 0.0
    segment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "vx_mps": self.vx_mps,
            "yaw_rad_s": self.yaw_rad_s,
            "height_m": self.height_m,
            "duration_s": self.duration_s,
            "segment": self.segment,
        }


@dataclass(frozen=True)
class TurnPhase:
    phase_id: str
    commands: tuple[TurnCommand, ...]
    workspace_radius_m: float
    initial_headings: int = 8
    policy_hz: int = POLICY_HZ
    physics_hz: int = PHYSICS_HZ
    train_duration_s: float = 8.0
    eval_duration_s: float = 20.0
    command_selection: str = "per_episode"

    def command_at(self, elapsed_s: float) -> TurnCommand:
        if not self.commands:
            raise ValueError(f"turn phase {self.phase_id!r} has no commands")
        remaining = max(0.0, float(elapsed_s))
        for command in self.commands:
            if remaining < max(command.duration_s, 1.0e-9):
                return command
            remaining -= command.duration_s
        return self.commands[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "commands": [command.to_dict() for command in self.commands],
            "workspace_radius_m": self.workspace_radius_m,
            "initial_headings": self.initial_headings,
            "policy_hz": self.policy_hz,
            "physics_hz": self.physics_hz,
            "train_duration_s": self.train_duration_s,
            "eval_duration_s": self.eval_duration_s,
            "command_selection": self.command_selection,
        }


@dataclass(frozen=True)
class TurnSpawn:
    x_m: float
    y_m: float
    terrain_height_m: float
    slope_deg: float
    relief_m: float
    heading_yaw_rad: float = 0.0
    source_grid_index_yx: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "terrain_height_m": self.terrain_height_m,
            "slope_deg": self.slope_deg,
            "relief_m": self.relief_m,
            "heading_yaw_rad": self.heading_yaw_rad,
            "source_grid_index_yx": list(self.source_grid_index_yx),
        }


class TurnController(Protocol):
    """Protocol implemented by a user-owned robot controller.

    ``reset`` and ``step`` may write ``data.ctrl`` or apply forces.  The core
    runner never assumes a joint order.  ``observe`` can return arbitrary
    scalar metrics for an external policy; it may return ``None``.
    """

    def reset(self, model: Any, data: Any, spawn: TurnSpawn, seed: int) -> None: ...

    def step(self, model: Any, data: Any, command: TurnCommand, step_index: int) -> None: ...

    def observe(self, model: Any, data: Any) -> Mapping[str, float] | None: ...


class NoOpTurnController:
    """Small controller used for infrastructure smoke tests."""

    def reset(self, model: Any, data: Any, spawn: TurnSpawn, seed: int) -> None:
        del model, data, spawn, seed

    def step(self, model: Any, data: Any, command: TurnCommand, step_index: int) -> None:
        del command, step_index
        if getattr(model, "nu", 0):
            data.ctrl[:] = 0.0

    def observe(self, model: Any, data: Any) -> Mapping[str, float] | None:
        del model, data
        return None


def turn_phase(phase_id: str) -> TurnPhase:
    """Return the deterministic command schedule for one registered phase."""

    phase_name = str(phase_id)
    if phase_name == "spin":
        commands = tuple(
            TurnCommand(0.0, yaw, height, segment=f"spin_{yaw:+.1f}_{height:.2f}")
            for height in (0.18, 0.24)
            for yaw in (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)
        )
        return TurnPhase(phase_name, commands, 0.75, command_selection="per_episode")
    if phase_name == "arc":
        commands = tuple(
            TurnCommand(vx, yaw, height, segment=f"arc_{vx:+.1f}_{yaw:+.1f}_{height:.2f}")
            for height in (0.18, 0.24)
            for vx in (-0.3, 0.3)
            for yaw in (-0.6, -0.3, 0.3, 0.6)
        )
        return TurnPhase(phase_name, commands, 1.5, command_selection="per_episode")
    if phase_name == "reversal":
        commands = tuple(
            TurnCommand(vx, yaw, height, duration_s=2.0, segment=f"reversal_{index}_{vx:+.1f}")
            for height in (0.18, 0.24)
            for vx in (0.0, -0.3, 0.3)
            for index, sequence in enumerate(((-0.6, 0.0, 0.6), (0.6, 0.0, -0.6)))
            for yaw in sequence
        )
        return TurnPhase(phase_name, commands, 1.5, command_selection="sequence")
    raise ValueError("unknown turn phase; choose one of: spin, arc, reversal")


def turn_phase_registry() -> dict[str, dict[str, Any]]:
    return {name: turn_phase(name).to_dict() for name in ("spin", "arc", "reversal")}


def screen_turn_spawns(
    asset: FieldAsset,
    *,
    robot_mjcf: str | Path | None = None,
    count: int = 8,
) -> tuple[TurnSpawn, ...]:
    """Select up to eight deterministic, terrain-only turn spawn points."""

    del robot_mjcf  # reserved for a future robot footprint adapter
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(int(count), 8)
    candidates = find_spawn_candidates(
        asset,
        count=count,
        footprint_radius_m=0.35,
        boundary_margin_m=0.50,
        minimum_separation_m=2.0,
        max_slope_deg=3.0,
        max_relief_m=0.02,
        ground_height_range_m=(-0.05, 0.05),
    )
    return tuple(
        TurnSpawn(
            x_m=float(candidate.x_m),
            y_m=float(candidate.y_m),
            terrain_height_m=float(candidate.terrain_height_m),
            slope_deg=float(candidate.maximum_footprint_slope_deg),
            relief_m=float(candidate.footprint_relief_upper_bound_m),
            source_grid_index_yx=tuple(candidate.grid_index_yx),
        )
        for candidate in candidates
    )


def turn_spawn_manifest(
    asset: FieldAsset,
    *,
    robot_mjcf: str | Path | None = None,
    count: int = 8,
) -> dict[str, Any]:
    """Return a reproducible descriptor for screened starts."""

    descriptor = scenario_descriptor(asset, "turn_basic", profile="collision_only")
    robot_hash = None
    if robot_mjcf is not None:
        robot_hash = _sha256_file(Path(robot_mjcf))
    spawns = screen_turn_spawns(asset, robot_mjcf=robot_mjcf, count=count)
    payload = {
        "scenario_id": "turn_basic",
        "source_manifest_sha256": asset.manifest_sha256,
        "profile_hash": descriptor["profile_hash"],
        "robot_mjcf_sha256": robot_hash,
        "selection_contract": get_scenario("turn_basic").spawn,
        "spawns": [spawn.to_dict() for spawn in spawns],
        "phases": turn_phase_registry(),
    }
    payload["spawn_manifest_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def load_turn_controller(spec: str | None, model: Any, data: Any) -> TurnController:
    """Load ``module:factory`` for the generic turning protocol."""

    if spec is None:
        return NoOpTurnController()
    if ":" not in spec:
        raise ValueError("controller must use module:object syntax")
    module_name, object_name = spec.split(":", 1)
    target = getattr(_load_module(module_name), object_name)
    if not callable(target):
        raise TypeError(f"controller target {spec!r} is not callable")
    try:
        candidate = target(model, data)
    except TypeError:
        candidate = target()
    if not hasattr(candidate, "step") or not hasattr(candidate, "reset"):
        raise TypeError(
            f"controller target {spec!r} must return reset(model,data,spawn,seed) and "
            "step(model,data,command,step_index)"
        )
    if not hasattr(candidate, "observe"):
        candidate.observe = lambda model, data: None  # type: ignore[attr-defined]
    return candidate


def reset_turn_spawn(
    model: Any,
    data: Any,
    spawn: TurnSpawn,
    *,
    heading_yaw_rad: float = 0.0,
) -> None:
    """Place a generic freejoint robot at a screened XY start.

    The vertical pose and leg configuration remain the robot adapter's
    responsibility.  This helper only changes XY and heading, avoiding a
    robot-specific height assumption.
    """

    import mujoco

    mujoco.mj_resetData(model, data)
    joint_id = _freejoint_id(model)
    if joint_id is None:
        mujoco.mj_forward(model, data)
        return
    qadr = int(model.jnt_qposadr[joint_id])
    vadr = int(model.jnt_dofadr[joint_id])
    data.qpos[qadr] = float(spawn.x_m)
    data.qpos[qadr + 1] = float(spawn.y_m)
    data.qpos[qadr + 3 : qadr + 7] = _yaw_quaternion(heading_yaw_rad)
    data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    _lift_freejoint_clear_of_field(model, data, joint_id=joint_id, qadr=qadr)


def _lift_freejoint_clear_of_field(
    model: Any,
    data: Any,
    *,
    joint_id: int,
    qadr: int,
) -> None:
    """Resolve only initial robot/field penetration without changing geometry."""

    import mujoco

    del joint_id
    field_geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if (
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").endswith(
                "/rmuc2026_field_collision"
            )
            or (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                == "rmuc2026_field_collision"
            )
        )
    }
    if not field_geom_ids:
        return
    for _ in range(4):
        minimum_distance = 0.0
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            if int(contact.geom1) in field_geom_ids or int(contact.geom2) in field_geom_ids:
                minimum_distance = min(minimum_distance, float(contact.dist))
        if minimum_distance >= 0.0:
            return
        data.qpos[qadr + 2] += -minimum_distance + 0.005
        mujoco.mj_forward(model, data)


@dataclass
class TurnEpisodeResult:
    seed: int
    phase: str
    spawn: TurnSpawn
    steps: int
    sim_time_s: float
    warnings: dict[str, int] = field(default_factory=dict)
    nonfinite_count: int = 0
    contacts: int = 0
    max_penetration_m: float = 0.0
    max_tilt_deg: float | None = None
    min_height_m: float | None = None
    yaw_rate_rmse: float | None = None
    forward_velocity_rmse: float | None = None
    torque_saturation_rate: float | None = None
    failed: bool = False
    failure_reason: str | None = None
    elapsed_wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "phase": self.phase,
            "spawn": self.spawn.to_dict(),
            "steps": self.steps,
            "sim_time_s": self.sim_time_s,
            "warnings": dict(self.warnings),
            "nonfinite_count": self.nonfinite_count,
            "contacts": self.contacts,
            "max_penetration_m": self.max_penetration_m,
            "max_tilt_deg": self.max_tilt_deg,
            "min_height_m": self.min_height_m,
            "yaw_rate_rmse": self.yaw_rate_rmse,
            "forward_velocity_rmse": self.forward_velocity_rmse,
            "torque_saturation_rate": self.torque_saturation_rate,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "elapsed_wall_s": self.elapsed_wall_s,
        }


def run_turn_episode(
    model: Any,
    data: Any,
    controller: TurnController,
    phase: TurnPhase,
    spawn: TurnSpawn,
    *,
    seed: int,
    duration_s: float,
    heading_yaw_rad: float = 0.0,
) -> TurnEpisodeResult:
    """Run one environment and collect generic physical diagnostics."""

    import mujoco

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    steps_target = max(1, int(round(duration_s / float(model.opt.timestep))))
    spawn = replace(spawn, heading_yaw_rad=float(heading_yaw_rad))
    reset_turn_spawn(model, data, spawn, heading_yaw_rad=heading_yaw_rad)
    controller.reset(model, data, spawn, seed)
    result = TurnEpisodeResult(
        seed=seed, phase=phase.phase_id, spawn=spawn, steps=0, sim_time_s=0.0
    )
    yaw_errors: list[float] = []
    velocity_errors: list[float] = []
    saturation_steps = 0
    saturation_samples = 0
    minimum_height = float("inf")
    maximum_tilt = 0.0
    maximum_penetration = 0.0
    started = time.perf_counter()
    command = phase.commands[0]
    observed_yaw_errors: list[float] = []
    observed_velocity_errors: list[float] = []
    for step in range(steps_target):
        policy_interval_steps = max(1, int(round(1.0 / (POLICY_HZ * float(model.opt.timestep)))))
        if step == 0 and getattr(model, "nu", 0):
            # The first 2 ms policy cycle is deliberately zeroed so external
            # controllers start from a settled reset state.
            data.ctrl[:] = 0.0
        elif step >= policy_interval_steps and step % policy_interval_steps == 0:
            command = phase.command_at(float(data.time))
            controller.step(model, data, command, step)
        try:
            mujoco.mj_step(model, data)
        except FloatingPointError as exc:
            result.failed = True
            result.failure_reason = f"mujoco:{exc}"
            break
        result.steps = step + 1
        result.sim_time_s = float(data.time)
        result.contacts = max(result.contacts, int(data.ncon))
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if not finite:
            result.nonfinite_count += 1
            result.failed = True
            result.failure_reason = "nonfinite_state"
            break
        warnings = _warning_counts(data)
        result.warnings.update(warnings)
        if warnings:
            result.failed = True
            result.failure_reason = "mujoco_warning"
            break
        for contact_index in range(int(data.ncon)):
            maximum_penetration = max(
                maximum_penetration, max(0.0, -float(data.contact[contact_index].dist))
            )
        root = _freejoint_observation(model, data)
        if root is not None:
            minimum_height = min(minimum_height, root["height_m"])
            maximum_tilt = max(maximum_tilt, root["tilt_deg"])
            yaw_errors.append(root["yaw_rate_rad_s"] - command.yaw_rad_s)
            velocity_errors.append(root["forward_velocity_mps"] - command.vx_mps)
            if root["height_m"] < 0.07:
                result.failed = True
                result.failure_reason = "base_height_below_0.07m"
                break
            if root["tilt_deg"] > 60.0:
                result.failed = True
                result.failure_reason = "tilt_above_60deg"
                break
            distance = math.hypot(root["x_m"] - spawn.x_m, root["y_m"] - spawn.y_m)
            if distance > phase.workspace_radius_m:
                result.failed = True
                result.failure_reason = "workspace_envelope"
                break
        observed = controller.observe(model, data)
        if observed:
            if any(not math.isfinite(float(value)) for value in observed.values()):
                result.nonfinite_count += 1
                result.failed = True
                result.failure_reason = "nonfinite_controller_observation"
                break
            if "yaw_rate_error_rad_s" in observed:
                observed_yaw_errors.append(float(observed["yaw_rate_error_rad_s"]))
            if "forward_velocity_error_mps" in observed:
                observed_velocity_errors.append(float(observed["forward_velocity_error_mps"]))
        saturation, samples = _torque_saturation(model, data)
        saturation_steps += saturation
        saturation_samples += samples
    result.elapsed_wall_s = time.perf_counter() - started
    result.max_penetration_m = maximum_penetration
    result.max_tilt_deg = None if maximum_tilt == 0.0 and not yaw_errors else maximum_tilt
    result.min_height_m = None if minimum_height == float("inf") else minimum_height
    result.yaw_rate_rmse = _rmse(observed_yaw_errors or yaw_errors)
    result.forward_velocity_rmse = _rmse(observed_velocity_errors or velocity_errors)
    result.torque_saturation_rate = (
        saturation_steps / saturation_samples if saturation_samples else None
    )
    return result


def run_turn_batch(
    asset: FieldAsset | str | Path,
    *,
    robot_mjcf: str | Path,
    controller: str | None = None,
    phase: str = "spin",
    workers: int = 1,
    envs_per_worker: int = 1,
    duration_s: float = 8.0,
    seed: int = 20260922,
    profile: str = "collision_only",
) -> dict[str, Any]:
    """Run independent turning environments with spawn-based multiprocessing."""

    if workers <= 0 or envs_per_worker <= 0:
        raise ValueError("workers and envs_per_worker must be positive")
    if profile != "collision_only":
        raise ValueError("turn_basic runner requires profile='collision_only'")
    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    robot_path = Path(robot_mjcf).expanduser().resolve()
    if not robot_path.is_file():
        raise FileNotFoundError(robot_path)
    phase_spec = turn_phase(phase)
    spawn_manifest = turn_spawn_manifest(field, robot_mjcf=robot_path)
    spawns = tuple(TurnSpawn(**_spawn_kwargs(row)) for row in spawn_manifest["spawns"])
    if not spawns:
        raise ValueError("turn_basic found no valid spawn points in this runtime pack")
    payload = {
        "asset_root": str(field.root),
        "robot_mjcf": str(robot_path),
        "controller": controller,
        "phase": phase,
        "duration_s": float(duration_s),
        "seed": int(seed),
        "profile": profile,
        "spawns": [spawn.to_dict() for spawn in spawns],
        "envs_per_worker": int(envs_per_worker),
    }
    started = time.perf_counter()
    if workers == 1:
        worker_results = [_turn_worker((0, payload))]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            worker_results = pool.map(_turn_worker, [(index, payload) for index in range(workers)])
    elapsed = time.perf_counter() - started
    episodes = [item for worker in worker_results for item in worker["episodes"]]
    total_steps = sum(int(item["steps"]) for item in episodes)
    warnings = _merge_counts(item["warnings"] for item in episodes)
    result = {
        "status": "PASS"
        if not warnings and all(not item["failed"] for item in episodes)
        else "FAIL",
        "backend": "mujoco",
        "scenario_id": "turn_basic",
        "phase": phase,
        "profile": profile,
        "source_manifest_sha256": field.manifest_sha256,
        "profile_hash": spawn_manifest["profile_hash"],
        "robot_mjcf_sha256": _sha256_file(robot_path),
        "spawn_manifest_sha256": spawn_manifest["spawn_manifest_sha256"],
        "workers": workers,
        "envs_per_worker": envs_per_worker,
        "environment_count": len(episodes),
        "duration_s": duration_s,
        "seed": seed,
        "steps": total_steps,
        "steps_per_second": total_steps / elapsed if elapsed else None,
        "worker_steps_per_second": [worker["steps_per_second"] for worker in worker_results],
        "worker_resource_max_rss_bytes": [
            worker["resource_max_rss_bytes"] for worker in worker_results
        ],
        "warnings": warnings,
        "nonfinite_count": sum(int(item["nonfinite_count"]) for item in episodes),
        "contacts": sum(int(item["contacts"]) for item in episodes),
        "failed_episodes": sum(bool(item["failed"]) for item in episodes),
        "failure_reasons": _merge_failure_reasons(item.get("failure_reason") for item in episodes),
        "episodes": episodes,
        "phase_spec": phase_spec.to_dict(),
        "resource_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "elapsed_wall_s": elapsed,
    }
    return result


def _turn_worker(args: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
    worker_index, payload = args
    import mujoco

    field = FieldAsset.open(payload["asset_root"], verify=True)
    model, _ = compose_with_robot(
        field,
        payload["robot_mjcf"],
        profile=payload["profile"],
    )
    phase_spec = turn_phase(str(payload["phase"]))
    spawns = tuple(TurnSpawn(**_spawn_kwargs(row)) for row in payload["spawns"])
    started = time.perf_counter()
    episodes: list[dict[str, Any]] = []
    for env_index in range(int(payload["envs_per_worker"])):
        data = mujoco.MjData(model)
        env_seed = int(payload["seed"]) + worker_index * int(payload["envs_per_worker"]) + env_index
        spawn = spawns[(worker_index * int(payload["envs_per_worker"]) + env_index) % len(spawns)]
        heading = (
            2.0 * math.pi * ((env_seed % phase_spec.initial_headings) / phase_spec.initial_headings)
        )
        episode_spawn = replace(spawn, heading_yaw_rad=heading)
        controller = load_turn_controller(payload.get("controller"), model, data)
        episode_phase = _episode_phase_for_seed(phase_spec, env_seed)
        episode = run_turn_episode(
            model,
            data,
            controller,
            episode_phase,
            episode_spawn,
            seed=env_seed,
            duration_s=float(payload["duration_s"]),
            heading_yaw_rad=heading,
        )
        episodes.append(episode.to_dict())
    elapsed = time.perf_counter() - started
    return {
        "worker_index": worker_index,
        "environment_count": len(episodes),
        "steps": sum(int(item["steps"]) for item in episodes),
        "steps_per_second": sum(int(item["steps"]) for item in episodes) / elapsed
        if elapsed
        else None,
        "resource_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "episodes": episodes,
    }


def _episode_phase_for_seed(phase: TurnPhase, seed: int) -> TurnPhase:
    """Select one command row per episode while retaining reversal timing."""

    if not phase.commands:
        return phase
    if phase.phase_id == "reversal":
        row_size = 3
        row_count = max(1, len(phase.commands) // row_size)
        row = int(seed) % row_count
        return replace(phase, commands=phase.commands[row * row_size : (row + 1) * row_size])
    if phase.phase_id not in {"spin", "arc"}:
        return phase
    command = phase.commands[int(seed) % len(phase.commands)]
    return replace(phase, commands=(command,))


def _freejoint_id(model: Any) -> int | None:
    import mujoco

    for joint_id, joint_type in enumerate(model.jnt_type):
        if int(joint_type) == int(mujoco.mjtJoint.mjJNT_FREE):
            return joint_id
    return None


def _freejoint_observation(model: Any, data: Any) -> dict[str, float] | None:
    joint_id = _freejoint_id(model)
    if joint_id is None:
        return None
    qadr = int(model.jnt_qposadr[joint_id])
    vadr = int(model.jnt_dofadr[joint_id])
    quaternion = np.asarray(data.qpos[qadr + 3 : qadr + 7], dtype=float)
    yaw = _quaternion_yaw(quaternion)
    linear_velocity = np.asarray(data.qvel[vadr : vadr + 3], dtype=float)
    return {
        "x_m": float(data.qpos[qadr]),
        "y_m": float(data.qpos[qadr + 1]),
        "height_m": float(data.qpos[qadr + 2]),
        "yaw_rad": yaw,
        "yaw_rate_rad_s": float(data.qvel[vadr + 5]),
        "forward_velocity_mps": float(
            math.cos(yaw) * linear_velocity[0] + math.sin(yaw) * linear_velocity[1]
        ),
        "tilt_deg": _quaternion_tilt_deg(quaternion),
    }


def _warning_counts(data: Any) -> dict[str, int]:
    import mujoco

    return {
        mujoco.mjtWarning(index).name: int(warning.number)
        for index, warning in enumerate(data.warning)
        if int(warning.number)
    }


def _torque_saturation(model: Any, data: Any) -> tuple[int, int]:
    ranges = np.asarray(getattr(model, "actuator_forcerange", []), dtype=float)
    forces = np.asarray(getattr(data, "actuator_force", []), dtype=float)
    if ranges.ndim != 2 or ranges.shape[0] == 0 or forces.size != ranges.shape[0]:
        return 0, 0
    bounded = np.isfinite(ranges).all(axis=1) & (ranges[:, 1] > ranges[:, 0])
    if not np.any(bounded):
        return 0, 0
    saturated = np.logical_and(
        bounded,
        np.logical_or(forces >= ranges[:, 1] - 1.0e-6, forces <= ranges[:, 0] + 1.0e-6),
    )
    return int(np.any(saturated)), 1


def _rmse(values: Sequence[float]) -> float | None:
    if not values:
        return None
    values_array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values_array * values_array)))


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)


def _quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quaternion_tilt_deg(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(math.degrees(math.acos(float(np.clip(up_z, -1.0, 1.0)))))


def _spawn_kwargs(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "x_m": float(row["x_m"]),
        "y_m": float(row["y_m"]),
        "terrain_height_m": float(row["terrain_height_m"]),
        "slope_deg": float(row["slope_deg"]),
        "relief_m": float(row["relief_m"]),
        "heading_yaw_rad": float(row.get("heading_yaw_rad", 0.0)),
        "source_grid_index_yx": tuple(row.get("source_grid_index_yx", (0, 0))),
    }


def _merge_counts(mappings: Sequence[Mapping[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            result[key] = result.get(key, 0) + int(value)
    return result


def _merge_failure_reasons(reasons: Sequence[str | None]) -> dict[str, int]:
    result: dict[str, int] = {}
    for reason in reasons:
        if reason:
            result[reason] = result.get(reason, 0) + 1
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "PHYSICS_HZ",
    "POLICY_HZ",
    "TurnCommand",
    "TurnController",
    "TurnEpisodeResult",
    "TurnPhase",
    "TurnSpawn",
    "NoOpTurnController",
    "load_turn_controller",
    "reset_turn_spawn",
    "run_turn_batch",
    "run_turn_episode",
    "screen_turn_spawns",
    "turn_phase",
    "turn_phase_registry",
    "turn_spawn_manifest",
]
