"""MuJoCo dynamics probe for the two fixed RMUC 2026 fly ramps.

The probe adds one constrained 120 mm wheel to the field-only model.  The
wheel can move along a ramp centreline, move vertically, and rotate about its
axle.  A bounded velocity servo supplies traction while contact, vertical
motion, and wheel rotation remain MuJoCo dynamics.  No field geom, contact
parameter, or height sample is changed.

Run the complete 12-trial matrix with::

    python -m rmuc2026_mujoco.wheel_probe PACK --output wheel-probe.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .errors import MujocoModelError
from .manifest import FieldAsset
from .mjcf import FIELD_COLLISION_GEOM_NAME, HFIELD_NAME, inject_exact_heightfield
from .query import HeightFieldData, load_heightfield
from .ramp_audit import (
    FIXED_FLY_RAMPS,
    REFERENCE_WHEEL_DIAMETER_M,
    WHEEL_PROBE_ENDPOINT_BASIS,
    WHEEL_PROBE_LOW_SIDE_APPROACH_M,
    WHEEL_PROBE_SPEEDS_M_S,
    FlyRampGeometry,
    wheel_probe_route_endpoints,
)


PROBE_BODY_NAME = "rmuc2026_wheel_probe_body"
PROBE_WHEEL_GEOM_NAME = "rmuc2026_wheel_probe"
PROBE_PATH_JOINT_NAME = "rmuc2026_wheel_probe_path"
PROBE_VERTICAL_JOINT_NAME = "rmuc2026_wheel_probe_vertical"
PROBE_SPIN_JOINT_NAME = "rmuc2026_wheel_probe_spin"
LOW_SIDE_APPROACH_M = WHEEL_PROBE_LOW_SIDE_APPROACH_M


@dataclass(frozen=True)
class WheelProbeConfig:
    """Numerical settings and explicit anomaly thresholds for the probe."""

    timestep_s: float = 0.001
    settle_time_s: float = 0.25
    timeout_scale: float = 2.0
    timeout_margin_s: float = 1.0
    wheel_width_m: float = 0.08
    wheel_mass_kg: float = 2.0
    velocity_gain_s_inv: float = 80.0
    maximum_drive_force_n: float = 200.0
    minimum_speed_fraction: float = 0.15
    minimum_speed_floor_m_s: float = 0.03
    maximum_stall_duration_s: float = 0.25
    seam_half_window_m: float = 0.10
    maximum_seam_stall_duration_s: float = 0.15
    maximum_airborne_duration_s: float = 0.08
    maximum_support_clearance_m: float = 0.025
    maximum_vertical_speed_m_s: float = 1.5
    maximum_median_speed_error_fraction: float = 0.15
    maximum_median_speed_error_floor_m_s: float = 0.05
    support_profile_step_m: float = 0.0025

    def validated(self) -> WheelProbeConfig:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("wheel-probe configuration values must be positive and finite")
        if self.timeout_scale < 1.0:
            raise ValueError("timeout_scale must be at least 1")
        if self.wheel_width_m >= 0.5:
            raise ValueError("wheel_width_m is implausibly large for the fixed-ramp probe")
        return self


@dataclass(frozen=True)
class WheelSupportProfile:
    """Minimum wheel-centre height over the exact runtime heightfield."""

    along_m: np.ndarray
    center_height_m: np.ndarray


@dataclass
class PreparedWheelProbe:
    """A route-specific model plus the read-only support profile it measures."""

    model: Any
    data: Any
    ramp: FlyRampGeometry
    support_profile: WheelSupportProfile
    manifest_sha256: str
    runtime_profile: str


@dataclass(frozen=True)
class _ProbeIds:
    body: int
    wheel_geom: int
    field_geom: int
    path_qpos: int
    vertical_qpos: int
    spin_qpos: int
    path_dof: int
    vertical_dof: int
    spin_dof: int


def build_wheel_probe_model(
    asset: FieldAsset,
    ramp: FlyRampGeometry,
    *,
    runtime_profile: str = "collision_only",
    config: WheelProbeConfig = WheelProbeConfig(),
) -> PreparedWheelProbe:
    """Compile a route-specific probe model from a validated runtime pack.

    The field is loaded from its collision-only entrypoint and receives only a
    probe body.  Exact NPZ height samples are injected after compilation, just
    like the normal field loader.
    """

    config.validated()
    mujoco = _mujoco()
    entrypoint = asset.entrypoint_for(runtime_profile)
    try:
        spec = mujoco.MjSpec.from_file(str(entrypoint))
    except (RuntimeError, ValueError) as exc:
        raise MujocoModelError(f"MuJoCo could not parse wheel-probe field profile: {exc}") from exc
    spec.option.timestep = config.timestep_s

    spawn = asset.recommended_spawn
    low_x = ramp.low_edge_center_xyz_m[0] - float(spawn["x_before_translation_m"])
    low_y = ramp.low_edge_center_xyz_m[1] - float(spawn["y_before_translation_m"])
    uphill = np.asarray(ramp.uphill_unit_xy, dtype=np.float64)
    lateral = np.asarray((-uphill[1], uphill[0]), dtype=np.float64)
    half_width = config.wheel_width_m / 2.0

    body = spec.worldbody.add_body(name=PROBE_BODY_NAME, pos=[low_x, low_y, 0.0])
    body.add_joint(
        name=PROBE_PATH_JOINT_NAME,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[uphill[0], uphill[1], 0.0],
        damping=0.02,
    )
    body.add_joint(
        name=PROBE_VERTICAL_JOINT_NAME,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 0.0, 1.0],
        damping=0.02,
    )
    body.add_joint(
        name=PROBE_SPIN_JOINT_NAME,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[lateral[0], lateral[1], 0.0],
        damping=0.001,
    )
    body.add_geom(
        name=PROBE_WHEEL_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[
            -half_width * lateral[0],
            -half_width * lateral[1],
            0.0,
            half_width * lateral[0],
            half_width * lateral[1],
            0.0,
        ],
        size=[REFERENCE_WHEEL_DIAMETER_M / 2.0],
        mass=config.wheel_mass_kg,
        friction=[1.0, 0.005, 0.0001],
        condim=6,
        rgba=[0.92, 0.22, 0.10, 1.0],
    )
    try:
        model = spec.compile()
    except (RuntimeError, ValueError) as exc:
        raise MujocoModelError(f"MuJoCo could not compile wheel-probe model: {exc}") from exc
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    _require_unique_heightfield(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    profile = _build_support_profile(asset, ramp, step_m=config.support_profile_step_m)
    return PreparedWheelProbe(
        model=model,
        data=data,
        ramp=ramp,
        support_profile=profile,
        manifest_sha256=asset.manifest_sha256,
        runtime_profile=runtime_profile,
    )


def run_wheel_probe_trial(
    model: Any,
    data: Any,
    *,
    ramp: FlyRampGeometry,
    direction: Literal["uphill", "downhill"],
    speed_m_s: float,
    start_center_height_m: float,
    support_profile: WheelSupportProfile | None = None,
    config: WheelProbeConfig = WheelProbeConfig(),
) -> dict[str, Any]:
    """Run one bounded trial on an already prepared MuJoCo model.

    A caller supplying a model must use the public probe body/joint names.  The
    model is rejected unless it has exactly one heightfield asset and one
    heightfield collision geom.  ``start_center_height_m`` is the wheel-centre
    world height at the fixed start coordinate.  A support profile enables the
    clearance gate; without it, that gate is reported as not measured.
    """

    config.validated()
    if direction not in ("uphill", "downhill"):
        raise ValueError("direction must be 'uphill' or 'downhill'")
    speed = float(speed_m_s)
    start_height = float(start_center_height_m)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("speed_m_s must be positive and finite")
    if not math.isfinite(start_height):
        raise ValueError("start_center_height_m must be finite")

    mujoco = _mujoco()
    _require_unique_heightfield(model)
    ids = _probe_ids(model)
    timestep = float(model.opt.timestep)
    if not math.isclose(timestep, config.timestep_s, rel_tol=0.0, abs_tol=1.0e-12):
        raise MujocoModelError(
            f"wheel-probe model timestep {timestep} disagrees with config {config.timestep_s}"
        )

    sign = 1.0 if direction == "uphill" else -1.0
    low_approach_s, high_seam_s = wheel_probe_route_endpoints(ramp)
    start_s = low_approach_s if direction == "uphill" else high_seam_s
    finish_s = high_seam_s if direction == "uphill" else low_approach_s
    required_progress = sign * (finish_s - start_s)
    target_velocity = sign * speed
    radius = REFERENCE_WHEEL_DIAMETER_M / 2.0

    mujoco.mj_resetData(model, data)
    data.qpos[ids.path_qpos] = start_s
    data.qpos[ids.vertical_qpos] = start_height
    data.qpos[ids.spin_qpos] = 0.0
    mujoco.mj_forward(model, data)

    settle_steps = max(1, int(math.ceil(config.settle_time_s / timestep)))
    for _ in range(settle_steps):
        data.qfrc_applied[:] = 0.0
        position_error = start_s - float(data.qpos[ids.path_qpos])
        hold_acceleration = (
            config.velocity_gain_s_inv * (0.0 - float(data.qvel[ids.path_dof]))
            + config.velocity_gain_s_inv * position_error / config.settle_time_s
        )
        data.qfrc_applied[ids.path_dof] = float(
            np.clip(
                config.wheel_mass_kg * hold_acceleration,
                -config.maximum_drive_force_n,
                config.maximum_drive_force_n,
            )
        )
        mujoco.mj_step(model, data)

    # Keep the settled vertical contact state but place the route coordinate
    # exactly at the declared start before measurements begin.
    data.qpos[ids.path_qpos] = start_s
    initial_support_slope = (
        _support_profile_slope(support_profile, start_s) if support_profile is not None else 0.0
    )
    initial_vertical_velocity = target_velocity * initial_support_slope
    initial_spin_velocity = (
        target_velocity * math.sqrt(1.0 + initial_support_slope * initial_support_slope) / radius
    )
    # Start at the requested horizontal speed without injecting an artificial
    # normal impact.  At a seam, the support-envelope tangent can differ
    # substantially from the ramp's nominal plane, so initialize vertical and
    # rolling speed from the exact heightfield-derived support profile.
    data.qvel[ids.path_dof] = target_velocity
    data.qvel[ids.vertical_dof] = initial_vertical_velocity
    data.qvel[ids.spin_dof] = initial_spin_velocity
    mujoco.mj_forward(model, data)

    maximum_duration = required_progress / speed * config.timeout_scale + config.timeout_margin_s
    maximum_steps = max(1, int(math.ceil(maximum_duration / timestep)))
    finite_state = True
    finite_contact_forces = True
    reached_finish = False
    position_teleport_detected = False
    maximum_contact_force = 0.0
    previous_s = start_s
    directed_progress_trace: list[float] = []
    directed_speed_trace: list[float] = []
    coordinate_trace: list[float] = []
    vertical_speed_trace: list[float] = []
    clearance_trace: list[float] = []
    airborne_trace: list[bool] = []
    stall_trace: list[bool] = []
    seam_stall_trace: list[bool] = []
    contact_sample_count = 0

    slope_rise_per_run = math.tan(math.radians(ramp.slope_angle_degrees))
    minimum_directed_speed = max(
        config.minimum_speed_floor_m_s,
        config.minimum_speed_fraction * speed,
    )
    for _step in range(maximum_steps):
        coordinate = float(data.qpos[ids.path_qpos])
        path_velocity = float(data.qvel[ids.path_dof])
        on_slope = (
            ramp.cad_low_seam_along_m - 0.03 <= coordinate <= ramp.cad_high_seam_along_m + 0.03
        )
        gravity_feedforward = config.wheel_mass_kg * 9.81 * slope_rise_per_run if on_slope else 0.0
        drive_force = (
            config.wheel_mass_kg * config.velocity_gain_s_inv * (target_velocity - path_velocity)
            + gravity_feedforward
        )
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[ids.path_dof] = float(
            np.clip(
                drive_force,
                -config.maximum_drive_force_n,
                config.maximum_drive_force_n,
            )
        )
        mujoco.mj_step(model, data)

        coordinate = float(data.qpos[ids.path_qpos])
        delta_s = coordinate - previous_s
        if abs(delta_s) > max(0.05, 5.0 * speed * timestep):
            position_teleport_detected = True
        previous_s = coordinate
        directed_progress = sign * (coordinate - start_s)
        directed_speed = sign * float(data.qvel[ids.path_dof])
        wheel_contacts, contact_force, contacts_finite = _wheel_heightfield_contacts(
            model, data, ids
        )
        maximum_contact_force = max(maximum_contact_force, contact_force)
        finite_contact_forces = finite_contact_forces and contacts_finite
        contact_sample_count += int(wheel_contacts > 0)

        current_finite = bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
            and np.isfinite(data.qfrc_constraint).all()
        )
        finite_state = finite_state and current_finite
        if not current_finite or not contacts_finite:
            break

        in_measured_surface = (
            ramp.cad_low_seam_along_m - radius <= coordinate <= ramp.cad_high_seam_along_m
        )
        airborne_trace.append(in_measured_surface and wheel_contacts == 0)
        vertical_speed_trace.append(float(data.qvel[ids.vertical_dof]))
        if support_profile is not None and in_measured_surface:
            expected_center = float(
                np.interp(
                    coordinate,
                    support_profile.along_m,
                    support_profile.center_height_m,
                )
            )
            body_height = float(data.xpos[ids.body, 2])
            clearance_trace.append(body_height - expected_center)
        coordinate_trace.append(coordinate)
        directed_progress_trace.append(directed_progress)
        directed_speed_trace.append(directed_speed)

        away_from_start_and_finish = (
            directed_progress >= 0.05 and directed_progress <= required_progress - 0.05
        )
        stalled = away_from_start_and_finish and directed_speed < minimum_directed_speed
        stall_trace.append(stalled)
        near_seam = (
            abs(coordinate - ramp.cad_low_seam_along_m) <= config.seam_half_window_m
            or abs(coordinate - ramp.cad_high_seam_along_m) <= config.seam_half_window_m
        )
        seam_stall_trace.append(stalled and near_seam)
        if directed_progress >= required_progress:
            reached_finish = True
            break

    warning_counts = _warning_counts(data)
    warning_total = sum(warning_counts.values())
    duration = len(directed_progress_trace) * timestep
    maximum_progress = max(directed_progress_trace, default=0.0)
    maximum_stall = _maximum_true_duration(stall_trace, timestep)
    maximum_seam_stall = _maximum_true_duration(seam_stall_trace, timestep)
    maximum_airborne = _maximum_true_duration(airborne_trace, timestep)
    maximum_airborne_event = _maximum_true_event(
        airborne_trace,
        coordinate_trace,
        timestep,
    )
    maximum_clearance = max(clearance_trace, default=0.0)
    maximum_vertical_speed = max((abs(value) for value in vertical_speed_trace), default=0.0)
    warmup = min(len(directed_speed_trace), int(math.ceil(0.20 / timestep)))
    tracking_values = directed_speed_trace[warmup:] or directed_speed_trace
    median_speed = float(np.median(tracking_values)) if tracking_values else 0.0
    median_speed_error = abs(median_speed - speed)
    maximum_allowed_speed_error = max(
        config.maximum_median_speed_error_floor_m_s,
        config.maximum_median_speed_error_fraction * speed,
    )
    abnormal_bounce = (
        maximum_airborne > config.maximum_airborne_duration_s + 1.0e-12
        or maximum_clearance > config.maximum_support_clearance_m + 1.0e-12
        or maximum_vertical_speed > config.maximum_vertical_speed_m_s + 1.0e-12
    )
    edge_snag = maximum_seam_stall > config.maximum_seam_stall_duration_s + 1.0e-12
    checks = {
        "reached_finish": reached_finish,
        "finite_state": finite_state,
        "finite_contact_forces": finite_contact_forces,
        "no_solver_warning": warning_total == 0,
        "no_pose_teleport": not position_teleport_detected,
        "no_stall": maximum_stall <= config.maximum_stall_duration_s + 1.0e-12,
        "no_edge_snag": not edge_snag,
        "no_abnormal_bounce": not abnormal_bounce,
        "commanded_speed_tracked": median_speed_error <= maximum_allowed_speed_error + 1.0e-12,
    }
    passed = all(checks.values())
    return {
        "trial_id": f"{ramp.route_id}:{direction}:{speed:.1f}",
        "route_id": ramp.route_id,
        "source_part_index": ramp.source_part_index,
        "direction": direction,
        "commanded_speed_m_s": speed,
        "wheel_diameter_m": REFERENCE_WHEEL_DIAMETER_M,
        "wheel_width_m": config.wheel_width_m,
        "dynamic_status": "PASS" if passed else "FAIL",
        "duration_s": duration,
        "step_count": len(directed_progress_trace),
        "start_along_ramp_m": start_s,
        "finish_along_ramp_m": finish_s,
        "endpoint_basis": WHEEL_PROBE_ENDPOINT_BASIS,
        "cad_low_seam_along_m": ramp.cad_low_seam_along_m,
        "cad_high_seam_along_m": ramp.cad_high_seam_along_m,
        "nominal_horizontal_run_m": ramp.horizontal_run_m,
        "low_side_approach_m": LOW_SIDE_APPROACH_M,
        "required_progress_m": required_progress,
        "maximum_progress_m": maximum_progress,
        "final_along_ramp_m": coordinate_trace[-1] if coordinate_trace else start_s,
        "median_directed_speed_m_s": median_speed,
        "median_speed_abs_error_m_s": median_speed_error,
        "initial_support_slope_dz_ds": initial_support_slope,
        "initial_velocity_m_s": {
            "along_ramp_horizontal": target_velocity,
            "vertical": initial_vertical_velocity,
            "wheel_spin_rad_s": initial_spin_velocity,
        },
        "maximum_stall_duration_s": maximum_stall,
        "maximum_seam_stall_duration_s": maximum_seam_stall,
        "maximum_airborne_duration_s": maximum_airborne,
        "maximum_airborne_event": maximum_airborne_event,
        "maximum_support_clearance_m": maximum_clearance if support_profile is not None else None,
        "maximum_abs_vertical_speed_m_s": maximum_vertical_speed,
        "contact_step_count": contact_sample_count,
        "maximum_contact_force_n": maximum_contact_force,
        "solver_warnings": warning_counts,
        "checks": checks,
        "field_physics_modified": False,
        "probe_constraints": ["ramp_centerline_slide", "vertical_slide", "free_wheel_spin"],
    }


def run_fixed_fly_ramp_wheel_probe(
    asset: FieldAsset | str | Path,
    *,
    output_path: str | Path | None = None,
    runtime_profile: str = "collision_only",
    config: WheelProbeConfig = WheelProbeConfig(),
) -> dict[str, Any]:
    """Run both ramps, both directions, and all three required speeds."""

    if not isinstance(asset, FieldAsset):
        asset = FieldAsset.open(asset, verify=True)
    config.validated()
    trials: list[dict[str, Any]] = []
    for ramp in FIXED_FLY_RAMPS:
        prepared = build_wheel_probe_model(
            asset,
            ramp,
            runtime_profile=runtime_profile,
            config=config,
        )
        for direction in ("uphill", "downhill"):
            low_approach_s, high_seam_s = wheel_probe_route_endpoints(ramp)
            start_s = low_approach_s if direction == "uphill" else high_seam_s
            start_height = (
                float(
                    np.interp(
                        start_s,
                        prepared.support_profile.along_m,
                        prepared.support_profile.center_height_m,
                    )
                )
                + 0.002
            )
            for speed in WHEEL_PROBE_SPEEDS_M_S:
                trials.append(
                    run_wheel_probe_trial(
                        prepared.model,
                        prepared.data,
                        ramp=ramp,
                        direction=direction,
                        speed_m_s=speed,
                        start_center_height_m=start_height,
                        support_profile=prepared.support_profile,
                        config=config,
                    )
                )

    passed = all(trial["dynamic_status"] == "PASS" for trial in trials)
    result = {
        "schema_version": 1,
        "artifact_type": "rmuc2026_fixed_fly_ramp_dynamic_wheel_probe",
        "status": "PASS" if passed else "FAIL",
        "validation_scope_status": "DRAFT_BLOCKED",
        "runtime_pack": {
            "root": str(asset.root),
            "manifest_sha256": asset.manifest_sha256,
            "runtime_profile": runtime_profile,
            "collision_kind": str(asset.collision["kind"]),
            "heightfield_resolution_m": asset.collision.get("resolution_m"),
        },
        "probe": {
            "wheel_diameter_m": REFERENCE_WHEEL_DIAMETER_M,
            "trial_count": len(trials),
            "required_trial_count": 12,
            "speeds_m_s": list(WHEEL_PROBE_SPEEDS_M_S),
            "directions": ["uphill", "downhill"],
            "endpoint_basis": WHEEL_PROBE_ENDPOINT_BASIS,
            "low_side_approach_m": LOW_SIDE_APPROACH_M,
            "config": asdict(config),
            "field_physics_modified": False,
            "collision_owner": "existing_single_heightfield_only",
        },
        "summary": {
            "passed_trials": sum(trial["dynamic_status"] == "PASS" for trial in trials),
            "failed_trials": sum(trial["dynamic_status"] != "PASS" for trial in trials),
            "solver_warning_count": sum(sum(trial["solver_warnings"].values()) for trial in trials),
            "finite_state_all_trials": all(trial["checks"]["finite_state"] for trial in trials),
            "finite_contact_forces_all_trials": all(
                trial["checks"]["finite_contact_forces"] for trial in trials
            ),
        },
        "trials": trials,
        "claim_boundary": (
            "This constrained 120 mm centreline wheel probe covers only fixed STEP parts 392 "
            "and 397 on the runtime pack's single heightfield. It does not validate free "
            "steering, robot suspension or policy behavior, landing after launch, multilevel "
            "topology, dynamic facilities, or the whole field."
        ),
    }
    if output_path is not None:
        _write_json_atomic(Path(output_path), result)
    return result


def _build_support_profile(
    asset: FieldAsset,
    ramp: FlyRampGeometry,
    *,
    step_m: float,
) -> WheelSupportProfile:
    field = load_heightfield(asset)
    radius = REFERENCE_WHEEL_DIAMETER_M / 2.0
    along = np.arange(
        ramp.cad_low_seam_along_m - LOW_SIDE_APPROACH_M - radius,
        ramp.cad_high_seam_along_m + radius + step_m * 0.5,
        step_m,
        dtype=np.float64,
    )
    offsets = np.arange(-radius, radius + step_m * 0.5, step_m, dtype=np.float64)
    circle = np.sqrt(np.maximum(radius * radius - offsets * offsets, 0.0))
    source_s = (along[:, None] + offsets[None, :]).reshape(-1)
    spawn = asset.recommended_spawn
    low_x = ramp.low_edge_center_xyz_m[0] - float(spawn["x_before_translation_m"])
    low_y = ramp.low_edge_center_xyz_m[1] - float(spawn["y_before_translation_m"])
    source_x = low_x + source_s * ramp.uphill_unit_xy[0]
    source_y = low_y + source_s * ramp.uphill_unit_xy[1]
    terrain = _sample_mujoco_heightfield(field, source_x, source_y).reshape(
        len(along), len(offsets)
    )
    center = np.max(terrain + circle[None, :], axis=1)
    if not np.isfinite(center).all():
        raise MujocoModelError(f"non-finite wheel support profile for {ramp.route_id}")
    return WheelSupportProfile(along_m=along, center_height_m=center)


def _sample_mujoco_heightfield(
    field: HeightFieldData,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    if (
        np.any(x < field.x_m[0])
        or np.any(x > field.x_m[-1])
        or np.any(y < field.y_m[0])
        or np.any(y > field.y_m[-1])
    ):
        raise MujocoModelError("fixed fly-ramp wheel support profile is outside the heightfield")
    columns = np.clip(np.searchsorted(field.x_m, x, side="left") - 1, 0, len(field.x_m) - 2)
    rows = np.clip(np.searchsorted(field.y_m, y, side="left") - 1, 0, len(field.y_m) - 2)
    x0 = field.x_m[columns]
    y0 = field.y_m[rows]
    dx = field.x_m[columns + 1] - x0
    dy = field.y_m[rows + 1] - y0
    u = (x - x0) / dx
    v = (y - y0) / dy
    z00 = field.height_m[rows, columns]
    z01 = field.height_m[rows, columns + 1]
    z10 = field.height_m[rows + 1, columns]
    z11 = field.height_m[rows + 1, columns + 1]
    lower = v <= u
    dz_dx = np.where(lower, (z01 - z00) / dx, (z11 - z10) / dx)
    dz_dy = np.where(lower, (z11 - z01) / dy, (z10 - z00) / dy)
    return z00 + dz_dx * (x - x0) + dz_dy * (y - y0)


def _probe_ids(model: Any) -> _ProbeIds:
    mujoco = _mujoco()

    def object_id(kind: Any, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise MujocoModelError(f"wheel-probe model is missing {name!r}")
        return value

    path_joint = object_id(mujoco.mjtObj.mjOBJ_JOINT, PROBE_PATH_JOINT_NAME)
    vertical_joint = object_id(mujoco.mjtObj.mjOBJ_JOINT, PROBE_VERTICAL_JOINT_NAME)
    spin_joint = object_id(mujoco.mjtObj.mjOBJ_JOINT, PROBE_SPIN_JOINT_NAME)
    return _ProbeIds(
        body=object_id(mujoco.mjtObj.mjOBJ_BODY, PROBE_BODY_NAME),
        wheel_geom=object_id(mujoco.mjtObj.mjOBJ_GEOM, PROBE_WHEEL_GEOM_NAME),
        field_geom=object_id(mujoco.mjtObj.mjOBJ_GEOM, FIELD_COLLISION_GEOM_NAME),
        path_qpos=int(model.jnt_qposadr[path_joint]),
        vertical_qpos=int(model.jnt_qposadr[vertical_joint]),
        spin_qpos=int(model.jnt_qposadr[spin_joint]),
        path_dof=int(model.jnt_dofadr[path_joint]),
        vertical_dof=int(model.jnt_dofadr[vertical_joint]),
        spin_dof=int(model.jnt_dofadr[spin_joint]),
    )


def _require_unique_heightfield(model: Any) -> None:
    mujoco = _mujoco()
    hfield_geoms = np.flatnonzero(np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_HFIELD))
    if int(model.nhfield) != 1 or len(hfield_geoms) != 1:
        raise MujocoModelError(
            "wheel probe requires exactly one heightfield asset and one heightfield geom"
        )
    field_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FIELD_COLLISION_GEOM_NAME))
    if field_id < 0 or int(hfield_geoms[0]) != field_id:
        raise MujocoModelError(
            f"the unique heightfield geom must be named {FIELD_COLLISION_GEOM_NAME!r}"
        )


def _wheel_heightfield_contacts(
    model: Any,
    data: Any,
    ids: _ProbeIds,
) -> tuple[int, float, bool]:
    mujoco = _mujoco()
    count = 0
    maximum_force = 0.0
    finite = True
    force = np.zeros(6, dtype=np.float64)
    expected_pair = {ids.wheel_geom, ids.field_geom}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} != expected_pair:
            continue
        count += 1
        mujoco.mj_contactForce(model, data, index, force)
        finite = finite and bool(np.isfinite(force).all())
        if finite:
            maximum_force = max(maximum_force, float(np.linalg.norm(force[:3])))
    return count, maximum_force, finite


def _warning_counts(data: Any) -> dict[str, int]:
    mujoco = _mujoco()
    return {
        name.removeprefix("mjWARN_"): int(data.warning[int(value)].number)
        for name, value in mujoco.mjtWarning.__members__.items()
        if name != "mjNWARNING"
    }


def _maximum_true_duration(values: Sequence[bool], timestep: float) -> float:
    maximum = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum * timestep


def _maximum_true_event(
    values: Sequence[bool],
    coordinates: Sequence[float],
    timestep: float,
) -> dict[str, float | None]:
    best_start = -1
    best_stop = -1
    current_start = -1
    for index, value in enumerate(values):
        if value and current_start < 0:
            current_start = index
        if current_start >= 0 and (not value or index == len(values) - 1):
            stop = index if value and index == len(values) - 1 else index - 1
            if stop - current_start > best_stop - best_start:
                best_start, best_stop = current_start, stop
            current_start = -1
    if best_start < 0:
        return {
            "duration_s": 0.0,
            "start_along_ramp_m": None,
            "end_along_ramp_m": None,
        }
    return {
        "duration_s": (best_stop - best_start + 1) * timestep,
        "start_along_ramp_m": float(coordinates[best_start]),
        "end_along_ramp_m": float(coordinates[best_stop]),
    }


def _support_profile_slope(profile: WheelSupportProfile, coordinate: float) -> float:
    index = int(np.searchsorted(profile.along_m, coordinate, side="left"))
    lower = max(0, index - 1)
    upper = min(len(profile.along_m) - 1, index + 1)
    if lower == upper:
        return 0.0
    delta_s = float(profile.along_m[upper] - profile.along_m[lower])
    return float((profile.center_height_m[upper] - profile.center_height_m[lower]) / delta_s)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise MujocoModelError("MuJoCo is not installed") from exc
    return mujoco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rmuc2026_mujoco.wheel_probe",
        description="Run the fixed-fly-ramp 120 mm wheel dynamics matrix.",
    )
    parser.add_argument("pack", type=Path, help="validated RMUC 2026 runtime-pack directory")
    parser.add_argument("--output", type=Path, help="write the full JSON report atomically")
    parser.add_argument(
        "--runtime-profile",
        default="collision_only",
        help="runtime profile to compile (default: collision_only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_fixed_fly_ramp_wheel_probe(
        args.pack,
        output_path=args.output,
        runtime_profile=args.runtime_profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a module entrypoint
    raise SystemExit(main())


__all__ = [
    "LOW_SIDE_APPROACH_M",
    "PROBE_BODY_NAME",
    "PROBE_PATH_JOINT_NAME",
    "PROBE_SPIN_JOINT_NAME",
    "PROBE_VERTICAL_JOINT_NAME",
    "PROBE_WHEEL_GEOM_NAME",
    "PreparedWheelProbe",
    "WheelProbeConfig",
    "WheelSupportProfile",
    "build_wheel_probe_model",
    "run_fixed_fly_ramp_wheel_probe",
    "run_wheel_probe_trial",
]
