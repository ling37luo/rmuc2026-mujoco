"""Free-flight dynamics probe for the two fixed RMUC 2026 fly ramps.

The rolling-wheel probe in :mod:`rmuc2026_mujoco.wheel_probe` stops at the
takeoff seam.  This module continues the same constrained 120 mm wheel with
zero longitudinal force after takeoff, measures the runtime heightfield's gap,
and distinguishes a lip impact from a wheel that reaches the landing top.

Run the ten-trial characterization matrix with::

    python -m rmuc2026_mujoco.jump_probe PACK --output jump-probe.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .errors import MujocoModelError
from .manifest import FieldAsset
from .query import load_heightfield
from .ramp_audit import (
    FIXED_FLY_RAMPS,
    REFERENCE_WHEEL_DIAMETER_M,
    FlyRampGeometry,
    wheel_probe_route_endpoints,
)
from .wheel_probe import (
    WheelProbeConfig,
    WheelSupportProfile,
    _mujoco,
    _probe_ids,
    _require_unique_heightfield,
    _sample_mujoco_heightfield,
    _support_profile_slope,
    _warning_counts,
    _wheel_heightfield_contacts,
    _write_json_atomic,
    build_wheel_probe_model,
)


# RMUC 2026 rulebook V2.0.0, section 4.4.2, Figure 4-38.  The plan length and
# width describe the marked interaction surface.  They are not the outer AABB
# of the structural STEP parts, which include borders and supports.
OFFICIAL_INTERACTION_SURFACE_PLAN_LENGTH_M = 1.145
OFFICIAL_INTERACTION_SURFACE_PLAN_WIDTH_M = 0.860
OFFICIAL_RAMP_SLOPE_DEGREES = 17.0
OFFICIAL_JUMP_GAP_M = 0.650
OFFICIAL_LANDING_PLATFORM_HEIGHT_M = 0.200
OFFICIAL_RAMP_LOW_EDGE_HEIGHT_M = 0.203
OFFICIAL_RAMP_HIGH_EDGE_HEIGHT_M = 0.553

FREE_FLIGHT_SPEEDS_M_S = (1.5, 1.8, 2.0, 2.2, 2.5)


@dataclass(frozen=True)
class JumpProbeConfig:
    """Bounded settings for runtime gap measurement and free-flight trials."""

    official_gap_m: float = OFFICIAL_JUMP_GAP_M
    maximum_gap_error_m: float = 0.015
    profile_step_m: float = 0.001
    gap_floor_sample_start_m: float = 0.15
    gap_floor_sample_end_m: float = 0.40
    landing_search_half_width_m: float = 0.10
    landing_face_minimum_rise_m: float = 0.020
    landing_top_inset_start_m: float = 0.030
    landing_top_inset_end_m: float = 0.150
    landing_height_tolerance_m: float = 0.015
    post_landing_observation_s: float = 0.15
    timeout_s: float = 4.0
    required_success_speed_m_s: float = 2.2

    def validated(self) -> JumpProbeConfig:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("jump-probe configuration values must be positive and finite")
        if self.gap_floor_sample_start_m >= self.gap_floor_sample_end_m:
            raise ValueError("jump-probe gap-floor sample interval is not ordered")
        if self.landing_top_inset_start_m >= self.landing_top_inset_end_m:
            raise ValueError("jump-probe landing-top sample interval is not ordered")
        if self.landing_search_half_width_m >= self.official_gap_m:
            raise ValueError("jump-probe landing search window is implausibly wide")
        return self


@dataclass(frozen=True)
class LandingProfile:
    """One centreline gap and landing edge measured from the runtime field."""

    takeoff_seam_along_m: float
    landing_edge_along_m: float
    measured_gap_m: float
    gap_floor_height_m: float
    landing_top_height_m: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def detect_landing_profile(
    along_m: np.ndarray,
    height_m: np.ndarray,
    *,
    takeoff_seam_along_m: float,
    config: JumpProbeConfig = JumpProbeConfig(),
) -> LandingProfile:
    """Detect the post-gap landing face from an ordered centreline profile."""

    config.validated()
    along = np.asarray(along_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    if along.ndim != 1 or height.shape != along.shape or len(along) < 3:
        raise ValueError("jump-probe profile must be matching one-dimensional arrays")
    if not np.isfinite(along).all() or not np.isfinite(height).all():
        raise ValueError("jump-probe profile must contain finite values")
    if np.any(np.diff(along) <= 0.0):
        raise ValueError("jump-probe profile coordinates must be strictly increasing")

    takeoff = float(takeoff_seam_along_m)
    floor_mask = (along >= takeoff + config.gap_floor_sample_start_m) & (
        along <= takeoff + config.gap_floor_sample_end_m
    )
    if not np.any(floor_mask):
        raise MujocoModelError("jump-probe profile does not cover the gap floor interval")
    floor_height = float(np.median(height[floor_mask]))

    expected_edge = takeoff + config.official_gap_m
    search_mask = (along >= expected_edge - config.landing_search_half_width_m) & (
        along <= expected_edge + config.landing_search_half_width_m
    )
    candidates = np.flatnonzero(
        search_mask & (height >= floor_height + config.landing_face_minimum_rise_m)
    )
    if len(candidates) == 0:
        raise MujocoModelError("jump-probe could not find the landing face near the official gap")
    landing_edge = float(along[int(candidates[0])])

    top_mask = (along >= landing_edge + config.landing_top_inset_start_m) & (
        along <= landing_edge + config.landing_top_inset_end_m
    )
    if not np.any(top_mask):
        raise MujocoModelError("jump-probe profile does not cover the landing top interval")
    top_height = float(np.median(height[top_mask]))
    if top_height <= floor_height + config.landing_face_minimum_rise_m:
        raise MujocoModelError("jump-probe detected no raised landing top")
    return LandingProfile(
        takeoff_seam_along_m=takeoff,
        landing_edge_along_m=landing_edge,
        measured_gap_m=landing_edge - takeoff,
        gap_floor_height_m=floor_height,
        landing_top_height_m=top_height,
    )


def measure_runtime_landing_profile(
    asset: FieldAsset,
    ramp: FlyRampGeometry,
    *,
    config: JumpProbeConfig = JumpProbeConfig(),
) -> LandingProfile:
    """Sample and measure one landing edge on the exact runtime heightfield."""

    config.validated()
    field = load_heightfield(asset)
    takeoff = ramp.cad_high_seam_along_m
    start = takeoff + min(
        config.gap_floor_sample_start_m,
        config.official_gap_m - config.landing_search_half_width_m,
    )
    stop = (
        takeoff
        + config.official_gap_m
        + config.landing_search_half_width_m
        + config.landing_top_inset_end_m
        + config.profile_step_m
    )
    along = np.arange(start, stop, config.profile_step_m, dtype=np.float64)
    spawn = asset.recommended_spawn
    low_x = ramp.low_edge_center_xyz_m[0] - float(spawn["x_before_translation_m"])
    low_y = ramp.low_edge_center_xyz_m[1] - float(spawn["y_before_translation_m"])
    x = low_x + along * ramp.uphill_unit_xy[0]
    y = low_y + along * ramp.uphill_unit_xy[1]
    height = _sample_mujoco_heightfield(field, x, y)
    return detect_landing_profile(
        along,
        height,
        takeoff_seam_along_m=takeoff,
        config=config,
    )


def run_jump_probe_trial(
    model: Any,
    data: Any,
    *,
    ramp: FlyRampGeometry,
    landing: LandingProfile,
    speed_m_s: float,
    start_center_height_m: float,
    support_profile: WheelSupportProfile,
    wheel_config: WheelProbeConfig = WheelProbeConfig(),
    jump_config: JumpProbeConfig = JumpProbeConfig(),
) -> dict[str, Any]:
    """Drive to the takeoff seam, coast in flight, and inspect the landing."""

    wheel_config.validated()
    jump_config.validated()
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
    if not math.isclose(timestep, wheel_config.timestep_s, rel_tol=0.0, abs_tol=1.0e-12):
        raise MujocoModelError(
            f"jump-probe model timestep {timestep} disagrees with config {wheel_config.timestep_s}"
        )
    low_approach_s, high_seam_s = wheel_probe_route_endpoints(ramp)
    radius = REFERENCE_WHEEL_DIAMETER_M / 2.0

    mujoco.mj_resetData(model, data)
    data.qpos[ids.path_qpos] = low_approach_s
    data.qpos[ids.vertical_qpos] = start_height
    data.qpos[ids.spin_qpos] = 0.0
    mujoco.mj_forward(model, data)
    settle_steps = max(1, int(math.ceil(wheel_config.settle_time_s / timestep)))
    for _ in range(settle_steps):
        data.qfrc_applied[:] = 0.0
        position_error = low_approach_s - float(data.qpos[ids.path_qpos])
        hold_acceleration = (
            wheel_config.velocity_gain_s_inv * (0.0 - float(data.qvel[ids.path_dof]))
            + wheel_config.velocity_gain_s_inv * position_error / wheel_config.settle_time_s
        )
        data.qfrc_applied[ids.path_dof] = float(
            np.clip(
                wheel_config.wheel_mass_kg * hold_acceleration,
                -wheel_config.maximum_drive_force_n,
                wheel_config.maximum_drive_force_n,
            )
        )
        mujoco.mj_step(model, data)

    data.qpos[ids.path_qpos] = low_approach_s
    initial_slope = _support_profile_slope(support_profile, low_approach_s)
    data.qvel[ids.path_dof] = speed
    data.qvel[ids.vertical_dof] = speed * initial_slope
    data.qvel[ids.spin_dof] = speed * math.sqrt(1.0 + initial_slope**2) / radius
    mujoco.mj_forward(model, data)

    maximum_steps = max(1, int(math.ceil(jump_config.timeout_s / timestep)))
    post_landing_steps = max(1, int(math.ceil(jump_config.post_landing_observation_s / timestep)))
    finite_state = True
    finite_contacts = True
    became_airborne = False
    first_recontact: dict[str, float] | None = None
    top_landing: dict[str, float] | None = None
    top_landing_step: int | None = None
    completed_post_landing_observation = False
    post_landing_minimum_height = math.inf
    post_landing_contact_steps = 0
    maximum_along = low_approach_s
    maximum_height = -math.inf
    minimum_height = math.inf
    contact_steps = 0
    takeoff: dict[str, float] | None = None
    slope_rise_per_run = math.tan(math.radians(ramp.slope_angle_degrees))

    for step in range(maximum_steps):
        coordinate = float(data.qpos[ids.path_qpos])
        velocity = float(data.qvel[ids.path_dof])
        data.qfrc_applied[:] = 0.0
        if coordinate < high_seam_s:
            drive_force = (
                wheel_config.wheel_mass_kg * wheel_config.velocity_gain_s_inv * (speed - velocity)
                + wheel_config.wheel_mass_kg * 9.81 * slope_rise_per_run
            )
            data.qfrc_applied[ids.path_dof] = float(
                np.clip(
                    drive_force,
                    -wheel_config.maximum_drive_force_n,
                    wheel_config.maximum_drive_force_n,
                )
            )
        mujoco.mj_step(model, data)

        coordinate = float(data.qpos[ids.path_qpos])
        center_height = float(data.xpos[ids.body, 2])
        velocity = float(data.qvel[ids.path_dof])
        vertical_velocity = float(data.qvel[ids.vertical_dof])
        maximum_along = max(maximum_along, coordinate)
        maximum_height = max(maximum_height, center_height)
        minimum_height = min(minimum_height, center_height)
        contacts, _force, contacts_finite = _wheel_heightfield_contacts(model, data, ids)
        contact_steps += int(contacts > 0)
        finite_contacts = finite_contacts and contacts_finite
        current_finite = bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
            and np.isfinite(data.qfrc_constraint).all()
        )
        finite_state = finite_state and current_finite
        if not current_finite or not contacts_finite:
            break

        if not became_airborne and coordinate >= high_seam_s and contacts == 0:
            became_airborne = True
            takeoff = {
                "along_m": coordinate,
                "horizontal_velocity_m_s": velocity,
                "vertical_velocity_m_s": vertical_velocity,
                "time_s": step * timestep,
            }
        elif became_airborne and first_recontact is None and contacts > 0:
            first_recontact = {
                "along_m": coordinate,
                "center_height_m": center_height,
                "horizontal_velocity_m_s": velocity,
                "time_s": step * timestep,
            }

        on_landing_top = (
            contacts > 0
            and coordinate >= landing.landing_edge_along_m + radius
            and center_height
            >= landing.landing_top_height_m + radius - jump_config.landing_height_tolerance_m
        )
        if on_landing_top and top_landing is None:
            top_landing_step = step
            top_landing = {
                "along_m": coordinate,
                "center_height_m": center_height,
                "horizontal_velocity_m_s": velocity,
                "time_s": step * timestep,
            }
        if top_landing_step is not None:
            post_landing_minimum_height = min(post_landing_minimum_height, center_height)
            post_landing_contact_steps += int(contacts > 0)
            if step - top_landing_step >= post_landing_steps:
                completed_post_landing_observation = True
                break

        if (
            coordinate > landing.landing_edge_along_m + 1.2
            or center_height < landing.gap_floor_height_m - 0.5
            or (first_recontact is not None and top_landing is None and velocity < -0.2)
        ):
            break

    warnings = _warning_counts(data)
    stable_top_landing = bool(
        top_landing is not None
        and completed_post_landing_observation
        and post_landing_contact_steps > 0
        and post_landing_minimum_height
        >= landing.landing_top_height_m + radius - 2.0 * jump_config.landing_height_tolerance_m
    )
    required_to_land = speed >= jump_config.required_success_speed_m_s - 1.0e-12
    checks = {
        "finite_state": finite_state,
        "finite_contact_forces": finite_contacts,
        "no_solver_warning": sum(warnings.values()) == 0,
        "became_airborne": became_airborne,
        "first_recontact_detected": first_recontact is not None,
        "reached_landing_top": top_landing is not None,
        "stable_post_landing_observation": stable_top_landing,
        "required_speed_landed": stable_top_landing if required_to_land else None,
    }
    core_pass = all(
        checks[name]
        for name in (
            "finite_state",
            "finite_contact_forces",
            "no_solver_warning",
            "became_airborne",
            "first_recontact_detected",
        )
    )
    required_pass = core_pass and (stable_top_landing or not required_to_land)
    if stable_top_landing:
        outcome = "LANDED_ON_TOP"
    elif first_recontact is not None:
        outcome = "SHORT_OR_LIP_IMPACT"
    elif became_airborne:
        outcome = "NO_RECONTACT_WITHIN_BOUNDS"
    else:
        outcome = "NO_TAKEOFF"
    return {
        "trial_id": f"{ramp.route_id}:free_flight:{speed:.1f}",
        "route_id": ramp.route_id,
        "source_part_index": ramp.source_part_index,
        "commanded_approach_speed_m_s": speed,
        "required_success_speed_m_s": jump_config.required_success_speed_m_s,
        "required_to_land": required_to_land,
        "status": "PASS" if required_pass else "FAIL",
        "outcome": outcome,
        "takeoff": takeoff,
        "first_recontact": first_recontact,
        "top_landing": top_landing,
        "completed_post_landing_observation": completed_post_landing_observation,
        "post_landing_minimum_center_height_m": (
            post_landing_minimum_height if top_landing is not None else None
        ),
        "post_landing_contact_steps": post_landing_contact_steps,
        "maximum_along_m": maximum_along,
        "minimum_center_height_m": minimum_height,
        "maximum_center_height_m": maximum_height,
        "contact_step_count": contact_steps,
        "solver_warnings": warnings,
        "checks": checks,
        "field_physics_modified": False,
        "probe_constraints": [
            "ramp_centerline_slide",
            "vertical_slide",
            "free_wheel_spin",
        ],
    }


def run_fixed_fly_ramp_jump_probe(
    asset: FieldAsset | str | Path,
    *,
    output_path: str | Path | None = None,
    runtime_profile: str = "collision_only",
    speeds_m_s: Sequence[float] = FREE_FLIGHT_SPEEDS_M_S,
    wheel_config: WheelProbeConfig = WheelProbeConfig(),
    jump_config: JumpProbeConfig = JumpProbeConfig(),
) -> dict[str, Any]:
    """Measure both gaps and characterize free flight across both ramps."""

    if not isinstance(asset, FieldAsset):
        asset = FieldAsset.open(asset, verify=True)
    wheel_config.validated()
    jump_config.validated()
    speeds = tuple(float(value) for value in speeds_m_s)
    if not speeds or not np.isfinite(speeds).all() or any(value <= 0.0 for value in speeds):
        raise ValueError("jump-probe speeds must be positive and finite")
    if tuple(sorted(speeds)) != speeds or len(set(speeds)) != len(speeds):
        raise ValueError("jump-probe speeds must be unique and increasing")
    if not any(value >= jump_config.required_success_speed_m_s for value in speeds):
        raise ValueError("jump-probe speeds do not cover the required success speed")

    dimensions: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for ramp in FIXED_FLY_RAMPS:
        landing = measure_runtime_landing_profile(asset, ramp, config=jump_config)
        gap_error = abs(landing.measured_gap_m - jump_config.official_gap_m)
        dimensions.append(
            {
                "route_id": ramp.route_id,
                "source_part_index": ramp.source_part_index,
                "official_marked_interaction_surface": {
                    "plan_length_m": OFFICIAL_INTERACTION_SURFACE_PLAN_LENGTH_M,
                    "plan_width_m": OFFICIAL_INTERACTION_SURFACE_PLAN_WIDTH_M,
                    "slope_degrees": OFFICIAL_RAMP_SLOPE_DEGREES,
                    "gap_m": OFFICIAL_JUMP_GAP_M,
                    "landing_platform_height_m": OFFICIAL_LANDING_PLATFORM_HEIGHT_M,
                    "low_edge_height_m": OFFICIAL_RAMP_LOW_EDGE_HEIGHT_M,
                    "high_edge_height_m": OFFICIAL_RAMP_HIGH_EDGE_HEIGHT_M,
                },
                "cad_structural_slope_face": {
                    "horizontal_run_m": ramp.horizontal_run_m,
                    "width_m": ramp.surface_width_m,
                    "slope_degrees": ramp.slope_angle_degrees,
                    "basis": "hash_bound_official_step_including_border_and_support",
                },
                "runtime_heightfield": landing.as_dict(),
                "gap_abs_error_m": gap_error,
                "gap_dimension_pass": gap_error <= jump_config.maximum_gap_error_m + 1.0e-12,
                "comparison_note": (
                    "Rulebook plan dimensions describe the marked interaction surface; the "
                    "STEP slope-face extents include structural border/support and are not "
                    "expected to equal 1.145 by 0.860 m."
                ),
            }
        )
        prepared = build_wheel_probe_model(
            asset,
            ramp,
            runtime_profile=runtime_profile,
            config=wheel_config,
        )
        low_approach_s, _high_seam_s = wheel_probe_route_endpoints(ramp)
        start_height = (
            float(
                np.interp(
                    low_approach_s,
                    prepared.support_profile.along_m,
                    prepared.support_profile.center_height_m,
                )
            )
            + 0.002
        )
        for speed in speeds:
            trials.append(
                run_jump_probe_trial(
                    prepared.model,
                    prepared.data,
                    ramp=ramp,
                    landing=landing,
                    speed_m_s=speed,
                    start_center_height_m=start_height,
                    support_profile=prepared.support_profile,
                    wheel_config=wheel_config,
                    jump_config=jump_config,
                )
            )

    required_trials = [trial for trial in trials if trial["required_to_land"]]
    dimensions_pass = all(record["gap_dimension_pass"] for record in dimensions)
    core_dynamics_pass = all(
        trial["checks"][name]
        for trial in trials
        for name in (
            "finite_state",
            "finite_contact_forces",
            "no_solver_warning",
            "became_airborne",
            "first_recontact_detected",
        )
    )
    required_landings_pass = bool(required_trials) and all(
        trial["checks"]["stable_post_landing_observation"] for trial in required_trials
    )
    result = {
        "schema_version": 1,
        "artifact_type": "rmuc2026_fixed_fly_ramp_free_flight_wheel_probe",
        "status": (
            "PASS" if dimensions_pass and core_dynamics_pass and required_landings_pass else "FAIL"
        ),
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
            "wheel_width_m": wheel_config.wheel_width_m,
            "speeds_m_s": list(speeds),
            "drive": (
                "bounded velocity servo before the CAD takeoff seam; zero longitudinal "
                "force after the seam"
            ),
            "constraints": [
                "ramp_centerline_slide",
                "vertical_slide",
                "free_wheel_spin",
            ],
            "wheel_config": asdict(wheel_config),
            "jump_config": asdict(jump_config),
            "field_physics_modified": False,
            "collision_owner": "existing_single_heightfield_only",
        },
        "dimensions": dimensions,
        "summary": {
            "dimension_records_passed": sum(record["gap_dimension_pass"] for record in dimensions),
            "dimension_record_count": len(dimensions),
            "landed_on_top_trials": sum(trial["outcome"] == "LANDED_ON_TOP" for trial in trials),
            "short_or_lip_impact_trials": sum(
                trial["outcome"] == "SHORT_OR_LIP_IMPACT" for trial in trials
            ),
            "required_speed_trials_passed": sum(
                trial["status"] == "PASS" for trial in required_trials
            ),
            "required_speed_trial_count": len(required_trials),
            "solver_warning_count": sum(sum(trial["solver_warnings"].values()) for trial in trials),
        },
        "trials": trials,
        "claim_boundary": (
            "This longitudinal centreline wheel probe isolates takeoff, free flight, the "
            "runtime gap and landing-top contact. It does not validate lateral steering, "
            "robot suspension, policy behavior, chassis attitude, material fidelity, or the "
            "whole field. The 2.2 m/s required-success threshold is a diagnostic gate for "
            "this 120 mm wheel, not an official robot speed requirement."
        ),
    }
    if output_path is not None:
        _write_json_atomic(Path(output_path), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rmuc2026_mujoco.jump_probe",
        description="Characterize fixed-fly-ramp free flight and landing contact.",
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
    result = run_fixed_fly_ramp_jump_probe(
        args.pack,
        output_path=args.output,
        runtime_profile=args.runtime_profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a module entrypoint
    raise SystemExit(main())


__all__ = [
    "FREE_FLIGHT_SPEEDS_M_S",
    "JumpProbeConfig",
    "LandingProfile",
    "OFFICIAL_INTERACTION_SURFACE_PLAN_LENGTH_M",
    "OFFICIAL_INTERACTION_SURFACE_PLAN_WIDTH_M",
    "OFFICIAL_JUMP_GAP_M",
    "OFFICIAL_LANDING_PLATFORM_HEIGHT_M",
    "OFFICIAL_RAMP_HIGH_EDGE_HEIGHT_M",
    "OFFICIAL_RAMP_LOW_EDGE_HEIGHT_M",
    "OFFICIAL_RAMP_SLOPE_DEGREES",
    "detect_landing_profile",
    "measure_runtime_landing_profile",
    "run_fixed_fly_ramp_jump_probe",
    "run_jump_probe_trial",
]
