"""Bounded collision evidence for the two fixed RMUC 2026 fly ramps.

The measurements in this module never alter the heightfield.  They inspect the
single existing collision owner and publish a deterministic wheel-probe trial
matrix for a separate MuJoCo dynamics run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


REFERENCE_WHEEL_DIAMETER_M = 0.120
FINE_SEAM_RESOLUTION_M = 0.010
RAMP_INTERIOR_INSET_M = 0.150
RAMP_INTERIOR_P95_ERROR_MAX_M = 0.001
SEAM_LOCATION_ERROR_MAX_M = 0.015
SEAM_PLANE_MATCH_TOLERANCE_M = 0.002
WHEEL_PROBE_SPEEDS_M_S = (0.3, 0.5, 1.0)
WHEEL_PROBE_LOW_SIDE_APPROACH_M = 0.18
WHEEL_PROBE_ENDPOINT_BASIS = "hash_bound_unsimplified_cad_ray_seams"
SOURCE_GEOMETRY_EVIDENCE_SHA256 = "8cd0aa8e0ed344debae21ad034d182f793f6a8b4690a1d306354376d8c34af7f"
SOURCE_GLTF_SHA256 = "5bc8042c0fcb90e6f23f1f6624dbf8af1177b582a56254b4df092ee04ef23705"
CAD_SEAM_REFERENCE_SPACING_M = 0.0025


@dataclass(frozen=True)
class FlyRampGeometry:
    """An audited ramp plane in the builder's pre-spawn coordinate frame."""

    route_id: str
    source_part_index: int
    low_edge_center_xyz_m: tuple[float, float, float]
    high_edge_center_xyz_m: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    uphill_unit_xy: tuple[float, float]
    horizontal_run_m: float
    surface_width_m: float
    slope_angle_degrees: float
    cad_low_seam_along_m: float
    cad_high_seam_along_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "source_part_index": self.source_part_index,
            "low_edge_center_xyz_m": list(self.low_edge_center_xyz_m),
            "high_edge_center_xyz_m": list(self.high_edge_center_xyz_m),
            "normal_xyz": list(self.normal_xyz),
            "uphill_unit_xy": list(self.uphill_unit_xy),
            "horizontal_run_m": self.horizontal_run_m,
            "surface_width_m": self.surface_width_m,
            "slope_angle_degrees": self.slope_angle_degrees,
            "cad_low_seam_along_m": self.cad_low_seam_along_m,
            "cad_high_seam_along_m": self.cad_high_seam_along_m,
        }


# These small numeric records are derived from the hash-bound official STEP
# part audit.  They contain no mesh, image, or rulebook asset.
FIXED_FLY_RAMPS = (
    FlyRampGeometry(
        route_id="fly_ramp_north_to_gap",
        source_part_index=392,
        low_edge_center_xyz_m=(1.405374363692241, 8.624889702713807, 0.1682647208451014),
        high_edge_center_xyz_m=(0.19661292174524192, 8.521394279870014, 0.5406570176361774),
        normal_xyz=(0.29237172608900674, 0.025033173932493675, 0.9559770467885724),
        uphill_unit_xy=(-0.996354549271394, -0.08530892184406735),
        horizontal_run_m=1.2131840446063435,
        surface_width_m=0.9633610298002253,
        slope_angle_degrees=17.064103547064715,
        cad_low_seam_along_m=0.025,
        cad_high_seam_along_m=1.180,
    ),
    FlyRampGeometry(
        route_id="fly_ramp_south_to_gap",
        source_part_index=397,
        low_edge_center_xyz_m=(-1.4053601691014155, -5.376202408365563, 0.16826472216939942),
        high_edge_center_xyz_m=(-0.19659879111710055, -5.272707028052935, 0.5406570184194796),
        normal_xyz=(-0.2923717412977785, -0.025033166272021404, 0.9559770423377867),
        uphill_unit_xy=(0.9963545518675088, 0.08530889152307497),
        horizontal_run_m=1.2131839772485442,
        surface_width_m=0.9633614686297474,
        slope_angle_degrees=17.06410441610087,
        cad_low_seam_along_m=0.025,
        cad_high_seam_along_m=1.180,
    ),
)


def wheel_probe_trial_plan() -> list[dict[str, Any]]:
    """Return the complete, deterministic dynamics trial matrix.

    Build-time geometry inspection cannot prove a wheel traverses a ramp under
    MuJoCo dynamics.  Each row therefore remains explicitly pending until a
    runner records a finite trace, solver warnings, progress, contacts and
    penetration for that exact trial.
    """

    return [
        {
            "trial_id": f"{ramp.route_id}:{direction}:{speed:.1f}",
            "route_id": ramp.route_id,
            "source_part_index": ramp.source_part_index,
            "direction": direction,
            "commanded_speed_m_s": speed,
            "wheel_diameter_m": REFERENCE_WHEEL_DIAMETER_M,
            "start_along_ramp_m": (
                wheel_probe_route_endpoints(ramp)[0]
                if direction == "uphill"
                else wheel_probe_route_endpoints(ramp)[1]
            ),
            "finish_along_ramp_m": (
                wheel_probe_route_endpoints(ramp)[1]
                if direction == "uphill"
                else wheel_probe_route_endpoints(ramp)[0]
            ),
            "endpoint_basis": WHEEL_PROBE_ENDPOINT_BASIS,
            "cad_low_seam_along_m": ramp.cad_low_seam_along_m,
            "cad_high_seam_along_m": ramp.cad_high_seam_along_m,
            "nominal_horizontal_run_m": ramp.horizontal_run_m,
            "low_side_approach_m": WHEEL_PROBE_LOW_SIDE_APPROACH_M,
            "lateral_offset_m": 0.0,
            "dynamic_status": "NOT_RUN_DURING_BUILD",
        }
        for ramp in FIXED_FLY_RAMPS
        for direction in ("uphill", "downhill")
        for speed in WHEEL_PROBE_SPEEDS_M_S
    ]


def wheel_probe_route_endpoints(ramp: FlyRampGeometry) -> tuple[float, float]:
    """Return the low approach and high seam in the hash-bound CAD frame."""

    low_approach = ramp.cad_low_seam_along_m - WHEEL_PROBE_LOW_SIDE_APPROACH_M
    high_seam = ramp.cad_high_seam_along_m
    if not math.isfinite(low_approach) or not math.isfinite(high_seam):
        raise ValueError("CAD-bound wheel-probe endpoints must be finite")
    if high_seam <= ramp.cad_low_seam_along_m or low_approach >= high_seam:
        raise ValueError("CAD-bound wheel-probe endpoints are not ordered")
    return low_approach, high_seam


def audit_fixed_fly_ramps(
    x_m: np.ndarray,
    y_m: np.ndarray,
    height_m: np.ndarray,
) -> dict[str, Any]:
    """Measure plane fidelity and edge locations without changing collision.

    Coordinates must be the builder's axis-normalized, ground-shifted arrays
    before the recommended-spawn translation.  This makes the fixed STEP-part
    coordinates stable even if a finer grid selects a slightly different spawn
    sample.
    """

    x, y, height, resolution = _validated_grid(x_m, y_m, height_m)
    records = [_audit_one_ramp(x, y, height, ramp, resolution) for ramp in FIXED_FLY_RAMPS]
    fine_resolution = resolution <= FINE_SEAM_RESOLUTION_M + 1.0e-9
    static_gates_pass = fine_resolution and all(
        record["checks"]["static_pass"] for record in records
    )
    return {
        "schema_version": 1,
        "artifact_type": "rmuc2026_fixed_fly_ramp_heightfield_audit",
        "status": (
            "PASS_STATIC_PENDING_DYNAMIC"
            if static_gates_pass
            else "FAIL_STATIC"
            if fine_resolution
            else "NOT_RUN_REQUIRES_1CM"
        ),
        "coordinate_frame": "axis_normalized_ground_shifted_before_spawn_translation",
        "source_geometry_evidence_sha256": SOURCE_GEOMETRY_EVIDENCE_SHA256,
        "source_official_gltf_sha256": SOURCE_GLTF_SHA256,
        "cad_seam_reference": {
            "method": (
                "highest downward ray along each ramp centreline; largest dominant-plane "
                "matching segment containing its midpoint"
            ),
            "sample_spacing_m": CAD_SEAM_REFERENCE_SPACING_M,
            "plane_match_tolerance_m": SEAM_PLANE_MATCH_TOLERANCE_M,
            "source": "unsimplified axis-normalized official OpenCascade GLTF",
        },
        "heightfield_resolution_xy_m": [resolution, resolution],
        "required_maximum_resolution_m": FINE_SEAM_RESOLUTION_M,
        "fine_resolution_pass": fine_resolution,
        "gates": {
            "ramp_interior_inset_m": RAMP_INTERIOR_INSET_M,
            "ramp_interior_plane_abs_error_p95_max_m": RAMP_INTERIOR_P95_ERROR_MAX_M,
            "seam_location_abs_error_max_m": SEAM_LOCATION_ERROR_MAX_M,
            "seam_plane_match_tolerance_m": SEAM_PLANE_MATCH_TOLERANCE_M,
        },
        "ramps": records,
        "wheel_probe": {
            "geometry_probe_computable": all(
                record["wheel_geometry_probe"]["finite"] for record in records
            ),
            "diameter_m": REFERENCE_WHEEL_DIAMETER_M,
            "speeds_m_s": list(WHEEL_PROBE_SPEEDS_M_S),
            "directions": ["uphill", "downhill"],
            "trials": wheel_probe_trial_plan(),
            "dynamic_execution": {
                "status": "NOT_RUN_DURING_BUILD",
                "required_observations": [
                    "finite_qpos_qvel_and_contact_forces",
                    "solver_warning_count",
                    "centerline_progress_m",
                    "wheel_heightfield_penetration_m",
                    "stall_duration_s",
                ],
                "acceptance": [
                    "reaches_finish_without_pose_teleport",
                    "no_nan_or_solver_warning",
                    "no_stall_or_edge_snag",
                ],
            },
        },
        "heightfield_modified": False,
        "planar_refinement_applied": False,
        "collision_owner": "existing_single_heightfield_only",
        "validation_scope_status": "DRAFT_BLOCKED",
        "claim_boundary": (
            "static measurements cover the two fixed ramp center surfaces and seams only; "
            "pending wheel trials must be run in MuJoCo and do not validate the jump, robot "
            "policy, multilevel topology, or the whole field"
        ),
    }


def _validated_grid(
    x_m: np.ndarray,
    y_m: np.ndarray,
    height_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    if (
        x.ndim != 1
        or y.ndim != 1
        or len(x) < 2
        or len(y) < 2
        or height.shape != (len(y), len(x))
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(height).all()
    ):
        raise ValueError("finite 1D axes and a matching finite heightfield are required")
    dx = np.diff(x)
    dy = np.diff(y)
    if np.any(dx <= 0.0) or np.any(dy <= 0.0):
        raise ValueError("heightfield axes must increase")
    resolution = float(max(np.mean(dx), np.mean(dy)))
    tolerance = max(1.0e-9, resolution * 1.0e-7)
    if not np.allclose(dx, resolution, rtol=0.0, atol=tolerance) or not np.allclose(
        dy, resolution, rtol=0.0, atol=tolerance
    ):
        raise ValueError("fly-ramp audit requires a uniform square heightfield grid")
    return x, y, height, resolution


def _audit_one_ramp(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    ramp: FlyRampGeometry,
    resolution: float,
) -> dict[str, Any]:
    low = np.asarray(ramp.low_edge_center_xyz_m, dtype=np.float64)
    normal = np.asarray(ramp.normal_xyz, dtype=np.float64)
    uphill = np.asarray(ramp.uphill_unit_xy, dtype=np.float64)
    dx = x[None, :] - low[0]
    dy = y[:, None] - low[1]
    along = dx * uphill[0] + dy * uphill[1]
    lateral = -dx * uphill[1] + dy * uphill[0]
    plane = low[2] - (normal[0] * dx + normal[1] * dy) / normal[2]
    interior = (
        (along >= RAMP_INTERIOR_INSET_M)
        & (along <= ramp.horizontal_run_m - RAMP_INTERIOR_INSET_M)
        & (np.abs(lateral) <= ramp.surface_width_m / 2.0 - RAMP_INTERIOR_INSET_M)
    )
    if not np.any(interior):
        raise ValueError(f"heightfield does not cover ramp interior: {ramp.route_id}")
    interior_error = np.abs(height[interior] - plane[interior])
    p95 = float(np.quantile(interior_error, 0.95))

    profile_step = min(resolution / 4.0, 0.0025)
    profile_s = np.arange(
        -0.12,
        ramp.horizontal_run_m + 0.12 + profile_step * 0.5,
        profile_step,
        dtype=np.float64,
    )
    profile_x = low[0] + profile_s * uphill[0]
    profile_y = low[1] + profile_s * uphill[1]
    profile_height = _bilinear(x, y, height, profile_x, profile_y)
    profile_plane = (
        low[2] - (normal[0] * (profile_x - low[0]) + normal[1] * (profile_y - low[1])) / normal[2]
    )
    matching = np.abs(profile_height - profile_plane) <= SEAM_PLANE_MATCH_TOLERANCE_M
    start, stop = _matching_segment_around_midpoint(profile_s, matching, ramp.horizontal_run_m)
    low_error = abs(start - ramp.cad_low_seam_along_m)
    high_error = abs(stop - ramp.cad_high_seam_along_m)

    wheel = _wheel_geometry_probe(profile_s, profile_height, ramp.horizontal_run_m)
    plane_pass = p95 <= RAMP_INTERIOR_P95_ERROR_MAX_M + 1.0e-12
    seam_pass = max(low_error, high_error) <= SEAM_LOCATION_ERROR_MAX_M + 1.0e-12
    return {
        "geometry": ramp.as_dict(),
        "interior": {
            "sample_count": int(np.count_nonzero(interior)),
            "plane_abs_error_m": {
                "mean": float(np.mean(interior_error)),
                "p95": p95,
                "maximum": float(np.max(interior_error)),
            },
        },
        "seams": {
            "cad_reference_low_seam_along_m": ramp.cad_low_seam_along_m,
            "cad_reference_high_seam_along_m": ramp.cad_high_seam_along_m,
            "detected_low_seam_along_m": start,
            "detected_high_seam_along_m": stop,
            "low_edge_location_abs_error_m": low_error,
            "high_edge_location_abs_error_m": high_error,
            "maximum_location_abs_error_m": max(low_error, high_error),
            "method": (
                "compare the heightfield plane-matching centreline segment with the same "
                "hash-bound unsimplified CAD-ray segment"
            ),
        },
        "wheel_geometry_probe": wheel,
        "checks": {
            "interior_plane_p95": plane_pass,
            "low_seam_location": low_error <= SEAM_LOCATION_ERROR_MAX_M + 1.0e-12,
            "high_seam_location": high_error <= SEAM_LOCATION_ERROR_MAX_M + 1.0e-12,
            "wheel_geometry_finite": wheel["finite"],
            "static_pass": plane_pass and seam_pass and wheel["finite"],
        },
    }


def _bilinear(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> np.ndarray:
    if (
        np.any(query_x < x[0])
        or np.any(query_x > x[-1])
        or np.any(query_y < y[0])
        or np.any(query_y > y[-1])
    ):
        raise ValueError("heightfield does not cover fixed fly-ramp audit profile")
    columns = np.clip(np.searchsorted(x, query_x, side="right") - 1, 0, len(x) - 2)
    rows = np.clip(np.searchsorted(y, query_y, side="right") - 1, 0, len(y) - 2)
    tx = (query_x - x[columns]) / (x[columns + 1] - x[columns])
    ty = (query_y - y[rows]) / (y[rows + 1] - y[rows])
    return (
        (1.0 - tx) * (1.0 - ty) * height[rows, columns]
        + tx * (1.0 - ty) * height[rows, columns + 1]
        + (1.0 - tx) * ty * height[rows + 1, columns]
        + tx * ty * height[rows + 1, columns + 1]
    )


def _matching_segment_around_midpoint(
    profile_s: np.ndarray,
    matching: np.ndarray,
    run_m: float,
) -> tuple[float, float]:
    indices = np.flatnonzero(matching)
    if len(indices) == 0:
        return math.inf, -math.inf
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    segments = np.split(indices, breaks)
    midpoint = run_m / 2.0
    segment = min(
        segments,
        key=lambda values: (
            0.0
            if profile_s[values[0]] <= midpoint <= profile_s[values[-1]]
            else min(abs(profile_s[values[0]] - midpoint), abs(profile_s[values[-1]] - midpoint)),
            -len(values),
        ),
    )
    return float(profile_s[segment[0]]), float(profile_s[segment[-1]])


def _wheel_geometry_probe(
    profile_s: np.ndarray,
    profile_height: np.ndarray,
    horizontal_run_m: float,
) -> dict[str, Any]:
    radius = REFERENCE_WHEEL_DIAMETER_M / 2.0
    centers = profile_s[(profile_s >= -0.06) & (profile_s <= horizontal_run_m)]
    center_z = np.empty_like(centers)
    for index, center in enumerate(centers):
        distance = profile_s - center
        contact = np.abs(distance) <= radius
        circle = np.sqrt(np.maximum(radius * radius - distance[contact] ** 2, 0.0))
        center_z[index] = float(np.max(profile_height[contact] + circle))
    finite = bool(np.isfinite(center_z).all()) and len(center_z) >= 2
    return {
        "method": "120mm circular cross-section upper-envelope on bilinear centerline",
        "finite": finite,
        "center_sample_count": int(len(center_z)),
        "minimum_center_height_m": float(np.min(center_z)),
        "maximum_center_height_m": float(np.max(center_z)),
        "maximum_adjacent_center_height_change_m": float(np.max(np.abs(np.diff(center_z)))),
        "claim_boundary": "geometric computability only; contact dynamics are pending",
    }


__all__ = [
    "FINE_SEAM_RESOLUTION_M",
    "FIXED_FLY_RAMPS",
    "FlyRampGeometry",
    "REFERENCE_WHEEL_DIAMETER_M",
    "WHEEL_PROBE_SPEEDS_M_S",
    "WHEEL_PROBE_LOW_SIDE_APPROACH_M",
    "WHEEL_PROBE_ENDPOINT_BASIS",
    "audit_fixed_fly_ramps",
    "wheel_probe_trial_plan",
    "wheel_probe_route_endpoints",
]
