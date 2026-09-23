"""Continuous ordinary-slope routes on the unchanged runtime heightfield."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .query import HeightFieldData
from .wheel_probe import _sample_mujoco_heightfield


ROUTE_SCREEN = {
    "version": 1,
    "search_half_length_m": 2.0,
    "sample_spacing_m": 0.01,
    "endpoint_footprint_length_m": 0.70,
    "maximum_endpoint_relief_m": 0.025,
    "maximum_cross_route_relief_m": 0.03,
    "maximum_adjacent_height_change_m": 0.015,
    "maximum_reverse_height_change_m": 0.025,
}


def screen_slope_route(data: HeightFieldData, patch: dict[str, Any]) -> dict[str, Any]:
    """Require low/high footprints and a connected, full-width path between.

    Sampling includes the complete endpoint footprints. A plane fitted through
    a step, a cliff beyond the crest, or a lateral obstacle must not become a
    runnable route just because its average angle is small.
    """

    center = np.asarray(patch["center_xyz_m"][:2], dtype=float)
    uphill = np.asarray(patch["uphill_unit_xy"], dtype=float)
    uphill /= np.linalg.norm(uphill)
    lateral = np.array([-uphill[1], uphill[0]])
    width = float(patch["usable_width_m"])
    step = ROUTE_SCREEN["sample_spacing_m"]
    along = np.linspace(-2.0, 2.0, 401)
    across = np.linspace(-width / 2, width / 2, math.ceil(width / step) + 1)
    points = center + along[:, None, None] * uphill + across[None, :, None] * lateral
    inside = np.all(
        (points >= np.asarray(data.bounds_xy_m[0])) & (points <= np.asarray(data.bounds_xy_m[1])),
        axis=(1, 2),
    )
    heights = np.full(points.shape[:2], np.nan)
    heights[inside] = _sample_mujoco_heightfield(data, points[inside, :, 0], points[inside, :, 1])
    radius = round(ROUTE_SCREEN["endpoint_footprint_length_m"] / (2 * step))
    flat = [
        i
        for i in range(radius, len(along) - radius)
        if np.ptp(heights[i - radius : i + radius + 1]) <= ROUTE_SCREEN["maximum_endpoint_relief_m"]
    ]
    low = [i for i in flat if along[i] < -patch["horizontal_run_m"] / 2]
    high = [i for i in flat if along[i] > patch["horizontal_run_m"] / 2]
    if not low or not high:
        raise ValueError("missing_low_or_high_clear_footprint")
    start, finish = low[-1], high[0]
    corridor = heights[start - radius : finish + radius + 1]
    if not np.isfinite(corridor).all():
        raise ValueError("route_outside_heightfield")
    if np.ptp(corridor, axis=1).max() > ROUTE_SCREEN["maximum_cross_route_relief_m"]:
        raise ValueError("lateral_obstacle_or_cross_slope")
    max_step = float(np.abs(np.diff(corridor, axis=0)).max())
    if max_step > ROUTE_SCREEN["maximum_adjacent_height_change_m"]:
        raise ValueError("step_or_discontinuous_seam")
    centerline = np.median(corridor, axis=1)
    reverse_drop = float((np.maximum.accumulate(centerline) - centerline).max())
    if reverse_drop > ROUTE_SCREEN["maximum_reverse_height_change_m"]:
        raise ValueError("drop_or_gap_after_slope")
    low_z, high_z = (float(np.median(heights[i])) for i in (start, finish))
    if high_z - low_z < 0.03:
        raise ValueError("insufficient_route_height_gain")

    def xyz(i, z):
        return [*map(float, center + along[i] * uphill), z]

    return {
        "route_id": patch["patch_id"],
        "status": "ROUTE_HEIGHTFIELD_SCREENED",
        "low_xyz_m": xyz(start, low_z),
        "high_xyz_m": xyz(finish, high_z),
        "uphill_unit_xy": uphill.tolist(),
        "heading_yaw_rad": math.atan2(uphill[1], uphill[0]),
        "length_m": float(along[finish] - along[start]),
        "width_m": width,
        "height_gain_m": high_z - low_z,
        "maximum_adjacent_height_change_m": max_step,
        "maximum_cross_route_relief_m": float(np.ptp(corridor, axis=1).max()),
        "waypoints_xyz_m": [
            xyz(i, float(heights[i, len(across) // 2])) for i in range(start, finish + 1, 5)
        ]
        + [xyz(finish, high_z)],
        "dynamic_status": "NOT_RUN",
    }


def select_slope_route(catalog: dict[str, Any], patch_id: str | None = None) -> dict[str, Any]:
    """Select a runnable route, never silently fall back from a named patch."""

    candidates = catalog["patches"]
    if patch_id is not None:
        candidates = [p for p in candidates if p["patch_id"] == patch_id]
        if not candidates:
            raise ValueError(f"unknown slope patch {patch_id!r}")
    for patch in candidates:
        if patch.get("route") is not None:
            return patch["route"]
    reasons = "; ".join(f"{p['patch_id']}: {p.get('route_rejection')}" for p in candidates)
    raise ValueError(f"no screened slope route: {reasons}")
