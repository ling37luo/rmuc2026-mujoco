"""World-space approach and landing routes for the two audited fly ramps.

The fixed CAD records locate the ramps.  The runtime heightfield supplies the
actual surface heights and landing edge, so these routes follow the selected
pack without creating another field or changing its collision geometry.
"""

from __future__ import annotations

import math
from typing import Any

from .jump_probe import JumpProbeConfig, measure_runtime_landing_profile
from .manifest import FieldAsset
from .query import height_at, surface_at
from .ramp_audit import FIXED_FLY_RAMPS, FlyRampGeometry


APPROACH_BEFORE_LOW_EDGE_M = 0.60
LANDING_TARGET_INSET_M = 0.20
APPROACH_FOOTPRINT_RADIUS_M = 0.35

_RAMP_BY_SCENARIO = {
    "fly_ramp_north": FIXED_FLY_RAMPS[0],
    "fly_ramp_south": FIXED_FLY_RAMPS[1],
}


def fly_route_descriptor(asset: FieldAsset, scenario_id: str) -> dict[str, Any]:
    """Bind one audited centreline to the selected pack's collision surface.

    ``spawn.xyz_m`` is the terrain point, not a robot root pose.  A robot reset
    must use its own contact geometry to find the supported root height.
    """

    try:
        ramp = _RAMP_BY_SCENARIO[scenario_id]
    except KeyError as exc:
        raise ValueError(f"scenario {scenario_id!r} is not a fly ramp") from exc

    landing = measure_runtime_landing_profile(asset, ramp)
    if abs(landing.measured_gap_m - JumpProbeConfig().official_gap_m) > 0.015:
        raise ValueError(f"{ramp.route_id}: runtime landing gap disagrees with the audited ramp")

    approach_along = ramp.cad_low_seam_along_m - APPROACH_BEFORE_LOW_EDGE_M
    approach_xy = _world_xy(asset, ramp, approach_along)
    approach_surface = surface_at(
        asset,
        *approach_xy,
        window_radius_m=APPROACH_FOOTPRINT_RADIUS_M,
    )
    if (
        approach_surface.maximum_window_slope_deg > 6.0
        or approach_surface.local_relief_upper_bound_m > 0.03
    ):
        raise ValueError(f"{ramp.route_id}: approach footprint is not a clear low-slope surface")

    low_seam = _surface_point(asset, ramp, ramp.cad_low_seam_along_m)
    takeoff = _surface_point(asset, ramp, ramp.cad_high_seam_along_m)
    landing_edge_xy = _world_xy(asset, ramp, landing.landing_edge_along_m)
    landing_target = _surface_point(
        asset, ramp, landing.landing_edge_along_m + LANDING_TARGET_INSET_M
    )
    heading = math.atan2(ramp.uphill_unit_xy[1], ramp.uphill_unit_xy[0])
    approach = [*approach_xy, approach_surface.height_m]
    return {
        "scenario_id": scenario_id,
        "route_id": ramp.route_id,
        "source_manifest_sha256": asset.manifest_sha256,
        "validation_status": asset.manifest["validation_status"],
        "topology_verified": False,
        "source_part_index": ramp.source_part_index,
        "uphill_unit_xy": list(ramp.uphill_unit_xy),
        "heading_yaw_rad": heading,
        "surface_width_m": ramp.surface_width_m,
        "slope_angle_deg": ramp.slope_angle_degrees,
        "approach_along_m": approach_along,
        "low_seam_along_m": ramp.cad_low_seam_along_m,
        "takeoff_along_m": ramp.cad_high_seam_along_m,
        "landing_edge_along_m": landing.landing_edge_along_m,
        "landing_target_along_m": landing.landing_edge_along_m + LANDING_TARGET_INSET_M,
        "approach_xyz_m": approach,
        "low_seam_xyz_m": low_seam,
        "takeoff_xyz_m": takeoff,
        # The edge XY comes from the first raised runtime sample. Its Z is
        # the measured landing *top*; a heightfield edge itself is interpolated.
        "landing_edge_xyz_m": [*landing_edge_xy, landing.landing_top_height_m],
        "landing_target_xyz_m": landing_target,
        "landing_top_height_m": landing.landing_top_height_m,
        "gap_m": landing.measured_gap_m,
        "spawn": {"xyz_m": approach, "heading_yaw_rad": heading},
    }


def _world_xy(asset: FieldAsset, ramp: FlyRampGeometry, along_m: float) -> tuple[float, float]:
    spawn = asset.recommended_spawn
    return (
        ramp.low_edge_center_xyz_m[0]
        - float(spawn["x_before_translation_m"])
        + along_m * ramp.uphill_unit_xy[0],
        ramp.low_edge_center_xyz_m[1]
        - float(spawn["y_before_translation_m"])
        + along_m * ramp.uphill_unit_xy[1],
    )


def _surface_point(asset: FieldAsset, ramp: FlyRampGeometry, along_m: float) -> list[float]:
    x, y = _world_xy(asset, ramp, along_m)
    return [x, y, height_at(asset, x, y, interpolation="mujoco")]


__all__ = ["fly_route_descriptor"]
