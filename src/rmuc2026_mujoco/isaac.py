"""Framework-neutral input for an optional Isaac/Isaac Lab adapter.

This module deliberately imports no Isaac package.  It exposes the verified
heightfield and identity metadata an external Isaac consumer needs to build a
high-parallel environment without changing the field geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import FieldAsset
from .query import field_bounds, load_heightfield
from .scenarios import scenario_descriptor
from .slope_catalog import slope_catalog
from .turning import screen_turn_spawns, turn_phase_registry
from .training_region import TrainingRegion


@dataclass(frozen=True)
class IsaacHeightfieldInput:
    height_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    scenario: dict[str, Any]
    spawn_points: tuple[dict[str, Any], ...] = ()
    source: str = "verified_runtime_pack_heightfield"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.height_m.shape),
            "bounds_xy_m": [list(row) for row in self.bounds_xy_m],
            "scenario": self.scenario,
            "spawn_points": [dict(point) for point in self.spawn_points],
            "source": self.source,
        }


def load_isaac_heightfield(
    asset: FieldAsset | str | Path,
    *,
    scenario: str = "turn_basic",
    profile: str = "collision_only",
) -> IsaacHeightfieldInput:
    """Return data for an Isaac adapter; no Isaac dependency is required."""

    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    descriptor = scenario_descriptor(field, scenario, profile=profile)
    samples = load_heightfield(field)
    spawn_points: tuple[dict[str, Any], ...] = ()
    if scenario == "turn_basic":
        spawn_points = tuple(spawn.to_dict() for spawn in screen_turn_spawns(field))
        descriptor["turn_phase_registry"] = turn_phase_registry()
        descriptor["turn_spawn_points"] = [dict(point) for point in spawn_points]
    elif scenario == "slope_basic":
        descriptor["slope_catalog"] = slope_catalog(field)
        spawn_points = tuple(
            {
                "route_id": patch["patch_id"],
                "direction": direction,
                "position_xyz_m": patch["route"][point],
                "heading_yaw_rad": patch["route"]["heading_yaw_rad"] + offset,
            }
            for patch in descriptor["slope_catalog"]["patches"]
            if patch["route"] is not None
            for direction, point, offset in (
                ("uphill", "low_xyz_m", 0.0),
                ("downhill", "high_xyz_m", float(np.pi)),
            )
        )
    elif scenario in {"fly_ramp_north", "fly_ramp_south"}:
        from .fly_routes import fly_route_descriptor

        route = fly_route_descriptor(field, scenario)
        descriptor["fly_ramp_route"] = route
        spawn_points = (
            {
                "route_id": route["route_id"],
                "position_xyz_m": list(route["spawn"]["xyz_m"]),
                "heading_yaw_rad": float(route["spawn"]["heading_yaw_rad"]),
                "position_reference": "terrain_surface",
                "topology_verified": False,
            },
        )
    return IsaacHeightfieldInput(
        height_m=np.asarray(samples.height_m, dtype=np.float32),
        x_m=np.asarray(samples.x_m, dtype=np.float32),
        y_m=np.asarray(samples.y_m, dtype=np.float32),
        bounds_xy_m=field_bounds(field),
        scenario=descriptor,
        spawn_points=spawn_points,
    )


def load_isaac_training_region(region: TrainingRegion | str | Path) -> IsaacHeightfieldInput:
    """Provide the exact local grid and identity for an external Isaac adapter.

    IsaacLab's Newton XPBD backend is currently the contact-tested consumer of
    this 1 cm grid. This function only transfers data; it does not choose a
    physics backend or claim PhysX/MJWarp contact equivalence.
    """

    selected = region if isinstance(region, TrainingRegion) else TrainingRegion.open(region)
    samples = selected.heightfield()
    manifest = selected.manifest
    route = dict(manifest["route"])
    spawn = {
        "route_id": route["route_id"],
        "position_xyz_m": list(route["spawn"]["xyz_m"]),
        "heading_yaw_rad": float(route["spawn"]["heading_yaw_rad"]),
        "position_reference": "terrain_surface",
        "topology_verified": False,
    }
    descriptor = {
        "scenario_id": manifest["scenario_id"],
        "profile": manifest["profile"],
        "profile_hash": manifest["profile_hash"],
        "region_manifest_sha256": selected.manifest_sha256,
        "source_manifest_sha256": manifest["source"]["source_manifest_sha256"],
        "source_profile_hash": manifest["source"]["source_profile_hash"],
        "source_collision_samples_sha256": manifest["source"]["source_collision_samples_sha256"],
        "local_samples_sha256": manifest["files"]["collision/heightfield.npz"]["sha256"],
        "source_grid_slice_yx": manifest["source"]["source_grid_slice_yx"],
        "route": route,
        "validation_status": manifest["validation_status"],
    }
    return IsaacHeightfieldInput(
        height_m=samples.height_m.astype(np.float32),
        x_m=samples.x_m.astype(np.float32),
        y_m=samples.y_m.astype(np.float32),
        bounds_xy_m=samples.bounds_xy_m,
        scenario=descriptor,
        spawn_points=(spawn,),
        source="verified_cropped_training_region",
    )


__all__ = ["IsaacHeightfieldInput", "load_isaac_heightfield", "load_isaac_training_region"]
