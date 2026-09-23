"""Small, robot-agnostic RMUC field scenario registry.

The registry describes *where* a controller should run.  It does not contain
robot actions or a second copy of field geometry.  Coordinates that come from
the audited ramp source are kept in :mod:`ramp_audit`; the other scenarios use
selectors so they remain valid across runtime packs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .manifest import FieldAsset
from .ramp_audit import FIXED_FLY_RAMPS
from .slope_routes import ROUTE_SCREEN
from .slope_progress import SLOPE_TRAVERSAL_RULE


@dataclass(frozen=True)
class ScenarioSpec:
    """A field-only scenario description shared by viewers and trainers."""

    scenario_id: str
    profile: str
    purpose: str
    region: Mapping[str, Any]
    spawn: Mapping[str, Any]
    routes: tuple[Mapping[str, Any], ...]
    consumers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "profile": self.profile,
            "purpose": self.purpose,
            "region": dict(self.region),
            "spawn": dict(self.spawn),
            "routes": [dict(route) for route in self.routes],
            "consumers": list(self.consumers),
        }


def _ramp_route(route: Any) -> dict[str, Any]:
    return {
        "route_id": route.route_id,
        "source_part_index": route.source_part_index,
        "low_edge_center_xyz_m": list(route.low_edge_center_xyz_m),
        "high_edge_center_xyz_m": list(route.high_edge_center_xyz_m),
        "uphill_unit_xy": list(route.uphill_unit_xy),
        "horizontal_run_m": route.horizontal_run_m,
        "surface_width_m": route.surface_width_m,
        "slope_angle_degrees": route.slope_angle_degrees,
        "cad_low_seam_along_m": route.cad_low_seam_along_m,
        "cad_high_seam_along_m": route.cad_high_seam_along_m,
    }


_RAMP_NORTH, _RAMP_SOUTH = tuple(_ramp_route(route) for route in FIXED_FLY_RAMPS)

SCENARIOS: Mapping[str, ScenarioSpec] = {
    "full_eval": ScenarioSpec(
        "full_eval", "full", "complete field interaction baseline",
        {"selector": "field_global", "visual": "full"},
        {"strategy": "pack_recommended_spawn_or_external_reset"},
        (), ("human_view", "viewer", "mujoco", "isaac"),
    ),
    "turn_basic": ScenarioSpec(
        "turn_basic", "collision_only", "flat area for forward, reverse and in-place turning",
        {"selector": "flat_open_area", "visual": "omitted"},
        {
            "strategy": "screened_flat_spawn",
            "selection": "deterministic_heightfield_screen",
            "footprint_radius_m": 0.35,
            "boundary_margin_m": 0.50,
            "minimum_separation_m": 2.0,
            "max_slope_deg": 3.0,
            "max_relief_m": 0.02,
            "max_count": 8,
            "ground_height_range_m": [-0.05, 0.05],
            "reuse_across_parallel_envs": True,
            "phases": {
                "spin": {
                    "duration_train_s": 8.0,
                    "duration_eval_s": 20.0,
                    "vx_mps": [0.0],
                    "yaw_rad_s": [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6],
                    "height_m": [0.18, 0.24],
                    "initial_headings": 8,
                    "workspace_radius_m": 0.75,
                },
                "arc": {
                    "duration_train_s": 8.0,
                    "duration_eval_s": 20.0,
                    "vx_mps": [-0.3, 0.3],
                    "yaw_rad_s": [-0.6, -0.3, 0.3, 0.6],
                    "height_m": [0.18, 0.24],
                    "workspace_radius_m": 1.5,
                },
                "reversal": {
                    "duration_train_s": 8.0,
                    "duration_eval_s": 20.0,
                    "vx_mps": [0.0, -0.3, 0.3],
                    "yaw_sequence_rad_s": [
                        [-0.6, 0.0, 0.6],
                        [0.6, 0.0, -0.6],
                    ],
                    "segment_duration_s": 2.0,
                    "height_m": [0.18, 0.24],
                    "workspace_radius_m": 1.5,
                },
            },
        },
        (), ("human_view", "mujoco", "isaac"),
    ),
    "stairs_basic": ScenarioSpec(
        "stairs_basic", "collision_only", "stairs and their approach/exit connection",
        {"selector": "stairs_and_connection", "visual": "omitted"},
        {"strategy": "external_route_spawn", "approach_directions": ["forward", "reverse", "diagonal"]},
        (), ("human_view", "mujoco", "isaac"),
    ),
    "slope_basic": ScenarioSpec(
        "slope_basic", "collision_only", "ordinary traversable slopes below the dedicated fly ramps",
        {
            "selector": "ordinary_slope_catalog",
            "visual": "omitted",
            "catalog_id": "slope_basic",
            "slope_bands_deg": [[3.0, 6.0], [6.0, 10.0], [10.0, 15.0]],
            "route_length_m": 0.80,
            "usable_width_m": 0.60,
            "fly_ramps_excluded": ["fly_ramp_north", "fly_ramp_south"],
            "route_screen": dict(ROUTE_SCREEN),
        },
        {
            "strategy": "catalog_patch_spawn",
            "controller_owns_gear_and_action_mapping": True,
            "goal": "independent_uphill_downhill_or_roundtrip_without_flight_evaluation",
            "default_direction": "uphill",
            "directions": ["uphill", "downhill", "roundtrip"],
            "traversal_rule": dict(SLOPE_TRAVERSAL_RULE),
            "topology_verified": False,
        },
        (), ("human_view", "mujoco", "isaac"),
    ),
    "fly_ramp_north": ScenarioSpec(
        "fly_ramp_north", "collision_only", "audited north fly ramp",
        {"selector": "route", "route_id": _RAMP_NORTH["route_id"], "visual": "omitted"},
        {"strategy": "route_endpoints", "wheel_diameter_m": 0.12},
        (_RAMP_NORTH,), ("human_view", "mujoco", "isaac"),
    ),
    "fly_ramp_south": ScenarioSpec(
        "fly_ramp_south", "collision_only", "audited south fly ramp",
        {"selector": "route", "route_id": _RAMP_SOUTH["route_id"], "visual": "omitted"},
        {"strategy": "route_endpoints", "wheel_diameter_m": 0.12},
        (_RAMP_SOUTH,), ("human_view", "mujoco", "isaac"),
    ),
    "boundary_contact": ScenarioSpec(
        "boundary_contact", "collision_only", "field edge and perimeter contact",
        {"selector": "perimeter_and_boundary", "visual": "omitted"},
        {"strategy": "external_boundary_spawn", "boundary_margin_m": 0.25},
        (), ("human_view", "mujoco", "isaac"),
    ),
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    try:
        return SCENARIOS[str(scenario_id)]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        raise ValueError(f"unknown scenario {scenario_id!r}; choose one of: {choices}") from exc


def list_scenarios() -> list[dict[str, Any]]:
    return [SCENARIOS[name].to_dict() for name in SCENARIOS]


def scenario_descriptor(
    asset: FieldAsset,
    scenario_id: str,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Bind a registry entry to one verified pack without changing its files."""

    spec = get_scenario(scenario_id)
    selected_profile = profile or spec.profile
    if selected_profile not in asset.available_runtime_profiles:
        raise ValueError(f"scenario {scenario_id!r} requires unavailable profile {selected_profile!r}")
    payload = {
        "scenario": spec.to_dict(),
        "scenario_id": spec.scenario_id,
        "source_manifest_sha256": asset.manifest_sha256,
        "profile": selected_profile,
        "profile_entrypoint": str(asset.entrypoint_for(selected_profile).relative_to(asset.root)),
        "heightfield_samples_sha256": str(asset.collision["samples_sha256"]),
        "collision_image_sha256": str(asset.collision["image_sha256"]),
        "collision_kind": asset.collision["kind"],
        "field_bounds_xy_m": _bounds(asset),
        "profile_contract": {
            "version": 1,
            "name": selected_profile,
            "source_manifest_sha256": asset.manifest_sha256,
            "kept_collision": [dict(spec.region)],
            "simplified_or_removed": (
                ["visual_meshes", "rulebook_livery", "unrelated_dynamic_mechanisms"]
                if selected_profile == "collision_only"
                else []
            ),
            "spawn": dict(spec.spawn),
            "routes": [dict(route) for route in spec.routes],
            "heightfield_samples_sha256": str(asset.collision["samples_sha256"]),
            "collision_image_sha256": str(asset.collision["image_sha256"]),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["profile_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _bounds(asset: FieldAsset) -> list[list[float]]:
    samples = asset.file(str(asset.collision["samples_file"]), label="heightfield samples")
    # Keep this helper dependency-free; query.field_bounds already performs
    # the same transform and is used by callers that need exact bounds.
    import numpy as np

    with np.load(samples) as data:
        x = np.asarray(data["x_m"], dtype=float)
        y = np.asarray(data["y_m"], dtype=float)
    spawn = asset.recommended_spawn
    return [
        [float(x[0] - spawn["x_before_translation_m"]), float(y[0] - spawn["y_before_translation_m"])],
        [float(x[-1] - spawn["x_before_translation_m"]), float(y[-1] - spawn["y_before_translation_m"])],
    ]


__all__ = ["SCENARIOS", "ScenarioSpec", "get_scenario", "list_scenarios", "scenario_descriptor"]
