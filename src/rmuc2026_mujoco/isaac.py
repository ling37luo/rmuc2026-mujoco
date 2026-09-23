"""Framework-neutral input for an optional Isaac/Isaac Lab adapter.

This module deliberately imports no Isaac package.  It exposes the verified
heightfield and identity metadata an external Isaac consumer needs to build a
high-parallel environment without changing the field geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .errors import AssetIntegrityError, ManifestError
from .manifest import FieldAsset
from .manifest import sha256_file
from .query import field_bounds, load_heightfield
from .scenarios import scenario_descriptor
from .slope_catalog import slope_catalog
from .turning import screen_turn_spawns, turn_phase_registry
from .training_region import TrainingRegion


_OFFLINE_REGION_TYPE = "rmuc2026_isaac_heightfield_input"
_OFFLINE_REGION_SCHEMA = 1
_OFFLINE_GRID_FILE = "heightfield.npz"


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


def export_isaac_training_region(region: TrainingRegion | str | Path, output: str | Path) -> dict:
    """Export exact region samples and world coordinates for an offline consumer.

    Unlike the in-memory Isaac adapter's float32 arrays, this copies the
    source region's exact float64 NPZ. The files are backend-neutral and do
    not create or validate a PhysX collider.
    """

    selected = TrainingRegion.open(region.root if isinstance(region, TrainingRegion) else region)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"output already exists: {target}")
    source = selected.manifest["source"]
    target.mkdir(parents=True)
    shutil.copyfile(selected.root / "collision/heightfield.npz", target / _OFFLINE_GRID_FILE)
    descriptor = {
        "artifact_type": _OFFLINE_REGION_TYPE,
        "schema_version": _OFFLINE_REGION_SCHEMA,
        "scope": "offline_heightfield_data_only_no_physx_contact_validation",
        "scenario_id": selected.manifest["scenario_id"],
        "validation_status": selected.manifest["validation_status"],
        "grid_file": _OFFLINE_GRID_FILE,
        "grid_file_sha256": sha256_file(target / _OFFLINE_GRID_FILE),
        "grid": {
            **selected.manifest["grid"],
            "height_array_order": "height_m[y_index, x_index]",
            "world_frame": "xyz_m_z_up",
            "height_values": "absolute_world_z_m_no_extra_scale_or_offset",
            "sample_dtype": "float64",
        },
        "identity": {
            "region_manifest_sha256": selected.manifest_sha256,
            "region_profile_hash": selected.manifest["profile_hash"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_profile_hash": source["source_profile_hash"],
            "source_collision_samples_sha256": source["source_collision_samples_sha256"],
            "source_grid_slice_yx": source["source_grid_slice_yx"],
        },
        "route": selected.manifest["route"],
    }
    (target / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return descriptor


def load_isaac_training_region_export(
    output: str | Path, *, source_region: TrainingRegion | str | Path | None = None
) -> IsaacHeightfieldInput:
    """Check an offline export before constructing an external terrain collider.

    Pass ``source_region`` to verify the data and identity against the source;
    without it, provenance in the descriptor is self-reported.
    """

    root = Path(output).expanduser().resolve()
    descriptor_path = root / "descriptor.json"
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Isaac region descriptor is invalid: {exc}") from exc
    if (
        descriptor.get("artifact_type") != _OFFLINE_REGION_TYPE
        or descriptor.get("schema_version") != _OFFLINE_REGION_SCHEMA
        or descriptor.get("grid_file") != _OFFLINE_GRID_FILE
    ):
        raise ManifestError("unsupported Isaac region descriptor")
    grid_path = root / _OFFLINE_GRID_FILE
    if grid_path.is_symlink() or not grid_path.is_file():
        raise AssetIntegrityError("Isaac region grid file is missing")
    if sha256_file(grid_path) != descriptor.get("grid_file_sha256"):
        raise AssetIntegrityError("Isaac region grid hash mismatch")
    with np.load(grid_path, allow_pickle=False) as payload:
        if set(payload.files) != {"x_m", "y_m", "height_m"}:
            raise ManifestError("Isaac region grid arrays are incomplete")
        x = np.asarray(payload["x_m"])
        y = np.asarray(payload["y_m"])
        height = np.asarray(payload["height_m"])
    record = descriptor["grid"]
    bounds = np.asarray(record["bounds_xy_m"], dtype=np.float64)
    resolution = np.asarray(record["resolution_xy_m"], dtype=np.float64)
    if (
        x.dtype != np.float64
        or y.dtype != np.float64
        or height.dtype != np.float64
        or height.shape != (record["rows_y"], record["columns_x"])
        or x.shape != (height.shape[1],)
        or y.shape != (height.shape[0],)
        or bounds.shape != (2, 2)
        or resolution.shape != (2,)
        or not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(height).all())
        or not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0))
        or not np.array_equal(bounds, [[x[0], y[0]], [x[-1], y[-1]]])
        or not np.allclose(np.diff(x), resolution[0], atol=1e-8, rtol=0)
        or not np.allclose(np.diff(y), resolution[1], atol=1e-8, rtol=0)
        or not np.allclose(
            [height.min(), height.max()],
            [record["minimum_height_m"], record["maximum_height_m"]],
            atol=1e-10,
            rtol=0,
        )
        or record.get("height_array_order") != "height_m[y_index, x_index]"
        or record.get("world_frame") != "xyz_m_z_up"
        or record.get("height_values") != "absolute_world_z_m_no_extra_scale_or_offset"
        or record.get("sample_dtype") != "float64"
        or record.get("orientation") != "world_y_ascending_world_x_ascending"
        or record.get("interpolation") != "mujoco_hfield_triangle"
    ):
        raise ManifestError("Isaac region grid geometry disagrees with descriptor")
    if source_region is not None:
        selected = TrainingRegion.open(
            source_region.root if isinstance(source_region, TrainingRegion) else source_region
        )
        source = selected.manifest["source"]
        expected_identity = {
            "region_manifest_sha256": selected.manifest_sha256,
            "region_profile_hash": selected.manifest["profile_hash"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_profile_hash": source["source_profile_hash"],
            "source_collision_samples_sha256": source["source_collision_samples_sha256"],
            "source_grid_slice_yx": source["source_grid_slice_yx"],
        }
        original = selected.heightfield()
        if (
            descriptor.get("identity") != expected_identity
            or not np.array_equal(x, original.x_m)
            or not np.array_equal(y, original.y_m)
            or not np.array_equal(height, original.height_m)
        ):
            raise ManifestError("Isaac region export differs from its verified source region")
    return IsaacHeightfieldInput(
        height_m=height,
        x_m=x,
        y_m=y,
        bounds_xy_m=((float(x[0]), float(y[0])), (float(x[-1]), float(y[-1]))),
        scenario={
            "scenario_id": descriptor["scenario_id"],
            "profile_hash": descriptor["identity"]["region_profile_hash"],
            "source_manifest_sha256": descriptor["identity"]["source_manifest_sha256"],
            "source_profile_hash": descriptor["identity"]["source_profile_hash"],
            "region_manifest_sha256": descriptor["identity"]["region_manifest_sha256"],
            "route": descriptor["route"],
            "validation_status": descriptor["validation_status"],
        },
        source="verified_offline_training_region",
    )


__all__ = [
    "IsaacHeightfieldInput",
    "export_isaac_training_region",
    "load_isaac_heightfield",
    "load_isaac_training_region",
    "load_isaac_training_region_export",
]
