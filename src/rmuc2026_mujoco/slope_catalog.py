"""Deterministic ordinary-slope training descriptors.

The catalog is derived from the verified collision heightfield at load time.  It
does not create another map or add collision geometry.  It deliberately stops
below the two audited fly ramps; those are exposed by the separate
``fly_ramp_north`` and ``fly_ramp_south`` scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from .manifest import FieldAsset
from .query import HEIGHTFIELD_CLAIM_BOUNDARY, HeightFieldData, load_heightfield
from .ramp_audit import FIXED_FLY_RAMPS
from .slope_routes import ROUTE_SCREEN, screen_slope_route


@dataclass(frozen=True)
class SlopeBand:
    """An ordinary, non-fly-ramp slope interval in degrees."""

    band_id: str
    minimum_deg: float
    maximum_deg: float
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "minimum_deg": self.minimum_deg,
            "maximum_deg": self.maximum_deg,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class SlopePatch:
    """A screened traversable-length patch on the single-valued heightfield."""

    patch_id: str
    band_id: str
    center_xyz_m: tuple[float, float, float]
    uphill_unit_xy: tuple[float, float]
    slope_angle_deg: float
    horizontal_run_m: float
    usable_width_m: float
    height_gain_m: float
    roughness_rms_m: float
    fit_residual_p95_m: float
    topology_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "band_id": self.band_id,
            "center_xyz_m": list(self.center_xyz_m),
            "uphill_unit_xy": list(self.uphill_unit_xy),
            "slope_angle_deg": self.slope_angle_deg,
            "horizontal_run_m": self.horizontal_run_m,
            "usable_width_m": self.usable_width_m,
            "height_gain_m": self.height_gain_m,
            "roughness_rms_m": self.roughness_rms_m,
            "fit_residual_p95_m": self.fit_residual_p95_m,
            "topology_verified": self.topology_verified,
            "claim_boundary": HEIGHTFIELD_CLAIM_BOUNDARY,
        }


ORDINARY_SLOPE_BANDS: tuple[SlopeBand, ...] = (
    SlopeBand("slope_3_6", 3.0, 6.0, "low-speed approach and basic climbing"),
    SlopeBand("slope_6_10", 6.0, 10.0, "ordinary uphill traversal"),
    SlopeBand("slope_10_15", 10.0, 15.0, "steeper traversal before the fly-ramp boundary"),
)

FLY_RAMP_SCENARIOS = ("fly_ramp_north", "fly_ramp_south")


def catalog_slope_patches(
    data: HeightFieldData,
    *,
    bands: Iterable[SlopeBand] = ORDINARY_SLOPE_BANDS,
    max_per_band: int = 4,
    sampling_m: float = 0.10,
    horizontal_run_m: float = 0.80,
    usable_width_m: float = 0.60,
    minimum_separation_m: float = 2.0,
) -> tuple[SlopePatch, ...]:
    """Find deterministic, plane-like ordinary slope patches.

    A raw heightfield gradient is only a candidate signal: vertical walls and
    mesh seams also have a large gradient.  Every returned patch therefore
    receives a local plane fit over a short route and a lateral width.  The
    result remains a runtime proxy and never claims CAD topology or robot
    clearance.
    """

    if isinstance(max_per_band, bool) or not isinstance(max_per_band, int) or max_per_band <= 0:
        raise ValueError("max_per_band must be a positive integer")
    if sampling_m <= 0.0 or horizontal_run_m <= 0.0 or usable_width_m <= 0.0:
        raise ValueError("sampling_m, horizontal_run_m and usable_width_m must be positive")
    if minimum_separation_m < 0.0:
        raise ValueError("minimum_separation_m must be non-negative")
    if data.height_m.ndim != 2 or data.height_m.shape != (data.y_m.size, data.x_m.size):
        raise ValueError("heightfield shape does not match its axes")
    if data.x_m.size < 2 or data.y_m.size < 2:
        raise ValueError("heightfield axes need at least two samples")
    if (
        not np.isfinite(data.x_m).all()
        or not np.isfinite(data.y_m).all()
        or not np.isfinite(data.height_m).all()
        or not np.all(np.diff(data.x_m) > 0.0)
        or not np.all(np.diff(data.y_m) > 0.0)
    ):
        raise ValueError("heightfield axes and samples must be finite and increasing")

    dx = float(np.median(np.diff(data.x_m)))
    dy = float(np.median(np.diff(data.y_m)))
    stride_x = max(1, int(round(sampling_m / dx)))
    stride_y = max(1, int(round(sampling_m / dy)))
    dz_dy, dz_dx = np.gradient(data.height_m, dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    margin_x = 0.5 * horizontal_run_m + 0.10
    margin_y = 0.5 * horizontal_run_m + 0.10
    x_indices = range(0, data.x_m.size, stride_x)
    y_indices = range(0, data.y_m.size, stride_y)
    candidates: list[SlopePatch] = []

    for band in tuple(bands):
        found: list[SlopePatch] = []
        for row in y_indices:
            y_value = float(data.y_m[row])
            if y_value < data.y_m[0] + margin_y or y_value > data.y_m[-1] - margin_y:
                continue
            for column in x_indices:
                x_value = float(data.x_m[column])
                if x_value < data.x_m[0] + margin_x or x_value > data.x_m[-1] - margin_x:
                    continue
                local_slope = float(slope_deg[row, column])
                if not (band.minimum_deg <= local_slope < band.maximum_deg):
                    continue
                gradient = np.array([float(dz_dx[row, column]), float(dz_dy[row, column])])
                gradient_norm = float(np.linalg.norm(gradient))
                if gradient_norm < 1.0e-9:
                    continue
                uphill = gradient / gradient_norm
                lateral = np.array([-uphill[1], uphill[0]])
                # A 5x5 grid checks both the route direction and its usable width.
                route_offsets = np.linspace(-horizontal_run_m / 2.0, horizontal_run_m / 2.0, 5)
                lateral_offsets = np.linspace(-usable_width_m / 2.0, usable_width_m / 2.0, 5)
                points = np.array(
                    [
                        [
                            x_value + float(uphill[0] * along + lateral[0] * across),
                            y_value + float(uphill[1] * along + lateral[1] * across),
                        ]
                        for along in route_offsets
                        for across in lateral_offsets
                    ],
                    dtype=float,
                )
                if not bool(_inside(data, points[:, 0], points[:, 1], margin=0.0).all()):
                    continue
                heights = _bilinear_points(data, points[:, 0], points[:, 1])
                design = np.column_stack(
                    (points[:, 0] - x_value, points[:, 1] - y_value, np.ones(points.shape[0]))
                )
                coefficients, *_ = np.linalg.lstsq(design, heights, rcond=None)
                residual = heights - design @ coefficients
                fit_slope = math.degrees(
                    math.atan(math.hypot(float(coefficients[0]), float(coefficients[1])))
                )
                if not (band.minimum_deg <= fit_slope < band.maximum_deg):
                    continue
                residual_p95 = float(np.percentile(np.abs(residual), 95.0))
                roughness = float(np.sqrt(np.mean(residual * residual)))
                if residual_p95 > 0.02 or roughness > 0.01:
                    continue
                fitted_gradient = np.array([float(coefficients[0]), float(coefficients[1])])
                fitted_norm = float(np.linalg.norm(fitted_gradient))
                if fitted_norm < 1.0e-9:
                    continue
                fitted_uphill = fitted_gradient / fitted_norm
                height_gain = fitted_norm * horizontal_run_m
                if height_gain < 0.03:
                    continue
                center = (
                    x_value,
                    y_value,
                    float(_bilinear_points(data, np.array([x_value]), np.array([y_value]))[0]),
                )
                patch = SlopePatch(
                    patch_id=f"{band.band_id}_{len(found):02d}",
                    band_id=band.band_id,
                    center_xyz_m=center,
                    uphill_unit_xy=(float(fitted_uphill[0]), float(fitted_uphill[1])),
                    slope_angle_deg=fit_slope,
                    horizontal_run_m=horizontal_run_m,
                    usable_width_m=usable_width_m,
                    height_gain_m=height_gain,
                    roughness_rms_m=roughness,
                    fit_residual_p95_m=residual_p95,
                )
                if all(_distance_xy(patch, prior) >= minimum_separation_m for prior in found):
                    found.append(patch)
                    if len(found) >= max_per_band:
                        break
            if len(found) >= max_per_band:
                break
        candidates.extend(found)
    return tuple(candidates)


def slope_catalog(
    asset: FieldAsset | str,
    *,
    max_per_band: int = 4,
    sampling_m: float = 0.10,
) -> dict[str, Any]:
    """Bind the ordinary-slope catalog to one verified collision-only pack."""

    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    if "collision_only" not in field.available_runtime_profiles:
        raise ValueError("ordinary slope catalog requires the collision_only runtime profile")
    data = load_heightfield(field)
    patches = catalog_slope_patches(
        data,
        max_per_band=max_per_band,
        sampling_m=sampling_m,
    )
    from .scenarios import scenario_descriptor

    descriptor = scenario_descriptor(field, "slope_basic", profile="collision_only")
    band_counts = {
        band.band_id: sum(patch.band_id == band.band_id for patch in patches)
        for band in ORDINARY_SLOPE_BANDS
    }
    records = []
    for patch in patches:
        record = patch.to_dict()
        try:
            route = screen_slope_route(data, record)
            xy = np.asarray(route["waypoints_xyz_m"])[:, :2]
            spawn = field.recommended_spawn
            offset = np.array([spawn["x_before_translation_m"], spawn["y_before_translation_m"]])
            for ramp in FIXED_FLY_RAMPS:
                relative = xy + offset - np.asarray(ramp.low_edge_center_xyz_m[:2])
                direction = np.asarray(ramp.uphill_unit_xy)
                along = relative @ direction
                across = relative @ np.array([-direction[1], direction[0]])
                if np.any(
                    (along >= -2.0)
                    & (along <= ramp.cad_high_seam_along_m + 2.0)
                    & (np.abs(across) <= (ramp.surface_width_m + route["width_m"]) / 2)
                ):
                    raise ValueError("dedicated_fly_ramp_corridor")
            record["route"] = route
            record["route_rejection"] = None
        except ValueError as exc:
            record["route"] = None
            record["route_rejection"] = str(exc)
        records.append(record)
    return {
        "schema_version": 2,
        "catalog_id": "slope_basic",
        "scenario_id": "slope_basic",
        "kind": "ordinary_slope_training_catalog",
        "status": (
            "READY_HEIGHTFIELD_SCREENED"
            if all(band_counts.values())
            else "PARTIAL_NO_PATCH_IN_ONE_OR_MORE_BANDS"
        ),
        "source_manifest_sha256": field.manifest_sha256,
        "profile": "collision_only",
        "profile_hash": descriptor["profile_hash"],
        "heightfield_samples_sha256": str(field.collision["samples_sha256"]),
        "collision_image_sha256": str(field.collision["image_sha256"]),
        "bands": [band.to_dict() for band in ORDINARY_SLOPE_BANDS],
        "patches": records,
        "band_counts": band_counts,
        "route_screen": dict(ROUTE_SCREEN),
        "runnable_route_count": sum(record["route"] is not None for record in records),
        "fly_ramps_separate": {
            "scenario_ids": list(FLY_RAMP_SCENARIOS),
            "reason": "fly ramps require dedicated takeoff, flight and landing evaluation",
        },
        "topology_verified": False,
        "claim_boundary": HEIGHTFIELD_CLAIM_BOUNDARY,
    }


def _inside(data: HeightFieldData, x: np.ndarray, y: np.ndarray, *, margin: float) -> np.ndarray:
    return (
        (x >= data.x_m[0] + margin)
        & (x <= data.x_m[-1] - margin)
        & (y >= data.y_m[0] + margin)
        & (y <= data.y_m[-1] - margin)
    )


def _bilinear_points(data: HeightFieldData, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    columns = np.clip(np.searchsorted(data.x_m, x, side="right") - 1, 0, data.x_m.size - 2)
    rows = np.clip(np.searchsorted(data.y_m, y, side="right") - 1, 0, data.y_m.size - 2)
    x0 = data.x_m[columns]
    x1 = data.x_m[columns + 1]
    y0 = data.y_m[rows]
    y1 = data.y_m[rows + 1]
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)
    z00 = data.height_m[rows, columns]
    z01 = data.height_m[rows, columns + 1]
    z10 = data.height_m[rows + 1, columns]
    z11 = data.height_m[rows + 1, columns + 1]
    return (1.0 - ty) * ((1.0 - tx) * z00 + tx * z01) + ty * ((1.0 - tx) * z10 + tx * z11)


def _distance_xy(first: SlopePatch, second: SlopePatch) -> float:
    dx = first.center_xyz_m[0] - second.center_xyz_m[0]
    dy = first.center_xyz_m[1] - second.center_xyz_m[1]
    return math.hypot(dx, dy)


__all__ = [
    "FLY_RAMP_SCENARIOS",
    "ORDINARY_SLOPE_BANDS",
    "SlopeBand",
    "SlopePatch",
    "catalog_slope_patches",
    "slope_catalog",
]
