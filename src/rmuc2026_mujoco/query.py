"""Read-only queries against the runtime pack's exact float heightfield.

The helpers in this module deliberately screen only the single-valued collision
surface stored in a runtime pack.  They never promote a ``DRAFT_BLOCKED`` pack
to a topology-verified field: a flat point on the roof of an underpass can
still look locally safe in a heightfield.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import numpy as np

from .errors import ManifestError, OutOfBoundsError
from .manifest import FieldAsset


HEIGHTFIELD_CLAIM_BOUNDARY = (
    "screened against the runtime pack's declared single-valued float heightfield only; "
    "CAD ray misses may have been filled with proxy ground and no per-cell hit-validity "
    "mask is available; underpasses, overhangs, stacked surfaces, vertical obstacles, "
    "dynamic facilities, semantic zones, and robot clearance remain unverified"
)
MAX_SPAWN_CANDIDATES = 4096


@dataclass(frozen=True)
class HeightFieldData:
    """World-space axes and height samples used by the MuJoCo loader."""

    x_m: np.ndarray
    y_m: np.ndarray
    height_m: np.ndarray

    @property
    def bounds_xy_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (float(self.x_m[0]), float(self.y_m[0])),
            (float(self.x_m[-1]), float(self.y_m[-1])),
        )


@dataclass(frozen=True)
class SurfaceSample:
    """Local heightfield geometry at one world-space position."""

    x_m: float
    y_m: float
    height_m: float
    normal_xyz: tuple[float, float, float]
    slope_deg: float
    maximum_window_slope_deg: float
    local_relief_upper_bound_m: float
    window_radius_m: float
    grid_resolution_xy_m: tuple[float, float]
    height_interpolation: str = "mujoco_hfield_triangle"
    topology_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "height_m": self.height_m,
            "normal_xyz": list(self.normal_xyz),
            "slope_deg": self.slope_deg,
            "maximum_window_slope_deg": self.maximum_window_slope_deg,
            "local_relief_upper_bound_m": self.local_relief_upper_bound_m,
            "window_radius_m": self.window_radius_m,
            "grid_resolution_xy_m": list(self.grid_resolution_xy_m),
            "height_interpolation": self.height_interpolation,
            "topology_verified": self.topology_verified,
            "claim_boundary": HEIGHTFIELD_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class SpawnCandidate:
    """A terrain-only spawn candidate screened over an enclosing square footprint."""

    x_m: float
    y_m: float
    terrain_height_m: float
    normal_xyz: tuple[float, float, float]
    surface_slope_deg: float
    maximum_footprint_slope_deg: float
    footprint_relief_upper_bound_m: float
    grid_boundary_clearance_m: float
    footprint_radius_m: float
    grid_index_yx: tuple[int, int]
    topology_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "terrain_height_m": self.terrain_height_m,
            "normal_xyz": list(self.normal_xyz),
            "surface_slope_deg": self.surface_slope_deg,
            "maximum_footprint_slope_deg": self.maximum_footprint_slope_deg,
            "footprint_relief_upper_bound_m": self.footprint_relief_upper_bound_m,
            "grid_boundary_clearance_m": self.grid_boundary_clearance_m,
            "footprint_radius_m": self.footprint_radius_m,
            "grid_index_yx": list(self.grid_index_yx),
            "topology_verified": self.topology_verified,
            "claim_boundary": HEIGHTFIELD_CLAIM_BOUNDARY,
        }


def load_heightfield(asset: FieldAsset) -> HeightFieldData:
    """Load the declared NPZ samples in the pack's translated world coordinates."""

    collision = asset.collision
    samples_path = asset.file(str(collision["samples_file"]), label="collision.samples_file")
    try:
        with np.load(samples_path, allow_pickle=False) as samples:
            x = np.asarray(samples["x_m"], dtype=np.float64)
            y = np.asarray(samples["y_m"], dtype=np.float64)
            height = np.asarray(samples["height_m"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ManifestError(f"heightfield NPZ is invalid: {exc}") from exc
    declared_shape = (int(collision["rows_y"]), int(collision["columns_x"]))
    if x.shape != (declared_shape[1],) or y.shape != (declared_shape[0],):
        raise ManifestError("heightfield axes do not match the declared shape")
    if height.shape != declared_shape:
        raise ManifestError("heightfield samples do not match the declared shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(height).all():
        raise ManifestError("heightfield axes and samples must be finite")
    if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
        raise ManifestError("heightfield axes must be strictly increasing")
    maximum = float(collision["maximum_height_m"])
    minimum = float(collision["minimum_height_m"])
    tolerance = max(1.0e-8, maximum * 1.0e-7)
    if float(np.min(height)) < minimum - tolerance or float(np.max(height)) > maximum + tolerance:
        raise ManifestError("heightfield samples lie outside the declared height range")

    spawn = asset.recommended_spawn
    offset_x = float(spawn["x_before_translation_m"])
    offset_y = float(spawn["y_before_translation_m"])
    offset_z = float(spawn["terrain_height_m"])
    data = HeightFieldData(x_m=x - offset_x, y_m=y - offset_y, height_m=height - offset_z)
    _validate_mujoco_grid_alignment(data, collision=collision, terrain_offset_z=offset_z)
    return data


def field_bounds(asset: FieldAsset) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return inclusive world-space XY bounds of the collision grid."""

    return load_heightfield(asset).bounds_xy_m


def height_at(
    asset: FieldAsset,
    x: float,
    y: float,
    *,
    out_of_bounds: Literal["raise", "clip", "nan"] = "raise",
    interpolation: Literal["bilinear", "mujoco"] = "bilinear",
) -> float:
    """Interpolate collision height at one world-space XY position.

    The default preserves the original public bilinear query. Use
    ``interpolation="mujoco"`` to match MuJoCo's two triangular heightfield
    faces per grid cell.
    """

    if out_of_bounds not in {"raise", "clip", "nan"}:
        raise ValueError("out_of_bounds must be 'raise', 'clip', or 'nan'")
    if interpolation not in {"bilinear", "mujoco"}:
        raise ValueError("interpolation must be 'bilinear' or 'mujoco'")
    data = load_heightfield(asset)
    x_value, y_value = _finite_point(x, y)
    outside = (
        x_value < data.x_m[0]
        or x_value > data.x_m[-1]
        or y_value < data.y_m[0]
        or y_value > data.y_m[-1]
    )
    if outside:
        if out_of_bounds == "raise":
            raise OutOfBoundsError(f"query ({x_value}, {y_value}) is outside {data.bounds_xy_m}")
        if out_of_bounds == "nan":
            return float("nan")
        x_value = float(np.clip(x_value, data.x_m[0], data.x_m[-1]))
        y_value = float(np.clip(y_value, data.y_m[0], data.y_m[-1]))

    if interpolation == "mujoco":
        return _mujoco_triangle_sample(data, x_value, y_value)[0]
    return _bilinear_grid_value(data, data.height_m, x_value, y_value)


def surface_at(
    asset: FieldAsset,
    x: float,
    y: float,
    *,
    window_radius_m: float = 0.15,
) -> SurfaceSample:
    """Return height, normal, slope, and local relief around one XY position.

    The height and normal use MuJoCo's triangular heightfield surface. The
    relief is a conservative upper bound over all vertices of cells touched by
    the requested square window; it is not a robot-clearance or topology check.
    """

    radius = _finite_nonnegative(window_radius_m, label="window_radius_m")
    data = load_heightfield(asset)
    x_value, y_value = _finite_point(x, y)
    _require_inside(data, x_value, y_value)
    height, dz_dx, dz_dy = _mujoco_triangle_sample(data, x_value, y_value)
    normal = _normal_from_gradient(dz_dx, dz_dy)
    row_start, row_stop = _sample_window(data.y_m, y_value, radius)
    column_start, column_stop = _sample_window(data.x_m, x_value, radius)
    local = data.height_m[row_start:row_stop, column_start:column_stop]
    maximum_face_slope = _cell_triangle_maximum_slope(
        data,
        rows=slice(row_start, row_stop),
        columns=slice(column_start, column_stop),
    )
    return SurfaceSample(
        x_m=x_value,
        y_m=y_value,
        height_m=height,
        normal_xyz=normal,
        slope_deg=_slope_degrees(dz_dx, dz_dy),
        maximum_window_slope_deg=float(np.max(maximum_face_slope)),
        local_relief_upper_bound_m=float(np.max(local) - np.min(local)),
        window_radius_m=radius,
        grid_resolution_xy_m=(
            float(np.median(np.diff(data.x_m))),
            float(np.median(np.diff(data.y_m))),
        ),
    )


def find_spawn_candidates(
    asset: FieldAsset,
    *,
    count: int = 8,
    footprint_radius_m: float = 0.35,
    boundary_margin_m: float = 0.25,
    minimum_separation_m: float = 1.0,
    max_slope_deg: float = 5.0,
    max_relief_m: float = 0.03,
    ground_height_range_m: tuple[float, float] | None = (-0.05, 0.05),
) -> tuple[SpawnCandidate, ...]:
    """Screen deterministic terrain-only spawn candidates from the heightfield.

    The footprint is approximated by an axis-aligned square enclosing the
    requested radius. Once the runtime-heightfield screening constraints pass,
    ordering favors proximity to the translated origin, then low relief, low
    slope, and the middle of the requested height band. Returned points still
    require robot-specific footprint, overhead-clearance, semantic-zone, ray-hit
    validity, and topology validation.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if count > MAX_SPAWN_CANDIDATES:
        raise ValueError(f"count must be <= {MAX_SPAWN_CANDIDATES}")
    radius = _finite_nonnegative(footprint_radius_m, label="footprint_radius_m")
    margin = _finite_nonnegative(boundary_margin_m, label="boundary_margin_m")
    separation = _finite_nonnegative(minimum_separation_m, label="minimum_separation_m")
    maximum_slope = _finite_nonnegative(max_slope_deg, label="max_slope_deg")
    if maximum_slope > 90.0:
        raise ValueError("max_slope_deg must be <= 90")
    maximum_relief = _finite_nonnegative(max_relief_m, label="max_relief_m")
    height_range = _height_range(ground_height_range_m)

    data = load_heightfield(asset)
    minimum_dx = float(np.min(np.diff(data.x_m)))
    minimum_dy = float(np.min(np.diff(data.y_m)))
    span_x = float(data.x_m[-1] - data.x_m[0])
    span_y = float(data.y_m[-1] - data.y_m[0])
    # Reject physically impossible footprints before radius / spacing can
    # overflow for an otherwise finite but extreme caller-provided value.
    if radius > 0.5 * min(span_x, span_y):
        return ()
    half_columns = max(1, int(math.ceil(radius / minimum_dx)))
    half_rows = max(1, int(math.ceil(radius / minimum_dy)))
    if 2 * half_columns + 1 > len(data.x_m) or 2 * half_rows + 1 > len(data.y_m):
        return ()

    local_minimum, local_maximum = _window_extrema(
        data.height_m,
        half_rows=half_rows,
        half_columns=half_columns,
    )
    relief = local_maximum - local_minimum
    cell_slope = _cell_triangle_maximum_slope(data)
    footprint_slope = _window_maximum(
        cell_slope,
        window_rows=2 * half_rows,
        window_columns=2 * half_columns,
    )
    x_centres = _interior_axis(data.x_m, half_columns)
    y_centres = _interior_axis(data.y_m, half_rows)
    centre_height = data.height_m[
        half_rows : len(data.y_m) - half_rows if half_rows else len(data.y_m),
        half_columns : len(data.x_m) - half_columns if half_columns else len(data.x_m),
    ]
    x_grid, y_grid = np.meshgrid(x_centres, y_centres)
    clearance = np.minimum.reduce(
        (
            x_grid - data.x_m[0],
            data.x_m[-1] - x_grid,
            y_grid - data.y_m[0],
            data.y_m[-1] - y_grid,
        )
    )
    valid = (
        (relief <= maximum_relief + 1.0e-12)
        & (footprint_slope <= maximum_slope + 1.0e-12)
        & (clearance >= radius + margin - 1.0e-12)
    )
    if height_range is not None:
        valid &= (local_minimum >= height_range[0] - 1.0e-12) & (
            local_maximum <= height_range[1] + 1.0e-12
        )
        preferred_height = 0.5 * (height_range[0] + height_range[1])
    else:
        preferred_height = 0.0
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return ()

    candidate_x = x_grid[rows, columns]
    candidate_y = y_grid[rows, columns]
    candidate_height = centre_height[rows, columns]
    candidate_relief = relief[rows, columns]
    candidate_max_slope = footprint_slope[rows, columns]
    candidate_distance = np.hypot(candidate_x, candidate_y)
    order = np.lexsort(
        (
            candidate_x,
            candidate_y,
            np.abs(candidate_height - preferred_height),
            candidate_max_slope,
            candidate_relief,
            candidate_distance,
        )
    )
    selected_indices: list[int] = []
    minimum_grid_spacing = min(minimum_dx, minimum_dy)
    field_diagonal = math.hypot(span_x, span_y)
    if separation <= minimum_grid_spacing + 1.0e-12:
        selected_indices = [int(index) for index in order[:count]]
    elif separation > field_diagonal:
        selected_indices = [int(order[0])]
    else:
        buckets: dict[tuple[int, int], list[int]] = {}
        minimum_distance_squared = separation * separation
        for ordered_index in order:
            candidate_index = int(ordered_index)
            x_value = float(candidate_x[candidate_index])
            y_value = float(candidate_y[candidate_index])
            key = (math.floor(x_value / separation), math.floor(y_value / separation))
            nearby = (
                other_index
                for offset_x in (-1, 0, 1)
                for offset_y in (-1, 0, 1)
                for other_index in buckets.get((key[0] + offset_x, key[1] + offset_y), ())
            )
            if any(
                (x_value - float(candidate_x[other_index])) ** 2
                + (y_value - float(candidate_y[other_index])) ** 2
                + 1.0e-12
                < minimum_distance_squared
                for other_index in nearby
            ):
                continue
            selected_indices.append(candidate_index)
            buckets.setdefault(key, []).append(candidate_index)
            if len(selected_indices) == count:
                break

    selected: list[SpawnCandidate] = []
    for candidate_index in selected_indices:
        x_value = float(candidate_x[candidate_index])
        y_value = float(candidate_y[candidate_index])
        reduced_row = int(rows[candidate_index])
        reduced_column = int(columns[candidate_index])
        source_row = reduced_row + half_rows
        source_column = reduced_column + half_columns
        _, dz_dx, dz_dy = _mujoco_triangle_sample(data, x_value, y_value)
        selected.append(
            SpawnCandidate(
                x_m=x_value,
                y_m=y_value,
                terrain_height_m=float(candidate_height[candidate_index]),
                normal_xyz=_normal_from_gradient(dz_dx, dz_dy),
                surface_slope_deg=_slope_degrees(dz_dx, dz_dy),
                maximum_footprint_slope_deg=float(candidate_max_slope[candidate_index]),
                footprint_relief_upper_bound_m=float(candidate_relief[candidate_index]),
                grid_boundary_clearance_m=float(clearance[reduced_row, reduced_column]),
                footprint_radius_m=radius,
                grid_index_yx=(source_row, source_column),
            )
        )
    return tuple(selected)


def _finite_nonnegative(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _finite_point(x: float, y: float) -> tuple[float, float]:
    x_value = float(x)
    y_value = float(y)
    if not np.isfinite([x_value, y_value]).all():
        raise ValueError("x and y must be finite")
    return x_value, y_value


def _height_range(
    value: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("ground_height_range_m must be a two-number tuple or None")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("ground_height_range_m must be finite and ordered")
    return low, high


def _require_inside(data: HeightFieldData, x: float, y: float) -> None:
    if x < data.x_m[0] or x > data.x_m[-1] or y < data.y_m[0] or y > data.y_m[-1]:
        raise OutOfBoundsError(f"query ({x}, {y}) is outside {data.bounds_xy_m}")


def _bilinear_grid_value(
    data: HeightFieldData,
    values: np.ndarray,
    x: float,
    y: float,
) -> float:
    column_hi = min(int(np.searchsorted(data.x_m, x, side="right")), len(data.x_m) - 1)
    row_hi = min(int(np.searchsorted(data.y_m, y, side="right")), len(data.y_m) - 1)
    column_lo = max(0, column_hi - 1)
    row_lo = max(0, row_hi - 1)
    x0, x1 = data.x_m[column_lo], data.x_m[column_hi]
    y0, y1 = data.y_m[row_lo], data.y_m[row_hi]
    tx = 0.0 if column_lo == column_hi else float((x - x0) / (x1 - x0))
    ty = 0.0 if row_lo == row_hi else float((y - y0) / (y1 - y0))
    low = (1.0 - tx) * values[row_lo, column_lo] + tx * values[row_lo, column_hi]
    high = (1.0 - tx) * values[row_hi, column_lo] + tx * values[row_hi, column_hi]
    return float((1.0 - ty) * low + ty * high)


def _mujoco_triangle_sample(
    data: HeightFieldData,
    x: float,
    y: float,
) -> tuple[float, float, float]:
    """Sample the same diagonal pair of planar faces used by MuJoCo hfields."""

    column = min(
        max(int(np.searchsorted(data.x_m, x, side="left")) - 1, 0),
        len(data.x_m) - 2,
    )
    row = min(
        max(int(np.searchsorted(data.y_m, y, side="left")) - 1, 0),
        len(data.y_m) - 2,
    )
    x0 = float(data.x_m[column])
    x1 = float(data.x_m[column + 1])
    y0 = float(data.y_m[row])
    y1 = float(data.y_m[row + 1])
    delta_x = x1 - x0
    delta_y = y1 - y0
    u = (x - x0) / delta_x
    v = (y - y0) / delta_y
    z00 = float(data.height_m[row, column])
    z01 = float(data.height_m[row, column + 1])
    z10 = float(data.height_m[row + 1, column])
    z11 = float(data.height_m[row + 1, column + 1])
    # MuJoCo resolves points on an internal grid edge toward the preceding
    # cell and points exactly on the cell diagonal toward the lower-right
    # triangle.  Matching both ties matters for spawn candidates because they
    # are deliberately reported at grid nodes.
    if v > u:
        dz_dx = (z11 - z10) / delta_x
        dz_dy = (z10 - z00) / delta_y
    else:
        dz_dx = (z01 - z00) / delta_x
        dz_dy = (z11 - z01) / delta_y
    height = z00 + dz_dx * (x - x0) + dz_dy * (y - y0)
    return float(height), float(dz_dx), float(dz_dy)


def _cell_triangle_maximum_slope(
    data: HeightFieldData,
    *,
    rows: slice = slice(None),
    columns: slice = slice(None),
) -> np.ndarray:
    """Return the steeper face per cell in the selected vertex window.

    Slice before allocating gradient arrays so local queries do not calculate
    slopes across the entire field. Keep the original grid spacing to preserve
    identical results even with floating-point variation along the world axes.
    """

    delta_x = float(data.x_m[1] - data.x_m[0])
    delta_y = float(data.y_m[1] - data.y_m[0])
    height = data.height_m[rows, columns]
    z00 = height[:-1, :-1]
    z01 = height[:-1, 1:]
    z10 = height[1:, :-1]
    z11 = height[1:, 1:]
    upper_left = np.degrees(
        np.arctan(
            np.hypot(
                (z11 - z10) / delta_x,
                (z10 - z00) / delta_y,
            )
        )
    )
    lower_right = np.degrees(
        np.arctan(
            np.hypot(
                (z01 - z00) / delta_x,
                (z11 - z01) / delta_y,
            )
        )
    )
    return np.maximum(upper_left, lower_right)


def _normal_from_gradient(dz_dx: float, dz_dy: float) -> tuple[float, float, float]:
    inverse_norm = 1.0 / math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
    return (-dz_dx * inverse_norm, -dz_dy * inverse_norm, inverse_norm)


def _slope_degrees(dz_dx: float, dz_dy: float) -> float:
    return math.degrees(math.atan(math.hypot(dz_dx, dz_dy)))


def _sample_window(axis: np.ndarray, value: float, radius: float) -> tuple[int, int]:
    first_cell = max(
        0,
        int(np.searchsorted(axis, value - radius, side="left")) - 1,
    )
    last_cell = min(
        len(axis) - 2,
        int(np.searchsorted(axis, value + radius, side="right")) - 1,
    )
    return first_cell, last_cell + 2


def _window_extrema(
    values: np.ndarray,
    *,
    half_rows: int,
    half_columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(values)
    maximum = np.asarray(values)
    if half_columns:
        width = 2 * half_columns + 1
        minimum = np.min(
            np.lib.stride_tricks.sliding_window_view(minimum, width, axis=1),
            axis=-1,
        )
        maximum = np.max(
            np.lib.stride_tricks.sliding_window_view(maximum, width, axis=1),
            axis=-1,
        )
    if half_rows:
        height = 2 * half_rows + 1
        minimum = np.min(
            np.lib.stride_tricks.sliding_window_view(minimum, height, axis=0),
            axis=-1,
        )
        maximum = np.max(
            np.lib.stride_tricks.sliding_window_view(maximum, height, axis=0),
            axis=-1,
        )
    return minimum, maximum


def _window_maximum(
    values: np.ndarray,
    *,
    window_rows: int,
    window_columns: int,
) -> np.ndarray:
    """Return a separable rectangular maximum without materializing all windows."""

    if window_rows <= 0 or window_columns <= 0:
        raise ValueError("window dimensions must be positive")
    maximum = np.max(
        np.lib.stride_tricks.sliding_window_view(values, window_columns, axis=1),
        axis=-1,
    )
    return np.max(
        np.lib.stride_tricks.sliding_window_view(maximum, window_rows, axis=0),
        axis=-1,
    )


def _interior_axis(axis: np.ndarray, half_window: int) -> np.ndarray:
    if half_window == 0:
        return axis
    return axis[half_window:-half_window]


def _validate_mujoco_grid_alignment(
    data: HeightFieldData,
    *,
    collision: Mapping[str, object],
    terrain_offset_z: float,
) -> None:
    """Fail closed if NPZ world axes disagree with the declared MuJoCo hfield."""

    try:
        half_x, half_y = (float(value) for value in collision["half_size_xy_m"])
        center_x, center_y, center_z = (
            float(value) for value in collision["geom_center_after_translation_m"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("collision grid placement is invalid") from exc
    if not np.isfinite([half_x, half_y, center_x, center_y, center_z]).all():
        raise ManifestError("collision grid placement must be finite")
    if half_x <= 0.0 or half_y <= 0.0:
        raise ManifestError("collision half sizes must be positive")

    delta_x = np.diff(data.x_m)
    delta_y = np.diff(data.y_m)
    expected_delta_x = 2.0 * half_x / (len(data.x_m) - 1)
    expected_delta_y = 2.0 * half_y / (len(data.y_m) - 1)
    if not np.allclose(delta_x, expected_delta_x, rtol=1.0e-9, atol=1.0e-10):
        raise ManifestError("heightfield x axis is not uniformly aligned with MuJoCo")
    if not np.allclose(delta_y, expected_delta_y, rtol=1.0e-9, atol=1.0e-10):
        raise ManifestError("heightfield y axis is not uniformly aligned with MuJoCo")
    expected_endpoints = np.asarray(
        [center_x - half_x, center_x + half_x, center_y - half_y, center_y + half_y]
    )
    actual_endpoints = np.asarray([data.x_m[0], data.x_m[-1], data.y_m[0], data.y_m[-1]])
    if not np.allclose(actual_endpoints, expected_endpoints, rtol=1.0e-9, atol=1.0e-10):
        raise ManifestError("heightfield world axes disagree with MuJoCo size/position")
    # The manifest records the logical pre-heightfield origin. Schema 3 shifts
    # only the emitted MJCF geom to its negative minimum elevation.
    if not math.isclose(center_z, -terrain_offset_z, rel_tol=1.0e-9, abs_tol=1.0e-10):
        raise ManifestError("heightfield world z translation disagrees with MuJoCo position")

    declared_resolution = collision.get("resolution_m")
    if declared_resolution is not None:
        resolution = float(declared_resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ManifestError("collision.resolution_m must be positive and finite")
        if not math.isclose(expected_delta_x, resolution, rel_tol=1.0e-7, abs_tol=1.0e-9):
            raise ManifestError("heightfield x spacing disagrees with collision.resolution_m")
        if not math.isclose(expected_delta_y, resolution, rel_tol=1.0e-7, abs_tol=1.0e-9):
            raise ManifestError("heightfield y spacing disagrees with collision.resolution_m")
