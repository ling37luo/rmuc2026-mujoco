"""Read-only queries against the runtime pack's exact float heightfield."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .errors import ManifestError, OutOfBoundsError
from .manifest import FieldAsset


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
    return HeightFieldData(x_m=x - offset_x, y_m=y - offset_y, height_m=height - offset_z)


def field_bounds(asset: FieldAsset) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return inclusive world-space XY bounds of the collision grid."""

    return load_heightfield(asset).bounds_xy_m


def height_at(
    asset: FieldAsset,
    x: float,
    y: float,
    *,
    out_of_bounds: Literal["raise", "clip", "nan"] = "raise",
) -> float:
    """Bilinearly interpolate collision height at one world-space XY position."""

    if out_of_bounds not in {"raise", "clip", "nan"}:
        raise ValueError("out_of_bounds must be 'raise', 'clip', or 'nan'")
    data = load_heightfield(asset)
    x_value = float(x)
    y_value = float(y)
    if not np.isfinite([x_value, y_value]).all():
        raise ValueError("x and y must be finite")
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

    column_hi = min(int(np.searchsorted(data.x_m, x_value, side="right")), len(data.x_m) - 1)
    row_hi = min(int(np.searchsorted(data.y_m, y_value, side="right")), len(data.y_m) - 1)
    column_lo = max(0, column_hi - 1)
    row_lo = max(0, row_hi - 1)
    x0, x1 = data.x_m[column_lo], data.x_m[column_hi]
    y0, y1 = data.y_m[row_lo], data.y_m[row_hi]
    tx = 0.0 if column_lo == column_hi else float((x_value - x0) / (x1 - x0))
    ty = 0.0 if row_lo == row_hi else float((y_value - y0) / (y1 - y0))
    low = (1.0 - tx) * data.height_m[row_lo, column_lo] + tx * data.height_m[row_lo, column_hi]
    high = (1.0 - tx) * data.height_m[row_hi, column_lo] + tx * data.height_m[row_hi, column_hi]
    return float((1.0 - ty) * low + ty * high)
