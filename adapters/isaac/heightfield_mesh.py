"""Exact grid topology for an Isaac/PhysX static triangle-mesh consumer.

No Isaac package is imported here. A USD Mesh stores points as float32, so a
runtime consumer must measure that conversion separately from this float64
source geometry. The diagonal is the MuJoCo hfield 00-to-11 split.
"""

from __future__ import annotations

import math

import numpy as np


def _checked_grid(
    x_m: np.ndarray, y_m: np.ndarray, height_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    if (
        x.ndim != 1
        or y.ndim != 1
        or x.size < 2
        or y.size < 2
        or height.shape != (y.size, x.size)
        or not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(height).all())
        or not (np.diff(x) > 0).all()
        or not (np.diff(y) > 0).all()
    ):
        raise ValueError("expected a finite, ascending height_m[y, x] grid")
    return x, y, height


def grid_to_triangle_mesh(
    x_m: np.ndarray, y_m: np.ndarray, height_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return local float64 points and upward-wound 00-to-11 triangle faces.

    XY points are relative to the first grid node. The USD parent Xform must
    translate by ``(x_m[0], y_m[0], 0)`` plus its environment offset. Z samples
    are already absolute world meters; no Z scaling or shifting is allowed.
    """

    x, y, height = _checked_grid(x_m, y_m, height_m)
    columns = x.size
    xx, yy = np.meshgrid(x - x[0], y - y[0])
    vertices = np.column_stack((xx.ravel(), yy.ravel(), height.ravel()))
    rows = np.arange(y.size - 1, dtype=np.int32)[:, None]
    cols = np.arange(columns - 1, dtype=np.int32)[None, :]
    base = (rows * columns + cols).ravel()
    faces = np.empty((2 * base.size, 3), dtype=np.int32)
    faces[0::2] = np.column_stack((base, base + 1, base + columns + 1))
    faces[1::2] = np.column_stack((base, base + columns + 1, base + columns))
    return vertices, faces


def triangle_height_at(
    x_m: np.ndarray, y_m: np.ndarray, height_m: np.ndarray, x: float, y: float
) -> float:
    """Evaluate the same planar triangle at an interior world-space XY point."""

    axis_x, axis_y, height = _checked_grid(x_m, y_m, height_m)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("query coordinates must be finite")
    if x < axis_x[0] or x > axis_x[-1] or y < axis_y[0] or y > axis_y[-1]:
        raise ValueError("query is outside the exported grid")
    col = min(max(int(np.searchsorted(axis_x, x, side="left")) - 1, 0), axis_x.size - 2)
    row = min(max(int(np.searchsorted(axis_y, y, side="left")) - 1, 0), axis_y.size - 2)
    u = (x - axis_x[col]) / (axis_x[col + 1] - axis_x[col])
    v = (y - axis_y[row]) / (axis_y[row + 1] - axis_y[row])
    z00 = height[row, col]
    z01 = height[row, col + 1]
    z10 = height[row + 1, col]
    z11 = height[row + 1, col + 1]
    if v > u:
        return float(z00 + (z11 - z10) * u + (z10 - z00) * v)
    return float(z00 + (z01 - z00) * u + (z11 - z01) * v)


def environment_offsets(
    count: int, x_m: np.ndarray, y_m: np.ndarray, *, gap_m: float = 1.0
) -> np.ndarray:
    """Separate N copies of the local collider without changing its samples."""

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or x.ndim != 1
        or y.ndim != 1
        or x.size < 2
        or y.size < 2
        or not np.isfinite(gap_m)
        or gap_m <= 0
    ):
        raise ValueError("invalid environment count, axes, or gap")
    columns = math.ceil(math.sqrt(count))
    spacing_x = float(x[-1] - x[0] + gap_m)
    spacing_y = float(y[-1] - y[0] + gap_m)
    return np.asarray(
        [(index % columns * spacing_x, index // columns * spacing_y) for index in range(count)],
        dtype=np.float64,
    )


__all__ = ["environment_offsets", "grid_to_triangle_mesh", "triangle_height_at"]
