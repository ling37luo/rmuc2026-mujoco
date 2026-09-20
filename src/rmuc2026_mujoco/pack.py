#!/usr/bin/env python3
"""Export the verified RMUC 2026 field build as a relocatable MuJoCo asset pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .livery import (
    LiveryProcessingError,
    SOURCE_BAKED_SCENE_CONTENT,
    SOURCE_SURFACE_GUIDE_KIND,
    mask_rulebook_page_edge,
)
from .wall_tip_repair import validate_wall_tip_repair_record


EXPECTED_ARTIFACT_TYPE = "rmuc2026_official_field_mujoco_asset"
EXPECTED_VISUAL_MESH_COUNT = 38
OUTPUT_ARTIFACT_TYPE = "rmuc2026_mujoco_runtime_asset_pack"
OUTPUT_XML = "rmuc2026_field.xml"
OUTPUT_COLLISION_ONLY_XML = "rmuc2026_field_collision_only.xml"
OUTPUT_SCHEMA_VERSION = 2
PUBLIC_DISPLAY_RGB_BLACK_FLOOR = 0.06
PUBLIC_DISPLAY_RGB_WHITE_CEILING = 0.76
PUBLIC_DISPLAY_RGB_GAMMA = 0.90
PUBLIC_DISPLAY_BASE_SURFACE_SCALE = 0.64
PUBLIC_DISPLAY_PROFILE = "cad_source_contrast_v2"
KEY_LIGHT_NAME = "rmuc2026_key_light"
FILL_LIGHT_NAME = "rmuc2026_fill_light"
SURFACE_GUIDE_KIND = SOURCE_SURFACE_GUIDE_KIND
SURFACE_GUIDE_TEXTURE_FILE = "visual/official_rulebook_v2_overhead_surface.png"
SURFACE_GUIDE_MESH_FILE = "visual/rmuc2026_surface_guide.obj"
SURFACE_GUIDE_GEOM_NAME = "rmuc2026_surface_guide"
SURFACE_GUIDE_GEOM_GROUP = 4
SURFACE_GUIDE_BAKED_CONTENT = SOURCE_BAKED_SCENE_CONTENT
SURFACE_GUIDE_SAMPLING_M = 0.05
SURFACE_GUIDE_CLEARANCE_M = 0.005
SURFACE_GUIDE_MINIMUM_ABOVE_TERRAIN_M = 0.002
SURFACE_GUIDE_MAXIMUM_ABOVE_TERRAIN_M = 0.008
SURFACE_GUIDE_MAXIMUM_CELL_HEIGHT_DELTA_M = 0.20
SURFACE_GUIDE_MAXIMUM_SURFACE_SLOPE = 0.8


class ExportBlocked(RuntimeError):
    """The source field build cannot safely produce the requested pack."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportBlocked(f"{label}必须是JSON对象")
    return value


def _finite_numbers(value: object, *, count: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ExportBlocked(f"{label}必须含{count}个数")
    numbers = [float(item) for item in value]
    if not all(math.isfinite(item) for item in numbers):
        raise ExportBlocked(f"{label}含非有限值")
    return numbers


def _json_safe_copy(value: object, label: str) -> object:
    """Refuse NaN/Infinity instead of writing a manifest that no parser accepts."""

    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExportBlocked(f"{label}不是可序列化的有限JSON：{exc}") from exc


def _source_count(source: dict[str, Any], key: str, *, label: str) -> int | None:
    """Copy an audit counter, or record None when the source never measured it.

    The runtime pack keeps the counters that say how much of the collision
    surface is synthetic.  A missing counter stays explicit rather than
    becoming a fabricated zero.
    """

    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExportBlocked(f"{label}.{key}必须是非负整数")
    return value


def _display_rgba(source_rgba: list[float], *, visual_role: str) -> list[float]:
    """Map source colours to a legible display palette without changing provenance."""

    rgb = [
        PUBLIC_DISPLAY_RGB_BLACK_FLOOR
        + (PUBLIC_DISPLAY_RGB_WHITE_CEILING - PUBLIC_DISPLAY_RGB_BLACK_FLOOR)
        * value**PUBLIC_DISPLAY_RGB_GAMMA
        for value in source_rgba[:3]
    ]
    if visual_role == "base_surface_shell":
        rgb = [value * PUBLIC_DISPLAY_BASE_SURFACE_SCALE for value in rgb]
    return [*rgb, source_rgba[3]]


def _source_file(
    field_root: Path,
    relative: object,
    expected_sha256: object,
    *,
    suffix: str,
) -> tuple[str, Path, str]:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ExportBlocked("源manifest文件路径必须是非空相对路径")
    source = (field_root / relative).resolve()
    if field_root not in source.parents or not source.is_file():
        raise ExportBlocked(f"源文件缺失或越界：{relative}")
    if source.suffix.lower() != suffix:
        raise ExportBlocked(f"源文件扩展名错误：{relative}")
    actual_sha256 = sha256_file(source)
    if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
        raise ExportBlocked(f"源文件SHA-256不匹配：{relative}")
    return Path(relative).as_posix(), source, actual_sha256


def _png_dimensions(path: Path, *, label: str) -> tuple[int, int]:
    """Read the mandatory PNG signature and IHDR dimensions without Pillow."""

    try:
        header = path.read_bytes()[:24]
    except OSError as exc:
        raise ExportBlocked(f"{label}无法读取：{exc}") from exc
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[8:12] != (13).to_bytes(4, "big")
        or header[12:16] != b"IHDR"
    ):
        raise ExportBlocked(f"{label}不是有效PNG")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        raise ExportBlocked(f"{label}尺寸无效")
    return width, height


def _source_surface_guide(
    field_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate the local rulebook guide and return its bounded world mapping."""

    surface = _json_object(manifest.get("surface_guide"), "surface_guide")
    if surface.get("kind") != SURFACE_GUIDE_KIND:
        raise ExportBlocked("surface_guide.kind不是支持的官方规则手册俯视图")
    if surface.get("physics") is not False:
        raise ExportBlocked("surface_guide必须明确声明physics=false")
    relative, source, digest = _source_file(
        field_root,
        surface.get("file"),
        surface.get("sha256"),
        suffix=".png",
    )
    width, height = _png_dimensions(source, label="surface_guide")
    for key, actual in (("width_px", width), ("height_px", height)):
        declared = surface.get(key)
        if isinstance(declared, bool) or not isinstance(declared, int) or declared != actual:
            raise ExportBlocked(f"surface_guide.{key}与PNG不一致")

    mapping = _json_object(surface.get("world_mapping"), "surface_guide.world_mapping")
    expected_mapping = {
        "image_left_to_world": "negative_x_red_side",
        "image_right_to_world": "positive_x_blue_side",
        "image_top_to_world": "positive_y",
        "rectangle": "official_cad_assembly_outer_xy_bounds",
        "calibration": "diagnostic_outer_bounds_fit_not_survey_homography",
    }
    if any(mapping.get(key) != value for key, value in expected_mapping.items()):
        raise ExportBlocked("surface_guide.world_mapping方向或标定合同损坏")
    raw_bounds = mapping.get("world_bounds_xy_m")
    if (
        not isinstance(raw_bounds, list)
        or len(raw_bounds) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in raw_bounds)
    ):
        raise ExportBlocked("surface_guide.world_mapping.world_bounds_xy_m必须是2x2数组")
    low = np.asarray(raw_bounds[0], dtype=np.float64)
    high = np.asarray(raw_bounds[1], dtype=np.float64)
    if not np.isfinite(np.concatenate((low, high))).all() or np.any(high <= low):
        raise ExportBlocked("surface_guide世界边界无效")
    size = high - low
    declared_size = np.asarray(mapping.get("world_size_xy_m"), dtype=np.float64)
    if declared_size.shape != (2,) or not np.isfinite(declared_size).all():
        raise ExportBlocked("surface_guide.world_size_xy_m必须含2个有限数")
    if not np.allclose(declared_size, size, rtol=0.0, atol=1.0e-6):
        raise ExportBlocked("surface_guide世界尺寸与边界不一致")

    dimensions = _json_object(manifest.get("dimensions"), "dimensions")
    outer = np.asarray(
        dimensions.get("cad_assembly_outer_bounds_after_translation_m"),
        dtype=np.float64,
    )
    if outer.shape != (2, 3) or not np.isfinite(outer).all():
        raise ExportBlocked("dimensions缺少有效的CAD外包围")
    if not np.allclose(outer[:, :2], np.stack((low, high)), rtol=0.0, atol=1.0e-6):
        raise ExportBlocked("surface_guide世界边界与CAD外包围不一致")
    if not math.isclose(width / height, size[0] / size[1], rel_tol=0.02, abs_tol=0.0):
        raise ExportBlocked("surface_guide像素比例与世界映射不一致")
    return {
        "source_relative": relative,
        "source_path": source,
        "source_sha256": digest,
        "world_bounds_xy_m": np.stack((low, high)),
    }


def _bounded_uniform_axis(
    values: np.ndarray,
    *,
    low: float,
    high: float,
    sampling_m: float,
    label: str,
) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ExportBlocked(f"{label}坐标轴损坏")
    steps = np.diff(values)
    if not np.all(steps > 0.0):
        raise ExportBlocked(f"{label}坐标轴必须严格递增")
    median_step = float(np.median(steps))
    tolerance = max(1.0e-6, 0.51 * median_step)
    if low < values[0] - tolerance or high > values[-1] + tolerance or high <= low:
        raise ExportBlocked(f"surface_guide与{label}坐标轴没有足够重叠")
    segment_count = max(1, int(math.ceil((high - low) / sampling_m)))
    return np.linspace(low, high, segment_count + 1, dtype=np.float64)


def _mujoco_triangle_height_samples(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    selected_x: np.ndarray,
    selected_y: np.ndarray,
) -> np.ndarray:
    """Sample MuJoCo's fixed-diagonal hfield triangles at exact mesh XY coordinates."""

    x_lower = np.clip(np.searchsorted(x, selected_x, side="left") - 1, 0, len(x) - 2)
    y_lower = np.clip(np.searchsorted(y, selected_y, side="left") - 1, 0, len(y) - 2)
    x_weight = (selected_x - x[x_lower]) / (x[x_lower + 1] - x[x_lower])
    y_weight = (selected_y - y[y_lower]) / (y[y_lower + 1] - y[y_lower])
    lower_left = height[np.ix_(y_lower, x_lower)]
    lower_right = height[np.ix_(y_lower, x_lower + 1)]
    upper_left = height[np.ix_(y_lower + 1, x_lower)]
    upper_right = height[np.ix_(y_lower + 1, x_lower + 1)]
    u = x_weight[None, :]
    v = y_weight[:, None]
    lower_triangle = lower_left + (lower_right - lower_left) * u + (upper_right - lower_right) * v
    upper_triangle = lower_left + (upper_right - upper_left) * u + (upper_left - lower_left) * v
    return np.where(v > u, upper_triangle, lower_triangle)


def _mujoco_triangle_height_points(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    points_x: np.ndarray,
    points_y: np.ndarray,
) -> np.ndarray:
    """Sample fixed-diagonal hfield triangles at pairwise XY coordinates."""

    x_lower = np.clip(np.searchsorted(x, points_x, side="left") - 1, 0, len(x) - 2)
    y_lower = np.clip(np.searchsorted(y, points_y, side="left") - 1, 0, len(y) - 2)
    u = (points_x - x[x_lower]) / (x[x_lower + 1] - x[x_lower])
    v = (points_y - y[y_lower]) / (y[y_lower + 1] - y[y_lower])
    ll = height[y_lower, x_lower]
    lr = height[y_lower, x_lower + 1]
    ul = height[y_lower + 1, x_lower]
    ur = height[y_lower + 1, x_lower + 1]
    lower = ll + (lr - ll) * u + (ur - lr) * v
    upper = ll + (ur - ul) * u + (ul - ll) * v
    return np.where(v > u, upper, lower)


def _surface_guide_signed_error_bounds(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    selected_x: np.ndarray,
    selected_y: np.ndarray,
    selected_height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound both signs of guide height error over each candidate cell.

    Inspect the 1 cm heightfield nodes, native-cell centres, guide-cell centres,
    and intersections between guide diagonals/edges and native grid lines.
    Chunk the Cartesian probes so a full field does not materialize millions
    of samples at once. Error is guide height minus authoritative terrain.
    """

    cells_y, cells_x = len(selected_y) - 1, len(selected_x) - 1
    minimum = np.full((cells_y, cells_x), np.inf)
    maximum = np.full((cells_y, cells_x), -np.inf)
    for probe_x, probe_y, native_nodes in (
        (x, y, True),
        (0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:]), False),
    ):
        x_inside = (probe_x >= selected_x[0]) & (probe_x <= selected_x[-1])
        x_cells = np.clip(np.searchsorted(selected_x, probe_x, side="right") - 1, 0, cells_x - 1)
        u = (probe_x - selected_x[x_cells]) / (selected_x[x_cells + 1] - selected_x[x_cells])
        for first_row in range(0, len(probe_y), 128):
            stop_row = min(first_row + 128, len(probe_y))
            section_y = probe_y[first_row:stop_row]
            y_inside = (section_y >= selected_y[0]) & (section_y <= selected_y[-1])
            y_cells = np.clip(
                np.searchsorted(selected_y, section_y, side="right") - 1,
                0,
                cells_y - 1,
            )
            v = (section_y - selected_y[y_cells]) / (selected_y[y_cells + 1] - selected_y[y_cells])
            ll = selected_height[np.ix_(y_cells, x_cells)]
            lr = selected_height[np.ix_(y_cells, x_cells + 1)]
            ul = selected_height[np.ix_(y_cells + 1, x_cells)]
            ur = selected_height[np.ix_(y_cells + 1, x_cells + 1)]
            lower = ll + (lr - ll) * u[None, :] + (ur - lr) * v[:, None]
            upper = ll + (ur - ul) * u[None, :] + (ul - ll) * v[:, None]
            guide = np.where(v[:, None] > u[None, :], upper, lower)
            terrain = (
                height[first_row:stop_row]
                if native_nodes
                else _mujoco_triangle_height_samples(x, y, height, probe_x, section_y)
            )
            error = guide - terrain
            inside = y_inside[:, None] & x_inside[None, :]
            np.minimum.at(
                minimum, (y_cells[:, None], x_cells[None, :]), np.where(inside, error, np.inf)
            )
            np.maximum.at(
                maximum, (y_cells[:, None], x_cells[None, :]), np.where(inside, error, -np.inf)
            )

    center_x = 0.5 * (selected_x[:-1] + selected_x[1:])
    center_y = 0.5 * (selected_y[:-1] + selected_y[1:])
    center_error = 0.5 * (selected_height[:-1, :-1] + selected_height[1:, 1:]) - (
        _mujoco_triangle_height_samples(x, y, height, center_x, center_y)
    )
    np.minimum(minimum, center_error, out=minimum)
    np.maximum(maximum, center_error, out=maximum)

    ll = selected_height[:-1, :-1]
    lr = selected_height[:-1, 1:]
    ul = selected_height[1:, :-1]
    ur = selected_height[1:, 1:]

    def include(
        points_x: np.ndarray, points_y: np.ndarray, guide: np.ndarray, valid: np.ndarray
    ) -> None:
        terrain = _mujoco_triangle_height_points(x, y, height, points_x, points_y)
        error = guide - terrain
        np.minimum(minimum, np.where(valid, error, np.inf), out=minimum)
        np.maximum(maximum, np.where(valid, error, -np.inf), out=maximum)

    # The diagonal and four outer edges are the only guide triangle edges.
    # Probe their crossings with every native horizontal/vertical grid line.
    first_x = np.searchsorted(x, selected_x[:-1], side="right")
    first_y = np.searchsorted(y, selected_y[:-1], side="right")
    max_x_lines = int(np.ceil(np.max(np.diff(selected_x)) / np.min(np.diff(x)))) + 2
    max_y_lines = int(np.ceil(np.max(np.diff(selected_y)) / np.min(np.diff(y)))) + 2
    for offset in range(max_x_lines):
        indices = first_x + offset
        native_x = x[np.clip(indices, 0, len(x) - 1)]
        valid_x = (indices < len(x)) & (native_x < selected_x[1:])
        u = (native_x - selected_x[:-1]) / np.diff(selected_x)
        points_x = np.broadcast_to(native_x[None, :], minimum.shape)
        diagonal_y = selected_y[:-1, None] + u[None, :] * np.diff(selected_y)[:, None]
        include(points_x, diagonal_y, ll + (ur - ll) * u[None, :], valid_x[None, :])
        for row_edge, guide in (
            (selected_y[:-1, None], ll + (lr - ll) * u[None, :]),
            (selected_y[1:, None], ul + (ur - ul) * u[None, :]),
        ):
            include(points_x, np.broadcast_to(row_edge, minimum.shape), guide, valid_x[None, :])
    for offset in range(max_y_lines):
        indices = first_y + offset
        native_y = y[np.clip(indices, 0, len(y) - 1)]
        valid_y = (indices < len(y)) & (native_y < selected_y[1:])
        v = (native_y - selected_y[:-1]) / np.diff(selected_y)
        points_y = np.broadcast_to(native_y[:, None], minimum.shape)
        diagonal_x = selected_x[None, :-1] + v[:, None] * np.diff(selected_x)[None, :]
        include(diagonal_x, points_y, ll + (ur - ll) * v[:, None], valid_y[:, None])
        for column_edge, guide in (
            (selected_x[None, :-1], ll + (ul - ll) * v[:, None]),
            (selected_x[None, 1:], lr + (ur - lr) * v[:, None]),
        ):
            include(np.broadcast_to(column_edge, minimum.shape), points_y, guide, valid_y[:, None])
    return minimum, maximum


def _surface_guide_local_error_bounds(
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_height: np.ndarray,
    cell_rows: np.ndarray,
    cell_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Probe only selected subcells against the original 1 cm triangles."""

    x0, x1 = grid_x[cell_columns], grid_x[cell_columns + 1]
    y0, y1 = grid_y[cell_rows], grid_y[cell_rows + 1]
    ll = grid_height[cell_rows, cell_columns]
    lr = grid_height[cell_rows, cell_columns + 1]
    ul = grid_height[cell_rows + 1, cell_columns]
    ur = grid_height[cell_rows + 1, cell_columns + 1]
    minimum = np.full(len(cell_rows), np.inf)
    maximum = np.full(len(cell_rows), -np.inf)

    def include(
        points_x: np.ndarray,
        points_y: np.ndarray,
        guide: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        terrain = _mujoco_triangle_height_points(x, y, height, points_x, points_y)
        error = guide - terrain
        if valid is not None:
            error_min = np.where(valid, error, np.inf)
            error_max = np.where(valid, error, -np.inf)
        else:
            error_min = error_max = error
        np.minimum(minimum, error_min, out=minimum)
        np.maximum(maximum, error_max, out=maximum)

    def guide_at(points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        u = (points_x - x0) / (x1 - x0)
        v = (points_y - y0) / (y1 - y0)
        lower = ll + (lr - ll) * u + (ur - lr) * v
        upper = ll + (ur - ul) * u + (ul - ll) * v
        return np.where(v > u, upper, lower)

    for points_x, points_y, guide in (
        (x0, y0, ll),
        (x1, y0, lr),
        (x0, y1, ul),
        (x1, y1, ur),
        (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (ll + ur)),
    ):
        include(points_x, points_y, guide)

    native_x_mid = 0.5 * (x[:-1] + x[1:])
    native_y_mid = 0.5 * (y[:-1] + y[1:])
    for probe_x, probe_y in ((x, y), (native_x_mid, native_y_mid)):
        first_x = np.searchsorted(probe_x, x0, side="right")
        first_y = np.searchsorted(probe_y, y0, side="right")
        max_x = int(math.ceil(float(np.max(x1 - x0)) / float(np.min(np.diff(probe_x))))) + 1
        max_y = int(math.ceil(float(np.max(y1 - y0)) / float(np.min(np.diff(probe_y))))) + 1
        for x_offset in range(max_x):
            index_x = first_x + x_offset
            points_x = probe_x[np.clip(index_x, 0, len(probe_x) - 1)]
            valid_x = (index_x < len(probe_x)) & (points_x < x1)
            for y_offset in range(max_y):
                index_y = first_y + y_offset
                points_y = probe_y[np.clip(index_y, 0, len(probe_y) - 1)]
                valid_y = (index_y < len(probe_y)) & (points_y < y1)
                include(points_x, points_y, guide_at(points_x, points_y), valid_x & valid_y)

    first_x = np.searchsorted(x, x0, side="right")
    first_y = np.searchsorted(y, y0, side="right")
    max_x = int(math.ceil(float(np.max(x1 - x0)) / float(np.min(np.diff(x))))) + 1
    max_y = int(math.ceil(float(np.max(y1 - y0)) / float(np.min(np.diff(y))))) + 1
    for offset in range(max_x):
        index = first_x + offset
        native_x = x[np.clip(index, 0, len(x) - 1)]
        valid = (index < len(x)) & (native_x < x1)
        u = (native_x - x0) / (x1 - x0)
        diagonal_y = y0 + u * (y1 - y0)
        include(native_x, diagonal_y, ll + (ur - ll) * u, valid)
        include(native_x, y0, ll + (lr - ll) * u, valid)
        include(native_x, y1, ul + (ur - ul) * u, valid)
    for offset in range(max_y):
        index = first_y + offset
        native_y = y[np.clip(index, 0, len(y) - 1)]
        valid = (index < len(y)) & (native_y < y1)
        v = (native_y - y0) / (y1 - y0)
        diagonal_x = x0 + v * (x1 - x0)
        include(diagonal_x, native_y, ll + (ur - ll) * v, valid)
        include(x0, native_y, ll + (ul - ll) * v, valid)
        include(x1, native_y, lr + (ur - lr) * v, valid)
    return minimum, maximum


def _write_surface_guide_mesh(
    path: Path,
    *,
    samples_path: Path,
    collision: dict[str, Any],
    recommended_spawn: dict[str, Any],
    world_bounds_xy_m: np.ndarray,
) -> dict[str, Any]:
    """Write the full non-contact rulebook surface used by the visual5 pack.

    The 5 cm grid follows MuJoCo's fixed-diagonal heightfield surface. Cells
    whose interior departs from the physical surface are refined to 2.5 cm;
    remaining bad cells are omitted rather than drawing false raised bridges
    or intersecting the terrain. Discontinuity gaps are a known visual limit,
    while the complete RGB texture remains available on retained cells. Texture
    alpha never selects geometry.
    """

    try:
        with np.load(samples_path, allow_pickle=False) as samples:
            x = np.asarray(samples["x_m"], dtype=np.float64)
            y = np.asarray(samples["y_m"], dtype=np.float64)
            height = np.asarray(samples["height_m"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ExportBlocked(f"surface_guide无法读取高度场样本：{exc}") from exc
    rows = int(collision["rows_y"])
    columns = int(collision["columns_x"])
    if x.shape != (columns,) or y.shape != (rows,) or height.shape != (rows, columns):
        raise ExportBlocked("surface_guide高度场样本形状与collision合同不一致")
    if not np.isfinite(height).all():
        raise ExportBlocked("surface_guide高度场样本含非有限值")
    x = x - float(recommended_spawn["x_before_translation_m"])
    y = y - float(recommended_spawn["y_before_translation_m"])
    height = height - float(recommended_spawn["terrain_height_m"])
    bounds = np.asarray(world_bounds_xy_m, dtype=np.float64)
    selected_x = _bounded_uniform_axis(
        x,
        low=float(bounds[0, 0]),
        high=float(bounds[1, 0]),
        sampling_m=SURFACE_GUIDE_SAMPLING_M,
        label="X",
    )
    selected_y = _bounded_uniform_axis(
        y,
        low=float(bounds[0, 1]),
        high=float(bounds[1, 1]),
        sampling_m=SURFACE_GUIDE_SAMPLING_M,
        label="Y",
    )
    selected_height = _mujoco_triangle_height_samples(x, y, height, selected_x, selected_y)
    mesh_rows, mesh_columns = len(selected_y), len(selected_x)
    cell_maximum = np.maximum.reduce(
        (
            selected_height[:-1, :-1],
            selected_height[:-1, 1:],
            selected_height[1:, :-1],
            selected_height[1:, 1:],
        )
    )
    cell_minimum = np.minimum.reduce(
        (
            selected_height[:-1, :-1],
            selected_height[:-1, 1:],
            selected_height[1:, :-1],
            selected_height[1:, 1:],
        )
    )
    minimum_error, maximum_error = _surface_guide_signed_error_bounds(
        x, y, height, selected_x, selected_y, selected_height
    )
    coarse_maximum_rise = np.minimum(
        SURFACE_GUIDE_MAXIMUM_CELL_HEIGHT_DELTA_M,
        SURFACE_GUIDE_MAXIMUM_SURFACE_SLOPE
        * np.hypot(np.diff(selected_y)[:, None], np.diff(selected_x)[None, :]),
    )
    smooth_cells = cell_maximum - cell_minimum <= coarse_maximum_rise
    below_terrain = minimum_error + SURFACE_GUIDE_CLEARANCE_M < (
        SURFACE_GUIDE_MINIMUM_ABOVE_TERRAIN_M - 1.0e-10
    )
    raised_bridge = maximum_error + SURFACE_GUIDE_CLEARANCE_M > (
        SURFACE_GUIDE_MAXIMUM_ABOVE_TERRAIN_M + 1.0e-10
    )
    coarse_visible = smooth_cells & ~below_terrain & ~raised_bridge

    # Locally refine only rejected 5 cm cells. Where a refined cell meets a
    # coarse triangle, put its midpoint exactly on the coarse edge; otherwise
    # a T-junction can open a visible crack even if both heights are valid.
    fine_x = np.linspace(bounds[0, 0], bounds[1, 0], 2 * (mesh_columns - 1) + 1)
    fine_y = np.linspace(bounds[0, 1], bounds[1, 1], 2 * (mesh_rows - 1) + 1)
    fine_height = _mujoco_triangle_height_samples(x, y, height, fine_x, fine_y)
    horizontal_coarse_edge = np.zeros((mesh_rows, mesh_columns - 1), dtype=bool)
    horizontal_coarse_edge[:-1] |= coarse_visible
    horizontal_coarse_edge[1:] |= coarse_visible
    horizontal_midpoints = fine_height[::2, 1::2]
    horizontal_linear = 0.5 * (selected_height[:, :-1] + selected_height[:, 1:])
    horizontal_midpoints[horizontal_coarse_edge] = horizontal_linear[horizontal_coarse_edge]
    vertical_coarse_edge = np.zeros((mesh_rows - 1, mesh_columns), dtype=bool)
    vertical_coarse_edge[:, :-1] |= coarse_visible
    vertical_coarse_edge[:, 1:] |= coarse_visible
    vertical_midpoints = fine_height[1::2, ::2]
    vertical_linear = 0.5 * (selected_height[:-1, :] + selected_height[1:, :])
    vertical_midpoints[vertical_coarse_edge] = vertical_linear[vertical_coarse_edge]

    fine_minimum_error, fine_maximum_error = _surface_guide_signed_error_bounds(
        x, y, height, fine_x, fine_y, fine_height
    )
    fine_cell_maximum = np.maximum.reduce(
        (
            fine_height[:-1, :-1],
            fine_height[:-1, 1:],
            fine_height[1:, :-1],
            fine_height[1:, 1:],
        )
    )
    fine_cell_minimum = np.minimum.reduce(
        (
            fine_height[:-1, :-1],
            fine_height[:-1, 1:],
            fine_height[1:, :-1],
            fine_height[1:, 1:],
        )
    )
    fine_maximum_rise = np.minimum(
        SURFACE_GUIDE_MAXIMUM_CELL_HEIGHT_DELTA_M,
        SURFACE_GUIDE_MAXIMUM_SURFACE_SLOPE
        * np.hypot(np.diff(fine_y)[:, None], np.diff(fine_x)[None, :]),
    )
    fine_valid = (
        (fine_cell_maximum - fine_cell_minimum <= fine_maximum_rise)
        & (
            fine_minimum_error + SURFACE_GUIDE_CLEARANCE_M
            >= SURFACE_GUIDE_MINIMUM_ABOVE_TERRAIN_M - 1.0e-10
        )
        & (
            fine_maximum_error + SURFACE_GUIDE_CLEARANCE_M
            <= SURFACE_GUIDE_MAXIMUM_ABOVE_TERRAIN_M + 1.0e-10
        )
    )
    fine_visible = fine_valid & np.repeat(np.repeat(~coarse_visible, 2, axis=0), 2, axis=1)
    fine_rejected = ~fine_valid & np.repeat(np.repeat(~coarse_visible, 2, axis=0), 2, axis=1)

    # A final local level gives narrow stair treads a chance to retain the
    # picture. Do not refine accepted cells or flatten the physical risers.
    quarter_x = np.linspace(bounds[0, 0], bounds[1, 0], 4 * (mesh_columns - 1) + 1)
    quarter_y = np.linspace(bounds[0, 1], bounds[1, 1], 4 * (mesh_rows - 1) + 1)
    quarter_height = _mujoco_triangle_height_samples(x, y, height, quarter_x, quarter_y)
    quarter_height[::2, ::2] = fine_height
    horizontal_fine_edge = np.zeros((len(fine_y), len(fine_x) - 1), dtype=bool)
    horizontal_fine_edge[:-1] |= fine_visible
    horizontal_fine_edge[1:] |= fine_visible
    quarter_horizontal_midpoints = quarter_height[::2, 1::2]
    fine_horizontal_linear = 0.5 * (fine_height[:, :-1] + fine_height[:, 1:])
    quarter_horizontal_midpoints[horizontal_fine_edge] = fine_horizontal_linear[
        horizontal_fine_edge
    ]
    vertical_fine_edge = np.zeros((len(fine_y) - 1, len(fine_x)), dtype=bool)
    vertical_fine_edge[:, :-1] |= fine_visible
    vertical_fine_edge[:, 1:] |= fine_visible
    quarter_vertical_midpoints = quarter_height[1::2, ::2]
    fine_vertical_linear = 0.5 * (fine_height[:-1, :] + fine_height[1:, :])
    quarter_vertical_midpoints[vertical_fine_edge] = fine_vertical_linear[vertical_fine_edge]

    # Four quarter-grid segments also meet a 5 cm coarse edge. Their shared
    # 1/4, 1/2 and 3/4 vertices must lie exactly on its straight triangle edge.
    for fraction in (1, 2, 3):
        horizontal = quarter_height[::4, fraction::4]
        horizontal_linear = selected_height[:, :-1] + (
            selected_height[:, 1:] - selected_height[:, :-1]
        ) * (fraction / 4.0)
        horizontal[horizontal_coarse_edge] = horizontal_linear[horizontal_coarse_edge]
        vertical = quarter_height[fraction::4, ::4]
        vertical_linear = selected_height[:-1, :] + (
            selected_height[1:, :] - selected_height[:-1, :]
        ) * (fraction / 4.0)
        vertical[vertical_coarse_edge] = vertical_linear[vertical_coarse_edge]

    quarter_candidates = np.repeat(np.repeat(fine_rejected, 2, axis=0), 2, axis=1)
    candidate_rows, candidate_columns = np.nonzero(quarter_candidates)
    quarter_visible = np.zeros(quarter_candidates.shape, dtype=bool)
    quarter_retained_minimum = math.inf
    quarter_retained_maximum = -math.inf
    if len(candidate_rows):
        quarter_minimum_error, quarter_maximum_error = _surface_guide_local_error_bounds(
            x,
            y,
            height,
            quarter_x,
            quarter_y,
            quarter_height,
            candidate_rows,
            candidate_columns,
        )
        qll = quarter_height[candidate_rows, candidate_columns]
        qlr = quarter_height[candidate_rows, candidate_columns + 1]
        qul = quarter_height[candidate_rows + 1, candidate_columns]
        qur = quarter_height[candidate_rows + 1, candidate_columns + 1]
        quarter_maximum_rise = np.minimum(
            SURFACE_GUIDE_MAXIMUM_CELL_HEIGHT_DELTA_M,
            SURFACE_GUIDE_MAXIMUM_SURFACE_SLOPE
            * np.hypot(
                quarter_y[candidate_rows + 1] - quarter_y[candidate_rows],
                quarter_x[candidate_columns + 1] - quarter_x[candidate_columns],
            ),
        )
        quarter_smooth = (
            np.maximum.reduce((qll, qlr, qul, qur)) - np.minimum.reduce((qll, qlr, qul, qur))
            <= quarter_maximum_rise
        )
        valid_candidates = (
            quarter_smooth
            & (
                quarter_minimum_error + SURFACE_GUIDE_CLEARANCE_M
                >= SURFACE_GUIDE_MINIMUM_ABOVE_TERRAIN_M - 1.0e-10
            )
            & (
                quarter_maximum_error + SURFACE_GUIDE_CLEARANCE_M
                <= SURFACE_GUIDE_MAXIMUM_ABOVE_TERRAIN_M + 1.0e-10
            )
        )
        quarter_visible[candidate_rows[valid_candidates], candidate_columns[valid_candidates]] = (
            True
        )
        if np.any(valid_candidates):
            quarter_retained_minimum = float(
                np.min(quarter_minimum_error[valid_candidates] + SURFACE_GUIDE_CLEARANCE_M)
            )
            quarter_retained_maximum = float(
                np.max(quarter_maximum_error[valid_candidates] + SURFACE_GUIDE_CLEARANCE_M)
            )

    if not np.any(coarse_visible) and not np.any(fine_visible) and not np.any(quarter_visible):
        raise ExportBlocked("surface_guide跨高度接缝过滤后没有可显示三角面")

    used_vertices = np.zeros(quarter_height.shape, dtype=bool)
    coarse_vertices = used_vertices[::4, ::4]
    coarse_vertices[:-1, :-1] |= coarse_visible
    coarse_vertices[:-1, 1:] |= coarse_visible
    coarse_vertices[1:, :-1] |= coarse_visible
    coarse_vertices[1:, 1:] |= coarse_visible
    fine_vertices = used_vertices[::2, ::2]
    fine_vertices[:-1, :-1] |= fine_visible
    fine_vertices[:-1, 1:] |= fine_visible
    fine_vertices[1:, :-1] |= fine_visible
    fine_vertices[1:, 1:] |= fine_visible
    used_vertices[:-1, :-1] |= quarter_visible
    used_vertices[:-1, 1:] |= quarter_visible
    used_vertices[1:, :-1] |= quarter_visible
    used_vertices[1:, 1:] |= quarter_visible
    vertex_indices = np.zeros(quarter_height.shape, dtype=np.int64)
    used_rows, used_columns = np.nonzero(used_vertices)
    vertex_indices[used_rows, used_columns] = np.arange(1, len(used_rows) + 1)

    lines = ["o rmuc2026_surface_guide"]
    lines.extend(
        f"v {quarter_x[column]:.9g} {quarter_y[row]:.9g} "
        f"{quarter_height[row, column] + SURFACE_GUIDE_CLEARANCE_M:.9g}"
        for row, column in zip(used_rows, used_columns)
    )
    span = bounds[1] - bounds[0]
    lines.extend(
        f"vt {float(np.clip((quarter_x[column] - bounds[0, 0]) / span[0], 0.0, 1.0)):.9g} "
        f"{float(np.clip((quarter_y[row] - bounds[0, 1]) / span[1], 0.0, 1.0)):.9g}"
        for row, column in zip(used_rows, used_columns)
    )

    def append_cell(row: int, column: int, stride: int) -> None:
        lower_left = int(vertex_indices[row, column])
        lower_right = int(vertex_indices[row, column + stride])
        upper_left = int(vertex_indices[row + stride, column])
        upper_right = int(vertex_indices[row + stride, column + stride])
        lines.append(
            f"f {lower_left}/{lower_left} {lower_right}/{lower_right} {upper_right}/{upper_right}"
        )
        lines.append(
            f"f {lower_left}/{lower_left} {upper_right}/{upper_right} {upper_left}/{upper_left}"
        )

    for row, column in zip(*np.nonzero(coarse_visible)):
        append_cell(4 * int(row), 4 * int(column), 4)
    for row, column in zip(*np.nonzero(fine_visible)):
        append_cell(2 * int(row), 2 * int(column), 2)
    for row, column in zip(*np.nonzero(quarter_visible)):
        append_cell(int(row), int(column), 1)
    coarse_visible_count = int(np.count_nonzero(coarse_visible))
    fine_visible_count = int(np.count_nonzero(fine_visible))
    quarter_visible_count = int(np.count_nonzero(quarter_visible))
    face_count = 2 * (coarse_visible_count + fine_visible_count + quarter_visible_count)
    cell_count = (mesh_rows - 1) * (mesh_columns - 1)
    refined_area = fine_visible_count / 4.0
    quarter_area = quarter_visible_count / 16.0
    visible_fraction = (coarse_visible_count + refined_area + quarter_area) / cell_count
    refined_per_coarse = fine_visible.reshape(mesh_rows - 1, 2, mesh_columns - 1, 2).sum(
        axis=(1, 3)
    )
    quarter_per_coarse = quarter_visible.reshape(mesh_rows - 1, 4, mesh_columns - 1, 4).sum(
        axis=(1, 3)
    )
    fully_omitted_cells = int(
        np.count_nonzero(~coarse_visible & (refined_per_coarse == 0) & (quarter_per_coarse == 0))
    )
    retained_minimum = min(
        float(np.min(minimum_error[coarse_visible] + SURFACE_GUIDE_CLEARANCE_M))
        if coarse_visible_count
        else math.inf,
        float(np.min(fine_minimum_error[fine_visible] + SURFACE_GUIDE_CLEARANCE_M))
        if fine_visible_count
        else math.inf,
        quarter_retained_minimum,
    )
    retained_maximum = max(
        float(np.max(maximum_error[coarse_visible] + SURFACE_GUIDE_CLEARANCE_M))
        if coarse_visible_count
        else -math.inf,
        float(np.max(fine_maximum_error[fine_visible] + SURFACE_GUIDE_CLEARANCE_M))
        if fine_visible_count
        else -math.inf,
        quarter_retained_maximum,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "method": "full_rulebook_surface_adaptive_5cm_2p5cm_1p25cm_signed_height_bounds",
        "target_spacing_m": SURFACE_GUIDE_SAMPLING_M,
        "refined_spacing_m": SURFACE_GUIDE_SAMPLING_M / 2.0,
        "local_spacing_m": SURFACE_GUIDE_SAMPLING_M / 4.0,
        "actual_max_spacing_xy_m": [
            float(np.max(np.diff(selected_x))),
            float(np.max(np.diff(selected_y))),
        ],
        "clearance_m": SURFACE_GUIDE_CLEARANCE_M,
        "minimum_above_terrain_m": SURFACE_GUIDE_MINIMUM_ABOVE_TERRAIN_M,
        "maximum_above_terrain_m": SURFACE_GUIDE_MAXIMUM_ABOVE_TERRAIN_M,
        "retained_sampled_minimum_above_terrain_m": retained_minimum,
        "retained_sampled_maximum_above_terrain_m": retained_maximum,
        "maximum_cell_height_delta_m": SURFACE_GUIDE_MAXIMUM_CELL_HEIGHT_DELTA_M,
        "maximum_surface_slope": SURFACE_GUIDE_MAXIMUM_SURFACE_SLOPE,
        "vertices": int(np.count_nonzero(used_vertices)),
        "faces": face_count,
        "omitted_cells": fully_omitted_cells,
        "cell_count": cell_count,
        "coarse_visible_cells": coarse_visible_count,
        "refined_visible_subcells": fine_visible_count,
        "local_visible_subcells": quarter_visible_count,
        "visible_area_fraction": visible_fraction,
        "height_discontinuity_filtered_cells": int(np.count_nonzero(~smooth_cells)),
        "interior_undercut_filtered_cells": int(np.count_nonzero(smooth_cells & below_terrain)),
        "interior_raised_bridge_filtered_cells": int(
            np.count_nonzero(smooth_cells & raised_bridge)
        ),
        "mesh_size_bytes": path.stat().st_size,
        "boundary_xy_height_interpolation": True,
        "source_texture_alpha_drives_topology": False,
    }


def _load_source_contract(field_build: Path) -> tuple[Path, Path, dict[str, Any]]:
    field_root = field_build.expanduser().resolve()
    manifest_path = field_root / "manifest.json"
    if not manifest_path.is_file():
        raise ExportBlocked(f"缺少源manifest：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportBlocked(f"源manifest无法解析：{exc}") from exc
    manifest = _json_object(manifest, "源manifest")
    if manifest.get("artifact_type") != EXPECTED_ARTIFACT_TYPE:
        raise ExportBlocked("源产物类型不是RMUC 2026官方场地MuJoCo资产")
    if manifest.get("status") != "PASS":
        raise ExportBlocked("源场地构建完整性状态不是PASS")
    scope = _json_object(manifest.get("validation_scope"), "validation_scope")
    if scope.get("status") != "DRAFT_BLOCKED":
        raise ExportBlocked("本导出器要求显式保留当前DRAFT_BLOCKED验证边界")
    visual = manifest.get("visual_meshes")
    if not isinstance(visual, list) or len(visual) != EXPECTED_VISUAL_MESH_COUNT:
        raise ExportBlocked(
            f"视觉网格必须恰好为{EXPECTED_VISUAL_MESH_COUNT}个，实际={len(visual) if isinstance(visual, list) else 'invalid'}"
        )
    return field_root, manifest_path, manifest


def _build_field_xml(
    manifest: dict[str, Any],
    *,
    include_visual_meshes: bool = True,
    livery: dict[str, str] | None = None,
) -> bytes:
    visual = manifest["visual_meshes"]
    collision = _json_object(manifest.get("collision"), "collision")
    half_size = _finite_numbers(collision.get("half_size_xy_m"), count=2, label="碰撞半尺寸")
    center = _finite_numbers(
        collision.get("geom_center_after_translation_m"), count=3, label="碰撞中心"
    )
    maximum_height = float(collision.get("maximum_height_m", math.nan))
    base_depth = float(collision.get("base_depth_m", math.nan))
    if not math.isfinite(maximum_height) or maximum_height <= 0.0:
        raise ExportBlocked("maximum_height_m必须是有限正数")
    if not math.isfinite(base_depth) or base_depth < 0.0:
        raise ExportBlocked("base_depth_m必须是有限非负数")

    model_name = (
        "rmuc2026_field_runtime_asset_pack"
        if include_visual_meshes
        else "rmuc2026_field_collision_only"
    )
    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.002", "solver": "Newton"})
    visual_config = ET.SubElement(root, "visual")
    ET.SubElement(
        visual_config,
        "headlight",
        {
            "ambient": "0.18 0.18 0.18",
            "diffuse": "0.40 0.40 0.40",
            "specular": "0.05 0.05 0.05",
        },
    )
    ET.SubElement(visual_config, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(visual_config, "quality", {"offsamples": "4"})
    asset = ET.SubElement(root, "asset")
    if include_visual_meshes:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "rmuc2026_sky",
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.16 0.22 0.30",
                "rgb2": "0.025 0.035 0.055",
                "width": "512",
                "height": "3072",
            },
        )
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "rmuc2026_collision",
            "file": Path(str(collision["image_file"])).as_posix(),
            "nrow": str(int(collision["rows_y"])),
            "ncol": str(int(collision["columns_x"])),
            "size": " ".join(f"{value:.9g}" for value in (*half_size, maximum_height, base_depth)),
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": KEY_LIGHT_NAME,
            "directional": "true",
            "castshadow": "false",
            "pos": "0 0 12",
            "dir": "-0.25 -0.35 -1",
            "diffuse": "0.55 0.53 0.50",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": FILL_LIGHT_NAME,
            "directional": "true",
            "castshadow": "false",
            "pos": "0 0 8",
            "dir": "0.45 0.20 -1",
            "diffuse": "0.10 0.12 0.15",
        },
    )
    if include_visual_meshes:
        for index, record_value in enumerate(visual):
            record = _json_object(record_value, f"visual_meshes[{index}]")
            rgba = _finite_numbers(
                record.get("rgba"), count=4, label=f"visual_meshes[{index}].rgba"
            )
            if not all(0.0 <= value <= 1.0 for value in rgba):
                raise ExportBlocked(f"visual_meshes[{index}].rgba越界")
            visual_role = str(record.get("visual_role", "cad_structure"))
            display_rgba = _display_rgba(rgba, visual_role=visual_role)
            material = f"rmuc2026_material_{index}"
            mesh = f"rmuc2026_visual_mesh_{index}"
            ET.SubElement(
                asset,
                "material",
                {
                    "name": material,
                    "rgba": " ".join(f"{value:.8g}" for value in display_rgba),
                    "specular": "0.04",
                    "shininess": "0.18",
                },
            )
            ET.SubElement(
                asset,
                "mesh",
                {"name": mesh, "file": Path(str(record["file"])).as_posix()},
            )
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": f"rmuc2026_visual_{index}",
                    "type": "mesh",
                    "mesh": mesh,
                    "material": material,
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "2" if visual_role == "base_surface_shell" else "1",
                },
            )
        if livery is not None:
            ET.SubElement(
                asset,
                "texture",
                {
                    "name": "rmuc2026_surface_guide_texture",
                    "type": "2d",
                    "file": livery["texture_file"],
                },
            )
            ET.SubElement(
                asset,
                "material",
                {
                    "name": "rmuc2026_surface_guide_material",
                    "texture": "rmuc2026_surface_guide_texture",
                    "texrepeat": "1 1",
                    "texuniform": "false",
                    "rgba": "1 1 1 1",
                    "specular": "0",
                    "shininess": "0",
                    "emission": "0.08",
                },
            )
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": "rmuc2026_surface_guide_mesh",
                    "file": livery["mesh_file"],
                },
            )
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": SURFACE_GUIDE_GEOM_NAME,
                    "type": "mesh",
                    "mesh": "rmuc2026_surface_guide_mesh",
                    "material": "rmuc2026_surface_guide_material",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": str(SURFACE_GUIDE_GEOM_GROUP),
                },
            )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "rmuc2026_field_collision",
            "type": "hfield",
            "hfield": "rmuc2026_collision",
            "pos": " ".join(f"{value:.9g}" for value in center),
            "rgba": "0.10 0.38 0.62 0" if include_visual_meshes else "0.16 0.42 0.68 1",
            "contype": "2",
            "conaffinity": "1",
            "friction": "1 0.005 0.0001",
            "solref": "0.02 1",
            "group": "3" if include_visual_meshes else "0",
        },
    )
    dimensions = _json_object(manifest.get("dimensions"), "dimensions")
    outer_bounds = dimensions.get("cad_assembly_outer_bounds_after_translation_m")
    if (
        not isinstance(outer_bounds, list)
        or len(outer_bounds) != 2
        or any(not isinstance(row, list) or len(row) != 3 for row in outer_bounds)
    ):
        raise ExportBlocked("dimensions缺少CAD外包围")
    low = _finite_numbers(outer_bounds[0], count=3, label="CAD外包围low")
    high = _finite_numbers(outer_bounds[1], count=3, label="CAD外包围high")
    if any(hi <= lo for lo, hi in zip(low, high)):
        raise ExportBlocked("CAD外包围无效")
    center_xy = [0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])]
    extent_xy = [high[0] - low[0], high[1] - low[1]]
    distance = (
        1.08
        * max(extent_xy[1], extent_xy[0] / (1280.0 / 720.0))
        / (2.0 * math.tan(math.radians(55.0 / 2.0)))
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "rmuc2026_overview",
            "mode": "fixed",
            "pos": f"{center_xy[0]:.9g} {center_xy[1]:.9g} {high[2] + distance:.9g}",
            "xyaxes": "1 0 0 0 1 0",
            "fovy": "55",
        },
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _file_record(root: Path, relative: str, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "file": relative,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def export_runtime_asset_pack(
    field_build: Path,
    output_dir: Path,
    *,
    include_surface_guide: bool = False,
) -> dict[str, Any]:
    field_root, source_manifest_path, source_manifest = _load_source_contract(field_build)
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise ExportBlocked(f"输出目录已存在，拒绝覆盖：{output}")
    if output == field_root or field_root in output.parents:
        raise ExportBlocked("输出目录不能位于源场地产物内部")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        visual_records: list[dict[str, object]] = []
        compact_visual_meshes: list[dict[str, object]] = []
        for index, record_value in enumerate(source_manifest["visual_meshes"]):
            record = _json_object(record_value, f"visual_meshes[{index}]")
            source_rgba = _finite_numbers(
                record.get("rgba"),
                count=4,
                label=f"visual_meshes[{index}].rgba",
            )
            visual_role = str(record.get("visual_role", "cad_structure"))
            relative, source, _digest = _source_file(
                field_root,
                record.get("file"),
                record.get("sha256"),
                suffix=".obj",
            )
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            visual_records.append(_file_record(staging, relative, "cad_visual_mesh"))
            compact_visual_meshes.append(
                {
                    "file": relative,
                    "sha256": str(record["sha256"]),
                    "material_id": str(record.get("material_id", f"mesh_{index}")),
                    "rgba": source_rgba,
                    "display_rgba": _display_rgba(source_rgba, visual_role=visual_role),
                    "visual_role": visual_role,
                }
            )

        collision = _json_object(source_manifest.get("collision"), "collision")
        copied_collision: list[tuple[str, str]] = []
        collision_sources: dict[str, Path] = {}
        for file_key, hash_key, suffix, role in (
            ("image_file", "image_sha256", ".png", "heightfield_bootstrap_png"),
            ("samples_file", "samples_sha256", ".npz", "heightfield_float_samples"),
        ):
            relative, source, _digest = _source_file(
                field_root,
                collision.get(file_key),
                collision.get(hash_key),
                suffix=suffix,
            )
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied_collision.append((relative, role))
            collision_sources[file_key] = source

        livery_xml: dict[str, str] | None = None
        livery_manifest: dict[str, object] | None = None
        livery_files: list[dict[str, object]] = []
        if include_surface_guide:
            source_surface = _source_surface_guide(field_root, source_manifest)
            texture_destination = staging / SURFACE_GUIDE_TEXTURE_FILE
            texture_destination.parent.mkdir(parents=True, exist_ok=True)
            world_bounds = np.asarray(source_surface["world_bounds_xy_m"], dtype=np.float64)
            # Preserve the complete image and its source hash in the field
            # build, but hide only the connected white PDF margin in the local
            # runtime texture. The RGB bleed avoids transparent white fringes.
            try:
                edge_processing = mask_rulebook_page_edge(
                    source_surface["source_path"], texture_destination
                )
            except LiveryProcessingError as exc:
                raise ExportBlocked(f"surface_guide页边处理失败：{exc}") from exc
            mesh_destination = staging / SURFACE_GUIDE_MESH_FILE
            recommended_spawn = _json_object(
                source_manifest.get("recommended_spawn"), "recommended_spawn"
            )
            _write_surface_guide_mesh(
                mesh_destination,
                samples_path=collision_sources["samples_file"],
                collision=collision,
                recommended_spawn=recommended_spawn,
                world_bounds_xy_m=world_bounds,
            )
            livery_xml = {
                "mesh_file": SURFACE_GUIDE_MESH_FILE,
                "texture_file": SURFACE_GUIDE_TEXTURE_FILE,
            }
            livery_manifest = {
                "kind": SURFACE_GUIDE_KIND,
                "geom_name": SURFACE_GUIDE_GEOM_NAME,
                "geom_group": SURFACE_GUIDE_GEOM_GROUP,
                "default_visible": False,
                "toggle_key": "G",
                "physics": False,
                "mesh_file": SURFACE_GUIDE_MESH_FILE,
                "mesh_sha256": sha256_file(mesh_destination),
                "texture_file": SURFACE_GUIDE_TEXTURE_FILE,
                "texture_sha256": sha256_file(texture_destination),
                "source_texture_sha256": source_surface["source_sha256"],
                "edge_processing": edge_processing,
                "contains_baked_scene_content": list(SURFACE_GUIDE_BAKED_CONTENT),
            }
            livery_files.extend(
                (
                    _file_record(staging, SURFACE_GUIDE_MESH_FILE, "surface_guide_visual_mesh"),
                    _file_record(
                        staging,
                        SURFACE_GUIDE_TEXTURE_FILE,
                        "official_rulebook_surface_texture",
                    ),
                )
            )

        xml_path = staging / OUTPUT_XML
        xml_path.write_bytes(
            _build_field_xml(
                source_manifest,
                include_visual_meshes=True,
                livery=livery_xml,
            )
        )
        collision_xml_path = staging / OUTPUT_COLLISION_ONLY_XML
        collision_xml_path.write_bytes(
            _build_field_xml(source_manifest, include_visual_meshes=False)
        )
        files = [
            _file_record(staging, OUTPUT_XML, "field_mjcf_full"),
            _file_record(staging, OUTPUT_COLLISION_ONLY_XML, "field_mjcf_collision_only"),
        ]
        files.extend(visual_records)
        files.extend(_file_record(staging, relative, role) for relative, role in copied_collision)
        files.extend(livery_files)

        source_identity = _json_object(source_manifest.get("source"), "source")
        scope = _json_object(source_manifest.get("validation_scope"), "validation_scope")
        recommended_spawn = _json_object(
            source_manifest.get("recommended_spawn"), "recommended_spawn"
        )
        dimensions = _json_object(source_manifest.get("dimensions"), "dimensions")
        compact_collision: dict[str, Any] = {
            "kind": collision.get("kind"),
            "image_file": str(collision["image_file"]),
            "image_sha256": str(collision["image_sha256"]),
            "samples_file": str(collision["samples_file"]),
            "samples_sha256": str(collision["samples_sha256"]),
            "rows_y": int(collision["rows_y"]),
            "columns_x": int(collision["columns_x"]),
            "half_size_xy_m": _finite_numbers(
                collision.get("half_size_xy_m"), count=2, label="collision.half_size_xy_m"
            ),
            "geom_center_after_translation_m": _finite_numbers(
                collision.get("geom_center_after_translation_m"),
                count=3,
                label="collision.geom_center_after_translation_m",
            ),
            "maximum_height_m": float(collision["maximum_height_m"]),
            "minimum_height_m": float(collision.get("minimum_height_m", 0.0)),
            "base_depth_m": float(collision["base_depth_m"]),
            "resolution_m": float(collision.get("resolution_m", math.nan)),
            "png_rows": collision.get("png_rows"),
            "ray_misses_filled_with_ground": _source_count(
                collision,
                "ray_misses_filled_with_ground",
                label="collision",
            ),
            "isolated_spikes_replaced": _source_count(
                collision,
                "isolated_spikes_replaced",
                label="collision",
            ),
            "claim_boundary": collision.get("claim_boundary"),
        }
        structural_audit = collision.get("structural_audit")
        if structural_audit is not None:
            compact_collision["structural_audit"] = _json_safe_copy(
                _json_object(structural_audit, "collision.structural_audit"),
                "collision.structural_audit",
            )
        fixed_ramp_audit = collision.get("fixed_fly_ramp_audit")
        if fixed_ramp_audit is not None:
            compact_collision["fixed_fly_ramp_audit"] = _json_safe_copy(
                _json_object(fixed_ramp_audit, "collision.fixed_fly_ramp_audit"),
                "collision.fixed_fly_ramp_audit",
            )
        wall_tip_repair = collision.get("verified_wall_tip_repair")
        if wall_tip_repair is not None:
            try:
                validate_wall_tip_repair_record(
                    wall_tip_repair, collision_samples_sha256=str(collision["samples_sha256"])
                )
            except ValueError as exc:
                raise ExportBlocked(f"官方墙端采样修复记录无效：{exc}") from exc
            compact_collision["verified_wall_tip_repair"] = _json_safe_copy(
                wall_tip_repair, "collision.verified_wall_tip_repair"
            )
        compact_manifest: dict[str, Any] = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "artifact_type": OUTPUT_ARTIFACT_TYPE,
            "status": "PASS",
            "validation_status": "DRAFT_BLOCKED",
            "source_identity": {
                "field_manifest_sha256": sha256_file(source_manifest_path),
                "field_artifact_type": source_manifest["artifact_type"],
                "field_build_status": source_manifest["status"],
                "official_step_sha256": source_identity.get("sha256"),
                "official_step_size_bytes": source_identity.get("size_bytes"),
                "official_step_product": source_identity.get("step_product"),
                "official_download_url": source_identity.get("official_download_url"),
                "license_status": "UNSPECIFIED",
                "redistribution_authorized": False,
                "asset_included_in_source_repository": False,
            },
            "visual_meshes": compact_visual_meshes,
            "visual_display": {
                "profile": PUBLIC_DISPLAY_PROFILE,
                "source_rgba_preserved_in_visual_meshes_rgba": True,
                "display_rgba_recorded_per_visual_mesh": True,
                "black_floor": PUBLIC_DISPLAY_RGB_BLACK_FLOOR,
                "white_ceiling": PUBLIC_DISPLAY_RGB_WHITE_CEILING,
                "gamma": PUBLIC_DISPLAY_RGB_GAMMA,
                "base_surface_scale": PUBLIC_DISPLAY_BASE_SURFACE_SCALE,
                "primary_light_casts_shadow": False,
                "physics_changed": False,
                "lighting": {
                    "key_light_name": KEY_LIGHT_NAME,
                    "fill_light_name": FILL_LIGHT_NAME,
                    "default_mode": "flat",
                    "modes": ["flat", "shadow"],
                    "toggle_key": "L",
                    "physics_changed": False,
                },
            },
            "visual_layers": ({"livery": livery_manifest} if livery_manifest else {}),
            "collision": compact_collision,
            "coordinate_frame": {
                "world_units": "metre-radian-kilogram-second",
                "z_up": True,
                "recommended_spawn": recommended_spawn,
                "npz_axes_before_world_translation": True,
                "world_x_m": "x_m - recommended_spawn.x_before_translation_m",
                "world_y_m": "y_m - recommended_spawn.y_before_translation_m",
                "world_z_m": "height_m - recommended_spawn.terrain_height_m",
            },
            "dimensions": {
                "official_core_battlefield_m": dimensions.get("official_core_battlefield_m"),
                "cad_assembly_outer_bounds_after_translation_m": dimensions.get(
                    "cad_assembly_outer_bounds_after_translation_m"
                ),
                "cad_assembly_outer_extents_m": dimensions.get("cad_assembly_outer_extents_m"),
            },
            "contents": {
                "entrypoint": OUTPUT_XML,
                "visual_obj_count": len(visual_records),
                "file_count_excluding_manifest": len(files),
                "files": files,
            },
            "runtime_profiles": {
                "default": "full",
                "profiles": {
                    "full": {
                        "entrypoint": OUTPUT_XML,
                        "includes_visual_meshes": True,
                        "visual_mesh_count": len(visual_records),
                        "intended_use": "interactive_visualization",
                    },
                    "collision_only": {
                        "entrypoint": OUTPUT_COLLISION_ONLY_XML,
                        "includes_visual_meshes": False,
                        "visual_mesh_count": 0,
                        "intended_use": "headless_physics",
                    },
                },
            },
            "heightfield_precision": {
                "name": "rmuc2026_collision",
                "png_file": str(collision["image_file"]),
                "float_samples_file": str(collision["samples_file"]),
                "rows_y": int(collision["rows_y"]),
                "columns_x": int(collision["columns_x"]),
                "maximum_height_m": float(collision["maximum_height_m"]),
                "npz_array": "height_m",
                "normalized_model_values": "clip(height_m / maximum_height_m, 0, 1)",
                "float_npz_injection_required_before_validated_physics": True,
                "png_role": "MuJoCo dimensions/bootstrap only",
            },
            "portability": {
                "all_mjcf_file_references_are_relative": True,
                "source_tree_required_after_export": False,
                "field_model_only": True,
                "robot_and_policy_included": False,
            },
            "excluded": {
                "colored_intermediate_glb": True,
                "official_rulebook_screenshot": not include_surface_guide,
                "rulebook_derived_ground_marking_overlay": True,
                "source_step": True,
                "telemetry_or_video": True,
            },
            "distribution": {
                "generated_locally": True,
                "third_party_geometry": True,
                "safe_to_publish_without_rightsholder_permission": False,
                "code_license_applies_to_asset_pack": False,
            },
            "validation_boundary": {
                "status": "DRAFT_BLOCKED",
                "profile_id": scope.get("profile_id"),
                "claim_boundary": scope.get("claim_boundary"),
                "blocking_reasons": scope.get("blocking_reasons"),
                "whole_field_topology_ready": scope.get("whole_field_topology_ready"),
                "final_policy_validation_ready": scope.get("final_policy_validation_ready"),
                "pack_pass_means": "file integrity, relative-path loadability, and preserved source identity only",
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(compact_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
        return compact_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _preview(field_build: Path, output_dir: Path) -> dict[str, object]:
    field_root, manifest_path, manifest = _load_source_contract(field_build)
    collision = _json_object(manifest.get("collision"), "collision")
    total_source_bytes = 0
    for index, record_value in enumerate(manifest["visual_meshes"]):
        record = _json_object(record_value, f"visual_meshes[{index}]")
        _relative, source, _digest = _source_file(
            field_root, record.get("file"), record.get("sha256"), suffix=".obj"
        )
        total_source_bytes += source.stat().st_size
    for file_key, hash_key, suffix in (
        ("image_file", "image_sha256", ".png"),
        ("samples_file", "samples_sha256", ".npz"),
    ):
        _relative, source, _digest = _source_file(
            field_root,
            collision.get(file_key),
            collision.get(hash_key),
            suffix=suffix,
        )
        total_source_bytes += source.stat().st_size
    return {
        "mode": "PREVIEW_ONLY",
        "source_manifest_sha256": sha256_file(manifest_path),
        "output_dir": str(output_dir.expanduser().resolve()),
        "visual_obj_count": len(manifest["visual_meshes"]),
        "payload_bytes_before_xml_and_compact_manifest": total_source_bytes,
        "validation_status": "DRAFT_BLOCKED",
        "excluded": ["47 MB colored GLB", "official rulebook screenshot", "source STEP"],
        "writes_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出可搬运的RMUC 2026 MuJoCo场地资产包")
    parser.add_argument("--field-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-surface-guide", action="store_true")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.preview:
            print(
                json.dumps(
                    _preview(args.field_build, args.output_dir), ensure_ascii=False, indent=2
                )
            )
            return 0
        result = export_runtime_asset_pack(
            args.field_build,
            args.output_dir,
            include_surface_guide=args.include_surface_guide,
        )
        print("RMUC2026_MUJOCO_ASSET_EXPORT=PASS")
        print(f"完成：{args.output_dir.expanduser().resolve()}")
        print(
            "边界：导出PASS仅证明相对路径与文件身份；场地验证仍为"
            f"{result['validation_status']}，不代表完整碰撞或策略通过。"
        )
        return 0
    except (ExportBlocked, OSError, ValueError, KeyError) as exc:
        print(f"RMUC2026_MUJOCO_ASSET_EXPORT=BLOCKED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
