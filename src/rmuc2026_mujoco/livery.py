"""Livery contracts, including read compatibility for filtered alpha packs.

Current runtime-pack export preserves the complete rulebook overhead render.
The filtered-marking implementation remains here so already generated schema-2
packs can still be validated and reproduced when explicitly needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


SOURCE_SURFACE_GUIDE_KIND = "official_rulebook_overhead_render_surface_guide"
GROUND_MARKING_OVERLAY_KIND = "rulebook_derived_ground_marking_overlay"
GROUND_MARKING_OVERLAY_TEXTURE_FILE = "visual/rmuc2026_ground_marking_overlay.png"
GROUND_MARKING_PROCESSING_ALGORITHM = "chromatic_component_filter_v1"
SOURCE_BAKED_SCENE_CONTENT = (
    "fixed field-module appearance",
    "obstacle appearance",
    "robots",
    "shadows",
)
GROUND_MARKING_RETAINED_CONTENT = (
    "red ground markings",
    "blue ground markings",
    "orange ground markings",
)
GROUND_MARKING_REMOVED_BY_DESIGN = (
    "neutral floor texture",
    "grayscale fixed field-module appearance",
    "grayscale obstacle appearance",
    "grayscale robots",
    "grayscale shadows",
)
GROUND_MARKING_KNOWN_LIMITATIONS = (
    "chromatic non-ground details can remain when large enough or connected to a retained marking",
    "low-saturation and white ground markings are deliberately omitted",
    "the rulebook image uses an outer-bounds fit rather than a surveyed homography",
)
GROUND_MARKING_MINIMUM_COMPONENT_SPAN_M = 0.45
GROUND_MARKING_MINIMUM_COMPONENT_AREA_M2 = 0.002
GROUND_MARKING_RGB_BLEED_RADIUS_PX = 4


class LiveryProcessingError(ValueError):
    """The local rulebook image cannot produce a bounded markings overlay."""


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel 8-connected dilation without an image-processing dependency."""

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    rows, columns = mask.shape
    for row_offset in range(3):
        for column_offset in range(3):
            result |= padded[
                row_offset : row_offset + rows,
                column_offset : column_offset + columns,
            ]
    return result


def _retained_components(
    seed: np.ndarray,
    *,
    pixel_size_xy_m: tuple[float, float],
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep only physically long chromatic components from the overhead render."""

    connected = _dilate(seed)
    visited = np.zeros_like(connected, dtype=bool)
    retained = np.zeros_like(connected, dtype=bool)
    rows, columns = connected.shape
    pixel_area_m2 = pixel_size_xy_m[0] * pixel_size_xy_m[1]
    components = 0
    retained_components = 0

    for start_row, start_column in np.argwhere(connected):
        row = int(start_row)
        column = int(start_column)
        if visited[row, column]:
            continue
        components += 1
        visited[row, column] = True
        stack = [(row, column)]
        members: list[tuple[int, int]] = []
        minimum_row = maximum_row = row
        minimum_column = maximum_column = column
        while stack:
            current_row, current_column = stack.pop()
            members.append((current_row, current_column))
            minimum_row = min(minimum_row, current_row)
            maximum_row = max(maximum_row, current_row)
            minimum_column = min(minimum_column, current_column)
            maximum_column = max(maximum_column, current_column)
            for row_offset in (-1, 0, 1):
                neighbour_row = current_row + row_offset
                if neighbour_row < 0 or neighbour_row >= rows:
                    continue
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbour_column = current_column + column_offset
                    if neighbour_column < 0 or neighbour_column >= columns:
                        continue
                    if (
                        connected[neighbour_row, neighbour_column]
                        and not visited[neighbour_row, neighbour_column]
                    ):
                        visited[neighbour_row, neighbour_column] = True
                        stack.append((neighbour_row, neighbour_column))

        span_x_m = (maximum_column - minimum_column + 1) * pixel_size_xy_m[0]
        span_y_m = (maximum_row - minimum_row + 1) * pixel_size_xy_m[1]
        area_m2 = len(members) * pixel_area_m2
        if (
            max(span_x_m, span_y_m) >= GROUND_MARKING_MINIMUM_COMPONENT_SPAN_M
            and area_m2 >= GROUND_MARKING_MINIMUM_COMPONENT_AREA_M2
        ):
            retained_components += 1
            member_rows, member_columns = zip(*members)
            retained[np.asarray(member_rows), np.asarray(member_columns)] = True

    return retained, {
        "detected": components,
        "retained": retained_components,
        "rejected": components - retained_components,
    }


def extract_ground_marking_overlay(
    source: Path,
    destination: Path,
    *,
    world_size_xy_m: Sequence[float],
) -> dict[str, Any]:
    """Write a transparent RGBA layer containing only robust chromatic markings.

    The rulebook illustration has no semantic mask, so this deliberately rejects
    all neutral pixels.  A physical connected-component gate removes isolated
    coloured details without encoding source-image pixel dimensions as policy.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised by minimal installs
        raise LiveryProcessingError(
            "Pillow is required to extract the optional ground-marking overlay"
        ) from exc

    source_path = Path(source)
    destination_path = Path(destination)
    try:
        with Image.open(source_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise LiveryProcessingError(f"cannot read rulebook surface guide: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise LiveryProcessingError("rulebook surface guide must be a non-empty RGB image")
    if rgb.shape[0] * rgb.shape[1] > 40_000_000:
        raise LiveryProcessingError("rulebook surface guide exceeds the processing pixel limit")

    size = np.asarray(world_size_xy_m, dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise LiveryProcessingError("world_size_xy_m must contain two positive finite values")
    rows, columns = rgb.shape[:2]
    pixel_size = (float(size[0] / columns), float(size[1] / rows))

    channels = rgb.astype(np.int16)
    red, green, blue = (channels[:, :, index] for index in range(3))
    red_seed = (red >= 90) & (red - green >= 28) & (red - blue >= 24)
    blue_seed = (blue >= 75) & (blue - red >= 24) & (blue - green >= 14)
    orange_seed = (red >= 135) & (green >= 45) & (red - green >= 38) & (green - blue >= 12)
    seed = red_seed | blue_seed | orange_seed
    component_mask, component_counts = _retained_components(
        seed,
        pixel_size_xy_m=pixel_size,
    )

    red_support = (red >= 65) & (red - green >= 12) & (red - blue >= 10)
    blue_support = (blue >= 55) & (blue - red >= 11) & (blue - green >= 7)
    orange_support = (red >= 95) & (green >= 32) & (red - green >= 18) & (green - blue >= 5)
    neighbourhood = _dilate(component_mask)
    retained_red = neighbourhood & red_support & ~orange_support
    retained_blue = neighbourhood & blue_support
    retained_orange = neighbourhood & orange_support
    retained = retained_red | retained_blue | retained_orange
    retained_pixels = int(np.count_nonzero(retained))
    if retained_pixels == 0:
        raise LiveryProcessingError("rulebook surface guide has no retained ground markings")

    rgba = np.zeros((rows, columns, 4), dtype=np.uint8)
    rgba[retained_red, :3] = (224, 54, 61)
    rgba[retained_blue, :3] = (57, 76, 225)
    rgba[retained_orange, :3] = (241, 111, 35)
    rgba[retained, 3] = 255
    # MuJoCo uses the alpha mask to decide which sparse 5 cm cells receive
    # faces, but texture filtering can still sample RGB just outside an opaque
    # stroke. Bleed only RGB into nearby transparent texels to prevent a black
    # fringe without changing the topology-driving alpha mask.
    bleed_assigned = retained.copy()
    for class_mask, colour in (
        (retained_red, (224, 54, 61)),
        (retained_blue, (57, 76, 225)),
        (retained_orange, (241, 111, 35)),
    ):
        bleed = class_mask.copy()
        for _ in range(GROUND_MARKING_RGB_BLEED_RADIUS_PX):
            bleed = _dilate(bleed)
        target = bleed & ~bleed_assigned
        rgba[target, :3] = colour
        bleed_assigned |= target
    edge_bleed_pixels = int(np.count_nonzero(bleed_assigned & ~retained))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(destination_path, format="PNG", optimize=True)

    total_pixels = rows * columns
    return {
        "algorithm": GROUND_MARKING_PROCESSING_ALGORITHM,
        "output_mode": "rgba_with_transparent_non_markings",
        "retained_content": list(GROUND_MARKING_RETAINED_CONTENT),
        "removed_by_design": list(GROUND_MARKING_REMOVED_BY_DESIGN),
        "minimum_component_span_m": GROUND_MARKING_MINIMUM_COMPONENT_SPAN_M,
        "minimum_component_area_m2": GROUND_MARKING_MINIMUM_COMPONENT_AREA_M2,
        "input_size_px": [columns, rows],
        "input_pixels": total_pixels,
        "strong_candidate_pixels": int(np.count_nonzero(seed)),
        "retained_pixels": retained_pixels,
        "transparent_pixels": total_pixels - retained_pixels,
        "retained_fraction": retained_pixels / total_pixels,
        "transparent_rgb_edge_bleed_radius_px": GROUND_MARKING_RGB_BLEED_RADIUS_PX,
        "transparent_rgb_edge_bleed_pixels": edge_bleed_pixels,
        "components": component_counts,
    }


__all__ = [
    "GROUND_MARKING_KNOWN_LIMITATIONS",
    "GROUND_MARKING_OVERLAY_KIND",
    "GROUND_MARKING_OVERLAY_TEXTURE_FILE",
    "GROUND_MARKING_PROCESSING_ALGORITHM",
    "GROUND_MARKING_REMOVED_BY_DESIGN",
    "GROUND_MARKING_RETAINED_CONTENT",
    "GROUND_MARKING_RGB_BLEED_RADIUS_PX",
    "LiveryProcessingError",
    "SOURCE_BAKED_SCENE_CONTENT",
    "SOURCE_SURFACE_GUIDE_KIND",
    "extract_ground_marking_overlay",
]
