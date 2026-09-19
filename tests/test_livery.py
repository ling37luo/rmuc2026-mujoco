from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rmuc2026_mujoco.livery import (
    LiveryProcessingError,
    extract_ground_marking_overlay,
    mask_rulebook_page_edge,
)


Image = pytest.importorskip("PIL.Image")


def test_full_guide_hides_only_connected_page_margin_without_moving_pixels(tmp_path: Path) -> None:
    pixels = np.full((50, 100, 3), (40, 50, 60), dtype=np.uint8)
    pixels[:2] = 255
    pixels[-2:] = 255
    pixels[:, :2] = 255
    pixels[:, -2:] = 255
    pixels[20:25, 40:50] = (245, 245, 245)  # interior white field marking
    pixels[25:28, 1] = (40, 50, 60)
    pixels[25:28, 2] = (250, 250, 250)  # white island near, but not joined to, margin
    source = tmp_path / "source.png"
    output = tmp_path / "runtime.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    source_bytes = source.read_bytes()

    report = mask_rulebook_page_edge(source, output)

    with Image.open(output) as image:
        result = np.asarray(image)
    assert report["masked_pixels"] > 0
    assert result.shape == (50, 100, 4)
    assert result[0, 10, 3] == 0
    assert result[25, 2, 3] == 255
    assert result[22, 45, 3] == 255
    assert np.array_equal(result[2:-2, 3:-3, :3], pixels[2:-2, 3:-3])
    assert np.max(result[0, 10, :3]) < 100  # dark RGB bleed prevents white fringe
    assert source.read_bytes() == source_bytes


def test_compat_filtered_overlay_keeps_long_colours_and_rejects_scene_pixels(
    tmp_path: Path,
) -> None:
    source_pixels = np.full((50, 100, 3), 118, dtype=np.uint8)
    source_pixels[15:35, 42:72] = (64, 64, 64)  # broad neutral baked shadow
    source_pixels[8:10, 8:72] = (205, 42, 48)  # red ground line
    source_pixels[30:35, 15:36] = (35, 48, 180)  # blue ground block
    source_pixels[40:42, 58:88] = (230, 105, 28)  # orange ground marking
    source_pixels[4:6, 90:92] = (220, 35, 40)  # isolated robot-status detail
    source_pixels[20:22, 5:75] = (240, 240, 240)  # neutral line is conservative omission
    source = tmp_path / "source.png"
    destination = tmp_path / "overlay.png"
    Image.fromarray(source_pixels, mode="RGB").save(source)
    source_bytes = source.read_bytes()

    report = extract_ground_marking_overlay(
        source,
        destination,
        world_size_xy_m=(10.0, 5.0),
    )

    with Image.open(destination) as output:
        rgba = np.asarray(output.convert("RGBA"))
    assert tuple(rgba[8, 20]) == (224, 54, 61, 255)
    assert tuple(rgba[7, 20]) == (224, 54, 61, 0)  # transparent RGB bleed avoids dark filtering
    assert tuple(rgba[32, 20]) == (57, 76, 225, 255)
    assert tuple(rgba[40, 70]) == (241, 111, 35, 255)
    assert rgba[25, 50, 3] == 0  # shadow
    assert rgba[4, 90, 3] == 0  # small chromatic non-ground component
    assert rgba[20, 20, 3] == 0  # white/neutral content
    assert source.read_bytes() == source_bytes
    assert report["components"]["retained"] == 3
    assert report["components"]["rejected"] >= 1
    assert report["retained_pixels"] + report["transparent_pixels"] == 5_000
    assert report["retained_fraction"] < 0.1

    second = tmp_path / "overlay-second.png"
    second_report = extract_ground_marking_overlay(
        source,
        second,
        world_size_xy_m=(10.0, 5.0),
    )
    assert second.read_bytes() == destination.read_bytes()
    assert second_report == report


def test_compat_filtered_overlay_fails_closed_without_retained_markings(tmp_path: Path) -> None:
    source = tmp_path / "neutral.png"
    Image.fromarray(np.full((16, 32, 3), 100, dtype=np.uint8), mode="RGB").save(source)

    with pytest.raises(LiveryProcessingError, match="no retained ground markings"):
        extract_ground_marking_overlay(
            source,
            tmp_path / "output.png",
            world_size_xy_m=(3.2, 1.6),
        )
