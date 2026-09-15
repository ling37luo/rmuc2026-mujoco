from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rmuc2026_mujoco import (
    FieldAsset,
    ManifestError,
    OutOfBoundsError,
    field_bounds,
    find_spawn_candidates,
    height_at,
    surface_at,
)


def _replace_heightfield(
    root: Path,
    *,
    world_x: np.ndarray,
    world_y: np.ndarray,
    world_height: np.ndarray,
) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spawn = manifest["coordinate_frame"]["recommended_spawn"]
    samples_path = root / manifest["collision"]["samples_file"]
    np.savez_compressed(
        samples_path,
        x_m=world_x + float(spawn["x_before_translation_m"]),
        y_m=world_y + float(spawn["y_before_translation_m"]),
        height_m=world_height + float(spawn["terrain_height_m"]),
    )
    digest = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    collision = manifest["collision"]
    collision.update(
        {
            "rows_y": len(world_y),
            "columns_x": len(world_x),
            "minimum_height_m": float(np.min(world_height) + spawn["terrain_height_m"]),
            "maximum_height_m": float(np.max(world_height) + spawn["terrain_height_m"]),
            "samples_sha256": digest,
            "half_size_xy_m": [
                0.5 * float(world_x[-1] - world_x[0]),
                0.5 * float(world_y[-1] - world_y[0]),
            ],
            "geom_center_after_translation_m": [
                0.5 * float(world_x[-1] + world_x[0]),
                0.5 * float(world_y[-1] + world_y[0]),
                -float(spawn["terrain_height_m"]),
            ],
        }
    )
    if "resolution_m" in collision:
        collision["resolution_m"] = float(world_x[1] - world_x[0])
    manifest["heightfield_precision"]["rows_y"] = len(world_y)
    manifest["heightfield_precision"]["columns_x"] = len(world_x)

    size = [
        *collision["half_size_xy_m"],
        collision["maximum_height_m"],
        collision["base_depth_m"],
    ]
    for profile in manifest["runtime_profiles"]["profiles"].values():
        relative = profile["entrypoint"]
        xml_path = root / relative
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()
        hfield = xml_root.find("./asset/hfield[@name='rmuc2026_collision']")
        geom = xml_root.find(".//geom[@name='rmuc2026_field_collision']")
        assert hfield is not None and geom is not None
        hfield.set("nrow", str(len(world_y)))
        hfield.set("ncol", str(len(world_x)))
        hfield.set("size", " ".join(str(value) for value in size))
        geom.set(
            "pos",
            " ".join(str(value) for value in collision["geom_center_after_translation_m"]),
        )
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    for record in manifest["contents"]["files"]:
        path = root / record["file"]
        if record["file"] == collision["samples_file"] or path.suffix == ".xml":
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record["size_bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_world_bounds_include_manifest_translation(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)
    assert field_bounds(asset) == ((-1.0, -1.0), (2.0, 1.0))


def test_height_query_is_bilinear_and_world_translated(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)
    assert height_at(asset, -0.5, -0.5) == pytest.approx(0.25)
    assert height_at(asset, 2.0, 1.0) == pytest.approx(1.5)


def test_height_query_out_of_bounds_modes(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)
    with pytest.raises(OutOfBoundsError):
        height_at(asset, 20.0, 0.0)
    assert height_at(asset, 20.0, 0.0, out_of_bounds="clip") == pytest.approx(1.0)
    assert height_at(asset, 20.0, 0.0, out_of_bounds="nan") != height_at(
        asset, 20.0, 0.0, out_of_bounds="nan"
    )


def test_surface_query_reports_world_normal_slope_and_boundary(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)

    sample = surface_at(asset, -0.5, -0.5, window_radius_m=0.51)

    assert sample.height_m == pytest.approx(0.25)
    assert sample.normal_xyz == pytest.approx((-0.235702, -0.235702, 0.942809), abs=1e-6)
    assert sample.slope_deg == pytest.approx(19.471221, abs=1e-6)
    assert sample.maximum_window_slope_deg == pytest.approx(19.471221, abs=1e-6)
    assert sample.local_relief_upper_bound_m == pytest.approx(1.0)
    assert sample.grid_resolution_xy_m == pytest.approx((1.0, 1.0))
    report = sample.to_dict()
    assert report["topology_verified"] is False
    assert "underpasses" in report["claim_boundary"]


def test_surface_query_rejects_outside_and_invalid_window(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)

    with pytest.raises(OutOfBoundsError):
        surface_at(asset, 20.0, 0.0)
    with pytest.raises(ValueError, match="window_radius_m"):
        surface_at(asset, 0.0, 0.0, window_radius_m=-0.1)


def test_spawn_candidates_are_deterministic_separated_and_explicitly_limited(
    field_asset_dir: Path,
) -> None:
    asset = FieldAsset.open(field_asset_dir)

    candidates = find_spawn_candidates(
        asset,
        count=2,
        footprint_radius_m=0.0,
        boundary_margin_m=0.0,
        minimum_separation_m=1.0,
        max_slope_deg=90.0,
        max_relief_m=2.0,
        ground_height_range_m=None,
    )

    assert len(candidates) == 2
    assert (candidates[0].x_m, candidates[0].y_m) == (0.0, 0.0)
    assert candidates == find_spawn_candidates(
        asset,
        count=2,
        footprint_radius_m=0.0,
        boundary_margin_m=0.0,
        minimum_separation_m=1.0,
        max_slope_deg=90.0,
        max_relief_m=2.0,
        ground_height_range_m=None,
    )
    for index, candidate in enumerate(candidates):
        assert candidate.topology_verified is False
        assert "robot clearance" in candidate.to_dict()["claim_boundary"]
        for other in candidates[index + 1 :]:
            distance_squared = (candidate.x_m - other.x_m) ** 2 + (candidate.y_m - other.y_m) ** 2
            assert distance_squared >= 1.0**2


def test_spawn_candidates_fail_closed_when_constraints_reject_the_grid(
    field_asset_dir: Path,
) -> None:
    asset = FieldAsset.open(field_asset_dir)

    assert (
        find_spawn_candidates(
            asset,
            max_slope_deg=0.0,
            max_relief_m=0.0,
            ground_height_range_m=None,
        )
        == ()
    )
    with pytest.raises(ValueError, match="positive integer"):
        find_spawn_candidates(asset, count=0)
    with pytest.raises(ValueError, match="<= 4096"):
        find_spawn_candidates(asset, count=4097)
    with pytest.raises(ValueError, match="ordered"):
        find_spawn_candidates(asset, ground_height_range_m=(1.0, -1.0))


def test_spawn_footprint_rejects_candidates_whose_window_contains_a_bump(
    field_asset_dir: Path,
) -> None:
    axis = np.linspace(-0.5, 0.5, 11)
    height = np.zeros((11, 11), dtype=np.float64)
    height[5, 5] = 0.2
    _replace_heightfield(
        field_asset_dir,
        world_x=axis,
        world_y=axis,
        world_height=height,
    )
    asset = FieldAsset.open(field_asset_dir)

    candidates = find_spawn_candidates(
        asset,
        count=20,
        footprint_radius_m=0.11,
        boundary_margin_m=0.0,
        minimum_separation_m=0.0,
        max_slope_deg=90.0,
        max_relief_m=0.05,
        ground_height_range_m=(-0.01, 0.01),
    )

    assert len(candidates) == 20
    assert all(
        not (abs(candidate.grid_index_yx[0] - 5) <= 2 and abs(candidate.grid_index_yx[1] - 5) <= 2)
        for candidate in candidates
    )

    height_band_candidates = find_spawn_candidates(
        asset,
        count=20,
        footprint_radius_m=0.11,
        boundary_margin_m=0.0,
        minimum_separation_m=0.0,
        max_slope_deg=90.0,
        max_relief_m=0.3,
        ground_height_range_m=(-0.01, 0.01),
    )
    assert len(height_band_candidates) == 20
    assert all(
        not (abs(candidate.grid_index_yx[0] - 5) <= 2 and abs(candidate.grid_index_yx[1] - 5) <= 2)
        for candidate in height_band_candidates
    )


def test_mujoco_triangle_query_detects_slope_hidden_by_node_gradients(
    field_asset_dir: Path,
) -> None:
    axis = np.asarray([-0.02, 0.0, 0.02], dtype=np.float64)
    height = np.zeros((3, 3), dtype=np.float64)
    height[1, 1] = 0.003
    _replace_heightfield(
        field_asset_dir,
        world_x=axis,
        world_y=axis,
        world_height=height,
    )
    asset = FieldAsset.open(field_asset_dir)

    assert height_at(asset, -0.01, -0.01) == pytest.approx(0.00075)
    assert height_at(asset, -0.01, -0.01, interpolation="mujoco") == pytest.approx(0.0015)
    sample = surface_at(asset, -0.01, -0.01, window_radius_m=0.0)
    assert sample.height_m == pytest.approx(0.0015)
    assert sample.slope_deg == pytest.approx(8.530766, abs=1.0e-6)
    assert sample.maximum_window_slope_deg == pytest.approx(8.530766, abs=1.0e-6)


def test_heightfield_query_rejects_nonuniform_world_axis(field_asset_dir: Path) -> None:
    _replace_heightfield(
        field_asset_dir,
        world_x=np.asarray([-1.0, -0.2, 1.0, 2.0]),
        world_y=np.asarray([-1.0, 0.0, 1.0]),
        world_height=np.zeros((3, 4), dtype=np.float64),
    )
    asset = FieldAsset.open(field_asset_dir)

    with pytest.raises(ManifestError, match="uniformly aligned"):
        field_bounds(asset)


def test_spawn_rejects_extreme_finite_radius_without_overflow(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)

    assert find_spawn_candidates(asset, footprint_radius_m=1.0e308) == ()
