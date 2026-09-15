from __future__ import annotations

from pathlib import Path

import pytest

from rmuc2026_mujoco import FieldAsset, OutOfBoundsError, field_bounds, height_at


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
