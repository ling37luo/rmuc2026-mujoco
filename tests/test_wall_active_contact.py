"""Fail-closed synthetic checks for the opt-in exact-wall contact candidate."""

from __future__ import annotations

import numpy as np
import pytest

from rmuc2026_mujoco.query import HeightFieldData
from rmuc2026_mujoco.wall_active_contact import _replace_wall_roof


def _fixture():
    trimesh = pytest.importorskip("trimesh")
    # The official walls sit a few centimetres below the surrounding floor at
    # their tips; burial is valid, but an air gap under the wall is not.
    wall = trimesh.creation.box(extents=(0.2, 0.3, 1.04))
    wall.apply_translation([0.0, 0.15, 0.48])
    x = np.arange(-0.2, 0.201, 0.01)
    y = np.arange(-0.1, 0.401, 0.01)
    height = np.zeros((len(y), len(x)))
    ix = np.flatnonzero((x >= wall.bounds[0, 0]) & (x <= wall.bounds[1, 0]))
    iy = np.flatnonzero((y >= wall.bounds[0, 1]) & (y <= wall.bounds[1, 1]))
    height[:, ix[-1] + 1 :] = 0.2
    height[np.ix_(iy, ix)] = 1.0
    return wall, HeightFieldData(x, y, height), ix, iy


def test_exact_roof_transfers_to_source_bottom_without_touching_outside() -> None:
    wall, field, ix, iy = _fixture()
    output = field.height_m.copy()
    report = _replace_wall_roof(wall, field, output, expected_nodes=len(ix) * len(iy))
    assert report["roof_nodes_transferred"] == len(ix) * len(iy)
    assert report["maximum_gap_beneath_source_wall_m"] < 1.0e-9
    assert report["maximum_source_wall_burial_beneath_floor_m"] == pytest.approx(0.04)
    np.testing.assert_array_equal(output[np.ix_(iy, ix)], 0.0)
    mask = np.ones(output.shape, dtype=bool)
    mask[np.ix_(iy, ix)] = False
    np.testing.assert_array_equal(output[mask], field.height_m[mask])
    np.testing.assert_array_equal(field.height_m[np.ix_(iy, ix)], 1.0)


def test_changed_roof_or_missing_source_bottom_rejects_without_mutation() -> None:
    wall, field, ix, iy = _fixture()
    field.height_m[iy[1], ix[1]] = 0.9
    output = field.height_m.copy()
    with pytest.raises(ValueError, match="exact source wall"):
        _replace_wall_roof(wall, field, output, expected_nodes=len(ix) * len(iy))
    np.testing.assert_array_equal(output, field.height_m)

    field.height_m[iy[1], ix[1]] = 1.0
    field.height_m[:, ix[0] - 2] = -0.1
    field.height_m[:, ix[-1] + 2] = -0.1
    output = field.height_m.copy()
    with pytest.raises(ValueError, match="gap beneath"):
        _replace_wall_roof(wall, field, output, expected_nodes=len(ix) * len(iy))
    np.testing.assert_array_equal(output, field.height_m)
