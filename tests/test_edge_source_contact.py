"""Static source-owned edge geometry checks; these do not approve robot traversal."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rmuc2026_mujoco import edge_source_contact
from rmuc2026_mujoco.query import HeightFieldData


def test_source_roof_prism_is_closed_and_transfers_only_owned_grid_nodes() -> None:
    trimesh = pytest.importorskip("trimesh")
    triangle = np.array(((0.0, 0.0, 1.0), (0.4, 0.0, 1.0), (0.0, 0.4, 1.0)))
    mesh = edge_source_contact._prism_from_roof_triangle(triangle)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(0.5 * 0.4 * 0.4 * 0.20006858)
    base = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    base.apply_translation((0.2, 0.2, 1.0 - 0.20006858 - 0.1))
    assert edge_source_contact._check_source_base(base, triangle) < 1.0e-9

    axis = np.arange(-0.1, 0.51, 0.01)
    heights = np.ones((len(axis), len(axis)))
    field = HeightFieldData(axis, axis, heights)
    updated = heights.copy()
    owned = np.zeros(heights.shape, dtype=bool)
    report = edge_source_contact._transfer_roof_nodes(field, updated, owned, triangle)
    assert report["transferred_roof_nodes"] > 500
    np.testing.assert_array_equal(updated[~owned], heights[~owned])
    np.testing.assert_allclose(updated[owned], 1.0 - 0.20006858)
    with pytest.raises(ValueError, match="same heightfield node"):
        edge_source_contact._transfer_roof_nodes(field, updated, owned, triangle)


def test_schema_two_cannot_encode_official_negative_edge_base(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        edge_source_contact.FieldAsset,
        "open",
        lambda _path: SimpleNamespace(collision={"minimum_height_m": 0.0}),
    )
    with pytest.raises(ValueError, match="negative-capable"):
        edge_source_contact.build_source_edge_contact_candidate(tmp_path, tmp_path)
