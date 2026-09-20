"""Synthetic geometry checks for the exact-source wall-end sampling gate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rmuc2026_mujoco.wall_tip_repair import (
    SOURCE_GLB_SHA256,
    _projected_top,
    repair_verified_wall_tips,
    validate_wall_tip_repair_record,
    verify_wall_tip_repair_samples,
)


def _hexagonal_prism(x_shift: float) -> SimpleNamespace:
    # A generic sloped six-point profile, not copied from the official CAD.
    profile = np.array([(0.0, 0.0), (3.95, 0.0), (3.95, 0.6), (2.5, 1.1), (1.0, 1.1), (0.0, 0.6)])
    vertices = np.array([(x_shift + side, y, z) for side in (0.0, 0.2) for y, z in profile])
    faces = []
    for offset in (0, 6):
        faces.extend((offset, offset + i, offset + i + 1) for i in range(1, 5))
    for i in range(6):
        j = (i + 1) % 6
        faces.extend(((i, j, j + 6), (i, j + 6, i + 6)))
    return SimpleNamespace(
        vertices=vertices,
        faces=np.asarray(faces),
        bounds=np.stack((vertices.min(axis=0), vertices.max(axis=0))),
    )


def _synthetic_field() -> tuple[list[object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(121, dtype=np.float64) * 0.01
    y = np.arange(396, dtype=np.float64) * 0.01
    height = np.zeros((len(y), len(x)), dtype=np.float64)
    meshes: list[object] = [None] * 404
    for part, shift in ((402, 0.0), (403, 1.0)):
        mesh = _hexagonal_prism(shift)
        meshes[part] = mesh
        ix = np.flatnonzero((x >= shift) & (x <= shift + 0.2))
        height[:, ix] = _projected_top(mesh, x[ix], y)
    original = height.copy()
    height[1, [1, 2, 3, 4]] = 0.0
    height[-2, [101, 102, 103]] = 0.0
    return meshes, x, y, height, original


def test_repairs_only_exact_source_covered_wall_end_nodes() -> None:
    meshes, x, y, height, original = _synthetic_field()
    old = height.copy()
    audit = repair_verified_wall_tips(meshes, x, y, height, source_glb_sha256=SOURCE_GLB_SHA256)
    assert audit["status"] == "PASS"
    assert audit["repaired_nodes"] == 7
    assert audit["new_collision_geoms"] == 0
    assert np.array_equal(height, original)
    assert np.count_nonzero(old != height) == 7


def test_source_hash_and_repair_count_fail_without_mutation() -> None:
    meshes, x, y, height, _ = _synthetic_field()
    old = height.copy()
    with pytest.raises(ValueError, match="exact audited official GLB"):
        repair_verified_wall_tips(meshes, x, y, height, source_glb_sha256="0" * 64)
    assert np.array_equal(height, old)
    height[1, 5] = 0.0
    changed = height.copy()
    with pytest.raises(ValueError, match="changed repair count"):
        repair_verified_wall_tips(meshes, x, y, height, source_glb_sha256=SOURCE_GLB_SHA256)
    assert np.array_equal(height, changed)


def test_manifest_record_binds_the_repaired_float_samples() -> None:
    meshes, x, y, height, _ = _synthetic_field()
    record = repair_verified_wall_tips(meshes, x, y, height, source_glb_sha256=SOURCE_GLB_SHA256)
    record["collision_samples_sha256"] = "a" * 64
    validate_wall_tip_repair_record(record, collision_samples_sha256="a" * 64)
    verify_wall_tip_repair_samples(record, x, y, height)
    with pytest.raises(ValueError, match="not bound"):
        validate_wall_tip_repair_record(record, collision_samples_sha256="b" * 64)
    changed = height.copy()
    changed[
        record["changed_samples"][0]["grid_row"], record["changed_samples"][0]["grid_column"]
    ] = 0
    with pytest.raises(ValueError, match="disagrees with the float grid"):
        verify_wall_tip_repair_samples(record, x, y, changed)
    record["parts"][1]["repaired_nodes"] = 4
    with pytest.raises(ValueError, match="count changed"):
        validate_wall_tip_repair_record(record, collision_samples_sha256="a" * 64)
