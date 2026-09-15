from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rmuc2026_mujoco._conversion import _clean_visual_mesh, _simplify_parts_then_group

trimesh = pytest.importorskip("trimesh")


def test_visual_cleanup_removes_coincident_opposite_and_degenerate_faces() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        faces=np.asarray(
            [
                [0, 1, 2],
                [2, 1, 3],  # same triangle with the opposite winding
                [0, 3, 1],  # collapses after exact-position vertex merge
            ]
        ),
        process=False,
    )

    report = _clean_visual_mesh(mesh)

    assert len(mesh.faces) == 1
    assert len(mesh.vertices) == 3
    assert mesh.area == pytest.approx(0.5)
    assert report["merged_vertices"] == 1
    assert report["invalid_faces_removed"] == 1
    assert report["duplicate_faces_removed"] == 1
    assert report["bounds_max_abs_change_m"] == pytest.approx(0.0)
    assert report["winding_consistent_after_repair"] is True


def test_visual_cleanup_is_a_noop_for_clean_geometry() -> None:
    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    original_vertices = np.asarray(mesh.vertices).copy()
    original_faces = np.asarray(mesh.faces).copy()
    original_bounds = np.asarray(mesh.bounds).copy()

    report = _clean_visual_mesh(mesh)

    assert len(mesh.vertices) == len(original_vertices)
    assert len(mesh.faces) == len(original_faces)
    assert np.array_equal(mesh.bounds, original_bounds)
    assert report["invalid_faces_removed"] == 0
    assert report["duplicate_faces_removed"] == 0
    assert report["winding_consistent_after_repair"] is True


def test_visual_export_pipeline_records_cleanup_and_keeps_collision_out_of_scope(
    tmp_path: Path,
) -> None:
    floor = trimesh.creation.box(extents=[10.0, 10.0, 0.1])
    floor.visual.face_colors = [240, 240, 220, 255]
    structure = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    structure.apply_translation([0.0, 0.0, 1.0])
    structure.visual.face_colors = [20, 20, 20, 255]

    records, _meshes, audit = _simplify_parts_then_group(
        [floor, structure],
        tmp_path,
        target_faces=50_000,
    )

    assert len(records) == 2
    assert all((tmp_path / record["file"]).is_file() for record in records)
    assert all(audit["hard_gates"].values())
    assert audit["visual_cleanup"]["geometry_simplification"] is False
    assert audit["visual_cleanup"]["collision_changed"] is False
