"""Backend-neutral geometry checks; these do not execute Isaac or PhysX."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "adapters/isaac/heightfield_mesh.py"
_SPEC = importlib.util.spec_from_file_location("heightfield_mesh", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MESH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MESH)


def _face_height(points: np.ndarray, face: np.ndarray, x: float, y: float) -> float:
    selected = points[face]
    weights = np.linalg.solve(
        np.vstack((selected[:, 0], selected[:, 1], np.ones(3))),
        np.asarray([x, y, 1.0]),
    )
    assert np.min(weights) >= -1e-12
    return float(weights @ selected[:, 2])


def test_mesh_faces_match_both_mujoco_triangle_planes() -> None:
    x = np.asarray([-10.0, -9.99])
    y = np.asarray([2.0, 2.01])
    # A saddle gives a distinct result for the other possible diagonal.
    z = np.asarray([[0.0, 0.0], [0.0, 1.0]])
    points, faces = _MESH.grid_to_triangle_mesh(x, y, z)
    assert points.shape == (4, 3)
    assert faces.tolist() == [[0, 1, 3], [0, 3, 2]]
    np.testing.assert_allclose(
        points[:, :2], [[0, 0], [0.01, 0], [0, 0.01], [0.01, 0.01]], rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(points[:, 2], [0, 0, 0, 1])
    for u, v, face in ((0.75, 0.25, faces[0]), (0.25, 0.75, faces[1])):
        px = float(x[0] + u * (x[1] - x[0]))
        py = float(y[0] + v * (y[1] - y[0]))
        local_x = px - x[0]
        local_y = py - y[0]
        assert _face_height(points, face, local_x, local_y) == pytest.approx(
            _MESH.triangle_height_at(x, y, z, px, py), abs=1e-12
        )
        triangle = points[face]
        assert np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[2] > 0


def test_mesh_preserves_float64_nodes_and_separates_sixteen_envs() -> None:
    x = np.linspace(-14.35, -8.22, 614)
    y = np.linspace(0.03, 2.15, 213)
    z = np.arange(y.size * x.size, dtype=np.float64).reshape(y.size, x.size) * 1e-7
    points, faces = _MESH.grid_to_triangle_mesh(x, y, z)
    assert points.shape == (213 * 614, 3)
    assert faces.shape == (2 * 212 * 613, 3)
    np.testing.assert_array_equal(points[:, 2], z.ravel())
    np.testing.assert_allclose(points[0, :2], [0.0, 0.0], atol=0)
    np.testing.assert_allclose(points[-1, :2], [6.13, 2.12], atol=1e-12)
    offsets = _MESH.environment_offsets(16, x, y)
    assert offsets.shape == (16, 2)
    assert len({tuple(row) for row in offsets}) == 16
    for a in range(16):
        for b in range(a):
            assert (
                abs(offsets[a, 0] - offsets[b, 0]) > x[-1] - x[0]
                or abs(offsets[a, 1] - offsets[b, 1]) > y[-1] - y[0]
            )


def test_mesh_rejects_nonfinite_and_outside_queries() -> None:
    x = np.asarray([0.0, 0.01])
    y = np.asarray([0.0, 0.01])
    z = np.zeros((2, 2))
    with pytest.raises(ValueError, match="finite"):
        _MESH.grid_to_triangle_mesh(x, y, np.asarray([[0.0, np.nan], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="outside"):
        _MESH.triangle_height_at(x, y, z, 0.02, 0.0)
    with pytest.raises(ValueError, match="count"):
        _MESH.environment_offsets(0, x, y)
