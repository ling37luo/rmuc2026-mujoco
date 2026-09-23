"""Backend-neutral geometry checks; these do not execute Isaac or PhysX."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "adapters/isaac/heightfield_mesh.py"
_SPEC = importlib.util.spec_from_file_location("heightfield_mesh", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MESH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MESH)
sys.path.insert(0, str(_MODULE_PATH.parent))
_PROBE_PATH = _MODULE_PATH.parent / "physx_fly_contact_smoke.py"
_PROBE_SPEC = importlib.util.spec_from_file_location("physx_fly_contact_smoke", _PROBE_PATH)
assert _PROBE_SPEC is not None and _PROBE_SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(_PROBE)


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


def test_fly_static_fence_probe_and_environment_separation() -> None:
    x = np.linspace(-14.35, -8.22, 614)
    y = np.linspace(0.03, 2.15, 213)
    box = {
        "name": "rmuc2026_perimeter_top",
        "pos_m": [-11.285, 1.7215, 1.0404],
        "size_m": [3.065, 0.025, 1.35],
        "contype": 2,
        "conaffinity": 1,
        "friction": [1.0, 0.005, 0.0001],
        "solref": [0.04, 1.0],
    }
    descriptor = {
        "schema_version": 2,
        "scenario_id": "fly_ramp_north",
        "collision": {"static_boxes": [box]},
    }
    assert _PROBE._static_boxes(descriptor, x, y) == [box]
    wall = _PROBE._wall_probe(box, [-8.9, 1.3])
    assert wall["direction_xyz"] == [0.0, 1.0, 0.0]
    assert wall["ray_origin_xyz_m"][1] == pytest.approx(box["pos_m"][1] - box["size_m"][1] - 0.2)
    assert wall["sphere_initial_xyz_m"][1] == pytest.approx(
        box["pos_m"][1] - box["size_m"][1] - 0.15
    )
    for count in (1, 4, 16):
        _PROBE._check_environment_box_separation([box], _MESH.environment_offsets(count, x, y))
    with pytest.raises(ValueError, match="overlap across environments"):
        _PROBE._check_environment_box_separation([box], np.array([[0.0, 0.0], [0.1, 0.0]]))


def test_fly_static_fence_contract_rejects_uncropped_or_missing_box() -> None:
    x = np.linspace(-14.35, -8.22, 614)
    y = np.linspace(0.03, 2.15, 213)
    descriptor = {"schema_version": 2, "scenario_id": "fly_ramp_north"}
    with pytest.raises(ValueError, match="collision.static_boxes"):
        _PROBE._static_boxes(descriptor, x, y)
    descriptor["collision"] = {"static_boxes": []}
    with pytest.raises(ValueError, match="no cropped static perimeter"):
        _PROBE._static_boxes(descriptor, x, y)
    descriptor["collision"]["static_boxes"] = [
        {
            "name": "rmuc2026_perimeter_top",
            "pos_m": [-11.285, 1.7215, 1.0404],
            "size_m": [14.025, 0.025, 1.35],
            "contype": 2,
            "conaffinity": 1,
            "friction": [1.0, 0.005, 0.0001],
            "solref": [0.04, 1.0],
        }
    ]
    with pytest.raises(ValueError, match="outside crop"):
        _PROBE._static_boxes(descriptor, x, y)
    assert _PROBE._static_boxes({"schema_version": 1}, x, y) == []


def test_raw_contact_observation_requires_exact_probe_and_collider_pair() -> None:
    sphere = "/World/env_0/probe_takeoff"
    terrain = "/World/env_0/terrain"
    other_terrain = "/World/env_1/terrain"
    fence = "/World/env_0/rmuc2026_perimeter_top"
    body_paths = {1: sphere, 2: terrain, 3: other_terrain, 4: fence}
    decode = body_paths.__getitem__

    direct = _PROBE._raw_contact_observation(
        [{"body0": np.uint64(1), "body1": np.uint64(2)}], sphere, terrain, decode
    )
    assert direct == {
        "raw_contact_count": 1,
        "observed_collider": terrain,
        "observed_colliders": [terrain],
        "valid": True,
    }
    reverse = _PROBE._raw_contact_observation(
        [{"body0": terrain, "body1": sphere}], sphere, terrain, decode
    )
    assert reverse["valid"] is True
    assert reverse["observed_collider"] == terrain

    for raw, observed in (
        ([], None),
        ([{"body0": 1, "body1": 3}], other_terrain),
        ([{"body0": 1, "body1": 4}], fence),
        ([{"body0": 2, "body1": 3}], None),
        ([{"body0": 1}], None),
    ):
        result = _PROBE._raw_contact_observation(raw, sphere, terrain, decode)
        assert result["valid"] is False
        assert result["observed_collider"] == observed

    mixed = _PROBE._raw_contact_observation(
        [{"body0": 1, "body1": 2}, {"body0": 1, "body1": 3}], sphere, terrain, decode
    )
    assert mixed["valid"] is False
    assert mixed["observed_collider"] is None
    assert mixed["observed_colliders"] == [terrain, other_terrain]


def test_physx_fly_probe_requires_source_heightfield_contact_metadata() -> None:
    terrain = {
        "name": "rmuc2026_field_collision",
        "contype": 2,
        "conaffinity": 1,
        "friction": [1.0, 0.005, 0.0001],
        "solref": [0.02, 1.0],
    }
    descriptor = {"schema_version": 3, "collision": {"heightfield": terrain}}
    assert _PROBE._heightfield_contact(descriptor) == terrain
    assert _PROBE._heightfield_contact({"schema_version": 2}) is None
    descriptor["collision"]["heightfield"] = {**terrain, "friction": [float("nan"), 0, 0]}
    with pytest.raises(ValueError, match="heightfield contact metadata"):
        _PROBE._heightfield_contact(descriptor)
    descriptor["collision"].pop("heightfield")
    with pytest.raises(ValueError, match="lacks collision.heightfield"):
        _PROBE._heightfield_contact(descriptor)
