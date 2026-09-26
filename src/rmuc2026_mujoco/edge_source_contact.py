"""Opt-in source-triangle contact candidates for the two raised field edges.

The upper and lower deck-to-base transitions have not passed robot traversal
gates.  This module only prepares deterministic geometry/heightfield evidence;
it deliberately does not write or activate a runtime pack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .manifest import FieldAsset
from .query import HeightFieldData, load_heightfield
from .wall_active_contact import _verified_source_scene


EDGE_SOURCE_FACES: dict[str, tuple[int, tuple[int, ...]]] = {
    "top": (236, tuple(range(2, 11))),
    "bottom": (244, (28,)),
}
BASE_SOURCE_PART = 552
SOURCE_DECK_THICKNESS_M = 0.20006858
ROOF_NODE_ERROR_LIMIT_M = 0.0011
BASE_SOURCE_ERROR_LIMIT_M = 2.0e-6


def _world_part(scene: Any, nodes: list[str], part: int, translation: np.ndarray) -> Any:
    if part >= len(nodes):
        raise ValueError(f"official source part {part} is absent")
    transform, geometry_name = scene.graph[nodes[part]]
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    mesh.apply_translation(translation)
    return mesh


def _prism_from_roof_triangle(triangle: np.ndarray) -> Any:
    """Return one bounded convex prism with the original roof triangle."""

    import trimesh

    roof = np.asarray(triangle, dtype=np.float64).copy()
    if roof.shape != (3, 3) or not np.isfinite(roof).all():
        raise ValueError("official roof triangle is invalid")
    area2 = np.linalg.det(np.column_stack((roof[1, :2] - roof[0, :2], roof[2, :2] - roof[0, :2])))
    if area2 < 0:
        roof[[1, 2]] = roof[[2, 1]]
    if abs(area2) < 1.0e-8:
        raise ValueError("official roof triangle has zero XY area")
    bottom = roof.copy()
    bottom[:, 2] -= SOURCE_DECK_THICKNESS_M
    vertices = np.vstack((roof, bottom))
    faces = np.array(
        (
            (0, 1, 2),
            (3, 5, 4),
            (0, 3, 4),
            (0, 4, 1),
            (1, 4, 5),
            (1, 5, 2),
            (2, 5, 3),
            (2, 3, 0),
        ),
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("source roof prism is not a closed outward convex mesh")
    return mesh


def _check_source_base(base_mesh: Any, roof_triangle: np.ndarray) -> float:
    """Check that the prism lower face is the actual part-552 top."""

    import trimesh

    barycentric = np.asarray(
        ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)),
        dtype=np.float64,
    )
    source_roof = barycentric @ roof_triangle
    origins = source_roof.copy()
    origins[:, 2] = float(max(np.max(roof_triangle[:, 2]), base_mesh.bounds[1, 2]) + 1.0)
    directions = np.tile((0.0, 0.0, -1.0), (len(origins), 1))
    locations, indices, _faces = trimesh.ray.ray_triangle.RayMeshIntersector(
        base_mesh
    ).intersects_location(origins, directions, multiple_hits=True)
    errors = []
    for index in range(len(origins)):
        hits = locations[indices == index, 2]
        if len(hits) == 0:
            raise ValueError("official base is absent beneath a source roof face")
        errors.append(abs(float(source_roof[index, 2] - max(hits)) - SOURCE_DECK_THICKNESS_M))
    maximum = max(errors)
    if maximum > BASE_SOURCE_ERROR_LIMIT_M:
        raise ValueError(f"official base differs from roof prism bottom: {maximum}")
    return maximum


def _transfer_roof_nodes(
    field: HeightFieldData,
    updated_height: np.ndarray,
    owned: np.ndarray,
    triangle: np.ndarray,
) -> dict[str, int | float]:
    """Transfer only roof-matching grid nodes to the source base."""

    x, y = field.x_m, field.y_m
    ix = np.flatnonzero((x >= min(triangle[:, 0])) & (x <= max(triangle[:, 0])))
    iy = np.flatnonzero((y >= min(triangle[:, 1])) & (y <= max(triangle[:, 1])))
    if len(ix) == 0 or len(iy) == 0:
        raise ValueError("source roof does not intersect the runtime heightfield")
    xx, yy = np.meshgrid(x[ix], y[iy])
    coordinates = np.vstack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
    matrix = np.vstack((triangle[:, 0], triangle[:, 1], np.ones(3)))
    weights = np.linalg.solve(matrix, coordinates)
    inside = np.all(weights >= -1.0e-9, axis=0).reshape(xx.shape)
    roof_height = (triangle[:, 2] @ weights).reshape(xx.shape)
    original = field.height_m[np.ix_(iy, ix)]
    error = original - roof_height
    transfer = inside & (np.abs(error) <= ROOF_NODE_ERROR_LIMIT_M)
    if np.any(inside & (error < -ROOF_NODE_ERROR_LIMIT_M)):
        raise ValueError("heightfield is below the official roof inside its footprint")
    current_owned = owned[np.ix_(iy, ix)]
    if np.any(current_owned & transfer):
        raise ValueError("two source prisms claim the same heightfield node")
    if int(np.count_nonzero(transfer)) < 0.99 * int(np.count_nonzero(inside)):
        raise ValueError("heightfield roof does not match the official source face")
    current = updated_height[np.ix_(iy, ix)]
    current[transfer] = roof_height[transfer] - SOURCE_DECK_THICKNESS_M
    updated_height[np.ix_(iy, ix)] = current
    current_owned[transfer] = True
    owned[np.ix_(iy, ix)] = current_owned
    return {
        "inside_grid_nodes": int(np.count_nonzero(inside)),
        "transferred_roof_nodes": int(np.count_nonzero(transfer)),
        "retained_higher_hfield_nodes": int(np.count_nonzero(inside & ~transfer)),
        "maximum_roof_node_error_m": float(np.max(np.abs(error[transfer]))),
    }


def build_source_edge_contact_candidate(
    source_build: Path,
    runtime_pack: Path,
    *,
    edges: tuple[str, ...] = ("top", "bottom"),
) -> tuple[HeightFieldData, dict[str, Any], dict[str, Any]]:
    """Build non-default top/bottom contact-owner experiments from official faces.

    A negative-capable hfield is required because the official base beneath
    these deck roofs is below the ordinary schema-2 zero plane.  Passing this
    geometric audit never changes the experimental activation status.
    """

    if not edges or len(set(edges)) != len(edges) or any(e not in EDGE_SOURCE_FACES for e in edges):
        raise ValueError("edges must be a unique nonempty subset of top and bottom")
    asset = FieldAsset.open(runtime_pack)
    if float(asset.collision["minimum_height_m"]) >= -0.1:
        raise ValueError("official edge base requires a negative-capable heightfield pack")
    field = load_heightfield(asset)
    scene, translation, source_identity = _verified_source_scene(
        Path(source_build).expanduser().resolve(), asset
    )
    nodes = list(scene.graph.nodes_geometry)
    base = _world_part(scene, nodes, BASE_SOURCE_PART, translation)
    updated = field.height_m.copy()
    owned = np.zeros(updated.shape, dtype=np.bool_)
    meshes: dict[str, Any] = {}
    face_reports: list[dict[str, Any]] = []
    for edge in edges:
        part, face_indices = EDGE_SOURCE_FACES[edge]
        source = _world_part(scene, nodes, part, translation)
        for face_index in face_indices:
            if face_index >= len(source.faces):
                raise ValueError(f"official {edge} roof face {face_index} is absent")
            triangle = source.vertices[source.faces[face_index]]
            base_error = _check_source_base(base, triangle)
            mesh = _prism_from_roof_triangle(triangle)
            node_report = _transfer_roof_nodes(field, updated, owned, triangle)
            name = f"rmuc2026_official_{edge}_part{part}_face{face_index}"
            meshes[name] = mesh
            face_reports.append(
                {
                    "name": name,
                    "edge": edge,
                    "source_part_index": part,
                    "source_face_index": face_index,
                    "maximum_source_base_error_m": base_error,
                    **node_report,
                }
            )
    world_minimum = float(asset.collision["minimum_height_m"]) - float(
        asset.recommended_spawn["terrain_height_m"]
    )
    if float(np.min(updated)) < world_minimum - 1.0e-8:
        raise ValueError("source base exceeds the runtime hfield negative range")
    report = {
        "status": "EXPERIMENTAL_BLOCKED",
        "activation": "DISABLED_BY_DEFAULT",
        "base_runtime_manifest_sha256": asset.manifest_sha256,
        "source_identity": source_identity,
        "edges": list(edges),
        "base_source_part_index": BASE_SOURCE_PART,
        "source_deck_thickness_m": SOURCE_DECK_THICKNESS_M,
        "contact_ownership": "source_face_prisms_roof_and_sides; hfield_source_base_under_transferred_nodes",
        "collision_bits": [2, 1],
        "friction": [1.0, 0.005, 0.0001],
        "solref": "0.02 1",
        "faces": face_reports,
        "roof_nodes_transferred": int(np.count_nonzero(owned)),
        "outside_transfer_mask_bitwise_unchanged": bool(
            np.array_equal(updated[~owned], field.height_m[~owned])
        ),
        "validation_boundary": (
            "source geometry and grid-node ownership only; top crossing and bottom 1.0 m/s "
            "robot gates remain blocked, and adjacent/shared-prism solver contacts are unverified"
        ),
    }
    return HeightFieldData(field.x_m, field.y_m, updated), meshes, report
