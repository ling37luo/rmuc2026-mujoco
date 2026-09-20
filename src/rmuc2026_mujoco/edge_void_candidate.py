"""Source-ray-bounded outer-edge void sample constructor.

The legacy heightfield fills every missed CAD ray with z=0.  That creates a
standable floor outside the source's finite boundary.  This function marks
only misses within an outer 1.25 m strip with a deep sentinel.  It does not
write a pack; the builder can use its samples only with the schema-3 origin,
scale, PNG and exact-float injection contract.  Writing them into an existing
schema-1/2 pack is invalid.

MuJoCo heightfields have no true holes.  The deep floor remains an artificial
surface below the playable area.  This module makes no runtime or source-pack
changes on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

from .collision_candidate import SOURCE_GLB_SHA256


EDGE_BAND_M = 1.25
VOID_DEPTH_M = 5.0
GRID_RESOLUTION_M = 0.01


@dataclass(frozen=True)
class EdgeVoidCandidate:
    height_m: np.ndarray
    changed_mask: np.ndarray
    source_hit_mask: np.ndarray
    record: dict[str, Any]


def _axes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if (
        x.ndim != 1
        or y.ndim != 1
        or len(x) < 3
        or len(y) < 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.allclose(np.diff(x), GRID_RESOLUTION_M, rtol=0.0, atol=1.0e-6)
        or not np.allclose(np.diff(y), GRID_RESOLUTION_M, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("edge-void candidate requires a finite, regular 1 cm grid")
    return x, y


def outer_edge_sample_mask(
    x_m: np.ndarray, y_m: np.ndarray, *, band_m: float = EDGE_BAND_M
) -> np.ndarray:
    """Select four finite outer strips; no interior sample can be changed."""

    x, y = _axes(x_m, y_m)
    if not math.isfinite(band_m) or band_m <= 0 or 2 * band_m >= min(np.ptp(x), np.ptp(y)):
        raise ValueError("edge strip width is invalid for this grid")
    columns = (x - x[0] <= band_m + 1.0e-9) | (x[-1] - x <= band_m + 1.0e-9)
    rows = (y - y[0] <= band_m + 1.0e-9) | (y[-1] - y <= band_m + 1.0e-9)
    return rows[:, None] | columns[None, :]


def raycast_outer_edge_hits(
    source_mesh: Any,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    band_m: float = EDGE_BAND_M,
    batch_size: int = 8192,
) -> np.ndarray:
    """Return source-hit validity for edge nodes in the source GLB frame.

    Interior nodes are deliberately marked true without raycasting because the
    candidate is forbidden from editing them.  The caller must separately
    verify the source GLB digest before invoking this function.
    """

    import trimesh

    x, y = _axes(x_m, y_m)
    if batch_size < 1:
        raise ValueError("ray batch size must be positive")
    edge = outer_edge_sample_mask(x, y, band_m=band_m)
    bounds = np.asarray(source_mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("source mesh bounds are invalid")
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(source_mesh)
    flat_edge = np.flatnonzero(edge)
    hit = np.ones(edge.size, dtype=np.bool_)
    hit[flat_edge] = False
    columns = len(x)
    origin_z = float(bounds[1, 2] + 1.0)
    for start in range(0, len(flat_edge), batch_size):
        selected = flat_edge[start : start + batch_size]
        row = selected // columns
        col = selected % columns
        origins = np.column_stack((x[col], y[row], np.full(len(selected), origin_z)))
        directions = np.zeros_like(origins)
        directions[:, 2] = -1.0
        _locations, ray_indices, _triangle_indices = intersector.intersects_location(
            origins, directions, multiple_hits=False
        )
        hit[selected[np.asarray(ray_indices, dtype=np.int64)]] = True
    return hit.reshape(edge.shape)


def make_edge_void_candidate(
    x_m: np.ndarray,
    y_m: np.ndarray,
    height_m: np.ndarray,
    source_hit_mask: np.ndarray,
    *,
    source_glb_sha256: str,
    band_m: float = EDGE_BAND_M,
    void_depth_m: float = VOID_DEPTH_M,
) -> EdgeVoidCandidate:
    """Encode only source-ray misses at the outer edge as deep void samples.

    Heights at every source-hit node and every interior node are byte-for-byte
    unchanged.  ``height_m`` remains a physical height array, with negative
    sentinel values; a caller must update the MJCF hfield origin, vertical
    scale, PNG normalization, exact float injection and pack schema together.
    """

    if source_glb_sha256 != SOURCE_GLB_SHA256:
        raise ValueError("edge-void candidate requires the audited official GLB")
    x, y = _axes(x_m, y_m)
    height = np.asarray(height_m, dtype=np.float64)
    hits = np.asarray(source_hit_mask)
    if (
        height.shape != (len(y), len(x))
        or not np.isfinite(height).all()
        or np.min(height) < 0.0
        or hits.shape != height.shape
        or hits.dtype != np.bool_
    ):
        raise ValueError("edge-void input heights or source-hit mask are invalid")
    if not math.isfinite(void_depth_m) or void_depth_m <= 1.0:
        raise ValueError("void sentinel depth must exceed the playable-height band")
    edge = outer_edge_sample_mask(x, y, band_m=band_m)
    if not np.all(hits[~edge]):
        raise ValueError("interior source-hit values must be true because they were not cast")
    changed = edge & ~hits
    candidate = height.copy()
    candidate[changed] = -float(void_depth_m)
    indices = np.argwhere(changed).astype("<i4", copy=False)
    record = {
        "status": "CANDIDATE_ONLY",
        "activation": "DISABLED",
        "source_glb_sha256": source_glb_sha256,
        "grid_resolution_m": GRID_RESOLUTION_M,
        "edge_band_m": float(band_m),
        "void_depth_m": float(void_depth_m),
        "edge_samples_cast": int(np.count_nonzero(edge)),
        "source_ray_misses_changed": int(len(indices)),
        "changed_row_column_le_i4_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "source_hit_mask_c_order_sha256": hashlib.sha256(hits.tobytes()).hexdigest(),
        "interior_nodes_changed": 0,
        "source_hit_nodes_changed": 0,
        "contact_owner": "existing_single_heightfield_only",
        "new_collision_geoms": 0,
        "required_hfield_origin_z_delta_m": -float(void_depth_m),
        "required_hfield_vertical_scale_delta_m": float(void_depth_m),
        "claim_boundary": (
            "A deep surrogate floor removes playable-height support where the exact official "
            "source ray misses in the finite outer strip. MuJoCo heightfields have no holes; "
            "this does not represent exact source sidewalls or a bottomless void."
        ),
    }
    return EdgeVoidCandidate(candidate, changed, hits.copy(), record)
