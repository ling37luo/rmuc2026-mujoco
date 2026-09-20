import numpy as np
import pytest

from rmuc2026_mujoco.collision_candidate import SOURCE_GLB_SHA256
from rmuc2026_mujoco.edge_void_candidate import (
    make_edge_void_candidate,
    outer_edge_sample_mask,
    raycast_outer_edge_hits,
)


def _grid():
    x = np.arange(0.0, 0.51, 0.01)
    y = np.arange(0.0, 0.51, 0.01)
    height = np.full((len(y), len(x)), 0.2)
    return x, y, height


def test_source_miss_only_changes_outer_strip_and_preserves_hit_heights():
    x, y, height = _grid()
    hits = np.ones(height.shape, dtype=bool)
    hits[0, 0] = False
    hits[1, 1] = False
    hits[0, 2] = True
    result = make_edge_void_candidate(
        x,
        y,
        height,
        hits,
        source_glb_sha256=SOURCE_GLB_SHA256,
        band_m=0.02,
        void_depth_m=20.0,
    )
    assert result.record["source_ray_misses_changed"] == 2
    assert result.record["interior_nodes_changed"] == 0
    assert result.record["new_collision_geoms"] == 0
    assert result.height_m[0, 0] == -20.0
    assert result.height_m[1, 1] == -20.0
    assert result.height_m[25, 25] == 0.2
    assert result.height_m[0, 2] == 0.2
    assert np.array_equal(height, np.full_like(height, 0.2))
    assert np.array_equal(result.changed_mask, outer_edge_sample_mask(x, y, band_m=0.02) & ~hits)


def test_source_hash_and_uncast_interior_fail_closed():
    x, y, height = _grid()
    hits = np.ones(height.shape, dtype=bool)
    with pytest.raises(ValueError, match="audited official GLB"):
        make_edge_void_candidate(x, y, height, hits, source_glb_sha256="0" * 64)
    hits[25, 25] = False
    with pytest.raises(ValueError, match="interior source-hit"):
        make_edge_void_candidate(
            x, y, height, hits, source_glb_sha256=SOURCE_GLB_SHA256, band_m=0.02
        )


def test_raycast_checks_edge_only_and_preserves_interior_scope():
    import trimesh

    x, y, _height = _grid()
    source = trimesh.creation.box(extents=(0.04, 0.50, 0.2))
    source.apply_translation((0.01, 0.25, 0.0))
    hits = raycast_outer_edge_hits(source, x, y, band_m=0.02)
    assert hits[25, 0]
    assert not hits[25, -1]
    assert hits[25, 25]  # deliberately uncast interior, never eligible for edits
