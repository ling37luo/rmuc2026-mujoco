"""Small geometry gates for the read-only, cross-resolution clearance review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rmuc2026_mujoco.clearance_review import (
    ClearanceReviewError,
    _bound_file,
    _coordinate_contract,
    review_entry_mask,
    review_samples,
)
from rmuc2026_mujoco.query import HeightFieldData


def test_current_heightfield_seal_is_recomputed_from_its_own_grid() -> None:
    heights = np.full((3, 3), 0.7)
    heights[2, 2] = 0.1
    current = HeightFieldData(
        x_m=np.array([0.0, 0.5, 1.0]),
        y_m=np.array([0.0, 0.5, 1.0]),
        height_m=heights,
    )
    result = review_samples(
        current,
        np.array([0.10, 0.90]),
        np.array([0.10, 0.90]),
        np.full((2, 2), 0.65),
        np.array([[1, 0], [0, 1]], dtype=np.uint8),
    )
    assert result["candidate_count"] == 2
    assert result["current_heightfield_sealed_count"] == 1
    assert result["current_heightfield_sealed_fraction"] == 0.5


def test_horizontal_channel_needs_contiguous_lanes_clear_at_every_height() -> None:
    blocked = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], dtype=np.uint8)
    result = review_entry_mask(blocked, spacing_m=0.02, required_width_m=0.05)
    assert result["fully_clear_lane_count"] == 4
    assert result["widest_connected_clear_width_m"] == pytest.approx(0.06)
    assert result["envelope_channel_observed"] is True
    blocked[1, 1] = 1
    assert review_entry_mask(blocked, 0.02, 0.05)["envelope_channel_observed"] is False


def test_old_evidence_hash_must_match_pinned_manifest(tmp_path: Path) -> None:
    evidence = tmp_path / "manifest.json"
    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(ClearanceReviewError, match="SHA-256 mismatch"):
        _bound_file(evidence, "0" * 64, "legacy evidence")


def test_coordinate_contract_excludes_resolution_but_detects_shift() -> None:
    source = {
        "conversion": {
            "axis_and_units": {"axis_order_output_xyz": [0, 1, 2], "unit_scale_to_metres": 1},
            "main_floor_height_before_shift_m": -1.8,
        },
        "recommended_spawn": {"x_before_translation_m": 11.16},
        "collision": {
            "resolution_m": 0.02,
            "geom_center_after_translation_m": [-11.16, -5.78, -0.01],
            "x_min_m": -14.9,
            "x_max_m": 14.9,
            "y_min_m": -6.4,
            "y_max_m": 9.6,
        },
    }
    finer = {
        **source,
        "collision": {**source["collision"], "resolution_m": 0.01},
    }
    assert _coordinate_contract(source) == _coordinate_contract(finer)
    shifted = {
        **finer,
        "recommended_spawn": {"x_before_translation_m": 11.26},
    }
    assert _coordinate_contract(source) != _coordinate_contract(shifted)
