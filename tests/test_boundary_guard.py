"""Bounded source-edge stop signals without changing simulated physics."""

from __future__ import annotations

import numpy as np
import pytest

from rmuc2026_mujoco import FieldAsset, FieldBoundaryGuard, ManifestError


def _guard() -> FieldBoundaryGuard:
    axis = np.linspace(0.0, 2.0, 21)
    miss = np.zeros((len(axis), len(axis)), dtype=np.bool_)
    miss[-1, :] = True
    return FieldBoundaryGuard(
        axis, axis, miss, edge_band_m=0.5, source_miss_steps=3, contact_cap_steps=4
    )


def test_outward_source_hit_contact_cap_stops_before_an_unbounded_edge_step() -> None:
    guard = _guard()
    assert (
        guard.observe(
            sim_time_s=0.0,
            base_xy_m=(1.0, 1.5),
            field_contact_count=50,
            max_field_pair_contacts=50,
        )
        is None
    )
    for step in range(1, 4):
        assert (
            guard.observe(
                sim_time_s=step * 0.01,
                base_xy_m=(1.0, 1.5 + step * 0.1),
                field_contact_count=50,
                max_field_pair_contacts=50,
            )
            is None
        )
    stop = guard.observe(
        sim_time_s=0.04,
        base_xy_m=(1.0, 1.9),
        field_contact_count=50,
        max_field_pair_contacts=50,
    )
    assert stop is not None
    assert stop["reason"] == "outward_edge_heightfield_contact_saturation"
    assert stop["outward_edge"] == "top"
    assert stop["physical_state_modified"] is False
    assert (
        guard.observe(
            sim_time_s=0.04,
            base_xy_m=(1.0, 1.9),
            field_contact_count=50,
            max_field_pair_contacts=50,
        )
        == stop
    )


def test_inward_or_central_contact_cap_does_not_trigger() -> None:
    guard = _guard()
    for step in range(10):
        assert (
            guard.observe(
                sim_time_s=step * 0.01,
                base_xy_m=(1.0, 1.9 - step * 0.02),
                field_contact_count=50,
                max_field_pair_contacts=50,
            )
            is None
        )
    guard.reset()
    for step in range(10):
        assert (
            guard.observe(
                sim_time_s=step * 0.01,
                base_xy_m=(0.9 + step * 0.01, 1.0),
                field_contact_count=50,
                max_field_pair_contacts=50,
            )
            is None
        )


def test_source_miss_requires_persistent_contact_loss_and_reset_clears_it() -> None:
    guard = _guard()
    for step in range(2):
        assert (
            guard.observe(
                sim_time_s=step * 0.01,
                base_xy_m=(1.0, 2.0),
                field_contact_count=0,
                max_field_pair_contacts=0,
            )
            is None
        )
    # Same-time observations cannot count twice toward persistence.
    assert (
        guard.observe(
            sim_time_s=0.01,
            base_xy_m=(1.0, 2.0),
            field_contact_count=0,
            max_field_pair_contacts=0,
        )
        is None
    )
    assert (
        guard.observe(
            sim_time_s=0.02,
            base_xy_m=(1.0, 2.0),
            field_contact_count=1,
            max_field_pair_contacts=1,
        )
        is None
    )
    for step in range(3, 5):
        assert (
            guard.observe(
                sim_time_s=step * 0.01,
                base_xy_m=(1.0, 2.0),
                field_contact_count=0,
                max_field_pair_contacts=0,
            )
            is None
        )
    stop = guard.observe(
        sim_time_s=0.05,
        base_xy_m=(1.0, 2.0),
        field_contact_count=0,
        max_field_pair_contacts=0,
    )
    assert stop is not None and stop["reason"] == "source_miss_without_field_contact"
    guard.reset()
    assert guard.stop_record is None
    assert (
        guard.observe(
            sim_time_s=0.0,
            base_xy_m=(1.0, 1.0),
            field_contact_count=1,
            max_field_pair_contacts=1,
        )
        is None
    )


def test_guard_requires_fully_verified_schema3_asset(field_asset_dir) -> None:
    with pytest.raises(ManifestError, match="hash-verified schema-3"):
        FieldBoundaryGuard.from_asset(FieldAsset.open(field_asset_dir, verify=False))
    with pytest.raises(ManifestError, match="hash-verified schema-3"):
        FieldBoundaryGuard.from_asset(FieldAsset.open(field_asset_dir, verify=True))
