from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rmuc2026_mujoco import builder
from rmuc2026_mujoco._conversion import (
    MAXIMUM_HEIGHTFIELD_SAMPLES,
    MINIMUM_HEIGHTFIELD_RESOLUTION_M,
    FieldBuildError,
    _choose_spawn,
    _heightfield_grid_axes,
    _square_local_range,
)
from rmuc2026_mujoco.ramp_audit import (
    FIXED_FLY_RAMPS,
    WHEEL_PROBE_SPEEDS_M_S,
    WHEEL_PROBE_LOW_SIDE_APPROACH_M,
    audit_fixed_fly_ramps,
    wheel_probe_trial_plan,
    wheel_probe_route_endpoints,
)


def _synthetic_fixed_ramps(*, resolution_m: float = 0.01, seam_shift_m: float = 0.0):
    x = np.arange(-2.0, 2.0 + resolution_m / 2.0, resolution_m)
    y = np.arange(-6.0, 9.2 + resolution_m / 2.0, resolution_m)
    height = np.zeros((len(y), len(x)), dtype=np.float64)
    for ramp in FIXED_FLY_RAMPS:
        low = np.asarray(ramp.low_edge_center_xyz_m)
        uphill = np.asarray(ramp.uphill_unit_xy)
        normal = np.asarray(ramp.normal_xyz)
        dx = x[None, :] - low[0]
        dy = y[:, None] - low[1]
        along = dx * uphill[0] + dy * uphill[1]
        lateral = -dx * uphill[1] + dy * uphill[0]
        plane = low[2] - (normal[0] * dx + normal[1] * dy) / normal[2]
        low_seam = ramp.cad_low_seam_along_m + seam_shift_m
        high_seam = ramp.cad_high_seam_along_m - seam_shift_m
        low_height = (
            low[2]
            - (normal[0] * low_seam * uphill[0] + normal[1] * low_seam * uphill[1]) / normal[2]
        )
        high_height = (
            low[2]
            - (normal[0] * high_seam * uphill[0] + normal[1] * high_seam * uphill[1]) / normal[2]
        )
        strip = np.abs(lateral) <= ramp.surface_width_m / 2.0
        height[strip & (along < low_seam)] = low_height
        sloped = strip & (along >= low_seam) & (along <= high_seam)
        height[sloped] = plane[sloped]
        height[strip & (along > high_seam)] = high_height
    return x, y, height


def test_one_centimetre_grid_is_allowed_below_sample_ceiling() -> None:
    assert MINIMUM_HEIGHTFIELD_RESOLUTION_M == pytest.approx(0.01)
    x, y = _heightfield_grid_axes(
        np.array([-14.9, -8.0, 0.0]),
        np.array([14.9, 8.0, 3.7]),
        resolution_m=0.01,
    )
    assert len(x) * len(y) < MAXIMUM_HEIGHTFIELD_SAMPLES
    assert np.diff(x) == pytest.approx(0.01)
    assert np.diff(y) == pytest.approx(0.01)


def test_one_centimetre_grid_preserves_established_two_centimetre_envelope() -> None:
    low = np.array([-14.875997066913486, -6.377222061157227, -1.9])
    high = np.array([14.876011208013486, 9.625908851623535, 1.9])

    x_2cm, y_2cm = _heightfield_grid_axes(low, high, resolution_m=0.02)
    x_1cm, y_1cm = _heightfield_grid_axes(low, high, resolution_m=0.01)

    np.testing.assert_allclose(x_1cm[[0, -1]], x_2cm[[0, -1]], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(y_1cm[[0, -1]], y_2cm[[0, -1]], rtol=0.0, atol=1.0e-10)
    np.testing.assert_array_equal(x_1cm[::2], x_2cm)
    np.testing.assert_array_equal(y_1cm[::2], y_2cm)


def test_one_centimetre_grid_preserves_established_spawn_selection() -> None:
    low = np.array([-3.0, -2.0, -0.1])
    high = np.array([3.0, 2.0, 1.0])
    x_2cm, y_2cm = _heightfield_grid_axes(low, high, resolution_m=0.02)
    x_1cm, y_1cm = _heightfield_grid_axes(low, high, resolution_m=0.01)
    height_2cm = np.zeros((len(y_2cm), len(x_2cm)), dtype=np.float64)
    height_1cm = np.zeros((len(y_1cm), len(x_1cm)), dtype=np.float64)

    assert _choose_spawn(x_1cm, y_1cm, height_1cm) == _choose_spawn(x_2cm, y_2cm, height_2cm)


def test_heightfield_grid_rejects_sample_count_before_meshgrid() -> None:
    with pytest.raises(FieldBuildError, match="采样上限"):
        _heightfield_grid_axes(
            np.array([-10.0, -10.0, 0.0]),
            np.array([10.0, 10.0, 1.0]),
            resolution_m=0.01,
            maximum_samples=1000,
        )


def test_linear_memory_local_range_matches_original_window_definition() -> None:
    height = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0, 0.0],
            [3.0, 2.0, 5.0, 1.0],
        ]
    )
    padded = np.pad(height, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    expected = np.ptp(windows, axis=(-1, -2))

    actual = _square_local_range(height, radius_cells=1)

    np.testing.assert_array_equal(actual, expected)


def test_fixed_ramp_audit_passes_exact_one_centimetre_surfaces_without_mutation() -> None:
    x, y, height = _synthetic_fixed_ramps()
    before = height.copy()

    report = audit_fixed_fly_ramps(x, y, height)

    np.testing.assert_array_equal(height, before)
    assert report["status"] == "PASS_STATIC_PENDING_DYNAMIC"
    assert report["collision_owner"] == "existing_single_heightfield_only"
    assert report["heightfield_modified"] is False
    assert report["planar_refinement_applied"] is False
    assert [row["geometry"]["source_part_index"] for row in report["ramps"]] == [392, 397]
    for row in report["ramps"]:
        assert row["checks"]["static_pass"]
        assert row["interior"]["plane_abs_error_m"]["p95"] <= 0.001
        assert row["seams"]["maximum_location_abs_error_m"] <= 0.015
        assert row["wheel_geometry_probe"]["finite"]


def test_fixed_ramp_audit_reports_shifted_seams_instead_of_overclaiming() -> None:
    x, y, height = _synthetic_fixed_ramps(seam_shift_m=0.03)

    report = audit_fixed_fly_ramps(x, y, height)

    assert report["status"] == "FAIL_STATIC"
    assert any(not row["checks"]["static_pass"] for row in report["ramps"])
    assert all(row["interior"]["plane_abs_error_m"]["p95"] <= 0.001 for row in report["ramps"])


def test_two_centimetre_grid_cannot_claim_one_centimetre_seam_acceptance() -> None:
    x, y, height = _synthetic_fixed_ramps(resolution_m=0.02)

    report = audit_fixed_fly_ramps(x, y, height)

    assert report["fine_resolution_pass"] is False
    assert report["status"] == "NOT_RUN_REQUIRES_1CM"


def test_wheel_probe_plan_covers_every_speed_ramp_and_direction() -> None:
    trials = wheel_probe_trial_plan()

    assert len(trials) == 2 * 2 * len(WHEEL_PROBE_SPEEDS_M_S)
    assert {row["commanded_speed_m_s"] for row in trials} == set(WHEEL_PROBE_SPEEDS_M_S)
    assert {row["direction"] for row in trials} == {"uphill", "downhill"}
    assert all(row["wheel_diameter_m"] == pytest.approx(0.12) for row in trials)
    assert all(row["dynamic_status"] == "NOT_RUN_DURING_BUILD" for row in trials)
    assert all(row["endpoint_basis"] == "hash_bound_unsimplified_cad_ray_seams" for row in trials)
    ramps = {ramp.route_id: ramp for ramp in FIXED_FLY_RAMPS}
    for row in trials:
        ramp = ramps[row["route_id"]]
        low_approach, high_seam = wheel_probe_route_endpoints(ramp)
        assert low_approach == pytest.approx(
            ramp.cad_low_seam_along_m - WHEEL_PROBE_LOW_SIDE_APPROACH_M
        )
        assert row["nominal_horizontal_run_m"] == pytest.approx(ramp.horizontal_run_m)
        if row["direction"] == "uphill":
            assert row["start_along_ramp_m"] == pytest.approx(low_approach)
            assert row["finish_along_ramp_m"] == pytest.approx(high_seam)
        else:
            assert row["start_along_ramp_m"] == pytest.approx(high_seam)
            assert row["finish_along_ramp_m"] == pytest.approx(low_approach)


def test_runtime_builder_forwards_surface_guide_without_retaining_full_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_build(_step: Path, output: Path, **_kwargs: object) -> dict[str, object]:
        output.mkdir()
        return {}

    def fake_export(
        source: Path, destination: Path, *, include_surface_guide: bool
    ) -> dict[str, object]:
        captured.update(
            source=source,
            destination=destination,
            include_surface_guide=include_surface_guide,
        )
        destination.mkdir()
        return {"status": "PASS"}

    def fake_decorate(source: Path, rulebook: Path, destination: Path) -> dict[str, object]:
        assert source.name == "full-build"
        assert rulebook == tmp_path / "rulebook.pdf"
        destination.mkdir()
        return {}

    monkeypatch.setattr(builder, "build_from_official_step", fake_build)
    monkeypatch.setattr("rmuc2026_mujoco._conversion.decorate_field_build", fake_decorate)
    monkeypatch.setattr("rmuc2026_mujoco.pack.export_runtime_asset_pack", fake_export)
    destination = tmp_path / "runtime-pack"

    result = builder.build_runtime_asset_pack(
        tmp_path / "official.step",
        destination,
        include_surface_guide=True,
        rulebook_pdf=tmp_path / "rulebook.pdf",
        minimum_free_bytes=0,
    )

    assert result == {"status": "PASS"}
    assert captured["destination"] == destination
    assert captured["include_surface_guide"] is True
    assert Path(captured["source"]).name == "decorated-build"
    assert not Path(captured["source"]).exists()
