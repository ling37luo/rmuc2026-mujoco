"""The boundary audit must reject a made-up continuous perimeter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from rmuc2026_mujoco.boundary_audit import (
    BoundaryAuditError,
    audit_source_boundaries,
    audit_triangle_bounds,
)
from rmuc2026_mujoco.download import OFFICIAL_STEP_SHA256
from rmuc2026_mujoco.manifest import sha256_file


def _panel(y_center: float, y_width: float) -> trimesh.Trimesh:
    panel = trimesh.creation.box(extents=(0.04, y_width, 1.0))
    panel.apply_translation((-0.4, y_center, 0.5))
    return panel


def test_disjoint_official_panels_cannot_justify_a_perimeter_wall() -> None:
    report = audit_triangle_bounds(
        [_panel(-0.75, 0.5), _panel(0.75, 0.5)],
        world_translation_m=(0.0, 0.0, 0.0),
        x_strip_m=(-1.0, 0.0),
        y_bounds_m=(-1.0, 1.0),
        bin_width_m=0.05,
    )
    for probe in report["probe_heights"]:
        assert probe["source_face_coverage_upper_bound_fraction"] == pytest.approx(0.55)
        assert probe["definite_no_face_intervals_y_m"] == [[-0.45, 0.45]]
        assert probe["largest_definite_no_face_gap_m"] == pytest.approx(0.9)
        assert probe["contributing_source_part_indices"] == [0, 1]
        assert probe["continuous_barrier_supported"] is False


def test_full_triangle_aabb_coverage_does_not_certify_a_wall() -> None:
    report = audit_triangle_bounds(
        [_panel(0.0, 2.0)],
        world_translation_m=(0.0, 0.0, 0.0),
        x_strip_m=(-1.0, 0.0),
        y_bounds_m=(-1.0, 1.0),
    )
    for probe in report["probe_heights"]:
        assert probe["source_face_coverage_upper_bound_fraction"] == 1.0
        assert probe["definite_no_face_intervals_y_m"] == []
        assert probe["continuous_barrier_supported"] is False


def test_thin_face_between_bin_centres_is_not_reported_as_a_definite_gap() -> None:
    report = audit_triangle_bounds(
        [_panel(-0.49, 0.002)],
        world_translation_m=(0.0, 0.0, 0.0),
        x_strip_m=(-1.0, 0.0),
        y_bounds_m=(-1.0, 1.0),
        bin_width_m=0.05,
    )
    for probe in report["probe_heights"]:
        assert probe["source_face_coverage_upper_bound_bins"] == 1
        assert probe["definite_no_face_intervals_y_m"] == [[-1.0, -0.5], [-0.45, 1.0]]


def _source(tmp_path: Path) -> Path:
    glb = tmp_path / "source.glb"
    glb.write_bytes(trimesh.Scene([_panel(-0.75, 0.5), _panel(0.75, 0.5)]).export(file_type="glb"))
    collision_dir = tmp_path / "collision"
    collision_dir.mkdir()
    samples = collision_dir / "samples.npz"
    samples.write_bytes(b"synthetic collision samples")
    manifest = {
        "artifact_type": "rmuc2026_official_field_mujoco_asset",
        "status": "PASS",
        "source": {"sha256": OFFICIAL_STEP_SHA256},
        "conversion": {
            "axis_and_units": {
                "axis_order_output_xyz": [0, 1, 2],
                "unit_scale_to_metres": 1.0,
            },
            "main_floor_height_before_shift_m": 0.0,
            "colored_intermediate_glb": {"file": glb.name, "sha256": sha256_file(glb)},
        },
        "collision": {
            "x_min_m": -1.0,
            "x_max_m": 1.0,
            "y_min_m": -1.0,
            "y_max_m": 1.0,
            "samples_file": "collision/samples.npz",
            "samples_sha256": sha256_file(samples),
        },
        "recommended_spawn": {
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_registry_is_disabled_and_bound_to_source_hashes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = audit_source_boundaries(
        source,
        inboard_width_m=0.8,
        expected_source_manifest_sha256=sha256_file(source),
    )
    assert result["status"] == "AUDIT_ONLY"
    assert result["activation"] == "disabled"
    assert result["source_manifest_sha256"] == sha256_file(source)
    assert result["source_glb_sha256"] == sha256_file(tmp_path / "source.glb")
    assert result["candidates"][0]["source_part_indices"] == [0, 1]
    reports = result["candidates"][0]["evidence"]["edge_reports"]
    assert set(reports) == {"negative_x", "positive_x", "negative_y", "positive_y"}
    assert reports["negative_x"]["probe_heights"][0]["definite_no_face_intervals_transverse_m"] == [
        [-0.45, 0.45]
    ]


def test_source_change_rejected_before_geometry_audit(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(BoundaryAuditError, match="source manifest SHA-256 mismatch"):
        audit_source_boundaries(source, expected_source_manifest_sha256="0" * 64)
    original = json.loads(source.read_text(encoding="utf-8"))
    original["conversion"]["colored_intermediate_glb"]["sha256"] = "0" * 64
    source.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(BoundaryAuditError, match="source CAD GLB SHA-256 mismatch"):
        audit_source_boundaries(source)


def test_collision_samples_are_verified_before_registry_output(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (tmp_path / "collision" / "samples.npz").write_bytes(b"changed")
    with pytest.raises(BoundaryAuditError, match="source collision samples SHA-256 mismatch"):
        audit_source_boundaries(source, inboard_width_m=0.8)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("status", "DRAFT_BLOCKED", "passing official field build"),
        ("source", {"sha256": "0" * 64}, "audited official STEP"),
    ],
)
def test_nonofficial_or_unpassed_source_is_rejected(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    source = _source(tmp_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest[key] = value
    source.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BoundaryAuditError, match=message):
        audit_source_boundaries(source, inboard_width_m=0.8)
