from __future__ import annotations

import json
from pathlib import Path

from rmuc2026_mujoco import DownloadedStep, FieldAsset
from rmuc2026_mujoco.cli import _configure_camera, build_parser, main


def test_verify_cli_emits_machine_readable_pass(field_asset_dir: Path, capsys) -> None:
    assert main(["verify", str(field_asset_dir)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["collision_shape"] == [3, 4]
    assert output["validation_status"] == "DRAFT_BLOCKED"


def test_source_cli_is_read_only_and_reports_unverified_redistribution(capsys) -> None:
    assert main(["source"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["download_started"] is False
    assert output["included_in_package"] is False
    assert output["redistribution_permission"] == "UNVERIFIED"
    assert output["redistribution_authorized"] is False
    assert output["license_review"] == "NO_EXPLICIT_GRANT_FOUND"
    review = output["rulebook_review"]
    assert review["official_rule_centre"] == (
        "https://bbs.robomaster.com/wiki/20204847/809871?source=7"
    )
    assert review["latest_reviewed_version"] == "V2.2.0"
    assert review["latest_reviewed_publication_date"] == "2026-08-07"
    assert review["latest_reviewed_size_bytes"] == 21_018_604
    assert review["latest_reviewed_sha256"] == (
        "88ae3c7d0bbd7312c8095eff2603910437e292a0b53f46c9378f26144c09a6f5"
    )
    assert review["latest_reviewed_url"].endswith("%E5%86%8CV2.2.0%EF%BC%8820260807%EF%BC%89.pdf")
    assert review["comparison_scope"] == "rulebook_chapter_4_only"
    assert review["static_field_chapter_vs_v2_0_0"] == "NO_CHANGE_DETECTED"
    assert review["comparison_method"] == "normalised_text_and_embedded_image_hashes"
    assert review["future_updates_require_new_review"] is True


def test_setup_cli_has_a_safe_user_cache_default() -> None:
    args = build_parser().parse_args(["setup", "local-field-pack", "--acknowledge-reference-only"])

    assert args.step_cache.name == "RMUC2026_V2.0.0.stp"
    assert args.step_cache.parent.name == "rmuc2026-mujoco"
    assert args.output == Path("local-field-pack")


def test_view_cli_defaults_to_full_without_friction_override() -> None:
    args = build_parser().parse_args(["view", "local-field-pack"])

    assert args.profile == "full"
    assert args.friction_preset is None
    assert args.camera == "overview"


def test_view_cli_accepts_headless_profile_and_unofficial_friction() -> None:
    args = build_parser().parse_args(
        ["view", "local-field-pack", "--profile", "collision_only", "--friction", "low"]
    )

    assert args.profile == "collision_only"
    assert args.friction_preset == "low"


def test_overview_camera_uses_verified_field_bounds(field_asset_dir: Path) -> None:
    class Camera:
        lookat = [99.0, 99.0, 99.0]
        distance = 0.0
        azimuth = 0.0
        elevation = 0.0

    class Viewer:
        cam = Camera()

    viewer = Viewer()
    _configure_camera(viewer, FieldAsset.open(field_asset_dir), "overview")

    assert viewer.cam.lookat == [0.5, 0.0, 0.0]
    assert viewer.cam.distance == 2.55
    assert viewer.cam.azimuth == 135.0
    assert viewer.cam.elevation == -50.0


def test_setup_cli_downloads_then_builds_locally(monkeypatch, tmp_path: Path, capsys) -> None:
    step = tmp_path / "cache" / "RMUC2026_V2.0.0.stp"
    output = tmp_path / "runtime-pack"
    calls: list[tuple[str, Path]] = []

    def fake_download(destination, *, acknowledge_reference_only, progress):
        assert acknowledge_reference_only is True
        assert progress is not None
        calls.append(("download", Path(destination)))
        return DownloadedStep(
            path=step,
            size_bytes=1_254_821_405,
            sha256="8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33",
            reused=False,
        )

    def fake_build(step_path, output_dir, **options):
        assert Path(step_path) == step
        assert options == {
            "target_visual_faces": 450_000,
            "heightfield_resolution_m": 0.02,
        }
        calls.append(("build", Path(output_dir)))
        return {
            "artifact_type": "rmuc2026_mujoco_runtime_asset_pack",
            "status": "PASS",
            "validation_status": "DRAFT_BLOCKED",
        }

    monkeypatch.setattr("rmuc2026_mujoco.cli.download_official_step", fake_download)
    monkeypatch.setattr("rmuc2026_mujoco.cli.build_runtime_asset_pack", fake_build)

    assert (
        main(
            [
                "setup",
                str(output),
                "--step-cache",
                str(step),
                "--acknowledge-reference-only",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert calls == [("download", step), ("build", output)]
    assert result["status"] == "PASS"
    assert result["official_step_reused"] is False


def test_setup_cli_rejects_unsafe_build_resolution(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "setup",
                str(tmp_path / "output"),
                "--heightfield-resolution",
                "0.5",
                "--acknowledge-reference-only",
            ]
        )
        == 2
    )
    assert "within [0.02, 0.10]" in capsys.readouterr().err


def test_surface_cli_emits_geometry_and_claim_boundary(
    field_asset_dir: Path,
    capsys,
) -> None:
    assert (
        main(
            [
                "surface",
                str(field_asset_dir),
                "-0.5",
                "-0.5",
                "--window-radius",
                "0.51",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["height_m"] == 0.25
    assert output["query_schema_version"] == 1
    assert output["query_algorithm"] == "mujoco_hfield_triangle_v1"
    assert output["manifest_sha256"]
    assert output["heightfield_samples_sha256"]
    assert output["topology_verified"] is False
    assert "underpasses" in output["claim_boundary"]


def test_spawns_cli_reports_screening_contract_without_overclaiming(
    field_asset_dir: Path,
    capsys,
) -> None:
    assert (
        main(
            [
                "spawns",
                str(field_asset_dir),
                "--count",
                "2",
                "--footprint-radius",
                "0",
                "--boundary-margin",
                "0",
                "--minimum-separation",
                "1",
                "--max-slope-deg",
                "90",
                "--max-relief",
                "2",
                "--ground-height-min",
                "-1",
                "--ground-height-max",
                "2",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "RUNTIME_PROXY_SCREENED_NOT_TOPOLOGY_VERIFIED"
    assert output["query_schema_version"] == 1
    assert output["query_algorithm"] == "mujoco_hfield_triangle_spawn_screen_v1"
    assert output["manifest_sha256"]
    assert output["heightfield_samples_sha256"]
    assert output["candidate_count"] == 2
    assert output["topology_verified"] is False
    assert output["selection_contract"]["slope_interpolation"] == "mujoco_hfield_triangle"
    assert (
        output["selection_contract"]["ground_height_range_applies_to"] == "entire_screening_window"
    )
    assert all(candidate["topology_verified"] is False for candidate in output["candidates"])


def test_spawns_cli_rejects_inverted_height_range(field_asset_dir: Path, capsys) -> None:
    assert (
        main(
            [
                "spawns",
                str(field_asset_dir),
                "--ground-height-min",
                "1",
                "--ground-height-max",
                "-1",
            ]
        )
        == 2
    )
    assert "ordered" in capsys.readouterr().err
