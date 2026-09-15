from __future__ import annotations

import json
from pathlib import Path

from rmuc2026_mujoco import DownloadedStep
from rmuc2026_mujoco.cli import build_parser, main


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


def test_setup_cli_has_a_safe_user_cache_default() -> None:
    args = build_parser().parse_args(["setup", "local-field-pack", "--acknowledge-reference-only"])

    assert args.step_cache.name == "RMUC2026_V2.0.0.stp"
    assert args.step_cache.parent.name == "rmuc2026-mujoco"
    assert args.output == Path("local-field-pack")


def test_view_cli_defaults_to_full_without_friction_override() -> None:
    args = build_parser().parse_args(["view", "local-field-pack"])

    assert args.profile == "full"
    assert args.friction_preset is None


def test_view_cli_accepts_headless_profile_and_unofficial_friction() -> None:
    args = build_parser().parse_args(
        ["view", "local-field-pack", "--profile", "collision_only", "--friction", "low"]
    )

    assert args.profile == "collision_only"
    assert args.friction_preset == "low"


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
