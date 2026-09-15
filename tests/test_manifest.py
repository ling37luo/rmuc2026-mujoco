from __future__ import annotations

import json
from pathlib import Path

import pytest

from rmuc2026_mujoco import AssetIntegrityError, FieldAsset, ManifestError, verify_asset


def test_open_and_verify_runtime_pack(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)
    report = verify_asset(field_asset_dir)
    assert asset.root == field_asset_dir.resolve()
    assert asset.entrypoint.name == "rmuc2026_field.xml"
    assert asset.available_runtime_profiles == ("full", "collision_only")
    assert asset.entrypoint_for("collision_only").name == "rmuc2026_field_collision_only.xml"
    assert report.visual_mesh_count == 1
    assert report.verified_file_count == 5
    assert report.collision_shape == (3, 4)
    assert report.validation_status == "DRAFT_BLOCKED"


def test_hash_tamper_fails_closed(field_asset_dir: Path) -> None:
    (field_asset_dir / "visual/tetra.obj").write_text("o changed\n", encoding="ascii")
    with pytest.raises(AssetIntegrityError, match="size mismatch|SHA-256 mismatch"):
        FieldAsset.open(field_asset_dir)


def test_path_escape_fails_even_without_hash_verification(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contents"]["files"][1]["file"] = "../outside.obj"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (field_asset_dir.parent / "outside.obj").write_text("o outside\n", encoding="ascii")
    with pytest.raises(AssetIntegrityError, match="escapes"):
        FieldAsset.open(field_asset_dir, verify=False)


def test_legacy_build_manifest_is_not_public_input(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_type"] = "rmuc2026_official_field_mujoco_asset"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="runtime pack"):
        FieldAsset.open(field_asset_dir, verify=False)


def test_asset_pack_cannot_overstate_redistribution_rights(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["distribution"]["safe_to_publish_without_rightsholder_permission"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="distribution boundary"):
        FieldAsset.open(field_asset_dir, verify=False)


def test_asset_pack_cannot_overstate_topology_readiness(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_boundary"]["whole_field_topology_ready"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="whole-field collision topology"):
        FieldAsset.open(field_asset_dir, verify=False)


def test_runtime_profile_metadata_fails_closed(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_profiles"]["profiles"]["collision_only"]["includes_visual_meshes"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="includes_visual_meshes is inconsistent"):
        FieldAsset.open(field_asset_dir, verify=False)


def test_collision_only_profile_rejects_hidden_mesh_reference(field_asset_dir: Path) -> None:
    collision_xml = field_asset_dir / "rmuc2026_field_collision_only.xml"
    collision_xml.write_text(
        collision_xml.read_text(encoding="utf-8").replace(
            "<asset>",
            '<asset><mesh name="hidden" file="visual/tetra.obj"/>',
        ),
        encoding="utf-8",
    )
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["contents"]["files"]
        if item["file"] == "rmuc2026_field_collision_only.xml"
    )
    import hashlib

    record["size_bytes"] = collision_xml.stat().st_size
    record["sha256"] = hashlib.sha256(collision_xml.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="declares 1 mesh assets; expected 0"):
        FieldAsset.open(field_asset_dir)


def test_pre_profile_runtime_pack_remains_full_only_compatible(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["runtime_profiles"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    asset = FieldAsset.open(field_asset_dir)
    assert asset.available_runtime_profiles == ("full",)
    assert asset.entrypoint_for("full") == asset.entrypoint
    with pytest.raises(ManifestError, match="collision_only.*unavailable"):
        asset.entrypoint_for("collision_only")
