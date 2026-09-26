"""Package plumbing and manifest checks for the opt-in source-wall layer."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rmuc2026_mujoco import AssetIntegrityError, FieldAsset, ManifestError, load_model
from rmuc2026_mujoco.cli import main
from rmuc2026_mujoco.collision_candidate import SOURCE_GLB_SHA256
from rmuc2026_mujoco.lite_profile import export_interactive_lite_pack
from rmuc2026_mujoco.manifest import sha256_file
from rmuc2026_mujoco.pack import export_runtime_asset_pack
from rmuc2026_mujoco.query import HeightFieldData, load_heightfield
from rmuc2026_mujoco import source_contact_pack
from test_pack_profiles import _synthetic_source_build


@pytest.fixture
def lite_pack_with_stub_source_walls(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Exercise pack wiring without depending on the local official STEP asset."""

    pytest.importorskip("fast_simplification")
    trimesh = pytest.importorskip("trimesh")
    source = _synthetic_source_build(tmp_path / "source_build")
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    visual = source / source_manifest["visual_meshes"][1]["file"]
    payload = trimesh.creation.icosphere(subdivisions=2).export(file_type="obj")
    visual.write_bytes(payload.encode("utf-8") if isinstance(payload, str) else payload)
    source_manifest["visual_meshes"][1]["sha256"] = sha256_file(visual)
    source_manifest["collision"]["resolution_m"] = 1.0
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    full_pack = tmp_path / "full_pack"
    lite_pack = tmp_path / "lite_pack"
    export_runtime_asset_pack(source, full_pack)
    export_interactive_lite_pack(full_pack, lite_pack, target_visual_faces=300)

    def stub_source_walls(_source_build: Path, runtime_pack: Path):
        asset = FieldAsset.open(runtime_pack, verify=True)
        original = load_heightfield(asset)
        changed = original.height_m.copy()
        changed[0, 0] += 0.001
        left = trimesh.creation.box(extents=(0.1, 0.2, 0.5))
        right = left.copy()
        right.apply_translation((0.4, 0.0, 0.0))
        return (
            HeightFieldData(original.x_m, original.y_m, changed),
            {402: left, 403: right},
            {
                "source_identity": {
                    "source_manifest_sha256": asset.manifest["source_identity"][
                        "field_manifest_sha256"
                    ],
                    "source_glb_sha256": SOURCE_GLB_SHA256,
                },
                # Synthetic fixtures cannot reproduce 15,800 official roof
                # samples; those are checked by the real-source constructor.
                "walls": [
                    {"source_part_index": 402, "roof_nodes_transferred": 7900},
                    {"source_part_index": 403, "roof_nodes_transferred": 7900},
                ],
                "validation_boundary": "synthetic package contract test only",
            },
        )

    monkeypatch.setattr(source_contact_pack, "build_verified_wall_replacement", stub_source_walls)
    return source, lite_pack


def _rewrite_manifest(pack: Path, payload: dict) -> None:
    (pack / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_exported_wall_layer_has_identical_physics_in_all_profiles(
    lite_pack_with_stub_source_walls: tuple[Path, Path], tmp_path: Path
) -> None:
    source, lite_pack = lite_pack_with_stub_source_walls
    output = tmp_path / "wall_pack"
    result = source_contact_pack.export_source_wall_pack(source, lite_pack, output)
    asset = FieldAsset.open(output, verify=True)
    layer = result["collision"]["source_contact_layer"]
    assert result["schema_version"] == 4
    assert result["validation_status"] == "DRAFT_BLOCKED"
    assert layer["status"] == "EXPERIMENTAL_BLOCKED"
    assert layer["roof_nodes_transferred"] == 15800
    assert asset.available_runtime_profiles == ("full", "interactive_lite", "collision_only")
    assert (
        asset.collision["samples_sha256"] != FieldAsset.open(lite_pack).collision["samples_sha256"]
    )

    models = {name: load_model(asset, profile=name)[0] for name in asset.available_runtime_profiles}
    reference = models["full"]
    for name, model in models.items():
        np.testing.assert_array_equal(model.hfield_data, reference.hfield_data, err_msg=name)
        for part in (402, 403):
            geom_name = f"rmuc2026_official_wall_{part}"
            geom_id = model.geom(geom_name).id
            assert (int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id])) == (
                2,
                1,
            )
            np.testing.assert_allclose(model.geom_friction[geom_id], (1.0, 0.005, 0.0001))
            np.testing.assert_allclose(model.geom_solref[geom_id], (0.005, 1.0))


def test_wall_asset_hash_and_manifest_contact_contract_reject_tampering(
    lite_pack_with_stub_source_walls: tuple[Path, Path], tmp_path: Path
) -> None:
    source, lite_pack = lite_pack_with_stub_source_walls
    output = tmp_path / "wall_pack"
    source_contact_pack.export_source_wall_pack(source, lite_pack, output)
    mesh = output / "collision/official_wall_402.obj"
    mesh.write_bytes(mesh.read_bytes() + b"# changed\n")
    with pytest.raises(AssetIntegrityError, match="size mismatch|SHA-256 mismatch"):
        FieldAsset.open(output, verify=True)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    mesh.write_bytes(mesh.read_bytes()[:-10])
    manifest["collision"]["source_contact_layer"]["mesh_geoms"][0]["contype"] = 0
    _rewrite_manifest(output, manifest)
    with pytest.raises(ManifestError, match="bits are invalid"):
        FieldAsset.open(output, verify=False)


def test_rehashed_lite_xml_cannot_change_wall_contact(
    lite_pack_with_stub_source_walls: tuple[Path, Path], tmp_path: Path
) -> None:
    source, lite_pack = lite_pack_with_stub_source_walls
    output = tmp_path / "wall_pack"
    source_contact_pack.export_source_wall_pack(source, lite_pack, output)
    asset = FieldAsset.open(output)
    lite_xml = asset.entrypoint_for("interactive_lite")
    xml = ET.parse(lite_xml)
    wall = xml.getroot().find("./worldbody/geom[@name='rmuc2026_official_wall_402']")
    assert wall is not None
    wall.set("friction", "0.2 0.005 0.0001")
    xml.write(lite_xml, encoding="utf-8", xml_declaration=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    file_record = next(
        item for item in manifest["contents"]["files"] if item["file"] == lite_xml.name
    )
    file_record["sha256"] = sha256_file(lite_xml)
    file_record["size_bytes"] = lite_xml.stat().st_size
    _rewrite_manifest(output, manifest)
    with pytest.raises(ManifestError, match="source wall.*friction|source wall.*changed contact"):
        FieldAsset.open(output, verify=True)


def test_source_wall_pack_cli_and_schema_two_rejection(
    lite_pack_with_stub_source_walls: tuple[Path, Path], tmp_path: Path, capsys
) -> None:
    source, lite_pack = lite_pack_with_stub_source_walls
    output = tmp_path / "cli_wall_pack"
    assert main(["source-wall-pack", str(source), str(lite_pack), str(output)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["source_contact_status"] == "EXPERIMENTAL_BLOCKED"
    assert response["schema_version"] == 4
    assert FieldAsset.open(output, verify=True).available_runtime_profiles == (
        "full",
        "interactive_lite",
        "collision_only",
    )
    schema_two = tmp_path / "schema_two"
    export_runtime_asset_pack(source, schema_two)
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="schema-4"):
        source_contact_pack.export_source_wall_pack(source, schema_two, rejected)
    assert not rejected.exists()
