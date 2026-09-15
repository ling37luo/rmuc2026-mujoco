from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib

import pytest

from rmuc2026_mujoco import AssetIntegrityError, FieldAsset, ManifestError, verify_asset


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _grayscale16_png(width: int, height: int, *, compressed: bytes | None = None) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    if compressed is None:
        scanlines = b"".join(b"\x00" + b"\x00\x00" * width for _ in range(height))
        compressed = zlib.compress(scanlines)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _refresh_declared_file(root: Path, manifest: dict[str, object], relative: str) -> None:
    path = root / relative
    record = next(
        item
        for item in manifest["contents"]["files"]
        if item["file"] == relative  # type: ignore[index]
    )
    record["size_bytes"] = path.stat().st_size
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


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


def test_heightfield_manifest_rejects_excessive_sample_capacity(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collision"]["rows_y"] = 100_000
    manifest["collision"]["columns_x"] = 100_000
    manifest["heightfield_precision"]["rows_y"] = 100_000
    manifest["heightfield_precision"]["columns_x"] = 100_000
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match="sample limit"):
        FieldAsset.open(field_asset_dir)


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
    _refresh_declared_file(field_asset_dir, manifest, "rmuc2026_field_collision_only.xml")
    _write_manifest(field_asset_dir, manifest)

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


@pytest.mark.parametrize(
    ("selector", "attribute", "replacement", "error"),
    [
        ("./asset/hfield[@name='rmuc2026_collision']", "nrow", "4", "shape"),
        ("./asset/hfield[@name='rmuc2026_collision']", "ncol", "5", "shape"),
        (
            "./asset/hfield[@name='rmuc2026_collision']",
            "size",
            "1.5 1 2.001 .05",
            "hfield size disagrees",
        ),
        (
            ".//geom[@name='rmuc2026_field_collision']",
            "pos",
            ".500001 0 -.5",
            "geom pos disagrees",
        ),
        (
            "./asset/hfield[@name='rmuc2026_collision']",
            "size",
            "1.5 1 nan .05",
            "finite numbers",
        ),
        (
            ".//geom[@name='rmuc2026_field_collision']",
            "pos",
            ".5 0",
            "3 finite numbers",
        ),
    ],
)
def test_runtime_profile_xml_collision_contract_fails_closed_after_rehash(
    field_asset_dir: Path,
    selector: str,
    attribute: str,
    replacement: str,
    error: str,
) -> None:
    relative = "rmuc2026_field.xml"
    xml_path = field_asset_dir / relative
    tree = ET.parse(xml_path)
    element = tree.getroot().find(selector)
    assert element is not None
    element.set(attribute, replacement)
    tree.write(xml_path, encoding="unicode")

    manifest = json.loads((field_asset_dir / "manifest.json").read_text(encoding="utf-8"))
    _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match=error):
        FieldAsset.open(field_asset_dir)


def test_legacy_file_hfield_shape_is_read_from_declared_png_ihdr(
    field_asset_dir: Path,
) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_relative = "collision/heightfield.png"
    image = field_asset_dir / image_relative
    image.write_bytes(_grayscale16_png(4, 3))
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest["collision"]["image_sha256"] = image_sha
    _refresh_declared_file(field_asset_dir, manifest, image_relative)

    for relative in ("rmuc2026_field.xml", "rmuc2026_field_collision_only.xml"):
        xml_path = field_asset_dir / relative
        tree = ET.parse(xml_path)
        hfield = tree.getroot().find("./asset/hfield[@name='rmuc2026_collision']")
        assert hfield is not None
        hfield.attrib.pop("nrow")
        hfield.attrib.pop("ncol")
        hfield.set("file", image_relative)
        tree.write(xml_path, encoding="unicode")
        _refresh_declared_file(field_asset_dir, manifest, relative)

    _write_manifest(field_asset_dir, manifest)
    assert FieldAsset.open(field_asset_dir).collision["rows_y"] == 3


def test_legacy_file_hfield_png_shape_mismatch_fails_closed(field_asset_dir: Path) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_relative = "collision/heightfield.png"
    image = field_asset_dir / image_relative
    image.write_bytes(_grayscale16_png(5, 3))
    manifest["collision"]["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    _refresh_declared_file(field_asset_dir, manifest, image_relative)

    relative = "rmuc2026_field.xml"
    xml_path = field_asset_dir / relative
    tree = ET.parse(xml_path)
    hfield = tree.getroot().find("./asset/hfield[@name='rmuc2026_collision']")
    assert hfield is not None
    hfield.attrib.pop("nrow")
    hfield.attrib.pop("ncol")
    hfield.set("file", image_relative)
    tree.write(xml_path, encoding="unicode")
    _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match="shape|dimensions"):
        FieldAsset.open(field_asset_dir)


def test_file_hfield_checks_png_shape_even_when_xml_declares_shape(
    field_asset_dir: Path,
) -> None:
    manifest_path = field_asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_relative = "collision/heightfield.png"
    image = field_asset_dir / image_relative
    image.write_bytes(_grayscale16_png(5, 3))
    manifest["collision"]["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    _refresh_declared_file(field_asset_dir, manifest, image_relative)

    for relative in ("rmuc2026_field.xml", "rmuc2026_field_collision_only.xml"):
        xml_path = field_asset_dir / relative
        tree = ET.parse(xml_path)
        hfield = tree.getroot().find("./asset/hfield[@name='rmuc2026_collision']")
        assert hfield is not None
        hfield.set("file", image_relative)
        tree.write(xml_path, encoding="unicode")
        _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match="PNG dimensions|shape"):
        FieldAsset.open(field_asset_dir)


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("crc", "invalid CRC"),
        ("zlib", "IDAT zlib"),
        ("truncated", "IEND"),
    ],
)
def test_file_hfield_png_must_be_complete_and_decodable(
    field_asset_dir: Path,
    corruption: str,
    error: str,
) -> None:
    manifest = json.loads((field_asset_dir / "manifest.json").read_text(encoding="utf-8"))
    image_relative = "collision/heightfield.png"
    image = field_asset_dir / image_relative
    payload = _grayscale16_png(4, 3)
    if corruption == "crc":
        damaged = bytearray(payload)
        damaged[41] ^= 1  # first byte of the IDAT payload, without updating its CRC
        payload = bytes(damaged)
    elif corruption == "zlib":
        payload = _grayscale16_png(4, 3, compressed=b"not-zlib")
    else:
        payload = payload[:-12]  # complete IEND chunk
    image.write_bytes(payload)
    manifest["collision"]["image_sha256"] = hashlib.sha256(payload).hexdigest()
    _refresh_declared_file(field_asset_dir, manifest, image_relative)

    relative = "rmuc2026_field.xml"
    xml_path = field_asset_dir / relative
    tree = ET.parse(xml_path)
    hfield = tree.getroot().find("./asset/hfield[@name='rmuc2026_collision']")
    assert hfield is not None
    hfield.set("file", image_relative)
    tree.write(xml_path, encoding="unicode")
    _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match=error):
        FieldAsset.open(field_asset_dir)


def test_runtime_profile_rejects_rotated_canonical_collision_geom(
    field_asset_dir: Path,
) -> None:
    relative = "rmuc2026_field.xml"
    xml_path = field_asset_dir / relative
    tree = ET.parse(xml_path)
    geom = tree.getroot().find("./worldbody/geom[@name='rmuc2026_field_collision']")
    assert geom is not None
    geom.set("quat", "0.9238795 0 0.3826834 0")
    tree.write(xml_path, encoding="unicode")

    manifest = json.loads((field_asset_dir / "manifest.json").read_text(encoding="utf-8"))
    _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match="axis-aligned|transform"):
        FieldAsset.open(field_asset_dir)


def test_runtime_profile_rejects_nested_canonical_collision_geom(
    field_asset_dir: Path,
) -> None:
    relative = "rmuc2026_field.xml"
    xml_path = field_asset_dir / relative
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("./worldbody")
    geom = root.find("./worldbody/geom[@name='rmuc2026_field_collision']")
    assert worldbody is not None and geom is not None
    worldbody.remove(geom)
    body = ET.SubElement(worldbody, "body", {"pos": "1 0 0"})
    body.append(geom)
    tree.write(xml_path, encoding="unicode")

    manifest = json.loads((field_asset_dir / "manifest.json").read_text(encoding="utf-8"))
    _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match="direct worldbody child"):
        FieldAsset.open(field_asset_dir)


def test_runtime_profile_rejects_inherited_default_collision_transform(
    field_asset_dir: Path,
) -> None:
    relative = "rmuc2026_field.xml"
    xml_path = field_asset_dir / relative
    tree = ET.parse(xml_path)
    root = tree.getroot()
    default = ET.Element("default")
    ET.SubElement(default, "geom", {"quat": "0.9238795 0 0.3826834 0"})
    root.insert(0, default)
    tree.write(xml_path, encoding="unicode")

    manifest = json.loads((field_asset_dir / "manifest.json").read_text(encoding="utf-8"))
    _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match="default geom.*transform"):
        FieldAsset.open(field_asset_dir)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("include", "must not load MJCF include files"),
        ("mesh_path", "visual mesh files disagree"),
        ("mesh_scale", "retain manifest coordinates"),
        ("file_texture", "unsupported file-backed assets"),
    ],
)
def test_runtime_profile_rejects_undeclared_or_transformed_external_assets(
    field_asset_dir: Path,
    mutation: str,
    error: str,
) -> None:
    relative = "rmuc2026_field.xml"
    xml_path = field_asset_dir / relative
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if mutation == "include":
        ET.SubElement(root, "include", {"file": "outside.xml"})
    elif mutation == "mesh_path":
        mesh = root.find("./asset/mesh")
        assert mesh is not None
        mesh.set("file", "../../outside.obj")
    elif mutation == "mesh_scale":
        mesh = root.find("./asset/mesh")
        assert mesh is not None
        mesh.set("scale", "2 2 2")
    else:
        asset = root.find("./asset")
        assert asset is not None
        ET.SubElement(asset, "texture", {"name": "outside", "file": "outside.png"})
    tree.write(xml_path, encoding="unicode")

    manifest = json.loads((field_asset_dir / "manifest.json").read_text(encoding="utf-8"))
    _refresh_declared_file(field_asset_dir, manifest, relative)
    _write_manifest(field_asset_dir, manifest)

    with pytest.raises(ManifestError, match=error):
        FieldAsset.open(field_asset_dir)
