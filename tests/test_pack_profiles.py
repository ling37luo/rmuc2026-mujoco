from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest
from PIL import Image
import mujoco

from rmuc2026_mujoco import (
    AssetIntegrityError,
    FieldAsset,
    FieldBoundaryGuard,
    ManifestError,
    compose_with_robot,
    height_at,
    load_model,
)
from rmuc2026_mujoco.collision_candidate import SOURCE_GLB_SHA256
from rmuc2026_mujoco.download import OFFICIAL_STEP_SHA256, OFFICIAL_STEP_SIZE
from rmuc2026_mujoco.pack import (
    PUBLIC_DISPLAY_BASE_SURFACE_SCALE,
    PUBLIC_DISPLAY_PROFILE,
    ExportBlocked,
    FILL_LIGHT_NAME,
    KEY_LIGHT_NAME,
    SURFACE_GUIDE_BAKED_CONTENT,
    SURFACE_GUIDE_KIND,
    SURFACE_GUIDE_TEXTURE_FILE,
    _display_rgba,
    _mujoco_triangle_height_samples,
    _write_surface_guide_mesh,
    export_runtime_asset_pack,
)
from rmuc2026_mujoco.livery import (
    GROUND_MARKING_KNOWN_LIMITATIONS,
    GROUND_MARKING_OVERLAY_KIND,
    GROUND_MARKING_PROCESSING_ALGORITHM,
    GROUND_MARKING_REMOVED_BY_DESIGN,
    GROUND_MARKING_RETAINED_CONTENT,
)
from rmuc2026_mujoco.lite_profile import export_interactive_lite_pack
from rmuc2026_mujoco.perimeter_fence import (
    FENCE_NAMES,
    HEIGHTFIELD_EDGE_FENCE_SCHEMA,
    LEGACY_FENCE_SCHEMA,
    RAMP_CLEARANCE_FENCE_SCHEMA,
    SOFT_CONTACT_FENCE_SCHEMA,
    export_fenced_pack,
    perimeter_fence_contract,
)
from rmuc2026_mujoco.ramp_audit import FIXED_FLY_RAMPS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _heightfield_bootstrap_png(
    height_m: np.ndarray, maximum_height_m: float, minimum_height_m: float = 0.0
) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    quantized = np.rint(
        np.clip(
            (height_m - minimum_height_m) / (maximum_height_m - minimum_height_m),
            0.0,
            1.0,
        )
        * 65535.0
    ).astype(np.uint16)
    rows, columns = quantized.shape
    ihdr = struct.pack(">IIBBBBB", columns, rows, 16, 0, 0, 0, 0)
    scanlines = b"".join(
        b"\x00" + np.flipud(quantized)[row].astype(">u2").tobytes() for row in range(rows)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _rgb8_png(width: int, height: int, *, rgb: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixel = bytes(rgb)
    scanlines = b"".join(b"\x00" + pixel * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _synthetic_source_build(root: Path, *, include_surface_guide: bool = False) -> Path:
    visual_dir = root / "visual"
    collision_dir = root / "collision"
    visual_dir.mkdir(parents=True)
    collision_dir.mkdir()
    visual_meshes = []
    tetra = "\n".join(
        [
            "o tetra",
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "v 0 0 1",
            "f 1 3 2",
            "f 1 2 4",
            "f 2 3 4",
            "f 3 1 4",
            "",
        ]
    )
    for index in range(38):
        relative = f"visual/mesh_{index:02d}.obj"
        path = root / relative
        path.write_text(tetra, encoding="ascii")
        visual_meshes.append(
            {
                "file": relative,
                "sha256": _sha(path),
                "material_id": f"synthetic_{index}",
                "rgba": [0.2, 0.4, 0.6, 1.0],
                "visual_role": "base_surface_shell" if index == 0 else "cad_structure",
            }
        )
    image = collision_dir / "heightfield.png"
    image.write_bytes(_heightfield_bootstrap_png(np.asarray([[0.0, 0.1], [0.1, 0.2]]), 1.0))
    samples = collision_dir / "heightfield.npz"
    np.savez_compressed(
        samples,
        x_m=np.asarray([-0.5, 0.5]),
        y_m=np.asarray([-0.5, 0.5]),
        height_m=np.asarray([[0.0, 0.1], [0.1, 0.2]]),
    )
    manifest = {
        "artifact_type": "rmuc2026_official_field_mujoco_asset",
        "status": "PASS",
        "source": {
            "sha256": OFFICIAL_STEP_SHA256,
            "size_bytes": OFFICIAL_STEP_SIZE,
            "step_product": "synthetic-test",
            "official_download_url": "https://example.invalid/official.stp",
        },
        "validation_scope": {
            "status": "DRAFT_BLOCKED",
            "profile_id": "synthetic-test",
            "claim_boundary": "synthetic test only",
            "blocking_reasons": ["synthetic"],
            "whole_field_topology_ready": False,
            "final_policy_validation_ready": False,
        },
        "visual_meshes": visual_meshes,
        "collision": {
            "kind": "top_surface_heightfield_proxy",
            "image_file": "collision/heightfield.png",
            "image_sha256": _sha(image),
            "samples_file": "collision/heightfield.npz",
            "samples_sha256": _sha(samples),
            "rows_y": 2,
            "columns_x": 2,
            "half_size_xy_m": [0.5, 0.5],
            "geom_center_after_translation_m": [0.0, 0.0, 0.0],
            "maximum_height_m": 1.0,
            "minimum_height_m": 0.0,
            "base_depth_m": 0.05,
            "resolution_m": 0.5,
            "png_rows": "flipped_y_for_mujoco_hfield_loader",
            "claim_boundary": "synthetic test only",
        },
        "recommended_spawn": {
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
        "dimensions": {
            "official_core_battlefield_m": [1.0, 1.0],
            "cad_assembly_outer_bounds_after_translation_m": [
                [-0.5, -0.5, 0.0],
                [0.5, 0.5, 1.0],
            ],
            "cad_assembly_outer_extents_m": [1.0, 1.0, 1.0],
        },
    }
    if include_surface_guide:
        guide = visual_dir / "official_rulebook_v2_overhead_surface.png"
        # Neutral pixels distinguish complete-picture preservation from the
        # earlier chromatic-only extractor, which rejected this image.
        guide.write_bytes(_rgb8_png(4, 4, rgb=(64, 64, 64)))
        manifest["surface_guide"] = {
            "kind": "official_rulebook_overhead_render_surface_guide",
            "file": "visual/official_rulebook_v2_overhead_surface.png",
            "sha256": _sha(guide),
            "width_px": 4,
            "height_px": 4,
            "physics": False,
            "world_mapping": {
                "image_left_to_world": "negative_x_red_side",
                "image_right_to_world": "positive_x_blue_side",
                "image_top_to_world": "positive_y",
                "rectangle": "official_cad_assembly_outer_xy_bounds",
                "world_bounds_xy_m": [[-0.5, -0.5], [0.5, 0.5]],
                "world_size_xy_m": [1.0, 1.0],
                "calibration": "diagnostic_outer_bounds_fit_not_survey_homography",
            },
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _negative_source_build(root: Path) -> Path:
    source = _synthetic_source_build(root)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collision = manifest["collision"]
    height = np.asarray([[-5.0, 0.1], [0.1, 0.2]])
    samples_path = source / collision["samples_file"]
    np.savez_compressed(
        samples_path,
        x_m=np.asarray([-0.5, 0.5]),
        y_m=np.asarray([-0.5, 0.5]),
        height_m=height,
    )
    image_path = source / collision["image_file"]
    image_path.write_bytes(_heightfield_bootstrap_png(height, 1.0, -5.0))
    mask_path = source / "collision/edge_void_mask.npz"
    np.savez_compressed(mask_path, changed_mask=np.asarray([[True, False], [False, False]]))
    collision["minimum_height_m"] = -5.0
    collision["resolution_m"] = 1.0
    collision["image_sha256"] = _sha(image_path)
    collision["samples_sha256"] = _sha(samples_path)
    collision["edge_void_provenance"] = {
        "status": "PASS",
        "source_glb_sha256": SOURCE_GLB_SHA256,
        "mask_file": "collision/edge_void_mask.npz",
        "mask_sha256": _sha(mask_path),
        "changed_count": 1,
        "edge_band_m": 1.25,
        "sentinel_height_m": -5.0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_negative_heightfield_uses_schema3_offset_encoding(tmp_path: Path) -> None:
    source = _negative_source_build(tmp_path / "source")
    output = tmp_path / "pack"
    exported = export_runtime_asset_pack(source, output)
    asset = FieldAsset.open(output)
    assert exported["schema_version"] == 3
    assert exported["collision"]["edge_void_provenance"]["changed_count"] == 1
    assert height_at(asset, -0.5, -0.5) == pytest.approx(-5.0)
    guard = FieldBoundaryGuard.from_asset(asset)
    assert guard._over_source_miss(-0.5, -0.5)
    assert not guard._over_source_miss(0.5, 0.5)
    model, _ = load_model(asset, profile="collision_only")
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "rmuc2026_collision")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc2026_field_collision")
    assert model.hfield_size[hid, 2] == pytest.approx(6.0)
    assert model.geom_pos[gid, 2] == pytest.approx(-5.0)
    expected = np.asarray([[0.0, 5.1 / 6.0], [5.1 / 6.0, 5.2 / 6.0]])
    np.testing.assert_allclose(model.hfield_data.reshape(2, 2), expected, atol=2e-7, rtol=0)


def test_interactive_lite_uses_same_collision_with_separate_hashed_visuals(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fast_simplification")
    import trimesh

    source = _synthetic_source_build(tmp_path / "source")
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    visual_path = source / source_manifest["visual_meshes"][1]["file"]
    visual_payload = trimesh.creation.icosphere(subdivisions=2).export(file_type="obj")
    visual_path.write_bytes(
        visual_payload.encode("utf-8") if isinstance(visual_payload, str) else visual_payload
    )
    source_manifest["visual_meshes"][1]["sha256"] = _sha(visual_path)
    source_manifest["collision"]["resolution_m"] = 1.0
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    full_pack = tmp_path / "full_pack"
    lite_pack = tmp_path / "lite_pack"
    export_runtime_asset_pack(source, full_pack)
    result = export_interactive_lite_pack(full_pack, lite_pack, target_visual_faces=300)
    asset = FieldAsset.open(lite_pack)
    assert result["schema_version"] == 4
    assert result["visual_lite"]["source_visual_faces"] > 300
    assert result["visual_lite"]["output_visual_faces"] <= 300
    assert asset.available_runtime_profiles == ("full", "interactive_lite", "collision_only")
    assert asset.manifest["validation_status"] == "DRAFT_BLOCKED"
    assert asset.entrypoint_for("interactive_lite").name == "rmuc2026_field_interactive_lite.xml"

    full, _ = load_model(asset, profile="full")
    lite, _ = load_model(asset, profile="interactive_lite")
    assert full.nmesh == lite.nmesh == 38
    assert full.ngeom == lite.ngeom
    np.testing.assert_array_equal(full.geom_contype, lite.geom_contype)
    np.testing.assert_array_equal(full.geom_conaffinity, lite.geom_conaffinity)
    np.testing.assert_array_equal(full.geom_friction, lite.geom_friction)
    np.testing.assert_array_equal(full.geom_solref, lite.geom_solref)
    np.testing.assert_array_equal(full.hfield_data, lite.hfield_data)

    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text(
        '<mujoco model="robot"><worldbody><body name="robot" pos="0 0 1">'
        '<freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    composed, _ = compose_with_robot(asset, robot_xml, profile="interactive_lite")
    assert composed.nmesh == 38

    lite_xml = asset.entrypoint_for("interactive_lite")
    original_xml = lite_xml.read_bytes()
    lite_xml.write_bytes(
        original_xml.replace(b'friction="1 0.005 0.0001"', b'friction=".2 .005 .0001"')
    )
    changed_manifest = json.loads((lite_pack / "manifest.json").read_text(encoding="utf-8"))
    xml_record = next(
        record
        for record in changed_manifest["contents"]["files"]
        if record["file"] == lite_xml.name
    )
    xml_record["size_bytes"] = lite_xml.stat().st_size
    xml_record["sha256"] = _sha(lite_xml)
    (lite_pack / "manifest.json").write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="changes MJCF beyond visual mesh files"):
        FieldAsset.open(lite_pack, verify=False)
    lite_xml.write_bytes(original_xml)
    (lite_pack / "manifest.json").write_text(json.dumps(result), encoding="utf-8")

    lite_record = result["visual_lite"]["meshes"][1]
    (lite_pack / lite_record["file"]).write_text("o tampered\n", encoding="ascii")
    with pytest.raises(AssetIntegrityError, match="size mismatch|SHA-256 mismatch"):
        FieldAsset.open(lite_pack)


def test_interactive_lite_rejects_a_noop_budget(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source")
    full_pack = tmp_path / "full_pack"
    export_runtime_asset_pack(source, full_pack)
    with pytest.raises(ExportBlocked, match="cannot improve"):
        export_interactive_lite_pack(full_pack, tmp_path / "lite_pack", target_visual_faces=1_000)


def test_schema3_rejects_mask_that_disagrees_with_negative_samples(tmp_path: Path) -> None:
    source = _negative_source_build(tmp_path / "source")
    output = tmp_path / "pack"
    export_runtime_asset_pack(source, output)
    mask_path = output / "collision/edge_void_mask.npz"
    np.savez_compressed(mask_path, changed_mask=np.zeros((2, 2), dtype=np.bool_))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha(mask_path)
    manifest["collision"]["edge_void_provenance"]["mask_sha256"] = digest
    record = next(
        item
        for item in manifest["contents"]["files"]
        if item["file"] == "collision/edge_void_mask.npz"
    )
    record["sha256"] = digest
    record["size_bytes"] = mask_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="mask does not match"):
        FieldAsset.open(output)


def test_export_emits_full_and_collision_only_runtime_profiles(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source")
    output = tmp_path / "pack"

    result = export_runtime_asset_pack(source, output)
    asset = FieldAsset.open(output)

    assert result["runtime_profiles"]["default"] == "full"
    assert asset.available_runtime_profiles == ("full", "collision_only")
    full = ET.parse(asset.entrypoint_for("full")).getroot()
    collision_only = ET.parse(asset.entrypoint_for("collision_only")).getroot()
    assert full.find("./option").attrib == {"timestep": "0.002", "solver": "Newton"}
    assert collision_only.find("./option").attrib == {
        "timestep": "0.002",
        "solver": "Newton",
    }
    assert len(full.findall("./asset/mesh")) == 38
    assert len([geom for geom in full.iter("geom") if geom.get("mesh")]) == 38
    materials = full.findall("./asset/material")
    assert len(materials) == 38
    assert [float(value) for value in materials[0].get("rgba", "").split()] == pytest.approx(
        _display_rgba([0.2, 0.4, 0.6, 1.0], visual_role="base_surface_shell")
    )
    assert [float(value) for value in materials[1].get("rgba", "").split()] == pytest.approx(
        _display_rgba([0.2, 0.4, 0.6, 1.0], visual_role="cad_structure")
    )
    assert all(
        base < structure
        for base, structure in zip(
            result["visual_meshes"][0]["display_rgba"][:3],
            result["visual_meshes"][1]["display_rgba"][:3],
        )
    )
    assert result["visual_display"]["profile"] == PUBLIC_DISPLAY_PROFILE
    assert result["visual_display"]["base_surface_scale"] == PUBLIC_DISPLAY_BASE_SURFACE_SCALE
    assert result["schema_version"] == 2
    assert result["visual_layers"] == {}
    assert result["visual_display"]["lighting"] == {
        "key_light_name": KEY_LIGHT_NAME,
        "fill_light_name": FILL_LIGHT_NAME,
        "default_mode": "flat",
        "modes": ["flat", "shadow"],
        "toggle_key": "L",
        "physics_changed": False,
    }
    assert full.find(f"./worldbody/light[@name='{KEY_LIGHT_NAME}']").get("castshadow") == "false"
    assert collision_only.findall("./asset/mesh") == []
    assert [geom for geom in collision_only.iter("geom") if geom.get("mesh")] == []
    collision_geom = collision_only.find("./worldbody/geom[@name='rmuc2026_field_collision']")
    assert collision_geom is not None
    assert collision_geom.get("rgba", "").split()[-1] == "1"


def test_fenced_pack_is_new_hash_bound_physical_pack(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source")
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["dimensions"]["official_core_battlefield_m"] = [28.0, 15.0]
    source_manifest["collision"]["resolution_m"] = 1.0
    source_manifest["collision"]["half_size_xy_m"] = [15.0, 8.0]
    source_manifest["collision"]["rows_y"] = 17
    source_manifest["collision"]["columns_x"] = 31
    height = np.zeros((17, 31), dtype=np.float64)
    height[-1, -1] = 0.2
    image = source / source_manifest["collision"]["image_file"]
    image.write_bytes(_heightfield_bootstrap_png(height, 1.0))
    source_manifest["collision"]["image_sha256"] = _sha(image)
    samples = source / source_manifest["collision"]["samples_file"]
    np.savez_compressed(
        samples,
        x_m=np.linspace(-15.0, 15.0, 31),
        y_m=np.linspace(-8.0, 8.0, 17),
        height_m=height,
    )
    source_manifest["collision"]["samples_sha256"] = _sha(samples)
    source_manifest["dimensions"]["cad_assembly_outer_bounds_after_translation_m"] = [
        [-14.876004137464, -8.001565456391, 0.0],
        [14.876004137464, 8.001565456391, 1.0],
    ]
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    plain = tmp_path / "plain"
    fenced = tmp_path / "fenced"
    export_runtime_asset_pack(source, plain)
    contract = export_fenced_pack(plain, fenced)
    plain_asset = FieldAsset.open(plain)
    fenced_asset = FieldAsset.open(fenced)
    assert plain_asset.manifest.get("perimeter_fence") is None
    assert fenced_asset.manifest["perimeter_fence"] == contract
    assert contract["official_fence_top_above_field_floor_m"] == 2.4
    assert len(contract["panels"]) == 4
    assert contract["outward_offset_xy_m"] == [0.0, 0.0]
    assert contract["contact"]["solref"] == [0.04, 1.0]
    alignment = contract["playable_deck_edge_alignment"]
    assert alignment["inferred_core_bounds_xy_m"] == [[-14.0, -7.5], [14.0, 7.5]]
    assert alignment["wall_inner_face_overlap_into_core_xy_m"] == pytest.approx([0.025, 0.025])
    assert alignment["traversable_lower_skirt_inside_wall_m"] == [0.0, 0.0, 0.0, 0.0]
    panels = {panel["name"]: panel for panel in contract["panels"]}
    half_thickness = contract["thickness_m"] / 2
    assert panels["rmuc2026_perimeter_left"]["pos"][0] + half_thickness == pytest.approx(-13.975)
    assert panels["rmuc2026_perimeter_right"]["pos"][0] - half_thickness == pytest.approx(13.975)
    assert panels["rmuc2026_perimeter_bottom"]["pos"][1] + half_thickness == pytest.approx(-7.475)
    assert panels["rmuc2026_perimeter_top"]["pos"][1] - half_thickness == pytest.approx(7.475)
    legacy = tmp_path / "legacy-fenced"
    legacy_contract = export_fenced_pack(plain, legacy, schema=LEGACY_FENCE_SCHEMA)
    assert FieldAsset.open(legacy).manifest["perimeter_fence"] == legacy_contract
    ramp_clearance = tmp_path / "ramp-clearance-fenced"
    ramp_clearance_contract = export_fenced_pack(
        plain, ramp_clearance, schema=RAMP_CLEARANCE_FENCE_SCHEMA
    )
    assert FieldAsset.open(ramp_clearance).manifest["perimeter_fence"] == ramp_clearance_contract
    assert ramp_clearance_contract["contact"]["solref"] == [0.02, 1.0]
    assert ramp_clearance_contract["panels"][3]["pos"][1] - legacy_contract["panels"][3]["pos"][
        1
    ] == pytest.approx(0.4)
    soft_contact = tmp_path / "soft-contact-fenced"
    soft_contact_contract = export_fenced_pack(
        plain, soft_contact, schema=SOFT_CONTACT_FENCE_SCHEMA
    )
    assert FieldAsset.open(soft_contact).manifest["perimeter_fence"] == soft_contact_contract
    assert soft_contact_contract["outward_offset_xy_m"] == [0.0, 0.4]
    assert soft_contact_contract["contact"]["solref"] == [0.04, 1.0]
    heightfield_edge = tmp_path / "heightfield-edge-fenced"
    heightfield_edge_contract = export_fenced_pack(
        plain, heightfield_edge, schema=HEIGHTFIELD_EDGE_FENCE_SCHEMA
    )
    assert FieldAsset.open(heightfield_edge).manifest["perimeter_fence"] == (
        heightfield_edge_contract
    )
    assert heightfield_edge_contract["outward_offset_xy_m"] == pytest.approx([0.975, 0.475])
    assert heightfield_edge_contract["heightfield_edge_alignment"]["heightfield_bounds_xy_m"] == [
        [-15.0, -8.0],
        [15.0, 8.0],
    ]
    for profile in ("full", "collision_only"):
        root = ET.parse(fenced_asset.entrypoint_for(profile)).getroot()
        geoms = [
            geom for geom in root.findall("./worldbody/geom") if geom.get("name") in FENCE_NAMES
        ]
        assert {geom.get("name") for geom in geoms} == set(FENCE_NAMES)
        assert all(geom.get("contype") == "2" and geom.get("conaffinity") == "1" for geom in geoms)
    model, _ = load_model(fenced_asset, profile="collision_only")
    assert all(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0 for name in FENCE_NAMES
    )

    full_xml = fenced_asset.entrypoint
    tree = ET.parse(full_xml)
    geom = tree.getroot().find("./worldbody/geom[@name='rmuc2026_perimeter_left']")
    assert geom is not None
    position = [float(value) for value in geom.get("pos", "").split()]
    position[0] += 1.0
    geom.set("pos", " ".join(str(value) for value in position))
    tree.write(full_xml, encoding="utf-8", xml_declaration=True)
    manifest_path = fenced / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(row for row in manifest["contents"]["files"] if row["file"] == full_xml.name)
    record["sha256"] = _sha(full_xml)
    record["size_bytes"] = full_xml.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="fence rmuc2026_perimeter_left pos disagrees"):
        FieldAsset.open(fenced)

    geom.set("pos", " ".join(str(value) for value in contract["panels"][0]["pos"]))
    geom.set("margin", "0.1")
    tree.write(full_xml, encoding="utf-8", xml_declaration=True)
    record["sha256"] = _sha(full_xml)
    record["size_bytes"] = full_xml.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="fence contact contract is invalid"):
        FieldAsset.open(fenced)


def test_fence_pack_rejects_source_void_profile_until_joint_physics_is_audited(
    tmp_path: Path,
) -> None:
    source = _negative_source_build(tmp_path / "source")
    plain = tmp_path / "plain"
    export_runtime_asset_pack(source, plain)
    with pytest.raises(ValueError, match="schema-2"):
        export_fenced_pack(plain, tmp_path / "fenced")


def test_current_perimeter_limits_both_pinned_fly_ramp_outer_overlap() -> None:
    manifest = {
        "schema_version": 2,
        "dimensions": {
            "official_core_battlefield_m": [28.0, 15.0],
            "cad_assembly_outer_bounds_after_translation_m": [
                [-26.04, -13.78, -0.151692429593],
                [3.712008274928, 2.223130912781, 3.649624647153],
            ],
        },
        "collision": {
            "geom_center_after_translation_m": [-11.16, -5.78, 0.0],
            "half_size_xy_m": [14.9, 8.02],
        },
        "coordinate_frame": {
            "recommended_spawn": {
                "x_before_translation_m": 11.164002933086,
                "y_before_translation_m": 7.402777938842,
                "terrain_height_m": 0.009595520909,
            }
        },
    }
    contract = perimeter_fence_contract(manifest)
    panels = {panel["name"].rsplit("_", 1)[-1]: panel for panel in contract["panels"]}
    spawn = manifest["coordinate_frame"]["recommended_spawn"]
    for ramp in FIXED_FLY_RAMPS:
        low = np.asarray(ramp.low_edge_center_xyz_m[:2]) - np.asarray(
            [spawn["x_before_translation_m"], spawn["y_before_translation_m"]]
        )
        uphill = np.asarray(ramp.uphill_unit_xy)
        lateral = np.asarray([-uphill[1], uphill[0]])
        corners = np.asarray(
            [
                low + along * uphill + across * lateral
                for along in (0.0, ramp.horizontal_run_m)
                for across in (-ramp.surface_width_m / 2, ramp.surface_width_m / 2)
            ]
        )
        side = "top" if "north" in ramp.route_id else "bottom"
        panel = panels[side]
        inner_face_y = panel["pos"][1] + (-panel["size"][1] if side == "top" else panel["size"][1])
        clearance = (
            inner_face_y - float(np.max(corners[:, 1]))
            if side == "top"
            else float(np.min(corners[:, 1])) - inner_face_y
        )
        assert -0.01 <= clearance <= 0.0


def test_export_preserves_bound_fixed_fly_ramp_audit(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = {
        "artifact_type": "rmuc2026_fixed_fly_ramp_heightfield_audit",
        "status": "PASS_STATIC_PENDING_DYNAMIC",
        "collision_owner": "existing_single_heightfield_only",
        "heightfield_modified": False,
        "planar_refinement_applied": False,
        "ramps": [{"source_part_index": 392}, {"source_part_index": 397}],
    }
    manifest["collision"]["fixed_fly_ramp_audit"] = audit
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = export_runtime_asset_pack(source, tmp_path / "pack")

    assert result["collision"]["fixed_fly_ramp_audit"] == audit


def test_export_can_include_fail_closed_non_contact_livery(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source", include_surface_guide=True)
    output = tmp_path / "pack"

    result = export_runtime_asset_pack(source, output, include_surface_guide=True)
    asset = FieldAsset.open(output)
    livery = result["visual_layers"]["livery"]
    assert livery["kind"] == SURFACE_GUIDE_KIND
    assert livery["geom_name"] == "rmuc2026_surface_guide"
    assert livery["geom_group"] == 4
    assert livery["default_visible"] is False
    assert livery["toggle_key"] == "G"
    assert livery["physics"] is False
    assert livery["mesh_file"] == "visual/rmuc2026_surface_guide.obj"
    assert livery["mesh_sha256"] == _sha(output / "visual/rmuc2026_surface_guide.obj")
    assert livery["texture_file"] == SURFACE_GUIDE_TEXTURE_FILE
    assert livery["texture_sha256"] == _sha(output / SURFACE_GUIDE_TEXTURE_FILE)
    assert livery["source_texture_sha256"] == _sha(
        source / "visual/official_rulebook_v2_overhead_surface.png"
    )
    assert livery["edge_processing"]["algorithm"] == "boundary_connected_near_white_alpha_v1"
    assert livery["contains_baked_scene_content"] == list(SURFACE_GUIDE_BAKED_CONTENT)
    assert "processing" not in livery
    assert "surface_sampling" not in livery
    source_texture = source / "visual/official_rulebook_v2_overhead_surface.png"
    with (
        Image.open(source_texture) as source_image,
        Image.open(output / SURFACE_GUIDE_TEXTURE_FILE) as runtime_image,
    ):
        assert runtime_image.mode == "RGBA"
        assert np.array_equal(
            np.asarray(runtime_image)[:, :, :3], np.asarray(source_image.convert("RGB"))
        )
        assert np.all(np.asarray(runtime_image)[:, :, 3] == 255)
    assert result["excluded"]["official_rulebook_screenshot"] is False
    assert result["excluded"]["rulebook_derived_ground_marking_overlay"] is True
    full = ET.parse(asset.entrypoint_for("full")).getroot()
    collision_only = ET.parse(asset.entrypoint_for("collision_only")).getroot()
    assert len(full.findall("./asset/mesh")) == 39
    geom = full.find("./worldbody/geom[@name='rmuc2026_surface_guide']")
    assert geom is not None
    assert (geom.get("group"), geom.get("contype"), geom.get("conaffinity")) == ("4", "0", "0")
    texture = full.find("./asset/texture[@name='rmuc2026_surface_guide_texture']")
    assert texture is not None
    assert texture.get("file") == livery["texture_file"]
    assert collision_only.find(".//*[@name='rmuc2026_surface_guide']") is None
    mesh_text = (output / livery["mesh_file"]).read_text(encoding="ascii")
    assert "vt 0 0" in mesh_text and "vt 1 1" in mesh_text and "\nf " in mesh_text


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda surface: surface.__setitem__("kind", "invented"), "kind"),
        (lambda surface: surface.__setitem__("physics", True), "physics=false"),
        (
            lambda surface: surface["world_mapping"].__setitem__(
                "image_top_to_world", "negative_y"
            ),
            "world_mapping",
        ),
        (lambda surface: surface.__setitem__("sha256", "0" * 64), "SHA-256"),
    ],
)
def test_surface_guide_source_contract_fails_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    error: str,
) -> None:
    source = _synthetic_source_build(tmp_path / "source", include_surface_guide=True)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest["surface_guide"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportBlocked, match=error):
        export_runtime_asset_pack(source, tmp_path / "pack", include_surface_guide=True)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("geom_group", 3, "schema-2 contract"),
        ("physics", True, "schema-2 contract"),
        ("mesh_sha256", "0" * 64, "SHA-256"),
        ("contains_baked_scene_content", ["floor"], "full-guide contract"),
        ("source_texture_sha256", "invalid", "source_texture_sha256"),
        ("edge_processing", {}, "edge_processing contract"),
    ],
)
def test_runtime_livery_manifest_contract_fails_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
    error: str,
) -> None:
    source = _synthetic_source_build(tmp_path / "source", include_surface_guide=True)
    output = tmp_path / "pack"
    export_runtime_asset_pack(source, output, include_surface_guide=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_layers"]["livery"][field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match=error):
        FieldAsset.open(output, verify=False)


def test_loader_keeps_earlier_filtered_marking_schema2_compatible(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source", include_surface_guide=True)
    output = tmp_path / "pack"
    export_runtime_asset_pack(source, output, include_surface_guide=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    livery = manifest["visual_layers"]["livery"]
    livery.pop("contains_baked_scene_content")
    livery.pop("source_texture_sha256")
    livery.pop("edge_processing")
    livery.update(
        {
            "kind": GROUND_MARKING_OVERLAY_KIND,
            "source_kind": SURFACE_GUIDE_KIND,
            "source_contains_baked_scene_content": list(SURFACE_GUIDE_BAKED_CONTENT),
            "processing": {
                "algorithm": GROUND_MARKING_PROCESSING_ALGORITHM,
                "output_mode": "rgba_with_transparent_non_markings",
                "retained_content": list(GROUND_MARKING_RETAINED_CONTENT),
                "removed_by_design": list(GROUND_MARKING_REMOVED_BY_DESIGN),
                "minimum_component_span_m": 0.45,
                "minimum_component_area_m2": 0.002,
                "input_size_px": [4, 4],
                "input_pixels": 16,
                "strong_candidate_pixels": 16,
                "retained_pixels": 16,
                "transparent_pixels": 0,
                "retained_fraction": 1.0,
                "transparent_rgb_edge_bleed_radius_px": 4,
                "transparent_rgb_edge_bleed_pixels": 0,
                "components": {"detected": 1, "retained": 1, "rejected": 0},
            },
            "surface_sampling": {
                "method": "sparse_marking_driven_5cm_mujoco_triangle_height_sampling",
                "target_spacing_m": 0.05,
                "actual_max_spacing_xy_m": [0.05, 0.05],
                "clearance_m": 0.018,
                "maximum_surface_height_m": 0.7,
                "maximum_cell_height_delta_m": 0.2,
                "vertices": 4,
                "faces": 2,
                "omitted_cells": 0,
                "cell_count": 1,
                "marking_cells": 1,
                "transparent_cells": 0,
                "height_filtered_marking_cells": 0,
                "mesh_size_bytes": (output / livery["mesh_file"]).stat().st_size,
                "boundary_xy_height_interpolation": True,
                "source_texture_alpha_drives_topology": True,
            },
            "known_limitations": list(GROUND_MARKING_KNOWN_LIMITATIONS),
        }
    )
    texture_record = next(
        record
        for record in manifest["contents"]["files"]
        if record["file"] == livery["texture_file"]
    )
    texture_record["role"] = "rulebook_derived_ground_marking_texture"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert FieldAsset.open(output).manifest["visual_layers"]["livery"]["kind"] == (
        GROUND_MARKING_OVERLAY_KIND
    )


@pytest.mark.parametrize(
    ("selector", "attribute", "replacement", "error"),
    [
        (
            "./worldbody/geom[@name='rmuc2026_surface_guide']",
            "contype",
            "2",
            "visual-only group 4",
        ),
        (
            "./worldbody/light[@name='rmuc2026_key_light']",
            "castshadow",
            "true",
            "default lighting must be flat",
        ),
    ],
)
def test_runtime_livery_xml_contract_fails_closed_after_rehash(
    tmp_path: Path,
    selector: str,
    attribute: str,
    replacement: str,
    error: str,
) -> None:
    source = _synthetic_source_build(tmp_path / "source", include_surface_guide=True)
    output = tmp_path / "pack"
    export_runtime_asset_pack(source, output, include_surface_guide=True)
    xml_path = output / "rmuc2026_field.xml"
    tree = ET.parse(xml_path)
    element = tree.getroot().find(selector)
    assert element is not None
    element.set(attribute, replacement)
    tree.write(xml_path, encoding="unicode")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        record for record in manifest["contents"]["files"] if record["file"] == "rmuc2026_field.xml"
    )
    record["sha256"] = _sha(xml_path)
    record["size_bytes"] = xml_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match=error):
        FieldAsset.open(output)


def test_surface_guide_uses_five_centimetre_exact_xy_sampling(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)
    height = 0.1 * x[None, :] + 0.2 * y[:, None]
    samples = tmp_path / "heightfield.npz"
    np.savez_compressed(samples, x_m=x, y_m=y, height_m=height)
    mesh = tmp_path / "guide.obj"
    bounds = np.asarray([[0.013, 0.017], [0.987, 0.983]])

    report = _write_surface_guide_mesh(
        mesh,
        samples_path=samples,
        collision={"rows_y": 101, "columns_x": 101},
        recommended_spawn={
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
        world_bounds_xy_m=bounds,
    )

    assert (
        report["method"] == "full_rulebook_surface_adaptive_5cm_2p5cm_1p25cm_signed_height_bounds"
    )
    assert max(report["actual_max_spacing_xy_m"]) <= 0.05 + 1.0e-9
    assert report["omitted_cells"] == 0
    assert report["visible_area_fraction"] == pytest.approx(1.0)
    assert report["retained_sampled_minimum_above_terrain_m"] >= 0.002 - 1.0e-9
    assert report["retained_sampled_maximum_above_terrain_m"] <= 0.008 + 1.0e-9
    assert report["source_texture_alpha_drives_topology"] is False
    lines = mesh.read_text(encoding="ascii").splitlines()
    vertices = [line for line in lines if line.startswith("v ")]
    first = [float(value) for value in vertices[0].split()[1:]]
    last = [float(value) for value in vertices[-1].split()[1:]]
    assert first == pytest.approx([0.013, 0.017, 0.1 * 0.013 + 0.2 * 0.017 + 0.005])
    assert last == pytest.approx([0.987, 0.983, 0.1 * 0.987 + 0.2 * 0.983 + 0.005])


def test_surface_guide_height_sampler_matches_mujoco_diagonal_on_saddle() -> None:
    x = np.asarray([0.0, 1.0])
    y = np.asarray([0.0, 1.0])
    height = np.asarray([[0.0, 0.0], [0.0, 1.0]])

    sampled = _mujoco_triangle_height_samples(
        x,
        y,
        height,
        np.asarray([0.25, 0.75]),
        np.asarray([0.25, 0.75]),
    )

    np.testing.assert_allclose(sampled, [[0.25, 0.25], [0.25, 0.75]])
    assert sampled[0, 1] != pytest.approx(0.1875)  # bilinear saddle value


def test_surface_guide_keeps_high_surfaces_and_omits_only_height_jumps(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)
    samples = tmp_path / "heightfield.npz"
    height = np.zeros((101, 101))
    height[:, 60:] = 2.0
    np.savez_compressed(samples, x_m=x, y_m=y, height_m=height)

    report = _write_surface_guide_mesh(
        tmp_path / "guide.obj",
        samples_path=samples,
        collision={"rows_y": 101, "columns_x": 101},
        recommended_spawn={
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
        world_bounds_xy_m=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )

    assert 0 < report["height_discontinuity_filtered_cells"] < report["cell_count"]
    assert report["visible_area_fraction"] < 1.0
    assert report["faces"] == 2 * (
        report["coarse_visible_cells"]
        + report["refined_visible_subcells"]
        + report["local_visible_subcells"]
    )
    vertices = [
        float(line.split()[3])
        for line in (tmp_path / "guide.obj").read_text(encoding="ascii").splitlines()
        if line.startswith("v ")
    ]
    assert max(vertices) == pytest.approx(2.005)


def test_surface_guide_omits_hidden_terrain_spike_inside_a_smooth_cell(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)
    height = np.zeros((101, 101))
    height[2, 2] = 0.3  # 5 cm cell corners remain zero despite the interior spike
    samples = tmp_path / "heightfield.npz"
    np.savez_compressed(samples, x_m=x, y_m=y, height_m=height)

    report = _write_surface_guide_mesh(
        tmp_path / "guide.obj",
        samples_path=samples,
        collision={"rows_y": 101, "columns_x": 101},
        recommended_spawn={
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
        world_bounds_xy_m=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )

    assert report["height_discontinuity_filtered_cells"] == 0
    assert report["interior_undercut_filtered_cells"] >= 1
    assert report["visible_area_fraction"] < 1.0


def test_surface_guide_rejects_hidden_pit_and_seals_refined_coarse_seam(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)
    height = np.zeros((101, 101))
    height[1, 1] = -0.3  # 5 cm cell corners miss the interior pit
    height[2, 5] = 0.002  # the native midpoint on its right edge is non-linear
    samples = tmp_path / "heightfield.npz"
    np.savez_compressed(samples, x_m=x, y_m=y, height_m=height)
    mesh = tmp_path / "guide.obj"

    report = _write_surface_guide_mesh(
        mesh,
        samples_path=samples,
        collision={"rows_y": 101, "columns_x": 101},
        recommended_spawn={
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
        world_bounds_xy_m=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )

    assert report["interior_raised_bridge_filtered_cells"] >= 1
    assert report["refined_visible_subcells"] >= 1
    assert report["visible_area_fraction"] < 1.0
    assert report["retained_sampled_minimum_above_terrain_m"] >= 0.002 - 1.0e-9
    assert report["retained_sampled_maximum_above_terrain_m"] <= 0.008 + 1.0e-9
    vertex_positions = [
        tuple(float(value) for value in line.split()[1:])
        for line in mesh.read_text(encoding="ascii").splitlines()
        if line.startswith("v ")
    ]
    faces = [
        tuple(vertex_positions[int(field.split("/")[0]) - 1] for field in line.split()[1:])
        for line in mesh.read_text(encoding="ascii").splitlines()
        if line.startswith("f ")
    ]
    assert not any(
        max(vertex[0] for vertex in face) <= 0.025 and max(vertex[1] for vertex in face) <= 0.025
        for face in faces
    )  # no triangle drapes over the pit
    assert any(
        min(vertex[0] for vertex in face) >= 0.05
        and max(vertex[0] for vertex in face) == pytest.approx(0.1)
        and max(vertex[1] for vertex in face) <= 0.05
        for face in faces
    )  # adjacent coarse cell remains intact
    seam = {
        round(vertex[1], 6): vertex[2]
        for vertex in vertex_positions
        if vertex[0] == pytest.approx(0.05) and vertex[1] <= 0.05
    }
    assert {0.0, 0.025, 0.05}.issubset(seam)
    assert seam[0.025] == pytest.approx(0.5 * (seam[0.0] + seam[0.05]))


def test_surface_guide_restores_stair_treads_without_draping_risers(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 101)
    y = np.linspace(0.0, 1.0, 101)
    height = np.zeros((101, 101))
    for edge_index in (6, 13, 21, 29, 38):
        height[:, edge_index:] += 0.08
    samples = tmp_path / "stairs.npz"
    np.savez_compressed(samples, x_m=x, y_m=y, height_m=height)
    mesh = tmp_path / "stairs.obj"

    report = _write_surface_guide_mesh(
        mesh,
        samples_path=samples,
        collision={"rows_y": 101, "columns_x": 101},
        recommended_spawn={
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
        world_bounds_xy_m=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )

    two_level_fraction = (
        report["coarse_visible_cells"] + report["refined_visible_subcells"] / 4.0
    ) / report["cell_count"]
    assert report["local_visible_subcells"] > 0
    assert report["visible_area_fraction"] - two_level_fraction >= 0.05
    assert report["retained_sampled_minimum_above_terrain_m"] >= 0.002 - 1.0e-9
    assert report["retained_sampled_maximum_above_terrain_m"] <= 0.008 + 1.0e-9

    vertices = [
        tuple(float(value) for value in line.split()[1:])
        for line in mesh.read_text(encoding="ascii").splitlines()
        if line.startswith("v ")
    ]
    point_array = np.asarray(vertices)
    checked_shared_midpoints = 0
    for line in mesh.read_text(encoding="ascii").splitlines():
        if not line.startswith("f "):
            continue
        face = [vertices[int(field.split("/")[0]) - 1] for field in line.split()[1:]]
        rise = max(vertex[2] for vertex in face) - min(vertex[2] for vertex in face)
        span_x = max(vertex[0] for vertex in face) - min(vertex[0] for vertex in face)
        span_y = max(vertex[1] for vertex in face) - min(vertex[1] for vertex in face)
        assert rise <= 0.8 * np.hypot(span_x, span_y) + 1.0e-8
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if start[0] == end[0] and abs(start[1] - end[1]) >= 0.025 - 1.0e-8:
                coordinate, fixed = 1, 0
            elif start[1] == end[1] and abs(start[0] - end[0]) >= 0.025 - 1.0e-8:
                coordinate, fixed = 0, 1
            else:
                continue
            fraction = (point_array[:, coordinate] - start[coordinate]) / (
                end[coordinate] - start[coordinate]
            )
            interior = (
                np.isclose(point_array[:, fixed], start[fixed], atol=1.0e-8)
                & (fraction > 1.0e-8)
                & (fraction < 1.0 - 1.0e-8)
            )
            if np.any(interior):
                expected = start[2] + fraction[interior] * (end[2] - start[2])
                np.testing.assert_allclose(point_array[interior, 2], expected, atol=1.0e-7)
                checked_shared_midpoints += int(np.count_nonzero(interior))
    assert checked_shared_midpoints > 0


def test_export_carries_heightfield_sample_provenance(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collision"]["ray_misses_filled_with_ground"] = 12_345
    manifest["collision"]["isolated_spikes_replaced"] = 678
    manifest["collision"]["structural_audit"] = {
        "representation": "single_height_per_xy_cell_2.5D",
        "underpasses_and_overhangs_preserved": False,
        "sealed_underpass_risk": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "pack"
    result = export_runtime_asset_pack(source, output)

    assert result["collision"]["ray_misses_filled_with_ground"] == 12_345
    assert result["collision"]["isolated_spikes_replaced"] == 678
    assert result["collision"]["structural_audit"]["sealed_underpass_risk"] is True
    asset = FieldAsset.open(output)
    assert asset.collision["ray_misses_filled_with_ground"] == 12_345
    assert asset.collision["isolated_spikes_replaced"] == 678


def test_export_records_absent_sample_provenance_as_null(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source")

    result = export_runtime_asset_pack(source, tmp_path / "pack")

    assert result["collision"]["ray_misses_filled_with_ground"] is None
    assert result["collision"]["isolated_spikes_replaced"] is None
    assert "structural_audit" not in result["collision"]


def test_export_rejects_a_negative_sample_provenance_counter(tmp_path: Path) -> None:
    source = _synthetic_source_build(tmp_path / "source")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collision"]["isolated_spikes_replaced"] = -1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportBlocked, match="非负整数"):
        export_runtime_asset_pack(source, tmp_path / "pack")
