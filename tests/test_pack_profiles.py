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

from rmuc2026_mujoco import FieldAsset, ManifestError
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _heightfield_bootstrap_png(height_m: np.ndarray, maximum_height_m: float) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    quantized = np.rint(np.clip(height_m / maximum_height_m, 0.0, 1.0) * 65535.0).astype(np.uint16)
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
    assert livery["contains_baked_scene_content"] == list(SURFACE_GUIDE_BAKED_CONTENT)
    assert "processing" not in livery
    assert "surface_sampling" not in livery
    source_texture = source / "visual/official_rulebook_v2_overhead_surface.png"
    assert (output / SURFACE_GUIDE_TEXTURE_FILE).read_bytes() == source_texture.read_bytes()
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


def test_surface_guide_uses_twenty_centimetre_exact_xy_sampling(tmp_path: Path) -> None:
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

    assert report["method"] == "full_rulebook_surface_20cm_mujoco_triangle_height_sampling"
    assert max(report["actual_max_spacing_xy_m"]) <= 0.20 + 1.0e-9
    assert report["omitted_cells"] == 0
    assert report["source_texture_alpha_drives_topology"] is False
    lines = mesh.read_text(encoding="ascii").splitlines()
    vertices = [line for line in lines if line.startswith("v ")]
    first = [float(value) for value in vertices[0].split()[1:]]
    last = [float(value) for value in vertices[-1].split()[1:]]
    assert first == pytest.approx([0.013, 0.017, 0.1 * 0.013 + 0.2 * 0.017 + 0.018])
    assert last == pytest.approx([0.987, 0.983, 0.1 * 0.987 + 0.2 * 0.983 + 0.018])


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
    assert report["omitted_cells"] == report["height_discontinuity_filtered_cells"]
    assert report["faces"] == 2 * (report["cell_count"] - report["omitted_cells"])
    vertices = [
        float(line.split()[3])
        for line in (tmp_path / "guide.obj").read_text(encoding="ascii").splitlines()
        if line.startswith("v ")
    ]
    assert max(vertices) == pytest.approx(2.018)


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
