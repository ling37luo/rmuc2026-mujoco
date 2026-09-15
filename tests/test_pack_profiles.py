from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from rmuc2026_mujoco import FieldAsset
from rmuc2026_mujoco.download import OFFICIAL_STEP_SHA256, OFFICIAL_STEP_SIZE
from rmuc2026_mujoco.pack import export_runtime_asset_pack


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_source_build(root: Path) -> Path:
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
                "visual_role": "cad_structure",
            }
        )
    image = collision_dir / "heightfield.png"
    image.write_bytes(b"synthetic bootstrap image")
    samples = collision_dir / "heightfield.npz"
    np.savez_compressed(
        samples,
        x_m=np.asarray([0.0, 1.0]),
        y_m=np.asarray([0.0, 1.0]),
        height_m=np.asarray([[0.0, 0.5], [0.5, 1.0]]),
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
            "kind": "conservative_top_surface_heightfield",
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
    assert len(full.findall("./asset/mesh")) == 38
    assert len([geom for geom in full.iter("geom") if geom.get("mesh")]) == 38
    assert collision_only.findall("./asset/mesh") == []
    assert [geom for geom in collision_only.iter("geom") if geom.get("mesh")] == []
    collision_geom = collision_only.find("./worldbody/geom[@name='rmuc2026_field_collision']")
    assert collision_geom is not None
    assert collision_geom.get("rgba", "").split()[-1] == "1"
