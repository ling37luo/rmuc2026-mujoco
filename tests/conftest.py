from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


OFFICIAL_STEP_SHA256 = "8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33"
OFFICIAL_STEP_SIZE = 1_254_821_405


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(root: Path, relative: str, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "file": relative,
        "role": role,
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture
def field_asset_dir(tmp_path: Path) -> Path:
    visual = tmp_path / "visual"
    collision = tmp_path / "collision"
    visual.mkdir()
    collision.mkdir()
    mesh = visual / "tetra.obj"
    mesh.write_text(
        "\n".join(
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
        ),
        encoding="ascii",
    )
    image = collision / "heightfield.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-test-only")
    samples = collision / "heightfield.npz"
    x = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    y = np.asarray([20.0, 21.0, 22.0], dtype=np.float64)
    height = np.asarray(
        [[0.5, 0.75, 1.0, 1.25], [0.75, 1.0, 1.25, 1.5], [1.0, 1.25, 1.5, 2.0]],
        dtype=np.float64,
    )
    np.savez_compressed(samples, x_m=x, y_m=y, height_m=height)
    entrypoint = tmp_path / "rmuc2026_field.xml"
    entrypoint.write_text(
        """<mujoco model="synthetic_field">
  <asset>
    <mesh name="field_mesh" file="visual/tetra.obj"/>
    <hfield name="rmuc2026_collision" nrow="3" ncol="4" size="1.5 1 2 .05"/>
  </asset>
  <worldbody>
    <geom name="field_visual" type="mesh" mesh="field_mesh" contype="0" conaffinity="0"/>
    <geom name="rmuc2026_field_collision" type="hfield" hfield="rmuc2026_collision"
          pos=".5 0 -.5"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    collision_entrypoint = tmp_path / "rmuc2026_field_collision_only.xml"
    collision_entrypoint.write_text(
        """<mujoco model="synthetic_field_collision_only">
  <asset>
    <hfield name="rmuc2026_collision" nrow="3" ncol="4" size="1.5 1 2 .05"/>
  </asset>
  <worldbody>
    <geom name="rmuc2026_field_collision" type="hfield" hfield="rmuc2026_collision"
          pos=".5 0 -.5" friction="1 .005 .0001"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    records = [
        _file_record(tmp_path, "rmuc2026_field.xml", "field_mjcf_full"),
        _file_record(
            tmp_path,
            "rmuc2026_field_collision_only.xml",
            "field_mjcf_collision_only",
        ),
        _file_record(tmp_path, "visual/tetra.obj", "cad_visual_mesh"),
        _file_record(tmp_path, "collision/heightfield.png", "heightfield_bootstrap_png"),
        _file_record(tmp_path, "collision/heightfield.npz", "heightfield_float_samples"),
    ]
    manifest = {
        "schema_version": 1,
        "artifact_type": "rmuc2026_mujoco_runtime_asset_pack",
        "status": "PASS",
        "validation_status": "DRAFT_BLOCKED",
        "source_identity": {
            "official_step_sha256": OFFICIAL_STEP_SHA256,
            "official_step_size_bytes": OFFICIAL_STEP_SIZE,
            "license_status": "UNSPECIFIED",
            "redistribution_authorized": False,
            "asset_included_in_source_repository": False,
        },
        "visual_meshes": [
            {
                "file": "visual/tetra.obj",
                "sha256": _sha(mesh),
                "material_id": "synthetic",
                "rgba": [0.2, 0.4, 0.6, 1.0],
                "visual_role": "cad_structure",
            }
        ],
        "collision": {
            "kind": "conservative_top_surface_heightfield",
            "image_file": "collision/heightfield.png",
            "image_sha256": _sha(image),
            "samples_file": "collision/heightfield.npz",
            "samples_sha256": _sha(samples),
            "rows_y": 3,
            "columns_x": 4,
            "maximum_height_m": 2.0,
            "minimum_height_m": 0.5,
            "base_depth_m": 0.05,
            "half_size_xy_m": [1.5, 1.0],
            "geom_center_after_translation_m": [0.5, 0.0, -0.5],
            "png_rows": "flipped_y_for_mujoco_hfield_loader",
        },
        "coordinate_frame": {
            "world_units": "metre-radian-kilogram-second",
            "z_up": True,
            "recommended_spawn": {
                "x_before_translation_m": 11.0,
                "y_before_translation_m": 21.0,
                "terrain_height_m": 0.5,
            },
            "npz_axes_before_world_translation": True,
            "world_x_m": "x_m - recommended_spawn.x_before_translation_m",
            "world_y_m": "y_m - recommended_spawn.y_before_translation_m",
            "world_z_m": "height_m - recommended_spawn.terrain_height_m",
        },
        "heightfield_precision": {
            "float_samples_file": "collision/heightfield.npz",
            "rows_y": 3,
            "columns_x": 4,
            "float_npz_injection_required_before_validated_physics": True,
        },
        "contents": {
            "entrypoint": "rmuc2026_field.xml",
            "visual_obj_count": 1,
            "file_count_excluding_manifest": len(records),
            "files": records,
        },
        "runtime_profiles": {
            "default": "full",
            "profiles": {
                "full": {
                    "entrypoint": "rmuc2026_field.xml",
                    "includes_visual_meshes": True,
                    "visual_mesh_count": 1,
                    "intended_use": "interactive_visualization",
                },
                "collision_only": {
                    "entrypoint": "rmuc2026_field_collision_only.xml",
                    "includes_visual_meshes": False,
                    "visual_mesh_count": 0,
                    "intended_use": "headless_physics",
                },
            },
        },
        "validation_boundary": {
            "status": "DRAFT_BLOCKED",
            "whole_field_topology_ready": False,
            "final_policy_validation_ready": False,
        },
        "distribution": {
            "generated_locally": True,
            "third_party_geometry": True,
            "safe_to_publish_without_rightsholder_permission": False,
            "code_license_applies_to_asset_pack": False,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path
