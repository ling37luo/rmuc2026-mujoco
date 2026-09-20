from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rmuc2026_mujoco import collision_candidate


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_wall_registry_is_source_bound_and_never_activates_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    glb = tmp_path / "official.glb"
    glb.write_bytes(b"synthetic source geometry")
    monkeypatch.setattr(collision_candidate, "SOURCE_GLB_SHA256", _sha(glb.read_bytes()))
    monkeypatch.setattr(collision_candidate, "VERTICAL_BARRIER_PARTS", (402,))

    x = np.linspace(-0.3, 0.3, 7)
    y = np.linspace(-0.3, 0.3, 7)
    height = np.zeros((7, 7), dtype=np.float64)
    height[2:5, 2:5] = 0.4
    samples = tmp_path / "collision.npz"
    np.savez(samples, x_m=x, y_m=y, height_m=height)
    parts = [
        {
            "source_part_index": index,
            "input_bounds_after_spawn_translation_m": [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.4]],
            "input_is_watertight": False,
        }
        for index in range(555)
    ]
    parts[553]["input_bounds_after_spawn_translation_m"] = [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.2]]
    source = {
        "artifact_type": "rmuc2026_official_field_mujoco_asset",
        "status": "PASS",
        "source": {"sha256": collision_candidate.OFFICIAL_STEP_SHA256},
        "conversion": {
            "colored_intermediate_glb": {"file": glb.name, "sha256": _sha(glb.read_bytes())},
            "visual_simplification": {"parts": parts},
        },
        "collision": {"samples_file": samples.name, "samples_sha256": _sha(samples.read_bytes())},
        "recommended_spawn": {
            "x_before_translation_m": 0.0,
            "y_before_translation_m": 0.0,
            "terrain_height_m": 0.0,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(source), encoding="utf-8")

    registry = collision_candidate.build_wall_collision_registry(tmp_path)
    assert registry["status"] == "AUDIT_ONLY"
    assert registry["activation"] == "disabled"
    assert len(registry["candidates"]) == 2
    wall = registry["candidates"][0]
    assert wall["bounds_world_m"] == parts[402]["input_bounds_after_spawn_translation_m"]
    assert wall["evidence"]["heightfield_center_z_m"] == pytest.approx(0.4)
    assert wall["evidence"]["source_aabb_is_collision_shape"] is False
    assert "HfieldContactOwnerUnresolved" in wall["blocking_reasons"]

    samples.write_bytes(samples.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        collision_candidate.build_wall_collision_registry(tmp_path)
