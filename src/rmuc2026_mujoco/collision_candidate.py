"""Source-bound wall collision audit; this module never creates active geoms.

The official assembly has recognizable vertical barriers, but the current
heightfield already represents their roofs and steep sides.  Adding boxes to
that surface would create a second contact owner.  This registry retains exact
part bounds and local heightfield evidence for a future *replacement* design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_registry_contract import (
    REGISTRY_ACTIVATION,
    REGISTRY_ARTIFACT_TYPE,
    REGISTRY_COORDINATE_FRAME,
    REGISTRY_STATUS,
    validate_candidate_registry,
)
from .download import OFFICIAL_STEP_SHA256


# Part numbers mean anything only for this exact official STEP-derived GLB.
SOURCE_GLB_SHA256 = "5bc8042c0fcb90e6f23f1f6624dbf8af1177b582a56254b4df092ee04ef23705"
VERTICAL_BARRIER_PARTS = (269, 272, 308, 390, 391, 395, 396, 402, 403, 404)
SCREENSHOT_SUPPORT_PART = 554
SCREENSHOT_GRILLE_PART = 553
SCREENSHOT_OVERLAPPING_STEP_PART = 476

# Source triangles of 402/403 were checked against the exact GLB above: all
# 20 faces lie on supporting planes of their 12-point convex hulls.  Their
# AABBs overfill the true hull by 27.8%, so a box is not an exact wall proxy.
CONVEX_WALL_EVIDENCE = {
    402: {
        "unique_vertices": 12,
        "source_triangles": 20,
        "hull_facets": 20,
        "non_supporting_source_triangles": 0,
        "hull_volume_m3": 0.7058543477399379,
        "aabb_volume_m3": 0.9020128358420485,
    },
    403: {
        "unique_vertices": 12,
        "source_triangles": 20,
        "hull_facets": 20,
        "non_supporting_source_triangles": 0,
        "hull_volume_m3": 0.7058552899465835,
        "aabb_volume_m3": 0.9020129447543842,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_file(root: Path, relative: object, digest: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("candidate source file must be a relative path")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"candidate source file is missing or outside source: {relative}")
    if not isinstance(digest, str) or _sha256(path) != digest:
        raise ValueError(f"candidate source file hash mismatch: {relative}")
    return path


def _bounds(parts: list[dict[str, Any]], index: int) -> list[list[float]]:
    try:
        record = parts[index]
        if record["source_part_index"] != index:
            raise ValueError("part index/order mismatch")
        raw = record["input_bounds_after_spawn_translation_m"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"source part {index} lacks traceable world bounds") from exc
    bounds = np.asarray(raw, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError(f"source part {index} has invalid world bounds")
    return bounds.tolist()


def _nearest(axis: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(axis, value))
    return min((max(index - 1, 0), min(index, len(axis) - 1)), key=lambda i: abs(axis[i] - value))


def _heightfield_evidence(
    x: np.ndarray, y: np.ndarray, height: np.ndarray, bounds: list[list[float]]
) -> dict[str, Any]:
    low, high = np.asarray(bounds, dtype=np.float64)
    cx, cy = (low[:2] + high[:2]) / 2.0
    if not (x[0] <= cx <= x[-1] and y[0] <= cy <= y[-1]):
        raise ValueError("wall candidate centre is outside the heightfield")
    ix, iy = _nearest(x, float(cx)), _nearest(y, float(cy))
    x_mask = (x >= low[0]) & (x <= high[0])
    y_mask = (y >= low[1]) & (y <= high[1])
    if not x_mask.any() or not y_mask.any():
        raise ValueError("wall candidate footprint has no heightfield samples")
    footprint = height[np.ix_(y_mask, x_mask)]
    # Cross the thin axis beyond each source AABB face by at least one 1 cm
    # sample.  These heights diagnose ownership; they do not define a mask.
    thin_axis = int(np.argmin(high[:2] - low[:2]))
    if thin_axis == 0:
        flank_low = float(height[iy, _nearest(x, float(low[0] - 0.05))])
        flank_high = float(height[iy, _nearest(x, float(high[0] + 0.05))])
    else:
        flank_low = float(height[_nearest(y, float(low[1] - 0.05)), ix])
        flank_high = float(height[_nearest(y, float(high[1] + 0.05)), ix])
    return {
        "source_aabb_is_collision_shape": False,
        "heightfield_grid_samples_in_source_xy_aabb": int(footprint.size),
        "heightfield_center_z_m": float(height[iy, ix]),
        "heightfield_footprint_min_z_m": float(np.min(footprint)),
        "heightfield_footprint_max_z_m": float(np.max(footprint)),
        "heightfield_flank_low_z_m": flank_low,
        "heightfield_flank_high_z_m": flank_high,
        "source_top_z_m": float(high[2]),
        "thin_axis": "x" if thin_axis == 0 else "y",
    }


def build_wall_collision_registry(source_build: Path) -> dict[str, Any]:
    """Record eleven exact-source wall candidates without enabling collision.

    Raises on altered source assets or missing geometry provenance.  The
    candidates intentionally lack geoms and hfield masks: source bounds alone
    cannot resolve duplicate contacts, terrain junctions, or facility semantics.
    """

    root = Path(source_build).expanduser().resolve()
    manifest_path = root / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    source = json.loads(raw)
    if (
        source.get("artifact_type") != "rmuc2026_official_field_mujoco_asset"
        or source.get("status") != "PASS"
    ):
        raise ValueError("wall candidate requires a passing official field build")
    official_step_sha256 = source["source"]["sha256"]
    if official_step_sha256 != OFFICIAL_STEP_SHA256:
        raise ValueError("wall candidate source is not the audited official STEP")
    glb = source["conversion"]["colored_intermediate_glb"]
    if glb.get("sha256") != SOURCE_GLB_SHA256:
        raise ValueError("wall candidate part indices do not match this GLB")
    _declared_file(root, glb["file"], SOURCE_GLB_SHA256)
    collision = source["collision"]
    samples_sha256 = collision["samples_sha256"]
    samples_path = _declared_file(root, collision["samples_file"], samples_sha256)
    parts = source["conversion"]["visual_simplification"]["parts"]
    if not isinstance(parts, list) or len(parts) <= SCREENSHOT_SUPPORT_PART:
        raise ValueError("source build has no complete original-part geometry audit")
    with np.load(samples_path, allow_pickle=False) as samples:
        x = np.asarray(samples["x_m"], dtype=np.float64) - float(
            source["recommended_spawn"]["x_before_translation_m"]
        )
        y = np.asarray(samples["y_m"], dtype=np.float64) - float(
            source["recommended_spawn"]["y_before_translation_m"]
        )
        height = np.asarray(samples["height_m"], dtype=np.float64) - float(
            source["recommended_spawn"]["terrain_height_m"]
        )
    if (
        x.ndim != 1
        or y.ndim != 1
        or height.shape != (len(y), len(x))
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(height).all()
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
    ):
        raise ValueError("wall candidate heightfield samples are invalid")

    candidates = []
    for index in VERTICAL_BARRIER_PARTS:
        bounds = _bounds(parts, index)
        evidence = _heightfield_evidence(x, y, height, bounds)
        evidence["source_part_input_watertight"] = parts[index]["input_is_watertight"]
        if index in CONVEX_WALL_EVIDENCE:
            evidence["source_convexity_audit"] = CONVEX_WALL_EVIDENCE[index]
        candidates.append(
            {
                "id": f"OfficialWallPart{index}",
                "category": "wall",
                "source_part_indices": [index],
                "bounds_world_m": bounds,
                "blocking_reasons": [
                    "HfieldContactOwnerUnresolved",
                    "WallJunctionUnverified",
                    "OfficialSemanticsUnverified",
                ],
                "evidence": evidence,
            }
        )

    support_bounds = _bounds(parts, SCREENSHOT_SUPPORT_PART)
    grille_bounds = _bounds(parts, SCREENSHOT_GRILLE_PART)
    candidates.append(
        {
            "id": "ScreenshotWallNearGrille",
            "category": "wall",
            "source_part_indices": [SCREENSHOT_SUPPORT_PART],
            "bounds_world_m": support_bounds,
            "blocking_reasons": [
                "HfieldContactOwnerUnresolved",
                "SupportHasSteppedGeometry",
                "GrilleSemanticsUnverified",
            ],
            "evidence": {
                **_heightfield_evidence(x, y, height, support_bounds),
                "adjacent_grille_source_part_index": SCREENSHOT_GRILLE_PART,
                "adjacent_grille_bounds_world_m": grille_bounds,
                "adjacent_grille_hfield": _heightfield_evidence(x, y, height, grille_bounds),
                "overlapping_step_source_part_index": SCREENSHOT_OVERLAPPING_STEP_PART,
                "source_part_input_watertight": parts[SCREENSHOT_SUPPORT_PART][
                    "input_is_watertight"
                ],
            },
        }
    )
    registry = {
        "schema_version": 1,
        "artifact_type": REGISTRY_ARTIFACT_TYPE,
        "source_manifest_sha256": manifest_sha256,
        "official_step_sha256": official_step_sha256,
        "source_glb_sha256": SOURCE_GLB_SHA256,
        "collision_samples_sha256": samples_sha256,
        "coordinate_frame": REGISTRY_COORDINATE_FRAME,
        "status": REGISTRY_STATUS,
        "activation": REGISTRY_ACTIVATION,
        "candidates": candidates,
    }
    return validate_candidate_registry(
        registry,
        source_manifest_sha256=manifest_sha256,
        official_step_sha256=official_step_sha256,
        source_glb_sha256=SOURCE_GLB_SHA256,
        collision_samples_sha256=samples_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    """Write a local audit registry without touching the source build."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_build", type=Path, help="verified local official field build")
    parser.add_argument("output_json", type=Path, help="new local audit registry JSON")
    args = parser.parse_args(argv)
    output = args.output_json.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        parser.error(f"output parent directory does not exist: {output.parent}")
    registry = build_wall_collision_registry(args.source_build)
    payload = json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(f"Wrote {len(registry['candidates'])} disabled wall candidates: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
