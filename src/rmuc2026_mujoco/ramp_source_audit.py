"""Check fly-ramp plane outliers against the highest official CAD surface.

This is a local, read-only audit. It never flattens a neighboring official
surface merely to satisfy the dominant ramp's planar error statistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .collision_candidate import SOURCE_GLB_SHA256
from .download import OFFICIAL_STEP_SHA256
from .manifest import FieldAsset, sha256_file
from .ramp_audit import FIXED_FLY_RAMPS, RAMP_INTERIOR_INSET_M


def _verified_glb(source_manifest_path: Path) -> tuple[Path, float, str]:
    root = source_manifest_path.expanduser().resolve().parent
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source.get("artifact_type") != "rmuc2026_official_field_mujoco_asset"
        or source.get("status") != "PASS"
        or source["source"]["sha256"] != OFFICIAL_STEP_SHA256
    ):
        raise ValueError("source build is not the pinned passing official field")
    record = source["conversion"]["colored_intermediate_glb"]
    if record["sha256"] != SOURCE_GLB_SHA256:
        raise ValueError("source GLB is not the audited unsimplified official geometry")
    relative = Path(str(record["file"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents or not path.is_file():
        raise ValueError("source GLB is missing or outside the build")
    if sha256_file(path) != SOURCE_GLB_SHA256:
        raise ValueError("source GLB SHA-256 mismatch")
    shift = -float(source["conversion"]["main_floor_height_before_shift_m"])
    if not np.isfinite(shift):
        raise ValueError("source ground shift is non-finite")
    return path, shift, sha256_file(source_manifest_path)


def audit_fly_ramp_source_overlap(asset: FieldAsset, source_manifest_path: Path) -> dict[str, Any]:
    """Measure every >1 mm interior plane outlier against the full source GLB."""

    if not asset.hashes_verified:
        raise ValueError("ramp source audit requires a fully verified runtime pack")
    glb_path, shift, source_manifest_sha = _verified_glb(source_manifest_path)
    import trimesh

    scene = trimesh.load(glb_path, force="scene", process=False)
    meshes = list(scene.dump(concatenate=False))
    if len(meshes) != 559:
        raise ValueError("source GLB part count/order changed")
    with np.load(asset.file(str(asset.collision["samples_file"])), allow_pickle=False) as samples:
        x = np.asarray(samples["x_m"], dtype=np.float64)
        y = np.asarray(samples["y_m"], dtype=np.float64)
        height = np.asarray(samples["height_m"], dtype=np.float64)
    if height.shape != (len(y), len(x)) or not np.isfinite(height).all():
        raise ValueError("runtime heightfield samples are invalid")

    records = []
    for ramp in FIXED_FLY_RAMPS:
        low = np.asarray(ramp.low_edge_center_xyz_m)
        normal = np.asarray(ramp.normal_xyz)
        uphill = np.asarray(ramp.uphill_unit_xy)
        dx, dy = x[None, :] - low[0], y[:, None] - low[1]
        along = dx * uphill[0] + dy * uphill[1]
        lateral = -dx * uphill[1] + dy * uphill[0]
        plane = low[2] - (normal[0] * dx + normal[1] * dy) / normal[2]
        interior = (
            (along >= RAMP_INTERIOR_INSET_M)
            & (along <= ramp.horizontal_run_m - RAMP_INTERIOR_INSET_M)
            & (np.abs(lateral) <= ramp.surface_width_m / 2 - RAMP_INTERIOR_INSET_M)
        )
        plane_error = np.abs(height - plane)
        outliers = np.argwhere(interior & (plane_error > 0.001))
        inliers = interior & (plane_error <= 0.001)
        if not np.any(interior):
            raise ValueError(f"ramp interior is absent: {ramp.route_id}")
        source_error_max = 0.0
        winner_counts: dict[str, int] = {}
        all_outliers_are_other_source_parts = True
        if len(outliers):
            query_x = x[outliers[:, 1]]
            query_y = y[outliers[:, 0]]
            origins = np.column_stack((query_x, query_y, np.full(len(outliers), 5.0)))
            directions = np.tile((0.0, 0.0, -1.0), (len(outliers), 1))
            top = np.full(len(outliers), -np.inf)
            winner = np.full(len(outliers), -1, dtype=np.int64)
            x_min, x_max = float(np.min(query_x)), float(np.max(query_x))
            y_min, y_max = float(np.min(query_y)), float(np.max(query_y))
            for part_index, mesh in enumerate(meshes):
                bounds = mesh.bounds
                if (
                    bounds[0, 0] > x_max
                    or bounds[1, 0] < x_min
                    or bounds[0, 1] > y_max
                    or bounds[1, 1] < y_min
                ):
                    continue
                points, ray_indices, _triangles = mesh.ray.intersects_location(
                    origins, directions, multiple_hits=True
                )
                if len(ray_indices) == 0:
                    continue
                heights = points[:, 2] + shift
                better = heights > top[ray_indices] + 1.0e-9
                if np.any(better):
                    chosen_rays = ray_indices[better]
                    np.maximum.at(top, chosen_rays, heights[better])
                    winner[chosen_rays] = part_index
            if not np.isfinite(top).all() or np.any(winner < 0):
                raise ValueError(f"source has no top ray for ramp outliers: {ramp.route_id}")
            source_error_max = float(np.max(np.abs(height[outliers[:, 0], outliers[:, 1]] - top)))
            winner_counts = {
                str(part): int(np.count_nonzero(winner == part)) for part in np.unique(winner)
            }
            all_outliers_are_other_source_parts = bool(np.all(winner != ramp.source_part_index))
        non_overlap_error_max = float(np.max(plane_error[inliers])) if np.any(inliers) else 0.0
        adjusted_max = max(non_overlap_error_max, source_error_max)
        records.append(
            {
                "route_id": ramp.route_id,
                "dominant_source_part_index": ramp.source_part_index,
                "interior_sample_count": int(np.count_nonzero(interior)),
                "overlap_outlier_count": len(outliers),
                "dominant_plane_error_max_m": float(np.max(plane_error[interior])),
                "outlier_highest_source_abs_error_max_m": source_error_max,
                "other_source_winner_counts": winner_counts,
                "all_outliers_have_another_source_owner": all_outliers_are_other_source_parts,
                "overlap_aware_error_max_m": adjusted_max,
                "static_pass": all_outliers_are_other_source_parts and adjusted_max <= 0.001,
            }
        )
    return {
        "artifact_type": "rmuc2026_fly_ramp_source_overlap_audit",
        "status": "PASS_STATIC" if all(row["static_pass"] for row in records) else "FAIL_STATIC",
        "runtime_manifest_sha256": asset.manifest_sha256,
        "runtime_heightfield_samples_sha256": asset.collision["samples_sha256"],
        "source_manifest_sha256": source_manifest_sha,
        "source_glb_sha256": SOURCE_GLB_SHA256,
        "interior_inset_m": RAMP_INTERIOR_INSET_M,
        "maximum_error_m": 0.001,
        "ramps": records,
        "heightfield_modified": False,
        "claim_boundary": (
            "Static source-top audit of the two 17-degree fly-ramp interiors only; "
            "seam position, dynamic wheel behavior, jumps, and the rest of the field require "
            "separate tests. Adjacent official CAD surfaces are not flattened."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fixed fly-ramp overlaps against local CAD")
    parser.add_argument("asset", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_fly_ramp_source_overlap(FieldAsset.open(args.asset), args.source_manifest)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "PASS_STATIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
