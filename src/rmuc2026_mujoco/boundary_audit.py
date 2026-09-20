"""Source-bound audit of the finite heightfield's four outer edges.

The result is deliberately audit-only. A triangle bounding box can overstate
where CAD has a face, but a bin it does *not* cover definitely has no source
face at that height anywhere in the inspected strip. Such gaps rule out a
continuous perimeter wall without inventing collision geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .candidate_registry_contract import REGISTRY_ARTIFACT_TYPE, validate_candidate_registry
from .download import OFFICIAL_STEP_SHA256
from .manifest import sha256_file


class BoundaryAuditError(ValueError):
    """A source asset or coordinate contract cannot support the audit."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise BoundaryAuditError(reason)


def _verified_source_file(source_root: Path, relative_name: str, digest: str, label: str) -> Path:
    candidate = (source_root / relative_name).resolve()
    _require(candidate.is_relative_to(source_root), f"{label} escapes source root")
    _require(candidate.is_file(), f"{label} is missing")
    _require(sha256_file(candidate) == digest, f"{label} SHA-256 mismatch")
    return candidate


def _intervals(mask: np.ndarray, start: float, spacing: float) -> list[list[float]]:
    intervals: list[list[float]] = []
    first: int | None = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and first is None:
            first = index
        elif not value and first is not None:
            intervals.append([round(start + first * spacing, 6), round(start + index * spacing, 6)])
            first = None
    return intervals


def audit_triangle_bounds(
    meshes: list[Any],
    *,
    world_translation_m: tuple[float, float, float],
    x_strip_m: tuple[float, float],
    y_bounds_m: tuple[float, float],
    # Include a 120 mm wheel centre and low curb height as well as body heights.
    probe_heights_m: tuple[float, ...] = (0.06, 0.15, 0.3, 0.6),
    bin_width_m: float = 0.05,
) -> dict[str, Any]:
    """Find definite source-face gaps along one edge, conservatively.

    Each triangle's world XYZ bounding box is treated as though it were full.
    This may count false coverage, so it cannot certify a wall. An uncovered
    bin, however, is a definite absence of source faces in this whole strip.
    """

    x_low, x_high = map(float, x_strip_m)
    y_low, y_high = map(float, y_bounds_m)
    probes = tuple(float(value) for value in probe_heights_m)
    _require(
        all(math.isfinite(v) for v in (*world_translation_m, x_low, x_high, y_low, y_high))
        and x_low < x_high
        and y_low < y_high,
        "invalid world transform or edge strip",
    )
    _require(probes and all(math.isfinite(value) for value in probes), "invalid probe heights")
    _require(math.isfinite(bin_width_m) and bin_width_m > 0, "invalid bin width")
    bin_count = math.ceil((y_high - y_low) / bin_width_m)
    _require(bin_count <= 100_000, "boundary audit has too many bins")
    bin_edges = np.linspace(y_low, y_high, bin_count + 1)
    actual_spacing = float(bin_edges[1] - bin_edges[0])
    occupancy = [np.zeros(bin_count + 1, dtype=np.int64) for _ in probes]
    contributing_parts: list[set[int]] = [set() for _ in probes]
    triangle_counts = [0 for _ in probes]
    translation = np.asarray(world_translation_m, dtype=np.float64)

    for part_index, mesh in enumerate(meshes):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        _require(
            vertices.ndim == 2
            and vertices.shape[1] == 3
            and faces.ndim == 2
            and faces.shape[1] == 3
            and np.isfinite(vertices).all(),
            f"source part {part_index} is not a finite triangle mesh",
        )
        if faces.size == 0:
            continue
        triangles = vertices[faces] + translation
        low = np.min(triangles, axis=1)
        high = np.max(triangles, axis=1)
        in_strip = (low[:, 0] <= x_high) & (high[:, 0] >= x_low)
        for probe_index, height in enumerate(probes):
            selected = in_strip & (low[:, 2] <= height) & (high[:, 2] >= height)
            if not np.any(selected):
                continue
            contributing_parts[probe_index].add(part_index)
            triangle_counts[probe_index] += int(np.count_nonzero(selected))
            # A bin is covered if *any part* of its interval overlaps a
            # triangle AABB. Sampling bin centres would wrongly label thin
            # source faces between centres as definite no-face gaps.
            starts = np.clip(
                np.searchsorted(bin_edges, low[selected, 1], side="left") - 1,
                0,
                bin_count,
            )
            stops = np.clip(
                np.searchsorted(bin_edges, high[selected, 1], side="right"),
                0,
                bin_count,
            )
            np.add.at(occupancy[probe_index], starts, 1)
            np.add.at(occupancy[probe_index], stops, -1)

    reports = []
    for index, height in enumerate(probes):
        covered = np.cumsum(occupancy[index][:-1]) > 0
        gaps = _intervals(~covered, y_low, actual_spacing)
        reports.append(
            {
                "height_m": height,
                "source_face_coverage_upper_bound_fraction": float(np.mean(covered)),
                "source_face_coverage_upper_bound_bins": int(np.count_nonzero(covered)),
                "total_bins": bin_count,
                "contributing_source_part_indices": sorted(contributing_parts[index]),
                "contributing_triangle_bounds": triangle_counts[index],
                "definite_no_face_intervals_y_m": gaps,
                "largest_definite_no_face_gap_m": max(
                    (high - low for low, high in gaps), default=0.0
                ),
                "continuous_barrier_supported": False,
            }
        )
    return {
        "algorithm": "conservative_source_triangle_aabb_upper_bound_v1",
        "x_strip_world_m": [x_low, x_high],
        "y_bounds_world_m": [y_low, y_high],
        "y_bin_width_m": actual_spacing,
        "probe_heights": reports,
        "claim_boundary": (
            "An uncovered bin has no source face at this height anywhere in the inspected "
            "strip. Covered bins are only upper bounds and do not prove a solid wall, "
            "robot blockage, or a safe collision primitive. No collider is generated."
        ),
    }


def _axis_report(report: dict[str, Any], axis: str) -> dict[str, Any]:
    probes = []
    for probe in report["probe_heights"]:
        converted = dict(probe)
        converted["definite_no_face_intervals_transverse_m"] = converted.pop(
            "definite_no_face_intervals_y_m"
        )
        probes.append(converted)
    return {
        "algorithm": report["algorithm"],
        "strip_axis_world": axis,
        "transverse_axis_world": "y" if axis == "x" else "x",
        "strip_bounds_world_m": report["x_strip_world_m"],
        "transverse_bounds_world_m": report["y_bounds_world_m"],
        "transverse_bin_width_m": report["y_bin_width_m"],
        "probe_heights": probes,
        "claim_boundary": report["claim_boundary"],
    }


def audit_source_boundaries(
    source_manifest_path: Path,
    *,
    inboard_width_m: float = 1.25,
    bin_width_m: float = 0.05,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit all four source-bound hfield edges without adding collision."""

    import trimesh

    path = Path(source_manifest_path).resolve()
    manifest_sha = sha256_file(path)
    if expected_source_manifest_sha256 is not None:
        _require(
            manifest_sha == expected_source_manifest_sha256, "source manifest SHA-256 mismatch"
        )
    source = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(source, dict), "source manifest must be a JSON object")
    _require(
        source.get("artifact_type") == "rmuc2026_official_field_mujoco_asset"
        and source.get("status") == "PASS",
        "boundary audit requires a passing official field build",
    )
    _require(
        source["source"]["sha256"] == OFFICIAL_STEP_SHA256,
        "boundary audit source is not the audited official STEP",
    )
    conversion = source["conversion"]
    axis = conversion["axis_and_units"]
    _require(
        axis["axis_order_output_xyz"] == [0, 1, 2] and float(axis["unit_scale_to_metres"]) == 1.0,
        "unsupported source GLB coordinate transform",
    )
    glb_record = conversion["colored_intermediate_glb"]
    glb_sha = glb_record["sha256"]
    glb_path = _verified_source_file(path.parent, glb_record["file"], glb_sha, "source CAD GLB")
    scene = trimesh.load(glb_path, force="scene", process=False)
    meshes = list(scene.dump(concatenate=False))
    _require(meshes, "source GLB contains no mesh parts")
    spawn = source["recommended_spawn"]
    collision = source["collision"]
    _verified_source_file(
        path.parent,
        collision["samples_file"],
        collision["samples_sha256"],
        "source collision samples",
    )
    x_bounds = (
        float(collision["x_min_m"]) - float(spawn["x_before_translation_m"]),
        float(collision["x_max_m"]) - float(spawn["x_before_translation_m"]),
    )
    y_bounds = (
        float(collision["y_min_m"]) - float(spawn["y_before_translation_m"]),
        float(collision["y_max_m"]) - float(spawn["y_before_translation_m"]),
    )
    _require(
        math.isfinite(inboard_width_m) and 0.0 < inboard_width_m <= 3.0,
        "invalid inboard width",
    )
    _require(
        x_bounds[1] - x_bounds[0] > 2 * inboard_width_m
        and y_bounds[1] - y_bounds[0] > 2 * inboard_width_m,
        "boundary strip is wider than the source field",
    )
    translation = (
        -float(spawn["x_before_translation_m"]),
        -float(spawn["y_before_translation_m"]),
        -float(conversion["main_floor_height_before_shift_m"]) - float(spawn["terrain_height_m"]),
    )
    swapped_meshes = [
        SimpleNamespace(vertices=np.asarray(mesh.vertices)[:, [1, 0, 2]], faces=mesh.faces)
        for mesh in meshes
    ]
    edges = {
        "negative_x": _axis_report(
            audit_triangle_bounds(
                meshes,
                world_translation_m=translation,
                x_strip_m=(x_bounds[0], x_bounds[0] + inboard_width_m),
                y_bounds_m=y_bounds,
                bin_width_m=bin_width_m,
            ),
            "x",
        ),
        "positive_x": _axis_report(
            audit_triangle_bounds(
                meshes,
                world_translation_m=translation,
                x_strip_m=(x_bounds[1] - inboard_width_m, x_bounds[1]),
                y_bounds_m=y_bounds,
                bin_width_m=bin_width_m,
            ),
            "x",
        ),
        "negative_y": _axis_report(
            audit_triangle_bounds(
                swapped_meshes,
                world_translation_m=(translation[1], translation[0], translation[2]),
                x_strip_m=(y_bounds[0], y_bounds[0] + inboard_width_m),
                y_bounds_m=x_bounds,
                bin_width_m=bin_width_m,
            ),
            "y",
        ),
        "positive_y": _axis_report(
            audit_triangle_bounds(
                swapped_meshes,
                world_translation_m=(translation[1], translation[0], translation[2]),
                x_strip_m=(y_bounds[1] - inboard_width_m, y_bounds[1]),
                y_bounds_m=x_bounds,
                bin_width_m=bin_width_m,
            ),
            "y",
        ),
    }
    all_indices = sorted(
        {
            index
            for edge in edges.values()
            for probe in edge["probe_heights"]
            for index in probe["contributing_source_part_indices"]
        }
    )
    registry = {
        "schema_version": 1,
        "artifact_type": REGISTRY_ARTIFACT_TYPE,
        "status": "AUDIT_ONLY",
        "activation": "disabled",
        "source_manifest_sha256": manifest_sha,
        "official_step_sha256": source["source"]["sha256"],
        "source_glb_sha256": glb_sha,
        "collision_samples_sha256": collision["samples_sha256"],
        "coordinate_frame": "translated_m_z_up",
        "candidates": [
            {
                "id": "four_edge_perimeter_source_gap_audit",
                "category": "boundary",
                "source_part_indices": all_indices,
                "bounds_world_m": [
                    [x_bounds[0], y_bounds[0], 0.0],
                    [x_bounds[1], y_bounds[1], 0.6],
                ],
                "blocking_reasons": [
                    "NO_BODY_HEIGHT_CONTINUOUS_SOURCE_FACE_BARRIER",
                    "FINITE_HFIELD_EDGE_UNPROTECTED",
                ],
                "evidence": {
                    "edge_reports": edges,
                    "collision_samples_sha256": collision["samples_sha256"],
                    "claim_boundary": (
                        "All four edges are read-only conservative source-face gap audits. "
                        "No covered bin proves robot blockage or justifies active collision."
                    ),
                },
            }
        ],
    }
    return validate_candidate_registry(
        registry,
        source_manifest_sha256=manifest_sha,
        official_step_sha256=source["source"]["sha256"],
        source_glb_sha256=glb_sha,
        collision_samples_sha256=collision["samples_sha256"],
    )


def audit_source_negative_x_boundary(
    source_manifest_path: Path,
    *,
    inboard_width_m: float = 1.25,
    bin_width_m: float = 0.05,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible entrypoint; now audits all four finite edges."""

    return audit_source_boundaries(
        source_manifest_path,
        inboard_width_m=inboard_width_m,
        bin_width_m=bin_width_m,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inboard-width", type=float, default=1.25)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    result = audit_source_boundaries(
        args.source_manifest,
        inboard_width_m=args.inboard_width,
        bin_width_m=args.bin_width,
        expected_source_manifest_sha256=args.expected_manifest_sha256,
    )
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RMUC2026_BOUNDARY_AUDIT=AUDIT_ONLY {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
