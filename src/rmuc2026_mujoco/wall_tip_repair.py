"""Restore verified wall-end roof samples lost by the generic spike filter.

Only the exact audited official GLB and established 1 cm grid are eligible.
The repair changes a few source-covered heightfield nodes; it never creates a
second collision geom or invents walls outside the official triangle footprint.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from .collision_candidate import SOURCE_GLB_SHA256


WALL_PARTS = (402, 403)
EXPECTED_REPAIRED_NODES = {402: 4, 403: 3}
END_BAND_M = 0.03
MIN_MISSING_ROOF_M = 0.15


def validate_wall_tip_repair_record(
    record: object, *, collision_samples_sha256: str
) -> Mapping[str, Any]:
    """Check that a source/runtime manifest does not overclaim this repair."""

    if not isinstance(record, dict):
        raise ValueError("wall-tip repair record must be an object")
    if record.get("collision_samples_sha256") != collision_samples_sha256:
        raise ValueError("wall-tip repair is not bound to these collision samples")
    status = record.get("status")
    if status == "NOT_APPLICABLE":
        if record.get("repaired_nodes") != 0 or not isinstance(record.get("reason"), str):
            raise ValueError("unapplied wall-tip repair must declare zero changes and a reason")
        return record
    if status != "PASS":
        raise ValueError("wall-tip repair status is invalid")
    if (
        record.get("source_glb_sha256") != SOURCE_GLB_SHA256
        or record.get("grid_resolution_m") != 0.01
        or record.get("source_part_indices") != list(WALL_PARTS)
        or record.get("repaired_nodes") != sum(EXPECTED_REPAIRED_NODES.values())
        or record.get("contact_owner") != "existing_single_heightfield"
        or record.get("new_collision_geoms") != 0
    ):
        raise ValueError("wall-tip repair source, grid, count, or contact owner changed")
    parts = record.get("parts")
    if not isinstance(parts, list) or len(parts) != len(WALL_PARTS):
        raise ValueError("wall-tip repair part records are incomplete")
    for part, row in zip(WALL_PARTS, parts):
        if not isinstance(row, dict) or (
            row.get("source_part_index") != part
            or row.get("repaired_nodes") != EXPECTED_REPAIRED_NODES[part]
        ):
            raise ValueError(f"wall-tip repair part {part} count changed")
        gap = row.get("max_missing_roof_m")
        if (
            isinstance(gap, bool)
            or not isinstance(gap, (int, float))
            or (not math.isfinite(gap) or gap < MIN_MISSING_ROOF_M)
        ):
            raise ValueError(f"wall-tip repair part {part} gap is invalid")
    samples = record.get("changed_samples")
    if not isinstance(samples, list) or len(samples) != record["repaired_nodes"]:
        raise ValueError("wall-tip repair must record every changed sample")
    sites = set()
    counts = {part: 0 for part in WALL_PARTS}
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("source_part_index") not in counts:
            raise ValueError("wall-tip repair sample part is invalid")
        part = sample["source_part_index"]
        counts[part] += 1
        row = sample.get("grid_row")
        column = sample.get("grid_column")
        if (
            isinstance(row, bool)
            or isinstance(column, bool)
            or not isinstance(row, int)
            or not isinstance(column, int)
            or row < 0
            or column < 0
            or (row, column) in sites
        ):
            raise ValueError("wall-tip repair sample grid index is invalid or repeated")
        sites.add((row, column))
        values = [
            sample.get(key)
            for key in (
                "x_before_translation_m",
                "y_before_translation_m",
                "height_before_m",
                "height_after_m",
            )
        ]
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
            or values[3] - values[2] < MIN_MISSING_ROOF_M
        ):
            raise ValueError("wall-tip repair sample heights or coordinates are invalid")
    if counts != EXPECTED_REPAIRED_NODES:
        raise ValueError("wall-tip repair changed-sample counts disagree with parts")
    return record


def verify_wall_tip_repair_samples(
    record: Mapping[str, Any], x: np.ndarray, y: np.ndarray, height: np.ndarray
) -> None:
    """Compare each declared repaired coordinate/height to the hashed float NPZ."""

    if record.get("status") != "PASS":
        return
    if x.ndim != 1 or y.ndim != 1 or height.shape != (len(y), len(x)):
        raise ValueError("wall-tip repair sample axes or grid shape are invalid")
    for sample in record["changed_samples"]:
        row = sample["grid_row"]
        column = sample["grid_column"]
        if row >= len(y) or column >= len(x):
            raise ValueError("wall-tip repair sample index exceeds the float grid")
        if not (
            math.isclose(float(x[column]), sample["x_before_translation_m"], abs_tol=1.0e-8)
            and math.isclose(float(y[row]), sample["y_before_translation_m"], abs_tol=1.0e-8)
            and math.isclose(float(height[row, column]), sample["height_after_m"], abs_tol=1.0e-8)
        ):
            raise ValueError("wall-tip repair changed sample disagrees with the float grid")


def _projected_top(mesh: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the upper envelope of exact source triangles on a small grid."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("verified wall mesh has invalid triangles")
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("verified wall mesh has invalid coordinates or indices")
    grid_x, grid_y = np.meshgrid(x, y)
    top = np.full(grid_x.shape, -np.inf, dtype=np.float64)
    for p, q, r in vertices[faces]:
        dx1, dy1 = q[:2] - p[:2]
        dx2, dy2 = r[:2] - p[:2]
        determinant = dx1 * dy2 - dy1 * dx2
        if abs(determinant) < 1.0e-12:
            continue
        dx = grid_x - p[0]
        dy = grid_y - p[1]
        u = (dx * dy2 - dy * dx2) / determinant
        v = (dx1 * dy - dy1 * dx) / determinant
        inside = (u >= -1.0e-10) & (v >= -1.0e-10) & (u + v <= 1.0 + 1.0e-10)
        z = p[2] + u * (q[2] - p[2]) + v * (r[2] - p[2])
        np.maximum(top, np.where(inside, z, -np.inf), out=top)
    return top


def repair_verified_wall_tips(
    meshes: list[Any],
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
    *,
    source_glb_sha256: str,
) -> dict[str, Any]:
    """Repair exactly seven source-covered 1 cm wall-end nodes in place.

    A changed part, raster origin, or library result fails closed. This helper
    is called before source NPZ/PNG hashes and the runtime pack are generated.
    """

    if source_glb_sha256 != SOURCE_GLB_SHA256:
        raise ValueError("wall-tip repair requires the exact audited official GLB")
    if len(meshes) <= max(WALL_PARTS):
        raise ValueError("wall-tip repair source part order is incomplete")
    if (
        x.ndim != 1
        or y.ndim != 1
        or height.shape != (len(y), len(x))
        or not np.isfinite(height).all()
        or not np.allclose(np.diff(x), 0.01, rtol=0.0, atol=1.0e-6)
        or not np.allclose(np.diff(y), 0.01, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("wall-tip repair requires the audited 1 cm heightfield grid")

    repairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    records = []
    changed_samples = []
    for part in WALL_PARTS:
        mesh = meshes[part]
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or len(mesh.faces) != 20
            or len(np.unique(np.asarray(mesh.vertices), axis=0)) != 12
            or not np.isclose(bounds[1, 0] - bounds[0, 0], 0.2, atol=1.0e-5)
        ):
            raise ValueError(f"verified wall part {part} geometry changed")
        ix = np.flatnonzero((x >= bounds[0, 0]) & (x <= bounds[1, 0]))
        iy = np.flatnonzero((y >= bounds[0, 1]) & (y <= bounds[1, 1]))
        if len(ix) == 0 or len(iy) == 0:
            raise ValueError(f"verified wall part {part} misses the heightfield grid")
        top = _projected_top(mesh, x[ix], y[iy])
        old = height[np.ix_(iy, ix)]
        near_end = np.minimum(y[iy] - bounds[0, 1], bounds[1, 1] - y[iy]) <= END_BAND_M
        missing = near_end[:, None] & np.isfinite(top) & (top - old > MIN_MISSING_ROOF_M)
        rows, cols = np.nonzero(missing)
        if len(rows) != EXPECTED_REPAIRED_NODES[part]:
            raise ValueError(
                f"verified wall part {part} changed repair count: {len(rows)} "
                f"!= {EXPECTED_REPAIRED_NODES[part]}"
            )
        global_rows, global_cols = iy[rows], ix[cols]
        new_height = top[rows, cols]
        repairs.append((global_rows, global_cols, new_height))
        changed_samples.extend(
            {
                "source_part_index": part,
                "grid_row": int(global_row),
                "grid_column": int(global_column),
                "x_before_translation_m": float(x[global_column]),
                "y_before_translation_m": float(y[global_row]),
                "height_before_m": float(old[local_row, local_column]),
                "height_after_m": float(new_height[index]),
            }
            for index, (local_row, local_column, global_row, global_column) in enumerate(
                zip(rows, cols, global_rows, global_cols)
            )
        )
        records.append(
            {
                "source_part_index": part,
                "repaired_nodes": len(rows),
                "max_missing_roof_m": float(np.max(new_height - old[rows, cols])),
            }
        )

    # Mutate only after both exact-source geometry and count gates pass.
    for rows, cols, new_height in repairs:
        height[rows, cols] = new_height
    return {
        "status": "PASS",
        "source_glb_sha256": source_glb_sha256,
        "grid_resolution_m": 0.01,
        "source_part_indices": list(WALL_PARTS),
        "repaired_nodes": sum(item["repaired_nodes"] for item in records),
        "parts": records,
        "changed_samples": changed_samples,
        "contact_owner": "existing_single_heightfield",
        "new_collision_geoms": 0,
    }


def derive_wall_tip_repaired_field_build(base_build: Path, output_dir: Path) -> dict[str, Any]:
    """Copy a verified local field build and change only seven collision nodes.

    The visual source meshes and optional local rulebook guide are copied
    byte-for-byte. No official asset is embedded in this repository.
    """

    import trimesh
    from PIL import Image

    from ._conversion import (
        FieldBuildError,
        _apply_verified_recorded_transform,
        _heightfield_structural_audit,
        _scene_meshes,
        _static_route_validation_scope,
        _verified_repack_input,
        _write_json_atomically,
        sha256_file,
    )
    from .ramp_audit import audit_fixed_fly_ramps

    base = Path(base_build).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output == base or base in output.parents:
        raise FieldBuildError("墙端修复输出必须是不存在且位于基础场地外的目录")
    manifest, glb_path, base_manifest_sha256, base_manifest_bytes = _verified_repack_input(base)
    if sha256_file(glb_path) != SOURCE_GLB_SHA256:
        raise FieldBuildError("墙端修复只支持审计过的官方GLB哈希")
    collision = manifest.get("collision")
    if not isinstance(collision, dict) or collision.get("resolution_m") != 0.01:
        raise FieldBuildError("墙端修复只支持官方1cm碰撞采样")
    if collision.get("verified_wall_tip_repair", {}).get("status") == "PASS":
        raise FieldBuildError("输入场地已包含墙端修复，拒绝重复应用")

    def verified_file(relative: object, digest: object) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise FieldBuildError("基础场地资源路径无效")
        candidate = base / relative
        path = candidate.resolve()
        if base not in path.parents or not path.is_file() or candidate.is_symlink():
            raise FieldBuildError(f"基础场地资源缺失、越界或为链接：{relative}")
        if not isinstance(digest, str) or sha256_file(path) != digest:
            raise FieldBuildError(f"基础场地资源哈希变化：{relative}")
        return path

    files = [
        (row.get("file"), row.get("sha256"))
        for row in manifest.get("visual_meshes", [])
        if isinstance(row, dict)
    ]
    if len(files) != len(manifest.get("visual_meshes", [])) or not files:
        raise FieldBuildError("基础场地视觉资源清单损坏")
    files.extend(
        (collision.get(file_key), collision.get(hash_key))
        for file_key, hash_key in (
            ("image_file", "image_sha256"),
            ("samples_file", "samples_sha256"),
        )
    )
    surface_guide = manifest.get("surface_guide")
    if isinstance(surface_guide, dict):
        files.append((surface_guide.get("file"), surface_guide.get("sha256")))
    for relative, digest in files:
        verified_file(relative, digest)
    if any(path.is_symlink() for path in base.rglob("*")):
        raise FieldBuildError("基础场地含符号链接，拒绝复制")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        shutil.copytree(base, staging, dirs_exist_ok=True, copy_function=shutil.copy2)
        samples_path = staging / str(collision["samples_file"])
        with np.load(samples_path, allow_pickle=False) as samples:
            x = np.asarray(samples["x_m"], dtype=np.float64)
            y = np.asarray(samples["y_m"], dtype=np.float64)
            height = np.asarray(samples["height_m"], dtype=np.float64).copy()
        if (
            x.ndim != 1
            or y.ndim != 1
            or height.shape != (len(y), len(x))
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isfinite(height).all()
        ):
            raise FieldBuildError("基础场地浮点高度场尺寸或值损坏")
        old_max = float(np.max(height))
        old_min = float(np.min(height))
        if not np.isclose(old_max, collision["maximum_height_m"], atol=1.0e-6):
            raise FieldBuildError("基础场地碰撞最大高度与样本不一致")
        if not np.isclose(old_min, collision["minimum_height_m"], atol=1.0e-6):
            raise FieldBuildError("基础场地碰撞最小高度与样本不一致")

        scene = trimesh.load(glb_path, force="scene", process=False)
        meshes = _scene_meshes(scene)
        _apply_verified_recorded_transform(meshes, manifest)
        try:
            repair = repair_verified_wall_tips(
                meshes, x, y, height, source_glb_sha256=SOURCE_GLB_SHA256
            )
        except ValueError as exc:
            raise FieldBuildError(f"官方墙端采样修复失败：{exc}") from exc
        if float(np.max(height)) != old_max or float(np.min(height)) != old_min:
            raise FieldBuildError("墙端修复意外改变高度场整体高低界")
        new_ramp_audit = audit_fixed_fly_ramps(x, y, height)
        if new_ramp_audit != collision.get("fixed_fly_ramp_audit"):
            raise FieldBuildError("墙端修复意外改变飞坡碰撞审计")
        np.savez_compressed(samples_path, x_m=x, y_m=y, height_m=height)
        image_path = staging / str(collision["image_file"])
        pixels = np.rint(np.clip(height / old_max, 0.0, 1.0) * 65535.0).astype(np.uint16)
        Image.fromarray(np.flipud(pixels), mode="I;16").save(image_path)
        old_hashes = {
            "image_sha256": collision["image_sha256"],
            "samples_sha256": collision["samples_sha256"],
        }
        collision["image_sha256"] = sha256_file(image_path)
        collision["samples_sha256"] = sha256_file(samples_path)
        collision["structural_audit"] = _heightfield_structural_audit(height)
        repair["collision_samples_sha256"] = collision["samples_sha256"]
        validate_wall_tip_repair_record(
            repair, collision_samples_sha256=collision["samples_sha256"]
        )
        collision["verified_wall_tip_repair"] = repair
        manifest["validation_scope"] = _static_route_validation_scope(collision)
        manifest["wall_tip_repaired_from"] = {
            "base_field_manifest_sha256": base_manifest_sha256,
            "source_glb_sha256": SOURCE_GLB_SHA256,
            "previous_collision_hashes": old_hashes,
            "source_visual_and_guide_unchanged": True,
        }
        _write_json_atomically(staging / "manifest.json", manifest)
        if (base / "manifest.json").read_bytes() != base_manifest_bytes:
            raise FieldBuildError("基础场地manifest在修复期间发生变化")
        if sha256_file(glb_path) != SOURCE_GLB_SHA256:
            raise FieldBuildError("基础官方GLB在修复期间发生变化")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
