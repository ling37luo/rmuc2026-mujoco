#!/usr/bin/env python3
"""Export the verified RMUC 2026 field build as a relocatable MuJoCo asset pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


EXPECTED_ARTIFACT_TYPE = "rmuc2026_official_field_mujoco_asset"
EXPECTED_VISUAL_MESH_COUNT = 38
OUTPUT_ARTIFACT_TYPE = "rmuc2026_mujoco_runtime_asset_pack"
OUTPUT_XML = "rmuc2026_field.xml"
OUTPUT_COLLISION_ONLY_XML = "rmuc2026_field_collision_only.xml"
PUBLIC_DISPLAY_RGB_BLACK_FLOOR = 0.10
PUBLIC_DISPLAY_RGB_WHITE_CEILING = 0.86
PUBLIC_DISPLAY_RGB_GAMMA = 0.85


class ExportBlocked(RuntimeError):
    """The source field build cannot safely produce the requested pack."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportBlocked(f"{label}必须是JSON对象")
    return value


def _finite_numbers(value: object, *, count: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ExportBlocked(f"{label}必须含{count}个数")
    numbers = [float(item) for item in value]
    if not all(math.isfinite(item) for item in numbers):
        raise ExportBlocked(f"{label}含非有限值")
    return numbers


def _display_rgba(source_rgba: list[float]) -> list[float]:
    """Lift CAD blacks and cap whites for legible, display-only contrast."""

    rgb = [
        PUBLIC_DISPLAY_RGB_BLACK_FLOOR
        + (PUBLIC_DISPLAY_RGB_WHITE_CEILING - PUBLIC_DISPLAY_RGB_BLACK_FLOOR)
        * value**PUBLIC_DISPLAY_RGB_GAMMA
        for value in source_rgba[:3]
    ]
    return [*rgb, source_rgba[3]]


def _source_file(
    field_root: Path,
    relative: object,
    expected_sha256: object,
    *,
    suffix: str,
) -> tuple[str, Path, str]:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ExportBlocked("源manifest文件路径必须是非空相对路径")
    source = (field_root / relative).resolve()
    if field_root not in source.parents or not source.is_file():
        raise ExportBlocked(f"源文件缺失或越界：{relative}")
    if source.suffix.lower() != suffix:
        raise ExportBlocked(f"源文件扩展名错误：{relative}")
    actual_sha256 = sha256_file(source)
    if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
        raise ExportBlocked(f"源文件SHA-256不匹配：{relative}")
    return Path(relative).as_posix(), source, actual_sha256


def _load_source_contract(field_build: Path) -> tuple[Path, Path, dict[str, Any]]:
    field_root = field_build.expanduser().resolve()
    manifest_path = field_root / "manifest.json"
    if not manifest_path.is_file():
        raise ExportBlocked(f"缺少源manifest：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportBlocked(f"源manifest无法解析：{exc}") from exc
    manifest = _json_object(manifest, "源manifest")
    if manifest.get("artifact_type") != EXPECTED_ARTIFACT_TYPE:
        raise ExportBlocked("源产物类型不是RMUC 2026官方场地MuJoCo资产")
    if manifest.get("status") != "PASS":
        raise ExportBlocked("源场地构建完整性状态不是PASS")
    scope = _json_object(manifest.get("validation_scope"), "validation_scope")
    if scope.get("status") != "DRAFT_BLOCKED":
        raise ExportBlocked("本导出器要求显式保留当前DRAFT_BLOCKED验证边界")
    visual = manifest.get("visual_meshes")
    if not isinstance(visual, list) or len(visual) != EXPECTED_VISUAL_MESH_COUNT:
        raise ExportBlocked(
            f"视觉网格必须恰好为{EXPECTED_VISUAL_MESH_COUNT}个，实际={len(visual) if isinstance(visual, list) else 'invalid'}"
        )
    return field_root, manifest_path, manifest


def _build_field_xml(
    manifest: dict[str, Any],
    *,
    include_visual_meshes: bool = True,
) -> bytes:
    visual = manifest["visual_meshes"]
    collision = _json_object(manifest.get("collision"), "collision")
    half_size = _finite_numbers(collision.get("half_size_xy_m"), count=2, label="碰撞半尺寸")
    center = _finite_numbers(
        collision.get("geom_center_after_translation_m"), count=3, label="碰撞中心"
    )
    maximum_height = float(collision.get("maximum_height_m", math.nan))
    base_depth = float(collision.get("base_depth_m", math.nan))
    if not math.isfinite(maximum_height) or maximum_height <= 0.0:
        raise ExportBlocked("maximum_height_m必须是有限正数")
    if not math.isfinite(base_depth) or base_depth < 0.0:
        raise ExportBlocked("base_depth_m必须是有限非负数")

    model_name = (
        "rmuc2026_field_runtime_asset_pack"
        if include_visual_meshes
        else "rmuc2026_field_collision_only"
    )
    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.002", "solver": "Newton"})
    visual_config = ET.SubElement(root, "visual")
    ET.SubElement(
        visual_config,
        "headlight",
        {
            "ambient": "0.18 0.18 0.18",
            "diffuse": "0.40 0.40 0.40",
            "specular": "0.05 0.05 0.05",
        },
    )
    ET.SubElement(visual_config, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(visual_config, "quality", {"offsamples": "4"})
    asset = ET.SubElement(root, "asset")
    if include_visual_meshes:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "rmuc2026_sky",
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.16 0.22 0.30",
                "rgb2": "0.025 0.035 0.055",
                "width": "512",
                "height": "3072",
            },
        )
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "rmuc2026_collision",
            "file": Path(str(collision["image_file"])).as_posix(),
            "nrow": str(int(collision["rows_y"])),
            "ncol": str(int(collision["columns_x"])),
            "size": " ".join(f"{value:.9g}" for value in (*half_size, maximum_height, base_depth)),
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "directional": "true",
            "castshadow": "false",
            "pos": "0 0 12",
            "dir": "-0.25 -0.35 -1",
            "diffuse": "0.55 0.53 0.50",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "directional": "true",
            "castshadow": "false",
            "pos": "0 0 8",
            "dir": "0.45 0.20 -1",
            "diffuse": "0.10 0.12 0.15",
        },
    )
    if include_visual_meshes:
        for index, record_value in enumerate(visual):
            record = _json_object(record_value, f"visual_meshes[{index}]")
            rgba = _finite_numbers(
                record.get("rgba"), count=4, label=f"visual_meshes[{index}].rgba"
            )
            if not all(0.0 <= value <= 1.0 for value in rgba):
                raise ExportBlocked(f"visual_meshes[{index}].rgba越界")
            display_rgba = _display_rgba(rgba)
            material = f"rmuc2026_material_{index}"
            mesh = f"rmuc2026_visual_mesh_{index}"
            ET.SubElement(
                asset,
                "material",
                {
                    "name": material,
                    "rgba": " ".join(f"{value:.8g}" for value in display_rgba),
                    "specular": "0.08",
                    "shininess": "0.25",
                },
            )
            ET.SubElement(
                asset,
                "mesh",
                {"name": mesh, "file": Path(str(record["file"])).as_posix()},
            )
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": f"rmuc2026_visual_{index}",
                    "type": "mesh",
                    "mesh": mesh,
                    "material": material,
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "2" if record.get("visual_role") == "base_surface_shell" else "1",
                },
            )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "rmuc2026_field_collision",
            "type": "hfield",
            "hfield": "rmuc2026_collision",
            "pos": " ".join(f"{value:.9g}" for value in center),
            "rgba": "0.10 0.38 0.62 0" if include_visual_meshes else "0.16 0.42 0.68 1",
            "contype": "2",
            "conaffinity": "1",
            "friction": "1 0.005 0.0001",
            "solref": "0.02 1",
            "group": "3" if include_visual_meshes else "0",
        },
    )
    dimensions = _json_object(manifest.get("dimensions"), "dimensions")
    outer_bounds = dimensions.get("cad_assembly_outer_bounds_after_translation_m")
    if (
        not isinstance(outer_bounds, list)
        or len(outer_bounds) != 2
        or any(not isinstance(row, list) or len(row) != 3 for row in outer_bounds)
    ):
        raise ExportBlocked("dimensions缺少CAD外包围")
    low = _finite_numbers(outer_bounds[0], count=3, label="CAD外包围low")
    high = _finite_numbers(outer_bounds[1], count=3, label="CAD外包围high")
    if any(hi <= lo for lo, hi in zip(low, high)):
        raise ExportBlocked("CAD外包围无效")
    center_xy = [0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])]
    extent_xy = [high[0] - low[0], high[1] - low[1]]
    distance = (
        1.08
        * max(extent_xy[1], extent_xy[0] / (1280.0 / 720.0))
        / (2.0 * math.tan(math.radians(55.0 / 2.0)))
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "rmuc2026_overview",
            "mode": "fixed",
            "pos": f"{center_xy[0]:.9g} {center_xy[1]:.9g} {high[2] + distance:.9g}",
            "xyaxes": "1 0 0 0 1 0",
            "fovy": "55",
        },
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _file_record(root: Path, relative: str, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "file": relative,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def export_runtime_asset_pack(field_build: Path, output_dir: Path) -> dict[str, Any]:
    field_root, source_manifest_path, source_manifest = _load_source_contract(field_build)
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise ExportBlocked(f"输出目录已存在，拒绝覆盖：{output}")
    if output == field_root or field_root in output.parents:
        raise ExportBlocked("输出目录不能位于源场地产物内部")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        visual_records: list[dict[str, object]] = []
        compact_visual_meshes: list[dict[str, object]] = []
        for index, record_value in enumerate(source_manifest["visual_meshes"]):
            record = _json_object(record_value, f"visual_meshes[{index}]")
            relative, source, _digest = _source_file(
                field_root,
                record.get("file"),
                record.get("sha256"),
                suffix=".obj",
            )
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            visual_records.append(_file_record(staging, relative, "cad_visual_mesh"))
            compact_visual_meshes.append(
                {
                    "file": relative,
                    "sha256": str(record["sha256"]),
                    "material_id": str(record.get("material_id", f"mesh_{index}")),
                    "rgba": _finite_numbers(
                        record.get("rgba"),
                        count=4,
                        label=f"visual_meshes[{index}].rgba",
                    ),
                    "visual_role": str(record.get("visual_role", "cad_structure")),
                }
            )

        collision = _json_object(source_manifest.get("collision"), "collision")
        copied_collision: list[tuple[str, str]] = []
        for file_key, hash_key, suffix, role in (
            ("image_file", "image_sha256", ".png", "heightfield_bootstrap_png"),
            ("samples_file", "samples_sha256", ".npz", "heightfield_float_samples"),
        ):
            relative, source, _digest = _source_file(
                field_root,
                collision.get(file_key),
                collision.get(hash_key),
                suffix=suffix,
            )
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied_collision.append((relative, role))

        xml_path = staging / OUTPUT_XML
        xml_path.write_bytes(_build_field_xml(source_manifest, include_visual_meshes=True))
        collision_xml_path = staging / OUTPUT_COLLISION_ONLY_XML
        collision_xml_path.write_bytes(
            _build_field_xml(source_manifest, include_visual_meshes=False)
        )
        files = [
            _file_record(staging, OUTPUT_XML, "field_mjcf_full"),
            _file_record(staging, OUTPUT_COLLISION_ONLY_XML, "field_mjcf_collision_only"),
        ]
        files.extend(visual_records)
        files.extend(_file_record(staging, relative, role) for relative, role in copied_collision)

        source_identity = _json_object(source_manifest.get("source"), "source")
        scope = _json_object(source_manifest.get("validation_scope"), "validation_scope")
        recommended_spawn = _json_object(
            source_manifest.get("recommended_spawn"), "recommended_spawn"
        )
        dimensions = _json_object(source_manifest.get("dimensions"), "dimensions")
        compact_manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": OUTPUT_ARTIFACT_TYPE,
            "status": "PASS",
            "validation_status": "DRAFT_BLOCKED",
            "source_identity": {
                "field_manifest_sha256": sha256_file(source_manifest_path),
                "field_artifact_type": source_manifest["artifact_type"],
                "field_build_status": source_manifest["status"],
                "official_step_sha256": source_identity.get("sha256"),
                "official_step_size_bytes": source_identity.get("size_bytes"),
                "official_step_product": source_identity.get("step_product"),
                "official_download_url": source_identity.get("official_download_url"),
                "license_status": "UNSPECIFIED",
                "redistribution_authorized": False,
                "asset_included_in_source_repository": False,
            },
            "visual_meshes": compact_visual_meshes,
            "collision": {
                "kind": collision.get("kind"),
                "image_file": str(collision["image_file"]),
                "image_sha256": str(collision["image_sha256"]),
                "samples_file": str(collision["samples_file"]),
                "samples_sha256": str(collision["samples_sha256"]),
                "rows_y": int(collision["rows_y"]),
                "columns_x": int(collision["columns_x"]),
                "half_size_xy_m": _finite_numbers(
                    collision.get("half_size_xy_m"), count=2, label="collision.half_size_xy_m"
                ),
                "geom_center_after_translation_m": _finite_numbers(
                    collision.get("geom_center_after_translation_m"),
                    count=3,
                    label="collision.geom_center_after_translation_m",
                ),
                "maximum_height_m": float(collision["maximum_height_m"]),
                "minimum_height_m": float(collision.get("minimum_height_m", 0.0)),
                "base_depth_m": float(collision["base_depth_m"]),
                "resolution_m": float(collision.get("resolution_m", math.nan)),
                "png_rows": collision.get("png_rows"),
                "claim_boundary": collision.get("claim_boundary"),
            },
            "coordinate_frame": {
                "world_units": "metre-radian-kilogram-second",
                "z_up": True,
                "recommended_spawn": recommended_spawn,
                "npz_axes_before_world_translation": True,
                "world_x_m": "x_m - recommended_spawn.x_before_translation_m",
                "world_y_m": "y_m - recommended_spawn.y_before_translation_m",
                "world_z_m": "height_m - recommended_spawn.terrain_height_m",
            },
            "dimensions": {
                "official_core_battlefield_m": dimensions.get("official_core_battlefield_m"),
                "cad_assembly_outer_bounds_after_translation_m": dimensions.get(
                    "cad_assembly_outer_bounds_after_translation_m"
                ),
                "cad_assembly_outer_extents_m": dimensions.get("cad_assembly_outer_extents_m"),
            },
            "contents": {
                "entrypoint": OUTPUT_XML,
                "visual_obj_count": len(visual_records),
                "file_count_excluding_manifest": len(files),
                "files": files,
            },
            "runtime_profiles": {
                "default": "full",
                "profiles": {
                    "full": {
                        "entrypoint": OUTPUT_XML,
                        "includes_visual_meshes": True,
                        "visual_mesh_count": len(visual_records),
                        "intended_use": "interactive_visualization",
                    },
                    "collision_only": {
                        "entrypoint": OUTPUT_COLLISION_ONLY_XML,
                        "includes_visual_meshes": False,
                        "visual_mesh_count": 0,
                        "intended_use": "headless_physics",
                    },
                },
            },
            "heightfield_precision": {
                "name": "rmuc2026_collision",
                "png_file": str(collision["image_file"]),
                "float_samples_file": str(collision["samples_file"]),
                "rows_y": int(collision["rows_y"]),
                "columns_x": int(collision["columns_x"]),
                "maximum_height_m": float(collision["maximum_height_m"]),
                "npz_array": "height_m",
                "normalized_model_values": "clip(height_m / maximum_height_m, 0, 1)",
                "float_npz_injection_required_before_validated_physics": True,
                "png_role": "MuJoCo dimensions/bootstrap only",
            },
            "portability": {
                "all_mjcf_file_references_are_relative": True,
                "source_tree_required_after_export": False,
                "field_model_only": True,
                "robot_and_policy_included": False,
            },
            "excluded": {
                "colored_intermediate_glb": True,
                "official_rulebook_screenshot": True,
                "source_step": True,
                "telemetry_or_video": True,
            },
            "distribution": {
                "generated_locally": True,
                "third_party_geometry": True,
                "safe_to_publish_without_rightsholder_permission": False,
                "code_license_applies_to_asset_pack": False,
            },
            "validation_boundary": {
                "status": "DRAFT_BLOCKED",
                "profile_id": scope.get("profile_id"),
                "claim_boundary": scope.get("claim_boundary"),
                "blocking_reasons": scope.get("blocking_reasons"),
                "whole_field_topology_ready": scope.get("whole_field_topology_ready"),
                "final_policy_validation_ready": scope.get("final_policy_validation_ready"),
                "pack_pass_means": "file integrity, relative-path loadability, and preserved source identity only",
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(compact_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
        return compact_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _preview(field_build: Path, output_dir: Path) -> dict[str, object]:
    field_root, manifest_path, manifest = _load_source_contract(field_build)
    collision = _json_object(manifest.get("collision"), "collision")
    total_source_bytes = 0
    for index, record_value in enumerate(manifest["visual_meshes"]):
        record = _json_object(record_value, f"visual_meshes[{index}]")
        _relative, source, _digest = _source_file(
            field_root, record.get("file"), record.get("sha256"), suffix=".obj"
        )
        total_source_bytes += source.stat().st_size
    for file_key, hash_key, suffix in (
        ("image_file", "image_sha256", ".png"),
        ("samples_file", "samples_sha256", ".npz"),
    ):
        _relative, source, _digest = _source_file(
            field_root,
            collision.get(file_key),
            collision.get(hash_key),
            suffix=suffix,
        )
        total_source_bytes += source.stat().st_size
    return {
        "mode": "PREVIEW_ONLY",
        "source_manifest_sha256": sha256_file(manifest_path),
        "output_dir": str(output_dir.expanduser().resolve()),
        "visual_obj_count": len(manifest["visual_meshes"]),
        "payload_bytes_before_xml_and_compact_manifest": total_source_bytes,
        "validation_status": "DRAFT_BLOCKED",
        "excluded": ["47 MB colored GLB", "official rulebook screenshot", "source STEP"],
        "writes_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出可搬运的RMUC 2026 MuJoCo场地资产包")
    parser.add_argument("--field-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.preview:
            print(
                json.dumps(
                    _preview(args.field_build, args.output_dir), ensure_ascii=False, indent=2
                )
            )
            return 0
        result = export_runtime_asset_pack(args.field_build, args.output_dir)
        print("RMUC2026_MUJOCO_ASSET_EXPORT=PASS")
        print(f"完成：{args.output_dir.expanduser().resolve()}")
        print(
            "边界：导出PASS仅证明相对路径与文件身份；场地验证仍为"
            f"{result['validation_status']}，不代表完整碰撞或策略通过。"
        )
        return 0
    except (ExportBlocked, OSError, ValueError, KeyError) as exc:
        print(f"RMUC2026_MUJOCO_ASSET_EXPORT=BLOCKED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
