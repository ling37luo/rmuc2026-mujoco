"""Derived, local fly-ramp heightfields for parallel training.

The grid is sliced at integer sample indices. It is never resampled or used to
modify the source runtime pack. The exported region is its own artifact type,
not a full-field ``FieldAsset`` profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from .errors import AssetIntegrityError, ManifestError, MujocoModelError
from .fly_routes import fly_route_descriptor
from .manifest import (
    FieldAsset,
    _png_scanline_payload,
    _unfilter_png_scanlines,
    sha256_file,
)
from .mjcf import (
    FIELD_ATTACH_PREFIX,
    HFIELD_NAME,
    _compile_spec,
    _load_spec,
    _mujoco,
    _strip_embedded_field_scene,
)
from .query import HeightFieldData, load_heightfield
from .scenarios import scenario_descriptor


REGION_ARTIFACT_TYPE = "rmuc2026_mujoco_training_region"
REGION_SCHEMA_VERSION = 1
REGION_SCENARIOS = ("fly_ramp_north", "fly_ramp_south")
_SIDE_MARGIN_M = 0.80
_END_MARGIN_M = 0.60


def _normalized_source_height(height: np.ndarray, record: Mapping[str, Any]) -> np.ndarray:
    source_height = height + float(record["terrain_offset_m"])
    maximum = float(record["maximum_source_height_m"])
    if record.get("legacy_clipped"):
        return np.clip(source_height / maximum, 0.0, 1.0)
    minimum = float(record["minimum_source_height_m"])
    return (source_height - minimum) / (maximum - minimum)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_hash(manifest: Mapping[str, Any]) -> str:
    version = manifest.get("profile_hash_contract_version", 1)
    if version == 1:
        # Historical local candidates bound only the source-grid contract.
        return _hash_payload(manifest["source"])
    if version != 2:
        raise ManifestError("unsupported training-region profile hash contract")
    try:
        payload = {
            "source": manifest["source"],
            "route": manifest["route"],
            "grid": manifest["grid"],
            "files": manifest["files"],
        }
    except KeyError as exc:
        raise ManifestError(f"training-region profile contract is missing {exc}") from exc
    return _hash_payload(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _heightfield_png(normalized: np.ndarray) -> bytes:
    """Make MuJoCo's required 16-bit bootstrap image without a build dependency."""

    pixels = np.rint(np.clip(normalized, 0.0, 1.0) * 65535).astype(">u2")
    rows, columns = pixels.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in np.flipud(pixels))
    header = struct.pack(">IIBBBBB", columns, rows, 16, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _grid_slice(axis: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    if lower < axis[0] or upper > axis[-1]:
        raise ValueError("fly-ramp training region extends beyond the source heightfield")
    start = max(0, int(np.searchsorted(axis, lower, side="right")) - 1)
    stop = min(len(axis), int(np.searchsorted(axis, upper, side="left")) + 1)
    if stop - start < 3:
        raise ValueError("fly-ramp training region has too few heightfield samples")
    return start, stop


def _crop_indices(world: HeightFieldData, route: Mapping[str, Any]) -> tuple[int, int, int, int]:
    direction = np.asarray(route["uphill_unit_xy"], dtype=np.float64)
    lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    start = np.asarray(route["approach_xyz_m"][:2], dtype=np.float64) - direction * _END_MARGIN_M
    end = (
        np.asarray(route["landing_target_xyz_m"][:2], dtype=np.float64) + direction * _END_MARGIN_M
    )
    corners = np.asarray(
        [point + sign * _SIDE_MARGIN_M * lateral for point in (start, end) for sign in (-1, 1)]
    )
    col0, col1 = _grid_slice(world.x_m, float(corners[:, 0].min()), float(corners[:, 0].max()))
    row0, row1 = _grid_slice(world.y_m, float(corners[:, 1].min()), float(corners[:, 1].max()))
    return row0, row1, col0, col1


def _source_field_xml(asset: FieldAsset) -> ET.Element:
    root = ET.parse(asset.entrypoint_for("collision_only")).getroot()
    geom = root.find("./worldbody/geom[@name='rmuc2026_field_collision']")
    if geom is None or geom.get("type") != "hfield":
        raise ManifestError("source collision_only profile has no expected field heightfield")
    return root


def _verify_grid_alignment(
    xml_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    grid: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    """Require MuJoCo's actual hfield coordinates to match exported Isaac axes."""

    try:
        root = ET.parse(xml_path).getroot()
        hfield = root.find(f"./asset/hfield[@name='{HFIELD_NAME}']")
        geom = root.find("./worldbody/geom[@name='rmuc2026_field_collision']")
        if hfield is None or geom is None or geom.get("hfield") != HFIELD_NAME:
            raise ValueError("field hfield or geom is missing")
        rows, columns = int(hfield.attrib["nrow"]), int(hfield.attrib["ncol"])
        half_x, half_y, vertical_range, _base_depth = (
            float(value) for value in hfield.attrib["size"].split()
        )
        center_x, center_y, base_z = (float(value) for value in geom.attrib["pos"].split())
        bounds = np.asarray(grid["bounds_xy_m"], dtype=float)
        resolution = np.asarray(grid["resolution_xy_m"], dtype=float)
    except (OSError, ET.ParseError, KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"training-region MJCF/grid geometry is invalid: {exc}") from exc
    if (
        (rows, columns) != (len(y), len(x))
        or rows < 3
        or columns < 3
        or bounds.shape != (2, 2)
        or resolution.shape != (2,)
        or not np.isfinite(bounds).all()
        or not np.isfinite(resolution).all()
        or min(half_x, half_y, vertical_range) <= 0
    ):
        raise ManifestError("training-region MJCF/grid dimensions disagree")
    expected_bounds = np.asarray(
        [[center_x - half_x, center_y - half_y], [center_x + half_x, center_y + half_y]]
    )
    axes_bounds = np.asarray([[x[0], y[0]], [x[-1], y[-1]]])
    expected_resolution = np.asarray([2 * half_x / (columns - 1), 2 * half_y / (rows - 1)])
    if (
        not np.allclose(axes_bounds, expected_bounds, atol=1e-8, rtol=0)
        or not np.allclose(bounds, axes_bounds, atol=1e-8, rtol=0)
        or not np.allclose(resolution, expected_resolution, atol=1e-8, rtol=0)
        or not np.allclose(np.diff(x), expected_resolution[0], atol=1e-8, rtol=0)
        or not np.allclose(np.diff(y), expected_resolution[1], atol=1e-8, rtol=0)
    ):
        raise ManifestError("training-region NPZ XY axes disagree with MJCF/grid coordinates")
    normalization = source.get("source_vertical_normalization")
    if normalization is not None and not np.allclose(
        [base_z, vertical_range],
        [normalization["xml_geom_base_z_m"], normalization["xml_hfield_vertical_range_m"]],
        atol=1e-9,
        rtol=0,
    ):
        raise ManifestError("training-region MJCF vertical scale disagrees with source")


def _write_region_xml(
    source_root: ET.Element,
    path: Path,
    world: HeightFieldData,
) -> None:
    center_x = float((world.x_m[0] + world.x_m[-1]) / 2)
    center_y = float((world.y_m[0] + world.y_m[-1]) / 2)
    half_x = float((world.x_m[-1] - world.x_m[0]) / 2)
    half_y = float((world.y_m[-1] - world.y_m[0]) / 2)
    source_hfield = source_root.find("./asset/hfield[@name='rmuc2026_collision']")
    source_geom = source_root.find("./worldbody/geom[@name='rmuc2026_field_collision']")
    assert source_hfield is not None and source_geom is not None
    source_vertical_range = source_hfield.attrib["size"].split()[2]
    source_vertical_base = source_geom.attrib["pos"].split()[2]
    root = ET.Element("mujoco", {"model": "rmuc2026_fly_training_region"})
    for tag in ("compiler", "option", "visual"):
        original = source_root.find(tag)
        if original is not None:
            root.append(ET.fromstring(ET.tostring(original)))
    asset_node = ET.SubElement(root, "asset")
    ET.SubElement(
        asset_node,
        "hfield",
        {
            "name": HFIELD_NAME,
            "file": "collision/heightfield.png",
            "nrow": str(len(world.y_m)),
            "ncol": str(len(world.x_m)),
            "size": f"{half_x:.17g} {half_y:.17g} {source_vertical_range} 0.05",
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    source_worldbody = source_root.find("worldbody")
    assert source_worldbody is not None
    for light in source_worldbody.findall("light"):
        worldbody.append(ET.fromstring(ET.tostring(light)))
    geom = ET.fromstring(ET.tostring(source_geom))
    geom.set("pos", f"{center_x:.17g} {center_y:.17g} {source_vertical_base}")
    worldbody.append(geom)
    # The audited ramps sit near the original perimeter. Keep the exact fence
    # geoms so a lateral excursion sees the same barrier as the full field.
    for fence in source_worldbody.findall("geom"):
        if str(fence.get("name", "")).startswith("rmuc2026_perimeter_"):
            worldbody.append(ET.fromstring(ET.tostring(fence)))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def export_training_region(
    asset: FieldAsset,
    output: str | Path,
    *,
    scenario_id: str,
    approach_distance_m: float = 0.9,
) -> dict[str, Any]:
    """Export an exact-grid local collision region from one verified field pack."""

    if scenario_id not in REGION_SCENARIOS:
        raise ValueError(f"scenario_id must be one of {REGION_SCENARIOS}")
    if not asset.hashes_verified:
        raise AssetIntegrityError("source field pack must have verified file hashes")
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise ValueError(f"output already exists: {root}")
    route = fly_route_descriptor(asset, scenario_id, approach_distance_m)
    source_world = load_heightfield(asset)
    row0, row1, col0, col1 = _crop_indices(source_world, route)
    world = HeightFieldData(
        x_m=source_world.x_m[col0:col1].copy(),
        y_m=source_world.y_m[row0:row1].copy(),
        height_m=source_world.height_m[row0:row1, col0:col1].copy(),
    )
    minimum = float(world.height_m.min())
    maximum = float(world.height_m.max())
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("cropped heightfield has no valid height range")
    source_root = _source_field_xml(asset)
    source_vertical = {
        "terrain_offset_m": float(asset.recommended_spawn["terrain_height_m"]),
        "minimum_source_height_m": float(asset.collision["minimum_height_m"]),
        "maximum_source_height_m": float(asset.collision["maximum_height_m"]),
        "xml_hfield_vertical_range_m": float(
            source_root.find("./asset/hfield[@name='rmuc2026_collision']").attrib["size"].split()[2]
        ),
        "xml_geom_base_z_m": float(
            source_root.find("./worldbody/geom[@name='rmuc2026_field_collision']")
            .attrib["pos"]
            .split()[2]
        ),
    }
    if int(asset.manifest["schema_version"]) != 3:
        # Match the legacy pack loader's clipped normalization exactly.
        source_vertical["legacy_clipped"] = True
    normalized = _normalized_source_height(world.height_m, source_vertical)
    region_collision = root / "collision"
    region_collision.mkdir(parents=True)
    np.savez_compressed(
        region_collision / "heightfield.npz",
        x_m=world.x_m,
        y_m=world.y_m,
        height_m=world.height_m,
    )
    (region_collision / "heightfield.png").write_bytes(_heightfield_png(normalized))
    _write_region_xml(source_root, root / "field.xml", world)
    files = {
        relative: {
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in ("collision/heightfield.npz", "collision/heightfield.png", "field.xml")
    }
    source_profile = scenario_descriptor(asset, scenario_id, profile="collision_only")
    profile_contract = {
        "scenario_id": scenario_id,
        "source_manifest_sha256": asset.manifest_sha256,
        "source_collision_samples_sha256": str(asset.collision["samples_sha256"]),
        "source_profile_hash": source_profile["profile_hash"],
        "source_grid_slice_yx": [[row0, row1], [col0, col1]],
        "source_vertical_normalization": source_vertical,
        "heightfield_samples_sha256": files["collision/heightfield.npz"]["sha256"],
        "kept_collision": [
            "exact source heightfield samples in selected rectangle",
            "source perimeter fence",
        ],
        "removed": ["visual meshes", "livery", "heightfield samples outside selected rectangle"],
    }
    grid_record = {
        "rows_y": len(world.y_m),
        "columns_x": len(world.x_m),
        "bounds_xy_m": [list(world.bounds_xy_m[0]), list(world.bounds_xy_m[1])],
        "minimum_height_m": minimum,
        "maximum_height_m": maximum,
        "resolution_xy_m": [float(np.diff(world.x_m).mean()), float(np.diff(world.y_m).mean())],
        "orientation": "world_y_ascending_world_x_ascending",
        "interpolation": "mujoco_hfield_triangle",
    }
    manifest = {
        "artifact_type": REGION_ARTIFACT_TYPE,
        "schema_version": REGION_SCHEMA_VERSION,
        "profile_hash_contract_version": 2,
        "validation_status": str(asset.manifest["validation_status"]),
        "topology_verified": False,
        "scenario_id": scenario_id,
        "profile": "fly_ramp_training_region",
        "source": profile_contract,
        "route": route,
        "grid": grid_record,
        "files": files,
    }
    manifest["profile_hash"] = _profile_hash(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class TrainingRegion:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str

    @classmethod
    def open(cls, root: str | Path, *, verify: bool = True) -> TrainingRegion:
        path = Path(root).expanduser().resolve()
        manifest_path = path / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ManifestError(f"training-region manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"training-region manifest is invalid: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("artifact_type") != REGION_ARTIFACT_TYPE:
            raise ManifestError("not an RMUC training-region artifact")
        if manifest.get("schema_version") != REGION_SCHEMA_VERSION:
            raise ManifestError("unsupported training-region schema")
        if manifest.get("scenario_id") not in REGION_SCENARIOS:
            raise ManifestError("unsupported training-region scenario")
        if manifest.get("validation_status") != "DRAFT_BLOCKED":
            raise ManifestError("training region overstates source validation")
        source = manifest.get("source")
        if not isinstance(source, dict) or manifest.get("profile_hash") != _profile_hash(manifest):
            raise ManifestError("training-region profile hash does not match its source contract")
        files = manifest.get("files")
        expected = {"collision/heightfield.npz", "collision/heightfield.png", "field.xml"}
        if not isinstance(files, dict) or set(files) != expected:
            raise ManifestError("training-region file table is incomplete")
        for relative in expected:
            candidate = path / relative
            record = files[relative]
            if candidate.is_symlink() or not candidate.is_file() or not isinstance(record, dict):
                raise AssetIntegrityError(f"training-region file is missing: {relative}")
            if candidate.stat().st_size != record.get("size_bytes"):
                raise AssetIntegrityError(f"training-region file size mismatch: {relative}")
            if verify and sha256_file(candidate) != record.get("sha256"):
                raise AssetIntegrityError(f"training-region file hash mismatch: {relative}")
        with np.load(path / "collision/heightfield.npz", allow_pickle=False) as samples:
            x = np.asarray(samples["x_m"], dtype=np.float64)
            y = np.asarray(samples["y_m"], dtype=np.float64)
            height = np.asarray(samples["height_m"], dtype=np.float64)
        grid = manifest.get("grid")
        if not isinstance(grid, dict) or height.shape != (
            grid.get("rows_y"),
            grid.get("columns_x"),
        ):
            raise ManifestError("training-region grid shape mismatch")
        if x.shape != (height.shape[1],) or y.shape != (height.shape[0],):
            raise ManifestError("training-region grid axes mismatch")
        if not (np.isfinite(height).all() and np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
            raise ManifestError("training-region grid contains nonfinite or unsorted samples")
        if not np.allclose(
            [height.min(), height.max()],
            [grid["minimum_height_m"], grid["maximum_height_m"]],
            atol=1e-10,
            rtol=0,
        ):
            raise ManifestError("training-region height range mismatch")
        _verify_grid_alignment(path / "field.xml", x, y, grid, source)
        if verify:
            normalization = source.get("source_vertical_normalization")
            if normalization is None:
                normalized = (height - height.min()) / (height.max() - height.min())
            else:
                normalized = _normalized_source_height(height, normalization)
            filtered, width, image_height = _png_scanline_payload(
                path / "collision/heightfield.png",
                label="training-region bootstrap PNG",
                expected_rows=height.shape[0],
                expected_columns=height.shape[1],
            )
            pixels = _unfilter_png_scanlines(
                filtered,
                width=width,
                height=image_height,
                label="training-region bootstrap PNG",
            )
            observed = np.frombuffer(pixels, dtype=">u2").reshape(height.shape)[::-1]
            expected_pixels = np.rint(np.clip(normalized, 0, 1) * 65535).astype(np.uint16)
            if np.any(np.abs(observed.astype(np.int32) - expected_pixels.astype(np.int32)) > 1):
                raise AssetIntegrityError("training-region PNG disagrees with float samples")
        return cls(path, manifest, sha256_file(manifest_path))

    def heightfield(self) -> HeightFieldData:
        with np.load(self.root / "collision/heightfield.npz", allow_pickle=False) as samples:
            return HeightFieldData(
                x_m=np.asarray(samples["x_m"], dtype=np.float64),
                y_m=np.asarray(samples["y_m"], dtype=np.float64),
                height_m=np.asarray(samples["height_m"], dtype=np.float64),
            )


def _inject_region_heightfield(model: Any, region: TrainingRegion, name: str) -> None:
    mujoco = _mujoco()
    hfield_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name))
    if hfield_id < 0:
        raise MujocoModelError(f"composed model lacks region heightfield {name!r}")
    height = region.heightfield().height_m
    rows, columns = height.shape
    if (int(model.hfield_nrow[hfield_id]), int(model.hfield_ncol[hfield_id])) != (rows, columns):
        raise MujocoModelError("compiled region heightfield shape disagrees with samples")
    normalization = region.manifest["source"].get("source_vertical_normalization")
    if normalization is None:
        # Compatibility with local regions exported before the source-scale
        # parity correction; no public version has used that normalization.
        minimum = float(region.manifest["grid"]["minimum_height_m"])
        maximum = float(region.manifest["grid"]["maximum_height_m"])
        normalized = (height - minimum) / (maximum - minimum)
    else:
        normalized = _normalized_source_height(height, normalization)
    offset = int(model.hfield_adr[hfield_id])
    model.hfield_data[offset : offset + rows * columns] = normalized.reshape(-1)


def load_training_region_model(region: TrainingRegion) -> tuple[Any, Any]:
    mujoco = _mujoco()
    spec = _load_spec(mujoco, region.root / "field.xml", label="training region")
    model = _compile_spec(spec, label="training region")
    _inject_region_heightfield(model, region, HFIELD_NAME)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def compose_training_region_with_robot(
    region: TrainingRegion, robot_xml: str | Path
) -> tuple[Any, Any]:
    """Compose the same local region with any external robot MJCF."""

    mujoco = _mujoco()
    robot_path = Path(robot_xml).expanduser().resolve()
    if robot_path.is_symlink() or not robot_path.is_file():
        raise MujocoModelError(f"robot XML is missing: {robot_path}")
    robot_spec = _load_spec(mujoco, robot_path, label="robot XML")
    _strip_embedded_field_scene(robot_spec)
    field_spec = _load_spec(mujoco, region.root / "field.xml", label="training region")
    robot_spec.copy_during_attach = True
    frame = robot_spec.worldbody.add_frame(name="rmuc2026_field_frame")
    robot_spec.attach(field_spec, frame=frame, prefix=FIELD_ATTACH_PREFIX)
    model = _compile_spec(robot_spec, label="composed robot and training region")
    _inject_region_heightfield(model, region, f"{FIELD_ATTACH_PREFIX}{HFIELD_NAME}")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


__all__ = [
    "TrainingRegion",
    "compose_training_region_with_robot",
    "export_training_region",
    "load_training_region_model",
]
